"""Codebase RAG — separate per-user DB for Aiko's own source.

Stores the entire repo under <USER_SPACE_ROOT>/<user_id>/knowledge/codebase.db
so Aiko can answer "where is X?", "how does Y work?" via regular RAG.

Optimized for Jetson Orin Nano 8GB:
  - SQLite + sqlite-vec (vec0, cosine), WAL, no external service.
  - 640-d harrier-oss-270m embeddings, batched 32, cached.
  - 900-char chunks, ~150 token budget, fits 8GB RAM.
  - Incremental: file SHA1, skip unchanged; prune stale docs.
  - Hybrid RRF: vec KNN (oversampled) + FTS5 OR, entity boost, threshold.

GraphRAG note: plain RAG is sufficient on-device. A lightweight
knowledge-graph layer is approximated by entity-overlap boosting
(entities extracted via memorize.extract_entities) and co-mention
edges — no separate graph DB needed on 8GB.

Usage:
  uv run python -m cognition.knowledge.codebase --ingest
  uv run python -m cognition.knowledge.codebase --search "how does attention gate work?"
  from cognition.knowledge.codebase import search_codebase, codebase_context_for, ingest_codebase
"""
from __future__ import annotations

import hashlib
import os
import re
import sqlite3
import time
import uuid
from pathlib import Path
from collections import OrderedDict
import threading

from system.log import get_logger
from system.config import env_float
from system.userspace import current_user_id, user_state_path
from cognition.memory.vecstore import (
    HarrierEmbedder,
    initialize_store_db,
    rank_by_id,
    rrf_score,
    user_scoped_fts_search,
    user_scoped_vec_knn,
    utc_now_iso,
)
from cognition.knowledge.schema import EMBED_DIMS, KNOWLEDGE_QUERY_INSTRUCT

log = get_logger(__name__)

CODEBASE_DB_PATH = "knowledge/codebase.db"
CODEBASE_CHUNK_CHARS = 900
CODEBASE_CHUNK_OVERLAP = 120
CODEBASE_RRF_K = 60
CODEBASE_KNN_LIMIT = 20
CODEBASE_FTS_LIMIT = 20
CODEBASE_RECALL_THRESHOLD = 0.015
CODEBASE_KNN_MIN_SIM = 0.12
CODEBASE_CONTEXT_CHARS = 4000
CODEBASE_ENTITY_BOOST = 0.003

# Repo root for ingestion (where this file lives two levels up -> repo root)
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Excludes — never index these
_EXCLUDE_DIRS = {
    ".git", ".venv", "__pycache__", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", ".benchmarks", "node_modules", ".aiko",
    "models", "checkpoints", "outputs", "logs", "data", "datasets",
    "archive", "papers", "assets",
}
_EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".so", ".o", ".a", ".onnx", ".bin", ".pt", ".pth", ".gguf", ".npz", ".db", ".sqlite", ".pickle", ".pkl"}
_TEXT_EXTS = {
    ".py", ".md", ".yaml", ".yml", ".toml", ".json", ".txt", ".sh", ".ini", ".cfg",
    ".rst", ".js", ".ts", ".tsx", ".jsx", ".html", ".css", ".sql", ".dockerfile",
}

_DDL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS codebase_docs (
    id          TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    path        TEXT NOT NULL,
    title       TEXT NOT NULL,
    sha1        TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS codebase_chunks (
    id              TEXT PRIMARY KEY,
    doc_id          TEXT NOT NULL REFERENCES codebase_docs(id) ON DELETE CASCADE,
    user_id         TEXT NOT NULL,
    chunk_index     INTEGER NOT NULL,
    text            TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    entities        TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_codebase_docs_user ON codebase_docs(user_id);
CREATE INDEX IF NOT EXISTS idx_codebase_docs_path ON codebase_docs(path);
CREATE INDEX IF NOT EXISTS idx_codebase_chunks_user ON codebase_chunks(user_id);
CREATE INDEX IF NOT EXISTS idx_codebase_chunks_doc ON codebase_chunks(doc_id);
CREATE VIRTUAL TABLE IF NOT EXISTS codebase_chunks_fts USING fts5(
    text, id UNINDEXED, content='codebase_chunks', content_rowid='rowid'
);
CREATE VIRTUAL TABLE IF NOT EXISTS codebase_chunks_vec USING vec0(
    id TEXT PRIMARY KEY, embedding FLOAT[640] distance_metric=cosine
);
CREATE TRIGGER IF NOT EXISTS codebase_chunks_ai AFTER INSERT ON codebase_chunks BEGIN
    INSERT INTO codebase_chunks_fts(rowid, text, id) VALUES (new.rowid, new.text, new.id);
END;
CREATE TRIGGER IF NOT EXISTS codebase_chunks_ad AFTER DELETE ON codebase_chunks BEGIN
    INSERT INTO codebase_chunks_fts(codebase_chunks_fts, rowid, text, id) VALUES ('delete', old.rowid, old.text, old.id);
    DELETE FROM codebase_chunks_vec WHERE id = old.id;
END;
CREATE TRIGGER IF NOT EXISTS codebase_chunks_au AFTER UPDATE OF text ON codebase_chunks BEGIN
    INSERT INTO codebase_chunks_fts(codebase_chunks_fts, rowid, text, id) VALUES ('delete', old.rowid, old.text, old.id);
    INSERT INTO codebase_chunks_fts(rowid, text, id) VALUES (new.rowid, new.text, new.id);
END;
"""

def connect_codebase(user_id: str | None = None) -> sqlite3.Connection:
    return initialize_store_db(CODEBASE_DB_PATH, _DDL, user_id=user_id, vector=True)

def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:12]

def _should_index(path: Path, rel: str) -> bool:
    # dir excludes
    for part in Path(rel).parts:
        if part in _EXCLUDE_DIRS:
            return False
    if path.is_symlink():
        return False
    # binary/size guard — skip >1.5 MB files
    try:
        if path.stat().st_size > 1_500_000:
            return False
    except Exception:
        return False
    # suffix guard
    suf = path.suffix.lower()
    if suf in _EXCLUDE_SUFFIXES:
        return False
    # allowlist: known text + no suffix but small (e.g. Dockerfile, Makefile)
    if suf and suf not in _TEXT_EXTS:
        # still allow if file looks like text (no null bytes)
        try:
            chunk = path.read_bytes()[:512]
            if b"\x00" in chunk:
                return False
        except Exception:
            return False
    return True

def _chunk_text(text: str, chunk_chars: int = CODEBASE_CHUNK_CHARS, overlap: int = CODEBASE_CHUNK_OVERLAP) -> list[str]:
    text = text.strip()
    if not text:
        return []
    # For code, prefer to break on blank lines / function boundaries when possible
    # Simple sliding window with overlap, but try to snap to newline near boundary.
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_chars, n)
        # snap end to newline if within 80 chars of window end
        if end < n:
            nl = text.rfind("\n", start + chunk_chars - 80, end)
            if nl > start + 200:
                end = nl + 1
        chunks.append(text[start:end].strip())
        if end >= n:
            break
        start = end - overlap
        if start < 0:
            start = 0
    return [c for c in chunks if len(c.strip()) >= 40]

def ingest_codebase(user_id: str | None = None, repo_root: Path | None = None, force: bool = False, embedder: HarrierEmbedder | None = None) -> dict:
    """Ingest entire repo into <USER_SPACE_ROOT>/<user_id>/knowledge/codebase.db.

    Incremental by SHA1: unchanged files skipped; stale docs pruned.
    Returns stats dict.
    """
    uid = user_id or current_user_id()
    root = Path(repo_root) if repo_root else _REPO_ROOT
    if not root.is_dir():
        return {"ok": False, "error": f"repo_root not found: {root}"}
    embedder = embedder or HarrierEmbedder()
    conn = connect_codebase(uid)
    try:
        # Existing docs by path
        existing = {row["path"]: (row["id"], row["sha1"]) for row in conn.execute("SELECT id, path, sha1 FROM codebase_docs WHERE user_id=?", (uid,)).fetchall()}
        seen_paths: set[str] = set()
        added_docs = 0
        added_chunks = 0
        skipped = 0
        # Collect files
        files: list[Path] = []
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            try:
                rel = str(p.relative_to(root))
            except Exception:
                continue
            if not _should_index(p, rel):
                continue
            files.append(p)
        files.sort()
        log.info("codebase ingest: scanning %d files under %s for user %s", len(files), root, uid)
        # Batch embed queue
        pending_chunks: list[tuple[str, str, int, str]] = []  # (chunk_id, doc_id, idx, text)
        for fpath in files:
            rel = str(fpath.relative_to(root))
            seen_paths.add(rel)
            try:
                text = fpath.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            # Prefix with file path for context
            full_text = f"File: {rel}\n\n" + text
            sha = _sha1(full_text)
            doc_id, old_sha = existing.get(rel, (None, None))
            if not force and doc_id and old_sha == sha:
                skipped += 1
                continue
            # Remove old doc if exists (cascades chunks + vec)
            if doc_id:
                conn.execute("DELETE FROM codebase_docs WHERE id=? AND user_id=?", (doc_id, uid))
            doc_id = uuid.uuid4().hex
            title = rel
            conn.execute("INSERT INTO codebase_docs(id, user_id, path, title, sha1, created_at) VALUES(?,?,?,?,?,?)",
                         (doc_id, uid, rel, title, sha, utc_now_iso()))
            added_docs += 1
            chunks = _chunk_text(full_text)
            for idx, chk in enumerate(chunks):
                cid = uuid.uuid4().hex
                # entities for boost (lightweight)
                try:
                    from cognition.memory.memorize import extract_entities, entities_to_json
                    ents = extract_entities(chk[:500])
                    ents_json = entities_to_json(ents[:6]) if ents else "[]"
                except Exception:
                    ents_json = "[]"
                conn.execute("INSERT INTO codebase_chunks(id, doc_id, user_id, chunk_index, text, created_at, entities) VALUES(?,?,?,?,?,?,?)",
                             (cid, doc_id, uid, idx, chk, utc_now_iso(), ents_json))
                pending_chunks.append((cid, doc_id, idx, chk))
                added_chunks += 1
            # Flush embeddings in batches to keep RAM low on 8GB
            if len(pending_chunks) >= 48:
                _embed_and_insert(conn, pending_chunks, embedder)
                conn.commit()
                pending_chunks.clear()
        if pending_chunks:
            _embed_and_insert(conn, pending_chunks, embedder)
            conn.commit()
        # Prune stale docs (files deleted since last ingest)
        pruned = 0
        for path, (doc_id, _) in list(existing.items()):
            if path not in seen_paths:
                conn.execute("DELETE FROM codebase_docs WHERE id=? AND user_id=?", (doc_id, uid))
                pruned += 1
        conn.commit()
        conn.execute("ANALYZE")
        conn.commit()
        return {"ok": True, "user_id": uid, "files_scanned": len(files), "docs_added": added_docs, "chunks_added": added_chunks, "skipped_unchanged": skipped, "pruned": pruned, "db": str(connect_codebase(uid).execute("PRAGMA database_list").fetchone()[2] if False else CODEBASE_DB_PATH)}
    except Exception as e:
        log.warning("ingest_codebase failed: %s", e)
        return {"ok": False, "error": str(e)}
    finally:
        try: conn.close()
        except Exception: pass

def _embed_and_insert(conn: sqlite3.Connection, pending: list[tuple[str,str,int,str]], embedder: HarrierEmbedder | None) -> None:
    if not pending:
        return
    texts = [t[3] for t in pending]
    vecs = None
    if embedder is not None:
        try:
            vecs = []
            for i in range(0, len(texts), 32):
                batch = texts[i:i+32]
                vecs.extend(list(embedder.embed(batch)))
        except Exception as e:
            log.warning("codebase embed failed (FTS still works): %s", e)
            vecs = None
    import sqlite_vec
    import numpy as np
    if vecs is None:
        # Fallback: zero vectors so vec table stays consistent but KNN won't match; FTS still works
        zero = np.zeros(640, dtype=np.float32)
        for (cid, _, _, _) in pending:
            conn.execute("INSERT INTO codebase_chunks_vec(id, embedding) VALUES(?,?)", (cid, sqlite_vec.serialize_float32(zero)))
        return
    for (cid, _, _, _), vec in zip(pending, vecs):
        conn.execute("INSERT INTO codebase_chunks_vec(id, embedding) VALUES(?,?)", (cid, sqlite_vec.serialize_float32(vec)))

# ── search (hybrid RRF, mirroring cognition/knowledge/search.py) ──────────

_SEARCH_CACHE: OrderedDict[tuple[str,str,int], tuple[float, list[dict]]] = OrderedDict()
_CACHE_LOCK = threading.RLock()
_CACHE_TTL = env_float("CODEBASE_CACHE_TTL", 300.0)
_CACHE_MAX = 64

def _knn(conn, query: str, embedder, uid: str, limit: int):
    if embedder is None or not (query or "").strip():
        return []
    try:
        vec = embedder.embed_query(query, instruct="Retrieve relevant code that answers the query")
    except Exception as e:
        log.debug("codebase _knn embed failed, FTS-only: %s", e)
        return []
    return user_scoped_vec_knn(conn, vec_table="codebase_chunks_vec", owner_table="codebase_chunks", owner_alias="c", vector=vec, user_id=uid, limit=limit, threshold=CODEBASE_KNN_MIN_SIM)

def _fts(conn, query: str, uid: str, limit: int):
    return user_scoped_fts_search(conn, fts_table="codebase_chunks_fts", owner_table="codebase_chunks", owner_alias="c", query=query, user_id=uid, limit=limit)

def search_codebase(query: str, limit: int = 5, *, embedder=None, user_id: str | None = None) -> list[dict]:
    uid = user_id or current_user_id()
    conn = connect_codebase(uid)
    try:
        rk = rank_by_id(_knn(conn, query, embedder, uid, CODEBASE_KNN_LIMIT))
        rf = rank_by_id(_fts(conn, query, uid, CODEBASE_FTS_LIMIT))
        ids = set(rk) | set(rf)
        if not ids:
            return []
        ph = ",".join("?"*len(ids))
        rows = conn.execute(f"SELECT c.id, c.text, c.chunk_index, c.created_at, c.entities, d.path, d.title FROM codebase_chunks c JOIN codebase_docs d ON d.id=c.doc_id WHERE c.id IN ({ph}) AND c.user_id=?", list(ids)+[uid]).fetchall()
        by_id = {r["id"]: r for r in rows}
        # entity boost
        try:
            from cognition.memory.memorize import entities_from_json, entity_overlap_score
        except Exception:
            entities_from_json = lambda x: []
            entity_overlap_score = lambda q, e: 0
        scored=[]
        for cid in ids:
            s=rrf_score(cid, rk, rf, k=CODEBASE_RRF_K)
            row=by_id.get(cid)
            if row is None: continue
            try:
                ents=entities_from_json(row["entities"] if "entities" in row.keys() else "[]")
                s+= CODEBASE_ENTITY_BOOST * entity_overlap_score(query, ents)
            except Exception: pass
            if s >= CODEBASE_RECALL_THRESHOLD:
                scored.append((s,cid))
        scored.sort(key=lambda p: (-p[0], by_id[p[1]]["created_at"]))
        return [dict(by_id[cid])|{"score": s} for s,cid in scored[:limit]]
    finally:
        try: conn.close()
        except Exception: pass

def codebase_context_for(query: str, limit: int = 5, max_chars: int | None = None, *, embedder=None, user_id: str | None = None) -> str:
    max_chars = CODEBASE_CONTEXT_CHARS if max_chars is None else max_chars
    hits = search_codebase(query, limit=limit, embedder=embedder, user_id=user_id)
    if not hits:
        return "<codebase_context>\nNo matching codebase context found.\n</codebase_context>"
    blocks=[]
    rem=max_chars
    for h in hits:
        if rem<=0: break
        body=h["text"][:rem]
        blocks.append(f'<code_chunk path="{h.get("path","")}" score="{h.get("score",0):.4f}">\n{body}\n</code_chunk>')
        rem-=len(body)
    return "<codebase_context>\n" + "\n\n".join(blocks) + "\n</codebase_context>"

if __name__ == "__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--ingest", action="store_true", help="ingest repo into <USER_SPACE_ROOT>/<user_id>/knowledge/codebase.db")
    ap.add_argument("--search", type=str, help="search query")
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--user", type=str, default=None)
    args=ap.parse_args()
    if args.ingest:
        res=ingest_codebase(user_id=args.user, force=args.force)
        print(res)
    if args.search:
        from cognition.memory.vecstore import HarrierEmbedder
        emb=HarrierEmbedder()
        hits=search_codebase(args.search, limit=args.limit, embedder=emb, user_id=args.user)
        for h in hits:
            print(f"[{h['score']:.4f}] {h['path']}#{h['chunk_index']}")
            print(h['text'][:600])
            print("---")
        if hits:
            print(codebase_context_for(args.search, limit=args.limit, embedder=emb, user_id=args.user)[:2000])
