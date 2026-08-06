"""
cognition/knowledge/backend.py

Persistent learned knowledge store for Aiko.

This module is the machine-writable knowledge RAG layer: durable facts,
excerpts, study notes, and user-approved document/PDF text that Aiko should
be able to retrieve later. Human-authored wiki/skill/persona/config files live
in :mod:`agentic.wiki`; this module owns the vector/FTS store for learned
knowledge.

Storage mirrors memory's retrieval shape without memory's forgetting lifecycle:
rows are chunked at ingest time, embedded once, stored in encrypted SQLite when
SQLITE_ENCRYPTION is enabled, and retrieved with Reciprocal Rank Fusion over
sqlite-vec KNN plus FTS5 lexical search.
"""
from __future__ import annotations

import os
import re
import sqlite3
import threading
import time
import uuid
import zipfile
import xml.etree.ElementTree as ET
from collections import OrderedDict
from datetime import datetime, timedelta
from html import escape
from pathlib import Path
from typing import Iterable, Protocol
from defusedxml import ElementTree as DET

import numpy as np

from system.config import load_config
load_config()

from cognition import reason
from cognition.memory.vecstore import initialize_store_db, insert_vector, rank_by_id, rrf_score, user_scoped_fts_search, user_scoped_vec_knn, utc_now_iso
from cognition.memory.memorize import extract_entities, entities_to_json, entities_from_json, entity_overlap_score

from system.log import get_logger
from system.userspace import current_user_id, user_workspace_root

log = get_logger(__name__)

EMBED_DIMS = int(os.getenv("EMBED_DIMS", "640"))
KNOWLEDGE_DB_PATH = os.getenv("KNOWLEDGE_DB_PATH", "knowledge/knowledge.db")
KNOWLEDGE_RRF_K = int(os.getenv("KNOWLEDGE_RRF_K", "60"))
KNOWLEDGE_KNN_LIMIT = int(os.getenv("KNOWLEDGE_KNN_LIMIT", "20"))
KNOWLEDGE_FTS_LIMIT = int(os.getenv("KNOWLEDGE_FTS_LIMIT", "20"))
KNOWLEDGE_RECALL_SCORE_THRESHOLD = float(os.getenv("KNOWLEDGE_RECALL_SCORE_THRESHOLD", "0.012"))
KNOWLEDGE_CHUNK_CHARS = int(os.getenv("KNOWLEDGE_STORE_CHUNK_CHARS", os.getenv("KNOWLEDGE_CHUNK_CHARS", "900")))
KNOWLEDGE_CONTEXT_CHARS = int(os.getenv("KNOWLEDGE_CONTEXT_CHARS", "3500"))
KNOWLEDGE_KNN_MIN_SIMILARITY = float(os.getenv("KNOWLEDGE_KNN_MIN_SIMILARITY", "0.15"))
KNOWLEDGE_QUERY_INSTRUCT = os.getenv(
    "KNOWLEDGE_QUERY_INSTRUCT",
    "Retrieve durable learned knowledge relevant to the request",
).strip()
KNOWLEDGE_WORKSPACE_DIR = os.getenv("KNOWLEDGE_WORKSPACE_DIR", "library").strip().strip("/") or "library"
KNOWLEDGE_ENTITY_BOOST = float(os.getenv("KNOWLEDGE_ENTITY_BOOST", "0.003"))
KNOWLEDGE_WRITE_DEDUP_THRESHOLD = float(os.getenv("KNOWLEDGE_WRITE_DEDUP_THRESHOLD", "0.95"))
# Phase 18: near-dup → supersede old chunk instead of skip (when enabled).
KNOWLEDGE_SUPERSEDE_ON_DEDUP = os.getenv("KNOWLEDGE_SUPERSEDE_ON_DEDUP", "1").strip().lower() in {"1", "true", "yes", "on"}
KNOWLEDGE_SPREADING_ENABLED = os.getenv("KNOWLEDGE_SPREADING_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}
KNOWLEDGE_SPREADING_MAX_EXTRA = max(0, int(os.getenv("KNOWLEDGE_SPREADING_MAX_EXTRA", "2")))
KNOWLEDGE_SPREADING_SCORE_WEIGHT = float(os.getenv("KNOWLEDGE_SPREADING_SCORE_WEIGHT", "0.003"))

# ── search cache (mirrors memory/memorize.py's pattern) ─────────────────────

_KNOWLEDGE_SEARCH_CACHE: OrderedDict[
    tuple[str, str, int], tuple[float, list[dict]]
] = OrderedDict()
_KNOWLEDGE_SEARCH_CACHE_LOCK = threading.RLock()
_KNOWLEDGE_SEARCH_CACHE_TTL: float = 20.0
_KNOWLEDGE_SEARCH_CACHE_MAX: int = 128

_LAST_KNOWLEDGE_CLEAR_TIME: float = 0.0
_KNOWLEDGE_MIN_CLEAR_INTERVAL: float = 0.5  # seconds — debounce window


def _cache_key(query: str, user_id: str, limit: int) -> tuple[str, str, int]:
    return (user_id, " ".join((query or "").lower().split()), limit)


def _search_cache_get(query: str, user_id: str, limit: int) -> list[dict] | None:
    key = _cache_key(query, user_id, limit)
    now = time.monotonic()
    with _KNOWLEDGE_SEARCH_CACHE_LOCK:
        cached = _KNOWLEDGE_SEARCH_CACHE.get(key)
        if cached is not None and now - cached[0] <= _KNOWLEDGE_SEARCH_CACHE_TTL:
            _KNOWLEDGE_SEARCH_CACHE.move_to_end(key)
            return [dict(r) for r in cached[1]]
        if cached:
            _KNOWLEDGE_SEARCH_CACHE.pop(key, None)
    return None


def _search_cache_set(query: str, user_id: str, limit: int, results: list[dict]) -> None:
    key = _cache_key(query, user_id, limit)
    now = time.monotonic()
    with _KNOWLEDGE_SEARCH_CACHE_LOCK:
        _KNOWLEDGE_SEARCH_CACHE[key] = (now, [dict(r) for r in results])
        while len(_KNOWLEDGE_SEARCH_CACHE) > _KNOWLEDGE_SEARCH_CACHE_MAX:
            _KNOWLEDGE_SEARCH_CACHE.popitem(last=False)


def _maybe_clear_knowledge_cache() -> None:
    """Time-debounced invalidation: clear the cache on write, but only if
    at least _KNOWLEDGE_MIN_CLEAR_INTERVAL has elapsed since the last clear.

    Knowledge writes are rare (learn/research pipeline). After a write, the
    user is likely to ask about what was just taught — the debounce ensures
    the next read always sees fresh data at human-paced gaps, while batch
    writes within the same window (multiple chunks from one source) keep the
    cache warm.
    """
    global _LAST_KNOWLEDGE_CLEAR_TIME
    now = time.monotonic()
    if now - _LAST_KNOWLEDGE_CLEAR_TIME >= _KNOWLEDGE_MIN_CLEAR_INTERVAL:
        with _KNOWLEDGE_SEARCH_CACHE_LOCK:
            _KNOWLEDGE_SEARCH_CACHE.clear()
        _LAST_KNOWLEDGE_CLEAR_TIME = now


class Embedder(Protocol):
    def embed_query(self, text: str, instruct: str = "") -> object: ...
    def embed_queries(self, texts: list[str], instruct: str = "") -> object: ...


_DDL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS learned_docs (
    id          TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    title       TEXT NOT NULL,
    source      TEXT NOT NULL DEFAULT '',
    kind        TEXT NOT NULL DEFAULT 'ingested',
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS learned_chunks (
    id              TEXT PRIMARY KEY,
    doc_id          TEXT NOT NULL REFERENCES learned_docs(id) ON DELETE CASCADE,
    user_id         TEXT NOT NULL,
    chunk_index     INTEGER NOT NULL,
    text            TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    access_count    INTEGER NOT NULL DEFAULT 0,
    last_accessed   TEXT,
    entities        TEXT NOT NULL DEFAULT '[]',
    status          TEXT NOT NULL DEFAULT 'active',
    supersedes_id   TEXT
);

CREATE INDEX IF NOT EXISTS idx_learned_docs_user ON learned_docs(user_id);
CREATE INDEX IF NOT EXISTS idx_learned_chunks_doc ON learned_chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_learned_chunks_user ON learned_chunks(user_id);
CREATE INDEX IF NOT EXISTS idx_learned_chunks_created ON learned_chunks(created_at);

CREATE VIRTUAL TABLE IF NOT EXISTS learned_chunks_fts USING fts5(
    text,
    id UNINDEXED,
    content='learned_chunks',
    content_rowid='rowid'
);

CREATE VIRTUAL TABLE IF NOT EXISTS learned_chunks_vec USING vec0(
    id TEXT PRIMARY KEY,
    embedding FLOAT[{dims}]
);

-- Archive table for cold storage of rarely-accessed chunks
CREATE TABLE IF NOT EXISTS learned_chunks_archive (
    id              TEXT PRIMARY KEY,
    doc_id          TEXT NOT NULL,
    user_id         TEXT NOT NULL,
    chunk_index     INTEGER NOT NULL,
    text            TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    access_count    INTEGER NOT NULL DEFAULT 0,
    last_accessed   TEXT,
    archived_at     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_learned_chunks_archive_user ON learned_chunks_archive(user_id);
CREATE INDEX IF NOT EXISTS idx_learned_chunks_archive_created ON learned_chunks_archive(created_at);

CREATE TRIGGER IF NOT EXISTS learned_chunks_ai AFTER INSERT ON learned_chunks BEGIN
    INSERT INTO learned_chunks_fts(rowid, text, id) VALUES (new.rowid, new.text, new.id);
END;

CREATE TRIGGER IF NOT EXISTS learned_chunks_ad AFTER DELETE ON learned_chunks BEGIN
    INSERT INTO learned_chunks_fts(learned_chunks_fts, rowid, text, id)
    VALUES ('delete', old.rowid, old.text, old.id);
    DELETE FROM learned_chunks_vec WHERE id = old.id;
END;

CREATE TRIGGER IF NOT EXISTS learned_chunks_au AFTER UPDATE OF text ON learned_chunks BEGIN
    INSERT INTO learned_chunks_fts(learned_chunks_fts, rowid, text, id)
    VALUES ('delete', old.rowid, old.text, old.id);
    INSERT INTO learned_chunks_fts(rowid, text, id) VALUES (new.rowid, new.text, new.id);
END;
""".format(dims=EMBED_DIMS)


def _connect(user_id: str | None = None) -> sqlite3.Connection:
    conn = initialize_store_db(KNOWLEDGE_DB_PATH, _DDL, user_id=user_id, vector=True)
    _ensure_knowledge_schema_migrated(conn, user_id)
    return conn


def _ensure_knowledge_schema_migrated(conn: sqlite3.Connection, user_id: str | None = None) -> None:
    """Add missing columns and indexes to knowledge tables."""
    cols = [r[1] for r in conn.execute("PRAGMA table_info(learned_chunks)").fetchall()]
    if "access_count" not in cols:
        conn.execute("ALTER TABLE learned_chunks ADD COLUMN access_count INTEGER NOT NULL DEFAULT 0")
        log.info("[knowledge] Added missing access_count column to learned_chunks")
    if "last_accessed" not in cols:
        conn.execute("ALTER TABLE learned_chunks ADD COLUMN last_accessed TEXT")
        log.info("[knowledge] Added missing last_accessed column to learned_chunks")
    if "archived_at" not in cols:
        conn.execute("ALTER TABLE learned_chunks ADD COLUMN archived_at TEXT")
        log.info("[knowledge] Added missing archived_at column to learned_chunks")
    if "entities" not in cols:
        conn.execute("ALTER TABLE learned_chunks ADD COLUMN entities TEXT NOT NULL DEFAULT '[]'")
    if "status" not in cols:
        conn.execute("ALTER TABLE learned_chunks ADD COLUMN status TEXT NOT NULL DEFAULT 'active'")
    if "supersedes_id" not in cols:
        conn.execute("ALTER TABLE learned_chunks ADD COLUMN supersedes_id TEXT")

    conn.execute("CREATE INDEX IF NOT EXISTS idx_learned_chunks_access ON learned_chunks(access_count, last_accessed)")

    # Check learned_chunks_archive table
    try:
        conn.execute("SELECT COUNT(*) FROM learned_chunks_archive LIMIT 1")
        try:
            archive_cols = [r[1] for r in conn.execute("PRAGMA table_info(learned_chunks_archive)").fetchall()]
            if "access_count" not in archive_cols:
                conn.execute("ALTER TABLE learned_chunks_archive ADD COLUMN access_count INTEGER NOT NULL DEFAULT 0")
            if "last_accessed" not in archive_cols:
                conn.execute("ALTER TABLE learned_chunks_archive ADD COLUMN last_accessed TEXT")
        except sqlite3.OperationalError:
            log.debug("knowledge: archive column migration failed (expected on first run)")
    except sqlite3.OperationalError:
        log.debug("knowledge: archive table does not exist yet")

    conn.commit()


def _now() -> str:
    return utc_now_iso()


def _sanitize_text(text: str, max_chars: int = 200_000) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())[:max_chars]





def _safe_workspace_path(relative_path: str, user_id: str | None = None) -> Path:
    root = user_workspace_root(user_id).resolve()
    target = (root / relative_path).expanduser().resolve()
    if root not in target.parents and target != root:
        raise ValueError("path must stay inside the user workspace")
    return target


def _xml_text_from_zip(path: Path, members: Iterable[str]) -> str:
    chunks: list[str] = []
    with zipfile.ZipFile(path) as zf:
        for member in members:
            try:
                data = zf.read(member)
            except KeyError:
                continue
            root = DET.fromstring(data)
            chunks.extend(t.strip() for t in root.itertext() if t and t.strip())
    return "\n".join(chunks)


def _xlsx_text(path: Path) -> str:
    with zipfile.ZipFile(path) as zf:
        shared: list[str] = []
        try:
            root = DET.fromstring(zf.read("xl/sharedStrings.xml"))
            shared = [" ".join(t.strip() for t in si.itertext() if t and t.strip()) for si in root]
        except KeyError:
            log.debug("knowledge: xlsx has no shared strings")
        out: list[str] = []
        for name in sorted(n for n in zf.namelist() if n.startswith("xl/worksheets/") and n.endswith(".xml")):
            root = DET.fromstring(zf.read(name))
            for c in root.iter():
                if not c.tag.endswith("}c"):
                    continue
                cell_type = c.attrib.get("t")
                value = None
                for child in c:
                    if child.tag.endswith("}v"):
                        value = child.text
                        break
                if value is None:
                    continue
                if cell_type == "s":
                    try:
                        value = shared[int(value)]
                    except Exception:
                        log.warning("knowledge: failed to decode shared string")
                out.append(str(value))
    return "\n".join(out)


def _epub_text(path: Path) -> str:
    texts: list[str] = []
    with zipfile.ZipFile(path) as zf:
        for name in sorted(zf.namelist()):
            if name.casefold().endswith((".xhtml", ".html", ".htm")):
                raw = zf.read(name).decode("utf-8", errors="replace")
                raw = re.sub(r"<script[\s\S]*?</script>|<style[\s\S]*?</style>", " ", raw, flags=re.I)
                texts.append(re.sub(r"<[^>]+>", " ", raw))
    return "\n".join(texts)


def extract_text_from_file(relative_path: str, *, user_id: str | None = None, max_chars: int = 200_000) -> tuple[str, str]:
    """Extract text from a workspace document for learned-knowledge ingest.

    Supports plain text/Markdown/config files directly, HTML via trafilatura
    when installed, and PDF via pypdf/PyPDF2 when installed. Returns
    (text, source_path). Raises ValueError with a user-facing reason when the
    file cannot be read/extracted.
    """
    path = _safe_workspace_path(relative_path, user_id)
    if not path.is_file():
        raise ValueError(f"workspace file not found: {relative_path}")
    suffix = path.suffix.casefold()
    if suffix in {".txt", ".md", ".rst", ".json", ".yaml", ".yml", ".toml", ".csv", ".tsv", ".log", ".py", ".js", ".ts", ".html", ".htm", ".tex", ".latex", ".rtf"}:
        raw = path.read_text(encoding="utf-8", errors="replace")
        if suffix in {".html", ".htm"}:
            try:
                import trafilatura  # type: ignore
                raw = trafilatura.extract(raw, include_links=False, include_tables=False) or raw
            except Exception:
                log.warning("knowledge: trafilatura extraction failed")
        return _sanitize_text(raw, max_chars), str(path.relative_to(user_workspace_root(user_id)))
    if suffix == ".docx":
        text = _xml_text_from_zip(path, ["word/document.xml"])
        return _sanitize_text(text, max_chars), str(path.relative_to(user_workspace_root(user_id)))
    if suffix == ".xlsx":
        return _sanitize_text(_xlsx_text(path), max_chars), str(path.relative_to(user_workspace_root(user_id)))
    if suffix == ".epub":
        return _sanitize_text(_epub_text(path), max_chars), str(path.relative_to(user_workspace_root(user_id)))
    if suffix == ".pdf":
        reader_cls = None
        try:
            from pypdf import PdfReader  # type: ignore
            reader_cls = PdfReader
        except Exception:
            try:
                from PyPDF2 import PdfReader  # type: ignore
                reader_cls = PdfReader
            except Exception as exc:
                raise ValueError("PDF ingest needs pypdf or PyPDF2 installed") from exc
        reader = reader_cls(str(path))
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
            if sum(len(p) for p in pages) >= max_chars:
                break
        return _sanitize_text("\n\n".join(pages), max_chars), str(path.relative_to(user_workspace_root(user_id)))
    raise ValueError(f"unsupported knowledge file type: {suffix or 'no extension'}")


def ingest_file(
    relative_path: str,
    *,
    title: str | None = None,
    kind: str = "ingested",
    embedder: Embedder | None = None,
    user_id: str | None = None,
) -> str | None:
    """Extract a workspace file and store it in learned knowledge RAG."""
    try:
        text, source = extract_text_from_file(relative_path, user_id=user_id)
    except ValueError as exc:
        log.warning("Failed to extract knowledge file %s: %s", relative_path, exc)
        return None
    return ingest_text(title or Path(relative_path).stem.replace("_", " ").title(), text, source=source, kind=kind, embedder=embedder, user_id=user_id)

def ingest_text(
    title: str,
    text: str,
    *,
    source: str = "",
    kind: str = "ingested",
    embedder: Embedder | None = None,
    user_id: str | None = None,
) -> str | None:
    """Chunk, embed, and persist durable learned knowledge."""
    clean = _sanitize_text(text)
    if not clean:
        return None
    uid = user_id or current_user_id()
    doc_id = str(uuid.uuid4())
    created_at = _now()
    chunks = reason.chunk_text(clean, KNOWLEDGE_CHUNK_CHARS) or [clean]
    conn = _connect(uid)
    try:
        conn.execute(
            "INSERT INTO learned_docs(id,user_id,title,source,kind,created_at) VALUES(?,?,?,?,?,?)",
            (doc_id, uid, (title or "Untitled knowledge")[:200], source[:500], kind[:50], created_at),
        )
        vectors = []
        if embedder is not None:
            batch = reason.embed_batch_or_none(embedder, chunks)
            vectors = list(batch) if batch is not None and len(batch) == len(chunks) else []

        for index, chunk in enumerate(chunks):  # always loop chunks
            ents_json = entities_to_json(extract_entities(chunk))
            vec = vectors[index] if index < len(vectors) else None

            supersedes_id = None
            if KNOWLEDGE_WRITE_DEDUP_THRESHOLD > 0 and vec is not None:
                try:
                    neighbors = user_scoped_vec_knn(
                        conn,
                        vec_table="learned_chunks_vec",
                        owner_table="learned_chunks",
                        owner_alias="c",
                        vector=vec,
                        user_id=uid,
                        limit=1,
                    )
                    if neighbors:
                        dist = float(neighbors[0]["dist"])
                        sim = 1.0 - dist
                        if sim >= KNOWLEDGE_WRITE_DEDUP_THRESHOLD:
                            old_id = str(neighbors[0]["id"])
                            if KNOWLEDGE_SUPERSEDE_ON_DEDUP and old_id:
                                row = conn.execute(
                                    "SELECT status FROM learned_chunks WHERE id = ? AND user_id = ?",
                                    (old_id, uid),
                                ).fetchone()
                                st = "active"
                                if row is not None:
                                    try:
                                        st = (row["status"] or "active")
                                    except Exception:
                                        st = "active"
                                if str(st).strip().lower() == "superseded":
                                    # Already replaced — skip insert (true dedup) or treat as no parent
                                    log.debug("knowledge dedup skip already-superseded sim=%.3f", sim)
                                    continue
                                supersedes_id = old_id
                                try:
                                    conn.execute(
                                        "UPDATE learned_chunks SET status = 'superseded' "
                                        "WHERE id = ? AND user_id = ? "
                                        "AND (status = 'active' OR status IS NULL)",
                                        (old_id, uid),
                                    )
                                except Exception as sup_exc:
                                    log.debug("knowledge supersede mark failed: %s", sup_exc)
                                log.debug("knowledge supersede chunk sim=%.3f old=%s", sim, old_id[:8])
                            else:
                                log.debug("knowledge dedup skip chunk sim=%.3f", sim)
                                continue
                except Exception as dedup_exc:
                    log.debug("knowledge dedup skipped: %s", dedup_exc)

            chunk_id = str(uuid.uuid4())
            conn.execute(
                "INSERT INTO learned_chunks(id,doc_id,user_id,chunk_index,text,created_at,entities,status,supersedes_id) VALUES(?,?,?,?,?,?,?,?,?)",
                (chunk_id, doc_id, uid, index, chunk, created_at, ents_json, "active", supersedes_id),
            )
            if vec is not None:
                insert_vector(conn, "learned_chunks_vec", chunk_id, vec)
        conn.commit()
        _maybe_clear_knowledge_cache()
        return doc_id
    except Exception as exc:
        conn.rollback()
        log.warning("Failed to ingest knowledge: %s", exc)
        return None
    finally:
        conn.close()



def _knowledge_sources(conn: sqlite3.Connection, uid: str) -> set[str]:
    rows = conn.execute("SELECT source FROM learned_docs WHERE user_id=?", (uid,)).fetchall()
    return {str(row["source"]) for row in rows if row["source"]}


def ingest_workspace_knowledge_folder(*, embedder: Embedder | None = None, user_id: str | None = None) -> list[str]:
    """Ingest new files dropped under <workspace>/knowledge into the KB DB.

    The scan is idempotent by source path: files already present in learned_docs.source
    are skipped. Unsupported files are logged and left in place for a future run.
    """
    uid = user_id or current_user_id()
    root = user_workspace_root(uid)
    folder = (root / KNOWLEDGE_WORKSPACE_DIR).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    conn = _connect(uid)
    try:
        known = _knowledge_sources(conn, uid)
    finally:
        conn.close()
    doc_ids: list[str] = []
    for path in sorted(p for p in folder.rglob("*") if p.is_file() and not p.name.startswith(".")):
        rel = str(path.relative_to(root))
        if rel in known:
            continue
        try:
            doc_id = ingest_file(rel, kind="workspace_drop", embedder=embedder, user_id=uid)
        except Exception as exc:
            log.warning("skipping workspace knowledge file %s: %s", rel, exc)
            continue
        if doc_id:
            known.add(rel)
            doc_ids.append(doc_id)
            log.info("ingested workspace knowledge file %s as %s", rel, doc_id)
    return doc_ids


def _knn(conn: sqlite3.Connection, query: str, embedder: Embedder | None, uid: str, limit: int) -> list[sqlite3.Row]:
    if embedder is None or not (query or "").strip():
        return []
    vector = embedder.embed_query(query, instruct=KNOWLEDGE_QUERY_INSTRUCT)
    return user_scoped_vec_knn(
        conn,
        vec_table="learned_chunks_vec",
        owner_table="learned_chunks",
        owner_alias="c",
        vector=vector,
        user_id=uid,
        limit=limit,
        threshold=KNOWLEDGE_KNN_MIN_SIMILARITY,
    )


def _fts(conn: sqlite3.Connection, query: str, uid: str, limit: int) -> list[sqlite3.Row]:
    return user_scoped_fts_search(
        conn,
        fts_table="learned_chunks_fts",
        owner_table="learned_chunks",
        owner_alias="c",
        query=query,
        user_id=uid,
        limit=limit,
    )



def _knowledge_spread_extra(conn: sqlite3.Connection, uid: str, hits: list[dict], limit: int) -> list[dict]:
    """Phase 18 optional: pull active chunks sharing an entity with hits."""
    if not KNOWLEDGE_SPREADING_ENABLED or limit <= 0 or not hits:
        return []
    seen = {str(h.get("id") or "") for h in hits}
    entities: set[str] = set()
    for h in hits:
        try:
            for e in entities_from_json(h.get("entities") or "[]"):
                if e:
                    entities.add(str(e).strip().lower())
        except Exception:
            continue
    if not entities:
        return []
    extra: list[dict] = []
    try:
        rows = conn.execute(
            "SELECT id, text, chunk_index, created_at, entities, status FROM learned_chunks "
            "WHERE user_id = ? AND (status = 'active' OR status IS NULL) LIMIT 200",
            (uid,),
        ).fetchall()
        for row in rows:
            rid = str(row["id"])
            if rid in seen:
                continue
            try:
                ents = {str(e).strip().lower() for e in entities_from_json(row["entities"] or "[]") if e}
            except Exception:
                continue
            if len(ents & entities) < 2:
                continue
            d = dict(row)
            d["score"] = float(KNOWLEDGE_SPREADING_SCORE_WEIGHT)
            d["_from_spreading"] = True
            extra.append(d)
            seen.add(rid)
            if len(extra) >= limit:
                break
    except Exception as exc:
        log.debug("knowledge spreading skipped: %s", exc)
    return extra


def search_knowledge(
    query: str,
    limit: int = 5,
    *,
    embedder: Embedder | None = None,
    user_id: str | None = None,
) -> list[dict]:
    uid = user_id or current_user_id()
    conn = _connect(uid)
    try:
        rank_knn = rank_by_id(_knn(conn, query, embedder, uid, KNOWLEDGE_KNN_LIMIT))
        rank_fts = rank_by_id(_fts(conn, query, uid, KNOWLEDGE_FTS_LIMIT))
        ids = set(rank_knn) | set(rank_fts)
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        rows = conn.execute(
            f"""
            SELECT c.id, c.text, c.chunk_index, c.created_at, c.entities, c.status,
                d.title, d.source, d.kind, d.id AS doc_id
            FROM learned_chunks c
            JOIN learned_docs d ON d.id = c.doc_id
            WHERE c.id IN ({placeholders})
                AND (c.status = 'active' OR c.status IS NULL)
            """,
            list(ids),
        ).fetchall()
        by_id = {row["id"]: row for row in rows}
        scored: list[tuple[float, str]] = []
        for cid in ids:
            score = rrf_score(cid, rank_knn, rank_fts, k=KNOWLEDGE_RRF_K)
            row = by_id.get(cid)
            if row is None:
                continue
            ents = entities_from_json(row["entities"] if "entities" in row.keys() else "[]")
            score += KNOWLEDGE_ENTITY_BOOST * entity_overlap_score(query, ents)
            if score >= KNOWLEDGE_RECALL_SCORE_THRESHOLD:
                scored.append((score, cid))
        scored.sort(key=lambda pair: (-pair[0], by_id[pair[1]]["created_at"]))
        results = [dict(by_id[cid]) | {"score": score} for score, cid in scored[:limit]]
        if KNOWLEDGE_SPREADING_ENABLED and KNOWLEDGE_SPREADING_MAX_EXTRA > 0:
            results.extend(_knowledge_spread_extra(conn, uid, results, KNOWLEDGE_SPREADING_MAX_EXTRA))
        return results
    except Exception as exc:
        log.warning("Knowledge search failed: %s", exc)
        return []
    finally:
        conn.close()

def _attr(value: object) -> str:
    return escape(str(value or ""), quote=True)


def knowledge_context_for(
    query: str,
    limit: int = 5,
    max_chars: int | None = None,
    embedder: Embedder | None = None,
    user_id: str | None = None,
) -> str:
    """Retrieve knowledge context for a query, tracking access counts.

    Args:
        query: Search query
        limit: Max number of results
        max_chars: Max total characters in returned context (default: KNOWLEDGE_CONTEXT_CHARS)
        embedder: Optional embedder for vector search
        user_id: Optional user ID (defaults to current_user_id)
    """
    uid = user_id or current_user_id()
    remaining = max_chars or KNOWLEDGE_CONTEXT_CHARS

    cached = _search_cache_get(query, uid, limit)
    if cached is not None:
        # Track access for cached results too
        chunk_ids = [r["id"] for r in cached]
        _increment_access_count(chunk_ids, uid)
        return _format_knowledge_context(cached, remaining)

    hits = search_knowledge(query, limit=limit, embedder=embedder, user_id=uid)
    if hits:
        _search_cache_set(query, uid, limit, hits)

    if not hits:
        return "<knowledge_context>\nNo matching learned knowledge found.\n</knowledge_context>"

    # Track access for non-cached results
    chunk_ids = [h["id"] for h in hits]
    _increment_access_count(chunk_ids, uid)

    return _format_knowledge_context(hits, remaining)


# ── Access tracking & maintenance ─────────────────────────────────────────────


def _increment_access_count(chunk_ids: list[str], user_id: str | None = None) -> None:
    """Increment access count and update last_accessed for given chunk IDs."""
    if not chunk_ids:
        return
    uid = user_id or current_user_id()
    conn = _connect(uid)
    try:
        now = utc_now_iso()
        placeholders = ",".join("?" * len(chunk_ids))
        conn.execute(
            f"UPDATE learned_chunks SET access_count = access_count + 1, last_accessed = ? WHERE id IN ({placeholders})",
            [now] + chunk_ids,
        )
        conn.commit()
    except Exception as exc:
        log.warning("increment access count failed: %s", exc)
    finally:
        conn.close()


def _format_knowledge_context(results: list[dict], max_chars: int | None = None) -> str:
    remaining = max_chars or KNOWLEDGE_CONTEXT_CHARS
    blocks = []
    for row in results:
        if remaining <= 0:
            break
        body = row["text"][:remaining]
        blocks.append(
            f'<knowledge_chunk doc_id="{_attr(row.get("doc_id", ""))}" title="{_attr(row.get("title", ""))}" '
            f'kind="{_attr(row.get("kind", ""))}" source="{_attr(row.get("source", ""))}" score="{row.get("score", row.get("recall_score", 0)):.4f}">\n'
            f'{body}\n</knowledge_chunk>'
        )
        remaining -= len(body)
    return "<knowledge_context>\n" + "\n\n".join(blocks) + "\n</knowledge_context>"


# ── Monthly maintenance: prune & archive ──────────────────────────────────────

def prune_knowledge(
    *,
    keep_days: int = 30,
    min_access: int = 2,
    archive_days: int = 90,
    delete_days: int = 180,
    dedupe_threshold: float = 0.95,
    user_id: str | None = None,
    embedder=None,
) -> dict:
    """
    Prune knowledge DB: archive cold chunks, delete never-accessed old chunks,
    deduplicate near-duplicates. Returns stats dict.
    """
    uid = user_id or current_user_id()
    conn = _connect(uid)
    stats = {"archived": 0, "deleted": 0, "deduped": 0, "errors": 0}
    now = utc_now_iso()
    try:
        # 1. Archive: move cold chunks (old + low access) to archive table
        archive_cutoff = (datetime.fromisoformat(now.replace('Z', '+00:00')) - timedelta(days=archive_days)).isoformat()
        conn.execute(
            """
            INSERT INTO learned_chunks_archive
            (id, doc_id, user_id, chunk_index, text, created_at, access_count, last_accessed, archived_at)
            SELECT id, doc_id, user_id, chunk_index, text, created_at, access_count, last_accessed, ?
            FROM learned_chunks
            WHERE user_id = ?
              AND created_at < ?
              AND access_count < ?
              AND id NOT IN (SELECT id FROM learned_chunks_archive WHERE user_id = ?)
            """,
            (now, uid, archive_cutoff, min_access, uid),
        )
        stats["archived"] = conn.total_changes

        # Delete archived from main table
        conn.execute(
            "DELETE FROM learned_chunks WHERE id IN (SELECT id FROM learned_chunks_archive WHERE user_id = ? AND archived_at = ?)",
            (uid, now),
        )

        # 2. Delete: remove never-accessed chunks older than delete_days
        delete_cutoff = (datetime.fromisoformat(now.replace('Z', '+00:00')) - timedelta(days=delete_days)).isoformat()
        conn.execute(
            """
            DELETE FROM learned_chunks
            WHERE user_id = ?
              AND created_at < ?
              AND access_count = 0
              AND id NOT IN (SELECT id FROM learned_chunks_archive WHERE user_id = ?)
            """,
            (uid, delete_cutoff, uid),
        )
        stats["deleted"] = conn.total_changes - stats["archived"]

        # 3. Deduplicate: find near-duplicate chunks via embedding similarity
        if embedder is not None:
            stats["deduped"] = _deduplicate_chunks(conn, uid, embedder, dedupe_threshold)

        conn.commit()
    except Exception as exc:
        conn.rollback()
        log.warning("prune_knowledge failed: %s", exc)
        stats["errors"] = 1
    finally:
        conn.close()
    return stats


def _deduplicate_chunks(conn: sqlite3.Connection, user_id: str, embedder, threshold: float) -> int:
    """Find near-duplicate chunks via embedding similarity and archive duplicates."""
    rows = conn.execute(
        """
        SELECT id, text, chunk_index FROM learned_chunks
        WHERE user_id = ? AND id NOT IN (SELECT id FROM learned_chunks_archive WHERE user_id = ?)
        ORDER BY chunk_index
        """,
        (user_id, user_id),
    ).fetchall()

    if len(rows) < 2:
        return 0

    texts = [row["text"] for row in rows]
    ids = [row["id"] for row in rows]
    indices = [row["chunk_index"] for row in rows]

    try:
        if hasattr(embedder, "embed_queries"):
            batch = embedder.embed_queries(texts)
        else:
            batch = [embedder.embed_query(t) for t in texts]
        vectors = np.array(batch, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        vectors = vectors / norms
        sim = vectors @ vectors.T
    except Exception:
        return 0

    to_archive = set()
    n = len(ids)
    for i in range(n):
        if ids[i] in to_archive:
            continue
        for j in range(i + 1, n):
            if ids[j] in to_archive:
                continue
            if sim[i, j] >= threshold:
                # Archive the one with higher chunk_index (later occurrence)
                if indices[i] < indices[j]:
                    to_archive.add(ids[j])
                else:
                    to_archive.add(ids[i])

    if to_archive:
        now = utc_now_iso()
        for cid in to_archive:
            row = conn.execute("SELECT * FROM learned_chunks WHERE id = ?", (cid,)).fetchone()
            if row:
                conn.execute(
                    """
                    INSERT INTO learned_chunks_archive
                    (id, doc_id, user_id, chunk_index, text, created_at, access_count, last_accessed, archived_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (row["id"], row["doc_id"], row["user_id"], row["chunk_index"],
                     row["text"], row["created_at"], row["access_count"],
                     row["last_accessed"], now),
                )
                conn.execute("DELETE FROM learned_chunks WHERE id = ?", (cid,))

    return len(to_archive)


def vacuum_knowledge_db(user_id: str | None = None) -> None:
    """VACUUM the knowledge DB to reclaim space after deletions."""
    uid = user_id or current_user_id()
    conn = _connect(uid)
    try:
        conn.execute("VACUUM")
        conn.execute("ANALYZE")
    finally:
        conn.close()
