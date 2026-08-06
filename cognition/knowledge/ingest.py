"""Knowledge ingestion and source extraction."""
from __future__ import annotations

import re
import sqlite3
import uuid
import zipfile
from pathlib import Path
from typing import Iterable

from defusedxml import ElementTree as DET

from cognition import reason
from cognition.memory.vecstore import insert_vector, user_scoped_vec_knn
from cognition.memory.memorize import extract_entities, entities_to_json
from system.log import get_logger
from system.userspace import current_user_id, user_workspace_root

from .schema import (
    connect,
    Embedder,
    KNOWLEDGE_CHUNK_CHARS,
    KNOWLEDGE_SUPERSEDE_ON_DEDUP,
    KNOWLEDGE_WORKSPACE_DIR,
    KNOWLEDGE_WRITE_DEDUP_THRESHOLD,
    KnowledgeSchema,
    now,
)
from .search import _maybe_clear_knowledge_cache

log = get_logger(__name__)


class KnowledgeIngest:
    """Owns file extraction and learned-knowledge write path."""

    def __init__(self, schema: KnowledgeSchema | None = None, embedder: Embedder | None = None):
        self.schema = schema or KnowledgeSchema()
        self.embedder = embedder

    def extract_text_from_file(
        self, relative_path: str, *, user_id: str | None = None, max_chars: int = 200_000
    ) -> tuple[str, str]:
        return extract_text_from_file(relative_path, user_id=user_id, max_chars=max_chars)

    def ingest_file(
        self,
        relative_path: str,
        *,
        title: str | None = None,
        kind: str = "ingested",
        embedder: Embedder | None = None,
        user_id: str | None = None,
    ) -> str | None:
        return ingest_file(
            relative_path,
            title=title,
            kind=kind,
            embedder=embedder if embedder is not None else self.embedder,
            user_id=user_id,
            schema=self.schema,
        )
    
    def ingest_text(
        self,
        title: str,
        text: str,
        *,
        source: str = "",
        kind: str = "ingested",
        embedder: Embedder | None = None,
        user_id: str | None = None,
    ) -> str | None:
        return ingest_text(
            title,
            text,
            source=source,
            kind=kind,
            embedder=embedder if embedder is not None else self.embedder,
            user_id=user_id,
            schema=self.schema,
        )
    
    def ingest_workspace_knowledge_folder(
        self, *, embedder: Embedder | None = None, user_id: str | None = None
    ) -> list[str]:
        return ingest_workspace_knowledge_folder(
            embedder=embedder if embedder is not None else self.embedder,
            user_id=user_id,
            schema=self.schema,
        )

def _sanitize_text(text: str, max_chars: int = 200_000) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())[:max_chars]


def _safe_workspace_path(relative_path: str, user_id: str | None = None) -> Path:
    root = user_workspace_root(user_id).resolve()
    target = (root / relative_path).expanduser().resolve()
    if root not in target.parents and target != root:
        raise ValueError("path must stay inside the user workspace")
    return target


def _xml_text_from_zip(path: Path, members: Iterable[str], max_chars: int = 200_000) -> str:
    chunks: list[str] = []
    accumulated = 0
    with zipfile.ZipFile(path) as zf:
        for member in members:
            if accumulated >= max_chars:
                break
            try:
                info = zf.getinfo(member)
                remaining_budget = max_chars - accumulated
                if info.file_size > remaining_budget:
                    # Skip members that exceed remaining budget
                    continue
                data = zf.read(member)
            except KeyError:
                continue
            root = DET.fromstring(data)
            for t in root.itertext():
                if t and t.strip():
                    text = t.strip()
                    chunks.append(text)
                    accumulated += len(text) + 1  # +1 for newline
                    if accumulated >= max_chars:
                        break
    return "\n".join(chunks)


def _xlsx_text(path: Path, max_chars: int = 200_000) -> str:
    accumulated = 0
    with zipfile.ZipFile(path) as zf:
        shared: list[str] = []
        try:
            info = zf.getinfo("xl/sharedStrings.xml")
            if info.file_size <= max_chars:
                root = DET.fromstring(zf.read("xl/sharedStrings.xml"))
                shared = [" ".join(t.strip() for t in si.itertext() if t and t.strip()) for si in root]
        except KeyError:
            log.debug("knowledge: xlsx has no shared strings")
        out: list[str] = []
        for name in sorted(n for n in zf.namelist() if n.startswith("xl/worksheets/") and n.endswith(".xml")):
            if accumulated >= max_chars:
                break
            try:
                info = zf.getinfo(name)
                remaining_budget = max_chars - accumulated
                if info.file_size > remaining_budget:
                    continue
                root = DET.fromstring(zf.read(name))
            except KeyError:
                continue
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
                value_str = str(value)
                out.append(value_str)
                accumulated += len(value_str) + 1  # +1 for newline
                if accumulated >= max_chars:
                    break
    return "\n".join(out)


def _epub_text(path: Path, max_chars: int = 200_000) -> str:
    texts: list[str] = []
    accumulated = 0
    with zipfile.ZipFile(path) as zf:
        for name in sorted(zf.namelist()):
            if accumulated >= max_chars:
                break
            if name.casefold().endswith((".xhtml", ".html", ".htm")):
                try:
                    info = zf.getinfo(name)
                    remaining_budget = max_chars - accumulated
                    if info.file_size > remaining_budget:
                        continue
                    raw = zf.read(name).decode("utf-8", errors="replace")
                except KeyError:
                    continue
                raw = re.sub(r"<script[\s\S]*?</script>|<style[\s\S]*?</style>", " ", raw, flags=re.I)
                cleaned = re.sub(r"<[^>]+>", " ", raw)
                texts.append(cleaned)
                accumulated += len(cleaned) + 1  # +1 for newline
                if accumulated >= max_chars:
                    break
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
        try:
            text = _xml_text_from_zip(path, ["word/document.xml"], max_chars)
            return _sanitize_text(text, max_chars), str(path.relative_to(user_workspace_root(user_id)))
        except Exception as exc:
            raise ValueError(f"Failed to extract .docx file: {exc}") from exc
    if suffix == ".xlsx":
        try:
            return _sanitize_text(_xlsx_text(path, max_chars), max_chars), str(path.relative_to(user_workspace_root(user_id)))
        except Exception as exc:
            raise ValueError(f"Failed to extract .xlsx file: {exc}") from exc
    if suffix == ".epub":
        try:
            return _sanitize_text(_epub_text(path, max_chars), max_chars), str(path.relative_to(user_workspace_root(user_id)))
        except Exception as exc:
            raise ValueError(f"Failed to extract .epub file: {exc}") from exc
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
        try:
            reader = reader_cls(str(path))
            pages = []
            for page in reader.pages:
                pages.append(page.extract_text() or "")
                if sum(len(p) for p in pages) >= max_chars:
                    break
            return _sanitize_text("\n\n".join(pages), max_chars), str(path.relative_to(user_workspace_root(user_id)))
        except Exception as exc:
            raise ValueError(f"Failed to read or extract PDF file: {exc}") from exc
    raise ValueError(f"unsupported knowledge file type: {suffix or 'no extension'}")


def ingest_file(
    relative_path: str,
    *,
    title: str | None = None,
    kind: str = "ingested",
    embedder: Embedder | None = None,
    user_id: str | None = None,
    schema: KnowledgeSchema | None = None,
) -> str | None:
    """Extract a workspace file and store it in learned knowledge RAG."""
    try:
        text, source = extract_text_from_file(relative_path, user_id=user_id)
    except ValueError as exc:
        log.warning("Failed to extract knowledge file %s: %s", relative_path, exc)
        return None
    return ingest_text(title or Path(relative_path).stem.replace("_", " ").title(), text, source=source, kind=kind, embedder=embedder, user_id=user_id, schema=schema,)

def ingest_text(
    title: str,
    text: str,
    *,
    source: str = "",
    kind: str = "ingested",
    embedder: Embedder | None = None,
    user_id: str | None = None,
    schema: KnowledgeSchema | None = None,
) -> str | None:
    """Chunk, embed, and persist durable learned knowledge."""
    clean = _sanitize_text(text)
    if not clean:
        return None
    uid = user_id or current_user_id()
    doc_id = str(uuid.uuid4())
    created_at = now()
    chunks = reason.chunk_text(clean, KNOWLEDGE_CHUNK_CHARS) or [clean]
    conn = connect(uid)
    try:
        conn.execute(
            "INSERT INTO learned_docs(id,user_id,title,source,kind,created_at) VALUES(?,?,?,?,?,?)",
            (doc_id, uid, (title or "Untitled knowledge")[:200], source[:500], kind[:50], created_at),
        )
        vectors = []
        if embedder is not None:
            batch = reason.embed_batch_or_none(embedder, chunks)
            vectors = list(batch) if batch is not None and len(batch) == len(chunks) else []

        written_chunks = 0
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
            written_chunks += 1
            if vec is not None:
                insert_vector(conn, "learned_chunks_vec", chunk_id, vec)

        if written_chunks == 0:
            # No chunks were written, rollback the document insert
            conn.rollback()
            return None

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
    conn = connect(uid)
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
