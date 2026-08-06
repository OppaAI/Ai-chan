"""Schema, connection, and shared constants for the knowledge store."""
from __future__ import annotations

import os
import sqlite3
from typing import Protocol

from system.config import load_config
load_config()

from cognition.memory.vecstore import initialize_store_db, utc_now_iso
from system.log import get_logger
from system.userspace import current_user_id

log = get_logger(__name__)

# ── config ───────────────────────────────────────────────────────────────────

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
KNOWLEDGE_SUPERSEDE_ON_DEDUP = os.getenv("KNOWLEDGE_SUPERSEDE_ON_DEDUP", "1").strip().lower() in {"1", "true", "yes", "on"}
KNOWLEDGE_SPREADING_ENABLED = os.getenv("KNOWLEDGE_SPREADING_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}
KNOWLEDGE_SPREADING_MAX_EXTRA = max(0, int(os.getenv("KNOWLEDGE_SPREADING_MAX_EXTRA", "2")))
KNOWLEDGE_SPREADING_SCORE_WEIGHT = float(os.getenv("KNOWLEDGE_SPREADING_SCORE_WEIGHT", "0.003"))


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
    supersedes_id   TEXT,
    archived_at     TEXT
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

CREATE TABLE IF NOT EXISTS knowledge_prune_meta (
    user_id   TEXT PRIMARY KEY,
    last_id   TEXT NOT NULL DEFAULT '',
    updated_at TEXT
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


class KnowledgeSchema:
    """Owns DB connection, DDL, and schema migrations for learned knowledge."""

    def connect(self, user_id: str | None = None) -> sqlite3.Connection:
        return connect(user_id)

    def ensure_migrated(self, conn: sqlite3.Connection, user_id: str | None = None) -> None:
        ensure_knowledge_schema_migrated(conn, user_id)

    def vacuum(self, user_id: str | None = None) -> None:
        vacuum_knowledge_db(user_id)


def connect(user_id: str | None = None) -> sqlite3.Connection:
    """Legacy free-function shim: sole implementation that
    :class:`KnowledgeSchema.connect` delegates to. Kept for the historical
    import path."""
    conn = initialize_store_db(KNOWLEDGE_DB_PATH, _DDL, user_id=user_id, vector=True)
    ensure_knowledge_schema_migrated(conn, user_id)
    return conn


def ensure_knowledge_schema_migrated(conn: sqlite3.Connection, user_id: str | None = None) -> None:
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

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS knowledge_prune_meta (
            user_id   TEXT PRIMARY KEY,
            last_id   TEXT NOT NULL DEFAULT '',
            updated_at TEXT
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_learned_chunks_access ON learned_chunks(access_count, last_accessed)")

    archive_cols = [r[1] for r in conn.execute("PRAGMA table_info(learned_chunks_archive)").fetchall()]
    if archive_cols:
        # Table exists, check for missing columns
        if "access_count" not in archive_cols:
            conn.execute("ALTER TABLE learned_chunks_archive ADD COLUMN access_count INTEGER NOT NULL DEFAULT 0")
        if "last_accessed" not in archive_cols:
            conn.execute("ALTER TABLE learned_chunks_archive ADD COLUMN last_accessed TEXT")

    conn.commit()


def now() -> str:
    return utc_now_iso()


def vacuum_knowledge_db(user_id: str | None = None) -> None:
    """Legacy free-function shim: sole implementation that
    :class:`KnowledgeSchema.vacuum` delegates to. VACUUM the knowledge DB to
    reclaim space after deletions."""
    uid = user_id or current_user_id()
    conn = connect(uid)
    try:
        conn.execute("VACUUM")
        conn.execute("ANALYZE")
    finally:
        conn.close()
