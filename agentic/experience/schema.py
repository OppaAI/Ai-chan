"""Schema, connection, and shared constants for the experience store.

Experience is not user memory and not wiki/knowledge. It is Aiko's procedural
trace of what she tried: goal, ordered tools, outcomes, verification score, and
a short result excerpt. Records do not decay or get forgotten; they are capped
only to prevent unbounded growth/noise. Because tool arguments can contain
incidental sensitive data, only argument keys and sanitized excerpts are stored,
and the SQLite DB uses the same optional SQLCipher encryption path as memory.
"""
from __future__ import annotations

import os
import re
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from system.config import load_config
load_config()

try:
    from cognition.memory.vecstore import delete_by_id, initialize_store_db, insert_vector, rank_by_id, rrf_score, user_scoped_fts_search, user_scoped_vec_knn, utc_now_iso
    from cognition.memory.memorize import extract_entities, entities_to_json, entities_from_json, entity_overlap_score
except ImportError:  # lightweight practice.py/test environments may not have numpy/sqlite-vec
    from datetime import datetime, timezone
    def utc_now_iso(): return datetime.now(timezone.utc).isoformat()
    def initialize_store_db(path, ddl, user_id=None, vector=True):
        from system.userspace import user_state_dir
        db_path = Path(path)
        if not db_path.is_absolute():
            db_path = user_state_dir(user_id) / db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        # Drop sqlite-vec-only statements for fallback mode.
        safe = re.sub(r"CREATE VIRTUAL TABLE IF NOT EXISTS experiences_vec USING vec0\([^;]+;", "", ddl, flags=re.S)
        safe = re.sub(r"DELETE FROM experiences_vec WHERE id = old.id;", "", safe)
        conn.executescript(safe)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(experiences)").fetchall()]
        if "entities" not in cols:
            conn.execute("ALTER TABLE experiences ADD COLUMN entities TEXT NOT NULL DEFAULT '[]'")
            conn.commit()
        return conn
    def insert_vector(*args, **kwargs): return None
    def delete_by_id(conn, table, row_id): conn.execute(f"DELETE FROM {table} WHERE id=?", (row_id,))
    def rank_by_id(rows): return {row["id"]: i for i, row in enumerate(rows)}
    def rrf_score(eid, *rankings, k=60): return 1.0
    def user_scoped_vec_knn(*args, **kwargs): return []
    def user_scoped_fts_search(conn, fts_table, owner_table, owner_alias, query, user_id, limit):
        return conn.execute(f"SELECT * FROM {owner_table} WHERE user_id=? AND record_text LIKE ? LIMIT ?", (user_id, f"%{query}%", limit)).fetchall()
    def extract_entities(text): return []
    def entities_to_json(ents): return "[]"
    def entities_from_json(raw): return []
    def entity_overlap_score(query, ents): return 0.0
from system.log import get_logger

log = get_logger(__name__)

EMBED_DIMS = int(os.getenv("EMBED_DIMS", "640"))
EXPERIENCE_DB_PATH = os.getenv("EXPERIENCE_DB_PATH", "experience/experience.db")
EXPERIENCE_QUERY_INSTRUCT = os.getenv("EXPERIENCE_QUERY_INSTRUCT", "Retrieve similar past agentic task runs").strip()
EXPERIENCE_RRF_K = int(os.getenv("EXPERIENCE_RRF_K", "60"))
EXPERIENCE_KNN_LIMIT = int(os.getenv("EXPERIENCE_KNN_LIMIT", "20"))
EXPERIENCE_FTS_LIMIT = int(os.getenv("EXPERIENCE_FTS_LIMIT", "20"))
EXPERIENCE_RECALL_SCORE_THRESHOLD = float(os.getenv("EXPERIENCE_RECALL_SCORE_THRESHOLD", "0.012"))
EXPERIENCE_MAX_ROWS = int(os.getenv("EXPERIENCE_MAX_ROWS", "5000"))
EXPERIENCE_CONTEXT_CHARS = int(os.getenv("EXPERIENCE_CONTEXT_CHARS", "2500"))
EXPERIENCE_ENTITY_BOOST = float(os.getenv("EXPERIENCE_ENTITY_BOOST", "0.003"))
EXPERIENCE_AUTO_RELATE_THRESHOLD = float(os.getenv("EXPERIENCE_AUTO_RELATE_THRESHOLD", "0.90"))
EXPERIENCE_SUPERSEDE_ON_NEAR_DUP = os.getenv("EXPERIENCE_SUPERSEDE_ON_NEAR_DUP", "1").strip().lower() in {"1", "true", "yes", "on"}
EXPERIENCE_SUPERSEDE_THRESHOLD = float(os.getenv("EXPERIENCE_SUPERSEDE_THRESHOLD", "0.95"))
EXPERIENCE_SPREADING_ENABLED = os.getenv("EXPERIENCE_SPREADING_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}
EXPERIENCE_SPREADING_MAX_EXTRA = max(0, int(os.getenv("EXPERIENCE_SPREADING_MAX_EXTRA", "2")))
EXPERIENCE_SPREADING_SCORE_WEIGHT = float(os.getenv("EXPERIENCE_SPREADING_SCORE_WEIGHT", "0.003"))

_SECRET_RE = re.compile(r"(?i)(api[_-]?key|token|secret|password)(\s*[:=]\s*)([^\s,;]+)")

_DDL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS experiences (
    id             TEXT PRIMARY KEY,
    user_id        TEXT NOT NULL,
    goal           TEXT NOT NULL,
    record_text    TEXT NOT NULL,
    steps_json     TEXT NOT NULL,
    outcome        TEXT NOT NULL,
    score          REAL NOT NULL,
    answer_excerpt TEXT NOT NULL,
    entities       TEXT NOT NULL DEFAULT '[]',
    created_at     TEXT NOT NULL,
    status         TEXT NOT NULL DEFAULT 'active',
    supersedes_id  TEXT
);

CREATE INDEX IF NOT EXISTS idx_experiences_user ON experiences(user_id);
CREATE INDEX IF NOT EXISTS idx_experiences_created ON experiences(created_at);

CREATE VIRTUAL TABLE IF NOT EXISTS experiences_fts USING fts5(
    record_text,
    id UNINDEXED,
    content='experiences',
    content_rowid='rowid'
);

CREATE VIRTUAL TABLE IF NOT EXISTS experiences_vec USING vec0(
    id TEXT PRIMARY KEY,
    embedding FLOAT[{dims}] distance_metric=cosine
);

-- Engram relations: explicit links between experiences (continuation, contradiction, refines, synthesizes)
CREATE TABLE IF NOT EXISTS engram_relations (
    from_engram TEXT NOT NULL,
    to_engram   TEXT NOT NULL,
    relation_type TEXT NOT NULL,  -- 'continuation', 'contradiction', 'refines', 'synthesizes'
    confidence  REAL NOT NULL DEFAULT 1.0,
    created_at  TEXT NOT NULL,
    PRIMARY KEY (from_engram, to_engram, relation_type),
    FOREIGN KEY (from_engram) REFERENCES experiences(id) ON DELETE CASCADE,
    FOREIGN KEY (to_engram)   REFERENCES experiences(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_engram_relations_from ON engram_relations(from_engram);
CREATE INDEX IF NOT EXISTS idx_engram_relations_to   ON engram_relations(to_engram);

CREATE TRIGGER IF NOT EXISTS experiences_ai AFTER INSERT ON experiences BEGIN
    INSERT INTO experiences_fts(rowid, record_text, id) VALUES (new.rowid, new.record_text, new.id);
END;

CREATE TRIGGER IF NOT EXISTS experiences_ad AFTER DELETE ON experiences BEGIN
    INSERT INTO experiences_fts(experiences_fts, rowid, record_text, id)
    VALUES ('delete', old.rowid, old.record_text, old.id);
    DELETE FROM experiences_vec WHERE id = old.id;
END;

CREATE TRIGGER IF NOT EXISTS experiences_au AFTER UPDATE OF record_text ON experiences BEGIN
    INSERT INTO experiences_fts(experiences_fts, rowid, record_text, id)
    VALUES ('delete', old.rowid, old.record_text, old.id);
    INSERT INTO experiences_fts(rowid, record_text, id) VALUES (new.rowid, new.record_text, new.id);
END;
""".format(dims=EMBED_DIMS)


class Embedder(Protocol):
    def embed_query(self, text: str, instruct: str = "") -> object: ...


@dataclass
class ExperienceStep:
    tool: str
    ok: bool
    error_type: str | None = None
    arg_keys: list[str] = field(default_factory=list)
    args_preview: dict[str, str] = field(default_factory=dict)


class ExperienceSchema:
    """Owns DB connection, DDL, and schema migrations for experiences."""

    def connect(self, user_id: str | None = None) -> sqlite3.Connection:
        return connect(user_id)

    def ensure_migrated(self, conn: sqlite3.Connection) -> None:
        ensure_experience_schema_migrated(conn)


def _db_path() -> str:
    """Resolve EXPERIENCE_DB_PATH from the facade namespace at call time.

    Tests and studio diagnostics can ``monkeypatch``/override the attribute on
    the ``agentic.experience`` package, so prefer that binding when present."""
    pkg = sys.modules.get("agentic.experience")
    if pkg is not None and hasattr(pkg, "EXPERIENCE_DB_PATH"):
        return pkg.EXPERIENCE_DB_PATH
    return EXPERIENCE_DB_PATH


def connect(user_id: str | None = None) -> sqlite3.Connection:
    """Legacy free-function shim: sole implementation that
    :class:`ExperienceSchema.connect` delegates to. Reads the DB path at call
    time so tests/studio can override agentic.experience.EXPERIENCE_DB_PATH."""
    conn = initialize_store_db(_db_path(), _DDL, user_id=user_id, vector=True)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(experiences)").fetchall()]
    if "entities" not in cols:
        conn.execute("ALTER TABLE experiences ADD COLUMN entities TEXT NOT NULL DEFAULT '[]'")
        conn.commit()
    ensure_experience_schema_migrated(conn)
    return conn


def ensure_experience_schema_migrated(conn: sqlite3.Connection) -> None:
    """Phase 18: status + supersedes_id on experiences."""
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(experiences)").fetchall()}
        if "status" not in cols:
            conn.execute("ALTER TABLE experiences ADD COLUMN status TEXT NOT NULL DEFAULT 'active'")
        if "supersedes_id" not in cols:
            conn.execute("ALTER TABLE experiences ADD COLUMN supersedes_id TEXT")
        conn.commit()
    except Exception as exc:
        log.debug("experience schema migrate skipped: %s", exc)


def sanitize(text: str, max_chars: int = 500) -> str:
    """Legacy free-function shim kept for the historical import path."""
    t = _SECRET_RE.sub(r"\1\2[redacted]", text or "")
    t = re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+", r"\1[redacted]", t)
    return re.sub(r"\s+", " ", t).strip()[:max_chars]


def now() -> str:
    return utc_now_iso()


def _sanitize(text: str, max_chars: int = 500) -> str:
    return sanitize(text, max_chars)


def _now() -> str:
    return now()


# Legacy alias: graph_export.py and tests import _connect directly.
_connect = connect
