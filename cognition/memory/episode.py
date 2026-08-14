"""
cognition/memory/episode.py

Episodic Memory Cortex (EMC) — true episodic store for Aiko.

Stores first-person, time-stamped episodes that come from working-memory
eviction. Separate from the existing memories / knowledge / experience tables.

Design rules (PR EMC-1):
  - Same per-user SQLite file as the rest of memory (Option A).
  - Same sqlite-vec + FTS5 technology.
  - Missing human-EM fields stay NULL / empty — never invent values.
  - This PR only adds storage + basic bind/flush API.
    Eviction, recall integration, and dream distillation come later.

Public surface:
    from cognition.memory.episode import EpisodicStore, ensure_episode_schema
"""
from __future__ import annotations

import json
import sqlite3
import struct
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from system.log import get_logger
from system.userspace import current_user_id
from cognition.memory.vecstore import HarrierEmbedder, initialize_store_db
from cognition.memory.env import env_bool, env_int, env_float

log = get_logger(__name__)

# ── tunables (also mirrored in config/memory.yaml) ────────────────────────────

EMC_ENABLED = env_bool("EMC_ENABLED", "1")
EMC_EMBED_ON_FLUSH = env_bool("EMC_EMBED_ON_FLUSH", "1")
EMC_FLUSH_BATCH = max(1, env_int("EMC_FLUSH_BATCH", 32))
EMC_STAGING_MAX = max(10, env_int("EMC_STAGING_MAX", 200))

EMBED_DIMS = int(__import__("os").getenv("EMBED_DIMS", "640"))


# ── schema ────────────────────────────────────────────────────────────────────

_EMC_DDL = f"""
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS emc_storage (
    id               INTEGER PRIMARY KEY,
    user_id          TEXT    NOT NULL,
    timestamp        TEXT    NOT NULL,          -- ISO-8601 of the original event
    date             TEXT    NOT NULL,          -- YYYY-MM-DD
    trace            TEXT    NOT NULL,          -- episode content (\"what\")
    encoding         BLOB,                      -- fp32 vector (NULL until encoded)
    created_at       TEXT    DEFAULT (datetime('now')),

    -- Human EM components (NULL if unknown — never guess)
    valence_tag      TEXT,                      -- pos | neg | neutral
    arousal_score    REAL,                      -- -2.0 … +2.0
    salience_score   REAL,
    entities         TEXT,                      -- JSON array
    source           TEXT,                      -- chat | voice | tool | dream …
    session_id       TEXT,

    -- Lifecycle
    last_recalled_at TEXT,
    recall_count     INTEGER NOT NULL DEFAULT 0,
    superseded_by    INTEGER,
    valid_from       TEXT,
    valid_until      TEXT
);

CREATE TABLE IF NOT EXISTS emc_staging (
    id               INTEGER PRIMARY KEY,
    user_id          TEXT    NOT NULL,
    timestamp        TEXT    NOT NULL,
    date             TEXT    NOT NULL,
    trace            TEXT    NOT NULL,
    valence_tag      TEXT,
    arousal_score    REAL,
    salience_score   REAL,
    entities         TEXT,
    source           TEXT,
    session_id       TEXT,
    created_at       TEXT    DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_emc_user_ts
    ON emc_storage(user_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_emc_user_date
    ON emc_storage(user_id, date);
CREATE INDEX IF NOT EXISTS idx_emc_staging_user
    ON emc_staging(user_id, id);

-- sqlite-vec virtual table (rowid aligned with emc_storage.id)
CREATE VIRTUAL TABLE IF NOT EXISTS emc_vec USING vec0(
    embedding float[{EMBED_DIMS}]
);

-- FTS5 lexical index
CREATE VIRTUAL TABLE IF NOT EXISTS emc_fts USING fts5(
    trace,
    content='emc_storage',
    content_rowid='id',
    tokenize='porter'
);
"""


def ensure_episode_schema(conn: sqlite3.Connection) -> list[str]:
    """Idempotent creation of EMC tables + indexes. Returns list of actions taken."""
    actions: list[str] = []
    try:
        conn.executescript(_EMC_DDL)
        conn.commit()
        actions.append("emc_schema_ensured")
    except sqlite3.Error as e:
        log.warning("ensure_episode_schema: %s", e)
    return actions


# ── helpers ───────────────────────────────────────────────────────────────────

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _date_from_ts(ts: str) -> str:
    """Extract YYYY-MM-DD from an ISO timestamp; fall back to today UTC."""
    try:
        return ts[:10]
    except Exception:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _pack_vector(vec: list[float] | Any) -> bytes:
    v = [float(x) for x in vec]
    return struct.pack(f"{len(v)}f", *v)


def _entities_json(entities: list[str] | None) -> str | None:
    if not entities:
        return None
    return json.dumps(list(entities), ensure_ascii=False)


# ── data class ────────────────────────────────────────────────────────────────

@dataclass
class Episode:
    """One consolidated episodic memory record."""
    id: int | None = None
    user_id: str = ""
    timestamp: str = ""
    date: str = ""
    trace: str = ""
    valence_tag: str | None = None
    arousal_score: float | None = None
    salience_score: float | None = None
    entities: list[str] = field(default_factory=list)
    source: str | None = None
    session_id: str | None = None
    created_at: str = ""
    recall_count: int = 0
    relevancy: float = 0.0  # transient — set at recall time only


# ── main store ────────────────────────────────────────────────────────────────

class EpisodicStore:
    """
    Episodic Memory store (EMC).

    Lifecycle for this PR:
        bind()  →  emc_staging
        flush_staging()  →  emc_storage (+ optional embed into emc_vec / emc_fts)

    Later PRs will add:
        - eviction from working memory / conversation history
        - regular buffer drain policy
        - recall path + joint token budget with SM
        - dream distillation
    """

    def __init__(
        self,
        db_path: str,
        *,
        user_id: str | None = None,
        embedder: HarrierEmbedder | None = None,
    ) -> None:
        self._user_id = user_id or current_user_id()
        self._db_path = db_path
        self._embedder = embedder or HarrierEmbedder()
        self._lock = threading.RLock()
        self._conn = self._connect()
        with self._lock:
            ensure_episode_schema(self._conn)

    def _connect(self) -> sqlite3.Connection:
        # Re-use the same bootstrap path style as the rest of memory.
        # We pass a minimal DDL; ensure_episode_schema does the real work.
        return initialize_store_db(
            self._db_path,
            "PRAGMA journal_mode=WAL;",
            user_id=self._user_id,
            vector=True,
        )

    # ── write path ────────────────────────────────────────────────────────────

    def bind(
        self,
        *,
        timestamp: str,
        trace: str,
        user_id: str | None = None,
        valence_tag: str | None = None,
        arousal_score: float | None = None,
        salience_score: float | None = None,
        entities: list[str] | None = None,
        source: str | None = None,
        session_id: str | None = None,
    ) -> int:
        """
        Stage one episode. Returns staging_id.

        Never invents missing fields — pass None / omit and the column stays NULL.
        """
        if not EMC_ENABLED:
            return -1
        uid = user_id or self._user_id
        ts = (timestamp or "").strip() or _utc_now_iso()
        content = (trace or "").strip()
        if not content:
            return -1

        date = _date_from_ts(ts)
        ent_json = _entities_json(entities)

        with self._lock:
            cur = self._conn.execute(
                """
                INSERT INTO emc_staging (
                    user_id, timestamp, date, trace,
                    valence_tag, arousal_score, salience_score,
                    entities, source, session_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    uid, ts, date, content,
                    valence_tag, arousal_score, salience_score,
                    ent_json, source, session_id,
                ),
            )
            self._conn.commit()
            staging_id = int(cur.lastrowid)
            log.debug("EMC bind staging_id=%s user=%s chars=%d", staging_id, uid, len(content))
            return staging_id

    def flush_staging(self, limit: int | None = None) -> int:
        """
        Move up to `limit` staged rows into emc_storage.
        Optionally embeds and writes into emc_vec + emc_fts.
        Returns number of episodes flushed.
        """
        if not EMC_ENABLED:
            return 0
        batch = limit if limit is not None else EMC_FLUSH_BATCH

        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id, user_id, timestamp, date, trace,
                       valence_tag, arousal_score, salience_score,
                       entities, source, session_id
                FROM emc_staging
                WHERE user_id = ?
                ORDER BY id ASC
                LIMIT ?
                """,
                (self._user_id, batch),
            ).fetchall()

            if not rows:
                return 0

            flushed = 0
            for row in rows:
                staging_id = row[0]
                try:
                    storage_id = self._inscribe(row)
                    # remove from staging
                    self._conn.execute("DELETE FROM emc_staging WHERE id = ?", (staging_id,))
                    flushed += 1
                    log.debug("EMC flush staging=%s → storage=%s", staging_id, storage_id)
                except Exception as e:
                    log.warning("EMC flush failed staging_id=%s: %s", staging_id, e)

            self._conn.commit()
            return flushed

    def _inscribe(self, row: sqlite3.Row | tuple) -> int:
        """Insert one staged row into emc_storage (+ vec/fts if enabled)."""
        (
            _sid, user_id, timestamp, date, trace,
            valence_tag, arousal_score, salience_score,
            entities, source, session_id,
        ) = row

        cur = self._conn.execute(
            """
            INSERT INTO emc_storage (
                user_id, timestamp, date, trace, encoding,
                valence_tag, arousal_score, salience_score,
                entities, source, session_id
            ) VALUES (?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id, timestamp, date, trace,
                valence_tag, arousal_score, salience_score,
                entities, source, session_id,
            ),
        )
        storage_id = int(cur.lastrowid)

        # FTS
        try:
            self._conn.execute(
                "INSERT INTO emc_fts(rowid, trace) VALUES (?, ?)",
                (storage_id, trace),
            )
        except sqlite3.Error as e:
            log.debug("EMC FTS insert: %s", e)

        # Vector (optional, can be deferred)
        if EMC_EMBED_ON_FLUSH:
            try:
                vec = list(self._embedder.embed([trace]))[0]
                blob = _pack_vector(vec)
                self._conn.execute(
                    "UPDATE emc_storage SET encoding = ? WHERE id = ?",
                    (blob, storage_id),
                )
                self._conn.execute(
                    "INSERT INTO emc_vec(rowid, embedding) VALUES (?, ?)",
                    (storage_id, blob),
                )
            except Exception as e:
                log.debug("EMC embed on flush skipped: %s", e)

        return storage_id

    # ── read helpers (minimal for this PR) ────────────────────────────────────

    def staging_count(self, user_id: str | None = None) -> int:
        uid = user_id or self._user_id
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM emc_staging WHERE user_id = ?", (uid,)
            ).fetchone()
            return int(row[0]) if row else 0

    def storage_count(self, user_id: str | None = None) -> int:
        uid = user_id or self._user_id
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM emc_storage WHERE user_id = ?", (uid,)
            ).fetchone()
            return int(row[0]) if row else 0

    def get_stats(self) -> dict[str, Any]:
        return {
            "enabled": EMC_ENABLED,
            "user_id": self._user_id,
            "staging": self.staging_count(),
            "storage": self.storage_count(),
            "embed_on_flush": EMC_EMBED_ON_FLUSH,
        }

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass
