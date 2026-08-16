"""
cognition/memory/episode.py

Episodic Memory Cortex (EMC) — true episodic store for Aiko.

Stores first-person, time-stamped episodes that come from working-memory
eviction. Separate from the existing memories / knowledge / experience tables.

Design rules:
  - Same per-user SQLite file as the rest of memory (Option A).
  - Same sqlite-vec + FTS5 technology.
  - Missing human-EM fields stay NULL / empty — never invent values.

EMC-1: storage + bind/flush API
EMC-2: turn ingest + buffer drain (eviction path)
EMC-6: coherent episode formation — group related staging rows on flush

Public surface:
    from cognition.memory.episode import EpisodicStore, ensure_episode_schema
"""
from __future__ import annotations

import json
import os
import queue
import sqlite3
import struct
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from system.log import get_logger
from system.userspace import current_user_id
from cognition.memory.vecstore import HarrierEmbedder, initialize_store_db
from cognition.memory.env import env_bool, env_int, env_float
from cognition.memory.episode_group import (
    EMC_GROUP_ENABLED,
    group_staging_rows as _group_staging_rows,
    merge_staging_group as _merge_staging_group,
)

log = get_logger(__name__)

# ── tunables (also mirrored in config/memory.yaml) ────────────────────────────

EMC_ENABLED = env_bool("EMC_ENABLED", "1")
EMC_EMBED_ON_FLUSH = env_bool("EMC_EMBED_ON_FLUSH", "1")
EMC_FLUSH_BATCH = max(1, env_int("EMC_FLUSH_BATCH", 32))
EMC_STAGING_MAX = max(10, env_int("EMC_STAGING_MAX", 200))

# EMC-2: eviction / buffer drain
EMC_EVICT_ENABLED = env_bool("EMC_EVICT_ENABLED", "1")
EMC_EVICT_MIN_CHARS = max(1, env_int("EMC_EVICT_MIN_CHARS", 40))
EMC_FLUSH_EVERY_TURNS = max(0, env_int("EMC_FLUSH_EVERY_TURNS", 8))
MEMORY_WM_CAPACITY = max(1, env_int("MEMORY_WM_CAPACITY", "7"))
MEMORY_WM_CAPACITY = max(1, env_int("MEMORY_WM_CAPACITY", "7"))
EMC_FLUSH_ON_STAGING = max(1, env_int("EMC_FLUSH_ON_STAGING", 24))

EMBED_DIMS = int(os.getenv("EMBED_DIMS", "640"))


# ── schema ────────────────────────────────────────────────────────────────────

_EMC_DDL = f"""
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS emc_storage (
    id               INTEGER PRIMARY KEY,
    user_id          TEXT    NOT NULL,
    timestamp        TEXT    NOT NULL,
    date             TEXT    NOT NULL,
    trace            TEXT    NOT NULL,
    encoding         BLOB,
    created_at       TEXT    DEFAULT (datetime('now')),
    valence_tag      TEXT,
    arousal_score    REAL,
    salience_score   REAL,
    entities         TEXT,
    source           TEXT,
    session_id       TEXT,
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

CREATE VIRTUAL TABLE IF NOT EXISTS emc_vec USING vec0(
    embedding float[{EMBED_DIMS}] distance_metric=cosine
);

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
    # vec0 defaults to L2; rebuild emc_vec with cosine metric if it predates
    # distance_metric=cosine so MATCH ranking matches the cosine semantics of
    # every KNN caller.
    try:
        from cognition.memory.vecstore import ensure_vec0_cosine_metric
        ensure_vec0_cosine_metric(conn)
    except Exception as e:
        log.debug("emc_vec metric migration: %s", e)
    # EMC-2: install turn-ingest hooks (idempotent, best-effort)
    try:
        from cognition.memory.emc2_wire import apply_emc2_hooks
        apply_emc2_hooks()
        actions.append("emc2_hooks")
    except Exception as e:
        log.debug("emc2 hooks: %s", e)
    return actions


# ── helpers ───────────────────────────────────────────────────────────────────

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _date_from_ts(ts: str) -> str:
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


def _format_turn_trace(user_text: str, assistant_text: str) -> str:
    """Compact first-person-ish episode text from one turn pair."""
    u = (user_text or "").strip()
    a = (assistant_text or "").strip()
    parts = []
    if u:
        parts.append(f"User: {u}")
    if a:
        if len(a) > 600:
            a = a[:600].rstrip() + "…"
        parts.append(f"Aiko: {a}")
    return "\n".join(parts)


def _is_trivial_turn(user_text: str, assistant_text: str) -> bool:
    """Skip pure filler / very short turns. Never invent content."""
    combined = f"{(user_text or '').strip()} {(assistant_text or '').strip()}".strip()
    if len(combined) < EMC_EVICT_MIN_CHARS:
        return True
    try:
        from cognition.memory.search import _is_trivial_input
        if _is_trivial_input(user_text or "") and len((assistant_text or "").strip()) < 80:
            return True
    except Exception:
        pass
    return False


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
    relevancy: float = 0.0


# ── main store ────────────────────────────────────────────────────────────────

class EpisodicStore:
    """
    Episodic Memory store (EMC).

    EMC-1 lifecycle:
        bind()  →  emc_staging
        flush_staging()  →  emc_storage (+ optional embed)

    EMC-2 lifecycle:
        ingest_turn(user, assistant)  →  bind (if non-trivial)
        maybe_flush()  →  flush when staging high or every N turns
        flush_all() / close() on session end

    EMC-6 (on flush_staging when EMC_GROUP_ENABLED):
        consecutive staging rows that share session_id and fall within
        the time gap are merged into one emc_storage episode.
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
        # EMC recall result cache (see episode_recall.search). Keyed by
        # (user_id, query, limit); bounded + TTL so episodic recall doesn't
        # re-run KNN+FTS (and the per-id touch loop) on every turn.
        self._recall_cache: OrderedDict[tuple[str, str, int], tuple[float, list[dict]]] = OrderedDict()
        self._recall_cache_lock = threading.RLock()
        # EMC embed-off-thread: embedding each episode on the conversation
        # thread stalls turns (1+ HTTP round-trips per flush). Instead the
        # embed runs on a dedicated worker thread; _inscribe only queues the
        # (storage_id, trace) pair after the storage row + FTS insert commit.
        self._embed_queue: "queue.Queue[tuple[int, str] | None]" = queue.Queue()
        self._embed_worker: threading.Thread | None = None
        self._turns_since_flush = 0
        with self._lock:
            ensure_episode_schema(self._conn)

    def _connect(self) -> sqlite3.Connection:
        return initialize_store_db(
            self._db_path,
            "PRAGMA journal_mode=WAL;",
            user_id=self._user_id,
            vector=True,
        )

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
        """Stage one episode. Returns staging_id. Missing fields stay NULL."""
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

    def ingest_turn(
        self,
        user_text: str,
        assistant_text: str,
        *,
        user_id: str | None = None,
        timestamp: str | None = None,
        session_id: str | None = None,
        source: str = "chat",
        valence_tag: str | None = None,
        arousal_score: float | None = None,
        salience_score: float | None = None,
        entities: list[str] | None = None,
        auto_flush: bool = True,
    ) -> int:
        """EMC-2: accept one conversation turn pair into the episodic buffer."""
        if not EMC_ENABLED or not EMC_EVICT_ENABLED:
            return -1
        if _is_trivial_turn(user_text, assistant_text):
            log.debug("EMC ingest skipped (trivial turn)")
            return -1

        trace = _format_turn_trace(user_text, assistant_text)
        if not trace.strip():
            return -1

        # Normalize user_id to prevent staging under a different user than self._user_id
        normalized_user_id = self._user_id
        if user_id is not None and user_id != self._user_id:
            log.warning(
                "EMC ingest_turn: user_id mismatch (provided=%s, instance=%s). "
                "Using instance user_id to prevent orphaned staging rows.",
                user_id, self._user_id
            )

        staging_id = self.bind(
            timestamp=timestamp or _utc_now_iso(),
            trace=trace,
            user_id=normalized_user_id,
            valence_tag=valence_tag,
            arousal_score=arousal_score,
            salience_score=salience_score,
            entities=entities,
            source=source,
            session_id=session_id,
        )

        with self._lock:
            self._turns_since_flush += 1

        if auto_flush and staging_id > 0:
            self.maybe_flush()

        return staging_id

    def maybe_flush(self) -> int:
        if not EMC_ENABLED:
            return 0
        staging = self.staging_count()
        should = staging >= EMC_FLUSH_ON_STAGING
        if EMC_FLUSH_EVERY_TURNS > 0 and self._turns_since_flush >= EMC_FLUSH_EVERY_TURNS:
            should = should or staging > 0
        # WM capacity: if staging exceeds the working memory cap, force flush
        # to evict the oldest rows and make room for new information.
        if staging > MEMORY_WM_CAPACITY:
            should = True
        if not should:
            return 0
        n = self.flush_staging()
        with self._lock:
            self._turns_since_flush = 0
        return n

    def flush_all(self) -> int:
        total = 0
        while True:
            n = self.flush_staging(limit=EMC_FLUSH_BATCH)
            total += n
            if n < EMC_FLUSH_BATCH:
                break
        with self._lock:
            self._turns_since_flush = 0
        if total and EMC_EMBED_ON_FLUSH:
            # Draining is best-effort: wait up to a few seconds for the
            # off-thread embed worker to finish so episodes are KNN-searchable
            # before the session ends (close()/switch_user()).
            self.drain_embeds(timeout=5.0)
        return total

    def drain_embeds(self, timeout: float | None = None) -> None:
        """Block until queued episode embeds finish, or `timeout` elapses.
        No-op if embedding is disabled or nothing is queued."""
        if not EMC_EMBED_ON_FLUSH or self._embed_worker is None:
            return
        try:
            if timeout is None:
                self._embed_queue.join()
                return
            deadline = time.monotonic() + max(0.0, timeout)
            with self._embed_queue.all_tasks_done:
                while self._embed_queue.unfinished_tasks:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return
                    self._embed_queue.all_tasks_done.wait(remaining)
        except Exception:
            pass

def flush_staging(self, limit: int | None = None, max_rows: int | None = None) -> int:
        """Move staging rows → emc_storage.

        EMC-6: when ``EMC_GROUP_ENABLED``, consecutive related rows are merged
        into one coherent episode (same session, within time gap, bounded by
        max turns/chars). Otherwise 1:1 as in EMC-1/2.

        ``max_rows`` — Phase 21: working memory capacity limit. If provided,
        only the newest ``max_rows`` staging rows are flushed to emc_storage;
        the oldest rows are deleted to make room.
        """
        if not EMC_ENABLED:
            return 0
        batch = limit if limit is not None else EMC_FLUSH_BATCH

        with self._lock:
            if max_rows is not None:
                # Determine which rows to consider for flushing
                all_rows = self._conn.execute(
                    """
                    SELECT id, user_id, timestamp, date, trace,
                           valence_tag, arousal_score, salience_score,
                           entities, source, session_id
                    FROM emc_staging
                    WHERE user_id = ?
                    ORDER BY id ASC
                    """,
                    (self._user_id,),
                ).fetchall()
                if len(all_rows) <= max_rows:
                    rows = all_rows
                else:
                    # Keep the newest max_rows (highest ids), delete the rest
                    rows = all_rows[-max_rows:]
                    # Delete the oldest rows without inscribing them
                    old_ids = [str(r[0]) for r in all_rows[:-max_rows]]
                    self._conn.executemany(
                        "DELETE FROM emc_staging WHERE id = ?",
                        [(sid,) for sid in old_ids],
                    )
            else:
                rows = self._conn.execute(
                    """
                    SELECT id, user_id, timestamp, date, trace,
                           valence_tag, arousal_score, salience_score,
                           entities, source, session_id
                    FROM emc_staging
                    WHERE user_id = ?
                    ORDER BY id ASC
                    LIMIT ?,
                    """,
                    (self._user_id, batch),
                ).fetchall()

            if not rows:
                return 0

            flushed = 0
            groups = (
                _group_staging_rows(rows)
                if EMC_GROUP_ENABLED
                else [[r] for r in rows]
            )

            for group in groups:
                staging_ids = [int(r[0]) for r in group]
                try:
                    merged = _merge_staging_group(group)
                    storage_id = self._inscribe(merged)
                    self._conn.executemany(
                        "DELETE FROM emc_staging WHERE id = ?",
                        [(sid,) for sid in staging_ids],
                    )
                    flushed += len(staging_ids)
                    if len(staging_ids) > 1:
                        log.info(
                            "EMC-6 grouped staging=%s → storage=%s turns=%d",
                            staging_ids, storage_id, len(staging_ids),
                        )
                    else:
                        log.debug(
                            "EMC flush staging=%s → storage=%s",
                            staging_ids[0], storage_id,
                        )
                except Exception as e:
                    log.warning(
                        "EMC flush failed staging_ids=%s: %s", staging_ids, e
                    )

            self._conn.commit()
            if flushed:
                # New episodes make cached recall results stale.
                with self._recall_cache_lock:
                    self._recall_cache.clear()
            return flushed

    def _inscribe(self, row: sqlite3.Row | tuple) -> int:
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

        try:
            self._conn.execute(
                "INSERT INTO emc_fts(rowid, trace) VALUES (?, ?)",
                (storage_id, trace),
            )
        except sqlite3.Error as e:
            log.debug("EMC FTS insert: %s", e)

        if EMC_EMBED_ON_FLUSH:
            # Off-thread: don't stall the conversation on an HTTP embed. The
            # row + FTS insert are already committed by flush_staging's
            # caller; the worker takes the lock only to write the vector.
            self._enqueue_embed(storage_id, trace)

        return storage_id

    def _enqueue_embed(self, storage_id: int, trace: str) -> None:
        """Queue a (storage_id, trace) embed for the background worker."""
        if self._embed_worker is None or not self._embed_worker.is_alive():
            self._embed_worker = threading.Thread(
                target=self._embed_worker_loop,
                name="emc-embed-worker",
                daemon=True,
            )
            self._embed_worker.start()
        self._embed_queue.put((storage_id, trace))

    def _embed_worker_loop(self) -> None:
        """Embed queued episodes off the conversation thread. Best-effort: a
        failed embed leaves the episode FTS-searchable but KNN-invisible; the
        next successful embed retries nothing automatically, matching the
        pre-existing best-effort semantics of the synchronous path."""
        while True:
            item = self._embed_queue.get()
            try:
                if item is None:
                    self._embed_queue.task_done()
                    return
                storage_id, trace = item
                vec = list(self._embedder.embed([trace]))[0]
                blob = _pack_vector(vec)
                with self._lock:
                    self._conn.execute(
                        "UPDATE emc_storage SET encoding = ? WHERE id = ?",
                        (blob, storage_id),
                    )
                    self._conn.execute(
                        "INSERT INTO emc_vec(rowid, embedding) VALUES (?, ?)",
                        (storage_id, blob),
                    )
                    self._conn.commit()
            except Exception as e:
                log.debug("EMC background embed skipped: %s", e)
            finally:
                self._embed_queue.task_done()

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
            "evict_enabled": EMC_EVICT_ENABLED,
            "group_enabled": EMC_GROUP_ENABLED,
            "user_id": self._user_id,
            "staging": self.staging_count(),
            "storage": self.storage_count(),
            "turns_since_flush": self._turns_since_flush,
            "embed_on_flush": EMC_EMBED_ON_FLUSH,
        }

    def close(self) -> None:
        try:
            self.flush_all()
        except Exception as e:
            log.debug("EMC flush_all on close: %s", e)
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass
