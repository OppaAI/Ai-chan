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
EMC-3: KNN + FTS5 + RRF recall + context formatting
EMC-4: dream-time distillation of episodic memory → semantic facts
EMC-6: coherent episode formation — group related staging rows on flush

Public surface:
    from cognition.memory.episode import (
        EpisodicStore,
        ensure_episode_schema,
        distill_episodes,
        attach_recall_to_store,
        attach_dream_hook,
    )
"""
from __future__ import annotations

import json
import os
import queue
import re
import sqlite3
import struct
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import sqlite_vec

from system.config import env_bool, env_float, env_int
from system.log import get_logger
from system.userspace import current_user_id
try:
    from system import brain_trace as _brain_trace
except Exception:
    _brain_trace = None
from cognition.memory.vecstore import HarrierEmbedder, initialize_store_db
from cognition.memory.vecstore import KNN_MATCH_K_MIN, KNN_MATCH_OVERSCAN

# ── inlined from lifecycle.py (dream-pass tunables, previously separate) ───
DREAM_MERGE_THRESHOLD = float(os.getenv("DREAM_MERGE_THRESHOLD", 0.88))
WRITE_DEDUP_THRESHOLD = float(os.getenv("WRITE_DEDUP_THRESHOLD", 0.95))
DREAM_BOOST_AMOUNT = int(os.getenv("DREAM_BOOST_AMOUNT", 2))
DREAM_SCHEMA_ENABLED = os.getenv("DREAM_SCHEMA_ENABLED", "1").lower() in {"1", "true", "yes", "on"}
DREAM_SCHEMA_MIN_MEMBERS = int(os.getenv("DREAM_SCHEMA_MIN_MEMBERS", "3"))
DREAM_SCHEMA_MAX_CLUSTERS = int(os.getenv("DREAM_SCHEMA_MAX_CLUSTERS", "12"))
DREAM_SCHEMA_VALENCE_MAJORITY = float(os.getenv("DREAM_SCHEMA_VALENCE_MAJORITY", "0.6"))
_SALIENCE_KEYWORDS = frozenset([
    "name", "called", "likes", "loves", "hates", "dislikes", "always", "never",
    "important", "remember", "favourite", "favorite", "birthday", "works",
    "lives", "studying", "job", "afraid", "dream", "goal",
    "deadline", "due", "appointment", "event", "hackathon", "wallet",
    "lost", "passport", "license", "meeting", "interview", "project",
])
_SALIENCE_RE = re.compile(
    r'\b(?:' + '|'.join(re.escape(k) for k in _SALIENCE_KEYWORDS) + r')\b',
    re.IGNORECASE,
)

# ── inlined from search.py (recall helpers, previously separate) ──────────
AI_NAME = os.getenv("AI_NAME", "Aiko").strip().lower()
_FILLER_WORDS = (
    "hi", "hey", "hello", "ok", "okay", "thanks", "thank you",
    "yes", "no", "yeah", "nah", "lol", "sure", "bye",
)
_GREETING_PHRASES = (
    "how are you", "how are you doing", "hows it going", "how's it going",
    "how are things", "how you doing", "whats up", "what's up",
)
_name_alt = re.escape(AI_NAME) if AI_NAME else ""
_TRIVIAL_PHRASES = sorted(_FILLER_WORDS + _GREETING_PHRASES, key=len, reverse=True)
_trivial_alt = "|".join(re.escape(p) for p in _TRIVIAL_PHRASES)
_CLAUSE_SPLIT_RE = re.compile(r"[,.!?]+")

def _is_trivial_input(text: str) -> bool:
    clauses = [c.strip().lower() for c in _CLAUSE_SPLIT_RE.split(text or "") if c.strip()]
    if not clauses:
        return True
    for clause in clauses:
        if _name_alt and re.fullmatch(_name_alt, clause, re.IGNORECASE):
            continue
        if re.fullmatch(_trivial_alt, clause, re.IGNORECASE):
            continue
        if len(clause.split()) == 1 and len(clause) <= 2:
            continue
        return False
    return True

_BROAD_RECALL_RE = re.compile(
    r"\b(what|anything|things|facts|memories?|remember|recall)\b.*\b(about me|about oppa|you remember|past|before)\b"
    r"|\b(remember|recall)\b.*\b(me|oppa)\b",
    re.IGNORECASE,
)

def _sanitize_fts_query(query: str) -> str | None:
    cleaned = re.sub(r'[^\w\s]', ' ', query or "")
    cleaned = ' '.join(cleaned.split())
    return cleaned or None

def _normalize_memory_text(text: str) -> str:
    return " ".join((text or "").split()).lower()

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
EMC_FLUSH_ON_STAGING = max(1, env_int("EMC_FLUSH_ON_STAGING", 24))

# EMC-3: recall
EMC_RECALL_ENABLED = env_bool("EMC_RECALL_ENABLED", "1")
EMC_RECALL_LIMIT = max(0, env_int("EMC_RECALL_LIMIT", 2))
EMC_KNN_LIMIT = max(1, env_int("EMC_KNN_LIMIT", 12))
EMC_FTS_LIMIT = max(1, env_int("EMC_FTS_LIMIT", 12))
EMC_RRF_K = max(1, env_int("EMC_RRF_K", 60))
EMC_CONTEXT_CHARS = max(100, env_int("EMC_CONTEXT_CHARS", 600))
EMC_CONTEXT_EPISODE_CHARS = max(40, env_int("EMC_CONTEXT_EPISODE_CHARS", 280))
EMC_JOINT_BUDGET = env_bool("EMC_JOINT_BUDGET", "1")
EMC_RECALL_CACHE_SIZE = max(1, env_int("EMC_RECALL_CACHE_SIZE", 128))
EMC_RECALL_CACHE_TTL = max(1.0, env_float("EMC_RECALL_CACHE_TTL", 300.0))

# EMC-4: dream distillation
EMC_DREAM_ENABLED = env_bool("EMC_DREAM_ENABLED", "1")
EMC_DREAM_LIMIT = max(0, env_int("EMC_DREAM_LIMIT", 12))
EMC_DREAM_BATCH = max(1, env_int("EMC_DREAM_BATCH", 4))
EMC_DREAM_MIN_CHARS = max(20, env_int("EMC_DREAM_MIN_CHARS", 60))
EMC_DREAM_MAX_TOKENS = max(64, env_int("EMC_DREAM_MAX_TOKENS", 256))

# EMC-6: coherent episode formation
EMC_GROUP_ENABLED = env_bool("EMC_GROUP_ENABLED", "1")
EMC_GROUP_MAX_GAP_SEC = max(0, env_int("EMC_GROUP_MAX_GAP_SEC", 900))
EMC_GROUP_MAX_TURNS = max(1, env_int("EMC_GROUP_MAX_TURNS", 6))
EMC_GROUP_MAX_CHARS = max(200, env_int("EMC_GROUP_MAX_CHARS", 2000))

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
    cognitive_json   TEXT,
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
    cognitive_json   TEXT,
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
    if _is_trivial_input(user_text or "") and len((assistant_text or "").strip()) < 80:
        return True
    return False


# ── EMC-6: coherent episode formation helpers ─────────────────────────────────

def _parse_ts(value: Any) -> float | None:
    s = str(value or "").strip()
    if not s:
        return None
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        # Truncate fractional seconds to 6 digits without breaking the offset
        if "." in s:
            head, rest = s.split(".", 1)
            i = 0
            while i < len(rest) and rest[i].isdigit():
                i += 1
            digits, suffix = rest[:i], rest[i:]
            s = head + "." + digits[:6] + suffix
        return datetime.fromisoformat(s).timestamp()
    except Exception:
        return None


def _group_staging_rows(rows: list) -> list[list]:
    """Partition ordered staging rows into coherent episode groups.

    Row tuple layout (from flush_staging SELECT):
      0 id, 1 user_id, 2 timestamp, 3 date, 4 trace,
      5 valence_tag, 6 arousal_score, 7 salience_score,
      8 entities, 9 source, 10 session_id, 11 cognitive_json)
    """
    if not rows:
        return []
    groups: list[list] = [[rows[0]]]
    for row in rows[1:]:
        cur = groups[-1]
        prev = cur[-1]
        same_session = (prev[10] or None) == (row[10] or None)
        t_prev = _parse_ts(prev[2])
        t_cur = _parse_ts(row[2])
        if EMC_GROUP_MAX_GAP_SEC == 0:
            gap_ok = True
        elif t_prev is not None and t_cur is not None:
            gap_ok = abs(t_cur - t_prev) <= float(EMC_GROUP_MAX_GAP_SEC)
        else:
            gap_ok = False
        turns_ok = len(cur) < EMC_GROUP_MAX_TURNS
        existing = sum(len(str(r[4] or "")) for r in cur)
        added = len(str(row[4] or ""))
        chars_ok = (existing + added + 8 * len(cur)) <= EMC_GROUP_MAX_CHARS
        if same_session and gap_ok and turns_ok and chars_ok:
            cur.append(row)
        else:
            groups.append([row])
    return groups


def _merge_staging_group(group: list) -> tuple:
    """Collapse a group into one row-shaped tuple for _inscribe."""
    if len(group) == 1:
        return group[0]
    first = group[0]
    traces = [str(r[4] or "").strip() for r in group if str(r[4] or "").strip()]
    trace = "\n\n".join(traces)
    valence_tag = None
    for r in group:
        if r[5] is not None and str(r[5]).strip():
            valence_tag = r[5]
            break
    arousal_vals = [float(r[6]) for r in group if r[6] is not None]
    arousal_score = max(arousal_vals) if arousal_vals else None
    salience_vals = [float(r[7]) for r in group if r[7] is not None]
    salience_score = max(salience_vals) if salience_vals else None
    ents: list[str] = []
    seen: set[str] = set()
    for r in group:
        raw = r[8]
        if raw is None or raw == "":
            continue
        try:
            data = json.loads(raw) if isinstance(raw, str) else raw
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(data, list):
            for e in data:
                s = str(e).strip()
                key = s.casefold()
                if s and key not in seen:
                    seen.add(key)
                    ents.append(s)
    entities = json.dumps(ents, ensure_ascii=False) if ents else None
    source = None
    for r in group:
        if r[9] is not None and str(r[9]).strip():
            source = r[9]
            break
    if source is None:
        source = "emc_group"
    session_id = None
    for r in group:
        if r[10] is not None and str(r[10]).strip():
            session_id = r[10]
            break
    cognitive_json = None
    for r in reversed(group):
        if len(r) > 11 and r[11] is not None and str(r[11]).strip():
            cognitive_json = r[11]
            break
    return (
        first[0], first[1], first[2], first[3], trace,
        valence_tag, arousal_score, salience_score,
        entities, source, session_id, cognitive_json,
    )


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

    EMC-3: search() — KNN + FTS5 + RRF over emc_storage
           format_for_context() — render hits as <episodic_context> block

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
        embed_cache: str | None = None,
    ) -> None:
        self._user_id = user_id or current_user_id()
        self._db_path = db_path
        self._embedder = embedder or HarrierEmbedder(cache_path=embed_cache)
        self._lock = threading.RLock()
        self._conn = self._connect()
        # EMC recall result cache (see search()). Keyed by
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
        for table in ("emc_storage", "emc_staging"):
            try:
                columns = {str(r[1]) for r in self._conn.execute(f"PRAGMA table_info({table})").fetchall()}
                if "cognitive_json" not in columns:
                    self._conn.execute(f"ALTER TABLE {table} ADD COLUMN cognitive_json TEXT")
                    self._conn.commit()
            except sqlite3.Error as e:
                log.debug("EMC metadata migration %s: %s", table, e)

    def _connect(self) -> sqlite3.Connection:
        return initialize_store_db(
            self._db_path,
            "PRAGMA journal_mode=WAL;",
            user_id=self._user_id,
            vector=True,
        )

    # ── EMC-1 / EMC-2: bind, flush, ingest ─────────────────────────────────────

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
        cognitive_state: dict | None = None,
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
        cognitive_json = json.dumps(cognitive_state, ensure_ascii=False, separators=(",", ":")) if cognitive_state else None

        with self._lock:
            cur = self._conn.execute(
                """
                INSERT INTO emc_staging (
                    user_id, timestamp, date, trace,
                    valence_tag, arousal_score, salience_score,
                    entities, source, session_id, cognitive_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    uid, ts, date, content,
                    valence_tag, arousal_score, salience_score,
                    ent_json, source, session_id, cognitive_json,
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
        cognitive_state: dict | None = None,
    ) -> int:
        """EMC-2: accept one conversation turn pair into the episodic buffer."""
        if not EMC_ENABLED or not EMC_EVICT_ENABLED:
            return -1
        if _is_trivial_turn(user_text, assistant_text):
            log.debug("EMC ingest skipped (trivial turn)")
            if _brain_trace and _brain_trace.TRACE_ENABLED:
                _brain_trace.record_step(
                    "episodic.ingest_turn",
                    layer="write",
                    inputs={"user_chars": len(user_text or ""),
                            "assistant_chars": len(assistant_text or "")},
                    outputs={"staged": False},
                    factors=["trivial turn (below EMC_EVICT_MIN_CHARS or _is_trivial_input)"],
                )
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

        if _brain_trace and _brain_trace.TRACE_ENABLED:
            _brain_trace.record_step(
                "episodic.ingest_turn",
                layer="write",
                inputs={"user_chars": len(user_text or ""),
                        "assistant_chars": len(assistant_text or ""),
                        "trace_chars": len(trace),
                        "staging_before": self.staging_count()},
                factors=[
                    "EMC-2: turn written to emc_staging; embed deferred until flush",
                    f"auto-flush trigger: staging>={EMC_FLUSH_ON_STAGING} or turns>={EMC_FLUSH_EVERY_TURNS}",
                ],
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
            cognitive_state=cognitive_state,
        )

        with self._lock:
            self._turns_since_flush += 1

        flushed = 0
        if auto_flush and staging_id > 0:
            flushed = self.maybe_flush()

        if _brain_trace and _brain_trace.TRACE_ENABLED:
            _brain_trace.record_step(
                "episodic.ingest_turn",
                layer="write",
                outputs={"staging_id": staging_id, "flushed_to_storage": flushed,
                         "staging_after": self.staging_count()},
            )
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
                           entities, source, session_id, cognitive_json
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
                           entities, source, session_id, cognitive_json
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
            if _brain_trace and _brain_trace.TRACE_ENABLED and flushed:
                grouped = sum(1 for g in (groups or []) if len(g) > 1)
                _brain_trace.record_step(
                    "episodic.flush_staging",
                    layer="write",
                    inputs={"staging_rows": flushed, "batch_limit": batch},
                    outputs={"episodes_written": flushed, "groups_merged": grouped},
                    factors=[
                        f"EMC-6 grouping: {grouped} group(s) merged from staging turns",
                        f"embed queue: {'enqueued' if EMC_EMBED_ON_FLUSH else 'skipped (EMC_EMBED_ON_FLUSH=0)'}",
                    ],
                )
            return flushed

    def _inscribe(self, row: sqlite3.Row | tuple) -> int:
        (
            _sid, user_id, timestamp, date, trace,
            valence_tag, arousal_score, salience_score,
            entities, source, session_id, cognitive_json,
        ) = row

        cur = self._conn.execute(
            """
            INSERT INTO emc_storage (
                user_id, timestamp, date, trace, encoding,
                valence_tag, arousal_score, salience_score,
                entities, source, session_id, cognitive_json
            ) VALUES (?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id, timestamp, date, trace,
                valence_tag, arousal_score, salience_score,
                entities, source, session_id, cognitive_json,
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

    # ── EMC-3: KNN + FTS5 + RRF recall ────────────────────────────────────────

    def search(
        self,
        query: str,
        *,
        limit: int | None = None,
        query_vector: list[float] | None = None,
        user_id: str | None = None,
    ) -> list[dict]:
        """KNN + FTS5 → RRF over emc_storage. Returns payload dicts."""
        if not EMC_ENABLED or not EMC_RECALL_ENABLED:
            return []
        top_k = EMC_RECALL_LIMIT if limit is None else max(0, int(limit))
        if top_k <= 0:
            return []
        uid = user_id or self._user_id
        q = (query or "").strip()
        if not q:
            return []

        # ── recall result cache ───────────────────────────────────────────────
        # Short-TTL per (user, query, limit) cache so every non-greeting turn
        # doesn't re-run KNN+FTS+RRF against emc_vec/emc_fts.
        cache_key = (uid, " ".join(q.casefold().split()), int(top_k))
        now = time.monotonic()
        with self._recall_cache_lock:
            cached = self._recall_cache.get(cache_key)
            if cached and now - cached[0] <= EMC_RECALL_CACHE_TTL:
                self._recall_cache.move_to_end(cache_key)
                hits = [dict(r) for r in cached[1]]
                try:
                    self._touch_episodes([r["id"] for r in hits])
                except Exception:
                    pass
                return hits
            if cached:
                self._recall_cache.pop(cache_key, None)

        vector = None
        try:
            if query_vector is not None:
                vector = list(query_vector)
            else:
                emb = self._embedder
                if hasattr(emb, "embed_query"):
                    vector = list(emb.embed_query(q))
                else:
                    vector = list(emb.embed([q]))[0]
        except Exception as e:
            log.debug("EMC search embed failed: %s", e)

        fts_query = _sanitize_fts_query(q)

        with self._lock:
            rank_knn: dict[int, int] = {}
            rank_fts: dict[int, int] = {}
            if vector is not None:
                try:
                    vec_blob = sqlite_vec.serialize_float32(vector)
                    k = max(int(EMC_KNN_LIMIT) * KNN_MATCH_OVERSCAN, KNN_MATCH_K_MIN)
                    knn_rows = self._conn.execute(
                        """
                        SELECT v.rowid AS id, v.distance AS dist
                        FROM emc_vec v
                        JOIN emc_storage s ON s.id = v.rowid
                        WHERE v.embedding MATCH ?
                          AND v.k = ?
                          AND s.user_id = ?
                          AND (s.superseded_by IS NULL)
                        ORDER BY v.distance ASC
                        LIMIT ?
                        """,
                        (vec_blob, k, uid, EMC_KNN_LIMIT),
                    ).fetchall()
                    rank_knn = {int(r[0]): i + 1 for i, r in enumerate(knn_rows)}
                except Exception as e:
                    log.debug("EMC KNN failed: %s", e)

            if fts_query:
                try:
                    fts_rows = self._conn.execute(
                        """
                        SELECT s.id
                        FROM emc_fts
                        JOIN emc_storage s ON s.id = emc_fts.rowid
                        WHERE emc_fts MATCH ?
                          AND s.user_id = ?
                          AND (s.superseded_by IS NULL)
                        ORDER BY rank
                        LIMIT ?
                        """,
                        (fts_query, uid, EMC_FTS_LIMIT),
                    ).fetchall()
                    rank_fts = {int(r[0]): i + 1 for i, r in enumerate(fts_rows)}
                except Exception as e:
                    log.debug("EMC FTS failed: %s", e)

            candidate_ids = set(rank_knn) | set(rank_fts)
            if not candidate_ids:
                return []

            scores: dict[int, float] = {}
            for eid in candidate_ids:
                s = 0.0
                if eid in rank_knn:
                    s += 1.0 / (EMC_RRF_K + rank_knn[eid])
                if eid in rank_fts:
                    s += 1.0 / (EMC_RRF_K + rank_fts[eid])
                scores[eid] = s

            ordered = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)[:top_k]
            if not ordered:
                return []

            placeholders = ",".join("?" * len(ordered))
            rows = self._conn.execute(
                f"""
                SELECT id, timestamp, date, trace, valence_tag, arousal_score,
                       salience_score, entities, source, session_id, recall_count, cognitive_json
                FROM emc_storage
                WHERE id IN ({placeholders})
                """,
                ordered,
            ).fetchall()
            by_id = {int(r[0]): r for r in rows}

            results: list[dict] = []
            for eid in ordered:
                row = by_id.get(eid)
                if not row:
                    continue
                entities = None
                if row[7]:
                    try:
                        entities = json.loads(row[7])
                    except Exception:
                        entities = None
                results.append({
                    "id": int(row[0]),
                    "timestamp": row[1],
                    "date": row[2],
                    "trace": row[3],
                    "memory": row[3],
                    "valence_tag": row[4],
                    "arousal_score": row[5],
                    "salience_score": row[6],
                    "entities": entities,
                    "source": row[8],
                    "session_id": row[9],
                    "recall_count": int(row[10] or 0),
                    "cognitive_state": json.loads(row[11]) if row[11] else None,
                    "_recall_score": scores.get(eid, 0.0),
                    "_emc": True,
                })

            self._touch_episodes([r["id"] for r in results])
            with self._recall_cache_lock:
                self._recall_cache[cache_key] = (now, [dict(r) for r in results])
                while len(self._recall_cache) > EMC_RECALL_CACHE_SIZE:
                    self._recall_cache.popitem(last=False)
            return results

    def _touch_episodes(self, ids: list[int]) -> None:
        if not ids:
            return
        now = _utc_now_iso()
        try:
            placeholders = ",".join("?" * len(ids))
            self._conn.execute(
                f"""
                UPDATE emc_storage
                SET recall_count = recall_count + 1,
                    last_recalled_at = ?
                WHERE id IN ({placeholders})
                """,
                [now] + ids,
            )
            self._conn.commit()
        except Exception as e:
            log.debug("EMC touch failed: %s", e)

    def format_for_context(self, episodes: list[dict], *, max_chars: int | None = None) -> str | None:
        """Format episodic hits as a compact <episodic_context> block."""
        if not episodes:
            return None
        budget = EMC_CONTEXT_CHARS if max_chars is None else max(40, int(max_chars))
        lines = [
            "<episodic_context>",
            "Past conversation moments (what happened), not durable facts. "
            "Use silently for continuity. Never quote this block or claim total recall.",
            "",
        ]
        kept = False
        for ep in episodes:
            trace = (ep.get("trace") or ep.get("memory") or "").strip()
            if not trace:
                continue
            kept = True
            if len(trace) > EMC_CONTEXT_EPISODE_CHARS:
                trace = trace[:EMC_CONTEXT_EPISODE_CHARS].rstrip() + "…"
            when = ep.get("date") or (ep.get("timestamp") or "")[:10] or ""
            if when:
                lines.append(f"  - [{when}] {trace}")
            else:
                lines.append(f"  - {trace}")
        if not kept:
            return None
        lines.append("</episodic_context>")
        block = "\n".join(lines)
        closing_tag = "\n</episodic_context>"
        if len(block) > budget:
            block = block[:budget - len(closing_tag)].rstrip() + closing_tag
        return block

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        try:
            self.flush_all()
        except Exception as e:
            log.debug("EMC flush_all on close: %s", e)
        # Stop the embed worker thread if it's running
        if self._embed_worker is not None and self._embed_worker.is_alive():
            try:
                self._embed_queue.put(None)
                self._embed_worker.join(timeout=2.0)
            except Exception as e:
                log.debug("EMC embed worker shutdown: %s", e)
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass


# ── EpisodicMemory: per-AikoMemorize integration facade ──────────────────────
# Owns the per-user EpisodicStore cache and exposes the methods AikoMemorize
# needs: queue_episode, _format_episodes_for_context, and lifecycle hooks
# for switch_user. Previously these were monkey-patched onto AikoMemorize at
# boot from a separate emc2_wire module — moving them onto a real class
# makes the relationship explicit, removes the runtime patch dance, and
# makes the integration testable in isolation.


class EpisodicMemory:
    """
    Per-AikoMemorize facade for the EMC episodic store.

    AikoMemorize holds one of these and delegates episodic ingest + recall
    to it. The facade owns the per-user store cache (lazy, one per user_id)
    so a single AikoMemorize process can serve multiple users in sequence
    without leaking connections across switches.
    """

    def __init__(self, memorize) -> None:
        # memorize is an AikoMemorize; we keep a back-ref so we can reach
        # its embedder, user_id resolver, and shared contextvar defaults
        # without copying them.
        self._memorize = memorize
        self._stores: dict[str, EpisodicStore] = {}
        self._lock = threading.RLock()

    def get_store(self, user_id: str | None = None) -> EpisodicStore | None:
        """Return (lazily creating) the EpisodicStore for a user. None if EMC is off."""
        if not EMC_ENABLED:
            return None
        uid = user_id or self._memorize.get_user_id()
        with self._lock:
            store = self._stores.get(uid)
            if store is not None:
                return store
            try:
                from cognition.memory.schema import _memory_db_path_for_user
                store = EpisodicStore(
                    _memory_db_path_for_user(uid),
                    user_id=uid,
                    embedder=self._memorize._mem._embedder,
                )
            except Exception as e:
                log.debug("episode store init failed: %s", e)
                return None
            self._stores[uid] = store
            # LRU-cap: close+evict oldest past 8 users so a multi-user LAN
            # box can't accumulate open SQLite handles without bound.
            # (close_all already clears on switch_user; this covers the
            # concurrent-users case that never switches.)
            while len(self._stores) > 8:
                oldest = next(iter(self._stores))
                if oldest == uid:
                    break
                try:
                    self._stores.pop(oldest).close()
                except Exception:
                    self._stores.pop(oldest, None)
            return store

    def queue_episode(
        self,
        user_input: str,
        response_text: str,
        cognitive_state: dict | None = None,
        user_id: str | None = None,
    ) -> None:
        """EMC-2: accept one conversation turn pair into the episodic buffer.

        Used by AikoThink._store_async; safe to call before any user is
        bound (no-op when EMC is disabled or the store can't be opened).
        """
        try:
            uid = self._memorize._resolve_user_id(user_id)
            store = self.get_store(user_id=uid)
            if store is None:
                return
            store.ingest_turn(
                user_input, response_text,
                user_id=uid, cognitive_state=cognitive_state,
            )
        except Exception as e:
            log.debug("queue_episode skipped: %s", e)

    def format_for_context(self, query: str, query_vector=None) -> str | None:
        """Render the <episodic_context> block for the current query, or None.

        Returns None when EMC is disabled, recall is disabled, the query is
        empty, the store can't be opened, or there are no hits. The caller
        (AikoMemorize.format_for_context) decides how to budget this block
        against the SM block.
        """
        if not EMC_ENABLED or not EMC_RECALL_ENABLED or EMC_RECALL_LIMIT <= 0:
            return None
        if not (query or "").strip():
            return None
        store = self.get_store()
        if store is None:
            return None
        try:
            hits = store.search(
                query,
                limit=EMC_RECALL_LIMIT,
                user_id=self._memorize.get_user_id(),
                query_vector=query_vector,
            )
        except Exception as e:
            log.debug("EMC format_for_context search failed: %s", e)
            return None
        if not hits:
            return None
        return store.format_for_context(hits)

    def close_all(self) -> None:
        """Flush + close every cached store. Called from AikoMemorize.switch_user."""
        with self._lock:
            for store in list(self._stores.values()):
                try:
                    store.flush_all()
                except Exception:
                    log.debug("episode store flush on switch_user failed")
                try:
                    store.close()
                except Exception:
                    log.debug("episode store close on switch_user failed")
            self._stores.clear()

    def close_one(self) -> None:
        """Close the singleton _episode_store (legacy single-store path)."""
        legacy = getattr(self._memorize, "_episode_store", None)
        if legacy is None:
            return
        try:
            legacy.flush_all()
            legacy.close()
        except Exception:
            log.debug("legacy episode store flush/close failed")
        self._memorize._episode_store = None


# ── EMC-3: backward-compat attach (idempotent) ───────────────────────────────
# Previously episode_recall.py monkey-patched search/format_for_context onto
# EpisodicStore at boot. With the merge they're already methods, but callers
# that still call attach_recall_to_store() keep working as a no-op.

def attach_recall_to_store() -> None:
    """No-op stub kept for backward-compat. Methods are now defined directly
    on EpisodicStore. Kept idempotent so existing boot code keeps working."""
    log.debug("EMC-3 attach_recall_to_store: no-op (methods are native)")


# ── EMC-4: dream distillation ────────────────────────────────────────────────

_DISTILL_PROMPT = """\
You extract durable long-term facts from past conversation moments.
Only keep stable facts about the user (preferences, identity, plans, relationships).
Skip greetings, one-off logistics, and anything already ephemeral.
Output a JSON array of strings. Empty array if nothing durable.
Each fact must be a single short sentence in third person about the user.

Moments:
{moments}

JSON array:
"""


def ensure_distilled_column(conn) -> None:
    """Add distilled_at + distilled_into to emc_storage if missing (idempotent).

    distilled_into is a JSON array of semantic-memory ids the episode
    consolidated into (EM→SM link used by the LTM/ITM studios).
    """
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(emc_storage)").fetchall()}
        if "distilled_at" not in cols:
            conn.execute("ALTER TABLE emc_storage ADD COLUMN distilled_at TEXT")
            conn.commit()
            log.info("EMC-4: added emc_storage.distilled_at")
        if "distilled_into" not in cols:
            conn.execute("ALTER TABLE emc_storage ADD COLUMN distilled_into TEXT")
            conn.commit()
            log.info("EMC-4: added emc_storage.distilled_into")
    except Exception as e:
        log.debug("EMC-4 distilled_at migration: %s", e)


def _parse_facts(raw: str) -> list[str]:
    text = (raw or "").strip()
    if not text:
        return []
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
    out: list[str] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
            elif isinstance(item, dict):
                fact = (item.get("fact") or item.get("text") or "").strip()
                if fact:
                    out.append(fact)
    return out


def _candidate_episodes(store, user_id: str, limit: int) -> list[dict]:
    ensure_distilled_column(store._conn)
    with store._lock:
        rows = store._conn.execute(
            """
            SELECT id, timestamp, date, trace, salience_score, recall_count
            FROM emc_storage
            WHERE user_id = ?
              AND (superseded_by IS NULL)
              AND (distilled_at IS NULL)
              AND length(trace) >= ?
            ORDER BY
              COALESCE(salience_score, 0) DESC,
              COALESCE(recall_count, 0) DESC,
              timestamp DESC
            LIMIT ?
            """,
            (user_id, EMC_DREAM_MIN_CHARS, limit),
        ).fetchall()
    return [
        {
            "id": int(r[0]),
            "timestamp": r[1],
            "date": r[2],
            "trace": r[3],
            "salience_score": r[4],
            "recall_count": int(r[5] or 0),
        }
        for r in rows
    ]


def _mark_distilled(store, ids: list[int], *, distilled_into: list[str] | None = None) -> None:
    if not ids:
        return
    now = _utc_now_iso()
    into_json = json.dumps(list(distilled_into or []), ensure_ascii=False)
    with store._lock:
        for eid in ids:
            store._conn.execute(
                "UPDATE emc_storage SET distilled_at = ?, distilled_into = ? WHERE id = ?",
                (now, into_json, eid),
            )
        store._conn.commit()


def _llm_distill(client, model: str, moments: list[str]) -> list[str]:
    if not moments or client is None:
        return []
    block = "\n\n".join(f"[{i+1}]\n{m}" for i, m in enumerate(moments))
    prompt = _DISTILL_PROMPT.format(moments=block[:6000])
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            max_tokens=EMC_DREAM_MAX_TOKENS,
            temperature=0.0,
            timeout=45.0,
        )
        raw = (resp.choices[0].message.content or "").strip()
        return _parse_facts(raw)
    except Exception as e:
        log.debug("EMC-4 LLM distill failed: %s", e)
        return []


def distill_episodes(
    memorize,
    *,
    user_id: str | None = None,
    dry_run: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    """Distill undistrilled episodes into semantic facts via memorize.add_raw."""
    result = {
        "candidates": 0,
        "distilled_episodes": 0,
        "facts_written": 0,
        "dry_run": dry_run,
        "enabled": bool(EMC_DREAM_ENABLED and EMC_ENABLED),
    }
    if not EMC_ENABLED or not EMC_DREAM_ENABLED:
        return result

    top = EMC_DREAM_LIMIT if limit is None else max(0, int(limit))
    if top <= 0:
        return result

    uid = user_id or memorize.get_user_id()
    store = None
    try:
        if hasattr(memorize, "episodic") and memorize.episodic is not None:
            store = memorize.episodic.get_store(uid)
        else:
            # Fallback for callers that don't yet have the EpisodicMemory
            # facade (e.g. legacy tests using a stripped-down AikoMemorize).
            store = memorize._get_episode_store(uid)
    except Exception as e:
        log.debug("EMC-4 no episode store: %s", e)
        return result
    if store is None:
        return result

    try:
        store.flush_all()
    except Exception as e:
        log.debug("EMC-4 flush_all failed: %s", e)

    candidates = _candidate_episodes(store, uid, top)
    result["candidates"] = len(candidates)
    if not candidates:
        return result

    backend = getattr(memorize, "_mem", None)
    client = getattr(backend, "_client", None) if backend else None
    model = getattr(backend, "_model", None) or "ministral"

    facts_written = 0
    distilled_ids: list[int] = []

    for i in range(0, len(candidates), EMC_DREAM_BATCH):
        batch = candidates[i : i + EMC_DREAM_BATCH]
        moments = [c["trace"] for c in batch if (c.get("trace") or "").strip()]
        facts = _llm_distill(client, model, moments)
        if dry_run:
            log.info(
                "EMC-4 dry-run batch episodes=%d facts=%d sample=%r",
                len(batch),
                len(facts),
                (facts[:2] if facts else []),
            )
            # dry-run: never mark; only count batches that would produce facts
            if facts:
                distilled_ids.extend(c["id"] for c in batch)
            continue

        # Only mark distilled when the LLM returned durable facts.
        # Empty extract → leave distilled_at NULL so the episodes can retry next dream.
        if not facts:
            log.debug(
                "EMC-4 empty extract; not marking episodes=%s",
                [c["id"] for c in batch],
            )
            continue

        batch_success = True
        batch_mem_ids: list[str] = []
        for fact in facts:
            try:
                mid = memorize.add_raw(fact, user_id=uid, pinned=False)
                if mid:
                    facts_written += 1
                    batch_mem_ids.append(str(mid))
            except Exception as e:
                log.debug("EMC-4 add_raw failed: %s", e)
                batch_success = False

        # Only mark distilled if all facts were written successfully
        if batch_success:
            ids = [c["id"] for c in batch]
            _mark_distilled(store, ids, distilled_into=batch_mem_ids)
            distilled_ids.extend(ids)

    result["distilled_episodes"] = len(distilled_ids)
    result["facts_written"] = facts_written
    log.info(
        "EMC-4 distill candidates=%d episodes=%d facts=%d dry_run=%s",
        result["candidates"],
        result["distilled_episodes"],
        facts_written,
        dry_run,
    )
    return result


def attach_dream_hook() -> None:
    """Wrap AikoMemorize.dream to run EMC distill after SM consolidation."""
    from cognition.memory.memorize import AikoMemorize

    if getattr(AikoMemorize, "_emc4_dream_patched", False):
        return

    _orig = AikoMemorize.dream

    def dream(self, user_id=None, dry_run=False, threshold=None, **kwargs):
        if threshold is None:
            threshold = DREAM_MERGE_THRESHOLD
        result = _orig(self, user_id=user_id, dry_run=dry_run, threshold=threshold, **kwargs)
        try:
            emc = distill_episodes(self, user_id=user_id, dry_run=dry_run)
            if isinstance(result, dict):
                result["emc_distill"] = emc
        except Exception as e:
            log.warning("EMC-4 distill after dream failed: %s", e)
            if isinstance(result, dict):
                result["emc_distill_error"] = str(e)
        return result

    AikoMemorize.dream = dream  # type: ignore[method-assign]
    AikoMemorize._emc4_dream_patched = True  # type: ignore[attr-defined]
    log.debug("EMC-4 dream hook attached")
