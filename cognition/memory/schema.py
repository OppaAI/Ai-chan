"""Memory-domain schema, constants, and low-level sqlite access.

Generic SQLite + sqlite-vec helpers (connection bootstrap, user-scoped vec KNN /
FTS, fetch_by_ids, rrf, rank) live in cognition.memory.vecstore — the shared
store layer for memory/knowledge/experience. This module holds what is
memory-specific: the ``memories`` DDL, per-user paths, env tunables, and the
low-level memories/memories_vec access helpers used by memorize._MemoryBackend.
"""
from __future__ import annotations

import os
import re
import sqlite3
import struct
import tempfile
import threading
import json

import sqlite_vec
from system.log import get_logger
from system.userspace import current_user_id
from cognition.memory.vecstore import initialize_store_db, resolve_user_db_path
from cognition.memory.env import env_bool, env_flag, env_float, env_int

log = get_logger(__name__)


_GUEST_DB: "tempfile.NamedTemporaryFile | None" = None
_GUEST_DB_LOCK = threading.Lock()


def _guest_memory_db() -> str:
    """Return a tempfile-backed path for the guest user's memory DB.

    Unlike ``:memory:`` (which lives entirely in process heap and grows
    unbounded), a tempfile is paged by the OS and reclaimed on restart.
    """
    global _GUEST_DB
    if _GUEST_DB is not None:
        return _GUEST_DB.name
    with _GUEST_DB_LOCK:
        if _GUEST_DB is not None:
            return _GUEST_DB.name
        _GUEST_DB = tempfile.NamedTemporaryFile(suffix=".db", delete=True)
        return _GUEST_DB.name

BOOT_LABELS = {
    'mem_embed':         'Opening sqlite-vec store and loading embedder...',
    'mem_display_name':  'Resolving display name...',
    'mem_cleanup':       'Running memory cleanup...',
    'mem_ready':         'Memory backend ready',
}

def _env_bool(name: str, default: str = "1") -> bool:
    return env_bool(name, default)

# ── constants ─────────────────────────────────────────────────────────────────

EMBED_MODEL = os.getenv("EMBED_MODEL", "ferrisS/harrier-oss-v1-270m-fastembed")
EMBED_DIMS  = int(os.getenv("EMBED_DIMS", "640"))
EMBED_QUERY_INSTRUCT = os.getenv("EMBED_QUERY_INSTRUCT", "Retrieve relevant memories that answer the query").strip()
RRF_K       = 60          # standard RRF constant — dampens outlier ranks
KNN_LIMIT   = 20          # candidates fetched before RRF re-rank (wide pass)
FTS_LIMIT   = 20          # candidates fetched before RRF re-rank (wide pass)
GRAPH_LIMIT = 20          # candidates fetched before RRF re-rank (wide pass)
QUICK_KNN_LIMIT = int(os.getenv("QUICK_KNN_LIMIT", "6"))   # narrow first-pass candidate count
QUICK_FTS_LIMIT = int(os.getenv("QUICK_FTS_LIMIT", "6"))   # narrow first-pass candidate count
QUICK_GRAPH_LIMIT = int(os.getenv("QUICK_GRAPH_LIMIT", "6"))  # narrow first-pass candidate count
# vec0 MATCH KNN oversampling: the user_id/status JOIN filters rows AFTER the
# vec0 scan picks the k nearest, so request a healthy multiple of the caller's
# limit and let the outer ORDER BY + LIMIT trim back down.
KNN_MATCH_OVERSCAN = int(os.getenv("KNN_MATCH_OVERSCAN", "16"))
KNN_MATCH_K_MIN = int(os.getenv("KNN_MATCH_K_MIN", "32"))
MEMORY_RECALL_SCORE_THRESHOLD = float(os.getenv("MEMORY_RECALL_SCORE_THRESHOLD", "0.015"))
MEMORY_RANK_RECENCY_WEIGHT = float(os.getenv("MEMORY_RANK_RECENCY_WEIGHT", "0.004"))
MEMORY_RANK_RECENCY_HALF_LIFE_DAYS = float(os.getenv("MEMORY_RANK_RECENCY_HALF_LIFE_DAYS", "30"))
MEMORY_RANK_ACCESS_WEIGHT = float(os.getenv("MEMORY_RANK_ACCESS_WEIGHT", "0.002"))
# Bumped from 0.002 -> 0.01 so pinned status is a meaningful tiebreaker
# under RRF (~0.016 at rank 1), without beating a clearly better unpinned hit.
MEMORY_RANK_PINNED_WEIGHT = float(os.getenv("MEMORY_RANK_PINNED_WEIGHT", "0.01"))
# Entity-graph tiebreaker. The term is MEMORY_RANK_GRAPH_WEIGHT / (RRF_K + rank),
# same shape as the KNN/FTS RRF terms (1.0 / (RRF_K + rank)) rather than a
# flat add like MEMORY_RANK_PINNED_WEIGHT — so its numerator needs to be
# larger than 0.01 to land in the same range as a flat 0.01 tiebreaker.
# At best rank (1), the term maxes out at WEIGHT / (RRF_K + 1) = WEIGHT / 61.
# To match MEMORY_RANK_PINNED_WEIGHT's flat +0.01 nudge at rank 1, WEIGHT
# needs to be ~0.01 * 61 ≈ 0.6 — a numerator of 0.01 here (an easy mistake,
# since it *looks* like the same scale as the pinned weight) actually
# maxes out around 0.00016, ~100x too small to move any ranking decision.
# 0.6 keeps graph a mild tiebreaker (never enough to bury a clearly
# stronger KNN/FTS hit) without being mathematically inert. Set to 0 to
# disable graph participation in scoring without touching any call sites.
MEMORY_RANK_GRAPH_WEIGHT = float(os.getenv("MEMORY_RANK_GRAPH_WEIGHT", "0.6"))

# Phase 3 entity importance (see cognition.memory.entity)
MEMORY_RANK_ENTITY_IMPORTANCE_WEIGHT = float(os.getenv("MEMORY_RANK_ENTITY_IMPORTANCE_WEIGHT", "0.008"))

MEMORY_SPREADING_ENABLED = os.getenv("MEMORY_SPREADING_ENABLED", "1").lower() in {"1", "true", "yes", "on"}
MEMORY_SPREADING_MAX_EXTRA = max(0, int(os.getenv("MEMORY_SPREADING_MAX_EXTRA", "5")))
MEMORY_SPREADING_SCORE_WEIGHT = float(os.getenv("MEMORY_SPREADING_SCORE_WEIGHT", "0.01"))

MEMORY_SEARCH_CACHE_SIZE = int(os.getenv("MEMORY_SEARCH_CACHE_SIZE", 128))
MEMORY_SEARCH_CACHE_TTL  = float(os.getenv("MEMORY_SEARCH_CACHE_TTL", 20.0))
MEMORY_CONTEXT_FACT_CHARS  = int(os.getenv("MEMORY_CONTEXT_FACT_CHARS", 220))
MEMORY_CONTEXT_TOTAL_CHARS = int(os.getenv("MEMORY_CONTEXT_TOTAL_CHARS", 1200))
MEMORY_LIFECYCLE_BATCH_SIZE = int(os.getenv("MEMORY_LIFECYCLE_BATCH_SIZE", 500))

# L3 persona cache — an always-hydrated, cheap blob of the most stable
# identity facts (kind='identity'), prepended to context every turn WITHOUT
# going through KNN/FTS/graph. Distinct from the RRF recall path so the common
# "who am I talking to" turn never pays for a full search, and so stable
# identity facts are never drowned out by turn-specific recall.
PERSONA_RECALL_LIMIT   = int(os.getenv("PERSONA_RECALL_LIMIT", 6))
PERSONA_CONTEXT_CHARS  = int(os.getenv("PERSONA_CONTEXT_CHARS", 600))
PERSONA_CACHE_TTL      = float(os.getenv("PERSONA_CACHE_TTL", 60.0))
# L2 scene blocks — a scene is itself a memory row with kind='scene'; the
# atomic facts it summarizes carry scene_id back to it. These knobs bound how
# many/hot scene summaries we surface as a cheap context bootstrap.
SCENE_CONTEXT_LIMIT    = int(os.getenv("SCENE_CONTEXT_LIMIT", 3))
SCENE_CONTEXT_CHARS    = int(os.getenv("SCENE_CONTEXT_CHARS", 600))
SCENE_MEMBER_LIMIT     = int(os.getenv("SCENE_MEMBER_LIMIT", 6))
# L0 conversation traceability — optional per-turn raw-log retention (off by
# default; Aiko's lean-memory stance keeps the journal as its only L0).
L0_CONVERSATION_LOG_ENABLED = _env_bool("L0_CONVERSATION_LOG_ENABLED", "0")

# Recall hard-timeout — protects the turn from a slow local embed/stall on the
# recall future. On timeout the recall is skipped (empty) instead of blocking.
MEMORY_RECALL_TIMEOUT  = float(os.getenv("MEMORY_RECALL_TIMEOUT", 5.0))

# Recency-among-relevant rerank — candidates clearing this score are
# reordered by created_at descending among themselves (see module docstring
# stage 3). Independent of MEMORY_RANK_RECENCY_WEIGHT's continuous blend.
MEMORY_RECENCY_RERANK_ENABLED = _env_bool("MEMORY_RECENCY_RERANK_ENABLED", "1")
MEMORY_RECENCY_RERANK_THRESHOLD = float(os.getenv("MEMORY_RECENCY_RERANK_THRESHOLD", "0.012"))

# Async write queue — idle-grace window before an enqueued write is allowed
# to run (avoids contending with the shared LLM mid-turn), and a hard cap so
# a write is never held back indefinitely if the caller's turn state gets
# stuck "active". See AikoMemorize.queue_write().
MEMORY_WRITE_IDLE_GRACE = float(os.getenv("MEMORY_WRITE_IDLE_GRACE", 3.0))
MEMORY_WRITE_MAX_WAIT = float(os.getenv("MEMORY_WRITE_MAX_WAIT", 45.0))

MEMORY_CROSS_STORE_ENABLED = os.getenv("MEMORY_CROSS_STORE_ENABLED", "1").lower() in {
    "1", "true", "yes", "on",
}

def _env_flag(name: str, default: str = "1") -> bool:
    return env_flag(name, default)

def _env_float(name: str, default: float) -> float:
    return env_float(name, default)

def _env_int(name: str, default: int) -> int:
    return env_int(name, default)

MEMORY_STATE_TAGS_ENABLED = _env_flag("MEMORY_STATE_TAGS_ENABLED", "1")
MEMORY_NEG_RECALL_AVOID = _env_flag("MEMORY_NEG_RECALL_AVOID", "1")
MEMORY_NEG_RECALL_AVOID_WEIGHT = _env_float("MEMORY_NEG_RECALL_AVOID_WEIGHT", 0.015)
MEMORY_NEG_RECALL_AVOID_EXCEPT = _env_flag("MEMORY_NEG_RECALL_AVOID_EXCEPT", "1")
MEMORY_SUPERSESSION_NARRATIVE = _env_flag("MEMORY_SUPERSESSION_NARRATIVE", "1")
MEMORY_SUPERSESSION_NARRATIVE_MAX = max(0, _env_int("MEMORY_SUPERSESSION_NARRATIVE_MAX", 2))

MEMORY_CROSS_STORE_CONTEXT_CHARS = max(0, _env_int("MEMORY_CROSS_STORE_CONTEXT_CHARS", 800))


def _default_user_id(user_id: str | None = None) -> str:
    return user_id or current_user_id()

STATUS_ACTIVE = "active"
STATUS_SUPERSEDED = "superseded"

KIND_FACT = "fact"
KIND_SCENE = "scene"
KIND_EPISODE = "episode"  # reserved; true EMC lives in emc_* tables (episode.py)
SOURCE_CHAT = "chat"
SOURCE_PIN = "pin"
SOURCE_LEGACY = "legacy"

_WS_RE = re.compile(r"\s+")

_PHASE_A_COLUMNS: tuple[tuple[str, str], ...] = (
    ("status", "TEXT NOT NULL DEFAULT 'active'"),
    ("supersedes_id", "TEXT"),
    ("kind", "TEXT NOT NULL DEFAULT 'fact'"),
    ("source", "TEXT NOT NULL DEFAULT 'legacy'"),
    ("entities", "TEXT NOT NULL DEFAULT '[]'"),
  # Phase 2 spacing: distinct local calendar days this memory was recalled.
    ("access_day_count", "INTEGER NOT NULL DEFAULT 0"),
  # Phase 4: turn-level emotion / salience tags (cheap, no LLM).
    ("valence_tag", "TEXT NOT NULL DEFAULT 'neutral'"),
    ("valence_score", "INTEGER"),  # -2..+2; NULL = legacy/unknown
    ("arousal_score", "INTEGER"),  # Phase 19: −2..+2 intensity; NULL = legacy
    ("salience_hit", "INTEGER NOT NULL DEFAULT 0"),
    ("state_json", "TEXT"),  # Phase 16: optional {"local_hour": 0-23}
)

# L2 scene blocks — one additive runtime column on memories. A scene row
# carries kind='scene' and is itself searchable; its atomic-fact members point
# back to it via scene_id. See ensure_l2_scene_schema().
_L2_SCENE_COLUMN: tuple[str, str] = ("scene_id", "TEXT")


def existing_columns(conn: sqlite3.Connection, table: str = "memories") -> set[str]:
    """Return the set of column names present on a table (for additive ALTERs)."""
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {str(r[1]) for r in rows}


def ensure_phase_a_schema(conn: sqlite3.Connection) -> list[str]:
    """Idempotent ALTER TABLE for Phase A columns + status index."""
    try:
        cols = existing_columns(conn)
    except sqlite3.Error:
        return []
    if "id" not in cols and "memory" not in cols:
        return []

    added: list[str] = []
    for name, decl in _PHASE_A_COLUMNS:
        if name in cols:
            continue
        try:
            conn.execute(f"ALTER TABLE memories ADD COLUMN {name} {decl}")
            added.append(name)
        except sqlite3.OperationalError as e:
            if "duplicate column" not in str(e).casefold():
                raise
    try:
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_user_status "
            "ON memories(user_id, status)"
        )
        conn.commit()
    except sqlite3.Error as e:
        log.debug("memory Phase A index: %s", e)
    if added:
        log.info("memory Phase A schema: added columns %s", added)
    return added


def ensure_l2_scene_schema(conn: sqlite3.Connection) -> list[str]:
    """Idempotent ALTER TABLE for the L2 scene_id column + kind index.

    Scene rows are distinguished by kind='scene'; the index makes the persona
    cache (kind='identity') and scene listing cheap without scanning the whole
    table. Calls ensure_phase_a_schema() first because 'kind' must exist to
    index it.
    """
    ensure_phase_a_schema(conn)
    added: list[str] = []
    cols = existing_columns(conn)
    name, decl = _L2_SCENE_COLUMN
    if name not in cols:
        try:
            conn.execute(f"ALTER TABLE memories ADD COLUMN {name} {decl}")
            added.append(name)
        except sqlite3.OperationalError as e:
            if "duplicate column" not in str(e).casefold():
                raise
    try:
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_user_kind "
            "ON memories(user_id, kind)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_user_scene "
            "ON memories(user_id, scene_id)"
        )
        conn.commit()
    except sqlite3.Error as e:
        log.debug("memory L2 scene index: %s", e)
    if added:
        log.info("memory L2 scene schema: added column %s", added)
    return added


def ensure_episode_schema(conn: sqlite3.Connection) -> list[str]:
    """Idempotent creation of EMC (episodic) tables.

    Delegates to cognition.memory.episode so the DDL lives in one place.
    Safe to call on every boot; no-op if tables already exist.
    """
    from cognition.memory.episode import ensure_episode_schema as _ensure
    return _ensure(conn)


def _active_sql(active_only: bool, alias: str = "m") -> str:
    """SQL fragment to restrict a query to active (non-superseded) memories.

    `alias` is the table alias used for `memories` in the calling query
    (default "m", matching _sqlite_knn_search/_fts_pass/_recent_or_important_memories).
    Callers with a different alias (e.g. _graph_pass, which joins memories
    as "mm") must pass it explicitly rather than string-replacing this
    function's output — the previous `.replace("m.status", "mm.status")`
    pattern silently produced broken SQL if this function's default alias
    or format ever changed.
    """
    if not active_only:
        return ""
    return f" AND ({alias}.status = 'active' OR {alias}.status IS NULL)"

_DDL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS memories (
    id               TEXT PRIMARY KEY,
    user_id          TEXT NOT NULL,
    memory           TEXT NOT NULL,
    created_at       TEXT NOT NULL,
    access_count     INTEGER NOT NULL DEFAULT 0,
    last_accessed_at TEXT NOT NULL DEFAULT 'never',
    pinned           INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    memory,
    id UNINDEXED,
    content='memories',
    content_rowid='rowid'
);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec USING vec0(
    id TEXT PRIMARY KEY,
    embedding FLOAT[{dims}] distance_metric=cosine
);

CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, memory, id)
    VALUES (new.rowid, new.memory, new.id);
END;

CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, memory, id)
    VALUES ('delete', old.rowid, old.memory, old.id);
END;

CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE OF memory ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, memory, id)
    VALUES ('delete', old.rowid, old.memory, old.id);
    INSERT INTO memories_fts(rowid, memory, id)
    VALUES (new.rowid, new.memory, new.id);
END;
""".format(dims=EMBED_DIMS)

def _sqlite_set_payload(
    conn: sqlite3.Connection,
    mem_id: str,
    payload: dict,
) -> None:
    """Update arbitrary column subset for a single memory row."""
    if not payload:
        return
    cols = ", ".join(f"{k} = ?" for k in payload)
    vals = list(payload.values()) + [mem_id]
    conn.execute(f"UPDATE memories SET {cols} WHERE id = ?", vals)
    conn.commit()


def _sqlite_batch_get_payloads(
    conn: sqlite3.Connection,
    mem_ids: list[str],
) -> dict:
    """
    Batch-fetch access_count + last_accessed_at in a single query.
    Returns {mem_id: (access_count, last_accessed_at)}.
    """
    if not mem_ids:
        return {}
    placeholders = ",".join("?" * len(mem_ids))
    rows = conn.execute(
        f"SELECT id, access_count, last_accessed_at FROM memories WHERE id IN ({placeholders})",
        mem_ids,
    ).fetchall()
    return {
        r["id"]: (r["access_count"] or 0, r["last_accessed_at"] or "never")
        for r in rows
    }


def _sqlite_get_vector(conn: sqlite3.Connection, mem_id: str) -> list[float]:
    """
    Retrieve the raw embedding for one memory from the vec0 table.
    Returns [] on miss or error.
    """
    row = conn.execute(
        "SELECT embedding FROM memories_vec WHERE id = ?", (mem_id,)
    ).fetchone()
    if row and row[0]:
        raw = row[0]
        if len(raw) % 4 != 0:  # not divisible by float32 size
            log.warning(f"Corrupted vector for {mem_id}, dropping")
            return []
        try:
            n = len(raw) // 4
            return list(struct.unpack(f"{n}f", raw))
        except struct.error as e:
            log.error(f"Vector deserialization failed: {e}")
            return []
    return []


def _sqlite_is_pinned(conn: sqlite3.Connection, mem_id: str) -> bool:
    """Return True if memories.pinned == 1 for this id. Defaults to False on error."""
    row = conn.execute(
        "SELECT pinned FROM memories WHERE id = ?", (mem_id,)
    ).fetchone()
    return bool(row and row[0])


def _sqlite_pinned_ids(conn: sqlite3.Connection, mem_ids: list[str]) -> set[str]:
    """Batch fetch pinned memory IDs from the canonical table."""
    ids = [str(mem_id) for mem_id in mem_ids if mem_id]
    if not ids:
        return set()
    placeholders = ",".join("?" * len(ids))
    rows = conn.execute(
        f"SELECT id FROM memories WHERE pinned = 1 AND id IN ({placeholders})",
        ids,
    ).fetchall()
    return {str(row["id"]) for row in rows}

def _sqlite_knn_search(
    conn: sqlite3.Connection,
    vector: list[float],
    user_id: str,
    limit: int,
    threshold: float | None = None,
    active_only: bool = True,
) -> list[sqlite3.Row]:
    """
    KNN cosine search against memories_vec, filtered by user_id, via the vec0
    MATCH index instead of a full-table scan. Tables are created with
    distance_metric=cosine (migrated on connect by ensure_vec0_cosine_metric),
    so `distance` is exactly 1 - cosine_similarity and the thresholds below
    match the historical vec_distance_cosine semantics.

    k is oversampled well past `limit` because the user_id/status filter is
    applied by the JOIN after the vec0 scan picks the k nearest rows overall;
    the outer ORDER BY + LIMIT restores the caller's requested candidate count.
    """
    vec_blob = sqlite_vec.serialize_float32(vector)
    status_sql = _active_sql(active_only)
    k = max(int(limit) * KNN_MATCH_OVERSCAN, KNN_MATCH_K_MIN)
    if threshold is not None:
        dist_ceil = 1.0 - threshold
        return conn.execute(
            """
            SELECT v.id, v.distance AS dist
            FROM memories_vec v
            JOIN memories m ON m.id = v.id
            WHERE v.embedding MATCH ?
              AND v.k = ?
              AND m.user_id = ?
              AND v.distance <= ?
              {status_sql}
            ORDER BY v.distance ASC
            LIMIT ?
            """.format(status_sql=status_sql),
            (vec_blob, k, user_id, dist_ceil, limit),
        ).fetchall()
    return conn.execute(
        """
        SELECT v.id, v.distance AS dist
        FROM memories_vec v
        JOIN memories m ON m.id = v.id
        WHERE v.embedding MATCH ?
          AND v.k = ?
          AND m.user_id = ?
          {status_sql}
        ORDER BY v.distance ASC
        LIMIT ?
        """.format(status_sql=status_sql),
        (vec_blob, k, user_id, limit),
    ).fetchall()


def _first_json_array(raw: str) -> str | None:
    """Extract the first complete top-level JSON array, correctly handling
    nested brackets and string escaping. Returns None if no array found."""
    i, n = 0, len(raw)
    start = raw.find("[")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for j in range(start, n):
        ch = raw[j]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    return raw[start:j + 1]
    return None

def _json_objects(text: str) -> list[dict]:
    """Pull every complete top-level JSON object out of some text.

    Used to salvage facts from a model response truncated mid-array (by a
    max_tokens cap) before the closing ``]`` ever arrives: each fully-formed
    ``{...}`` object is decoded and kept; a trailing incomplete object is
    skipped rather than failing the whole turn.
    """
    dec = json.JSONDecoder()
    out: list[dict] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch == "{":
            try:
                obj, end = dec.raw_decode(text, i)
            except json.JSONDecodeError:
                i += 1
                continue
            if isinstance(obj, dict):
                out.append(obj)
            i = end
        else:
            i += 1
    return out


def parse_json_array(raw) -> list | None:
    """Return an LLM response as a fact array without dropping truncated data.

    Tries a clean top-level-array parse first; if that fails (e.g. the response
    was cut off by a token cap), every complete JSON object is recovered so no
    already-extracted fact is lost. Returns a list or None when nothing usable.
    """
    if raw is None:
        return None
    if isinstance(raw, list):
        return raw
    text = str(raw).strip()
    if not text:
        return None
    arr = _first_json_array(text)
    if arr is not None:
        try:
            val = json.loads(arr)
            if isinstance(val, list):
                return val
        except json.JSONDecodeError:
            pass
    objs = _json_objects(text)
    return objs if objs else None

def _memory_db_path_for_user(uid: str) -> str:
    if uid == "guest":
        return _guest_memory_db()
    env_path = os.getenv("SQLITE_MEMORY_PATH", "").strip()
    if env_path:
        # NOTE: when set, every non-guest user shares this exact file path —
        # there's no per-user substitution here. Every query still filters
        # by the user_id column, so this doesn't corrupt or leak data across
        # users, but it does mean all non-guest users' memories live in one
        # physical .db file rather than one-file-per-user. Fine for a
        # single-tenant deployment (the common case this env var exists
        # for); if you're running this multi-tenant, don't set it.
        return os.path.expanduser(env_path)
    return str(resolve_user_db_path("memory/memory.db", user_id=uid))


def vacuum_memory_db(user_id: str | None = None) -> None:
    """Reclaim space after bulk memory deletes during maintenance."""
    uid = user_id or current_user_id()
    conn = initialize_store_db(_memory_db_path_for_user(uid), _DDL, user_id=uid, vector=True)
    try:
        # VACUUM cannot run inside a transaction. Python's sqlite3 module
        # implicitly opens one on the first DML statement even under
        # isolation_level="" (the default), so if initialize_store_db (or
        # anything else on this connection) issued an uncommitted INSERT/
        # UPDATE/DELETE before we get here, a bare `conn.execute("VACUUM")`
        # raises `OperationalError: cannot VACUUM from within a transaction`.
        # Commit anything pending, then force autocommit mode for the
        # VACUUM itself, then restore normal isolation.
        conn.commit()
        prior_isolation = conn.isolation_level
        conn.isolation_level = None
        try:
            conn.execute("VACUUM")
            conn.execute("ANALYZE")
        finally:
            conn.isolation_level = prior_isolation
    finally:
        conn.close()


__all__ = [
    "BOOT_LABELS",
    "EMBED_DIMS",
    "EMBED_MODEL",
    "EMBED_QUERY_INSTRUCT",
    "FTS_LIMIT",
    "GRAPH_LIMIT",
    "KIND_FACT",
    "KIND_SCENE",
    "KIND_EPISODE",
    "KNN_LIMIT",
    "L0_CONVERSATION_LOG_ENABLED",
    "MEMORY_CONTEXT_FACT_CHARS",
    "MEMORY_CONTEXT_TOTAL_CHARS",
    "MEMORY_CROSS_STORE_CONTEXT_CHARS",
    "MEMORY_CROSS_STORE_ENABLED",
    "MEMORY_LIFECYCLE_BATCH_SIZE",
    "MEMORY_NEG_RECALL_AVOID",
    "MEMORY_NEG_RECALL_AVOID_EXCEPT",
    "MEMORY_NEG_RECALL_AVOID_WEIGHT",
    "MEMORY_RANK_ACCESS_WEIGHT",
    "MEMORY_RANK_ENTITY_IMPORTANCE_WEIGHT",
    "MEMORY_RANK_GRAPH_WEIGHT",
    "MEMORY_RANK_PINNED_WEIGHT",
    "MEMORY_RANK_RECENCY_HALF_LIFE_DAYS",
    "MEMORY_RANK_RECENCY_WEIGHT",
    "MEMORY_RECALL_SCORE_THRESHOLD",
    "MEMORY_RECALL_TIMEOUT",
    "MEMORY_RECENCY_RERANK_ENABLED",
    "MEMORY_RECENCY_RERANK_THRESHOLD",
    "MEMORY_SEARCH_CACHE_SIZE",
    "MEMORY_SEARCH_CACHE_TTL",
    "MEMORY_SPREADING_ENABLED",
    "MEMORY_SPREADING_MAX_EXTRA",
    "MEMORY_SPREADING_SCORE_WEIGHT",
    "MEMORY_STATE_TAGS_ENABLED",
    "MEMORY_SUPERSESSION_NARRATIVE",
    "MEMORY_SUPERSESSION_NARRATIVE_MAX",
    "MEMORY_WRITE_IDLE_GRACE",
    "MEMORY_WRITE_MAX_WAIT",
    "PERSONA_CACHE_TTL",
    "PERSONA_CONTEXT_CHARS",
    "PERSONA_RECALL_LIMIT",
    "QUICK_FTS_LIMIT",
    "QUICK_GRAPH_LIMIT",
    "QUICK_KNN_LIMIT",
    "KNN_MATCH_K_MIN",
    "KNN_MATCH_OVERSCAN",
    "RRF_K",
    "SCENE_CONTEXT_CHARS",
    "SCENE_CONTEXT_LIMIT",
    "SCENE_MEMBER_LIMIT",
    "SOURCE_CHAT",
    "SOURCE_LEGACY",
    "SOURCE_PIN",
    "STATUS_ACTIVE",
    "STATUS_SUPERSEDED",
    "_DDL",
    "_GUEST_DB",
    "_GUEST_DB_LOCK",
    "_L2_SCENE_COLUMN",
    "_PHASE_A_COLUMNS",
    "_WS_RE",
    "_active_sql",
    "_default_user_id",
    "_env_bool",
    "_env_flag",
    "_env_float",
    "_env_int",
    "_first_json_array",
    "_guest_memory_db",
    "_memory_db_path_for_user",
    "_sqlite_batch_get_payloads",
    "_sqlite_get_vector",
    "_sqlite_is_pinned",
    "_sqlite_knn_search",
    "_sqlite_pinned_ids",
    "_sqlite_set_payload",
    "ensure_episode_schema",
    "ensure_l2_scene_schema",
    "ensure_phase_a_schema",
    "existing_columns",
    "parse_json_array",
    "vacuum_memory_db",
]
