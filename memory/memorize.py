"""
memory/memorize.py
Aiko's persistent memory — custom backend via sqlite-vec + HarrierEmbedder (GGUF/llama.cpp).
Abstracts all memory calls so think.py stays clean.

Memory lifecycle:
  - Every search() call increments access_count and updates last_accessed_at
    in the memories table, enabling Ebbinghaus-style exponential decay scoring.
  - dream() runs nightly (00:00) as a consolidation pass — no new vectors
    are written. It boosts salient memories, merges near-duplicates, then
    prunes decayed entries. Order matters: boost before prune so boosted
    memories aren't immediately swept.
  - cleanup() deletes memories below decay threshold, with grace period
    protection for newly created entries.
  - Decay logic lives in core/forget.py (pure math, no I/O).
  - Pinned memories (created via pin()) are permanently immune to decay
    cleanup and dream pruning. The pinned flag lives in the memories table.

Dream pass overview:
  1. Boost  — increment access_count on memories matching salience heuristics
              (keyword signals, high prior access, recency) so they survive decay.
  2. Merge  — cosine-similarity search per memory; near-duplicates above
              threshold are collapsed: keep the higher access_count copy,
              delete the redundant one to stay in sync.
              Pinned memories are never chosen as the loser in a merge.
  3. Prune  — standard cleanup() pass; runs after boost so newly protected
              memories aren't caught in the sweep.
              Pinned memories are skipped entirely.

Storage layout (single .db file):
  memories        — canonical record: id, user_id, memory, metadata
  memories_fts    — FTS5 virtual table for lexical search (BM25)
  memories_vec    — vec0 virtual table for KNN cosine search

Recall strategy — Reciprocal Rank Fusion (RRF), tiered quick/wide, with
recency-among-relevant reranking:

  score = 1/(k + rank_knn) + 1/(k + rank_fts)
  k=60 (standard RRF constant — dampens outlier ranks)

  KNN catches semantic similarity ("I love cats" <-> "I adore cats")
  FTS5 catches exact token matches ("Max", "birthday", proper nouns)
  RRF fuses both without weighting either arbitrarily.

  Stage 1 — tiered candidate fetch:
    Search runs a narrow "quick" pass first (QUICK_KNN_LIMIT/QUICK_FTS_LIMIT
    candidates). If that pass already fills `limit` results whose weakest
    final score clears MEMORY_RECALL_SCORE_THRESHOLD, it is used as-is —
    most turns stop here. Otherwise the search widens to the full
    KNN_LIMIT/FTS_LIMIT candidate pool and re-ranks from scratch. The query
    embedding is computed exactly once regardless of which path runs; only
    the (cheap) SQL scans are ever repeated.

  Stage 2 — scoring:
    On top of the fused RRF score, recall applies:
      - a small recency bonus (exponential decay, configurable half-life) —
        this is a continuous blend applied to every candidate, separate
        from stage 3's discrete recency-among-relevant reorder below.
      - a small access-count bonus (capped, normalized)
      - a small pinned bonus (MEMORY_RANK_PINNED_WEIGHT) — a mild
        tiebreaker only. There is no separate guarantee stage anymore:
        pinned candidates compete purely on this blended score like
        everything else.

  Stage 3 — recency-among-relevant rerank (MEMORY_RECENCY_RERANK_ENABLED):
    Candidates whose score clears MEMORY_RECENCY_RERANK_THRESHOLD are
    considered "relevant enough" and are reordered by created_at
    descending among themselves (most recent first), ahead of everything
    that didn't clear the bar. This is a genuine reorder — not another
    additive weight — so among several similarly-relevant memories, the
    newest one surfaces first rather than whichever happened to score
    marginally higher on RRF/access/pinned terms.

  Stage 4 — removed (previously: pinned reserve via
    MEMORY_PINNED_RESERVED_SLOTS). Pinned candidates now compete on the
    same blended score as everything else (RRF + recency + access +
    MEMORY_RANK_PINNED_WEIGHT tiebreaker) — no guaranteed slot. Removed
    because guaranteeing whole pinned daily-summary blocks a spot
    regardless of score let oversized entries blow the LLM context
    window on recall. Pinned entries are now atomic per-fact rows (see
    memory/reflect.py), so a normal score-based ranking is sufficient.

  Dedup-on-recall: before any of the above, candidates are collapsed by
  normalized memory text. If the same text exists as multiple rows
  (e.g. several pinned inserts of the same daily record), only the most
  recently created row survives into the ranked result set. This runs
  independently of write-time dedup and independently of dream() merge,
  so a duplicate that slipped through either of those (most commonly:
  pinned duplicates, which dream() can never delete) still can't occupy
  more than one of the returned slots.

Trivial-input skip:
  AikoMemorize.search() short-circuits to [] for turns that are nothing
  but filler (greetings, acks, the assistant's wake-word alone) BEFORE the
  cache lookup or the embedding call — this is the single choke point all
  callers (CLI, WebUI, voice, think.py) go through, so every input path
  gets the optimization without duplicating the check anywhere else. Any
  message with real content attached (a question, a name, a request)
  always searches normally, regardless of what it starts with.

Custom backend (replaces Qdrant + mem0):
  - _MemoryBackend handles LLM-based fact extraction, GGUF embeddings (HarrierEmbedder),
    and direct sqlite-vec upsert/search/delete/scroll.
  - Extraction prompt is tuned for small models: asks for a JSON array of
    atomic facts, strips <think> blocks for CoT models, skips trivial turns.
  - All schema fields (memory, user_id, created_at, access_count,
    last_accessed_at, pinned) are owned by this module — no hidden schema.
  - Both add() and add_raw() run the same write-time dedup check (cosine
    >= WRITE_DEDUP_THRESHOLD against existing vectors) before inserting.
    Previously add_raw() had no such guard, which let repeated calls
    (e.g. a nightly daily-record pin job re-running for the same day)
    insert unbounded duplicate rows that dream()'s merge pass could never
    clean up once pinned=1 was set.

Async write queue:
  - AikoMemorize.queue_write() lets a caller (think.py's chat/webchat
    turns) enqueue a fire-and-forget memory write without blocking the
    turn on LLM-based fact extraction. This module owns the worker
    thread/queue; the caller only needs to decide *when* it's safe to run
    (idle vs mid-turn), which it expresses via two small callables
    (is_active_turn, idle_since) rather than this module reaching into the
    caller's turn-tracking state directly.

Dependencies:
  pip install sqlite-vec llama-cpp-python tokenizers
"""
import json
import os
from collections import OrderedDict
import queue
import threading
import re
import sqlite3
import struct
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from system import bioclock
from memory.vecstore import initialize_store_db, resolve_user_db_path
from system.userspace import current_user_id, user_state_path
import sqlite_vec
from openai import OpenAI

from memory.forget import ACCESS_COUNT_CAP, compute_weighted_score, should_cleanup, CLEANUP_THRESHOLD
from system.log import get_logger
from memory.vecstore import HarrierEmbedder

log = get_logger(__name__)

# ── boot labels ───────────────────────────────────────────────────────────────

BOOT_LABELS = {
    'mem_sqlite_vec':  'Opening sqlite-vec memory store...',
    'mem_cleanup': 'Running memory cleanup...',
    'mem_ready':   'Memory backend ready',
}

# ── helpers ───────────────────────────────────────────────────────────────────

def _env_bool(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}

# ── constants ─────────────────────────────────────────────────────────────────

EMBED_MODEL = os.getenv("EMBED_MODEL", "ferrisS/harrier-oss-v1-270m-fastembed")
EMBED_DIMS  = int(os.getenv("EMBED_DIMS", "640"))
EMBED_QUERY_INSTRUCT = os.getenv("EMBED_QUERY_INSTRUCT", "Retrieve relevant memories that answer the query").strip()
RRF_K       = 60          # standard RRF constant — dampens outlier ranks
KNN_LIMIT   = 20          # candidates fetched before RRF re-rank (wide pass)
FTS_LIMIT   = 20          # candidates fetched before RRF re-rank (wide pass)
QUICK_KNN_LIMIT = int(os.getenv("QUICK_KNN_LIMIT", "6"))   # narrow first-pass candidate count
QUICK_FTS_LIMIT = int(os.getenv("QUICK_FTS_LIMIT", "6"))   # narrow first-pass candidate count
MEMORY_RECALL_SCORE_THRESHOLD = float(os.getenv("MEMORY_RECALL_SCORE_THRESHOLD", "0.015"))
MEMORY_RANK_RECENCY_WEIGHT = float(os.getenv("MEMORY_RANK_RECENCY_WEIGHT", "0.004"))
MEMORY_RANK_RECENCY_HALF_LIFE_DAYS = float(os.getenv("MEMORY_RANK_RECENCY_HALF_LIFE_DAYS", "30"))
MEMORY_RANK_ACCESS_WEIGHT = float(os.getenv("MEMORY_RANK_ACCESS_WEIGHT", "0.002"))
# Bumped from 0.002 -> 0.01: at the old weight, pinned status barely moved
# ranking relative to RRF terms (~0.016 at rank 1), so pinned facts weren't
# reliably outranking unpinned ones of similar relevance. 0.01 makes pinned
# status a meaningful tiebreaker while staying below a full RRF rank-1 term,
# so a highly relevant unpinned memory can still beat a barely-relevant
# pinned one. The hard guarantee for pinned visibility now lives in
MEMORY_RANK_PINNED_WEIGHT = float(os.getenv("MEMORY_RANK_PINNED_WEIGHT", "0.01"))
SEARCH_CACHE_SIZE = int(os.getenv("MEMORY_SEARCH_CACHE_SIZE", 128))
SEARCH_CACHE_TTL  = float(os.getenv("MEMORY_SEARCH_CACHE_TTL", 20.0))
MEMORY_CONTEXT_FACT_CHARS  = int(os.getenv("MEMORY_CONTEXT_FACT_CHARS", 220))
MEMORY_CONTEXT_TOTAL_CHARS = int(os.getenv("MEMORY_CONTEXT_TOTAL_CHARS", 1200))
LIFECYCLE_BATCH_SIZE = int(os.getenv("MEMORY_LIFECYCLE_BATCH_SIZE", 500))

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

USER_ID = current_user_id  # Backward-compatible callable alias; resolve at call time.


def _default_user_id(user_id: str | None = None) -> str:
    return user_id or current_user_id()

# ── trivial-input skip ────────────────────────────────────────────────────────
# Words that carry no retrievable intent on their own. Built dynamically so
# the assistant's configured name (identity.yaml -> AI_NAME) is also a valid
# stand-alone trivial input — e.g. "Hey Aiko" with nothing else attached.
# This check lives here (not in main.py) because AikoMemorize.search() is
# the single choke point every input path (CLI, WebUI, voice, think.py)
# already goes through — putting it in main.py would mean duplicating the
# check at every call site instead of once.
AI_NAME = os.getenv("AI_NAME", "Aiko").strip().lower()

_FILLER_WORDS = (
    "hi", "hey", "hello", "ok", "okay", "thanks", "thank you",
    "yes", "no", "yeah", "nah", "lol", "sure", "bye",
)

# Social/wellbeing phrases that carry no retrievable intent on their own —
# distinct from _FILLER_WORDS since these are multi-word and never a
# stand-alone ack.
_GREETING_PHRASES = (
    "how are you", "how are you doing", "hows it going", "how's it going",
    "how are things", "how you doing", "whats up", "what's up",
)

_name_alt = re.escape(AI_NAME) if AI_NAME else ""

# Combined trivial vocabulary: filler acks + greeting/wellbeing phrases.
# Sorted longest-first so e.g. "how are you doing" matches before the
# shorter "how are you" prefix inside the alternation.
_TRIVIAL_PHRASES = sorted(_FILLER_WORDS + _GREETING_PHRASES, key=len, reverse=True)
_trivial_alt = "|".join(re.escape(p) for p in _TRIVIAL_PHRASES)
_CLAUSE_SPLIT_RE = re.compile(r"[,.!?]+")


def _is_trivial_input(text: str) -> bool:
    """
    True when every clause of the message (split on , . ! ?) is filler,
    a greeting/wellbeing phrase, or the wake-word alone — i.e. no clause
    carries retrievable intent.

    Replaces the old single-anchor _TRIVIAL_INPUT_RE, which could only
    match one-or-two-token messages and had no concept of multi-word
    social phrases like "how are you doing". Splitting into clauses also
    handles ragged ASR transcripts like "Hi, I. How are you doing." —
    each clause is checked independently rather than requiring the whole
    string to match one rigid pattern.

    Any clause that doesn't fully match the trivial vocabulary (a real
    question, name, or request) makes the whole input non-trivial, so
    "hi aiko, what's the weather" still searches normally.
    """
    clauses = [c.strip().lower() for c in _CLAUSE_SPLIT_RE.split(text or "") if c.strip()]
    if not clauses:
        return True
    for clause in clauses:
        if _name_alt and re.fullmatch(_name_alt, clause, re.IGNORECASE):
            continue
        if not re.fullmatch(_trivial_alt, clause, re.IGNORECASE):
            return False
    return True

# Cosine similarity threshold for near-duplicate detection during dream pass
# and dedup-on-write. 0.95 on write is tight (near-identical only).
# 0.92 on dream merge catches slightly more semantic duplicates.
DREAM_MERGE_THRESHOLD = float(os.getenv("DREAM_MERGE_THRESHOLD", 0.92))
WRITE_DEDUP_THRESHOLD = float(os.getenv("WRITE_DEDUP_THRESHOLD", 0.95))

# access_count boost applied to salient memories during dream pass.
DREAM_BOOST_AMOUNT = int(os.getenv("DREAM_BOOST_AMOUNT", 2))

# Salience keywords — memories containing these are boosted during dream pass.
# Matched on word boundaries (see _SALIENCE_RE) so "works" doesn't match
# "networks"/"fireworks" and "lives" doesn't match "olives".
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

# Minimum conversation size (chars) worth sending to LLM for extraction.
_EXTRACT_MIN_CHARS = int(os.getenv("MEMORY_EXTRACT_MIN_CHARS", 80))
_EXTRACT_MAX_TOKENS = int(os.getenv("MEMORY_EXTRACT_MAX_TOKENS", 128))
_EXTRACT_TIMEOUT = float(os.getenv("MEMORY_EXTRACT_TIMEOUT", 18))

_BROAD_RECALL_RE = re.compile(
    r"\b(what|anything|things|facts|memories?|remember|recall)\b.*\b(about me|about oppa|you remember|past|before)\b"
    r"|\b(remember|recall)\b.*\b(me|oppa)\b",
    re.IGNORECASE,
)

# Language that signals the LLM is guessing rather than stating a known fact.
# Facts containing these signals are dropped before persistence.
# Matched on word/phrase boundaries (see _HEDGE_RE) so e.g. "Oppa believes in
# ghosts" isn't missed and "Oppa said I believe in hard work" isn't wrongly
# dropped just because "believe" overlaps with a substring of "believes".
_HEDGE_SIGNALS = frozenset([
    "might", "probably", "seems", "i think", "perhaps", "maybe",
    "appears", "possibly", "could be", "not sure", "i believe",
    "it sounds like", "it seems like",
])

_HEDGE_RE = re.compile(
    r'\b(?:' + '|'.join(re.escape(h) for h in _HEDGE_SIGNALS) + r')\b',
    re.IGNORECASE,
)

# Extraction prompt — temperature 0.0, explicit only-stated-facts rule.
_EXTRACT_PROMPT = """\
Extract memorable facts about Oppa from this conversation.
Oppa is the user (he/him). You are Aiko, the assistant.

Rules:
- Only include facts Oppa stated explicitly. Never infer or assume.
- Write facts as short, direct statements in third person about Oppa.
- No facts about Aiko's behavior, feelings, or responses.
- No uncertain language: never use might, probably, seems, maybe, perhaps, appears.
- If nothing is worth remembering, return: []

Return ONLY a JSON array of short strings. No markdown. No explanation.

Good examples:
["Oppa's birthday is June 3", "Oppa is building a robot called GRACE", "Oppa joined the Hugging Face Hackathon", "Oppa lost his wallet", "Oppa has a deadline on Friday", "Oppa dislikes mushrooms"]

Bad examples (do not produce these):
["Oppa might like cats", "It seems Oppa is tired", "Aiko should remember this"]

Conversation:
{conversation}"""


def _sanitize_fts_query(query: str) -> Optional[str]:
    """
    Strip characters that break FTS5 query parsing.
    FTS5 treats , " ( ) * ^ : - ' as syntax tokens — remove them all.
    Returns None when nothing usable remains (caller should skip the FTS5
    lookup entirely — a bare '*' is not a valid FTS5 "match everything"
    query and raises `sqlite3.OperationalError: unknown special query:`).
    """
    cleaned = re.sub(r'[^\w\s]', ' ', query or "")
    cleaned = ' '.join(cleaned.split())
    return cleaned or None


def _normalize_memory_text(text: str) -> str:
    """
    Normalize memory text for exact-duplicate comparison at recall time.
    Lowercased, whitespace-collapsed. Intentionally cheap/exact (not
    fuzzy) — recall-time dedup targets true copies (e.g. the same
    daily-record string inserted multiple times via add_raw), not
    semantic near-duplicates. Semantic near-duplicates are dream()'s job.
    """
    return " ".join((text or "").split()).lower()

# ── schema ────────────────────────────────────────────────────────────────────

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
    embedding FLOAT[{dims}]
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


# ── sqlite payload helpers ────────────────────────────────────────────────────

def _sqlite_get_payload(conn: sqlite3.Connection, mem_id: str) -> dict:
    """Fetch the full memories row for a single id. Returns {} if not found."""
    row = conn.execute(
        "SELECT * FROM memories WHERE id = ?", (mem_id,)
    ).fetchone()
    return dict(row) if row else {}


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
        n   = len(raw) // 4
        return list(struct.unpack(f"{n}f", raw))
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
    threshold: Optional[float] = None,
) -> list[sqlite3.Row]:
    """
    KNN cosine search against memories_vec, filtered by user_id.
    When threshold is supplied, only rows with dist <= (1 - threshold) are returned.
    """
    vec_blob = sqlite_vec.serialize_float32(vector)
    if threshold is not None:
        dist_ceil = 1.0 - threshold
        rows = conn.execute(
            """
            SELECT v.id, vec_distance_cosine(v.embedding, ?) AS dist
            FROM memories_vec v
            JOIN memories m ON m.id = v.id
            WHERE m.user_id = ?
              AND vec_distance_cosine(v.embedding, ?) <= ?
            ORDER BY dist ASC
            LIMIT ?
            """,
            (vec_blob, user_id, vec_blob, dist_ceil, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT v.id, vec_distance_cosine(v.embedding, ?) AS dist
            FROM memories_vec v
            JOIN memories m ON m.id = v.id
            WHERE m.user_id = ?
            ORDER BY dist ASC
            LIMIT ?
            """,
            (vec_blob, user_id, limit),
        ).fetchall()
    return rows


# ── memory backend ────────────────────────────────────────────────────────────

class _MemoryBackend:
    """
    sqlite-vec + FTS5 + RRF memory backend.

    Changes from original:
      - Extraction LLM runs at temperature=0.0 for deterministic fact output.
      - _extract_facts() filters hedging language via _HEDGE_RE before
        returning — uncertain facts are never persisted.
      - add() runs a dedup check per fact before insert: if a near-identical
        vector already exists (cosine >= WRITE_DEDUP_THRESHOLD), the fact is
        skipped rather than creating a redundant entry.
      - add_raw() now runs the same dedup check as add() (previously it had
        none, which allowed unbounded duplicate pinned inserts).
      - search() collapses exact-text duplicates before final ranking,
        keeping only the most recently created row per duplicate cluster;
        runs a tiered quick/wide candidate pass; applies a recency-among-
        relevant rerank; and finishes with a pinned-slot reserve. See
        module docstring for the full stage breakdown.
    """

    def __init__(
        self,
        db_path:         str,
        llm_base_url:    str,
        model:           str,
        embed_cache:     Optional[str] = None,
    ) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db_path  = db_path
        self._user_id = current_user_id()
        self._llm_base = llm_base_url.rstrip("/")
        self._model    = model
        self._client   = OpenAI(base_url=self._llm_base, api_key="not-needed")
        self._embedder = HarrierEmbedder()
        self._conn = self._connect()
        self._apply_schema()

    def _connect(self) -> sqlite3.Connection:
        return initialize_store_db(self._db_path, _DDL, user_id=self._user_id, vector=True)

    def _apply_schema(self) -> None:
        # Schema is applied by databank.initialize_store_db().
        pass

    # ── embedding ─────────────────────────────────────────────────────────────

    def _format_query_text(self, text: str) -> str:
        """Apply query-side instruction prefix for instruct embedding models."""
        if not EMBED_QUERY_INSTRUCT:
            return text
        return f"Instruct: {EMBED_QUERY_INSTRUCT}\nQuery: {text}"

    def _embed(self, text: str, *, query: bool = False) -> list[float]:
        """Embed a single string with HarrierEmbedder. Returns a plain float list."""
        if query:
            return self._embedder.embed_query(text).tolist()
        return list(self._embedder.embed([text]))[0].tolist()

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple strings in a single batched GGUF call."""
        return self._embedder.embed_batch(texts).tolist()

    # ── extraction ────────────────────────────────────────────────────────────

    def _should_extract(self, messages: list[dict]) -> bool:
        """Return False for trivial turns below minimum char threshold."""
        total = sum(
            len(m.get("content") or "")
            for m in messages
            if m.get("role") in ("user", "assistant")
            and (m.get("content") or "").strip()
        )
        return total >= _EXTRACT_MIN_CHARS

    def _extract_facts(self, messages: list[dict]) -> list[str]:
        """
        Send conversation to the OpenAI-compatible local LLM and parse the returned JSON fact array.

        Changes from original:
          - temperature=0.0 for deterministic output — reduces confabulation.
          - Post-parse hedge filter: facts containing uncertain language
            (_HEDGE_RE, word-boundary matched) are dropped before returning.
          - Only user/assistant turns with real content are sent.
        """
        if not self._should_extract(messages):
            return []

        clean_messages = [
            m for m in messages
            if m.get("role") in ("user", "assistant")
            and (m.get("content") or "").strip()
        ]

        while clean_messages and clean_messages[0].get("role") != "user":
            clean_messages.pop(0)

        while clean_messages and clean_messages[-1].get("role") == "assistant":
            if any(m.get("role") == "user" for m in clean_messages[:-1]):
                break
            clean_messages.pop()

        if not clean_messages:
            return []

        total = sum(len(m.get("content") or "") for m in clean_messages)
        if total < _EXTRACT_MIN_CHARS:
            return []

        convo = "\n".join(
            f"{m['role'].upper()}: {m['content'].strip()}"
            for m in clean_messages
        )

        prompt = _EXTRACT_PROMPT.format(conversation=convo)

        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                max_tokens=_EXTRACT_MAX_TOKENS,
                temperature=0.0,  # deterministic — reduces hallucinated facts
                timeout=_EXTRACT_TIMEOUT,
                stop=["\n\n", "```"],
            )
            raw = (resp.choices[0].message.content or "").strip()
        except Exception as e:
            log.warning(f"Extraction LLM call failed: {e}")
            return []

        # strip CoT think blocks before JSON parsing
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

        # strip accidental markdown fences
        raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()

        # take only the first top-level JSON array — model sometimes repeats output
        match = re.search(r"\[.*?\]", raw, re.DOTALL)
        if match:
            raw = match.group(0)

        try:
            facts = json.loads(raw)
            if isinstance(facts, list):
                facts = [f.strip() for f in facts if isinstance(f, str) and f.strip()]
            else:
                return []
        except json.JSONDecodeError:
            log.warning(f"Failed to parse extraction JSON: {raw[:200]!r}")
            return []

        # drop facts containing hedging/uncertain language (word-boundary match)
        clean_facts = []
        for fact in facts:
            if _HEDGE_RE.search(fact):
                log.debug(f"Dropped hedging fact: {fact!r}")
                continue
            clean_facts.append(fact)

        return clean_facts

    # ── write ─────────────────────────────────────────────────────────────────

    def add(self, messages: list[dict], user_id: str) -> list[str]:
        """
        Extract facts and persist each as a row in memories + memories_vec.

        Dedup-on-write: before inserting each fact, a KNN search checks for
        a near-identical vector already in the store (cosine >= WRITE_DEDUP_THRESHOLD).
        Duplicates are skipped to prevent redundant entries that compound into
        false confidence during recall.

        Embeddings for all extracted facts are computed in a single batched
        call rather than one-by-one.

        Returns list of new memory IDs. Empty list if nothing extracted.
        """
        facts = self._extract_facts(messages)
        if not facts:
            return []

        now = bioclock.local_now()
        ids = []

        try:
            vectors = self._embed_batch(facts)
        except Exception as e:
            log.warning(f"Batch embedding failed, aborting write: {e}")
            return []

        try:
            for fact, vector in zip(facts, vectors):
                mem_id = str(uuid.uuid4())
                # dedup check — skip if near-identical vector already exists
                existing = _sqlite_knn_search(
                    self._conn, vector, user_id,
                    limit=1, threshold=WRITE_DEDUP_THRESHOLD,
                )
                if existing:
                    log.debug(f"Skipping near-duplicate fact: {fact!r}")
                    continue

                # insert canonical record — FTS5 trigger fires automatically
                self._conn.execute(
                    """
                    INSERT INTO memories
                        (id, user_id, memory, created_at, access_count, last_accessed_at, pinned)
                    VALUES (?, ?, ?, ?, 0, 'never', 0)
                    """,
                    (mem_id, user_id, fact, now),
                )

                # insert embedding into vec0 table
                self._conn.execute(
                    "INSERT INTO memories_vec(id, embedding) VALUES (?, ?)",
                    (mem_id, sqlite_vec.serialize_float32(vector)),
                )

                ids.append(mem_id)

            self._conn.commit()
        except Exception as e:
            log.warning(f"Failed to upsert fact batch: {e}")
            self._conn.rollback()
            return []

        return ids

    def add_raw(self, memory: str, user_id: str, *, pinned: bool = False) -> str | None:
        """
        Persist one already-curated memory string without LLM extraction.

        Now runs the same write-time dedup check as add(): if a near-identical
        vector already exists (cosine >= WRITE_DEDUP_THRESHOLD), the insert is
        skipped and None is returned. This closes the gap that previously let
        repeated calls (e.g. a daily-record pin job re-running for the same
        day) accumulate unbounded duplicate rows — especially dangerous for
        pinned=True inserts, since dream()'s merge pass can never delete a
        pinned memory even as a duplicate loser.
        """
        text = (memory or "").strip()
        if not text:
            return None
        try:
            vector = self._embed(text)

            existing = _sqlite_knn_search(
                self._conn, vector, user_id,
                limit=1, threshold=WRITE_DEDUP_THRESHOLD,
            )
            if existing:
                log.debug(f"Skipping near-duplicate raw memory: {text[:80]!r}")
                return None

            mem_id = str(uuid.uuid4())
            now = datetime.now(timezone.utc).isoformat()
            self._conn.execute(
                """
                INSERT INTO memories
                    (id, user_id, memory, created_at, access_count, last_accessed_at, pinned)
                VALUES (?, ?, ?, ?, 0, 'never', ?)
                """,
                (mem_id, user_id, text, now, 1 if pinned else 0),
            )
            self._conn.execute(
                "INSERT INTO memories_vec(id, embedding) VALUES (?, ?)",
                (mem_id, sqlite_vec.serialize_float32(vector)),
            )
            self._conn.commit()
            return mem_id
        except Exception as e:
            log.warning("Failed to insert raw memory: %s", e)
            self._conn.rollback()
            return None

    # ── read ──────────────────────────────────────────────────────────────────

    def _fts_pass(self, fts_query: Optional[str], user_id: str, fts_limit: int) -> list[sqlite3.Row]:
        """Run one FTS5 BM25 pass. Returns [] if fts_query is None (nothing usable to match)."""
        if fts_query is None:
            return []
        return self._conn.execute(
            """
            SELECT f.id
            FROM memories_fts f
            JOIN memories m ON m.id = f.id
            WHERE memories_fts MATCH ?
            AND m.user_id = ?
            ORDER BY rank
            LIMIT ?
            """,
            (fts_query, user_id, fts_limit),
        ).fetchall()

    def _rank_and_score(
        self,
        rank_knn: dict,
        rank_fts: dict,
    ) -> tuple[list[str], dict, dict]:
        """
        Dedup + score one candidate pool (from either the quick or wide pass).

        1. Fetch full rows for the union of KNN/FTS candidate ids.
        2. Collapse exact-text duplicates, keeping the most recently created
           row per duplicate cluster.
        3. Score every surviving id: RRF fusion + recency/access/pinned bonuses.

        Returns (ids sorted best-first by score, {id: score}, {id: row}).
        Recency-among-relevant reranking and pinned-reserve are applied
        afterward by the caller (search()), not here — this method only
        produces the base score-ordered list.
        """
        all_ids = set(rank_knn) | set(rank_fts)
        if not all_ids:
            return [], {}, {}

        placeholders = ",".join("?" * len(all_ids))
        rows = self._conn.execute(
            f"SELECT * FROM memories WHERE id IN ({placeholders})",
            list(all_ids),
        ).fetchall()
        row_by_id = {row["id"]: row for row in rows}

        # ── recall-time dedup: collapse exact-text duplicates, keep newest ──
        # Handles the case dream() structurally can't: pinned duplicate rows
        # (dream's merge never deletes a pinned memory, even as the loser),
        # and any duplicate created between dream() runs.
        best_by_text: dict[str, str] = {}
        for mid in all_ids:
            row = row_by_id.get(mid)
            if row is None:
                continue
            norm = _normalize_memory_text(row["memory"])
            current_best = best_by_text.get(norm)
            if current_best is None:
                best_by_text[norm] = mid
                continue
            if row["created_at"] > row_by_id[current_best]["created_at"]:
                best_by_text[norm] = mid
        deduped_ids = set(best_by_text.values())

        def _recency_score(created_at: str) -> float:
            try:
                created = datetime.fromisoformat((created_at or "").replace("Z", "+00:00"))
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                age_days = max(0.0, (datetime.now(timezone.utc) - created).total_seconds() / 86400)
                return 0.5 ** (age_days / max(MEMORY_RANK_RECENCY_HALF_LIFE_DAYS, 1e-6))
            except Exception:
                return 0.0

        def final_score(mem_id: str) -> float:
            knn = rank_knn.get(mem_id, 0)
            fts = rank_fts.get(mem_id, 0)
            score = 0.0
            if knn:
                score += 1.0 / (RRF_K + knn)
            if fts:
                score += 1.0 / (RRF_K + fts)

            row = row_by_id.get(mem_id)
            if row is not None:
                score += MEMORY_RANK_RECENCY_WEIGHT * _recency_score(row["created_at"])
                score += MEMORY_RANK_ACCESS_WEIGHT * min(int(row["access_count"] or 0), ACCESS_COUNT_CAP) / max(ACCESS_COUNT_CAP, 1)
                if int(row["pinned"] or 0):
                    score += MEMORY_RANK_PINNED_WEIGHT
            return score

        scored_ids = sorted(deduped_ids, key=final_score, reverse=True)
        scores = {mid: final_score(mid) for mid in scored_ids}
        return scored_ids, scores, row_by_id

    def _apply_recency_rerank(
        self,
        scored_ids: list[str],
        scores: dict,
        row_by_id: dict,
    ) -> list[str]:
        """
        Stage 3 — recency-among-relevant reorder (see module docstring).

        Candidates whose score clears MEMORY_RECENCY_RERANK_THRESHOLD are
        pulled to the front, sorted by created_at descending among
        themselves (most recent first). Candidates below the threshold keep
        their original score-descending relative order and follow behind.

        This is a genuine reorder, not another additive weight: two
        similarly-relevant memories can swap places here even if their RRF
        scores differ, as long as both clear the bar.
        """
        if not MEMORY_RECENCY_RERANK_ENABLED or not scored_ids:
            return scored_ids

        relevant = [mid for mid in scored_ids if scores.get(mid, 0.0) >= MEMORY_RECENCY_RERANK_THRESHOLD]
        if not relevant:
            return scored_ids

        relevant_sorted = sorted(
            relevant,
            key=lambda mid: row_by_id[mid]["created_at"] if mid in row_by_id else "",
            reverse=True,
        )
        relevant_set = set(relevant)
        rest = [mid for mid in scored_ids if mid not in relevant_set]
        return relevant_sorted + rest

    def search(self, query: str, user_id: str, limit: int = 5) -> list[dict]:
        """
        KNN + FTS5 -> RRF fusion search, with a tiered quick/wide candidate
        pass, recency-among-relevant reranking, and a pinned-slot reserve.
        See module docstring for the full stage-by-stage description.

        1. Embed the query once (_embed) — this is the dominant cost
           regardless of which pass runs below, so it is never repeated.
        2. Quick pass: pull QUICK_KNN_LIMIT / QUICK_FTS_LIMIT candidates,
           dedup + score them. If that already fills `limit` results and the
           weakest of them clears MEMORY_RECALL_SCORE_THRESHOLD, use it as-is
           — most turns stop here and never pay for the wider SQL scan.
        3. Otherwise widen to KNN_LIMIT / FTS_LIMIT and re-rank the larger
           pool from scratch (rank positions shift when the pool grows, so
           this is a fresh scoring pass, not a merge with the quick pass).
        4. Reorder the resulting candidates by recency-among-relevant.
        5. Apply the pinned-slot reserve as a final guarantee.
        6. Truncate to `limit` and return as payload dicts.
        """
        vector = self._embed(query, query=True)
        fts_query = _sanitize_fts_query(query)

        # ── quick pass ──────────────────────────────────────────────────────
        quick_knn_rows = _sqlite_knn_search(self._conn, vector, user_id, QUICK_KNN_LIMIT)
        rank_knn_q = {row["id"]: i + 1 for i, row in enumerate(quick_knn_rows)}
        quick_fts_rows = self._fts_pass(fts_query, user_id, QUICK_FTS_LIMIT)
        rank_fts_q = {row["id"]: i + 1 for i, row in enumerate(quick_fts_rows)}

        scored_ids, scores, row_by_id = self._rank_and_score(rank_knn_q, rank_fts_q)

        confident = (
            len(scored_ids) >= limit
            and scores.get(scored_ids[limit - 1], 0.0) >= MEMORY_RECALL_SCORE_THRESHOLD
        )

        # ── widen only if the quick pass was under-filled or under-confident ──
        if not confident:
            wide_knn_rows = _sqlite_knn_search(self._conn, vector, user_id, KNN_LIMIT)
            rank_knn_w = {row["id"]: i + 1 for i, row in enumerate(wide_knn_rows)}
            wide_fts_rows = self._fts_pass(fts_query, user_id, FTS_LIMIT)
            rank_fts_w = {row["id"]: i + 1 for i, row in enumerate(wide_fts_rows)}
            scored_ids, scores, row_by_id = self._rank_and_score(rank_knn_w, rank_fts_w)

        ordered_ids = self._apply_recency_rerank(scored_ids, scores, row_by_id)
        top_ids = ordered_ids[:limit]

        results = []
        for mid in top_ids:
            if mid not in row_by_id:
                continue
            d = dict(row_by_id[mid])
            d["_recall_score"] = scores.get(mid, 0.0)
            results.append(d)
        return results

    def iter_all(self, user_id: str, batch_size: int = LIFECYCLE_BATCH_SIZE):
        """Yield memory records for a user in rowid order without one giant list."""
        last_rowid = 0
        while True:
            rows = self._conn.execute(
                """
                SELECT rowid, id, memory, created_at
                FROM memories
                WHERE user_id = ? AND rowid > ?
                ORDER BY rowid ASC
                LIMIT ?
                """,
                (user_id, last_rowid, batch_size),
            ).fetchall()
            if not rows:
                break
            for row in rows:
                last_rowid = row["rowid"]
                yield {"id": row["id"], "memory": row["memory"], "created_at": row["created_at"]}

    def get_all(self, user_id: str) -> list[dict]:
        """
        Return memory records for a user.

        Projected to (id, memory, created_at) only — the three fields
        actually read off get_all() results anywhere in this codebase.
        access_count/last_accessed_at/pinned are intentionally NOT included
        here; callers that need fresh values for those fetch them via
        _sqlite_batch_get_payloads()/_sqlite_is_pinned() instead, since
        get_all() snapshots can be stale by the time those checks run.
        """
        return list(self.iter_all(user_id=user_id))

    def get_since(self, since: datetime, user_id: str | None = None) -> list[dict]:
        """Return memories created on or after `since`, newest first."""
        user_id = _default_user_id(user_id)
        rows = self._conn.execute(
            """
            SELECT * FROM memories
            WHERE user_id = ? AND created_at >= ?
            ORDER BY created_at DESC
            """,
            (user_id, since.isoformat()),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_between(self, start: datetime, end: datetime, user_id: str | None = None) -> list[dict]:
        """Return memories created in [start, end), oldest first."""
        user_id = _default_user_id(user_id)
        rows = self._conn.execute(
            """
            SELECT * FROM memories
            WHERE user_id = ? AND created_at >= ? AND created_at < ?
            ORDER BY created_at ASC
            """,
            (user_id, start.isoformat(), end.isoformat()),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── delete ────────────────────────────────────────────────────────────────

    def delete(self, memory_id: str) -> None:
        """Delete a memory from all three tables."""
        self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self._conn.execute("DELETE FROM memories_vec WHERE id = ?", (memory_id,))
        self._conn.commit()

    def delete_all(self, user_id: str) -> None:
        """Delete every memory for a user from all three tables."""
        ids = [
            r["id"] for r in self._conn.execute(
                "SELECT id FROM memories WHERE user_id = ?", (user_id,)
            ).fetchall()
        ]
        if not ids:
            return
        placeholders = ",".join("?" * len(ids))
        self._conn.execute(
            f"DELETE FROM memories WHERE id IN ({placeholders})", ids
        )
        self._conn.execute(
            f"DELETE FROM memories_vec WHERE id IN ({placeholders})", ids
        )
        self._conn.commit()


# ── memorize ──────────────────────────────────────────────────────────────────

class AikoMemorize:
    """
    Persistent memory with Ebbinghaus decay lifecycle and nightly dream() pass.

    Boot sequence (called by wakeup.py in order):
        memorize = AikoMemorize()
        memorize.cleanup()

    Access tracking:
        Every search() call updates the memories table (access_count,
        last_accessed_at) so the decay formula has fresh data.

    Pinned memories:
        Created via pin() — the pinned=1 column flag makes them
        immune to cleanup(), dream prune, and dream merge (as the loser),
        and eligible for the pinned-slot reserve at recall time (see
        _MemoryBackend.search() stage 4).
        Recall-time dedup (in _MemoryBackend.search) still collapses
        multiple pinned rows with identical text down to the most recent
        one, since dream() structurally cannot do this for pinned rows.

    Async write queue:
        queue_write() lets a caller enqueue a fire-and-forget memory write
        (LLM-based fact extraction + persist) that runs on a dedicated
        background thread, without blocking the caller's turn. The caller
        expresses when it's safe to run via two callables (is_active_turn,
        idle_since) rather than this class inspecting the caller's state
        directly — see queue_write() below.

    Dream pass (call nightly at 00:00):
        1. Boost salient memories' access_count so they survive decay.
        2. Merge near-duplicate vectors — keeps higher-access copy.
        3. Prune decayed memories via cleanup().
    """

    def __init__(self, silent: bool = False) -> None:
        self._user_id_override = None
        self._silent = silent
        self._search_cache: OrderedDict[tuple[str, str, int], tuple[float, list[dict]]] = OrderedDict()
        self._search_cache_lock = threading.RLock()
        self._llm_base_url = os.getenv("LLM_BASE_URL", "http://localhost:8080/v1")
        self._model = os.getenv("EXTRACT_MODEL") or os.getenv("LLM_MODEL", "ministral")
        self._embed_cache = os.getenv("EMBED_CACHE_PATH") or os.getenv("FASTEMBED_CACHE_PATH")

        # Use a .pending path for pre-auth boot so user-space dirs are never
        # created before a real user logs in via the web UI.
        uid = current_user_id()
        if uid == "guest":
            db_path = ":memory:"   # pure in-RAM sqlite — zero disk footprint pre-login
        else:
            db_path = os.getenv("SQLITE_MEMORY_PATH") or str(resolve_user_db_path("memory/memory.db", user_id=uid))
        if not silent:
            log.info("Opening sqlite-vec memory store for %s ...", uid)
        self._mem = _MemoryBackend(
            db_path=db_path,
            llm_base_url=self._llm_base_url,
            model=self._model,
            embed_cache=self._embed_cache,
        )
        self._conn = self._mem._conn
        self._write_queue: "queue.Queue[tuple]" = queue.Queue()
        self._write_worker = threading.Thread(target=self._write_loop, daemon=True)
        self._write_worker.start()
        self._last_cache_clear_time: float = 0.0
        if not silent:
            log.info("Ready.")

    def _open(self, uid: str | None = None) -> None:
        """Open (or reopen) the sqlite-vec store for a given user_id."""
        uid = uid or self._user_id_override or current_user_id()
        if uid == "guest":
            db_path = ":memory:"
        else:
            db_path = os.getenv("SQLITE_MEMORY_PATH") or str(resolve_user_db_path("memory/memory.db", user_id=uid))
        if not self._silent:
            log.info("Opening sqlite-vec memory store for %s ...", uid)
        self._mem = _MemoryBackend(
            db_path=db_path,
            llm_base_url=self._llm_base_url,
            model=self._model,
            embed_cache=self._embed_cache,
        )
        self._conn = self._mem._conn
        if not self._silent:
            log.info("Memory store ready for %s.", uid)

    def switch_user(self, user_id: str) -> None:
        """Switch to a different user's memory store. Re-opens DB."""
        self._user_id_override = user_id
        if self._conn:
            try:
                self._conn.execute("PRAGMA optimize")
                self._conn.commit()
                self._conn.close()
            except Exception:
                pass
        self._open(user_id)

    def get_user_id(self) -> str:
        """Return the user_id this instance is currently opened for."""
        return self._user_id_override or self._mem._user_id

    # ── write ─────────────────────────────────────────────────────────────────

    def add(self, messages: list[dict], user_id: str | None = None) -> bool:
        """
        Store a conversation turn into long-term memory.
        Returns True on success, False on failure.
        """
        try:
            user_id = _default_user_id(user_id)
            t       = time.perf_counter()
            ids     = self._mem.add(messages, user_id=user_id)
            elapsed = time.perf_counter() - t
            if ids:
                self._maybe_clear_search_cache()
                log.info(f"Saved {len(ids)} memories in {elapsed:.2f}s")
            else:
                log.debug(f"No facts extracted ({elapsed:.2f}s) — nothing saved.")
            return True
        except Exception as e:
            log.error(f"Save failed: {e}")
            return False

    def pin(self, messages: list[dict], user_id: str | None = None) -> bool:
        """
        Store messages and immediately mark all resulting memories as pinned.
        Pinned memories are immune to cleanup, dream pruning, and merge losses.
        Returns True on success, False on any failure.
        """
        try:
            user_id = _default_user_id(user_id)
            ids = self._mem.add(messages, user_id=user_id)

            if not ids:
                query = "\n".join(
                    (m.get("content") or "").strip()
                    for m in messages
                    if (m.get("content") or "").strip()
                )
                ids = [
                    str(m.get("id"))
                    for m in self.search(query, user_id=user_id, limit=3)
                    if m.get("id")
                ]

            if not ids:
                log.warning("pin(): add succeeded but no memory IDs were found to pin.")
                return False

            for mem_id in ids:
                _sqlite_set_payload(self._conn, mem_id, {"pinned": 1})

            self._clear_search_cache()
            log.info(f"Pinned {len(ids)} memories: {ids}")
            return True
        except Exception as e:
            log.error(f"Pin failed: {e}")
            return False

    # ── async write queue ────────────────────────────────────────────────────

    def queue_write(
        self,
        user_input: str,
        response_text: str,
        *,
        is_active_turn=None,
        idle_since=None,
    ) -> None:
        """Queue an async memory write for a conversation turn.

        Runs on this instance's dedicated write-worker thread — the caller's
        turn is never blocked on LLM-based fact extraction. `is_active_turn`
        (callable[[], bool]) and `idle_since` (callable[[], float], a
        time.time()-style timestamp of the caller's last chat activity) let
        the write wait for an idle window before using the shared LLM,
        without this module needing to know how the caller tracks turn
        state. If either is omitted, the write runs as soon as it's
        dequeued with no idle wait.
        """
        user_id = self.get_user_id()  # resolved here, on the caller's thread — not in _write_loop
        self._write_queue.put((user_input, response_text, user_id, is_active_turn, idle_since))

    def _write_loop(self) -> None:
        while True:
            user_input, response_text, user_id, is_active_turn, idle_since = self._write_queue.get()
            try:
                self._wait_for_write_window(is_active_turn, idle_since)
                self.add([
                    {"role": "user", "content": user_input[:500]},
                    {"role": "assistant", "content": response_text[:800]},
                ], user_id=user_id)
            except Exception as e:
                log.error(f"Async memory write failed: {e}")
            finally:
                self._write_queue.task_done()

    def _wait_for_write_window(self, is_active_turn, idle_since) -> None:
        """Wait until the caller reports idle before running an extraction
        write on the shared LLM. No-ops immediately if the caller didn't
        supply idle-tracking callables."""
        if is_active_turn is None or idle_since is None:
            return
        deadline = time.monotonic() + max(0.0, MEMORY_WRITE_MAX_WAIT)
        while True:
            idle_for = time.time() - idle_since()
            if not is_active_turn() and idle_for >= MEMORY_WRITE_IDLE_GRACE:
                return
            if (
                MEMORY_WRITE_MAX_WAIT > 0
                and time.monotonic() >= deadline
                and not is_active_turn()
            ):
                return
            sleep_for = min(0.5, max(0.05, MEMORY_WRITE_IDLE_GRACE - idle_for))
            time.sleep(sleep_for)

    def wait_for_writes(self, timeout: float | None = None) -> bool:
        """Block until all queued async writes complete, or `timeout`
        elapses. Returns True if the queue drained, False on timeout."""
        if timeout is None:
            self._write_queue.join()
            return True
        deadline = time.monotonic() + max(0.0, timeout)
        with self._write_queue.all_tasks_done:
            while self._write_queue.unfinished_tasks:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._write_queue.all_tasks_done.wait(remaining)
        return True

    # ── read ──────────────────────────────────────────────────────────────────

    def search(self, query: str, user_id: str | None = None, limit: int = 5) -> list[dict]:
        """
        Retrieve top-k memories relevant to the current query.
        Side-effect: increments access_count and updates last_accessed_at
        for all returned memories in a single batched UPDATE.

        Trivial-input skip: turns that are nothing but filler (a greeting,
        an ack, the assistant's wake-word alone) return [] immediately,
        before the cache lookup or the embedding call. This is the single
        choke point every caller (CLI, WebUI, voice, think.py) goes
        through, so the skip applies everywhere without duplication. Any
        message with real content attached always searches normally.
        """
        user_id = _default_user_id(user_id)
        if _is_trivial_input(query or ""):
            log.debug(f"Skipping search for trivial input: {query!r}")
            return []

        if _BROAD_RECALL_RE.search(query or ""):
            results = self._recent_or_important_memories(user_id=user_id, limit=limit)
            self._touch_memories(results)
            return results

        cache_key = (user_id, " ".join((query or "").lower().split()), int(limit))
        now_s = time.monotonic()

        with self._search_cache_lock:
            cached = self._search_cache.get(cache_key)
            if cached and now_s - cached[0] <= SEARCH_CACHE_TTL:
                self._search_cache.move_to_end(cache_key)
                results = [dict(r) for r in cached[1]]
                log.debug("[memory] cache hit, scores=%s", [r.get("_recall_score") for r in results])  # temp
                self._touch_memories(results)
                return results
            if cached:
                self._search_cache.pop(cache_key, None)

        results = self._mem.search(query, user_id=user_id, limit=limit)
        self._touch_memories(results)

        with self._search_cache_lock:
            self._search_cache[cache_key] = (now_s, [dict(r) for r in results])
            while len(self._search_cache) > SEARCH_CACHE_SIZE:
                self._search_cache.popitem(last=False)

        return results

    def _recent_or_important_memories(self, user_id: str, limit: int) -> list[dict]:
        """
        Return useful memories for broad recall prompts.

        Deduplicated by normalized text (keeping the most recently created
        row per duplicate cluster) before the LIMIT is applied, so pinned
        duplicate rows (e.g. several identical daily-record pins) can't
        eat multiple slots of the broad-recall result set.

        Note: this is a separate code path from _MemoryBackend.search() and
        was already pinned-first (ORDER BY pinned DESC, ...) before the
        pinned-reserve / recency-rerank stages were added there — it is
        untouched by those changes.
        """
        # Fetch a wider candidate window than `limit` so dedup doesn't
        # leave fewer than `limit` results when duplicates are present.
        fetch_n = max(int(limit) * 4, int(limit) + 10)
        rows = self._conn.execute(
            """
            SELECT *
            FROM memories
            WHERE user_id = ?
            ORDER BY pinned DESC, created_at DESC, access_count DESC
            LIMIT ?
            """,
            (user_id, fetch_n),
        ).fetchall()

        best_by_text: dict[str, sqlite3.Row] = {}
        order: list[str] = []
        for row in rows:
            norm = _normalize_memory_text(row["memory"])
            existing = best_by_text.get(norm)
            if existing is None:
                best_by_text[norm] = row
                order.append(norm)
            elif row["created_at"] > existing["created_at"]:
                best_by_text[norm] = row

        deduped = [best_by_text[norm] for norm in order][:int(limit)]
        out = []
        for r in deduped:
            d = dict(r)
            d["_recall_score"] = 1.0  # broad recall is explicit — never filtered
            out.append(d)
        return out

    def _touch_memories(self, results: list[dict]) -> None:
        """Update decay access metadata for a search result set."""
        if not results:
            return
        now = datetime.now(timezone.utc).isoformat()
        mem_ids = [str(r.get("id", "")) for r in results if r.get("id")]
        if not mem_ids:
            return
        try:
            placeholders = ",".join("?" * len(mem_ids))
            self._conn.execute(
                f"""
                UPDATE memories
                SET access_count = MIN(access_count + 1, 255),
                    last_accessed_at = ?
                WHERE id IN ({placeholders})
                """,
                [now] + mem_ids,
            )
            self._conn.commit()
        except Exception as e:
            log.warning(f"Access tracking failed for {mem_ids}: {e}")

    MIN_CLEAR_INTERVAL: float = 0.5  # seconds — debounce window for cache invalidation

    def _clear_search_cache(self) -> None:
        with self._search_cache_lock:
            self._search_cache.clear()

    def _maybe_clear_search_cache(self) -> None:
        """Time-debounced cache clearing — invalidate on write, but only if
        at least MIN_CLEAR_INTERVAL has elapsed since the last clear.

        Normal-paced conversation (one write per turn, seconds between them)
        always sees fresh data.  Rapid writes within the same debounce window
        (bulk import, batch writes) keep the cache warm instead of cold-starting
        on every single write — the only acceptable staleness window.
        """
        now = time.monotonic()
        if now - self._last_cache_clear_time >= self.MIN_CLEAR_INTERVAL:
            self._clear_search_cache()
            self._last_cache_clear_time = now

    def format_for_context(self, memories: list[dict]) -> Optional[str]:
        """
        Format retrieved memories into a compact string for injection
        into the conversation context. Returns None if nothing to inject.
        """
        if not memories:
            return None

        now   = bioclock.local_now()
        lines = [
            "<memory_context>",
            "Background facts about Oppa. Use silently. Never quote or reference this block directly.",
            "",
        ]
        for m in memories:
            text       = m.get("memory") or m.get("text")
            if not text:
                continue
            if len(text) > MEMORY_CONTEXT_FACT_CHARS:
                text = text[:MEMORY_CONTEXT_FACT_CHARS].rstrip() + "..."
            created_at = m.get("created_at")
            if created_at:
                try:
                    ts    = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    delta = now - ts
                    days  = delta.days
                    if days == 0:
                        age = "today"
                    elif days == 1:
                        age = "yesterday"
                    else:
                        age = f"{days} days ago"
                    lines.append(f"  - [{age}] {text}")
                except Exception:
                    lines.append(f"  - {text}")
            else:
                lines.append(f"  - {text}")

        lines.append("</memory_context>")
        block = "\n".join(lines)
        if len(block) > MEMORY_CONTEXT_TOTAL_CHARS:
            block = block[:MEMORY_CONTEXT_TOTAL_CHARS].rstrip() + "\n</memory_context>"
        return block

    # ── dream pass ────────────────────────────────────────────────────────────

    def dream(
        self,
        user_id:   str | None = None,
        dry_run:   bool  = False,
        threshold: float = DREAM_MERGE_THRESHOLD,
    ) -> dict:
        """
        Nightly memory consolidation pass.

        Stages (in order):
          1. Boost  — salient memories get +DREAM_BOOST_AMOUNT access_count.
          2. Merge  — near-duplicate pairs (cosine >= threshold) are collapsed.
          3. Prune  — standard decay cleanup runs last.

        all_mems is fetched once and passed through to cleanup() so the
        prune stage doesn't re-scan the table from scratch.

        Returns dict: {boosted, merged, pruned, duration_s}
        """
        user_id = _default_user_id(user_id)
        t_start = time.perf_counter()
        log.info(f"{'(dry-run) ' if dry_run else ''}Starting consolidation pass...")

        mem_ids: list[str] = []
        boosted = 0

        for batch in self._iter_memory_batches(user_id):
            batch_ids = [str(m.get("id", "")) for m in batch if m.get("id")]
            if not batch_ids:
                continue
            mem_ids.extend(batch_ids)
            payload_map = self._batch_get_payloads(batch_ids)
            pinned_ids = _sqlite_pinned_ids(self._conn, batch_ids)
            boosted += self._dream_boost(batch, payload_map, pinned_ids=pinned_ids, dry_run=dry_run)

        if not mem_ids:
            log.info("No memories found — nothing to do.")
            return {"boosted": 0, "merged": 0, "pruned": 0, "duration_s": 0.0}

        pinned_ids = _sqlite_pinned_ids(self._conn, mem_ids)
        merged = self._dream_merge(mem_ids, user_id=user_id, threshold=threshold, pinned_ids=pinned_ids, dry_run=dry_run)
        prune_result = self.cleanup(user_id=user_id, dry_run=dry_run)
        pruned = prune_result.get("deleted", 0)

        duration = round(time.perf_counter() - t_start, 2)
        log.info(
            f"{'(dry-run) ' if dry_run else ''}"
            f"Done — boosted={boosted}, merged={merged}, pruned={pruned}, "
            f"duration={duration}s"
        )
        return {"boosted": boosted, "merged": merged, "pruned": pruned, "duration_s": duration}

    def _dream_boost(
        self,
        all_mems:    list[dict],
        payload_map: dict,
        pinned_ids:  set[str] | None = None,
        dry_run:     bool = False,
    ) -> int:
        """
        Increment access_count on memories matching salience heuristics.
        Pinned memories pass through unchanged.
        Returns count of memories boosted.
        """
        now     = datetime.now(timezone.utc)
        boost_ids: list[str] = []
        pinned_ids = pinned_ids or set()

        for m in all_mems:
            mem_id = str(m.get("id", ""))
            if not mem_id:
                continue
            if mem_id in pinned_ids:
                continue

            text     = m.get("memory") or ""
            ac, _la  = payload_map.get(mem_id, (0, "never"))

            is_recent  = False
            created_at = m.get("created_at", "")
            if created_at:
                try:
                    ts        = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    is_recent = (now - ts).days <= 7
                except Exception:
                    pass

            is_salient = (
                bool(_SALIENCE_RE.search(text))
                or ac >= 3
                or is_recent
            )

            if not is_salient:
                continue

            boost_ids.append(mem_id)

        if boost_ids and not dry_run:
            try:
                placeholders = ",".join("?" * len(boost_ids))
                self._conn.execute(
                    f"""
                    UPDATE memories
                    SET access_count = MIN(access_count + ?, 255)
                    WHERE id IN ({placeholders})
                    """,
                    [DREAM_BOOST_AMOUNT] + boost_ids,
                )
                self._conn.commit()
            except Exception as e:
                log.warning(f"Batch boost failed for {len(boost_ids)} memories: {e}")
                self._conn.rollback()
                return 0

        boosted = len(boost_ids)
        if boosted:
            log.info(f"{'(dry-run) ' if dry_run else ''}Boosted {boosted} memories.")
        return boosted

    def _dream_merge(
        self,
        mem_ids:   list[str],
        user_id:   str,
        threshold: float = DREAM_MERGE_THRESHOLD,
        pinned_ids: set[str] | None = None,
        dry_run:   bool  = False,
    ) -> int:
        """
        Detect and collapse near-duplicate memory vectors.
        Pinned memories are never chosen as the loser.
        Returns count of memories deleted as duplicates.
        """
        deleted_ids: set[str] = set()
        pinned_ids = pinned_ids or set()
        merged = 0

        for mem_id in mem_ids:
            if mem_id in deleted_ids:
                continue
            if mem_id in pinned_ids:
                continue

            vector = _sqlite_get_vector(self._conn, mem_id)
            if not vector:
                continue

            try:
                neighbor_rows = _sqlite_knn_search(
                    self._conn, vector, user_id, limit=4, threshold=threshold
                )
            except Exception as e:
                log.warning(f"Similarity search failed for {mem_id}: {e}")
                continue

            for row in neighbor_rows:
                neighbor_id = row["id"]
                if neighbor_id == mem_id:
                    continue
                if neighbor_id in deleted_ids:
                    continue

                similarity = 1.0 - row["dist"]
                n_merged = self._resolve_duplicate(
                    mem_id, neighbor_id, similarity, pinned_ids=pinned_ids, dry_run=dry_run
                )
                if n_merged:
                    deleted_ids.add(neighbor_id)
                    merged += 1

        if merged:
            log.info(f"{'(dry-run) ' if dry_run else ''}Merged {merged} duplicate memories.")
        return merged

    def _resolve_duplicate(
        self,
        id_a:    str,
        id_b:    str,
        score:   float,
        pinned_ids: set[str] | None = None,
        dry_run: bool = False,
    ) -> bool:
        """
        Compare two near-duplicate memories and delete the weaker one.
        Pinned memories are never deleted. Tie goes to id_a (query origin).
        Returns True if a deletion occurred.
        """
        pinned_ids = pinned_ids or set()
        if id_a in pinned_ids or id_b in pinned_ids:
            log.info(f"Skipping merge: one or both of ({id_a}, {id_b}) is pinned.")
            return False

        payload_map = self._batch_get_payloads([id_a, id_b])
        ac_a, _     = payload_map.get(id_a, (0, "never"))
        ac_b, _     = payload_map.get(id_b, (0, "never"))
        row_map = {
            row["id"]: row["created_at"]
            for row in self._conn.execute(
                "SELECT id, created_at FROM memories WHERE id IN (?, ?)", (id_a, id_b)
            ).fetchall()
        }
        if ac_a == ac_b:
            loser = id_b if row_map.get(id_a, "") >= row_map.get(id_b, "") else id_a
        else:
            loser = id_b if ac_a > ac_b else id_a

        if dry_run:
            log.info(
                f"(dry-run) Would merge: score={score:.3f} "
                f"ac_a={ac_a} ac_b={ac_b} → delete {loser}"
            )
            return True

        try:
            self._mem.delete(memory_id=loser)
            log.info(
                f"Merged duplicate (score={score:.3f}, "
                f"ac_a={ac_a}, ac_b={ac_b}) → deleted {loser}"
            )
            return True
        except Exception as e:
            log.warning(f"Merge delete failed for {loser}: {e}")
            return False

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def cleanup(
        self,
        user_id:   str | None = None,
        threshold: float = CLEANUP_THRESHOLD,
        dry_run:   bool  = False,
        _all_mems: Optional[list[dict]] = None,
        _pinned_ids: Optional[set[str]] = None,
    ) -> dict:
        """
        Prune decayed memories below threshold score.
        Grace period (14 days) protects newly created memories.
        Pinned memories are unconditionally kept.

        _all_mems: internal — when called from dream(), the already-fetched
        memory list is passed through here to avoid a redundant get_all() scan.

        Returns dict: {deleted, kept, failed, candidates (dry_run only)}.
        """
        user_id = _default_user_id(user_id)
        source = [_all_mems] if _all_mems is not None else self._iter_memory_batches(user_id)

        kept = 0
        deleted: list[str] = []
        failed: list[dict] = []
        dry_candidates: list[dict] = []
        saw_any = False

        for batch in source:
            if not batch:
                continue
            saw_any = True
            batch_kept, candidates = self._cleanup_candidates(
                batch,
                _pinned_ids=_pinned_ids,
            )
            kept += batch_kept

            if dry_run:
                dry_candidates.extend(candidates)
                continue

            for c in candidates:
                try:
                    self._mem.delete(memory_id=c["id"])
                    deleted.append(c["id"])
                except Exception as e:
                    failed.append({"id": c["id"], "error": str(e)})

        if not saw_any:
            return {"deleted": 0, "kept": 0, "failed": 0}

        if dry_run:
            dry_candidates.sort(key=lambda x: x["weighted_score"])
            log.info(f"Dry run: {len(dry_candidates)} candidates for deletion, {kept} kept.")
            return {"deleted": 0, "kept": kept, "failed": 0, "candidates": dry_candidates}

        if deleted:
            self._clear_search_cache()
            self.optimize()

        log.info(f"Cleanup: deleted={len(deleted)}, kept={kept}, failed={len(failed)}")
        return {"deleted": len(deleted), "kept": kept, "failed": len(failed)}

    def _iter_memory_batches(self, user_id: str, batch_size: int = LIFECYCLE_BATCH_SIZE):
        """Yield lifecycle scan batches without retaining the full table."""
        batch: list[dict] = []
        for mem in self._mem.iter_all(user_id=user_id, batch_size=batch_size):
            batch.append(mem)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def _cleanup_candidates(
        self,
        all_mems: list[dict],
        _pinned_ids: Optional[set[str]] = None,
    ) -> tuple[int, list[dict]]:
        mem_ids     = [str(m.get("id", "")) for m in all_mems if m.get("id")]
        payload_map = self._batch_get_payloads(mem_ids)
        pinned_ids  = _pinned_ids if _pinned_ids is not None else _sqlite_pinned_ids(self._conn, mem_ids)

        candidates = []
        kept       = 0

        for m in all_mems:
            mem_id     = str(m.get("id", ""))
            ac, la     = payload_map.get(mem_id, (0, "never"))
            created_at = m.get("created_at", "")

            if mem_id in pinned_ids:
                kept += 1
                continue

            if should_cleanup(ac, la, created_at):
                w = compute_weighted_score(ac, la)
                candidates.append({
                    "id":               mem_id,
                    "memory":           m.get("memory", "")[:120],
                    "access_count":     ac,
                    "weighted_score":   round(w, 4),
                    "last_accessed_at": la,
                })
            else:
                kept += 1

        candidates.sort(key=lambda x: x["weighted_score"])
        return kept, candidates

    def optimize(self) -> None:
        """Run SQLite's lightweight planner/index maintenance hook."""
        try:
            self._conn.execute("PRAGMA optimize")
            self._conn.commit()
        except Exception as e:
            log.debug(f"SQLite optimize skipped: {e}")

    # ── debug ─────────────────────────────────────────────────────────────────

    def get_all(self, user_id: str | None = None) -> list[dict]:
        """Return all stored memories for a user."""
        user_id = _default_user_id(user_id)
        return self._mem.get_all(user_id=user_id)

    def add_raw(self, memory: str, user_id: str | None = None, *, pinned: bool = False, metadata: Optional[dict] = None) -> str | None:
        """Persist one already-curated memory string without LLM extraction."""
        # metadata is accepted for call-site clarity; the current schema stores
        # only the curated text plus pinned flag.
        user_id = _default_user_id(user_id)
        mem_id = self._mem.add_raw(memory, user_id=user_id, pinned=pinned)
        if mem_id:
            self._maybe_clear_search_cache()
        return mem_id

    def get_since(self, since: datetime, user_id: str | None = None) -> list[dict]:
        """Return memories created on or after `since`, newest first."""
        user_id = _default_user_id(user_id)
        return self._mem.get_since(since, user_id=user_id)

    def get_between(self, start: datetime, end: datetime, user_id: str | None = None) -> list[dict]:
        """Return memories created in [start, end), oldest first."""
        user_id = _default_user_id(user_id)
        return self._mem.get_between(start, end, user_id=user_id)

    def delete(self, memory_id: str) -> None:
        """Delete one memory from the store and clear search cache."""
        self._mem.delete(memory_id)
        self._clear_search_cache()

    def clear(self, user_id: str | None = None) -> None:
        """Wipe all memories for a user. Use carefully."""
        user_id = _default_user_id(user_id)
        self._mem.delete_all(user_id=user_id)
        self._clear_search_cache()
        log.info(f"Cleared all memories for user '{user_id}'.")

    # ── internal ──────────────────────────────────────────────────────────────

    def _batch_get_payloads(self, mem_ids: list[str]) -> dict:
        """Batch retrieve access_count + last_accessed_at in a single query."""
        return _sqlite_batch_get_payloads(self._conn, mem_ids)

    def _get_vector(self, mem_id: str) -> list[float]:
        """Retrieve the raw embedding vector for a single memory."""
        return _sqlite_get_vector(self._conn, mem_id)

    def embed_text(self, text: str, *, query: bool = False) -> list[float]:
        """Embed one text string with the configured memory embedding model."""
        return self._mem._embed(text, query=query)

    def embed_texts(self, texts: list[str], *, query: bool = False) -> list[list[float]]:
        """Embed multiple strings with the configured memory embedding model."""
        if query:
            return self._mem._embedder.embed_queries(texts).tolist()   # applies instruct prefix
        return self._mem._embed_batch(texts)                           # document side — no prefix

    def _is_pinned(self, mem_id: str) -> bool:
        """Return True if memories.pinned == 1 for this id."""
        return _sqlite_is_pinned(self._conn, mem_id)
