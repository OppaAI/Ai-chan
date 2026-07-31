"""
memory/memorize.py
Aiko's persistent memory — custom backend via sqlite-vec + HarrierEmbedder (GGUF/llama.cpp).
Abstracts all memory calls so think.py stays clean.

Phase A/B memory metadata (formerly memory/memory_meta.py) is merged in here
as native methods: status/supersedes/kind/source/entities columns, write-time
dedup+supersede, rule-based entity tags, and active-only recall. No runtime
monkey-patching.

Memory lifecycle:
  - Every search() call increments access_count and updates last_accessed_at
    in the memories table, enabling Ebbinghaus-style exponential decay scoring.
  - dream() runs nightly (00:00) as a consolidation pass — no new vectors
    are written. It boosts salient memories, merges near-duplicates, then
    prunes decayed entries. Order matters: boost before prune so boosted
    memories aren't immediately swept.
  - cleanup() deletes memories below decay threshold, with grace period
    protection for newly created entries.
  - Decay logic lives in memory/forget.py (pure math, no I/O).
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

Clock convention:
  - created_at is ALWAYS stored as datetime.now(timezone.utc).isoformat() —
    every write path (add(), add_raw(), pin()) agrees on this, and every
    reader (_rank_and_score's recency scoring, _dream_boost's is_recent
    check, get_since()/get_between()'s string range comparisons) depends
    on that consistency. Do not reintroduce a local-time write path here.
  - format_for_context()'s user-facing age labels ("today", "yesterday",
    "N days ago") are the one place local time still matters: they convert
    the UTC created_at into the local day before diffing against
    bioclock.local_now(), so "today"/"yesterday" reflect the person's
    calendar day rather than UTC's.

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
from __future__ import annotations

import json
import os
from collections import OrderedDict
from itertools import combinations
from typing import Any, Iterable
import queue
import threading
import re
import sqlite3
import struct
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from system import bioclock
from memory.vecstore import initialize_store_db, resolve_user_db_path
from system.userspace import current_display_name, current_user_id, user_state_path
import sqlite_vec
from openai import OpenAI

from memory.forget import ACCESS_COUNT_CAP, compute_weighted_score, should_cleanup, CLEANUP_THRESHOLD
from system.log import get_logger
from memory.vecstore import HarrierEmbedder

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


# ── boot labels ───────────────────────────────────────────────────────────────

BOOT_LABELS = {
    'mem_embed':         'Opening sqlite-vec store and loading embedder...',
    'mem_display_name':  'Resolving display name...',
    'mem_cleanup':       'Running memory cleanup...',
    'mem_ready':         'Memory backend ready',
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
# Bumped from 0.002 -> 0.01 so pinned status is a meaningful tiebreaker
# under RRF (~0.016 at rank 1), without beating a clearly better unpinned hit.
MEMORY_RANK_PINNED_WEIGHT = float(os.getenv("MEMORY_RANK_PINNED_WEIGHT", "0.01"))
MEMORY_SEARCH_CACHE_SIZE = int(os.getenv("MEMORY_SEARCH_CACHE_SIZE", 128))
MEMORY_SEARCH_CACHE_TTL  = float(os.getenv("MEMORY_SEARCH_CACHE_TTL", 20.0))
MEMORY_CONTEXT_FACT_CHARS  = int(os.getenv("MEMORY_CONTEXT_FACT_CHARS", 220))
MEMORY_CONTEXT_TOTAL_CHARS = int(os.getenv("MEMORY_CONTEXT_TOTAL_CHARS", 1200))
MEMORY_LIFECYCLE_BATCH_SIZE = int(os.getenv("MEMORY_LIFECYCLE_BATCH_SIZE", 500))

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
        if re.fullmatch(_trivial_alt, clause, re.IGNORECASE):
            continue
        # stray ASR fragments (bare pronouns/fillers with no verb) carry no
        # retrievable intent on their own -- e.g. "I" from "Hi, I. How are..."
        if len(clause.split()) == 1 and len(clause) <= 2:
            continue
        return False
    return True

# Cosine similarity threshold for near-duplicate detection during dream pass
# and dedup-on-write. 0.95 on write is tight (near-identical only).
# 0.88 on dream merge catches slightly more semantic duplicates without being too aggressive.
DREAM_MERGE_THRESHOLD = float(os.getenv("DREAM_MERGE_THRESHOLD", 0.88))
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
Extract memorable facts about {user_name} from this conversation.
{user_name} is the user (he/him). You are Aiko, the assistant.

Rules:
- Only include facts {user_name} stated explicitly. Never infer or assume.
- Write facts as short, direct statements in third person about {user_name}.
- No facts about Aiko's behavior, feelings, or responses.
- No uncertain language: never use might, probably, seems, maybe, perhaps, appears.
- If nothing is worth remembering, return: []

Return ONLY a JSON array of short strings. No markdown. No explanation.

Good examples:
["{user_name}'s birthday is June 3", "{user_name} is building a robot called GRACE", "{user_name} joined the Hugging Face Hackathon", "{user_name} lost his wallet", "{user_name} has a deadline on Friday", "{user_name} dislikes mushrooms"]

Bad examples (do not produce these):
["{user_name} might like cats", "It seems {user_name} is tired", "Aiko should remember this"]

Conversation:
{conversation}"""


def _sanitize_fts_query(query: str) -> str | None:
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


# ── Phase A/B memory metadata ─────────────────────────────────────────────────
# Rule-based entity tags + kind, write-op classification (add/supersede/noop),
# and additive schema columns. Zero extra LLM calls — latency-safe on Jetson.

STATUS_ACTIVE = "active"
STATUS_SUPERSEDED = "superseded"

KIND_FACT = "fact"
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
)

# ── Phase B: entity tagging (formerly memory/entities.py) ────────────────────
# Pure regex / heuristics — no LLM, no NER model. Runs on already-extracted
# fact strings at write time so the hot path stays Jetson-friendly.

# Multi-word Proper Case spans: "Hugging Face", "San Francisco"
_PROPER_SPAN_RE = re.compile(
    r"\b([A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+){0,4})\b"
)
# Short ALLCAPS tokens often used as project codes: GRACE, ROS2, API
_ALLCAPS_RE = re.compile(r"\b([A-Z]{2,12}[0-9]*)\b")
# Quoted names: "Max", 'Aiko'
_QUOTED_RE = re.compile(r"[\"']([^\"']{2,40})[\"']")
# Explicit name patterns: called X, named X, project X
_CALLED_RE = re.compile(
    r"\b(?:called|named|project|robot|dog|cat|company|team)\s+([A-Z][\w.-]{1,40})",
    re.IGNORECASE,
)

_STOP_ENTITIES = frozenset({
    "the", "a", "an", "and", "or", "but", "for", "with", "from", "into",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
    "today", "yesterday", "tomorrow", "user", "assistant", "aiko",
    "he", "she", "they", "his", "her", "their", "this", "that",
})

# Kind heuristics (keyword → kind). First match wins.
_KIND_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("identity", ("name is", "birthday", "lives in", "nationality", "age is", "is from ")),
    ("preference", ("likes", "loves", "hates", "dislikes", "prefers", "favorite", "favourite")),
    ("plan", ("deadline", "due ", "will ", "going to", "plans to", "wants to")),
    ("event", ("hackathon", "interview", "meeting", "appointment", "lost ", "joined")),
)


def _clean_entity(raw: str) -> str | None:
    s = (raw or "").strip(" .,;:!?()[]{}").strip()
    if len(s) < 2 or len(s) > 80:
        return None
    if s.casefold() in _STOP_ENTITIES:
        return None
    # Drop pure numbers
    if s.isdigit():
        return None
    return s


def extract_entities(text: str, *, max_entities: int = 12) -> list[str]:
    """Extract entity-like tokens from a memory fact string.

    Deterministic and cheap. Prefer precision over recall — empty is fine.
    """
    if not (text or "").strip():
        return []

    found: list[str] = []
    seen: set[str] = set()

    def _add(raw: str) -> None:
        ent = _clean_entity(raw)
        if not ent:
            return
        key = ent.casefold()
        if key in seen:
            return
        seen.add(key)
        found.append(ent)

    for m in _QUOTED_RE.finditer(text):
        _add(m.group(1))
    for m in _CALLED_RE.finditer(text):
        _add(m.group(1))
    for m in _PROPER_SPAN_RE.finditer(text):
        span = m.group(1)
        # Sentence-initial single word is usually grammar capitalization, not an entity.
        if m.start() == 0 and " " not in span:
            continue
        _add(span)
    for m in _ALLCAPS_RE.finditer(text):
        _add(m.group(1))

    return found[:max_entities]


def classify_kind(text: str, default: str = "fact") -> str:
    """Heuristic memory kind from fact text. No LLM."""
    low = (text or "").casefold()
    for kind, needles in _KIND_RULES:
        if any(n in low for n in needles):
            return kind
    return default


def entity_overlap_score(query: str, entities: Iterable[str]) -> float:
    """Return 0..1 fraction of entities mentioned in the query (casefold)."""
    ents = [e for e in entities if e]
    if not ents:
        return 0.0
    q = (query or "").casefold()
    hits = sum(1 for e in ents if e.casefold() in q)
    return hits / len(ents)


def normalize_memory_text(text: str) -> str:
    """Normalize for write-op classification (lowercase, collapse whitespace)."""
    return _WS_RE.sub(" ", (text or "").strip()).lower()


def entities_to_json(entities: list[str] | None) -> str:
    """Serialize a deduped entity list into a JSON string column value."""
    if not entities:
        return "[]"
    cleaned: list[str] = []
    seen: set[str] = set()
    for e in entities:
        s = str(e).strip()
        if not s:
            continue
        key = s.casefold()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(s[:80])
        if len(cleaned) >= 16:
            break
    return json.dumps(cleaned, ensure_ascii=False)


def entities_from_json(raw: Any) -> list[str]:
    """Parse an entities column value (JSON string or raw list) back to list."""
    if raw is None or raw == "":
        return []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, json.JSONDecodeError):
        return []
    if not isinstance(data, list):
        return []
    return [str(x).strip() for x in data if str(x).strip()]


def classify_write_op(
    *,
    similarity: float | None,
    new_text: str,
    old_text: str | None,
    dedup_threshold: float,
) -> str:
    """Return 'noop' | 'supersede' | 'add' — rule-only, no LLM."""
    if similarity is None or similarity < dedup_threshold:
        return "add"
    if normalize_memory_text(new_text) == normalize_memory_text(old_text or ""):
        return "noop"
    return "supersede"


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


def _active_sql(active_only: bool) -> str:
    """SQL fragment to restrict a query to active (non-superseded) memories."""
    if not active_only:
        return ""
    return " AND (m.status = 'active' OR m.status IS NULL)"


def backfill_entities(
    conn: sqlite3.Connection,
    *,
    user_id: str | None = None,
    limit: int = 0,
    only_empty: bool = True,
) -> int:
    """Fill entities/kind for existing rows using rule-based extractors.

    No re-embed. Returns number of rows updated.
    """
    ensure_phase_a_schema(conn)
    cols = existing_columns(conn)
    if "entities" not in cols:
        return 0

    sql = "SELECT id, memory, entities, kind FROM memories WHERE 1=1"
    params: list[Any] = []
    if user_id:
        sql += " AND user_id = ?"
        params.append(user_id)
    if only_empty:
        sql += " AND (entities IS NULL OR entities = '' OR entities = '[]')"
    sql += " ORDER BY created_at DESC"
    if limit and limit > 0:
        sql += " LIMIT ?"
        params.append(int(limit))

    rows = conn.execute(sql, params).fetchall()
    updated = 0
    for row in rows:
        text = row["memory"] or ""
        ents = extract_entities(text)
        kind = classify_kind(text, default=str(row["kind"] or KIND_FACT))
        conn.execute(
            "UPDATE memories SET entities = ?, kind = ? WHERE id = ?",
            (entities_to_json(ents), kind, row["id"]),
        )
        updated += 1
    if updated:
        conn.commit()
        log.info("memory Phase B backfill: updated %d rows", updated)
    return updated

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
    KNN cosine search against memories_vec, filtered by user_id.
    When threshold is supplied, only rows with dist <= (1 - threshold) are returned.
    When active_only, superseded memories (status = 'superseded') are excluded.
    """
    vec_blob = sqlite_vec.serialize_float32(vector)
    status_sql = _active_sql(active_only)
    if threshold is not None:
        dist_ceil = 1.0 - threshold
        return conn.execute(
            """
            SELECT v.id, vec_distance_cosine(v.embedding, ?) AS dist
            FROM memories_vec v
            JOIN memories m ON m.id = v.id
            WHERE m.user_id = ?
              AND vec_distance_cosine(v.embedding, ?) <= ?
              {status_sql}
            ORDER BY dist ASC
            LIMIT ?
            """.format(status_sql=status_sql),
            (vec_blob, user_id, vec_blob, dist_ceil, limit),
        ).fetchall()
    return conn.execute(
        """
        SELECT v.id, vec_distance_cosine(v.embedding, ?) AS dist
        FROM memories_vec v
        JOIN memories m ON m.id = v.id
        WHERE m.user_id = ?
          {status_sql}
        ORDER BY dist ASC
        LIMIT ?
        """.format(status_sql=status_sql),
        (vec_blob, user_id, limit),
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
        i = j
    return None


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
        relevant rerank. Pinned rows only get MEMORY_RANK_PINNED_WEIGHT as
        a mild score bonus (no reserved slots). See module docstring for
        the full stage breakdown.

    Fixes applied:
      - Phase A schema (status/supersedes_id/kind/source/entities columns)
        is now migrated once at __init__ time, not lazily on first write.
        Previously a fresh boot or un-migrated DB would hit
        `search()` before any `add()`/`add_raw()` call and crash with
        "no such column: m.status", because `_active_sql()` referenced a
        column that had never been created.
      - `search()` now holds `self._db_lock` across its entire body (quick
        pass, wide pass, and the scoring/rank step in between), not just
        the first KNN call. The connection is shared with the async write
        worker thread, so partial locking left most of the read path
        racing against concurrent writes/dream/cleanup.
      - `iter_all()` now takes the lock around each page fetch (not held
        across the yield, to avoid blocking other threads for the whole
        duration a caller spends processing a batch).
      - `add()`'s created_at now uses datetime.now(timezone.utc).isoformat(),
        matching add_raw()/_touch_memories()/pin(). It previously used
        bioclock.local_now(), which produced two incompatible clock
        formats in the same column depending on write path — skewing
        _rank_and_score's recency scoring, silently breaking _dream_boost's
        is_recent check (naive-vs-aware subtraction raised, was swallowed
        by a bare except), and making get_since()/get_between()'s raw
        string range comparisons sort inconsistently.
      - `_wait_for_write_window()`'s hard-cap check no longer requires
        `not is_active_turn()` to also be true. The deadline now overrides
        a turn state stuck "active" forever — which is the exact scenario
        MEMORY_WRITE_MAX_WAIT exists to guard against, so gating the cap on
        that same condition meant it could never fire when it mattered.
    """

    def __init__(
        self,
        db_path:         str,
        llm_base_url:    str,
        model:           str,
        embed_cache:     str | None = None,
        user_id:         str | None = None,   # NEW
    ) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db_path  = db_path
        self._user_id  = user_id or current_user_id()   # CHANGED
        self._llm_base = llm_base_url.rstrip("/")
        self._model    = model
        self._client   = OpenAI(base_url=self._llm_base, api_key="not-needed")
        self._embedder = HarrierEmbedder()
        self._conn = self._connect()
        self._db_lock = threading.RLock()
        # FIX 1: migrate Phase A schema immediately, not lazily inside
        # add()/add_raw(). Otherwise a read-only path (search() on a fresh
        # boot or a DB that hasn't been written to yet) hits `_active_sql()`
        # referencing `m.status` before that column exists.
        with self._db_lock:
            ensure_phase_a_schema(self._conn)

    def _connect(self) -> sqlite3.Connection:
        return initialize_store_db(self._db_path, _DDL, user_id=self._user_id, vector=True)

    # ── embedding ─────────────────────────────────────────────────────────────

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

    def _extract_facts(self, messages: list[dict], display_name: str | None = None) -> list[str]:
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

        user_name = (display_name or current_display_name()).strip()
        convo = "\n".join(
            f"{user_name}: {m['content'].strip()}" if m["role"] == "user"
            else f"Aiko: {m['content'].strip()}"
            for m in clean_messages
        )

        prompt = _EXTRACT_PROMPT.format(conversation=convo, user_name=user_name)

        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                max_tokens=_EXTRACT_MAX_TOKENS,
                temperature=0.0,  # deterministic — reduces hallucinated facts
                timeout=_EXTRACT_TIMEOUT,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "facts",
                        "schema": {"type": "array", "items": {"type": "string"}},
                    },
                },
            )
            raw = (resp.choices[0].message.content or "").strip()
        except Exception as e:
            log.warning(f"Extraction LLM call failed: {e}")
            return []

        # Grammar-constrained output: no markdown fences, no repeated arrays,
        # no <think> preamble possible (schema forces token 0 to be '[').
        # Parsing is a plain json.loads — the old fence-strip/first-array
        # salvage logic is no longer needed for this call site.
        try:
            facts = json.loads(raw)
            if isinstance(facts, list):
                facts = [f.strip() for f in facts if isinstance(f, str) and f.strip()]
            else:
                return []
        except json.JSONDecodeError:
            log.warning(f"Failed to parse extraction JSON despite schema constraint: {raw[:200]!r}")
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

    def _insert_row(
        self,
        *,
        mem_id: str,
        user_id: str,
        text: str,
        now: str,
        vector: list[float],
        pinned: int = 0,
        source: str = SOURCE_CHAT,
        supersedes_id: str | None = None,
        kind: str | None = None,
        entities: list[str] | None = None,
    ) -> None:
        """Insert one memory row (Phase A columns when present) + its vector,
        and best-effort co-mention edges for the entity graph."""
        cols = existing_columns(self._conn)
        kind_val = kind or classify_kind(text, default=KIND_FACT)
        ents_list = entities if entities is not None else extract_entities(text)
        ents_json = entities_to_json(ents_list)
        if "status" in cols:
            self._conn.execute(
                """
                INSERT INTO memories
                    (id, user_id, memory, created_at, access_count, last_accessed_at, pinned,
                     status, supersedes_id, kind, source, entities)
                VALUES (?, ?, ?, ?, 0, 'never', ?, ?, ?, ?, ?, ?)
                """,
                (
                    mem_id, user_id, text, now, pinned,
                    STATUS_ACTIVE, supersedes_id, kind_val, source, ents_json,
                ),
            )
        else:
            self._conn.execute(
                """
                INSERT INTO memories
                    (id, user_id, memory, created_at, access_count, last_accessed_at, pinned)
                VALUES (?, ?, ?, ?, 0, 'never', ?)
                """,
                (mem_id, user_id, text, now, pinned),
            )
        self._conn.execute(
            "INSERT INTO memories_vec(id, embedding) VALUES (?, ?)",
            (mem_id, sqlite_vec.serialize_float32(vector)),
        )

        # Phase D: live co-mention edges (best-effort; never fail the write)
        try:
            if ents_list and len([e for e in ents_list if str(e).strip()]) >= 2:
                upsert_co_mentions(
                    self._conn,
                    user_id=user_id,
                    entities=ents_list,
                    memory_id=mem_id,
                    updated_at=now if isinstance(now, str) else None,
                )
        except Exception as e:
            log.debug("entity_relations upsert skipped: %s", e)

    def _maybe_supersede_neighbor(
        self, user_id: str, vector: list[float], text: str
    ) -> tuple[str, str | None]:
        """Classify the write op against the nearest existing memory: 'add',
        'noop' (near-duplicate, skip), or 'supersede' (replace old_id)."""
        existing = _sqlite_knn_search(
            self._conn, vector, user_id,
            limit=1, threshold=WRITE_DEDUP_THRESHOLD, active_only=True,
        )
        if not existing:
            return "add", None
        sim = 1.0 - float(existing[0]["dist"])
        old_id = str(existing[0]["id"])
        row = self._conn.execute(
            "SELECT memory, pinned FROM memories WHERE id = ?", (old_id,)
        ).fetchone()
        old_text = (row["memory"] if row else "") or ""
        pinned = bool(row and row["pinned"])
        op = classify_write_op(
            similarity=sim,
            new_text=text,
            old_text=old_text,
            dedup_threshold=WRITE_DEDUP_THRESHOLD,
        )
        if op == "supersede" and pinned:
            return "add", None
        if op == "supersede":
            return "supersede", old_id
        return op, None

    def add(self, messages: list[dict], user_id: str, display_name: str | None = None) -> list[str]:
        """
        Extract facts and persist each as a row in memories + memories_vec.

        Write-path dedup/supersede: before inserting each fact, a KNN search
        checks for a near-identical vector already in the store. Near-duplicate
        text is skipped ('noop'); text that changed but is semantically the same
        supersedes the older row (status -> 'superseded').

        Embeddings for all extracted facts are computed in a single batched
        call rather than one-by-one.

        Returns list of new memory IDs. Empty list if nothing extracted.
        """
        facts = self._extract_facts(messages, display_name=display_name)
        if not facts:
            return []

        # created_at is UTC everywhere (matches add_raw()/_touch_memories()/
        # pin()) — see the class docstring's "Fixes applied" note. Mixing
        # local_now() here and UTC elsewhere broke every downstream
        # comparison: _rank_and_score's recency scoring, _dream_boost's
        # is_recent check, and get_since()/get_between()'s string range
        # comparisons.
        now = datetime.now(timezone.utc).isoformat()
        ids: list[str] = []

        try:
            vectors = self._embed_batch(facts)
        except Exception as e:
            log.warning("Batch embedding failed, aborting write: %s", e)
            return []

        with self._db_lock:
            try:
                for fact, vector in zip(facts, vectors):
                    op, supersedes_id = self._maybe_supersede_neighbor(user_id, vector, fact)
                    if op == "noop":
                        log.debug("Skipping near-duplicate fact: %r", fact)
                        continue
                    if op == "supersede" and supersedes_id:
                        cols = existing_columns(self._conn)
                        if "status" in cols:
                            self._conn.execute(
                                "UPDATE memories SET status = ? WHERE id = ?",
                                (STATUS_SUPERSEDED, supersedes_id),
                            )
                            log.info("Superseded memory %s with new fact", supersedes_id)
                    mem_id = str(uuid.uuid4())
                    self._insert_row(
                        mem_id=mem_id,
                        user_id=user_id,
                        text=fact,
                        now=now,
                        vector=vector,
                        pinned=0,
                        source=SOURCE_CHAT,
                        supersedes_id=supersedes_id,
                    )
                    ids.append(mem_id)
                self._conn.commit()
            except Exception as e:
                log.warning("Failed to upsert fact batch: %s", e)
                self._conn.rollback()
                return []
        return ids

    def add_raw(self, memory: str, user_id: str, *, pinned: bool = False) -> str | None:
        """
        Persist one already-curated memory string without LLM extraction.

        Runs the same write-time dedup/supersede check as add(): near-duplicates
        are skipped; semantically-equal-but-changed text supersedes the older
        row. This closes the gap that previously let repeated calls (e.g. a
        daily-record pin job re-running for the same day) accumulate unbounded
        duplicate rows — especially dangerous for pinned=True inserts, since
        dream()'s merge pass can never delete a pinned memory even as a
        duplicate loser.
        """
        text = (memory or "").strip()
        if not text:
            return None
        try:
            vector = self._embed(text)
        except Exception as e:
            log.warning("Failed to embed raw memory: %s", e)
            return None
        with self._db_lock:
            try:
                op, supersedes_id = self._maybe_supersede_neighbor(user_id, vector, text)
                if op == "noop":
                    log.debug("Skipping near-duplicate raw memory: %r", text[:80])
                    return None
                if op == "supersede" and supersedes_id:
                    cols = existing_columns(self._conn)
                    if "status" in cols:
                        self._conn.execute(
                            "UPDATE memories SET status = ? WHERE id = ?",
                            (STATUS_SUPERSEDED, supersedes_id),
                        )
                mem_id = str(uuid.uuid4())
                now = datetime.now(timezone.utc).isoformat()
                self._insert_row(
                    mem_id=mem_id,
                    user_id=user_id,
                    text=text,
                    now=now,
                    vector=vector,
                    pinned=1 if pinned else 0,
                    source=SOURCE_PIN if pinned else SOURCE_CHAT,
                    supersedes_id=supersedes_id,
                )
                self._conn.commit()
                return mem_id
            except Exception as e:
                log.warning("Failed to insert raw memory: %s", e)
                self._conn.rollback()
                return None

    # ── read ──────────────────────────────────────────────────────────────────

    def _fts_pass(self, fts_query: str | None, user_id: str, fts_limit: int, active_only: bool = True) -> list[sqlite3.Row]:
        """Run one FTS5 BM25 pass. Returns [] if fts_query is None (nothing usable to match).
        Caller must hold self._db_lock."""
        if fts_query is None:
            return []
        status_sql = _active_sql(active_only)
        return self._conn.execute(
            """
            SELECT f.id
            FROM memories_fts f
            JOIN memories m ON m.id = f.id
            WHERE memories_fts MATCH ?
            AND m.user_id = ?
            {status_sql}
            ORDER BY rank
            LIMIT ?
            """.format(status_sql=status_sql),
            (fts_query, user_id, fts_limit),
        ).fetchall()

    def _rank_and_score(
        self,
        rank_knn: dict,
        rank_fts: dict,
    ) -> tuple[list[str], dict, dict]:
        """
        Dedup + score one candidate pool (from either the quick or wide pass).
        Caller must hold self._db_lock.

        1. Fetch full rows for the union of KNN/FTS candidate ids.
        2. Collapse exact-text duplicates, keeping the most recently created
           row per duplicate cluster.
        3. Score every surviving id: RRF fusion + recency/access/pinned bonuses.

        Returns (ids sorted best-first by score, {id: score}, {id: row}).
        Recency-among-relevant reranking is applied afterward by the
        caller (search()), not here — this method only produces the base
        score-ordered list (RRF + recency + access + pinned weight).
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

    def search(self, query: str, user_id: str, limit: int = 5, vector: list[float] | None = None, include_history: bool = False) -> list[dict]:
        """
        KNN + FTS5 -> RRF fusion search, with a tiered quick/wide candidate
        pass and recency-among-relevant reranking. Pinned status is only a
        MEMORY_RANK_PINNED_WEIGHT score tiebreaker (no reserved slots).
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
        5. Truncate to `limit` and return as payload dicts.
           (Pinned is already folded into scores in step 2 via MEMORY_RANK_PINNED_WEIGHT.)

        vector — pre-computed query embedding; skips the _embed HTTP call.
        include_history — when False (default), superseded memories are excluded.

        FIX 2: the entire DB-touching portion of this method (both KNN
        passes, both FTS passes, and the scoring step) now runs under a
        single `self._db_lock` acquisition. Previously only the first quick
        KNN call was locked, leaving the FTS pass, the scoring pass (which
        reads the full `memories` rows), and the wide-pass fallback racing
        against the async write-worker thread on the same connection.
        """
        if vector is None:
            vector = self._embed(query, query=True)
        fts_query = _sanitize_fts_query(query)
        active_only = not include_history

        with self._db_lock:
            quick_knn_rows = _sqlite_knn_search(
                self._conn, vector, user_id, QUICK_KNN_LIMIT, active_only=active_only
            )
            rank_knn_q = {row["id"]: i + 1 for i, row in enumerate(quick_knn_rows)}
            quick_fts_rows = self._fts_pass(fts_query, user_id, QUICK_FTS_LIMIT, active_only=active_only)
            rank_fts_q = {row["id"]: i + 1 for i, row in enumerate(quick_fts_rows)}

            scored_ids, scores, row_by_id = self._rank_and_score(rank_knn_q, rank_fts_q)

            confident = (
                len(scored_ids) >= limit
                and scores.get(scored_ids[limit - 1], 0.0) >= MEMORY_RECALL_SCORE_THRESHOLD
            )

            # ── widen only if the quick pass was under-filled or under-confident ──
            if not confident:
                wide_knn_rows = _sqlite_knn_search(
                    self._conn, vector, user_id, KNN_LIMIT, active_only=active_only
                )
                rank_knn_w = {row["id"]: i + 1 for i, row in enumerate(wide_knn_rows)}
                wide_fts_rows = self._fts_pass(fts_query, user_id, FTS_LIMIT, active_only=active_only)
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

    def iter_all(self, user_id: str, batch_size: int = MEMORY_LIFECYCLE_BATCH_SIZE):
        """Yield memory records for a user in rowid order without one giant list.

        FIX: each page fetch is now taken under self._db_lock. The lock is
        NOT held across the yield, so a slow consumer (e.g. dream()/cleanup()
        processing a batch) doesn't block the write-worker thread for the
        whole duration — only the actual SQL scan is protected.
        """
        last_rowid = 0
        while True:
            with self._db_lock:
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
        return list(self.iter_all(user_id=user_id))

    def get_since(self, since: datetime, user_id: str | None = None) -> list[dict]:
        user_id = user_id or self._user_id
        with self._db_lock:
            rows = self._conn.execute(
                """
                SELECT * FROM memories
                WHERE user_id = ? AND created_at >= ?
                ORDER BY created_at DESC
                """,
                (user_id, since.isoformat()),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_between(self, start: datetime, end: datetime, user_id: str | None = None, limit: int = 0) -> list[dict]:
        user_id = user_id or self._user_id
        sql = """
            SELECT * FROM memories
            WHERE user_id = ? AND created_at >= ? AND created_at < ?
            ORDER BY created_at ASC
        """
        params: list[Any] = [user_id, start.isoformat(), end.isoformat()]
        if limit > 0:
            sql += " LIMIT ?"
            params.append(limit)
        with self._db_lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    # ── delete ────────────────────────────────────────────────────────────────

    def delete(self, memory_id: str) -> None:
        with self._db_lock:
            self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            self._conn.execute("DELETE FROM memories_vec WHERE id = ?", (memory_id,))
            self._conn.commit()

    def delete_all(self, user_id: str) -> None:
        with self._db_lock:
            self._conn.execute(
                "DELETE FROM memories_vec WHERE id IN (SELECT id FROM memories WHERE user_id = ?)",
                (user_id,),
            )
            self._conn.execute("DELETE FROM memories WHERE user_id = ?", (user_id,))
            self._conn.commit()


# ── Phase D: entity co-mention relations ─────────────────────────────────────
# Thin entity-relation layer (not a full graph DB): one ``entity_relations``
# table storing co-mention / explicit links between entity labels. Built
# primarily from co-occurrence on the same memory fact; no vector changes,
# no LLM. Write side lives here; read/export for the Studio lives in
# memory/studio/backend/graph_export.py.
#
# NOTE: this graph is currently write-only from AikoMemorize's perspective.
# _MemoryBackend.search() / AikoMemorize.search() (the RRF recall path) do
# NOT query entity_relations at all — co-mention edges are recorded on
# every write (_insert_row) but never consulted at recall time. If/when
# graph-aware recall is wanted, it would need an explicit read step here
# (e.g. boosting candidates connected to entities in the query) rather than
# relying on this table being populated as a side effect.

RELATION_CO_MENTION = "co_mentions"
RELATION_RELATED = "related_to"

_ENTITY_RELATIONS_DDL = """
CREATE TABLE IF NOT EXISTS entity_relations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     TEXT NOT NULL,
    entity_a    TEXT NOT NULL,
    entity_b    TEXT NOT NULL,
    relation    TEXT NOT NULL DEFAULT 'co_mentions',
    weight      REAL NOT NULL DEFAULT 1.0,
    memory_id   TEXT,
    updated_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_entity_rel_user ON entity_relations(user_id);
CREATE INDEX IF NOT EXISTS idx_entity_rel_a ON entity_relations(user_id, entity_a);
CREATE INDEX IF NOT EXISTS idx_entity_rel_b ON entity_relations(user_id, entity_b);
CREATE UNIQUE INDEX IF NOT EXISTS idx_entity_rel_pair
    ON entity_relations(user_id, entity_a, entity_b, relation);
"""


def _norm_entity(e: str) -> str:
    return (e or "").strip()


def _ordered_pair(a: str, b: str) -> tuple[str, str]:
    """Canonical unordered pair key (casefold order, display preserves first-seen casing via callers)."""
    aa, bb = _norm_entity(a), _norm_entity(b)
    if aa.casefold() <= bb.casefold():
        return aa, bb
    return bb, aa


def ensure_entity_relations_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_ENTITY_RELATIONS_DDL)
    conn.commit()


def upsert_co_mentions(
    conn: sqlite3.Connection,
    *,
    user_id: str,
    entities: Iterable[str],
    memory_id: str | None = None,
    updated_at: str | None = None,
) -> int:
    """Record co-mention pairs for entities on one memory. Returns pairs touched."""
    from memory.vecstore import utc_now_iso

    ents = []
    seen: set[str] = set()
    for e in entities:
        n = _norm_entity(e)
        if not n:
            continue
        key = n.casefold()
        if key in seen:
            continue
        seen.add(key)
        ents.append(n)
    if len(ents) < 2:
        return 0

    ensure_entity_relations_schema(conn)
    now = updated_at or utc_now_iso()
    touched = 0
    for a, b in combinations(ents, 2):
        ea, eb = _ordered_pair(a, b)
        if ea.casefold() == eb.casefold():
            continue
        conn.execute(
            """
            INSERT INTO entity_relations (user_id, entity_a, entity_b, relation, weight, memory_id, updated_at)
            VALUES (?, ?, ?, ?, 1.0, ?, ?)
            ON CONFLICT(user_id, entity_a, entity_b, relation) DO UPDATE SET
                weight = entity_relations.weight + 1.0,
                memory_id = COALESCE(excluded.memory_id, entity_relations.memory_id),
                updated_at = excluded.updated_at
            """,
            (user_id, ea, eb, RELATION_CO_MENTION, memory_id, now),
        )
        touched += 1
    return touched


def rebuild_entity_relations(
    conn: sqlite3.Connection,
    *,
    user_id: str,
    clear: bool = True,
) -> dict[str, int]:
    """Rebuild co-mention edges from memories.entities JSON for one user."""
    ensure_entity_relations_schema(conn)
    cols = {str(r[1]) for r in conn.execute("PRAGMA table_info(memories)").fetchall()}
    if "entities" not in cols:
        return {"pairs": 0, "memories": 0, "note": 1}

    if clear:
        conn.execute(
            "DELETE FROM entity_relations WHERE user_id = ? AND relation = ?",
            (user_id, RELATION_CO_MENTION),
        )

    rows = conn.execute(
        """
        SELECT id, entities FROM memories
        WHERE user_id = ?
          AND (status IS NULL OR status = 'active')
        """,
        (user_id,),
    ).fetchall()

    pairs = 0
    for row in rows:
        ents = entities_from_json(row["entities"])
        pairs += upsert_co_mentions(
            conn, user_id=user_id, entities=ents, memory_id=str(row["id"])
        )
    conn.commit()
    log.info("entity_relations rebuild user=%s memories=%d pairs=%d", user_id, len(rows), pairs)
    return {"pairs": pairs, "memories": len(rows)}


def _memory_db_path_for_user(uid: str) -> str:
    if uid == "guest":
        return _guest_memory_db()
    env_path = os.getenv("SQLITE_MEMORY_PATH", "").strip()
    if env_path:
        return os.path.expanduser(env_path)
    return str(resolve_user_db_path("memory/memory.db", user_id=uid))
    

def vacuum_memory_db(user_id: str | None = None) -> None:
    """Reclaim space after bulk memory deletes during maintenance."""
    uid = user_id or current_user_id()
    conn = initialize_store_db(_memory_db_path_for_user(uid), _DDL, user_id=uid, vector=True)
    try:
        conn.execute("VACUUM")
        conn.execute("ANALYZE")
        conn.commit()
    finally:
        conn.close()


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
        immune to cleanup(), dream prune, and dream merge (as the loser).
        At recall they compete on the same blended score as everything
        else, with only MEMORY_RANK_PINNED_WEIGHT as a mild tiebreaker
        (the old stage-4 reserved-slot path was removed).
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
        self._display_name: str | None = None
        self._search_cache: OrderedDict[tuple[str, str, int, bool], tuple[float, list[dict]]] = OrderedDict()
        self._search_cache_lock = threading.RLock()
        self._llm_base_url = os.getenv("LLM_BASE_URL", "http://localhost:8080/v1")
        self._model = os.getenv("EXTRACT_MODEL") or os.getenv("LLM_MODEL", "ministral")
        self._embed_cache = os.getenv("EMBED_CACHE_PATH") or os.getenv("FASTEMBED_CACHE_PATH")

        # Use a .pending path for pre-auth boot so user-space dirs are never
        # created before a real user logs in via the web UI.
        uid = current_user_id()
        db_path = _memory_db_path_for_user(uid)
        # Guest remains tempfile-backed to avoid unbounded heap growth.
        if not silent:
            log.info("Opening sqlite-vec memory store for %s ...", uid)
        self._mem = _MemoryBackend(
            db_path=db_path,
            llm_base_url=self._llm_base_url,
            model=self._model,
            embed_cache=self._embed_cache,
            user_id=uid,
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
        db_path = _memory_db_path_for_user(uid)
        if not self._silent:
            log.info("Opening sqlite-vec memory store for %s ...", uid)
        self._mem = _MemoryBackend(
            db_path=db_path,
            llm_base_url=self._llm_base_url,
            model=self._model,
            embed_cache=self._embed_cache,
            user_id=uid,
        )
        self._conn = self._mem._conn
        if not self._silent:
            log.info("Memory store ready for %s.", uid)

def switch_user(self, user_id: str) -> None:
    # Drain any pending writes first
    self.wait_for_writes(timeout=5.0)
    
    self._user_id_override = user_id
        self._display_name = None
        if self._conn:
            with self._mem._db_lock:
                try:
                    self._conn.execute("PRAGMA optimize")
                    self._conn.commit()
                    self._conn.close()
                except Exception:
                    log.warning("memorize: PRAGMA optimize failed")
        self._open(user_id)

    def get_user_id(self) -> str:
        """Return the user_id this instance is currently opened for."""
        return self._user_id_override or self._mem._user_id

    def set_display_name(self, name: str) -> None:
        """Set the display name for this user (e.g. GitHub login)."""
        self._display_name = name.strip() if name else None
    
    def get_display_name(self) -> str:
        """Return the display name for this user, or fall back to user_id."""
        return self._display_name or self.get_user_id()
      
    def _resolve_user_id(self, user_id: str | None = None) -> str:
        """Resolve the effective user_id for this call.

        An explicit argument always wins. Otherwise, falls back to THIS
        instance's own bound identity (get_user_id()) — never the ambient
        contextvar. An AikoMemorize instance is constructed for (or
        switch_user()'d to) a specific user; calls issued against it from
        another thread — the scheduler's daemon thread, the async write
        worker, a standalone script — must resolve against that bound
        identity, not whatever current_user_id() happens to return in
        the calling thread's own context (which is usually unset/"guest").
        This is the fix for the ambient-user-id bug class tracked across
        memory/ and sensory/.
        """
        return user_id or self.get_user_id()

    # ── write ─────────────────────────────────────────────────────────────────

    def add(self, messages: list[dict], user_id: str | None = None, display_name: str | None = None) -> bool:
        """
        Store a conversation turn into long-term memory.
        Returns True on success, False on failure.
        """
        try:
            user_id = self._resolve_user_id(user_id)
            t       = time.perf_counter()
            ids     = self._mem.add(messages, user_id=user_id, display_name=display_name)
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

    def pin(self, messages: list[dict], user_id: str | None = None, display_name: str | None = None) -> bool:
        """
        Store messages and immediately mark all resulting memories as pinned.
        Pinned memories are immune to cleanup, dream pruning, and merge losses.
        Returns True on success, False on any failure.
        """
        try:
            user_id = self._resolve_user_id(user_id)
            ids = self._mem.add(messages, user_id=user_id, display_name=display_name)

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
                with self._mem._db_lock:
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
        display_name = current_display_name()
        self._write_queue.put((user_input, response_text, user_id, display_name, is_active_turn, idle_since))

    def _write_loop(self) -> None:
        while True:
            user_input, response_text, user_id, display_name, is_active_turn, idle_since = self._write_queue.get()
            try:
                self._wait_for_write_window(is_active_turn, idle_since)
                self.add([
                    {"role": "user", "content": user_input[:500]},
                    {"role": "assistant", "content": response_text[:800]},
                ], user_id=user_id, display_name=display_name)
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
            # FIX: the hard cap now overrides is_active_turn() outright,
            # instead of requiring it to also be False. The old
            # `and not is_active_turn()` gate meant the cap could never
            # fire in exactly the scenario it exists for — a turn state
            # stuck "active" forever — letting the wait spin indefinitely.
            if MEMORY_WRITE_MAX_WAIT > 0 and time.monotonic() >= deadline:
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

    def search(self, query: str, user_id: str | None = None, limit: int = 5, query_vector: list[float] | None = None, include_history: bool = False) -> list[dict]:
        """
        Retrieve top-k memories relevant to the current query.
        Side-effect: increments access_count and updates last_accessed_at
        for all returned memories in a single batched UPDATE.
        """
        user_id = self._resolve_user_id(user_id)
        if _is_trivial_input(query or ""):
            log.debug(f"Skipping search for trivial input: {query!r}")
            return []
    
        if _BROAD_RECALL_RE.search(query or ""):
            results = self._recent_or_important_memories(
                user_id=user_id, limit=limit, include_history=include_history
            )
            self._touch_memories(results)
            return results[:int(limit)]
    
        cache_key = (user_id, " ".join((query or "").lower().split()), int(limit), bool(include_history))
        now_s = time.monotonic()
    
        with self._search_cache_lock:
            cached = self._search_cache.get(cache_key)
            if cached and now_s - cached[0] <= MEMORY_SEARCH_CACHE_TTL:
                self._search_cache.move_to_end(cache_key)
                results = [dict(r) for r in cached[1]]
                log.debug("[memory] cache hit, scores=%s", [r.get("_recall_score") for r in results])
                self._touch_memories(results)
                return results
            if cached:
                self._search_cache.pop(cache_key, None)
    
        # Run the core RRF search
        results = self._mem.search(
            query,
            user_id=user_id,
            limit=limit,
            vector=query_vector,
            include_history=include_history,
        )
    
        # Entity-aware rerank (optional, feature-gated)
        if os.getenv("AIKO_ENTITY_BOOST"):
            query_entities = self._extract_query_entities(query)
            if query_entities:
                results = self._boost_by_entity_relations(query_entities, results)
                log.debug(f"Boosted results by entities: {query_entities}")
    
        self._touch_memories(results)
        log.debug("[memory] search miss, scores=%s", [r.get("_recall_score") for r in results])
    
        # Search replay logging (optional, feature-gated)
        if os.getenv("AIKO_REPLAY_SEARCHES"):
            self._write_search_replay(query, results, user_id)
    
        # Cache and return
        with self._search_cache_lock:
            self._search_cache[cache_key] = (now_s, [dict(r) for r in results])
            while len(self._search_cache) > MEMORY_SEARCH_CACHE_SIZE:
                self._search_cache.popitem(last=False)
    
        return results
      
    def _write_search_replay(self, query: str, results: list[dict], user_id: str) -> None:
        """Append search to replay log (debug/tuning, env-gated)."""
        try:
            replay_path = Path.home() / ".aiko" / "memory" / "search_replay.jsonl"
            replay_path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "user_id": user_id,
                "query": query[:200],
                "result_count": len(results),
                "results": [
                    {
                        "id": r["id"],
                        "score": round(r.get("_recall_score", 0.0), 6),
                        "text": r["memory"][:100],
                    }
                    for r in results
                ],
            }
            with open(replay_path, "a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            log.debug(f"search replay write failed: {e}")
      
    def _recent_or_important_memories(self, user_id: str, limit: int, include_history: bool = False) -> list[dict]:
        """
        FIX 3: status filtering (active-only, unless include_history) is now
        applied in SQL, before the dedup+truncate step below — not by the
        caller after truncation. Previously the caller filtered out
        superseded rows AFTER this method had already cut the candidate
        pool down to `limit`, which could return fewer than `limit` results
        even when the wider fetch window had enough active memories to
        fill it.
        """
        fetch_n = max(int(limit) * 4, int(limit) + 10)
        status_sql = _active_sql(not include_history)
        with self._mem._db_lock:
            rows = self._conn.execute(
                """
                SELECT *
                FROM memories m
                WHERE m.user_id = ?
                  {status_sql}
                ORDER BY m.pinned DESC, m.created_at DESC, m.access_count DESC
                LIMIT ?
                """.format(status_sql=status_sql),
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
        if not results:
            return
        now = datetime.now(timezone.utc).isoformat()
        mem_ids = [str(r.get("id", "")) for r in results if r.get("id")]
        if not mem_ids:
            return
        with self._mem._db_lock:
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

    def format_for_context(self, memories: list[dict]) -> str | None:
        """
        Format retrieved memories into a compact string for injection
        into the conversation context. Returns None if nothing to inject.

        created_at is always stored in UTC (see _MemoryBackend.add()/
        add_raw()), but the age labels here ("today", "yesterday", "N days
        ago") should reflect the person's local calendar day, not UTC's.
        So each row's UTC created_at is converted into local time before
        diffing against bioclock.local_now() — diffing the raw UTC
        timestamp against a local "now" would misplace the day boundary
        by whatever the local UTC offset is (e.g. a memory from 11pm local
        last night could read as "today" or vice versa near midnight).
        """
        if not memories:
            return None

        now = bioclock.local_now()
        # Local tz offset used to convert stored UTC timestamps into the
        # same local frame as `now`, regardless of whether bioclock returns
        # a naive or tz-aware datetime.
        local_tz = datetime.now().astimezone().tzinfo
        now_is_aware = isinstance(now, datetime) and now.tzinfo is not None

        lines = [
            "<memory_context>",
            "Facts about the person you are speaking with — not a separate person. Use silently. Never quote or reference this block directly.",
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
                    ts = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    if ts.tzinfo is None:
                        # legacy rows written before the UTC-everywhere fix —
                        # treat as UTC rather than silently mismatching.
                        ts = ts.replace(tzinfo=timezone.utc)
                    ts_local = ts.astimezone(local_tz)
                    if not now_is_aware:
                        ts_local = ts_local.replace(tzinfo=None)
                    delta = now - ts_local
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
  
    # ── entity-aware recall ───────────────────────────────────────────────────
      
    def _extract_query_entities(self, query: str) -> list[str]:
        """Extract entities mentioned in the query using the same rules as _insert_row."""
        from memory.memorize import extract_entities
        return extract_entities(query, max_entities=5)
    
    def _boost_by_entity_relations(self, query_entities: list[str], results: list[dict]) -> list[dict]:
        """
        Rerank results: memories that co-mention query entities rank higher.
        Runs after RRF ranking but before return.
        """
        if not query_entities or not results:
            return results
        
        # Fetch all memories related to query entities
        with self._mem._db_lock:
            placeholders = ",".join("?" * len(query_entities))
            rows = self._conn.execute(
                f"""
                SELECT DISTINCT memory_id, SUM(weight) as relation_weight
                FROM entity_relations
                WHERE user_id = ? 
                  AND (entity_a IN ({placeholders}) OR entity_b IN ({placeholders}))
                  AND memory_id IS NOT NULL
                GROUP BY memory_id
                ORDER BY relation_weight DESC
                """,
                [self.get_user_id()] + query_entities + query_entities,
            ).fetchall()
        
        related_ids = {str(row["memory_id"]): row["relation_weight"] for row in rows}
        
        # Rerank: memories with entity overlap float to top, preserving RRF order otherwise
        def entity_boost_score(result: dict) -> tuple[int, float]:
            mid = str(result.get("id", ""))
            if mid in related_ids:
                return (0, -related_ids[mid])  # (0, ...) sorts before (1, ...), negated weight for descending
            return (1, 0)
        
        reranked = sorted(results, key=entity_boost_score)
        return reranked
  
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
        user_id = self._resolve_user_id(user_id)
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
            with self._mem._db_lock:
                pinned_ids = _sqlite_pinned_ids(self._conn, batch_ids)
            boosted += self._dream_boost(batch, payload_map, pinned_ids=pinned_ids, dry_run=dry_run)

        if not mem_ids:
            log.info("No memories found — nothing to do.")
            return {"boosted": 0, "merged": 0, "pruned": 0, "duration_s": 0.0}

        with self._mem._db_lock:
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
                    log.warning("memorize: failed to parse created_at")

            is_salient = (
                bool(_SALIENCE_RE.search(text))
                or ac >= 3
                or is_recent
            )

            if not is_salient:
                continue

            boost_ids.append(mem_id)

        if boost_ids and not dry_run:
            with self._mem._db_lock:
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

            with self._mem._db_lock:
                vector = _sqlite_get_vector(self._conn, mem_id)
            if not vector:
                continue

            with self._mem._db_lock:
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
        pinned_ids = pinned_ids or set()
        if id_a in pinned_ids or id_b in pinned_ids:
            log.info(f"Skipping merge: one or both of ({id_a}, {id_b}) is pinned.")
            return False

        payload_map = self._batch_get_payloads([id_a, id_b])
        ac_a, _     = payload_map.get(id_a, (0, "never"))
        ac_b, _     = payload_map.get(id_b, (0, "never"))
        with self._mem._db_lock:
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
        _all_mems: list[dict] | None = None,
        _pinned_ids: set[str] | None = None,
    ) -> dict:
        """
        Prune decayed memories below threshold score.
        Grace period (default 35 days) protects newly created memories.
        Pinned memories are unconditionally kept.

        _all_mems: internal — when called from dream(), the already-fetched
        memory list is passed through here to avoid a redundant get_all() scan.

        Returns dict: {deleted, kept, failed, candidates (dry_run only)}.
        """
        user_id = self._resolve_user_id(user_id)
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

    def _iter_memory_batches(self, user_id: str, batch_size: int = MEMORY_LIFECYCLE_BATCH_SIZE):
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
        _pinned_ids: set[str] | None = None,
    ) -> tuple[int, list[dict]]:
        mem_ids     = [str(m.get("id", "")) for m in all_mems if m.get("id")]
        payload_map = self._batch_get_payloads(mem_ids)
        with self._mem._db_lock:
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
        with self._mem._db_lock:
            try:
                self._conn.execute("PRAGMA optimize")
                self._conn.commit()
            except Exception as e:
                log.debug(f"SQLite optimize skipped: {e}")

    # ── debug ─────────────────────────────────────────────────────────────────

    def get_all(self, user_id: str | None = None) -> list[dict]:
        """Return all stored memories for a user."""
        user_id = self._resolve_user_id(user_id)
        return self._mem.get_all(user_id=user_id)

    def add_raw(self, memory: str, user_id: str | None = None, *, pinned: bool = False, metadata: dict | None = None) -> str | None:
        """Persist one already-curated memory string without LLM extraction."""
        # metadata is accepted for call-site clarity; the current schema stores
        # only the curated text plus pinned flag.
        user_id = self._resolve_user_id(user_id)
        mem_id = self._mem.add_raw(memory, user_id=user_id, pinned=pinned)
        if mem_id:
            self._maybe_clear_search_cache()
        return mem_id

    def get_since(self, since: datetime, user_id: str | None = None) -> list[dict]:
        """Return memories created on or after `since`, newest first."""
        user_id = self._resolve_user_id(user_id)
        return self._mem.get_since(since, user_id=user_id)

    def get_between(self, start: datetime, end: datetime, user_id: str | None = None) -> list[dict]:
        """Return memories created in [start, end), oldest first."""
        user_id = self._resolve_user_id(user_id)
        return self._mem.get_between(start, end, user_id=user_id)

    def delete(self, memory_id: str) -> None:
        """Delete one memory from the store and clear search cache."""
        self._mem.delete(memory_id)
        self._clear_search_cache()


    def clear(self, user_id: str | None = None) -> None:
        """Wipe all memories for a user. Use carefully."""
        user_id = self._resolve_user_id(user_id)
        self._mem.delete_all(user_id=user_id)
        self._clear_search_cache()
        log.info(f"Cleared all memories for user '{user_id}'.")

    # ── internal ──────────────────────────────────────────────────────────────

    def _batch_get_payloads(self, mem_ids: list[str]) -> dict:
        """Batch retrieve access_count + last_accessed_at in a single query."""
        return _sqlite_batch_get_payloads(self._conn, mem_ids)

    def embed_text(self, text: str, *, query: bool = False) -> list[float]:
        """Embed one text string with the configured memory embedding model."""
        return self._mem._embed(text, query=query)

    def embed_texts(self, texts: list[str], *, query: bool = False) -> list[list[float]]:
        """Embed multiple strings with the configured memory embedding model."""
        if query:
            return self._mem._embedder.embed_queries(texts).tolist()   # applies instruct prefix
        return self._mem._embed_batch(texts)                           # document side — no prefix
