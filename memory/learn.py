"""
memory/learn.py

Aiko's two research depths, both operating on a topic YOU (or the idle
learner loop, or the scheduled deep-study window) hand her — neither depth
picks its own topic on its own.

  - quick_studying: a single interactive-scale research pass. Thin alias
    over agentic.toolkit.research.deep_research — same TTL-cached, in-memory,
    ephemeral behavior you already have. Use this for "look this up for me
    now" and for the idle learner's short-idle-gap top-ups.

  - deep_studying: an autonomous, potentially long-running research pass
    meant for genuine idle/overnight time. Runs many iterations against a
    single topic, persists everything to a disk-backed scratch SQLite store
    for the duration of the call (so 50-100 iterations can revisit the same
    pages without re-fetching or re-embedding them), rate-limits itself per
    host so it doesn't hammer the same sites across iterations, and ends by
    distilling the accumulated evidence into compact atomic facts. It can
    also be interrupted gracefully mid-run via a threading.Event (see
    `stop_event` below) — this is what lets a scheduled window (e.g.
    "weekdays 05:00-18:00") stop a long-running session at the window's
    edge instead of only at max_iterations.

    The scratch store is deleted when the call returns — it is NOT a
    persistent index. It exists only to make one long call efficient, the
    same way a browser's per-page DOM exists only for that page's lifetime.
    Long-term persistence is the caller's job: deep_studying returns the
    distilled facts and (optionally) calls an on_distilled hook so whatever
    owns your actual knowledge graph / MSB schema can decide how to store
    them. This file does not know that schema and should not guess at it.

Also owns:
  - idle_learner_loop: the background process that decides, once Aiko's
    proactive check-in cycle has finished and she's gone quiet to rest
    (NOT merely "no chat messages recently" — see the docstring below), to
    pick a recent topic and study it via quick_studying, then persist the
    result to memory.
  - the scheduled deep-study window manager (_DeepStudySessionManager,
    deep_study_window_start/stop, register_deep_study_handlers): wires
    deep_studying into system.schedule's handler-based jobs so it runs only
    inside a configured wall-clock window (see
    schedule.ensure_deep_study_window_jobs) and stops cleanly at the
    window's edge.

Config note: this module's tunables (IDLE_LEARN_SECONDS,
QUICK_STUDY_MAX_ROUNDS, DEEP_STUDY_MAX_ITERATIONS, etc.) are read from
os.environ at import time via os.getenv(...). Those values are populated
from config/learn.yaml by system.config.load_config(). We call load_config()
explicitly at the top of this module (it's idempotent — see system/config.py's
_LOADED guard) so learn.yaml's values are honored regardless of what else
has or hasn't imported system.config yet. Without this, module-level
os.getenv() calls would silently fall back to hardcoded defaults if learn.py
happened to be imported before whatever else in the app calls load_config().

This module reads its own tunables directly from os.environ rather than
importing constants from agentic.toolkit.research — it owns its own view of
shared knobs like CONDENSE_CHUNK_CHARS/CONDENSE_MIN_SCORE/search-result-count
instead of reaching into another module's constants (which would also
silently go stale if that module's env var name ever changed).
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import os
import re
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse

from system.config import load_config

load_config()

from cognition import reason
from agentic.toolkit.research import (
    _ask_llm_json,
    _is_private_or_local_host,
    _web_search_raw,
    deep_research,
    web_fetch,
)

from system.log import get_logger

log = get_logger(__name__)

# Tunables this module owns its own env-backed view of (see module
# docstring) rather than importing from agentic.toolkit.research.
CONDENSE_CHUNK_CHARS = int(os.getenv("CONDENSE_CHUNK_CHARS", 500))
CONDENSE_MIN_SCORE = float(os.getenv("CONDENSE_MIN_SCORE", 0.15))
DEEP_SEARCH_MAX_RESULTS = int(os.getenv("SEARXNG_MAX_RESULTS", 5))


def _cosine(a: list[float], b: list[float]) -> float:
    """Plain single-pair cosine similarity. cognition.reason's batched cosine
    helper (reason.batch_cosine_scores) is numpy-oriented for scoring many
    chunks against one query vector at once; deep_studying's distillation
    step scores one (topic_vec, chunk_embedding) pair at a time against
    arbitrary accumulated chunks, so a tiny pure-python version avoids
    forcing a numpy import/allocation per chunk here."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# ── quick_studying ────────────────────────────────────────────────────────────

QUICK_STUDY_MAX_ROUNDS = int(os.getenv("QUICK_STUDY_MAX_ROUNDS", 3))


def quick_studying(
    topic: str,
    client=None,
    model: str | None = None,
    embedder=None,
    max_rounds: int = QUICK_STUDY_MAX_ROUNDS,
    num_searches: int | None = None,
    num_fetches: int | None = None,
) -> str:
    """Interactive-depth research on a topic. This is exactly deep_research —
    same TTL cache, same in-memory ephemeral scoring, same single-call cost
    model. Named separately so call sites (the idle learner's short-gap
    path, or a direct /research command) read as "which depth am I asking
    for" rather than exposing agentic.toolkit.research internals at every
    call site.

    max_rounds defaults to QUICK_STUDY_MAX_ROUNDS (config/learn.yaml),
    actually wired through now — previously this default was hardcoded to
    3 regardless of what QUICK_STUDY_MAX_ROUNDS was set to.

    num_searches/num_fetches are optional pass-throughs to deep_research's
    own per-call overrides (see agentic.toolkit.research.deep_research).
    Leave them None to use deep_research's own DEEP_RESEARCH_NUM_SEARCHES/
    DEEP_RESEARCH_NUM_FETCHES env defaults.
    """
    overrides: dict = {}
    if num_searches is not None:
        overrides["num_searches"] = num_searches
    if num_fetches is not None:
        overrides["num_fetches"] = num_fetches
    return deep_research(
        topic, client=client, model=model, embedder=embedder,
        max_rounds=max_rounds, **overrides,
    )


# ── idle learner loop ─────────────────────────────────────────────────────────

# Floor on how long chat must have been quiet before the loop will even
# consider studying — a safety minimum, not the primary gate anymore (see
# idle_learner_loop's docstring). Kept so a proactive "resting" flag that
# flips true immediately after a rest message can't trigger research in the
# same instant the rest note goes out.
IDLE_LEARN_SECONDS = int(os.getenv("IDLE_LEARN_SECONDS", 1800))
IDLE_LEARNER_CHECK_INTERVAL = float(os.getenv("IDLE_LEARNER_CHECK_INTERVAL", 300))


def idle_learner_loop(owner, check_interval: float = IDLE_LEARNER_CHECK_INTERVAL) -> None:
    """Background autonomous learning loop.

    Intended to be launched as a daemon thread by the owning AikoThink
    instance at startup:

        threading.Thread(target=learn.idle_learner_loop, args=(owner,), daemon=True).start()

    `owner` supplies everything this loop reads but doesn't own: chat
    history (owner._history / owner._history_lock), idle bookkeeping
    (owner._last_chat_time), the TTS handle (owner._speak), the LLM
    client/model (owner._client / owner._llm_model), the shared embedder
    (owner._memorize._mem._embedder), and the memory store (owner._memorize).
    This function owns the decision logic — what counts as a study-worthy
    topic, and dedup against already-learned topics — not the bookkeeping
    itself.

    "Idle" here is intentionally NOT just "no chat messages for a while."
    It means the proactive check-in system (see config/proactive.yaml —
    PROACTIVE_REST_AFTER_SECONDS / PROACTIVE_REST_MESSAGE) has already run
    its own idle cycle to completion and gone quiet to rest. Studying
    during an active proactive check-in window would compete with those
    check-ins for the same idle time; studying only once Aiko has already
    decided to rest keeps the two systems from stepping on each other.

    This loop looks for that signal on `owner` in this order:
      1. owner.is_proactive_resting() — a callable, if present.
      2. owner._proactive_resting — a plain bool attribute, if present.
    AikoThink (cognition/think.py) now implements is_proactive_resting() as part
    of its own proactive-checkin state machine, driven by
    config/proactive.yaml. If owner exposes neither, this loop falls back
    to IDLE_LEARN_SECONDS alone so it still degrades to the old
    timer-only behavior rather than never firing.

    Every check_interval: check idle/resting conditions, and if met, pick
    the most recent substantial user message as a topic, skip it if
    already studied this session or already in memory, then run
    quick_studying on it and persist the distilled result.
    """
    while True:
        time.sleep(check_interval)

        chat_idle_long_enough = (time.time() - owner._last_chat_time) >= IDLE_LEARN_SECONDS

        is_resting_fn = getattr(owner, "is_proactive_resting", None)
        if callable(is_resting_fn):
            proactive_resting = bool(is_resting_fn())
        elif hasattr(owner, "_proactive_resting"):
            proactive_resting = bool(getattr(owner, "_proactive_resting"))
        else:
            # No proactive-state hook found on owner — fall back to the
            # old behavior (chat-idle timer only) rather than never firing.
            proactive_resting = True

        if not (chat_idle_long_enough and proactive_resting):
            continue

        if owner._speak and owner._speak.is_playing():
            continue

        log.info("[learner] Aiko is idle and resting. Starting autonomous learning...")
        study_uid = owner._memorize.get_user_id()

        try:
            with owner._history_lock:
                candidates = [
                    m["content"] for m in owner._history
                    if m["role"] == "user" and len(m["content"].split()) > 3
                ]

            if not candidates:
                log.info("[learner] skipped: no eligible candidate topics in current history.")
                continue

            topic = candidates[-1]  # simplistic: look at last user query
            learned_tag = f"[self-learned:{topic}]"
            if any(learned_tag in (m.get("content") or "") for m in owner._history):
                log.info("[learner] skipped: topic already tagged as learned this session: %r", topic)
                continue
            existing = owner._memorize.search(learned_tag, limit=1)
            if existing:
                log.info(
                    "[learner] skipped: topic already found in memory (closest match: %r), topic=%r",
                    (existing[0].get("memory") or existing[0].get("text") or "")[:120],
                    topic,
                )
                continue

            log.info("[learner] researching topic: %r", topic)
            result = quick_studying(
                topic,
                client=owner._client,
                model=owner._llm_model,
                embedder=owner._memorize._mem._embedder,
            )

            owner._memorize.add([
                {"role": "system", "content": learned_tag},
                {"role": "assistant", "content": result[:800]},
            ])
            log.info("[learner] learned about %r — summary: %s", topic, result[:300].replace("\n", " "))
        except Exception as e:
            log.error(f"[learner] Autonomous learning failed: {e}")


# ── deep_studying config ──────────────────────────────────────────────────────

DEEP_STUDY_MAX_ITERATIONS = int(os.getenv("DEEP_STUDY_MAX_ITERATIONS", 60))
DEEP_STUDY_RESULTS_PER_QUERY = int(os.getenv("DEEP_STUDY_RESULTS_PER_QUERY", DEEP_SEARCH_MAX_RESULTS))
DEEP_STUDY_FETCH_TOP = int(os.getenv("DEEP_STUDY_FETCH_TOP", 3))
DEEP_STUDY_MAX_CHARS_PER_PAGE = int(os.getenv("DEEP_STUDY_MAX_CHARS_PER_PAGE", 2000))
DEEP_STUDY_CHUNK_CHARS = int(os.getenv("DEEP_STUDY_CHUNK_CHARS", CONDENSE_CHUNK_CHARS))
DEEP_STUDY_MIN_SCORE = float(os.getenv("DEEP_STUDY_MIN_SCORE", CONDENSE_MIN_SCORE))
DEEP_STUDY_TOP_K_FOR_DISTILLATION = int(os.getenv("DEEP_STUDY_TOP_K_FOR_DISTILLATION", 40))
DEEP_STUDY_SEED_QUERIES = int(os.getenv("DEEP_STUDY_SEED_QUERIES", 6))

# Per-host politeness: don't hit the same site more than once every N
# seconds across the whole session, regardless of how many iterations want
# to fetch from it. This matters at 50-100 iterations in a way it doesn't
# at deep_research's 1-3 rounds — a topic can easily surface the same
# authoritative domain (e.g. a project's own GitHub) repeatedly.
DEEP_STUDY_PER_HOST_MIN_INTERVAL = float(os.getenv("DEEP_STUDY_PER_HOST_MIN_INTERVAL", 3.0))

DEEP_STUDY_SCRATCH_DIR = os.getenv(
    "DEEP_STUDY_SCRATCH_DIR",
    str(Path.home() / ".aiko" / "dream"),
)

DEEP_STUDY_DECISION_MAX_TOKENS = int(os.getenv("DEEP_STUDY_DECISION_MAX_TOKENS", 250))
DEEP_STUDY_SYNTHESIS_MAX_TOKENS = int(os.getenv("DEEP_STUDY_SYNTHESIS_MAX_TOKENS", 900))


class _HostRateLimiter:
    """Enforce a minimum interval between requests to the same host. Cheap,
    in-process, session-scoped — not a general-purpose rate limiter, just
    enough to keep 50-100 iterations from looking like a scraper hammering
    the same domain."""

    def __init__(self, min_interval: float):
        self._min_interval = max(0.0, min_interval)
        self._last_call: dict[str, float] = {}
        self._lock = threading.Lock()

    def wait(self, url: str) -> None:
        host = urlparse(url).hostname or ""
        if not host or self._min_interval <= 0:
            return
        with self._lock:
            last = self._last_call.get(host)
            now = time.monotonic()
            if last is not None:
                elapsed = now - last
                remaining = self._min_interval - elapsed
            else:
                remaining = 0.0
            self._last_call[host] = max(now, (last or 0.0) + self._min_interval)
        if remaining > 0:
            time.sleep(remaining)


class _ScratchStore:
    """Disk-backed scratch store for one deep_studying call.

    Plain sqlite3, not sqlite-vec — deliberately. At the scale this runs at
    (bounded by DEEP_STUDY_MAX_ITERATIONS * a per-iteration chunk cap), the
    total chunk count stays in the thousands, not millions. A linear scan
    with pure-Python cosine (same primitive already used elsewhere) is fast
    enough at that scale and avoids taking on a sqlite-vec dependency for a
    file that gets deleted at the end of the call anyway. If deep_studying's
    scope grows to genuinely large corpora later, THAT would be the point to
    revisit this as an actual vector index — not before.

    Embeddings are stored as JSON-encoded float lists. Simple, inspectable
    with `sqlite3 scratch.db` if something goes wrong mid-session, and
    avoids a numpy/struct-packing dependency for what is, again, a file
    that's deleted when the call returns.
    """

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._lock = threading.Lock()
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS pages (
                url TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                fetched_at REAL NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT NOT NULL,
                chunk TEXT NOT NULL,
                chunk_hash TEXT NOT NULL UNIQUE,
                embedding TEXT,
                query TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def has_page(self, url: str) -> bool:
        with self._lock:
            row = self._conn.execute("SELECT 1 FROM pages WHERE url = ?", (url,)).fetchone()
        return row is not None

    def get_page_text(self, url: str) -> str | None:
        with self._lock:
            row = self._conn.execute("SELECT text FROM pages WHERE url = ?", (url,)).fetchone()
        return row[0] if row else None

    def save_page(self, url: str, text: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO pages (url, text, fetched_at) VALUES (?, ?, ?)",
                (url, text, time.time()),
            )
            self._conn.commit()

    def save_chunk(self, url: str, chunk: str, chunk_hash: str, embedding: list[float] | None, query: str) -> None:
        embedding_json = json.dumps(embedding) if embedding is not None else None
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT INTO chunks (url, chunk, chunk_hash, embedding, query) VALUES (?, ?, ?, ?, ?)",
                    (url, chunk, chunk_hash, embedding_json, query),
                )
                self._conn.commit()
            except sqlite3.IntegrityError:
                pass  # duplicate chunk_hash — already have this exact chunk, skip

    def all_chunks(self, limit: int = 0, desc: bool = False) -> list[tuple[str, str, list[float] | None]]:
        sql = "SELECT url, chunk, embedding FROM chunks"
        if desc:
            sql += " ORDER BY id DESC"
        if limit > 0:
            sql += f" LIMIT {limit}"
        with self._lock:
            rows = self._conn.execute(sql).fetchall()
        out = []
        for url, chunk, embedding_json in rows:
            embedding = json.loads(embedding_json) if embedding_json else None
            out.append((url, chunk, embedding))
        return out

    def chunk_count(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        return row[0] if row else 0

    def close_and_delete(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
        try:
            self.path.unlink(missing_ok=True)
        except Exception as e:
            log.warning("deep_studying: failed to delete scratch store %s: %s", self.path, e)


def _seed_subqueries(topic: str, client, model: str, n: int) -> list[str]:
    """Ask the model to break a broad topic into n concrete starting
    queries. Falls back to just the topic itself if no client/model is
    available or the call fails — deep_studying still works, it just
    explores less broadly on iteration 1."""
    if not (client and model):
        return [topic]
    prompt = (
        "Break the following research topic into concrete, differently-angled "
        "search queries suitable for a web search engine. Return ONLY compact "
        "JSON: {\"queries\": [string, ...]}.\n"
        f"Return at most {n} queries. Prefer specific technical or factual "
        "angles over restating the topic broadly.\n\n"
        f"Topic: {topic}"
    )
    decision = _ask_llm_json(client, model, prompt, DEEP_STUDY_DECISION_MAX_TOKENS)
    if not decision:
        return [topic]
    queries = [str(q).strip() for q in decision.get("queries", []) if str(q).strip()]
    return queries[:n] if queries else [topic]


def _next_subquery(
    topic: str,
    client,
    model: str,
    explored: list[str],
    evidence_preview: str,
) -> dict | None:
    """Ask the model for the next sub-query to explore, or a stop signal.
    Same continue/next_query/reason shape as deep_research's decision call,
    but scoped to "what haven't we covered yet about this broad topic" over
    many more iterations rather than "is this one question answered yet".
    """
    prompt = (
        "You are conducting an extended, multi-session self-study on a broad "
        "topic during idle time. Given what has already been explored and the "
        "evidence gathered so far, decide the next concrete sub-query to "
        "explore, or whether the topic has been covered well enough to stop.\n"
        "Return ONLY compact JSON: "
        "{\"continue\": bool, \"next_query\": string, \"reason\": string}.\n"
        "Set continue=false only when further searching is very unlikely to "
        "surface meaningfully new angles on the topic.\n"
        "next_query should explore a genuinely new angle, not restate a "
        "prior query.\n\n"
        f"Overall topic: {topic}\n\n"
        f"Sub-queries already explored: {explored}\n\n"
        f"Evidence gathered so far (preview):\n{evidence_preview}"
    )
    return _ask_llm_json(client, model, prompt, DEEP_STUDY_DECISION_MAX_TOKENS)


def _hash_chunk(chunk: str) -> str:
    return hashlib.sha1(chunk.strip().lower().encode("utf-8", "ignore")).hexdigest()


def _fetch_one(url: str, rate_limiter: _HostRateLimiter, store: _ScratchStore, max_chars: int) -> str | None:
    """Fetch a URL respecting the scratch store's own dedup (skip re-fetch
    if already in this session) and the per-host rate limiter. Returns the
    page text, or None on failure/skip."""
    if store.has_page(url):
        return store.get_page_text(url)

    parsed = urlparse(url)
    if not parsed.hostname or _is_private_or_local_host(parsed.hostname):
        return None

    rate_limiter.wait(url)
    text = web_fetch(url, max_chars=max_chars, use_cache=False)  # scratch store IS the cache here
    if text.startswith("[fetch failed"):
        return None
    store.save_page(url, text)
    return text


def _score_and_store_page(
    url: str,
    text: str,
    query: str,
    embedder,
    query_vec,
    use_embedder: bool,
    store: _ScratchStore,
) -> None:
    for chunk in reason.chunk_text(text, DEEP_STUDY_CHUNK_CHARS):
        chunk_hash = _hash_chunk(chunk)
        embedding = None
        if use_embedder:
            batch = reason.embed_batch_or_none(embedder, [chunk])
            if batch is not None and len(batch) > 0:
                embedding = list(batch[0])
            else:
                try:
                    embedding = embedder.embed_query(chunk)
                except Exception:
                    embedding = None
        store.save_chunk(url, chunk, chunk_hash, embedding, query)


def _distill(
    store: _ScratchStore,
    topic: str,
    embedder,
    client,
    model: str,
    top_k: int,
    min_score: float,
) -> tuple[str, list[tuple[float, str, str]]]:
    """Rank all accumulated chunks against the ORIGINAL topic (not each
    sub-query — the point of distillation is relevance to the overall
    subject Aiko was asked to study), dedup, and optionally run one
    synthesis pass. Returns (distilled_text, ranked_chunks) so a caller can
    also inspect/store the raw ranked evidence if their knowledge-graph
    write path wants source-level granularity rather than just prose.
    """
    all_chunks = store.all_chunks()
    if not all_chunks:
        return f"[deep_studying: no evidence gathered for topic: {topic}]", []

    use_embedder = embedder is not None and hasattr(embedder, "embed_query")
    topic_vec = None
    if use_embedder:
        try:
            topic_vec = embedder.embed_query(topic)
        except Exception:
            use_embedder = False

    scored: list[tuple[float, str, str]] = []
    for url, chunk, embedding in all_chunks:
        if use_embedder and topic_vec is not None and embedding is not None:
            try:
                score = _cosine(topic_vec, embedding)
            except Exception:
                score = reason.keyword_overlap_score(topic, chunk)
        else:
            score = reason.keyword_overlap_score(topic, chunk)
        scored.append((score, url, chunk))

    ranked = sorted(
        (item for item in scored if item[0] >= min_score),
        key=lambda item: item[0],
        reverse=True,
    )[:top_k]

    if not ranked:
        return (
            f"[deep_studying: {len(all_chunks)} chunk(s) gathered for topic: {topic}, "
            "but none cleared the relevance threshold against the original topic — "
            "sub-queries may have drifted; do not fabricate findings]"
        ), []

    evidence_bundle = "\n\n".join(
        f"[source: {url} | relevance: {score:.2f}]\n{chunk}" for score, url, chunk in ranked
    )

    if client and model:
        prompt = (
            "Synthesize the following research evidence, gathered over an "
            "extended self-study session, into a compact set of atomic facts "
            "and takeaways about the topic below. Prefer short, standalone "
            "statements over narrative prose — this output will be stored as "
            "durable knowledge, not read as an essay. Note unresolved "
            "questions or conflicting information explicitly. Do not invent "
            "facts not present in the evidence.\n\n"
            f"Topic: {topic}\n\n"
            f"Evidence ({len(ranked)} excerpt(s) from {len(set(u for _, u, _ in ranked))} source(s)):\n"
            f"{evidence_bundle}"
        )
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                max_tokens=DEEP_STUDY_SYNTHESIS_MAX_TOKENS,
                temperature=0.2,
            )
            synthesis = (resp.choices[0].message.content or "").strip()
            if synthesis:
                return synthesis, ranked
        except Exception as e:
            log.warning("deep_studying: synthesis call failed, returning raw evidence: %s", e)

    return evidence_bundle, ranked


def deep_studying(
    topic: str,
    client=None,
    model: str | None = None,
    embedder=None,
    max_iterations: int = DEEP_STUDY_MAX_ITERATIONS,
    fetch_top: int = DEEP_STUDY_FETCH_TOP,
    results_per_query: int = DEEP_STUDY_RESULTS_PER_QUERY,
    per_host_min_interval: float = DEEP_STUDY_PER_HOST_MIN_INTERVAL,
    on_distilled=None,
    session_id: str | None = None,
    stop_event: threading.Event | None = None,
) -> str:
    """Autonomous, extended research on a single topic, meant for genuine
    idle time (overnight dream cycles, or a scheduled deep-study window —
    see register_deep_study_handlers), not interactive use.

    Runs up to max_iterations search-and-fetch rounds against one topic,
    expanding into new sub-queries each round (LLM-driven if client/model
    are given, else it explores the seed queries and stops). Every fetched
    page and scored chunk is persisted to a disk-backed scratch SQLite store
    for the duration of this call, so revisiting an already-fetched page
    across iterations costs nothing — no re-fetch, no re-embed. The store
    is deleted before this function returns.

    Stops early when: the model signals the topic is sufficiently covered,
    no new sub-query can be found, max_iterations is reached, OR
    stop_event is set (checked at the top of every iteration and again
    right before asking for the next sub-query). stop_event is how a
    scheduled window's "stop" edge (e.g. 18:00 on weekdays) interrupts a
    long-running session gracefully: whatever has been gathered so far is
    still distilled and handed to on_distilled before returning, it just
    won't start a new iteration. There is no hard external rate-limit
    handling here beyond the per-host minimum interval — if your search
    backend (SearXNG) itself starts throttling, _web_search_raw's error
    path will surface that as an empty round and this loop will simply
    stop, same as deep_research does today.

    on_distilled, if given, is called as on_distilled(topic, distilled_text,
    ranked_chunks) after distillation, BEFORE the scratch store is deleted —
    this is the hook for writing into your actual knowledge graph / MSB.
    This module doesn't know that schema, so it doesn't write to it
    directly; it hands you the distilled output and the source-level ranked
    chunks (score, url, chunk_text) and lets the caller decide the write.

    Returns the distilled text (same thing on_distilled receives).
    """
    if not topic or not topic.strip():
        return "[deep_studying failed: empty topic]"
    topic = topic.strip()

    session_id = session_id or uuid.uuid4().hex[:12]
    scratch_path = Path(DEEP_STUDY_SCRATCH_DIR) / f"{session_id}.db"
    store = _ScratchStore(scratch_path)
    rate_limiter = _HostRateLimiter(per_host_min_interval)

    use_embedder = embedder is not None and hasattr(embedder, "embed_query")
    adaptive = client is not None and bool(model)

    log.info("[deep_studying] starting session=%s topic=%r max_iterations=%d", session_id, topic, max_iterations)

    try:
        pending_queries: list[str] = _seed_subqueries(topic, client, model, DEEP_STUDY_SEED_QUERIES) if adaptive else [topic]
        explored: list[str] = []
        seen_urls: set[str] = set()
        iterations_run = 0

        while pending_queries and iterations_run < max_iterations:
            if stop_event is not None and stop_event.is_set():
                log.info("[deep_studying] session=%s stop_event set — winding down after %d iteration(s).", session_id, iterations_run)
                break

            current_query = pending_queries.pop(0)
            if current_query in explored:
                continue
            explored.append(current_query)
            iterations_run += 1

            results, error = _web_search_raw(current_query, results_per_query, pageno=1)
            if error:
                log.info("[deep_studying] search failed for %r: %s", current_query, error)
                continue
            if not results:
                continue

            new_urls = [
                r["url"].strip() for r in results
                if r.get("url") and r["url"].strip() not in seen_urls
            ][:fetch_top]
            seen_urls.update(new_urls)

            query_vec = None
            if use_embedder:
                try:
                    query_vec = embedder.embed_query(current_query)
                except Exception:
                    pass

            for url in new_urls:
                if stop_event is not None and stop_event.is_set():
                    break
                text = _fetch_one(url, rate_limiter, store, DEEP_STUDY_MAX_CHARS_PER_PAGE)
                if not text:
                    continue
                _score_and_store_page(url, text, current_query, embedder, query_vec, use_embedder, store)

            log.info(
                "[deep_studying] iteration %d/%d query=%r chunks_so_far=%d",
                iterations_run, max_iterations, current_query, store.chunk_count(),
            )

            if not adaptive:
                continue  # no model to pick new sub-queries — just work through the seed list

            if stop_event is not None and stop_event.is_set():
                break

            if not pending_queries and iterations_run < max_iterations:
                # Ask for the next sub-query only once the seed queue is
                # drained, so seeded breadth always gets explored before
                # the adaptive loop starts narrowing further.
                evidence_preview = ""
                all_chunks = store.all_chunks(limit=20, desc=True)
                if all_chunks:
                    preview_chunks = [c for _, c, _ in all_chunks]
                    evidence_preview = "\n---\n".join(preview_chunks)[:4000]

                decision = _next_subquery(topic, client, model, explored, evidence_preview)
                if not decision or not decision.get("continue"):
                    log.info("[deep_studying] model signaled stop: %s", (decision or {}).get("reason"))
                    break
                next_query = str(decision.get("next_query") or "").strip()
                if not next_query or next_query in explored:
                    break
                pending_queries.append(next_query)

        distilled, ranked_chunks = _distill(
            store, topic, embedder, client, model,
            DEEP_STUDY_TOP_K_FOR_DISTILLATION, DEEP_STUDY_MIN_SCORE,
        )

        log.info(
            "[deep_studying] finished session=%s iterations=%d chunks=%d sources=%d",
            session_id, iterations_run, store.chunk_count(),
            len(set(u for _, u, _ in ranked_chunks)),
        )

        if on_distilled is not None:
            try:
                on_distilled(topic, distilled, ranked_chunks)
            except Exception as e:
                log.error("[deep_studying] on_distilled hook failed: %s", e)

        return distilled

    finally:
        store.close_and_delete()


# ── scheduled deep-study window ───────────────────────────────────────────────
# Ties deep_studying into system.schedule's handler-based jobs so it only
# runs inside a configured wall-clock window (see
# schedule.ensure_deep_study_window_jobs: weekdays 05:00-18:00, weekends
# 05:00-10:00 by default) instead of purely on the idle-learner's own
# opportunistic timing. At most one window session runs at a time; the
# *_stop handler signals it to wind down rather than killing the thread,
# so whatever's been gathered so far still gets distilled and persisted.

def _pick_window_topic(memorize) -> str | None:
    """Very simple default topic picker for scheduled deep-study windows.

    ADAPT THIS: this module deliberately doesn't know GRACE/Aiko's actual
    curiosity/topic-selection policy, so it just checks for an optional
    `pending_deep_study_topic` attribute on the memory store (a place your
    own code — e.g. reflection, or a `/deep-study <topic>` command — could
    stash a topic ahead of time) and otherwise declines to start rather
    than guessing at a topic. Replace with real topic-selection logic
    (e.g. pull the least-covered item from GRACE's knowledge graph, or the
    top item in a backlog) whenever that's ready.
    """
    return getattr(memorize, "pending_deep_study_topic", None) or None


class _DeepStudySessionManager:
    """Tracks at most one running deep_studying session at a time, so the
    scheduled start/stop handlers (see the four jobs seeded by
    schedule.ensure_deep_study_window_jobs) can launch and gracefully
    interrupt it without deep_studying itself knowing anything about
    schedules, windows, or wall-clock time.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None

    def is_running(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def start(self, memorize, client=None, model=None, topic: str | None = None) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                log.info("[deep_study_window] start requested but a session is already running — skipping.")
                return
            stop_event = threading.Event()
            self._stop_event = stop_event

        study_topic = topic or _pick_window_topic(memorize)
        if not study_topic:
            log.info("[deep_study_window] no topic available to study — skipping window start.")
            return

        def _run() -> None:
            log.info("[deep_study_window] starting deep_studying for topic=%r", study_topic)
            try:
                embedder = getattr(getattr(memorize, "_mem", None), "_embedder", None)
                study_uid = memorize.get_user_id()

                def _on_distilled(distilled_topic, text, ranked_chunks):
                    memorize.add([
                        {"role": "system", "content": f"[deep-studied:{distilled_topic}]"},
                        {"role": "assistant", "content": text[:4000]},
                    ], user_id=study_uid)

                distilled = deep_studying(
                    study_topic,
                    client=client,
                    model=model,
                    embedder=embedder,
                    stop_event=stop_event,
                    on_distilled=_on_distilled,
                )
                log.info("[deep_study_window] finished — %s", distilled[:200].replace("\n", " "))
            except Exception as e:
                log.error("[deep_study_window] session failed: %s", e)

        thread = threading.Thread(target=_run, name="aiko-deep-study-window", daemon=True)
        with self._lock:
            self._thread = thread
        thread.start()

    def stop(self, memorize=None) -> None:
        with self._lock:
            stop_event = self._stop_event
        if stop_event is not None:
            log.info("[deep_study_window] stop requested — signaling session to wind down.")
            stop_event.set()
        # Deliberately no join() here — this runs on the scheduler thread,
        # and deep_studying may be mid-fetch; let it exit on its own next
        # loop check (see stop_event checks in deep_studying) rather than
        # blocking the scheduler thread until it does.


_deep_study_manager = _DeepStudySessionManager()


def deep_study_window_start(memorize, client=None, model=None) -> None:
    """Registered as schedule.json handler "deep_study_start". Bind
    client/model with functools.partial when registering (see
    register_deep_study_handlers) — the scheduler always calls handlers as
    fn(memorize), so those extra kwargs need to already be bound in."""
    _deep_study_manager.start(memorize, client=client, model=model)


def deep_study_window_stop(memorize) -> None:
    """Registered as schedule.json handler "deep_study_stop"."""
    _deep_study_manager.stop(memorize)


def register_deep_study_handlers(client=None, model=None, timezone: str | None = None) -> None:
    """Call once at app startup to wire deep_studying into the scheduler's
    window (weekdays 05:00-18:00, weekends 05:00-10:00 by default) and seed
    the four recurring jobs that bound it.

        # at startup, after system.schedule's ScheduleRunner exists:
        from memory import learn
        learn.register_deep_study_handlers(client=llm_client, model=llm_model)

    This is now called automatically from system.wakeup.AikoWakeup.boot() —
    see system/wakeup.py — once AikoThink (and therefore its LLM client/model)
    exists, so app authors normally don't need to call this by hand.
    """
    from system import schedule as _schedule

    _schedule.register_system_handler(
        "deep_study_start",
        functools.partial(deep_study_window_start, client=client, model=model),
    )
    _schedule.register_system_handler("deep_study_stop", deep_study_window_stop)
    _schedule.ensure_deep_study_window_jobs(timezone=timezone)
    log.info("[deep_study_window] handlers registered and window jobs ensured.")
