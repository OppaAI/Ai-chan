"""
core/learn.py

Aiko's two research depths, both operating on a topic YOU (or the idle
learner loop) hand her — neither one picks its own topic.

  - quick_studying: a single interactive-scale research pass. Thin alias
    over core.tools.deep_research — same TTL-cached, in-memory, ephemeral
    behavior you already have. Use this for "look this up for me now" and
    for the idle learner's short-idle-gap top-ups.

  - deep_studying: an autonomous, potentially long-running research pass
    meant for genuine idle/overnight time. Runs many iterations against a
    single topic, persists everything to a disk-backed scratch SQLite store
    for the duration of the call (so 50-100 iterations can revisit the same
    pages without re-fetching or re-embedding them), rate-limits itself per
    host so it doesn't hammer the same sites across iterations, and ends by
    distilling the accumulated evidence into compact atomic facts.

    The scratch store is deleted when the call returns — it is NOT a
    persistent index. It exists only to make one long call efficient, the
    same way a browser's per-page DOM exists only for that page's lifetime.
    Long-term persistence is the caller's job: deep_studying returns the
    distilled facts and (optionally) calls an on_distilled hook so whatever
    owns your actual knowledge graph / MSB schema can decide how to store
    them. This file does not know that schema and should not guess at it.

Also owns the idle learner loop (idle_learner_loop) — the background
process that decides, during genuine chat idle time, to pick a recent topic
and study it via quick_studying, then persist the result to memory. The
loop's *scheduling* (what counts as "idle", whether TTS is currently
playing) is read directly off the owning AikoThink instance passed in;
this module doesn't own that bookkeeping, only what happens once idle
conditions are met.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse

from core.tools import (
    _ask_llm_json,
    _chunk_text,
    _cosine,
    _embed_batch,
    _is_private_or_local_host,
    _keyword_overlap_score,
    _score_chunk,
    _web_search_raw,
    deep_research,
    web_fetch,
    CONDENSE_CHUNK_CHARS,
    CONDENSE_MIN_SCORE,
    DEEP_SEARCH_MAX_RESULTS,
)

from core.log import get_logger

log = get_logger(__name__)

# ── quick_studying ────────────────────────────────────────────────────────────

def quick_studying(
    topic: str,
    client=None,
    model: str | None = None,
    embedder=None,
    max_rounds: int = 3,
) -> str:
    """Interactive-depth research on a topic. This is exactly deep_research —
    same TTL cache, same in-memory ephemeral scoring, same single-call cost
    model. Named separately so call sites (the idle learner's short-gap
    path, or a direct /research command) read as "which depth am I asking
    for" rather than exposing tools.py internals at every call site.
    """
    return deep_research(topic, client=client, model=model, embedder=embedder, max_rounds=max_rounds)


# ── idle learner loop ─────────────────────────────────────────────────────────

# How long (seconds) chat must be idle before the loop will pick a topic and
# study it. Owned here now, not in think.py — this is a learn.py policy
# about when autonomous learning is worth doing, not a chat-facade concern.
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

    Every iteration: sleep, check idle conditions, and if idle for long
    enough, pick the most recent substantial user message as a topic,
    skip it if already studied this session or already in memory, then
    run quick_studying on it and persist the distilled result.
    """
    while True:
        time.sleep(check_interval)
        if time.time() - owner._last_chat_time < IDLE_LEARN_SECONDS:
            continue  # user has been active recently, don't interrupt

        if owner._speak and owner._speak.is_playing():
            continue

        log.info("[learner] Aiko is idle. Starting autonomous learning...")
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
    with pure-Python cosine (same primitive tools.py already uses) is fast
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

    def all_chunks(self) -> list[tuple[str, str, list[float] | None]]:
        with self._lock:
            rows = self._conn.execute("SELECT url, chunk, embedding FROM chunks").fetchall()
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
    import hashlib
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
    for _, chunk in _chunk_text(url, text, DEEP_STUDY_CHUNK_CHARS):
        chunk_hash = _hash_chunk(chunk)
        embedding = None
        if use_embedder:
            batch = _embed_batch(embedder, [chunk])
            if batch:
                embedding = batch[0]
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
                score = _keyword_overlap_score(topic, chunk)
        else:
            score = _keyword_overlap_score(topic, chunk)
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
) -> str:
    """Autonomous, extended research on a single topic, meant for genuine
    idle time (overnight dream cycles), not interactive use.

    Runs up to max_iterations search-and-fetch rounds against one topic,
    expanding into new sub-queries each round (LLM-driven if client/model
    are given, else it explores the seed queries and stops). Every fetched
    page and scored chunk is persisted to a disk-backed scratch SQLite store
    for the duration of this call, so revisiting an already-fetched page
    across iterations costs nothing — no re-fetch, no re-embed. The store
    is deleted before this function returns.

    Stops early when: the model signals the topic is sufficiently covered,
    no new sub-query can be found, or max_iterations is reached. There is no
    hard external rate-limit handling here beyond the per-host minimum
    interval — if your search backend (SearXNG) itself starts throttling,
    _web_search_raw's error path will surface that as an empty round and
    this loop will simply stop, same as deep_research does today.

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

            if not pending_queries and iterations_run < max_iterations:
                # Ask for the next sub-query only once the seed queue is
                # drained, so seeded breadth always gets explored before
                # the adaptive loop starts narrowing further.
                evidence_preview = ""
                all_chunks = store.all_chunks()
                if all_chunks:
                    preview_chunks = [c for _, c, _ in all_chunks[-20:]]
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
