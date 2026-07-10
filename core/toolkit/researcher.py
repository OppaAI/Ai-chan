"""
core/toolkit/researcher.py

Web search and page extraction tools.

This module provides web research capabilities for agentic workflows:

  - web_search()    — search via configured SearXNG instance
  - web_fetch()     — fetch and extract text from URLs
  - deep_search()   — snippet-only search pass for task-mode workflows
  - deep_research() — adaptive fetched-source research with synthesis

Requires a running SearXNG instance (SEARXNG_URL env var).
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import ipaddress
import json
import os
import re
import socket
import time
import threading
from urllib.parse import urlparse

import importlib
import importlib.util

import numpy as np

from core.log import get_logger
from core import reason

log = get_logger(__name__)

SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:8888")
MAX_RESULTS = int(os.getenv("SEARXNG_MAX_RESULTS", 5))

# -- deep_search (fixed, non-adaptive snippet search pass) --
# Task-mode deep_search is intentionally snippet-only. It uses the same raw
# SearXNG primitive as web_search, but is exposed to agentic workflows while
# web_search/web_fetch remain chat-mode primitives.
DEEP_SEARCH_NUM_SEARCHES = int(os.getenv("DEEP_SEARCH_NUM_SEARCHES", 1))
DEEP_SEARCH_NUM_FETCHES = int(os.getenv("DEEP_SEARCH_NUM_FETCHES", 0))  # legacy override; keep 0 for snippet-only deep_search
DEEP_SEARCH_MAX_CHARS_PER_PAGE = int(os.getenv("DEEP_SEARCH_MAX_CHARS_PER_PAGE", 2000))
DEEP_SEARCH_MAX_WORKERS = int(os.getenv("DEEP_SEARCH_MAX_WORKERS", 4))

# -- deep_research (adaptive fetched-source research) --
# Deep research uses search only to discover URLs, then fetches pages and
# condenses/synthesizes evidence. It has its own fetch knobs so deep_search can
# stay snippet-only. These are read once as module-level DEFAULTS; deep_research()
# itself now accepts num_searches/num_fetches/max_chars_per_page as real
# function args (see below) so a caller — e.g. core.learn.quick_studying —
# can override per-call instead of only ever getting these env defaults.
DEEP_RESEARCH_NUM_SEARCHES = int(os.getenv("DEEP_RESEARCH_NUM_SEARCHES", os.getenv("DEEP_SEARCH_NUM_SEARCHES", 1)))
DEEP_RESEARCH_NUM_FETCHES = int(os.getenv("DEEP_RESEARCH_NUM_FETCHES", 2))
DEEP_RESEARCH_MAX_CHARS_PER_PAGE = int(os.getenv("DEEP_RESEARCH_MAX_CHARS_PER_PAGE", os.getenv("DEEP_SEARCH_MAX_CHARS_PER_PAGE", 2000)))
DEEP_RESEARCH_MAX_ROUNDS = int(os.getenv("DEEP_RESEARCH_MAX_ROUNDS", 3))
DEEP_RESEARCH_EVIDENCE_CHARS_FOR_DECISION = int(os.getenv("DEEP_RESEARCH_EVIDENCE_CHARS_FOR_DECISION", 6000))
DEEP_RESEARCH_EVIDENCE_CHARS_FOR_SYNTHESIS = int(os.getenv("DEEP_RESEARCH_EVIDENCE_CHARS_FOR_SYNTHESIS", 8000))
DEEP_RESEARCH_DECISION_MAX_TOKENS = int(os.getenv("DEEP_RESEARCH_DECISION_MAX_TOKENS", 200))
DEEP_RESEARCH_SYNTHESIS_MAX_TOKENS = int(os.getenv("DEEP_RESEARCH_SYNTHESIS_MAX_TOKENS", 600))

# -- in-memory evidence condensation (numpy-vectorized relevance filtering) --
# A FILTER, not a rewrite: chunks are scored for relevance and either kept
# verbatim or dropped entirely. Summarization only happens later, in
# deep_research's separate LLM synthesis call.
CONDENSE_CHUNK_CHARS = int(os.getenv("CONDENSE_CHUNK_CHARS", 500))
CONDENSE_TOP_K = int(os.getenv("CONDENSE_TOP_K", 8))
CONDENSE_MIN_SCORE = float(os.getenv("CONDENSE_MIN_SCORE", 0.15))
# Caps embedding calls PER fetch pipeline invocation (per deep_search call,
# i.e. per round) — not a lifetime cap.
CONDENSE_MAX_CHUNKS_TO_SCORE = int(os.getenv("CONDENSE_MAX_CHUNKS_TO_SCORE", 60))

# -- web_fetch download guard --
WEB_FETCH_MAX_DOWNLOAD_BYTES = int(os.getenv("WEB_FETCH_MAX_DOWNLOAD_BYTES", 5_000_000))
WEB_FETCH_TIMEOUT_SECONDS = int(os.getenv("WEB_FETCH_TIMEOUT_SECONDS", 8))
WEB_FETCH_USER_AGENT = os.getenv(
    "WEB_FETCH_USER_AGENT",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36",
)

# -- short-lived in-process query cache --
CACHE_TTL_SECONDS = int(os.getenv("TOOLS_CACHE_TTL_SECONDS", 900))  # 15 min
CACHE_MAX_ENTRIES = int(os.getenv("TOOLS_CACHE_MAX_ENTRIES", 256))

_cache_lock = threading.Lock()
_search_cache: dict[str, tuple[float, list[dict]]] = {}
_fetch_cache: dict[str, tuple[float, str]] = {}


def _cache_get(cache: dict, key: str):
    with _cache_lock:
        entry = cache.get(key)
        if entry is None:
            return None
        ts, value = entry
        if time.monotonic() - ts > CACHE_TTL_SECONDS:
            cache.pop(key, None)
            return None
        return value


def _cache_set(cache: dict, key: str, value) -> None:
    with _cache_lock:
        if len(cache) >= CACHE_MAX_ENTRIES:
            oldest_key = min(cache, key=lambda k: cache[k][0], default=None)
            if oldest_key is not None:
                cache.pop(oldest_key, None)
        cache[key] = (time.monotonic(), value)


def _web_search_raw(query: str, max_results: int, pageno: int = 1) -> tuple[list[dict] | None, str | None]:
    """Low-level SearXNG call returning (results, error). Cached in-process
    for CACHE_TTL_SECONDS keyed on (query, max_results, pageno)."""
    cache_key = f"{query}|{max_results}|{pageno}"
    cached = _cache_get(_search_cache, cache_key)
    if cached is not None:
        return cached, None

    if importlib.util.find_spec("requests") is None:
        return None, "[search failed: requests is not installed]"
    requests = importlib.import_module("requests")
    try:
        response = requests.get(
            f"{SEARXNG_URL}/search",
            params={"q": query, "format": "json", "pageno": pageno},
            timeout=8,
        )
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        return None, f"[search failed: {e}]"
    except ValueError:
        return None, "[search failed: invalid JSON response]"

    results = data.get("results", [])[:max_results]
    _cache_set(_search_cache, cache_key, results)
    return results, None


def web_search(query: str, max_results: int = MAX_RESULTS) -> str:
    """Search the web via SearXNG and return compact numbered results."""
    results, error = _web_search_raw(query, max_results, pageno=1)
    if error:
        return error
    if not results:
        return f"[no results found for: {query}]"

    lines = [f"[Web search results for: {query}]"]
    for i, result in enumerate(results, 1):
        title = result.get("title", "").strip()
        url = result.get("url", "").strip()
        content = result.get("content", "").strip()
        lines.append(f"{i}. {title}\n   {url}\n   {content}")

    return "\n\n".join(lines)


def _is_private_or_local_host(hostname: str) -> bool:
    try:
        for _family, _type, _proto, _canonname, sockaddr in socket.getaddrinfo(hostname, None):
            raw_ip = sockaddr[0]
            ip = ipaddress.ip_address(raw_ip.split("%")[0])
            if (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip.is_reserved
                or ip.is_multicast
            ):
                return True
        return False
    except OSError:
        return True


def web_fetch(
    url: str,
    max_chars: int = 4000,
    max_download_bytes: int = WEB_FETCH_MAX_DOWNLOAD_BYTES,
    use_cache: bool = True,
) -> str:
    """Fetch a single URL and extract its main article/body text with trafilatura.

    This is the one-and-only "fetch a page" primitive in the toolkit. Both
    the model's direct fetch calls and deep_search's internal pipelined
    page reads route through this function.

    Downloads are streamed and capped at max_download_bytes, aborted
    mid-stream BEFORE trafilatura ever runs extraction and BEFORE max_chars
    truncation — this is what bounds worst-case memory for a single fetch.

    Successful fetches are cached in-process for CACHE_TTL_SECONDS keyed on
    (url, max_chars). Failed fetches are never cached.
    """
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return f"[fetch failed: unsupported URL scheme: {parsed.scheme or 'none'}]"
    if _is_private_or_local_host(parsed.hostname):
        return "[fetch failed: URL host is not allowed]"

    cache_key = f"{url}|{max_chars}"
    if use_cache:
        cached = _cache_get(_fetch_cache, cache_key)
        if cached is not None:
            return cached

    if importlib.util.find_spec("requests") is None:
        return "[fetch failed: requests is not installed]"
    if importlib.util.find_spec("trafilatura") is None:
        return "[fetch failed: trafilatura is not installed]"
    requests = importlib.import_module("requests")
    trafilatura = importlib.import_module("trafilatura")

    try:
        with requests.get(
            url,
            stream=True,
            timeout=WEB_FETCH_TIMEOUT_SECONDS,
            headers={"User-Agent": WEB_FETCH_USER_AGENT},
        ) as resp:
            resp.raise_for_status()

            content_length = resp.headers.get("content-length")
            if content_length is not None:
                try:
                    if int(content_length) > max_download_bytes:
                        return "[fetch failed: page too large]"
                except ValueError:
                    pass

            chunks: list[bytes] = []
            total = 0
            for chunk in resp.iter_content(chunk_size=65536):
                if not chunk:
                    continue
                total += len(chunk)
                if total > max_download_bytes:
                    return "[fetch failed: page exceeded size limit during download]"
                chunks.append(chunk)
            downloaded = b"".join(chunks)
    except requests.exceptions.RequestException as e:
        return f"[fetch failed: {e}]"

    if not downloaded:
        return "[fetch failed: empty response]"

    try:
        text = trafilatura.extract(downloaded, include_links=False, include_tables=False) or ""
    except Exception as e:
        return f"[fetch failed: {e}]"

    result = text[:max_chars] if text else "[fetch failed: no extractable text]"
    if use_cache and text:
        _cache_set(_fetch_cache, cache_key, result)
    return result


def _score_url_chunks(
    url_chunks: list[tuple[str, str]], query: str, embedder, max_chunks_to_score: int,
) -> list[tuple[float, str, str]]:
    """Score (url, chunk) pairs in one batched numpy pass via
    reason instead of a per-chunk Python loop. Falls back to
    keyword overlap per chunk if no embedder is available or embedding
    fails."""
    url_chunks = url_chunks[:max_chunks_to_score]
    if not url_chunks:
        return []
    texts = [c for _u, c in url_chunks]
    if embedder is not None and hasattr(embedder, "embed_query"):
        try:
            query_vec = np.asarray(embedder.embed_query(query), dtype=np.float32)
            chunk_vecs = reason.embed_batch_or_none(embedder, texts)
            if chunk_vecs is not None and chunk_vecs.shape[0] == len(texts):
                scores = reason.batch_cosine_scores(query_vec, chunk_vecs)
                return [(float(scores[i]), url_chunks[i][0], url_chunks[i][1]) for i in range(len(url_chunks))]
        except Exception:
            pass  # fall through to keyword scoring below
    return [(reason.keyword_overlap_score(query, c), u, c) for u, c in url_chunks]


def _fetch_and_score_pipeline(
    urls: list[str],
    query: str,
    embedder,
    max_chars_per_page: int,
    chunk_chars: int = CONDENSE_CHUNK_CHARS,
    max_workers: int = DEEP_SEARCH_MAX_WORKERS,
    max_chunks_to_score: int = CONDENSE_MAX_CHUNKS_TO_SCORE,
) -> tuple[list[tuple[float, str, str]], list[tuple[str, str]], list[tuple[str, str]]]:
    """Fetch multiple URLs concurrently, scoring each page's chunks for
    relevance the moment that page finishes downloading — not after every
    URL has finished.

    Returns (scored_chunks, pages, url_outcomes).
    """
    if not urls:
        return [], [], []

    log.info("[fetch_pipeline] attempting %d url(s): %s", len(urls), urls)

    scored: list[tuple[float, str, str]] = []
    pages: list[tuple[str, str]] = []
    url_outcomes: list[tuple[str, str]] = []
    chunks_scored = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(urls)))) as pool:
        future_to_url = {pool.submit(web_fetch, url, max_chars_per_page): url for url in urls}
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                text = future.result()
            except Exception as e:
                log.warning("[fetch_pipeline] exception fetching %s: %s", url, e)
                url_outcomes.append((url, f"exception: {e}"))
                continue

            if text.startswith("[fetch failed"):
                log.info("[fetch_pipeline] failed %s: %s", url, text)
                url_outcomes.append((url, text))
                continue

            log.info("[fetch_pipeline] fetched %s (%d chars)", url, len(text))
            url_outcomes.append((url, f"ok ({len(text)} chars)"))
            pages.append((url, text))

            remaining_budget = max_chunks_to_score - chunks_scored
            if remaining_budget <= 0:
                continue

            page_chunks = [(url, c) for c in reason.chunk_text(text, chunk_chars)][:remaining_budget]
            page_scored = _score_url_chunks(page_chunks, query, embedder, remaining_budget)
            scored.extend(page_scored)
            chunks_scored += len(page_scored)

    log.info(
        "[fetch_pipeline] done: %d/%d succeeded, %d chunk(s) scored",
        len(pages), len(urls), chunks_scored,
    )
    return scored, pages, url_outcomes


def _format_url_manifest(url_outcomes: list[tuple[str, str]]) -> str:
    if not url_outcomes:
        return "[no URLs attempted]"
    lines = [f"[URL manifest — {len(url_outcomes)} attempted]"]
    for url, status in url_outcomes:
        lines.append(f"- {url} — {status}")
    return "\n".join(lines)


def _finalize_condensed(
    scored_chunks: list[tuple[float, str, str]],
    query: str,
    top_k: int = CONDENSE_TOP_K,
    min_score: float = CONDENSE_MIN_SCORE,
) -> str:
    """Dedup, filter, rank, and format already-scored chunks. Filtering is
    literal: chunks below min_score are dropped, not truncated or reworded.
    If nothing clears the bar, returns an explicit sentinel."""
    if not scored_chunks:
        return "[no fetched content available to condense]"

    seen_hashes: set[str] = set()
    deduped: list[tuple[float, str, str]] = []
    for score, url, chunk in scored_chunks:
        h = hashlib.sha1(chunk.strip().lower().encode("utf-8", "ignore")).hexdigest()
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        deduped.append((score, url, chunk))

    relevant = sorted(
        (item for item in deduped if item[0] >= min_score),
        key=lambda item: item[0],
        reverse=True,
    )[:top_k]

    if not relevant:
        return (
            f"[no relevant content found among fetched sources for: {query} — "
            "fetched pages did not match the query closely enough to include; "
            "do not fabricate an answer from them, disclose the gap instead]"
        )

    lines = [f"[Condensed evidence for: {query} — {len(relevant)} relevant excerpt(s)]"]
    for score, url, chunk in relevant:
        lines.append(f"[source: {url} | relevance: {score:.2f}]\n{chunk}")
    return "\n\n".join(lines)


def condense_evidence(
    pages: list[tuple[str, str]],
    query: str,
    embedder=None,
    top_k: int = CONDENSE_TOP_K,
    chunk_chars: int = CONDENSE_CHUNK_CHARS,
    min_score: float = CONDENSE_MIN_SCORE,
    max_chunks_to_score: int = CONDENSE_MAX_CHUNKS_TO_SCORE,
) -> str:
    """Convenience wrapper for callers that already have raw (url, text)
    pages in hand and just want them chunked, scored, and condensed."""
    url_chunks: list[tuple[str, str]] = []
    for url, text in pages:
        url_chunks.extend((url, c) for c in reason.chunk_text(text, chunk_chars))
        if len(url_chunks) >= max_chunks_to_score:
            break
    scored = _score_url_chunks(url_chunks, query, embedder, max_chunks_to_score)
    return _finalize_condensed(scored, query, top_k=top_k, min_score=min_score)


def _deep_search_impl(
    query: str,
    num_searches: int,
    num_fetches: int,
    max_chars_per_page: int,
    max_workers: int,
    embedder,
    exclude_urls: set[str] | None,
) -> tuple[str, set[str]]:
    """Fixed, non-adaptive search pass with optional fetch/condense.

    With num_fetches=0 this returns snippets/URLs only, which is now the
    default public deep_search behavior. Deep_research calls this helper with
    its own positive fetch count to do fetched-source work.

    Returns (formatted_bundle, urls_actually_fetched) — the URL set lets
    deep_research exclude already-seen URLs across rounds without a
    separate re-fetch-avoidance mechanism.
    """
    if not query or not query.strip():
        return "[search failed: empty query]", set()

    num_searches = max(1, num_searches)
    num_fetches = max(0, num_fetches)
    all_results: list[dict] = []
    errors: list[str] = []

    log.info("[deep_search] %d search call(s) for: %s", num_searches, query)

    if num_searches == 1:
        results, error = _web_search_raw(query, MAX_RESULTS, pageno=1)
        if error:
            errors.append(error)
        elif results:
            all_results.extend(results)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(num_searches, max_workers)) as pool:
            futures = [pool.submit(_web_search_raw, query, MAX_RESULTS, p) for p in range(1, num_searches + 1)]
            for future in futures:
                results, error = future.result()
                if error:
                    errors.append(error)
                elif results:
                    all_results.extend(results)

    if not all_results:
        return (errors[0] if errors else f"[no results found for: {query}]"), set()

    exclude_urls = exclude_urls or set()
    seen_urls: set[str] = set()
    deduped_results = []
    for r in all_results:
        url = (r.get("url") or "").strip()
        if not url or url in seen_urls or url in exclude_urls:
            continue
        seen_urls.add(url)
        deduped_results.append(r)

    snippet_lines = [f"[Web search results for: {query} ({len(deduped_results)} unique across {num_searches} search call(s))]"]
    for i, r in enumerate(deduped_results, 1):
        snippet_lines.append(f"{i}. {r.get('title', '').strip()}\n   {r.get('url', '').strip()}\n   {r.get('content', '').strip()}")
    snippet_bundle = "\n\n".join(snippet_lines)

    fetch_urls = [r["url"].strip() for r in deduped_results[:num_fetches] if r.get("url")]
    if num_fetches <= 0:
        return snippet_bundle, set()
    scored_chunks, fetched_pages, url_outcomes = _fetch_and_score_pipeline(
        fetch_urls, query, embedder, max_chars_per_page, max_workers=max_workers,
    )
    manifest = _format_url_manifest(url_outcomes)
    fetched_url_set = {url for url, _text in fetched_pages}

    if not fetched_pages:
        return f"{snippet_bundle}\n\n{manifest}", fetched_url_set

    condensed = _finalize_condensed(scored_chunks, query)
    return f"{snippet_bundle}\n\n{manifest}\n\n{condensed}", fetched_url_set


def deep_search(
    query: str,
    num_searches: int = DEEP_SEARCH_NUM_SEARCHES,
    num_fetches: int = DEEP_SEARCH_NUM_FETCHES,
    max_chars_per_page: int = DEEP_SEARCH_MAX_CHARS_PER_PAGE,
    max_workers: int = DEEP_SEARCH_MAX_WORKERS,
    embedder=None,
) -> str:
    """Agentic snippet-only search.

    By default num_fetches is 0, so this returns SearXNG result snippets/URLs
    only and never reads full pages. The num_fetches argument is retained as a
    backwards-compatible escape hatch; keep it 0 for the intended behavior."""
    text, _urls = _deep_search_impl(
        query, num_searches, num_fetches, max_chars_per_page, max_workers, embedder, None,
    )
    return text


def _ask_llm_json(client, model: str, prompt: str, max_tokens: int) -> dict | None:
    """Best-effort structured call for the adaptive research loop. Returns
    None on failure so callers fall back to a fixed round count."""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            max_tokens=max_tokens,
            temperature=0.0,
        )
        raw = (resp.choices[0].message.content or "").strip()
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        return json.loads(match.group(0) if match else raw)
    except Exception:
        return None


def deep_research(
    query: str,
    client=None,
    model: str | None = None,
    embedder=None,
    max_rounds: int = DEEP_RESEARCH_MAX_ROUNDS,
    num_searches: int = DEEP_RESEARCH_NUM_SEARCHES,
    num_fetches: int = DEEP_RESEARCH_NUM_FETCHES,
    max_chars_per_page: int = DEEP_RESEARCH_MAX_CHARS_PER_PAGE,
) -> str:
    """Multi-round adaptive fetched-source research.

    Uses search only to discover candidate URLs, then fetches full pages,
    condenses evidence, and optionally asks the LLM whether another fetched
    round/refined query is needed. This is the heavy research/self-learning
    tool; deep_search remains snippet-only.

    num_searches/num_fetches/max_chars_per_page default to this module's
    DEEP_RESEARCH_* env-backed constants but are now real function
    arguments — every round's underlying _deep_search_impl() call uses
    whatever was passed in here, not the module constants directly. This
    is what lets a caller (e.g. core.learn.quick_studying, or a
    deep_research(..., num_searches=3, num_fetches=1) call site) override
    per-call instead of only ever getting the env-var defaults.
    """
    if not query or not query.strip():
        return "[search failed: empty query]"

    rounds_text: list[str] = []
    seen_urls: set[str] = set()
    queries_used: list[str] = [query.strip()]
    current_query = query.strip()
    adaptive = client is not None and model

    for round_num in range(1, max_rounds + 1):
        log.info("[deep_research] round %d searching: %s", round_num, current_query)
        round_text, round_urls = _deep_search_impl(
            current_query,
            num_searches,
            num_fetches,
            max_chars_per_page,
            DEEP_SEARCH_MAX_WORKERS,
            embedder,
            seen_urls,
        )
        if round_text.startswith("[search failed") and round_num == 1:
            return round_text

        seen_urls |= round_urls
        rounds_text.append(f"[Round {round_num} — query: {current_query}]\n{round_text}")

        if not adaptive or round_num == max_rounds:
            break

        evidence_so_far = "\n\n".join(rounds_text)[-DEEP_RESEARCH_EVIDENCE_CHARS_FOR_DECISION:]
        decision_prompt = (
            "You are directing a multi-round web research process. Given the "
            "original question and the evidence gathered so far, decide whether "
            "another search round is needed.\n"
            "Return ONLY compact JSON: {\"continue\": bool, \"next_query\": string, \"reason\": string}.\n"
            "Set continue=false once the evidence is sufficient to answer the "
            "original question, if further searching is unlikely to add "
            "anything new, or if the evidence explicitly says nothing relevant "
            "was found and a differently-worded query is unlikely to help.\n"
            "next_query should be empty when continue=false.\n\n"
            f"Original question: {query}\n\n"
            f"Prior queries used: {queries_used}\n\n"
            f"Evidence gathered so far:\n{evidence_so_far}"
        )
        decision = _ask_llm_json(client, model, decision_prompt, DEEP_RESEARCH_DECISION_MAX_TOKENS)
        if not decision or not decision.get("continue"):
            break
        next_query = str(decision.get("next_query") or "").strip()
        if not next_query or next_query in queries_used:
            break
        current_query = next_query
        queries_used.append(next_query)

    if not rounds_text:
        return f"[no results found for: {query}]"

    log.info("[deep_research] done: %d round(s)", len(rounds_text))

    header = f"[Deep research: {len(rounds_text)} round(s) for: {query}]"
    if len(queries_used) > 1:
        header += f"\n[Query refinements: {' -> '.join(queries_used)}]"
    rounds_log = "\n\n".join(rounds_text)

    has_usable_evidence = any(
        "no relevant content" not in t and "no results found" not in t and "[search failed" not in t
        for t in rounds_text
    )

    if adaptive and has_usable_evidence:
        synthesis_prompt = (
            "Synthesize the following multi-round research evidence into a "
            "concise, well-organized answer to the original question. Note "
            "any unresolved gaps or conflicting information explicitly. If "
            "evidence says nothing relevant was found, say so plainly instead "
            "of guessing. Do not invent facts not present in the evidence.\n\n"
            f"Original question: {query}\n\n"
            f"Evidence:\n{rounds_log[:DEEP_RESEARCH_EVIDENCE_CHARS_FOR_SYNTHESIS]}"
        )
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": synthesis_prompt}],
                stream=False,
                max_tokens=DEEP_RESEARCH_SYNTHESIS_MAX_TOKENS,
                temperature=0.2,
            )
            synthesis = (resp.choices[0].message.content or "").strip()
            if synthesis:
                return f"{header}\n\n[Synthesis]\n{synthesis}\n\n{rounds_log}"
        except Exception:
            pass  # fall through to raw bundle below

    return f"{header}\n\n{rounds_log}"


def web_search_context(query: str, max_results: int = MAX_RESULTS) -> str | None:
    """Run web_search and wrap successful results as context for chat mode."""
    if not query or not query.strip():
        return "[search failed: empty query]"
    results = web_search(query, max_results)
    if results.startswith("[search failed") or results.startswith("[no results"):
        return None
    return f"{results}\n\nUser asked: {query}"
