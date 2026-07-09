"""Web search and page extraction tools."""

from __future__ import annotations

import concurrent.futures
import hashlib
import ipaddress
import json
import math
import os
import re
import socket
import time
import threading
from urllib.parse import urlparse

import importlib
import importlib.util

from core.log import get_logger

log = get_logger(__name__)

SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:8888")
MAX_RESULTS = int(os.getenv("SEARXNG_MAX_RESULTS", 5))

# -- deep_search (search + fetch pass, tunable for extra breadth) --
DEEP_SEARCH_MAX_RESULTS = int(os.getenv("DEEP_SEARCH_MAX_RESULTS", 3))
DEEP_SEARCH_FETCH_TOP = int(os.getenv("DEEP_SEARCH_FETCH_TOP", 2))
DEEP_SEARCH_MAX_CHARS_PER_PAGE = int(os.getenv("DEEP_SEARCH_MAX_CHARS_PER_PAGE", 2000))
DEEP_SEARCH_SEARCH_PAGES = int(os.getenv("DEEP_SEARCH_SEARCH_PAGES", 1))
DEEP_SEARCH_MAX_WORKERS = int(os.getenv("DEEP_SEARCH_MAX_WORKERS", 4))

# -- deep_research (multi-round adaptive research) --
DEEP_RESEARCH_MAX_ROUNDS = int(os.getenv("DEEP_RESEARCH_MAX_ROUNDS", 3))
DEEP_RESEARCH_FETCH_TOP = int(os.getenv("DEEP_RESEARCH_FETCH_TOP", 2))
DEEP_RESEARCH_MAX_CHARS_PER_PAGE = int(os.getenv("DEEP_RESEARCH_MAX_CHARS_PER_PAGE", 1500))
DEEP_RESEARCH_EVIDENCE_CHARS_FOR_DECISION = int(os.getenv("DEEP_RESEARCH_EVIDENCE_CHARS_FOR_DECISION", 6000))
DEEP_RESEARCH_EVIDENCE_CHARS_FOR_SYNTHESIS = int(os.getenv("DEEP_RESEARCH_EVIDENCE_CHARS_FOR_SYNTHESIS", 8000))
DEEP_RESEARCH_DECISION_MAX_TOKENS = int(os.getenv("DEEP_RESEARCH_DECISION_MAX_TOKENS", 200))
DEEP_RESEARCH_SYNTHESIS_MAX_TOKENS = int(os.getenv("DEEP_RESEARCH_SYNTHESIS_MAX_TOKENS", 600))

# -- in-memory evidence condensation (embedding-based relevance filtering) --
# Everything here is plain Python lists/tuples in process memory — nothing
# is written to disk. Condensing is a FILTER, not a rewrite: chunks are
# scored for relevance and either kept verbatim or dropped entirely. No text
# is summarized or altered by this step; that only happens later, in
# deep_research's separate LLM synthesis call.
CONDENSE_CHUNK_CHARS = int(os.getenv("CONDENSE_CHUNK_CHARS", 500))
CONDENSE_TOP_K = int(os.getenv("CONDENSE_TOP_K", 8))
CONDENSE_MIN_SCORE = float(os.getenv("CONDENSE_MIN_SCORE", 0.15))
# Caps embedding calls PER fetch pipeline invocation (i.e. per deep_search
# call, or per round inside deep_research) — not a lifetime cap. Since each
# page's chunks are now only ever embedded once (see _fetch_and_score_pipeline
# below), this bound is what actually determines worst-case embedding
# latency for a single round/call.
CONDENSE_MAX_CHUNKS_TO_SCORE = int(os.getenv("CONDENSE_MAX_CHUNKS_TO_SCORE", 60))

# -- web_fetch download guard --
# Caps the raw response body size read off the wire, BEFORE trafilatura ever
# runs extraction. Without this, a single oversized page (a data dump, a
# giant PDF served as HTML, a page with megabytes of inline JSON) gets fully
# downloaded into memory before max_chars truncation ever kicks in. On a
# Jetson-class device with a shared memory pool and an ONNX runtime session
# already resident, a handful of concurrent oversized fetches (multiplied by
# DEEP_SEARCH_MAX_WORKERS) can spike memory hard. This bound stops the
# download mid-stream instead of after the fact.
WEB_FETCH_MAX_DOWNLOAD_BYTES = int(os.getenv("WEB_FETCH_MAX_DOWNLOAD_BYTES", 5_000_000))
WEB_FETCH_TIMEOUT_SECONDS = int(os.getenv("WEB_FETCH_TIMEOUT_SECONDS", 8))
WEB_FETCH_USER_AGENT = os.getenv(
    "WEB_FETCH_USER_AGENT",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36",
)

# -- short-lived in-process query cache --
# Same idea as OpenClaw's default 15-minute web_search cache: cheap insurance
# against a single research round (or an agent loop retrying) re-issuing an
# identical query/URL within seconds of the last call. This is NOT a
# persisted index — it's a small in-memory dict with a TTL, gone the moment
# the process restarts. It doesn't change the JIT/ephemeral nature of the
# architecture; it just avoids paying the full fetch+embed cost twice for
# the exact same input inside one short window.
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
            # Evict the oldest entry rather than growing unbounded. Not a
            # true LRU — good enough for a short-TTL convenience cache.
            oldest_key = min(cache, key=lambda k: cache[k][0], default=None)
            if oldest_key is not None:
                cache.pop(oldest_key, None)
        cache[key] = (time.monotonic(), value)


def _web_search_raw(query: str, max_results: int, pageno: int = 1) -> tuple[list[dict] | None, str | None]:
    """Low-level SearXNG call returning (results, error). Kept separate from
    web_search() so callers merging multiple pages (deep_search's
    search_pages, deep_research's rounds) don't have to re-parse formatted
    text output — they get structured dicts directly.

    Cached in-process for CACHE_TTL_SECONDS keyed on (query, max_results,
    pageno) — repeat calls within the TTL skip the network round-trip
    entirely.
    """
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
    the model's direct fetch_page-style calls and deep_search/deep_research's
    internal pipelined page reads route through this function, so there is
    exactly one implementation to reason about.

    Downloads are streamed and capped at max_download_bytes — the download is
    aborted mid-stream the moment it exceeds the cap, BEFORE trafilatura ever
    runs extraction and BEFORE max_chars truncation. This is what actually
    bounds worst-case memory for a single fetch; max_chars alone only bounds
    the text kept *after* the full page was already downloaded.

    Successful fetches are cached in-process for CACHE_TTL_SECONDS keyed on
    (url, max_chars) — set use_cache=False to force a fresh fetch (e.g. for
    pages known to change quickly, like live scoreboards or breaking news).
    Failed fetches are never cached, so a transient network error doesn't
    poison the cache for the TTL window.
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
                    pass  # malformed header — fall through to the streamed cap below

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


def _extract_urls(raw_results: str, limit: int) -> list[str]:
    """Retained for any external caller still parsing web_search()'s
    formatted text; deep_search/deep_research use _web_search_raw directly."""
    result_blocks = raw_results.split("\n\n")[1:]
    urls = []
    for block in result_blocks[:limit]:
        url = next((line.strip() for line in block.splitlines() if line.strip().startswith("http")), None)
        if url:
            urls.append(url)
    return urls


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _keyword_overlap_score(query: str, text: str) -> float:
    """Fallback relevance score when no embedder is available — same
    graceful-degradation pattern already used in knowledge.py/skills.py."""
    q_terms = {t for t in re.findall(r"[a-z0-9]+", query.lower()) if len(t) > 2}
    if not q_terms:
        return 0.0
    t_terms = {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2}
    if not t_terms:
        return 0.0
    return len(q_terms & t_terms) / len(q_terms)


def _chunk_text(url: str, text: str, chunk_chars: int) -> list[tuple[str, str]]:
    chunks = []
    for i in range(0, len(text), chunk_chars):
        piece = text[i:i + chunk_chars].strip()
        if piece:
            chunks.append((url, piece))
    return chunks


def _embed_batch(embedder, texts: list[str]) -> list[list[float]] | None:
    """Best-effort batched embedding. Tries common batch-method names on the
    embedder in order, returns None if none are available so the caller can
    fall back to one-at-a-time embed_query calls.

    This is opt-in and speculative on purpose: HarrierEmbedder's actual
    interface isn't something this file has visibility into, so rather than
    assume a method name, we probe for the conventional ones
    (embed_documents is the LangChain-style convention; embed_batch and
    embed are plausible custom names) and gracefully do nothing if none
    exist. Batching only helps if the underlying onnxruntime session
    actually does one forward pass for the batch rather than looping
    internally — confirm that before relying on this for a speedup.
    """
    for method_name in ("embed_documents", "embed_batch", "embed"):
        method = getattr(embedder, method_name, None)
        if callable(method):
            try:
                result = method(texts)
                if result is not None and len(result) == len(texts):
                    return list(result)
            except Exception:
                continue
    return None


def _score_chunk(chunk: str, query: str, embedder, query_vec) -> float:
    """Score one chunk against the query. Caller supplies a precomputed
    query_vec (embedded once per pipeline call, not once per chunk)."""
    if embedder is not None and query_vec is not None:
        try:
            chunk_vec = embedder.embed_query(chunk)
            return _cosine(query_vec, chunk_vec)
        except Exception:
            pass
    return _keyword_overlap_score(query, chunk)


def _score_chunks_for_page(
    url: str,
    text: str,
    query: str,
    embedder,
    query_vec,
    use_embedder: bool,
    chunk_chars: int,
    remaining_budget: int,
) -> list[tuple[float, str, str]]:
    """Chunk one page's text and score every chunk, batching the embedding
    call for the whole page when the embedder supports it. Falls back to
    per-chunk _score_chunk calls (and further to keyword overlap) otherwise.
    remaining_budget caps how many chunks from THIS page get scored, so the
    global CONDENSE_MAX_CHUNKS_TO_SCORE bound is respected across pages.
    """
    page_chunks = _chunk_text(url, text, chunk_chars)[:remaining_budget]
    if not page_chunks:
        return []

    scored: list[tuple[float, str, str]] = []

    if use_embedder and query_vec is not None:
        chunk_texts = [c for _, c in page_chunks]
        batch_vecs = _embed_batch(embedder, chunk_texts)
        if batch_vecs is not None:
            for (_, chunk), vec in zip(page_chunks, batch_vecs):
                try:
                    score = _cosine(query_vec, vec)
                except Exception:
                    score = _keyword_overlap_score(query, chunk)
                scored.append((score, url, chunk))
            return scored

    # No batch method available (or it failed) — per-chunk fallback.
    for _, chunk in page_chunks:
        score = _score_chunk(chunk, query, embedder if use_embedder else None, query_vec)
        scored.append((score, url, chunk))
    return scored


def _fetch_and_score_pipeline(
    urls: list[str],
    query: str,
    embedder,
    max_chars_per_page: int,
    chunk_chars: int = CONDENSE_CHUNK_CHARS,
    max_workers: int = DEEP_SEARCH_MAX_WORKERS,
    max_chunks_to_score: int = CONDENSE_MAX_CHUNKS_TO_SCORE,
) -> tuple[list[tuple[float, str, str]], list[tuple[str, str]], list[tuple[str, str]]]:
    """Fetch multiple URLs concurrently, and score each page's chunks for
    relevance THE MOMENT that page finishes downloading — not after every
    URL has finished fetching.

    Returns (scored_chunks, pages, url_outcomes):
    - scored_chunks: (score, url, chunk_text) tuples for condensing.
    - pages: (url, full_text) for every URL that succeeded.
    - url_outcomes: (url, status) for EVERY url passed in, success or
      failure, in the order they completed — this is what lets
      deep_search/deep_research report a full URL manifest instead of only
      ever showing the successes.
    """
    if not urls:
        return [], [], []

    log.info("[fetch_pipeline] attempting %d url(s): %s", len(urls), urls)

    use_embedder = embedder is not None and hasattr(embedder, "embed_query")
    query_vec = None
    if use_embedder:
        try:
            query_vec = embedder.embed_query(query)
        except Exception:
            use_embedder = False

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

            page_scored = _score_chunks_for_page(
                url, text, query, embedder if use_embedder else None, query_vec,
                use_embedder, chunk_chars, remaining_budget,
            )
            scored.extend(page_scored)
            chunks_scored += len(page_scored)

    log.info(
        "[fetch_pipeline] done: %d/%d succeeded, %d chunk(s) scored",
        len(pages), len(urls), chunks_scored,
    )
    return scored, pages, url_outcomes


def _format_url_manifest(url_outcomes: list[tuple[str, str]]) -> str:
    """Render a per-URL success/failure manifest for inclusion in tool
    output, so the model (and anyone reading the transcript) can see
    exactly which URLs were attempted without needing separate log access."""
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
    """Dedup, filter, rank, and format already-scored chunks into the final
    bundle. No embedding happens here — this is pure filtering/sorting over
    scores computed earlier by _fetch_and_score_pipeline, which is what lets
    deep_research call this once per round without re-embedding prior
    rounds' chunks every time.

    Filtering is literal: chunks below min_score are dropped, not truncated
    or reworded. If nothing clears the bar, returns an explicit sentinel
    instead of silently handing back irrelevant text.
    """
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
    pages in hand (no fetching to overlap with) and just want them chunked,
    scored, and condensed in one call. deep_search/deep_research no longer
    use this internally — they use _fetch_and_score_pipeline so embedding
    overlaps with fetching — but this is kept for any other caller that
    wants condensing without a pipeline."""
    use_embedder = embedder is not None and hasattr(embedder, "embed_query")
    query_vec = None
    if use_embedder:
        try:
            query_vec = embedder.embed_query(query)
        except Exception:
            use_embedder = False

    scored: list[tuple[float, str, str]] = []
    chunks_scored = 0
    for url, text in pages:
        remaining_budget = max_chunks_to_score - chunks_scored
        if remaining_budget <= 0:
            break
        page_scored = _score_chunks_for_page(
            url, text, query, embedder if use_embedder else None, query_vec,
            use_embedder, chunk_chars, remaining_budget,
        )
        scored.extend(page_scored)
        chunks_scored += len(page_scored)

    return _finalize_condensed(scored, query, top_k=top_k, min_score=min_score)


def deep_search(
    query: str,
    max_results: int = DEEP_SEARCH_MAX_RESULTS,
    fetch_top: int = DEEP_SEARCH_FETCH_TOP,
    max_chars_per_page: int = DEEP_SEARCH_MAX_CHARS_PER_PAGE,
    search_pages: int = DEEP_SEARCH_SEARCH_PAGES,
    embedder=None,
) -> str:
    """Search (optionally across multiple SearXNG result pages, pulled
    concurrently), then fetch the top pages while scoring each one for
    relevance as it lands. Returned bundle now includes a full URL manifest
    (every attempted URL + success/failure) so fetch activity is visible
    without needing log access.
    """
    if not query or not query.strip():
        return "[search failed: empty query]"

    search_pages = max(1, search_pages)
    all_results: list[dict] = []
    errors: list[str] = []

    log.info("[deep_search] searching %d page(s) for: %s", search_pages, query)

    if search_pages == 1:
        results, error = _web_search_raw(query, max_results, pageno=1)
        if error:
            errors.append(error)
        elif results:
            all_results.extend(results)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(search_pages, DEEP_SEARCH_MAX_WORKERS)) as pool:
            futures = [pool.submit(_web_search_raw, query, max_results, p) for p in range(1, search_pages + 1)]
            for future in futures:
                results, error = future.result()
                if error:
                    errors.append(error)
                elif results:
                    all_results.extend(results)

    if not all_results:
        return errors[0] if errors else f"[no results found for: {query}]"

    seen_urls: set[str] = set()
    deduped_results = []
    for r in all_results:
        url = (r.get("url") or "").strip()
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        deduped_results.append(r)

    snippet_lines = [f"[Web search results for: {query} ({len(deduped_results)} unique across {search_pages} page(s))]"]
    for i, r in enumerate(deduped_results, 1):
        snippet_lines.append(f"{i}. {r.get('title', '').strip()}\n   {r.get('url', '').strip()}\n   {r.get('content', '').strip()}")
    snippet_bundle = "\n\n".join(snippet_lines)

    fetch_urls = [r["url"].strip() for r in deduped_results[:fetch_top] if r.get("url")]
    scored_chunks, fetched_pages, url_outcomes = _fetch_and_score_pipeline(
        fetch_urls, query, embedder, max_chars_per_page, max_workers=DEEP_SEARCH_MAX_WORKERS,
    )

    manifest = _format_url_manifest(url_outcomes)

    if not fetched_pages:
        return f"{snippet_bundle}\n\n{manifest}"

    condensed = _finalize_condensed(scored_chunks, query)
    return f"{snippet_bundle}\n\n{manifest}\n\n{condensed}"


def _ask_llm_json(client, model: str, prompt: str, max_tokens: int) -> dict | None:
    """Best-effort structured call for the adaptive research loop.

    Returns None on any failure so callers can fall back to a fixed,
    non-adaptive round count instead of crashing the whole research call.
    """
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
    fetch_top: int = DEEP_RESEARCH_FETCH_TOP,
    max_chars_per_page: int = DEEP_RESEARCH_MAX_CHARS_PER_PAGE,
) -> str:
    """Multi-round adaptive research... (docstring unchanged)

    The returned bundle now includes a full URL manifest accumulated across
    ALL rounds — every URL attempted in every round, with success/failure
    status — so total fetch activity for the whole multi-round call is
    visible in one place, not scattered across per-round logs.
    """
    if not query or not query.strip():
        return "[search failed: empty query]"

    rounds: list[str] = []
    all_scored_chunks: list[tuple[float, str, str]] = []
    all_url_outcomes: list[tuple[str, str]] = []  # accumulated across every round
    fetched_page_count = 0
    queries_used: list[str] = [query.strip()]
    seen_urls: set[str] = set()
    current_query = query.strip()
    adaptive = client is not None and model

    for round_num in range(1, max_rounds + 1):
        log.info("[deep_research] round %d searching: %s", round_num, current_query)
        results, error = _web_search_raw(current_query, DEEP_SEARCH_MAX_RESULTS, pageno=1)
        if error:
            if round_num == 1:
                return error
            break
        if not results:
            break

        new_urls = [
            r["url"].strip() for r in results
            if r.get("url") and r["url"].strip() not in seen_urls
        ][:fetch_top]
        seen_urls.update(new_urls)

        round_scored, round_pages, round_url_outcomes = _fetch_and_score_pipeline(
            new_urls, current_query, embedder, max_chars_per_page, max_workers=DEEP_SEARCH_MAX_WORKERS,
        )
        all_url_outcomes.extend(round_url_outcomes)

        if round_pages:
            all_scored_chunks.extend(round_scored)
            fetched_page_count += len(round_pages)
            rounds.append(f"[Round {round_num} — query: {current_query} — fetched {len(round_pages)} page(s)]")
        elif round_num == 1:
            snippet_lines = [f"[Round {round_num} — query: {current_query} — snippets only, no pages fetched]"]
            for r in results[:DEEP_SEARCH_MAX_RESULTS]:
                snippet_lines.append(f"- {r.get('title', '').strip()}: {r.get('url', '').strip()}")
            rounds.append("\n".join(snippet_lines))

        if not adaptive or round_num == max_rounds:
            break

        evidence_so_far = _finalize_condensed(
            all_scored_chunks, query, top_k=12
        )[:DEEP_RESEARCH_EVIDENCE_CHARS_FOR_DECISION]
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

    if not rounds:
        return f"[no results found for: {query}]"

    log.info(
        "[deep_research] done: %d round(s), %d url(s) attempted total, %d succeeded",
        len(rounds), len(all_url_outcomes), fetched_page_count,
    )

    header = f"[Deep research: {len(rounds)} round(s) for: {query}]"
    if len(queries_used) > 1:
        header += f"\n[Query refinements: {' -> '.join(queries_used)}]"

    manifest = _format_url_manifest(all_url_outcomes)

    condensed_evidence = (
        _finalize_condensed(all_scored_chunks, query) if fetched_page_count
        else "[no pages were successfully fetched across any round; snippets only]"
    )
    rounds_log = "\n\n".join(rounds)

    if adaptive and fetched_page_count:
        synthesis_prompt = (
            "Synthesize the following research evidence into a concise, "
            "well-organized answer to the original question. Note any "
            "unresolved gaps or conflicting information explicitly. If the "
            "evidence says nothing relevant was found, say so plainly instead "
            "of guessing. Do not invent facts not present in the evidence.\n\n"
            f"Original question: {query}\n\n"
            f"Evidence:\n{condensed_evidence[:DEEP_RESEARCH_EVIDENCE_CHARS_FOR_SYNTHESIS]}"
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
                return f"{header}\n\n[Synthesis]\n{synthesis}\n\n{manifest}\n\n[Condensed evidence]\n{condensed_evidence}"
        except Exception:
            pass  # fall through to raw bundle below

    return f"{header}\n\n{rounds_log}\n\n{manifest}\n\n{condensed_evidence}"


def web_search_context(query: str, max_results: int = MAX_RESULTS) -> str | None:
    """Run web_search and wrap successful results as context for chat mode."""
    if not query or not query.strip():
        return "[search failed: empty query]"
    results = web_search(query, max_results)
    if results.startswith("[search failed") or results.startswith("[no results"):
        return None
    return f"{results}\n\nUser asked: {query}"
