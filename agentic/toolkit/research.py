"""
toolkit/research.py

Web search and page extraction tools.

This module provides web research capabilities for agentic workflows:

  - web_search()    — search via configured SearXNG instance
  - web_fetch()     — fetch and extract text from URLs (requests + trafilatura)
  - deep_search()   — snippet-only search pass for task-mode workflows (FAST path)
  - deep_research() — adaptive fetched-source research with synthesis (ACCURATE path)

Requires a running SearXNG instance (SEARXNG_URL env var).

deep_search vs deep_research, by design intent:
  - deep_search is the cheap, latency-first primitive. It never fetches full
    pages by default and never touches Crawl4AI, robots.txt, or sitemaps.
  - deep_research is the accuracy-first, "trusted KB" primitive. Latency is
    a secondary concern here. It optionally uses Crawl4AI for higher-quality
    page extraction, expands candidate URLs via sitemap discovery, respects
    robots.txt ("source agreement") before fetching anything, and scores
    cross-source corroboration so multi-source claims outrank single-source
    ones in the condensed evidence it hands to the LLM synthesis step.

Crawl4AI is optional. If it isn't installed (`pip install crawl4ai` plus
`crawl4ai-setup` / `playwright install chromium` for the browser binary),
deep_research transparently falls back to the requests+trafilatura path —
same graceful-degradation pattern this file already uses for `requests` and
`trafilatura` themselves.
"""

from __future__ import annotations

import concurrent.futures
import functools
import hashlib
import io
import ipaddress
import json
import os
import re
import socket
import time
import threading
from dataclasses import dataclass, field
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import importlib
import importlib.util

import numpy as np

from system.log import get_logger
from cognition import reason

log = get_logger(__name__)

SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:8888")
MAX_RESULTS = int(os.getenv("SEARXNG_MAX_RESULTS", 5))

# -- deep_search (fixed, non-adaptive snippet search pass) --
# Task-mode deep_search is intentionally snippet-only. It uses the same raw
# SearXNG primitive as web_search, but is exposed to agentic workflows while
# web_search/web_fetch remain chat-mode primitives. deep_search NEVER uses
# Crawl4AI, sitemap expansion, or robots-gated fetching — those are
# deep_research-only, by design, to keep this path fast.
DEEP_SEARCH_NUM_SEARCHES = int(os.getenv("DEEP_SEARCH_NUM_SEARCHES", 1))
DEEP_SEARCH_NUM_FETCHES = int(os.getenv("DEEP_SEARCH_NUM_FETCHES", 0))  # legacy override; keep 0 for snippet-only deep_search
DEEP_SEARCH_MAX_CHARS_PER_PAGE = int(os.getenv("DEEP_SEARCH_MAX_CHARS_PER_PAGE", 2000))
DEEP_SEARCH_MAX_WORKERS = int(os.getenv("DEEP_SEARCH_MAX_WORKERS", 4))

# -- deep_research (adaptive fetched-source research) --
# Deep research uses search only to discover URLs, then fetches pages and
# condenses/synthesizes evidence. It has its own fetch knobs so deep_search can
# stay snippet-only. These are read once as module-level DEFAULTS; deep_research()
# itself now accepts num_searches/num_fetches/max_chars_per_page as real
# function args (see below) so a caller — e.g. memory.learn.quick_studying —
# can override per-call instead of only ever getting these env defaults.
#
# Tuned for accuracy-over-latency per JJ: more fetches, more rounds, more
# per-page budget than the old defaults. deep_search's knobs above are left
# untouched so it stays the fast path.
DEEP_RESEARCH_NUM_SEARCHES = int(os.getenv("DEEP_RESEARCH_NUM_SEARCHES", os.getenv("DEEP_SEARCH_NUM_SEARCHES", 1)))
DEEP_RESEARCH_NUM_FETCHES = int(os.getenv("DEEP_RESEARCH_NUM_FETCHES", 4))
DEEP_RESEARCH_MAX_CHARS_PER_PAGE = int(os.getenv("DEEP_RESEARCH_MAX_CHARS_PER_PAGE", 3500))
DEEP_RESEARCH_MAX_ROUNDS = int(os.getenv("DEEP_RESEARCH_MAX_ROUNDS", 4))
DEEP_RESEARCH_EVIDENCE_CHARS_FOR_DECISION = int(os.getenv("DEEP_RESEARCH_EVIDENCE_CHARS_FOR_DECISION", 8000))
DEEP_RESEARCH_EVIDENCE_CHARS_FOR_SYNTHESIS = int(os.getenv("DEEP_RESEARCH_EVIDENCE_CHARS_FOR_SYNTHESIS", 12000))
DEEP_RESEARCH_DECISION_MAX_TOKENS = int(os.getenv("DEEP_RESEARCH_DECISION_MAX_TOKENS", 200))
DEEP_RESEARCH_SYNTHESIS_MAX_TOKENS = int(os.getenv("DEEP_RESEARCH_SYNTHESIS_MAX_TOKENS", 700))  # raised to 1500 in config/agentic.yaml

# -- in-memory evidence condensation (numpy-vectorized relevance filtering) --
# A FILTER, not a rewrite: chunks are scored for relevance and either kept
# verbatim or dropped entirely. Summarization only happens later, in
# deep_research's separate LLM synthesis call.
#
# deep_search keeps the original (fast/cheap) knobs below. deep_research uses
# the separate RESEARCH_CONDENSE_* knobs further down — wider net, lower bar,
# because the corroboration bonus (see _apply_corroboration_bonus) promotes
# borderline items that get independently confirmed by a second domain.
CONDENSE_CHUNK_CHARS = int(os.getenv("CONDENSE_CHUNK_CHARS", 500))
CONDENSE_TOP_K = int(os.getenv("CONDENSE_TOP_K", 8))
CONDENSE_MIN_SCORE = float(os.getenv("CONDENSE_MIN_SCORE", 0.15))
# Caps embedding calls PER fetch pipeline invocation (per deep_search call,
# i.e. per round) — not a lifetime cap.
CONDENSE_MAX_CHUNKS_TO_SCORE = int(os.getenv("CONDENSE_MAX_CHUNKS_TO_SCORE", 60))

RESEARCH_CONDENSE_TOP_K = int(os.getenv("RESEARCH_CONDENSE_TOP_K", 12))
RESEARCH_CONDENSE_MIN_SCORE = float(os.getenv("RESEARCH_CONDENSE_MIN_SCORE", 0.12))
RESEARCH_CONDENSE_MAX_CHUNKS_TO_SCORE = int(os.getenv("RESEARCH_CONDENSE_MAX_CHUNKS_TO_SCORE", 100))

# -- cross-source corroboration ("sources agreement" scoring) --
# Independent confirmation from a SECOND domain boosts a chunk's relevance
# score; same-domain repeats don't count. This is separate from the
# robots.txt "may I fetch this" agreement below — this one is about whether
# multiple independent sources agree on a claim.
RESEARCH_AGREEMENT_BONUS = float(os.getenv("RESEARCH_AGREEMENT_BONUS", 0.12))
RESEARCH_AGREEMENT_SIMILARITY = float(os.getenv("RESEARCH_AGREEMENT_SIMILARITY", 0.5))
RESEARCH_AGREEMENT_SHINGLE_SIZE = int(os.getenv("RESEARCH_AGREEMENT_SHINGLE_SIZE", 5))

# -- Crawl4AI (optional, richer extraction for deep_research) --
# Requires: pip install crawl4ai && crawl4ai-setup (installs the Playwright
# browser). Gracefully no-ops if not installed — deep_research falls back to
# web_fetch (requests+trafilatura) for every URL in that case.
RESEARCH_USE_CRAWL4AI = os.getenv("RESEARCH_USE_CRAWL4AI", "1").lower() in {"1", "true", "yes", "on"}
CRAWL4AI_TIMEOUT_MS = int(os.getenv("CRAWL4AI_TIMEOUT_MS", 20000))
CRAWL4AI_MAX_CONCURRENT = int(os.getenv("CRAWL4AI_MAX_CONCURRENT", 4))
CRAWL4AI_WORD_COUNT_THRESHOLD = int(os.getenv("CRAWL4AI_WORD_COUNT_THRESHOLD", 40))

# -- robots.txt compliance ("source agreement" to be crawled) + sitemap --
RESEARCH_RESPECT_ROBOTS = os.getenv("RESEARCH_RESPECT_ROBOTS", "1").lower() in {"1", "true", "yes", "on"}
ROBOTS_CACHE_TTL_SECONDS = int(os.getenv("ROBOTS_CACHE_TTL_SECONDS", 3600))
RESEARCH_SITEMAP_ENABLED = os.getenv("RESEARCH_SITEMAP_ENABLED", "1").lower() in {"1", "true", "yes", "on"}
RESEARCH_SITEMAP_MAX_URLS = int(os.getenv("RESEARCH_SITEMAP_MAX_URLS", 6))
RESEARCH_SITEMAP_TIMEOUT_SECONDS = int(os.getenv("RESEARCH_SITEMAP_TIMEOUT_SECONDS", 6))

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

_robots_lock = threading.Lock()
_robots_cache: dict[str, tuple[float, RobotFileParser]] = {}


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
    last_error = None
    for attempt in range(3):
        try:
            response = requests.get(
                f"{SEARXNG_URL}/search",
                params={"q": query, "format": "json", "pageno": pageno},
                timeout=8,
            )
            if response.status_code == 429:
                last_error = f"[search failed: rate limited (attempt {attempt + 1})]"
                time.sleep(2.0 * (attempt + 1))
                continue
            response.raise_for_status()
            data = response.json()
            break
        except requests.exceptions.ConnectionError as e:
            last_error = f"[search failed: {e}]"
            time.sleep(1.0 * (attempt + 1))
            continue
        except ValueError:
            return None, "[search failed: invalid JSON response]"
        except requests.exceptions.RequestException as e:
            return None, f"[search failed: {e}]"
    else:
        return None, last_error or "[search failed: max retries]"

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

    This is the baseline "fetch a page" primitive in the toolkit — fast,
    dependency-light, no JS rendering. deep_research prefers Crawl4AI when
    available (see _crawl4ai_fetch_many) and falls back to this for anything
    Crawl4AI misses or when it isn't installed. deep_search always uses this
    directly (when it fetches at all).

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

            buf = io.BytesIO()
            total = 0
            for chunk in resp.iter_content(chunk_size=65536):
                if not chunk:
                    continue
                total += len(chunk)
                if total > max_download_bytes:
                    return "[fetch failed: page exceeded size limit during download]"
                buf.write(chunk)
            downloaded = buf.getvalue()
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


def read_paper_url(
    url: str,
    query: str = "",
    embedder=None,
    max_chars: int = 40000,
    condense_top_k: int = 12,
) -> str:
    """Fetch one EXACT URL — no search involved — for reading a specific
    paper/document the user already chose. Distinct from deep_research,
    which discovers URLs via search and may or may not land on the exact
    page given.

    Caveat: extraction is trafilatura, built for HTML article pages. Many
    paper links are raw PDFs (arxiv.org/pdf/...) and will likely return
    little or no text. Prefer the HTML/abstract page (arxiv.org/abs/... or
    an arXiv HTML rendering) over a direct PDF link where available.

    Without `query`: returns the first max_chars of extracted text — which
    will then be hard-truncated again to AGENT_TOOL_RESULT_MAX_CHARS (8000
    chars) once wrapped as a tool observation, so a long paper effectively
    gets reduced to its opening section only.

    With `query`: text is chunked and relevance-scored the same way
    deep_research condenses evidence, so what survives the 8000-char
    observation limit is the material most relevant to the task at hand,
    not just whatever came first in the document.
    """
    text = web_fetch(url, max_chars=max_chars, max_download_bytes=15_000_000)
    if text.startswith("[fetch failed"):
        return text
    if not query:
        return f"[Fetched paper content — {url}]\n\n{text}"
    condensed = condense_evidence([(url, text)], query, embedder=embedder, top_k=condense_top_k)
    return f"[Fetched paper content — {url}, condensed for: {query}]\n\n{condensed}"
  
# ── robots.txt compliance ("source agreement" to be crawled) ────────────────

def _get_robot_parser(origin: str) -> RobotFileParser:
    """Fetch and cache a RobotFileParser for one origin (scheme://netloc).
    Fails open (allow-all) if robots.txt is missing or unreachable, which
    matches standard crawler convention — explicit Disallow rules are still
    honored whenever robots.txt IS reachable."""
    with _robots_lock:
        entry = _robots_cache.get(origin)
        if entry and time.monotonic() - entry[0] < ROBOTS_CACHE_TTL_SECONDS:
            return entry[1]

    parser = RobotFileParser()
    parser.set_url(f"{origin}/robots.txt")
    if importlib.util.find_spec("requests") is not None:
        requests = importlib.import_module("requests")
        try:
            resp = requests.get(
                f"{origin}/robots.txt", timeout=5,
                headers={"User-Agent": WEB_FETCH_USER_AGENT},
            )
            if resp.status_code >= 400:
                parser.parse([])
            else:
                parser.parse(resp.text.splitlines())
        except Exception:
            parser.parse([])
    else:
        parser.parse([])

    with _robots_lock:
        _robots_cache[origin] = (time.monotonic(), parser)
    return parser


def _source_agreement_allows(url: str) -> bool:
    """deep_research's crawl-citizenship gate: only fetch pages the source's
    own robots.txt permits for our user-agent. Fails open on any lookup
    error (unreachable/unparseable robots.txt), same convention as most
    well-behaved crawlers."""
    if not RESEARCH_RESPECT_ROBOTS:
        return True
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return True
    origin = f"{parsed.scheme}://{parsed.netloc}"
    try:
        parser = _get_robot_parser(origin)
        return parser.can_fetch(WEB_FETCH_USER_AGENT, url)
    except Exception:
        return True


# ── sitemap discovery ────────────────────────────────────────────────────────

def _discover_sitemap_urls(origin: str, query_hint: str = "", max_urls: int = RESEARCH_SITEMAP_MAX_URLS) -> list[str]:
    """Best-effort sitemap.xml discovery for one domain, used by deep_research
    to widen candidate URLs beyond search-engine snippets when a source
    domain looks authoritative for the query (e.g. official docs). Checks
    robots.txt 'Sitemap:' directives first, falls back to /sitemap.xml.
    Returns at most max_urls URLs, ranked by keyword overlap with query_hint
    when provided."""
    if not RESEARCH_SITEMAP_ENABLED:
        return []
    if importlib.util.find_spec("requests") is None:
        return []
    requests = importlib.import_module("requests")

    candidates: list[str] = []
    try:
        robots_resp = requests.get(
            f"{origin}/robots.txt", timeout=RESEARCH_SITEMAP_TIMEOUT_SECONDS,
            headers={"User-Agent": WEB_FETCH_USER_AGENT},
        )
        if robots_resp.ok:
            for line in robots_resp.text.splitlines():
                if line.lower().startswith("sitemap:"):
                    candidates.append(line.split(":", 1)[1].strip())
    except Exception:
        pass
    if not candidates:
        candidates.append(f"{origin}/sitemap.xml")

    urls: list[str] = []
    for sitemap_url in candidates[:3]:
        try:
            resp = requests.get(
                sitemap_url, timeout=RESEARCH_SITEMAP_TIMEOUT_SECONDS,
                headers={"User-Agent": WEB_FETCH_USER_AGENT},
            )
            if not resp.ok:
                continue
            locs = re.findall(r"<loc>(.*?)</loc>", resp.text, flags=re.IGNORECASE | re.DOTALL)
            urls.extend(loc.strip() for loc in locs if loc.strip())
        except Exception:
            continue
        if urls:
            break  # first working sitemap source is enough

    if not urls:
        return []

    if query_hint:
        urls = sorted(urls, key=lambda u: reason.keyword_overlap_score(query_hint, u), reverse=True)
    return urls[:max_urls]


# ── Crawl4AI batch fetch (deep_research only) ────────────────────────────────

def _crawl4ai_fetch_many(urls: list[str], max_chars: int) -> dict[str, str]:
    """Fetch many URLs in ONE Crawl4AI session (single browser launch,
    concurrent pages via arun_many) instead of one browser per URL — a
    per-URL launch would be far too slow to be worth it. This is the batch
    path deep_research prefers; returns {} on any failure or when crawl4ai
    isn't installed, so the caller's per-URL fallback (web_fetch) covers
    every URL missing from the result.

    Already filters out robots-disallowed URLs itself as defense in depth,
    though _deep_search_impl also filters before calling this.
    """
    if not urls or importlib.util.find_spec("crawl4ai") is None:
        return {}

    allowed_urls = [u for u in urls if _source_agreement_allows(u)] if RESEARCH_RESPECT_ROBOTS else list(urls)
    if not allowed_urls:
        return {}

    try:
        import asyncio
        crawl4ai = importlib.import_module("crawl4ai")
        AsyncWebCrawler = crawl4ai.AsyncWebCrawler
        CrawlerRunConfig = crawl4ai.CrawlerRunConfig
        CacheMode = crawl4ai.CacheMode

        async def _run() -> dict[str, str]:
            config = CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS,
                page_timeout=CRAWL4AI_TIMEOUT_MS,
                word_count_threshold=CRAWL4AI_WORD_COUNT_THRESHOLD,
                excluded_tags=["nav", "footer", "header", "aside", "form"],
                exclude_external_links=True,
                exclude_social_media_links=True,
            )
            out: dict[str, str] = {}
            async with AsyncWebCrawler() as crawler:
                results = await crawler.arun_many(
                    urls=allowed_urls, config=config,
                    max_concurrent=CRAWL4AI_MAX_CONCURRENT,
                )
                for r in results:
                    if not r or not getattr(r, "success", False):
                        continue
                    md = getattr(r, "markdown", None)
                    text = getattr(md, "fit_markdown", None) or md or ""
                    text = str(text).strip()
                    if text:
                        out[getattr(r, "url", "")] = text[:max_chars]
            return out

        return asyncio.run(_run())
    except Exception as e:
        log.info("[crawl4ai] batch fetch failed for %d url(s): %s", len(urls), e)
        return {}


# ── cross-source corroboration ("sources agreement" scoring) ────────────────

def _apply_corroboration_bonus(
    scored_chunks: list[tuple[float, str, str]],
    bonus: float = RESEARCH_AGREEMENT_BONUS,
    similarity_threshold: float = RESEARCH_AGREEMENT_SIMILARITY,
    shingle_size: int = RESEARCH_AGREEMENT_SHINGLE_SIZE,
) -> list[tuple[float, str, str, int]]:
    """Boost chunks whose content is independently corroborated by a
    DIFFERENT domain. Returns (score, url, chunk, corroboration_count)
    tuples — count=1 means single-source.

    Uses cheap word-shingle Jaccard similarity rather than a second
    embedding pass: good enough to catch two sources saying substantially
    the same thing, with no extra model calls on top of the relevance
    scoring _score_url_chunks already did.
    """
    def _domain(u: str) -> str:
        try:
            netloc = urlparse(u).netloc.lower()
            return netloc[4:] if netloc.startswith("www.") else netloc
        except Exception:
            return u

    def _shingles(text: str) -> set[str]:
        words = text.lower().split()
        n = shingle_size
        if len(words) < n:
            return {" ".join(words)} if words else set()
        return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}

    domains = [_domain(u) for _s, u, _c in scored_chunks]
    shingle_sets = [_shingles(c) for _s, _u, c in scored_chunks]
    counts = [1] * len(scored_chunks)

    for i in range(len(scored_chunks)):
        if not shingle_sets[i]:
            continue
        for j in range(i + 1, len(scored_chunks)):
            if domains[i] == domains[j] or not shingle_sets[j]:
                continue
            inter = len(shingle_sets[i] & shingle_sets[j])
            union = len(shingle_sets[i] | shingle_sets[j])
            if union and inter / union >= similarity_threshold:
                counts[i] += 1
                counts[j] += 1

    boosted = []
    for (score, url, chunk), count in zip(scored_chunks, counts):
        adjusted = min(1.0, score + bonus * (count - 1)) if count > 1 else score
        boosted.append((adjusted, url, chunk, count))
    return boosted


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
    fetch_fn=web_fetch,
    batch_prefetch_fn=None,
) -> tuple[list[tuple[float, str, str]], list[tuple[str, str]], list[tuple[str, str]]]:
    """Fetch multiple URLs concurrently, scoring each page's chunks for
    relevance the moment that page finishes downloading — not after every
    URL has finished.

    If batch_prefetch_fn is given (deep_research's Crawl4AI batch path), it's
    called ONCE with the full url list up front to grab as many pages as
    possible in a single browser session; only URLs it doesn't cover fall
    through to the per-URL thread-pool path using fetch_fn (e.g. web_fetch).
    deep_search never passes batch_prefetch_fn, so its behavior is unchanged.

    Returns (scored_chunks, pages, url_outcomes).
    """
    if not urls:
        return [], [], []

    log.info("[fetch_pipeline] attempting %d url(s): %s", len(urls), urls)

    scored: list[tuple[float, str, str]] = []
    pages: list[tuple[str, str]] = []
    url_outcomes: list[tuple[str, str]] = []
    chunks_scored = 0

    def _process(url: str, text: str) -> None:
        nonlocal chunks_scored
        if text.startswith("[fetch failed"):
            log.info("[fetch_pipeline] failed %s: %s", url, text)
            url_outcomes.append((url, text))
            return
        log.info("[fetch_pipeline] fetched %s (%d chars)", url, len(text))
        url_outcomes.append((url, f"ok ({len(text)} chars)"))
        pages.append((url, text))
        remaining_budget = max_chunks_to_score - chunks_scored
        if remaining_budget <= 0:
            return
        page_chunks = [(url, c) for c in reason.chunk_text(text, chunk_chars)][:remaining_budget]
        page_scored = _score_url_chunks(page_chunks, query, embedder, remaining_budget)
        scored.extend(page_scored)
        chunks_scored += len(page_scored)

    prefetched: dict[str, str] = {}
    if batch_prefetch_fn is not None:
        try:
            prefetched = batch_prefetch_fn(urls, max_chars_per_page) or {}
        except Exception as e:
            log.info("[fetch_pipeline] batch prefetch failed: %s", e)
            prefetched = {}

    for url, text in prefetched.items():
        _process(url, text)

    remaining_urls = [u for u in urls if u not in prefetched]
    if remaining_urls:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(remaining_urls)))) as pool:
            future_to_url = {pool.submit(fetch_fn, url, max_chars_per_page): url for url in remaining_urls}
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    text = future.result()
                except Exception as e:
                    log.warning("[fetch_pipeline] exception fetching %s: %s", url, e)
                    url_outcomes.append((url, f"exception: {e}"))
                    continue
                _process(url, text)

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
    annotate_agreement: bool = False,
    agreement_bonus: float = RESEARCH_AGREEMENT_BONUS,
    agreement_similarity: float = RESEARCH_AGREEMENT_SIMILARITY,
) -> str:
    """Dedup, filter, rank, and format already-scored chunks. Filtering is
    literal: chunks below min_score are dropped, not truncated or reworded.
    If nothing clears the bar, returns an explicit sentinel.

    When annotate_agreement is True (deep_research), chunks are first passed
    through the cross-source corroboration bonus, and each surfaced excerpt
    is tagged 'corroborated x2' or 'single-source, unverified' so a reader
    (or the synthesis LLM) can weight confidence accordingly.
    """
    if not scored_chunks:
        return "[no fetched content available to condense]"

    if annotate_agreement:
        working = _apply_corroboration_bonus(scored_chunks, agreement_bonus, agreement_similarity)
    else:
        working = [(score, url, chunk, 1) for score, url, chunk in scored_chunks]

    seen_hashes: set[str] = set()
    deduped: list[tuple[float, str, str, int]] = []
    for score, url, chunk, count in working:
        h = hashlib.sha1(chunk.strip().lower().encode("utf-8", "ignore")).hexdigest()
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        deduped.append((score, url, chunk, count))

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
    if annotate_agreement:
        corroborated_n = sum(1 for item in relevant if item[3] > 1)
        lines.append(
            f"[Source agreement: {corroborated_n}/{len(relevant)} excerpt(s) corroborated by "
            "an independent domain; treat the rest as single-source and unverified]"
        )
    for score, url, chunk, count in relevant:
        trust = f"corroborated x{count}" if count > 1 else "single-source, unverified"
        if annotate_agreement:
            lines.append(f"[source: {url} | relevance: {score:.2f} | {trust}]\n{chunk}")
        else:
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
    embedder,
    *,
    num_searches: int = 1,
    num_fetches: int = 0,
    max_chars_per_page: int = 4000,
    max_workers: int = 3,
    exclude_urls: set[str] | None = None,
    fetch_fn=web_fetch,
    batch_prefetch_fn=None,
    respect_robots: bool = False,
    annotate_agreement: bool = False,
    condense_top_k: int = CONDENSE_TOP_K,
    condense_min_score: float = CONDENSE_MIN_SCORE,
    condense_max_chunks_to_score: int = CONDENSE_MAX_CHUNKS_TO_SCORE,
    expand_sitemap: bool = False,
    sitemap_max_urls: int = RESEARCH_SITEMAP_MAX_URLS,
) -> tuple[str, set[str]]:
    """Fixed, non-adaptive search pass with optional fetch/condense.

    With num_fetches=0 this returns snippets/URLs only, which is now the
    default public deep_search behavior. Deep_research calls this helper with
    its own positive fetch count plus the research-only knobs (Crawl4AI batch
    prefetch, robots gating, sitemap expansion, corroboration annotation) to
    do fetched-source work.

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

    if expand_sitemap and deduped_results:
        sitemap_urls: list[str] = []
        seen_domains: set[str] = set()
        for r in deduped_results[:3]:
            url = (r.get("url") or "").strip()
            if not url:
                continue
            try:
                parts = urlparse(url)
                domain = parts.netloc
                origin = f"{parts.scheme}://{domain}"
            except Exception:
                continue
            if not domain or domain in seen_domains:
                continue
            seen_domains.add(domain)
            found = _discover_sitemap_urls(origin, query, max_urls=sitemap_max_urls)
            sitemap_urls.extend(u for u in found if u not in fetch_urls and u not in exclude_urls and u not in sitemap_urls)
            if len(seen_domains) >= 2:
                break
        if sitemap_urls:
            log.info("[deep_research] sitemap expansion added %d url(s) for: %s", len(sitemap_urls[:sitemap_max_urls]), query)
            fetch_urls = fetch_urls + sitemap_urls[:sitemap_max_urls]

    skipped_outcomes: list[tuple[str, str]] = []
    if respect_robots:
        allowed_fetch_urls = []
        for u in fetch_urls:
            if _source_agreement_allows(u):
                allowed_fetch_urls.append(u)
            else:
                skipped_outcomes.append((u, "[skipped: disallowed by robots.txt]"))
        fetch_urls = allowed_fetch_urls

    scored_chunks, fetched_pages, url_outcomes = _fetch_and_score_pipeline(
        fetch_urls, query, embedder, max_chars_per_page, max_workers=max_workers,
        max_chunks_to_score=condense_max_chunks_to_score,
        fetch_fn=fetch_fn, batch_prefetch_fn=batch_prefetch_fn,
    )
    url_outcomes = skipped_outcomes + url_outcomes
    manifest = _format_url_manifest(url_outcomes)
    fetched_url_set = {url for url, _text in fetched_pages}

    if not fetched_pages:
        return f"{snippet_bundle}\n\n{manifest}", fetched_url_set

    condensed = _finalize_condensed(
        scored_chunks, query, top_k=condense_top_k, min_score=condense_min_score,
        annotate_agreement=annotate_agreement,
    )
    return f"{snippet_bundle}\n\n{manifest}\n\n{condensed}", fetched_url_set


def deep_search(
    query: str,
    num_searches: int = DEEP_SEARCH_NUM_SEARCHES,
    num_fetches: int = DEEP_SEARCH_NUM_FETCHES,
    max_chars_per_page: int = DEEP_SEARCH_MAX_CHARS_PER_PAGE,
    max_workers: int = DEEP_SEARCH_MAX_WORKERS,
    embedder=None,
) -> str:
    """Agentic snippet-only search — the FAST path.

    By default num_fetches is 0, so this returns SearXNG result snippets/URLs
    only and never reads full pages. The num_fetches argument is retained as a
    backwards-compatible escape hatch; keep it 0 for the intended behavior.
    Deliberately never uses Crawl4AI, robots gating, or sitemap expansion —
    those are deep_research-only."""
    text, _urls = _deep_search_impl(query, embedder, num_searches=num_searches, num_fetches=num_fetches, max_chars_per_page=max_chars_per_page, max_workers=max_workers)
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
    use_crawl4ai: bool = RESEARCH_USE_CRAWL4AI,
    respect_robots: bool = RESEARCH_RESPECT_ROBOTS,
    expand_sitemap: bool = RESEARCH_SITEMAP_ENABLED,
    condense_top_k: int = RESEARCH_CONDENSE_TOP_K,
    condense_min_score: float = RESEARCH_CONDENSE_MIN_SCORE,
    condense_max_chunks_to_score: int = RESEARCH_CONDENSE_MAX_CHUNKS_TO_SCORE,
) -> str:
    """Multi-round adaptive fetched-source research — the ACCURATE path.

    Uses search only to discover candidate URLs, optionally widens those
    candidates via sitemap discovery, fetches full pages (Crawl4AI batch
    session preferred, web_fetch fallback), only ever fetching URLs the
    source's robots.txt permits, condenses evidence with a cross-source
    corroboration bonus, and optionally asks the LLM whether another fetched
    round/refined query is needed. This is the heavy research/self-learning
    tool; deep_search remains snippet-only and fast.

    num_searches/num_fetches/max_chars_per_page/condense_* default to this
    module's DEEP_RESEARCH_*/RESEARCH_CONDENSE_* env-backed constants but are
    real function arguments — every round's underlying _deep_search_impl()
    call uses whatever was passed in here, not the module constants
    directly. This is what lets a caller (e.g. memory.learn.quick_studying)
    override per-call instead of only ever getting the env-var defaults.
    """
    if not query or not query.strip():
        return "[search failed: empty query]"

    fetch_fn = web_fetch
    batch_prefetch_fn = functools.partial(_crawl4ai_fetch_many) if use_crawl4ai else None

    rounds_text: list[str] = []
    seen_urls: set[str] = set()
    queries_used: list[str] = [query.strip()]
    current_query = query.strip()
    adaptive = client is not None and model

    for round_num in range(1, max_rounds + 1):
        log.info("[deep_research] round %d searching: %s", round_num, current_query)
        round_text, round_urls = _deep_search_impl(
            current_query,
            embedder,
            num_searches=num_searches,
            num_fetches=num_fetches,
            max_chars_per_page=max_chars_per_page,
            max_workers=DEEP_SEARCH_MAX_WORKERS,
            exclude_urls=seen_urls,
            fetch_fn=fetch_fn,
            batch_prefetch_fn=batch_prefetch_fn,
            respect_robots=respect_robots,
            annotate_agreement=True,
            condense_top_k=condense_top_k,
            condense_min_score=condense_min_score,
            condense_max_chunks_to_score=condense_max_chunks_to_score,
            expand_sitemap=expand_sitemap,
            sitemap_max_urls=RESEARCH_SITEMAP_MAX_URLS,
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
            "concise, well-organized answer to the original question. Each "
            "excerpt is tagged either 'corroborated xN' (confirmed by N "
            "independent sources) or 'single-source, unverified'. Prefer "
            "corroborated claims; explicitly flag single-source claims as "
            "such rather than stating them with full confidence. Note any "
            "unresolved gaps or conflicting information explicitly. If "
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
