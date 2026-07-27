"""
toolkit/websearch.py

Search the web, then read what you find. The "find me information"
module.

Provides a two-step pipeline:
  1. **Search** — query a SearXNG instance, get structured results
  2. **Fetch** — read one result URL via trafilatura (HTML article extraction)

This is a *discovery-oriented* module — you have a question and need
to find relevant pages, then read them. It does NOT handle non-HTML
documents (PDF/DOCX/etc.), does NOT do SSRF checks on its own (delegates
to ingest.py), and does NOT do multi-round research. Those live in
toolkit/ingest.py and toolkit/research.py respectively.

Public API:
  - web_search()              — SearXNG query → structured results (list[dict])
  - web_search_context()      — search → formatted numbered snippets for chat
  - web_search_and_fetch()    — search + fetch top result's extracted text
  - web_fetch()               — single known URL → extracted text (HTML/trafilatura)

Requires a running SearXNG instance (SEARXNG_URL env var).
"""

from __future__ import annotations
 
import io
import os
import re
import time
from dataclasses import dataclass, field
from urllib.parse import urlparse
 
import importlib
import importlib.util
 
from cognition import reason
from system.log import get_logger
from agentic.toolkit.cache import TTLCache
from agentic.toolkit.ingest import ingest_from_url, FETCH_URL_USER_AGENT, _check_host_ssrf

log = get_logger(__name__)
 
# ── Config ──────────────────────────────────────────────────────────────
SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:8888")
SEARXNG_MAX_RESULTS = int(os.getenv("SEARXNG_MAX_RESULTS", 5))
SEARXNG_TIMEOUT_SECONDS = int(os.getenv("SEARXNG_TIMEOUT_SECONDS", 8))
SEARXNG_MAX_RETRIES = int(os.getenv("SEARXNG_MAX_RETRIES", 3))
SEARXNG_RETRY_BASE_DELAY = float(os.getenv("SEARXNG_RETRY_BASE_DELAY", 1.0))
SEARXNG_RATE_LIMIT_DELAY = float(os.getenv("SEARXNG_RATE_LIMIT_DELAY", 2.0))
 
# -- web_fetch download guard --
WEB_FETCH_MAX_DOWNLOAD_BYTES = int(os.getenv("WEB_FETCH_MAX_DOWNLOAD_BYTES", 5_000_000))
WEB_FETCH_TIMEOUT_SECONDS = int(os.getenv("WEB_FETCH_TIMEOUT_SECONDS", 8))
WEB_FETCH_MAX_CHARS = int(os.getenv("WEB_FETCH_MAX_CHARS", 4000))
# ── Cache instance ──────────────────────────────────────────────────────
# -- shared TTL cache instances --
# Both search and fetch operations use the same TTL window but separate
# cache instances, allowing independent tuning if needed later.
_SEARCH_CACHE = TTLCache(
    ttl_seconds=int(os.getenv("TOOLS_CACHE_TTL_SECONDS", 900)),
    max_entries=int(os.getenv("TOOLS_CACHE_MAX_ENTRIES", 256)),
)
 
_FETCH_CACHE = TTLCache(
    ttl_seconds=int(os.getenv("TOOLS_CACHE_TTL_SECONDS", 900)),
    max_entries=int(os.getenv("TOOLS_CACHE_MAX_ENTRIES", 256)),
)

# ── Public API ──────────────────────────────────────────────────────────

def web_search(
    query: str, max_results: int, pageno: int = 1
) -> tuple[list[dict] | None, str | None]:
    """Search the web via SearXNG. Returns structured results.

    The core search primitive — unformatted structured data suitable for
    ranking, filtering, and programmatic use. For chat-display formatting
    use web_search_context().

    Handles retries on transient failures and caches results.

    Args:
        query: Search query string.
        max_results: Maximum results to return for this page.
        pageno: Pagination number (1-indexed, default 1).

    Returns:
        (results_list, None) on success, (None, error_msg) on failure.
        results_list is a list of dicts with 'title', 'url', 'content' keys.
        Returns empty list [] if SearXNG finds no results (not an error).
    """
    if not query.strip():
        return [], "[search failed: empty query]"

    cache_key = f"{query}|{max_results}|{pageno}"
    cached = _SEARCH_CACHE.get(cache_key)
    if cached is not None:
        return cached, None

    if importlib.util.find_spec("requests") is None:
        return None, "[search failed: requests is not installed]"
    requests = importlib.import_module("requests")
    
    last_error = None
    for attempt in range(SEARXNG_MAX_RETRIES):
        try:
            response = requests.get(
                f"{SEARXNG_URL}/search",
                params={"q": query, "format": "json", "pageno": pageno},
                timeout=SEARXNG_TIMEOUT_SECONDS,
            )
            if response.status_code == 429:
                time.sleep(SEARXNG_RATE_LIMIT_DELAY * (attempt + 1))
                continue
            response.raise_for_status()
            data = response.json()
            break
        except requests.exceptions.ConnectionError as e:
            last_error = f"[search failed: {e}]"
            time.sleep(SEARXNG_RETRY_BASE_DELAY * (attempt + 1))
            continue
        except ValueError:
            return None, "[search failed: invalid JSON response]"
        except requests.exceptions.RequestException as e:
            return None, f"[search failed: {e}]"
    else:
        return None, last_error or "[search failed: max retries]"

    results = data.get("results", [])[:max_results]
    _SEARCH_CACHE.set(cache_key, results)
    return results, None


def web_fetch(
    url: str,
    max_chars: int = WEB_FETCH_MAX_CHARS,
    max_download_bytes: int = WEB_FETCH_MAX_DOWNLOAD_BYTES,
    use_cache: bool = True,
) -> str:
    """Fetch a single URL and extract its main article/body text with trafilatura.

    The baseline "fetch a page" primitive — fast, dependency-light, no JS
    rendering. Downloads are streamed and capped at max_download_bytes BEFORE
    extraction and BEFORE max_chars truncation, bounding worst-case memory.

    Args:
        url: URL to fetch.
        max_chars: Maximum characters to return from extracted text.
        max_download_bytes: Hard byte limit on the HTTP response body.
        use_cache: If True, cache successful fetches in-process.

    Returns:
        Extracted text on success, or a bracketed error string starting with
        "[fetch failed:" on failure.
    """
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return f"[fetch failed: unsupported URL scheme: {parsed.scheme or 'none'}]"
    if _check_host_ssrf(parsed.hostname):
        return "[fetch failed: URL host is not allowed]"
 
    cache_key = f"{url}|{max_chars}"
    if use_cache:
        cached = _FETCH_CACHE.get(cache_key)
        if cached is not None:
            return cached
 
    if importlib.util.find_spec("trafilatura") is None:
        return "[fetch failed: trafilatura is not installed]"
    trafilatura = importlib.import_module("trafilatura")
 
    downloaded, error = ingest_from_url(url, max_download_bytes)
    if error:
        return error
 
    try:
        text = trafilatura.extract(downloaded, include_links=False, include_tables=False) or ""
    except Exception as e:
        return f"[fetch failed: {e}]"
 
    result = text[:max_chars] if text else "[fetch failed: no extractable text]"
    if use_cache and text:
        _FETCH_CACHE.set(cache_key, result)
    return result


def web_search_context(query: str, max_results: int = SEARXNG_MAX_RESULTS) -> str | None:
    """Search the web and return numbered snippets as chat context.

    Calls web_search() for structured results, then formats them as a
    readable numbered list. Returns None on failure so callers can
    fall back gracefully.

    Args:
        query: Search query string.
        max_results: Number of results to fetch (default SEARXNG_MAX_RESULTS=5).

    Returns:
        Formatted numbered list on success, or None if the search failed
        or returned no results.
    """
    if not query or not query.strip():
        return None
    results, error = web_search(query, max_results, pageno=1)
    if error or not results:
        return None

    lines = [f"[Web search results for: {query}]"]
    for i, result in enumerate(results, 1):
        title = result.get("title", "").strip()
        url = result.get("url", "").strip()
        content = result.get("content", "").strip()
        lines.append(f"{i}. {title}\n   {url}\n   {content}")

    return f"{'\n\n'.join(lines)}\n\nUser asked: {query}"


def web_search_and_fetch(query: str, max_results: int = SEARXNG_MAX_RESULTS) -> str:
    """Search the web and fetch the top result's content.

    Calls web_search() for structured results, formats them as a numbered
    list, then fetches the top result's URL via web_fetch().

    Args:
        query: Search query string.
        max_results: Number of search results to fetch (default 5).

    Returns:
        Formatted string with search results and fetched content, or error.
    """
    results, error = web_search(query, max_results, pageno=1)
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

    search_text = "\n\n".join(lines)
    top_url = results[0].get("url", "").strip()
    if not top_url:
        return search_text

    fetch_result = web_fetch(top_url, max_chars=WEB_FETCH_MAX_CHARS)
    return f"{search_text}\n\n---\n\nFetched content:\n{fetch_result}"





