"""
toolkit/websurf.py
 
Web search and fetch primitives.
 
Pure building blocks — no multi-step workflows. Callers compose
these via the graph executor (research_graph.py) or ReAct loop.
 
  - fetch_search_results()     — low-level SearXNG call (raw JSON)
  - web_fetch()                — single URL → extracted text (HTML/trafilatura)
  - web_search()              — SearXNG query → numbered snippets
  - web_search_context()       — chat-mode wrapper around web_search
  - web_search_and_fetch()     — web_search + fetch top result
 
Requires a running SearXNG instance (SEARXNG_URL env var).
"""

from __future__ import annotations
 
import concurrent.futures
import io
import ipaddress
import os
import re
import socket
import tempfile
import time
from dataclasses import dataclass, field
from urllib.parse import urlparse
 
import importlib
import importlib.util
 
from cognition import reason
from system.log import get_logger
from agentic.toolkit.cache import TTLCache 

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
WEB_FETCH_USER_AGENT = os.getenv(
    "WEB_FETCH_USER_AGENT",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36",
)

# MarkItDown handles non-HTML document formats by converting to markdown.
# Base install only (`pip install markitdown`) covers all of these with no
# heavy optional deps — deliberately NOT enabling the image-OCR or
# audio-transcription extras here, since those pull real weight for
# capabilities deep_read doesn't need.
_MARKITDOWN_CONTENT_TYPE_MAP = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "application/epub+zip": "epub",
    "text/csv": "csv",
    "application/xml": "xml",
    "text/xml": "xml",
    "application/zip": "zip",
    "application/x-ipynb+json": "ipynb",
}
_MARKITDOWN_EXTENSION_MAP = {
    ".pdf": "pdf", ".docx": "docx", ".pptx": "pptx", ".xlsx": "xlsx",
    ".epub": "epub", ".csv": "csv", ".xml": "xml", ".zip": "zip",
    ".ipynb": "ipynb", ".msg": "msg",
}
_MARKITDOWN_SUFFIX_MAP = {
    "pdf": ".pdf", "docx": ".docx", "pptx": ".pptx", "xlsx": ".xlsx",
    "epub": ".epub", "csv": ".csv", "xml": ".xml", "zip": ".zip",
    "ipynb": ".ipynb", "msg": ".msg",
}

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

def fetch_search_results(
    query: str, max_results: int, pageno: int = 1
) -> tuple[list[dict] | None, str | None]:
    """Fetch raw search results from SearXNG backend.

    Handles retries on transient failures and caches results. Returns
    unformatted structured data suitable for ranking and filtering.

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
    max_chars: int = 4000,
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
    if _is_private_or_local_host(parsed.hostname):
        return "[fetch failed: URL host is not allowed]"
 
    cache_key = f"{url}|{max_chars}"
    if use_cache:
        cached = _FETCH_CACHE.get(cache_key)
        if cached is not None:
            return cached
 
    if importlib.util.find_spec("trafilatura") is None:
        return "[fetch failed: trafilatura is not installed]"
    trafilatura = importlib.import_module("trafilatura")
 
    downloaded, error = _download_bytes(url, max_download_bytes)
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


def web_search(query: str, max_results: int = SEARXNG_MAX_RESULTS) -> str:
    """
    Search the web and return formatted numbered snippets.
    
    User-friendly wrapper around _fetch_search_results(). Formats raw results
    as a readable numbered list. For structured data (e.g., ranking), use
    _fetch_search_results() directly.
    
    Args:
        query: Search query string.
        max_results: Number of results to fetch (default SEARXNG_MAX_RESULTS=5).

    Returns:
        Formatted string on success, or a bracketed error string starting
        with "[search failed:" or "[no results found for:".

    Notes:
        Always returns a string; never raises exceptions.
        Results cached same as fetch_search_results().
        Only fetches page 1 (use fetch_search_results for pagination).
    """
    if not query.strip():
        return "[search failed: empty query]"
    results, error = fetch_search_results(query, max_results, pageno=1)
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


def web_search_context(query: str, max_results: int = SEARXNG_MAX_RESULTS) -> str | None:
    """Run web_search and wrap successful results as chat context.

    Args:
        query: Search query string.
        max_results: Number of results to fetch (default SEARXNG_MAX_RESULTS=5).

    Returns:
        Search results with "User asked: {query}" appended, or None if the
        search failed or returned no results.
    """

    if not query or not query.strip():
        return "[search failed: empty query]"
    results = web_search(query, max_results)
    if results.startswith("[search failed") or results.startswith("[no results"):
        return None
    return f"{results}\n\nUser asked: {query}"


def web_search_and_fetch(query: str, max_results: int = SEARXNG_MAX_RESULTS) -> str:
    """
    Perform web search and fetch top results.

    Executes web_search() then web_fetch() on the top result URL.
    Returns formatted search results + fetched content, or error message.

    Args:
        query: Search query string.
        max_results: Number of search results to fetch (default 5).

    Returns:
        Formatted string with search results and fetched content, or error.
    """
    search_result = web_search(query, max_results)
    if search_result.startswith("[search failed") or search_result.startswith("[no results"):
        return search_result

    # Extract URL from top result
    lines = search_result.split("\n")
    if len(lines) < 4:
        return "[fetch failed: unexpected search result format]"

    url_line = lines[2]
    match = re.search(r"^\s*(https?://[^\s]+)", url_line)
    if not match:
        return "[fetch failed: could not extract URL from search result]"

    url = match.group(1)
    fetch_result = web_fetch(url, max_chars=4000)

    return f"{search_result}\n\n---\n\nFetched content:\n{fetch_result}"

# ── Private helpers ─────────────────────────────────────────────────────

def _is_private_or_local_host(hostname: str) -> bool:
    """Check whether a hostname resolves to a private, local, or reserved IP.

    Used as a security guard: web_fetch and any other URL-fetching primitive
    rejects hosts that resolve to private/loopback/link-local/multicast ranges
    to prevent SSRF attacks.

    Args:
        hostname: The hostname to resolve (e.g. "192.168.1.1", "localhost").

    Returns:
        True if the hostname is private/local/unroutable, False if it appears
        to be a public routable address. Returns True on resolution failure
        (fail-closed).
    """
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


def _download_bytes(
    url: str,
    max_download_bytes: int,
    timeout: int = WEB_FETCH_TIMEOUT_SECONDS,
) -> tuple[bytes | None, str | None]:
    """Stream-download a URL, aborting mid-stream when the size limit is exceeded.

    The single shared size-guard for all fetch paths. Callers are responsible
    for scheme/host validation before calling this.

    Args:
        url: URL to download.
        max_download_bytes: Hard byte limit on the response body.
        timeout: Request timeout in seconds.

    Returns:
        (bytes, None) on success, (None, error_string) on failure.
    """
    if importlib.util.find_spec("requests") is None:
        return None, "[fetch failed: requests is not installed]"
    requests = importlib.import_module("requests")
    try:
        with requests.get(
            url,
            stream=True,
            timeout=timeout,
            headers={"User-Agent": WEB_FETCH_USER_AGENT},
        ) as resp:
            resp.raise_for_status()

            content_length = resp.headers.get("content-length")
            if content_length is not None:
                try:
                    if int(content_length) > max_download_bytes:
                        return None, "[fetch failed: page too large]"
                except ValueError:
                    pass

            buf = io.BytesIO()
            total = 0
            for chunk in resp.iter_content(chunk_size=65536):
                if not chunk:
                    continue
                total += len(chunk)
                if total > max_download_bytes:
                    return None, "[fetch failed: page exceeded size limit during download]"
                buf.write(chunk)
            downloaded = buf.getvalue()
    except requests.exceptions.RequestException as e:
        return None, f"[fetch failed: {e}]"

    if not downloaded:
        return None, "[fetch failed: empty response]"
    return downloaded, None


def _sniff_content_type(url: str) -> str:
    """Classify a URL's content type for deep_read's content-type routing.

    Tries a cheap HEAD request first (no body download), then falls back to
    the URL's file extension. Returns 'html' on total ambiguity.

    Args:
        url: The URL to classify.

    Returns:
        'html' (default/fallback) or a MarkItDown-handled format: 'pdf',
        'docx', 'pptx', 'xlsx', 'epub', 'csv', 'xml', 'zip', 'ipynb', 'msg'.
    """
    parsed = urlparse(url)
    ext = os.path.splitext(parsed.path)[1].lower()

    if importlib.util.find_spec("requests") is not None:
        requests = importlib.import_module("requests")
        try:
            resp = requests.head(
                url, timeout=5, allow_redirects=True,
                headers={"User-Agent": WEB_FETCH_USER_AGENT},
            )
            ctype = resp.headers.get("content-type", "").split(";")[0].strip().lower()
            if ctype in _MARKITDOWN_CONTENT_TYPE_MAP:
                return _MARKITDOWN_CONTENT_TYPE_MAP[ctype]
            if ctype.startswith("text/html") or ctype.startswith("application/xhtml"):
                return "html"
        except Exception:
            pass  # fall through to extension sniff below

    return _MARKITDOWN_EXTENSION_MAP.get(ext, "html")


def _extract_with_markitdown(data: bytes, content_type: str, max_chars: int) -> str:
    """Convert non-HTML document bytes to markdown text via MarkItDown.

    Writes bytes to a suffix-matched temp file for reliable format detection.

    Args:
        data: Raw document bytes.
        content_type: One of the MarkItDown-handled format keys ('pdf',
            'docx', 'pptx', 'xlsx', 'epub', 'csv', 'xml', 'zip', 'ipynb',
            'msg').
        max_chars: Maximum characters to return.

    Returns:
        Markdown text on success, or a bracketed error string starting with
        "[fetch failed:" on failure.
    """
    if importlib.util.find_spec("markitdown") is None:
        return "[fetch failed: markitdown is not installed]"
    markitdown_mod = importlib.import_module("markitdown")
    MarkItDown = markitdown_mod.MarkItDown

    suffix = _MARKITDOWN_SUFFIX_MAP.get(content_type, "")
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            tmp.write(data)
            tmp.flush()
            converter = MarkItDown()
            result = converter.convert(tmp.name)
            text = (getattr(result, "text_content", None) or "").strip()
    except Exception as e:
        return f"[fetch failed: markitdown conversion error: {e}]"

    if not text:
        return "[fetch failed: markitdown returned no extractable text]"
    return text[:max_chars]


# ── batch fetch + concurrent relevance scoring ──────────────────────────

def _fetch_and_score_pipeline(
    urls: list[str],
    query: str,
    embedder,
    max_chars_per_page: int,
    chunk_chars: int = 500,
    max_workers: int = 4,
    max_chunks_to_score: int = 60,
    fetch_fn=web_fetch,
    batch_prefetch_fn=None,
) -> tuple[list[tuple[float, str, str]], list[tuple[str, str]], list[tuple[str, str]]]:
    """Fetch multiple URLs concurrently and score each page's chunks for relevance.

    Processes pages as they finish (not after every URL completes). If
    batch_prefetch_fn is given, it is called once with the full URL list up
    front; uncovered URLs fall through to the per-URL thread-pool.

    Args:
        urls: URLs to fetch.
        query: Relevance query for chunk scoring.
        embedder: Embedder instance for semantic scoring.
        max_chars_per_page: Max characters to fetch per URL.
        chunk_chars: Character size for chunking page text.
        max_workers: Max concurrent fetches.
        max_chunks_to_score: Max total chunks to score across all pages.
        fetch_fn: Per-URL fetch function (default web_fetch).
        batch_prefetch_fn: Optional batch prefetch function for Crawl4AI.

    Returns:
        (scored_chunks, pages, url_outcomes) tuple:
        - scored_chunks: list of (score, url, chunk_text)
        - pages: list of (url, full_text) for successful fetches
        - url_outcomes: list of (url, status_string)
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
        from agentic.toolkit.research import _score_url_chunks
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