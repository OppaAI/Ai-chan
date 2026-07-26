"""
toolkit/websurf.py
 
Web search and fetch primitives.
 
Pure building blocks — no multi-step workflows. Callers compose
these via the graph executor (research_graph.py) or ReAct loop.
 
  - web_search()              — SearXNG query → numbered snippets
  - web_fetch()                — single URL → extracted text (HTML/trafilatura)
  - web_search_context()       — chat-mode wrapper around web_search
  - _web_search_raw()          — low-level SearXNG call (raw JSON)
  - _download_bytes()          — shared streamed/size-capped byte download
  - _sniff_content_type()      — HEAD/extension based format classification
  - _extract_with_markitdown() — non-HTML document → markdown text
  - _fetch_and_score_pipeline() — batch fetch + concurrent relevance scoring
 
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
 
SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:8888")
MAX_RESULTS = int(os.getenv("SEARXNG_MAX_RESULTS", 5))
 
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


def _web_search_raw(query: str, max_results: int, pageno: int = 1) -> tuple[list[dict] | None, str | None]:
    """Low-level SearXNG call returning (results, error). Cached in-process
    via TTLCache for TOOLS_CACHE_TTL_SECONDS, keyed on (query, max_results, pageno).
    
    Returns (results, None) on success, or (None, error_string) on failure.
    """
    cache_key = f"{query}|{max_results}|{pageno}"
    
    # Check cache first
    cached = _SEARCH_CACHE.get(cache_key)
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
    _SEARCH_CACHE.set(cache_key, results)  # <-- NEW: Use TTLCache.set()
    return results, None
 
 
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
    Crawl4AI misses or when it isn't installed.
 
    Downloads are streamed and capped at max_download_bytes via
    _download_bytes, aborted mid-stream BEFORE trafilatura ever runs
    extraction and BEFORE max_chars truncation — this is what bounds
    worst-case memory for a single fetch.
 
    Successful fetches are cached in-process for TOOLS_CACHE_TTL_SECONDS keyed on
    (url, max_chars). Failed fetches are never cached.
    
    REFACTORED: Uses TTLCache.get/set instead of _cache_get/_cache_set.
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


def _download_bytes(
    url: str,
    max_download_bytes: int,
    timeout: int = WEB_FETCH_TIMEOUT_SECONDS,
) -> tuple[bytes | None, str | None]:
    """Stream-download a URL, aborting mid-stream the moment
    max_download_bytes is exceeded — BEFORE any extraction step runs. This
    is the single shared size-guard: web_fetch's trafilatura/HTML path and
    deep_read's MarkItDown non-HTML path both call this instead of each
    re-implementing their own streaming loop, so the worst-case-memory bound
    only needs to be correct in one place.

    Returns (bytes, None) on success, or (None, error_string) on failure.
    Callers are responsible for scheme/host validation before calling this —
    it does not repeat the private/local-host or scheme checks web_fetch
    already does, since deep_read's content-type sniff runs first and both
    callers share the same validated URL.
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
    Crawl4AI misses or when it isn't installed.

    Downloads are streamed and capped at max_download_bytes via
    _download_bytes, aborted mid-stream BEFORE trafilatura ever runs
    extraction and BEFORE max_chars truncation — this is what bounds
    worst-case memory for a single fetch.

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
        _cache_set(_fetch_cache, cache_key, result)
    return result


def _sniff_content_type(url: str) -> str:
    """Best-effort content-type classification for research.py's deep_read
    routing: returns 'html' (default/fallback) or one of the MarkItDown-
    handled non-HTML kinds ('pdf', 'docx', 'pptx', 'xlsx', 'epub', 'csv',
    'xml', 'zip', 'ipynb', 'msg').

    Tries a cheap HEAD request first (no body download) and reads the
    Content-Type header; falls back to the URL's file extension if HEAD is
    blocked, times out, or the server omits/mislabels the header. Fails open
    to 'html' on total ambiguity — the HTML path degrading gracefully
    (via web_fetch's own "[fetch failed: no extractable text]") is a safer
    default than guessing a document format wrong.
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
    """Convert non-HTML document bytes (pdf/docx/pptx/xlsx/epub/csv/xml/zip/
    ipynb/msg) to markdown text via MarkItDown.

    MarkItDown's convert() wants a file path or file-like object with a
    recognizable extension; bytes are written to a suffix-matched temp file
    rather than relying on an in-memory stream, since MarkItDown's format
    detection is more reliable off a real extension than a bare stream.

    Requires the base `markitdown` package only (`pip install markitdown`)
    — no OCR/audio extras. Returns a bracketed failure string, same
    convention as web_fetch, on any failure so callers can check
    text.startswith("[fetch failed") uniformly regardless of which
    extraction path produced the text.
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
    """Fetch multiple URLs concurrently, scoring each page's chunks for
    relevance the moment that page finishes downloading — not after every
    URL has finished.

    If batch_prefetch_fn is given (deep_research's Crawl4AI batch path), it's
    called ONCE with the full url list up front to grab as many pages as
    possible in a single browser session; only URLs it doesn't cover fall
    through to the per-URL thread-pool path using fetch_fn (e.g. web_fetch).

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


def web_search_context(query: str, max_results: int = MAX_RESULTS) -> str | None:
    """Run web_search and wrap successful results as context for chat mode."""
    if not query or not query.strip():
        return "[search failed: empty query]"
    results = web_search(query, max_results)
    if results.startswith("[search failed") or results.startswith("[no results"):
        return None
    return f"{results}\n\nUser asked: {query}"