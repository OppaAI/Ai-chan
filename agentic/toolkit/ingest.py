"""
toolkit/ingest.py

URL → bytes → extracted text. The "how do I read what's at this URL?"
module.

Handles one specific URL at a time: download with size capping, SSRF
guard, content-type sniffing, and MarkItDown document conversion
(PDF/DOCX/XLSX/PPTX/EPUB/CSV/XML/ZIP/IPYNB/MSG/HTML).

This is a *resource-oriented* module — you already know the URL and
just need its content. It does NOT search, does NOT crawl, does NOT
rank, does NOT format for chat display. Those live in toolkit/websearch.py.

Provides `fetch_from_url` — a lightweight registered tool for agents
that just want to download and extract text from one URL without the
overhead of deep_read's Crawl4AI escalation or relevance filtering.
"""
from __future__ import annotations

import importlib
import importlib.util
import io
import ipaddress
import os
import socket
import tempfile
import time
from urllib.parse import urlparse

from agentic.registry import TOOLS, tool

FETCH_URL_USER_AGENT = os.getenv(
    "FETCH_URL_USER_AGENT",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36",
)

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
    "text/html": "html",
    "application/xhtml+xml": "html",
}
_MARKITDOWN_EXTENSION_MAP = {
    ".pdf": "pdf", ".docx": "docx", ".pptx": "pptx", ".xlsx": "xlsx",
    ".epub": "epub", ".csv": "csv", ".xml": "xml", ".zip": "zip",
    ".ipynb": "ipynb", ".msg": "msg", ".html": "html", ".htm": "html",
}
_MARKITDOWN_SUFFIX_MAP = {
    "pdf": ".pdf", "docx": ".docx", "pptx": ".pptx", "xlsx": ".xlsx",
    "epub": ".epub", "csv": ".csv", "xml": ".xml", "zip": ".zip",
    "ipynb": ".ipynb", "msg": ".msg", "html": ".html",
}

FETCH_FROM_URL_MAX_CHARS = int(os.getenv("FETCH_FROM_URL_MAX_CHARS", 4000))


def _check_host_ssrf(hostname: str) -> bool:
    """Check whether a hostname resolves to a private, local, or reserved IP.

    Used as a security guard: any URL-fetching primitive rejects hosts that
    resolve to private/loopback/link-local/multicast ranges to prevent SSRF
    attacks.

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


def ingest_from_url(
    url: str,
    max_bytes: int,
    timeout: int = 8,
) -> tuple[bytes | None, str | None]:
    """Stream-download a URL, aborting mid-stream when the size limit is exceeded.

    The shared size-guard for all fetch paths. Callers are responsible for
    scheme/host validation before calling this.

    Args:
        url: URL to download.
        max_bytes: Hard byte limit on the response body.
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
            headers={"User-Agent": FETCH_URL_USER_AGENT},
        ) as resp:
            resp.raise_for_status()

            content_length = resp.headers.get("content-length")
            if content_length is not None:
                try:
                    if int(content_length) > max_bytes:
                        return None, "[fetch failed: page too large]"
                except ValueError:
                    pass

            buf = io.BytesIO()
            total = 0
            for chunk in resp.iter_content(chunk_size=65536):
                if not chunk:
                    continue
                total += len(chunk)
                if total > max_bytes:
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
                headers={"User-Agent": FETCH_URL_USER_AGENT},
            )
            ctype = resp.headers.get("content-type", "").split(";")[0].strip().lower()
            if ctype in _MARKITDOWN_CONTENT_TYPE_MAP:
                return _MARKITDOWN_CONTENT_TYPE_MAP[ctype]
            if ctype.startswith("text/html") or ctype.startswith("application/xhtml"):
                return "html"
        except Exception:
            pass

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


@tool(TOOLS["fetch_from_url"])
def fetch_from_url(url: str, max_chars: int = FETCH_FROM_URL_MAX_CHARS) -> str:
    """Download and extract text from any URL.

    Lightweight fetch — no JS rendering, no relevance filtering, no
    Crawl4AI escalation. Routes HTML through MarkItDown conversion,
    non-HTML documents through the same MarkItDown pipeline.

    Args:
        url: The URL to fetch.
        max_chars: Maximum characters to return.

    Returns:
        Extracted text on success, or a bracketed error string starting
        with "[fetch failed:" on failure.
    """
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return f"[fetch failed: unsupported URL scheme: {parsed.scheme or 'none'}]"
    if _check_host_ssrf(parsed.hostname):
        return "[fetch failed: URL host is not allowed]"

    downloaded, error = ingest_from_url(url, max_bytes=5_000_000)
    if error:
        return error

    content_type = _sniff_content_type(url)
    if content_type == "html":
        raw = downloaded.decode("utf-8", errors="replace")
        return raw[:max_chars]

    result = _extract_with_markitdown(downloaded, content_type, max_chars)
    return result
