"""
toolkit/ingest.py

Document and file ingestion: convert non-HTML documents (PDF, DOCX, XLSX,
PPTX, EPUB, CSV, XML, ZIP, IPYNB, MSG) into markdown text.

Used by websurf's web_fetch and research's deep_read for content-type
routing and format-agnostic text extraction.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import tempfile
from urllib.parse import urlparse

from agentic.toolkit.websurf import WEB_FETCH_USER_AGENT

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
