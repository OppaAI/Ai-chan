"""Web search and page extraction tools."""

from __future__ import annotations

import ipaddress
import os
import socket
from urllib.parse import urlparse

import importlib
import importlib.util

SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:8081")
MAX_RESULTS = int(os.getenv("SEARXNG_MAX_RESULTS", 5))


def web_search(query: str, max_results: int = MAX_RESULTS) -> str:
    """Search the web via SearXNG and return compact numbered results."""
    if importlib.util.find_spec("requests") is None:
        return "[search failed: requests is not installed]"
    requests = importlib.import_module("requests")
    try:
        response = requests.get(
            f"{SEARXNG_URL}/search",
            params={"q": query, "format": "json"},
            timeout=8,
        )
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        return f"[search failed: {e}]"
    except ValueError:
        return "[search failed: invalid JSON response]"

    results = data.get("results", [])[:max_results]
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


def fetch_and_extract(url: str, max_chars: int = 4000) -> str:
    """Fetch a URL and extract its main article/body text with trafilatura."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return f"[fetch failed: unsupported URL scheme: {parsed.scheme or 'none'}]"
    if _is_private_or_local_host(parsed.hostname):
        return "[fetch failed: URL host is not allowed]"
    if importlib.util.find_spec("trafilatura") is None:
        return "[fetch failed: trafilatura is not installed]"
    trafilatura = importlib.import_module("trafilatura")
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return "[fetch failed: empty response]"
        text = trafilatura.extract(downloaded, include_links=False, include_tables=False) or ""
        return text[:max_chars] if text else "[fetch failed: no extractable text]"
    except Exception as e:
        return f"[fetch failed: {e}]"


def deep_search(query: str, max_results: int = 3, fetch_top: int = 2) -> str:
    """Search, fetch the top pages, and return one compact research bundle."""
    if not query or not query.strip():
        return "[search failed: empty query]"
    raw_results = web_search(query, max_results)
    if raw_results.startswith("[search failed") or raw_results.startswith("[no results"):
        return raw_results

    bundle = [raw_results]
    result_blocks = raw_results.split("\n\n")[1:]

    for i, result in enumerate(result_blocks[:fetch_top], 1):
        url = next((line.strip() for line in result.splitlines() if line.strip().startswith("http")), None)
        if not url:
            continue
        page = fetch_and_extract(url)
        if not page.startswith("[fetch failed"):
            bundle.append(f"\n[Full page {i}: {url}]\n{page[:2000]}")

    return "\n\n".join(bundle)


def web_search_context(query: str, max_results: int = MAX_RESULTS) -> str | None:
    """Run web_search and wrap successful results as context for chat mode."""
    if not query or not query.strip():
        return "[search failed: empty query]"
    results = web_search(query, max_results)
    if results.startswith("[search failed") or results.startswith("[no results"):
        return None
    return f"{results}\n\nUser asked: {query}"
