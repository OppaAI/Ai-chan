"""
core/tools.py

Aiko's tool belt — web search via SearXNG, deep page fetching, and bundling.
All tools are plain functions that return strings ready for context injection.
"""

import os
import requests
import trafilatura
from core.log import get_logger

log = get_logger(__name__)

# ── config ────────────────────────────────────────────────────────────────────

SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:8081")
MAX_RESULTS = int(os.getenv("SEARXNG_MAX_RESULTS", 5))

# ── web search ────────────────────────────────────────────────────────────────

def web_search(query: str, max_results: int = MAX_RESULTS) -> str:
    """Search the web via SearXNG and return a compact result string."""
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
    for i, r in enumerate(results, 1):
        title   = r.get("title", "").strip()
        url     = r.get("url", "").strip()
        content = r.get("content", "").strip()
        lines.append(f"{i}. {title}\n   {url}\n   {content}")

    return "\n\n".join(lines)

def fetch_and_extract(url: str, max_chars: int = 4000) -> str:
    """Fetch a URL and extract its main text content using trafilatura."""
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return "[fetch failed: empty response]"
        text = trafilatura.extract(downloaded, include_links=False, include_tables=False) or ""
        return text[:max_chars]
    except Exception as e:
        return f"[fetch failed: {e}]"

def deep_search(query: str, max_results: int = 3, fetch_top: int = 2) -> str:
    """Search → fetch top N pages → return compact bundle."""
    raw_results = web_search(query, max_results)
    if raw_results.startswith("[search failed") or raw_results.startswith("[no results"):
        return raw_results

    bundle = [raw_results]
    result_blocks = raw_results.split("\n\n")[1:]  # skip the header line

    for i, r in enumerate(result_blocks[:fetch_top], 1):
        url = next((l.strip() for l in r.splitlines() if l.strip().startswith("http")), None)
        if url:
            page = fetch_and_extract(url)
            if not page.startswith("[fetch failed"):
                bundle.append(f"\n[Full page {i}: {url}]\n{page[:2000]}")

    return "\n\n".join(bundle)

def web_search_context(query: str, max_results: int = MAX_RESULTS) -> str | None:
    """Run a standard web search and return a context-ready prompt string."""
    results = web_search(query, max_results)
    if results.startswith("[search failed") or results.startswith("[no results"):
        return None
    return f"[Web search results for: {query}]\n\n{results}\n\nUser asked: {query}"
