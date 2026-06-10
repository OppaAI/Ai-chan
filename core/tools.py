"""
core/tools.py

Aiko's tool belt — starting with web search via SearXNG.
All tools are plain functions that return strings ready for context injection.
"""

import os
import requests
from core.log import get_logger
log = get_logger(__name__)

# ── config ────────────────────────────────────────────────────────────────────

SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:8081")
MAX_RESULTS = int(os.getenv("SEARXNG_MAX_RESULTS", 3))

# ── web search ────────────────────────────────────────────────────────────────

def web_search(query: str, max_results: int = MAX_RESULTS) -> str:
    """
    Search the web via SearXNG and return a compact result string
    ready for injection into Aiko's context.

    Args:
        query (str): The search query.
        max_results (int): The maximum number of results to return.

    Returns:
        str: A compact result string ready for injection into Aiko's context.
    """
    try:
        response = requests.get(
            f"{SEARXNG_URL}/search",
            params={
                "q":      query,
                "format": "json",
            },
            timeout=8,
        )
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.ConnectionError:
        return f"[search failed: could not reach SearXNG at {SEARXNG_URL}]"
    except requests.exceptions.Timeout:
        return "[search failed: timed out]"
    except requests.exceptions.RequestException as e:
        return f"[search failed: request error: {e}]"
    except ValueError:
        return "[search failed: invalid JSON response]"
        
    if not isinstance(data, dict):
        return "[search failed: unexpected response format]"
        
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

def web_search_context(query: str, max_results: int = MAX_RESULTS) -> str | None:
    """
    Run a web search and return a context-ready prompt string,
    or None if search failed / no results.

    Args:
        query (str): The search query.
        max_results (int): The maximum number of results to return.

    Returns:
        str | None: A context-ready prompt string or None if search failed.
    """
    log.debug(f"[tools] searching: {query!r} at {SEARXNG_URL}")
    results = web_search(query, max_results)
    log.debug(f"[tools] result: {results[:200]!r}")
    if results.startswith("[search failed") or results.startswith("[no results"):
        return None
    return f"[Web search results for: {query}]\n\n{results}\n\nUser asked: {query}"