"""Web search and page extraction tools."""

from __future__ import annotations

import ipaddress
import json
import os
import re
import socket
from urllib.parse import urlparse

import importlib
import importlib.util

SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:8888")
MAX_RESULTS = int(os.getenv("SEARXNG_MAX_RESULTS", 5))

# -- deep_search (single search + fetch pass) --
DEEP_SEARCH_MAX_RESULTS = int(os.getenv("DEEP_SEARCH_MAX_RESULTS", 3))
DEEP_SEARCH_FETCH_TOP = int(os.getenv("DEEP_SEARCH_FETCH_TOP", 2))
DEEP_SEARCH_MAX_CHARS_PER_PAGE = int(os.getenv("DEEP_SEARCH_MAX_CHARS_PER_PAGE", 2000))

# -- deep_research (multi-round adaptive research) --
DEEP_RESEARCH_MAX_ROUNDS = int(os.getenv("DEEP_RESEARCH_MAX_ROUNDS", 3))
DEEP_RESEARCH_FETCH_TOP = int(os.getenv("DEEP_RESEARCH_FETCH_TOP", 2))
DEEP_RESEARCH_MAX_CHARS_PER_PAGE = int(os.getenv("DEEP_RESEARCH_MAX_CHARS_PER_PAGE", 1500))
DEEP_RESEARCH_EVIDENCE_CHARS_FOR_DECISION = int(os.getenv("DEEP_RESEARCH_EVIDENCE_CHARS_FOR_DECISION", 6000))
DEEP_RESEARCH_EVIDENCE_CHARS_FOR_SYNTHESIS = int(os.getenv("DEEP_RESEARCH_EVIDENCE_CHARS_FOR_SYNTHESIS", 8000))
DEEP_RESEARCH_DECISION_MAX_TOKENS = int(os.getenv("DEEP_RESEARCH_DECISION_MAX_TOKENS", 200))
DEEP_RESEARCH_SYNTHESIS_MAX_TOKENS = int(os.getenv("DEEP_RESEARCH_SYNTHESIS_MAX_TOKENS", 600))


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


def web_fetch(url: str, max_chars: int = 4000) -> str:
    """Fetch a single URL and extract its main article/body text with trafilatura.

    This is the one-and-only "fetch a page" primitive in the toolkit. Both
    the model's direct fetch_page-style calls and deep_search/deep_research's
    internal page reads route through this function, so there is exactly one
    implementation to reason about.
    """
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


def _extract_urls(raw_results: str, limit: int) -> list[str]:
    result_blocks = raw_results.split("\n\n")[1:]
    urls = []
    for block in result_blocks[:limit]:
        url = next((line.strip() for line in block.splitlines() if line.strip().startswith("http")), None)
        if url:
            urls.append(url)
    return urls


def deep_search(
    query: str,
    max_results: int = DEEP_SEARCH_MAX_RESULTS,
    fetch_top: int = DEEP_SEARCH_FETCH_TOP,
    max_chars_per_page: int = DEEP_SEARCH_MAX_CHARS_PER_PAGE,
) -> str:
    """Search, fetch the top pages, and return one compact research bundle.

    Single search + single fetch pass — the "search + fetch" tier, not a
    multi-round research loop. See deep_research for that.
    """
    if not query or not query.strip():
        return "[search failed: empty query]"
    raw_results = web_search(query, max_results)
    if raw_results.startswith("[search failed") or raw_results.startswith("[no results"):
        return raw_results

    bundle = [raw_results]
    for i, url in enumerate(_extract_urls(raw_results, fetch_top), 1):
        page = web_fetch(url)
        if not page.startswith("[fetch failed"):
            bundle.append(f"\n[Full page {i}: {url}]\n{page[:max_chars_per_page]}")

    return "\n\n".join(bundle)


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
    max_rounds: int = DEEP_RESEARCH_MAX_ROUNDS,
    fetch_top: int = DEEP_RESEARCH_FETCH_TOP,
    max_chars_per_page: int = DEEP_RESEARCH_MAX_CHARS_PER_PAGE,
) -> str:
    """Single-round adaptive research: search, read, decide whether to refine
    the query and search again, repeat, then return a synthesized bundle.

    This is the genuine "deep research" tier — an iterative loop across
    several search+fetch rounds — as distinct from deep_search's single
    search+fetch pass. When client/model are supplied (a small local LLM),
    the loop is adaptive: after each round the model is asked whether the
    gathered evidence answers the original query, and if not, what the next
    query should be. Without client/model it degrades gracefully to a single 
    search+fetch round (same as deep_search) so the tool never hard-fails just
    because no model was wired in.
    """
    if not query or not query.strip():
        return "[search failed: empty query]"

    rounds: list[str] = []
    queries_used: list[str] = [query.strip()]
    current_query = query.strip()
    adaptive = client is not None and model

    for round_num in range(1, max_rounds + 1):
        bundle = deep_search(
            current_query,
            max_results=DEEP_SEARCH_MAX_RESULTS,
            fetch_top=fetch_top,
            max_chars_per_page=max_chars_per_page,
        )
        if bundle.startswith("[search failed"):
            if round_num == 1:
                return bundle
            break
        if bundle.startswith("[no results"):
            break
        rounds.append(f"[Round {round_num} — query: {current_query}]\n{bundle}")

        if not adaptive or round_num == max_rounds:
            break

        evidence_so_far = "\n\n".join(rounds)[-DEEP_RESEARCH_EVIDENCE_CHARS_FOR_DECISION:]
        decision_prompt = (
            "You are directing a multi-round web research process. Given the "
            "original question and the evidence gathered so far, decide whether "
            "another search round is needed.\n"
            "Return ONLY compact JSON: {\"continue\": bool, \"next_query\": string, \"reason\": string}.\n"
            "Set continue=false once the evidence is sufficient to answer the "
            "original question, or if further searching is unlikely to add "
            "anything new. next_query should be empty when continue=false.\n\n"
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

    header = f"[Deep research: {len(rounds)} round(s) for: {query}]"
    if len(queries_used) > 1:
        header += f"\n[Query refinements: {' -> '.join(queries_used)}]"

    combined = "\n\n".join(rounds)

    if adaptive:
        synthesis_prompt = (
            "Synthesize the following multi-round research evidence into a "
            "concise, well-organized answer to the original question. Note any "
            "unresolved gaps or conflicting information. Do not invent facts "
            "not present in the evidence.\n\n"
            f"Original question: {query}\n\n"
            f"Evidence:\n{combined[:DEEP_RESEARCH_EVIDENCE_CHARS_FOR_SYNTHESIS]}"
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
                return f"{header}\n\n[Synthesis]\n{synthesis}\n\n[Raw evidence]\n{combined}"
        except Exception:
            pass  # fall through to raw bundle below

    return f"{header}\n\n{combined}"


def web_search_context(query: str, max_results: int = MAX_RESULTS) -> str | None:
    """Run web_search and wrap successful results as context for chat mode."""
    if not query or not query.strip():
        return "[search failed: empty query]"
    results = web_search(query, max_results)
    if results.startswith("[search failed") or results.startswith("[no results"):
        return None
    return f"{results}\n\nUser asked: {query}"
