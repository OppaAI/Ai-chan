"""
toolkit/research.py

Adaptive search + deep research, expressed as bounded-unroll subgraphs.
Plus deep_read: a single-known-URL fetch with content-type routing.

Both adaptive_search and deep_research_graph build and run their OWN small
PlanGraph instances via graph_engine.py's execute_graph(). This means they plug
into graph_engine._tool_map() as single opaque tools — so any playbook that
currently calls "deep_research" can switch to "deep_research_graph" with a
one-line args change.

deep_read is NOT a PlanGraph — a single known URL has no "escalate: widen the
candidate pool" move the way a search does, so there's nothing here that
benefits from loop_to/max_visits or graph-level concurrency. It's a flat
escalation chain (route by content-type → extract → optionally condense),
same tier of composition as adaptive_search/deep_research from the LLM's
perspective (one opaque tool call), just without the graph machinery
underneath.

Sub-modules:
  - adaptive_search:     snippet-only → fetch only if judge says ESCALATE
  - deep_research_graph: multi-round fetch + combine + synthesize
  - deep_read:            known URL → content-type routed fetch (+ condense)
"""
from __future__ import annotations

import concurrent.futures
import functools
import hashlib
import importlib
import importlib.util
import json
import os
import re
import threading
import time
import uuid
from typing import Any
from urllib.parse import urlparse

import numpy as np

from system.log import get_logger
from agentic.graph_engine import PlanGraph, PlanNode, execute_graph
from agentic.registry import tool
from agentic.toolkit.websurf import (
    fetch_search_results,
    _stream_download,
    web_fetch,
    SEARXNG_MAX_RESULTS,
    WEB_FETCH_USER_AGENT,
)
from agentic.toolkit.ingest import (
    _extract_with_markitdown,
    _sniff_content_type,
)
from agentic.toolkit.provenance import authority_bonus, query_looks_time_sensitive

log = get_logger(__name__)

# -- deep_research config (moved from websurf.py) --
# These are module-level DEFAULTS; deep_research() itself now accepts
# num_searches/num_fetches/max_chars_per_page as real function args so
# callers (e.g. memory.learn.quick_studying) can override per-call.
DEEP_RESEARCH_NUM_SEARCHES = int(os.getenv("DEEP_RESEARCH_NUM_SEARCHES", 1))
DEEP_RESEARCH_NUM_FETCHES = int(os.getenv("DEEP_RESEARCH_NUM_FETCHES", 4))
DEEP_RESEARCH_MAX_CHARS_PER_PAGE = int(os.getenv("DEEP_RESEARCH_MAX_CHARS_PER_PAGE", 3500))
DEEP_RESEARCH_MAX_ROUNDS = int(os.getenv("DEEP_RESEARCH_MAX_ROUNDS", 4))
DEEP_RESEARCH_MAX_WORKERS = int(os.getenv("DEEP_RESEARCH_MAX_WORKERS", 4))
DEEP_RESEARCH_EVIDENCE_CHARS_FOR_DECISION = int(os.getenv("DEEP_RESEARCH_EVIDENCE_CHARS_FOR_DECISION", 8000))
DEEP_RESEARCH_EVIDENCE_CHARS_FOR_SYNTHESIS = int(os.getenv("DEEP_RESEARCH_EVIDENCE_CHARS_FOR_SYNTHESIS", 12000))
DEEP_RESEARCH_DECISION_MAX_TOKENS = int(os.getenv("DEEP_RESEARCH_DECISION_MAX_TOKENS", 200))
DEEP_RESEARCH_SYNTHESIS_MAX_TOKENS = int(os.getenv("DEEP_RESEARCH_SYNTHESIS_MAX_TOKENS", 700))

# -- in-memory evidence condensation (numpy-vectorized relevance filtering) --
# A FILTER, not a rewrite: chunks are scored for relevance and either kept
# verbatim or dropped entirely. Summarization only happens later, in
# deep_research's separate LLM synthesis call.
#
# deep_research uses the RESEARCH_CONDENSE_* knobs further down — wider net,
# lower bar, because the corroboration bonus (see _apply_corroboration_bonus)
# promotes borderline items that get independently confirmed by a second domain.
CONDENSE_CHUNK_CHARS = int(os.getenv("CONDENSE_CHUNK_CHARS", 500))
CONDENSE_TOP_K = int(os.getenv("CONDENSE_TOP_K", 8))
CONDENSE_MIN_SCORE = float(os.getenv("CONDENSE_MIN_SCORE", 0.15))
# Caps embedding calls PER fetch pipeline invocation (per _deep_search_impl call,
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

# -- deep_read (research.py) support: content-type routing + MarkItDown --
# THIN_TEXT_CHARS_THRESHOLD is the "did trafilatura actually get anything
# useful" bar used by deep_read to decide whether to escalate
# an HTML fetch to Crawl4AI.
THIN_TEXT_CHARS_THRESHOLD = int(os.getenv("THIN_TEXT_CHARS_THRESHOLD", 200))

ADAPTIVE_SEARCH_MAX_ROUNDS = int(os.getenv("ADAPTIVE_SEARCH_MAX_ROUNDS", 2))

# Same tiering as the earlier heuristic plan, kept here so this module has
# no import dependency on the retired adaptive_search.py.
_COMPARISON_MARKERS = re.compile(r"\bvs\.?\b|\bversus\b|\bcompare[d]?\b", re.IGNORECASE)
_MULTI_PART_MARKERS = re.compile(r"\band\b.*\band\b|,.*,|\?.*\?", re.IGNORECASE)
_OPEN_ENDED_MARKERS = re.compile(
    r"\brecommend|\bideas?\b|\bhow (should|do) i\b|\bwhat are (the|some)\b|\boverview\b|\bresearch\b",
    re.IGNORECASE,
)
_EFFORT_TIERS = {
    "simple": {"num_fetches": 0, "rounds": 1},
    "medium": {"num_fetches": 3, "rounds": 2},
    "broad":  {"num_fetches": 6, "rounds": ADAPTIVE_SEARCH_MAX_ROUNDS},
}


# ── Crawl4AI batch fetch (deep_research / deep_read) ────────────────────────

async def _crawl4ai_fetch_many_async(urls: list[str], max_chars: int) -> dict[str, str]:
    """Fetch many URLs in ONE Crawl4AI session (single browser launch,
    concurrent pages via arun_many) instead of one browser per URL.
    Returns {} on any failure or when crawl4ai isn't installed.
    """
    if not urls or importlib.util.find_spec("crawl4ai") is None:
        return {}

    allowed_urls = [u for u in urls if _source_agreement_allows(u)] if RESEARCH_RESPECT_ROBOTS else list(urls)
    if not allowed_urls:
        return {}

    try:
        crawl4ai = importlib.import_module("crawl4ai")
        AsyncWebCrawler = crawl4ai.AsyncWebCrawler
        CrawlerRunConfig = crawl4ai.CrawlerRunConfig
        CacheMode = crawl4ai.CacheMode

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
    except Exception as e:
        log.info("[crawl4ai] batch fetch failed for %d url(s): %s", len(urls), e)
        return {}


def _crawl4ai_fetch_many(urls: list[str], max_chars: int) -> dict[str, str]:
    """Synchronous wrapper for _crawl4ai_fetch_many_async."""
    import asyncio
    return asyncio.run(_crawl4ai_fetch_many_async(urls, max_chars))


# ── robots.txt compliance ("source agreement" to be crawled) ─────────────────

_robots_lock = threading.Lock()
_robots_cache: dict[str, tuple[float, RobotFileParser]] = {}


def _get_robot_parser(origin: str) -> RobotFileParser:
    """Fetch and cache a RobotFileParser for one origin (scheme://netloc).
    Fails open (allow-all) if robots.txt is missing or unreachable."""
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
    """Crawl-citizenship gate: only fetch pages the source's own robots.txt
    permits for our user-agent. Fails open on any lookup error."""
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
    """Best-effort sitemap.xml discovery for one domain. Checks robots.txt
    'Sitemap:' directives first, falls back to /sitemap.xml."""
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
            break

    if not urls:
        return []

    if query_hint:
        urls = sorted(urls, key=lambda u: reason.keyword_overlap_score(query_hint, u), reverse=True)
    return urls[:max_urls]


def plan_effort(prompt: str = "") -> str:
    """Graph tool: classify query complexity, return JSON plan.
    Zero model calls — cheap regex heuristic, same tiers the earlier
    standalone module used. If you want LLM-based classification instead,
    swap this node's tool for a synthesize_report call with a
    classification prompt (same trick routing_classifier already uses)."""
    q = prompt.strip()
    if _OPEN_ENDED_MARKERS.search(q) or len(q.split()) > 18:
        tier = "broad"
    elif _COMPARISON_MARKERS.search(q) or _MULTI_PART_MARKERS.search(q):
        tier = "medium"
    else:
        tier = "simple"
    plan = {"tier": tier, "freshness_bias": query_looks_time_sensitive(q), **_EFFORT_TIERS[tier]}
    return json.dumps(plan)


def search_and_rank(prompt: str = "") -> str:
    """Graph tool: snippet-only search, results ranked by domain authority
    (not just search-engine order). Returns JSON list of candidates so the
    next node (fetch_and_condense_ranked) can pick the top N."""
    results, error = fetch_search_results(prompt, SEARXNG_MAX_RESULTS, pageno=1)
    if error or not results:
        return json.dumps({"error": error or "no results", "candidates": []})
    ranked = sorted(results, key=lambda r: authority_bonus(r.get("url", ""), prompt), reverse=True)
    candidates = [
        {"url": r.get("url", ""), "title": r.get("title", ""), "content": r.get("content", ""),
         "quality": round(authority_bonus(r.get("url", ""), prompt), 3)}
        for r in ranked
    ]
    return json.dumps({"candidates": candidates})


def fetch_and_condense_ranked(candidates_json: str = "[]", prompt: str = "",
                               num_fetches: str = "3", freshness_bias: str = "false",
                               embedder=None) -> str:
    """Graph tool: fetch the top-N already-ranked candidates, blend
    authority+recency into relevance scoring, return condensed evidence in
    the same '[source: url | relevance: N | corroborated xN]' shape
    _deep_search_impl/deep_research already produce, so downstream synthesize/
    combine_evidence nodes need no changes."""
    try:
        data = json.loads(candidates_json)
        candidates = data.get("candidates", []) if isinstance(data, dict) else []
    except (json.JSONDecodeError, AttributeError):
        candidates = []
    if not candidates:
        return "[no candidates available to fetch]"

    n = max(0, int(num_fetches)) if str(num_fetches).lstrip("-").isdigit() else 3
    urls = [c["url"] for c in candidates[:n] if c.get("url")]
    if not urls:
        return "[no fetchable urls among candidates]"

    scored_chunks, pages, url_outcomes = _fetch_and_score_pipeline(urls, prompt, embedder, 3500)
    if not pages:
        return "[no pages fetched successfully]"

    fresh = str(freshness_bias).lower() == "true"
    boosted = []
    for score, url, chunk in scored_chunks:
        bonus = next((c["quality"] for c in candidates if c.get("url") == url), 0.0)
        # recency isn't re-derived here (would need a per-URL network call
        # this node doesn't make) — authority bonus from the ranking step
        # is reused directly; add fetch_published_date() back in here if
        # per-chunk recency matters more than authority for your use case.
        boosted.append((min(1.0, max(0.0, score + bonus)), url, chunk))

    return _finalize_condensed(boosted, prompt, annotate_agreement=True)


def judge_sufficient(evidence: str = "", prompt: str = "", client=None, model=None, state=None) -> str:
    """Graph tool: is the evidence gathered so far enough to fully answer
    the original question? Returns literally 'SUFFICIENT' or 'ESCALATE' so
    run_if:{equals: ...} on downstream nodes can gate on it directly —
    same convention as routing_classifier's classify node."""
    if client is None or not model:
        # No LLM available: conservative default — only simple/round-1
        # snippet evidence is ever judged sufficient without a model.
        return "SUFFICIENT" if evidence and "[no " not in evidence[:30] else "ESCALATE"
    from agentic.toolkit.synthesize import synthesize_report
    prompt_text = (
        "Reply with exactly one word: SUFFICIENT if the evidence below fully "
        "and specifically answers the question with no need for more "
        "searching, or ESCALATE if it's incomplete, vague, or missing.\n\n"
        f"Question: {prompt}\n\nEvidence:\n{evidence[:3000]}"
    )
    out = synthesize_report(evidence=evidence, prompt=prompt_text, style="plain",
                             client=client, model=model, state=state)
    return "SUFFICIENT" if "SUFFICIENT" in str(out).upper() else "ESCALATE"


def _build_adaptive_search_subgraph(query: str, tier: str, max_rounds: int | None = None, freshness_bias: bool = False) -> PlanGraph:
    """Build adaptive search graph based on tier.
    - simple: search → judge (no fetch, 1 round max)
    - medium: search → fetch → judge (loop if ESCALATE, max 2 rounds)
    - broad: search → fetch → judge (loop if ESCALATE, max ADAPTIVE_SEARCH_MAX_ROUNDS)
    """
    max_rounds = max_rounds if max_rounds is not None else ADAPTIVE_SEARCH_MAX_ROUNDS

    if tier == "simple":
        # Simple: search → judge (no fetch, no loop)
        nodes = [
            PlanNode(id="search_r", tool="search_and_rank", args={"prompt": "$prompt"}),
            PlanNode(id="judge_r", tool="judge_sufficient", depends_on=("search_r",),
                     args={"evidence": "$result:search_r", "prompt": "$prompt"}),
        ]
    else:
        # Medium/Broad: search → fetch → judge (loop if ESCALATE)
        num_fetches = 3 if tier == "medium" else 6

        nodes = [
            PlanNode(id="search_r", tool="search_and_rank", args={"prompt": "$prompt"}),
            PlanNode(id="fetch_r", tool="fetch_and_condense_ranked", depends_on=("search_r",),
                     args={"candidates_json": "$result:search_r", "prompt": "$prompt",
                           "num_fetches": str(num_fetches), "freshness_bias": "true" if freshness_bias else "false"}),
            PlanNode(id="judge_r", tool="judge_sufficient", depends_on=("fetch_r",),
                     args={"evidence": "$result:fetch_r", "prompt": "$prompt"},
                     loop_to="search_r", loop_condition={"equals": "ESCALATE"},
                     max_visits=max_rounds),
        ]

    return PlanGraph(id="adaptive_search", name="Adaptive search", goal=query, nodes=tuple(nodes))


@tool(
    name="adaptive_search",
    description="The default tool for any internet lookup. Adaptively searches, judges if snippets suffice, and only fetches full pages if needed.",
    props={"query": {"type": "string"}},
    required=["query"],
    domain="research",
    react=True,
    graph=True,
    wiki=True,
)
def adaptive_search(query: str, embedder=None, client=None, model: str | None = None,
                     max_rounds: int | None = None) -> str:
    if not query or not query.strip():
        return "[search failed: empty query]"

    # First, determine the tier by calling plan_effort
    try:
        plan_json = plan_effort(query.strip())
        plan = json.loads(plan_json)
        tier = plan.get("tier", "simple")
        freshness_bias = plan.get("freshness_bias", False)
    except Exception:
        tier = "simple"
        freshness_bias = False

    # Build graph appropriate for the tier
    graph = _build_adaptive_search_subgraph(query.strip(), tier, max_rounds=max_rounds, freshness_bias=freshness_bias)
    result = execute_graph(graph, embedder=embedder, llm_client=client, llm_model=model)

    # Prefer fetch_r's content (for medium/broad), fall back to search/judge
    for node_id in ("fetch_r", "search_r", "judge_r"):
        match = next((r for r in result.results if r.node_id == node_id and r.ok), None)
        if match and match.content and not match.content.startswith("["):
            return match.content
    return result.final_answer


# ══════════════════════════════════════════════════════════════════════════
# deep_read — single known URL, content-type routed fetch (+ optional condense)
# ══════════════════════════════════════════════════════════════════════════
#
# Distinct from adaptive_search/deep_research, which discover URLs via
# search: deep_read is handed one exact URL the user (or agent) already
# chose. There's no "widen the candidate pool" escalation available here —
# just an ordered attempt chain per content type:
#
#   non-HTML (pdf/docx/pptx/xlsx/epub/csv/xml/zip/ipynb/msg)
#       → download bytes → MarkItDown conversion
#   HTML (or unknown/ambiguous)
#       → web_fetch (cheap, trafilatura)
#       → if failed/thin: escalate to a single-URL Crawl4AI fetch (JS render)
#
# Then: no query → return raw text (truncated); query given →
# condense_evidence relevance-filters it.

DEEP_READ_MAX_CHARS = int(os.getenv("DEEP_READ_MAX_CHARS", 40000))
DEEP_READ_MAX_DOWNLOAD_BYTES = int(os.getenv("DEEP_READ_MAX_DOWNLOAD_BYTES", 20_000_000))
DEEP_READ_CONDENSE_TOP_K = int(os.getenv("DEEP_READ_CONDENSE_TOP_K", 12))


# ── Crawl4AI batch fetch (deep_research & deep_read) ────────────────────────

def _crawl4ai_fetch_many(urls: list[str], max_chars: int) -> dict[str, str]:
    """Fetch many URLs in ONE Crawl4AI session (single browser launch,
    concurrent pages via arun_many) instead of one browser per URL — a
    per-URL launch would be far too slow to be worth it. This is the batch
    path deep_research prefers; returns {} on any failure or when crawl4ai
    isn't installed, so the caller's per-URL fallback (web_fetch) covers
    every URL missing from the result.

    Also used (with a single-URL list) by deep_read as the
    JS-rendering escalation path when a plain web_fetch comes back thin.

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


@tool(
    name="deep_read",
    description="Fetch and extract content from one EXACT known URL. Handles HTML pages (with JS-render escalation), PDFs, DOCX, PPTX, XLSX, EPUB, CSV, and more. Use when you already have the specific URL to read — not for discovery (use adaptive_search for that).",
    props={"url": {"type": "string", "description": "The exact URL to fetch and read."}, "query": {"type": "string", "description": "Optional focus query — if given, content is relevance-filtered to what matters for this question."}},
    required=["url"],
    domain="research",
    react=True,
    graph=True,
    wiki=True,
)
def deep_read(
    url: str,
    query: str = "",
    embedder=None,
    max_chars: int = DEEP_READ_MAX_CHARS,
    condense_top_k: int = DEEP_READ_CONDENSE_TOP_K,
) -> str:
    """Fetch one EXACT URL — no search involved — routing by content type
    and escalating when the cheap path comes back empty or thin.

    Non-HTML documents (pdf, docx, pptx, xlsx, epub, csv, xml, zip, ipynb,
    msg) are downloaded and converted via MarkItDown — this is what
    the legacy read_paper_url could never do, since trafilatura only
    understands HTML.

    HTML (or anything _sniff_content_type can't classify) goes through the
    cheap web_fetch/trafilatura path first. Only if that fails outright or
    returns fewer than THIN_TEXT_CHARS_THRESHOLD chars does it escalate to
    a single-URL Crawl4AI fetch (JS rendering) — mirrors the cheap-first,
    escalate-only-when-warranted pattern adaptive_search/judge_sufficient
    already use, so a single-URL read doesn't pay a browser-launch cost on
    every call, only on the ones that actually need it.

    Without `query`: returns the first max_chars of extracted/converted
    text (further truncated to AGENT_TOOL_RESULT_MAX_CHARS once wrapped as
    a tool observation).

    With `query`: text is chunked and relevance-scored via condense_evidence,
    same as deep_research's evidence condensation, so what survives the
    observation-length limit is what's actually relevant to the question.
    """
    if not url or not url.strip():
        return "[fetch failed: empty url]"
    url = url.strip()

    content_type = _sniff_content_type(url)

    if content_type != "html":
        downloaded, error = _stream_download(url, DEEP_READ_MAX_DOWNLOAD_BYTES)
        if error:
            return error
        text = _extract_with_markitdown(downloaded, content_type, max_chars)
    else:
        text = web_fetch(url, max_chars=max_chars, max_stream_download=DEEP_READ_MAX_DOWNLOAD_BYTES)
        is_thin = text.startswith("[fetch failed") or len(text.strip()) < THIN_TEXT_CHARS_THRESHOLD
        if is_thin and RESEARCH_USE_CRAWL4AI:
            crawled = _crawl4ai_fetch_many([url], max_chars)
            candidate = crawled.get(url, "").strip()
            current_len = 0 if text.startswith("[fetch failed") else len(text.strip())
            if candidate and len(candidate) > current_len:
                text = candidate

    if text.startswith("[fetch failed"):
        return text
    if not query:
        return f"[Fetched content — {url}]\n\n{text}"
    condensed = condense_evidence([(url, text)], query, embedder=embedder, top_k=condense_top_k)
    return f"[Fetched content — {url}, condensed for: {query}]\n\n{condensed}"


# ── evidence condensation ──────────────────────────────────────────────
# These helpers score, dedupe, filter, and format evidence chunks.
# Moved here from websurf.py to keep research logic in the research module.

import hashlib
import concurrent.futures

from cognition import reason


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


def _finalize_condensed(
    scored_chunks: list[tuple[float, str, str]],
    query: str,
    top_k: int = RESEARCH_CONDENSE_TOP_K,
    min_score: float = RESEARCH_CONDENSE_MIN_SCORE,
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


# ── fetch + scoring pipeline helpers ──────────────────────────────────
# These are research-specific tools for deep_research and adaptive_search.

def _fetch_and_score_pipeline(
    urls: list[str],
    query: str,
    embedder,
    max_chars_per_page: int,
    chunk_chars: int = CONDENSE_CHUNK_CHARS,
    max_workers: int = DEEP_RESEARCH_MAX_WORKERS,
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

    With num_fetches=0 this returns snippets/URLs only (snippet-only mode).
    Deep_research calls this helper with its own positive fetch count plus the
    research-only knobs (Crawl4AI batch prefetch, robots gating, sitemap
    expansion, corroboration annotation) to do fetched-source work.

    Returns (formatted_bundle, urls_actually_fetched) — the URL set lets
    deep_research exclude already-seen URLs across rounds without a
    separate re-fetch-avoidance mechanism.
    """
    if not query or not query.strip():
        return "[search failed: empty query]", set()

    num_searches = max(1, num_searches)
    num_fetches = max(0, num_fetches)

    log.info("[_deep_search_impl] query=%r searches=%d fetches=%d", query, num_searches, num_fetches)

    # Search phase
    snippets = []
    seen_urls = set()
    for i in range(num_searches):
        raw, err = fetch_search_results(query, SEARXNG_MAX_RESULTS, pageno=i + 1)
        if err:
            log.warning("[_deep_search_impl] search %d failed: %s", i + 1, err)
            continue
        results = raw or []
        for r in results:
            u = r.get("url")
            if u and u not in seen_urls:
                seen_urls.add(u)
                snippets.append(r)

    snippet_bundle = _format_snippet_bundle(snippets)

    if num_fetches == 0:
        return snippet_bundle, set()

    # Fetch phase
    candidates = [r.get("url") for r in snippets[:num_fetches] if r.get("url")]
    if not candidates:
        return snippet_bundle, set()

    urls_to_fetch = [u for u in candidates if u not in (exclude_urls or set())]
    if not urls_to_fetch:
        return snippet_bundle, set()

    # Score/fetch pipeline
    scored_chunks, pages, url_outcomes = _fetch_and_score_pipeline(
        urls_to_fetch,
        query,
        embedder,
        max_chars_per_page,
        chunk_chars=CONDENSE_CHUNK_CHARS,
        max_workers=max_workers,
        max_chunks_to_score=condense_max_chunks_to_score,
        fetch_fn=fetch_fn,
        batch_prefetch_fn=batch_prefetch_fn,
    )

    # Condense evidence
    if pages:
        condensed = _finalize_condensed(
            scored_chunks, query, top_k=condense_top_k, min_score=condense_min_score,
            annotate_agreement=annotate_agreement,
            agreement_bonus=RESEARCH_AGREEMENT_BONUS,
            agreement_similarity=RESEARCH_AGREEMENT_SIMILARITY,
        )
    else:
        condensed = "[no fetched content available to condense]"

    manifest = _format_url_manifest(url_outcomes)
    fetched_url_set = {url for url, _text in pages}

    if not pages:
        return f"{snippet_bundle}\n\n{manifest}", fetched_url_set

    return f"{snippet_bundle}\n\n{manifest}\n\n{condensed}", fetched_url_set


def _format_snippet_bundle(snippets: list[dict]) -> str:
    """Format a bundle of search snippets for the LLM."""
    if not snippets:
        return "[no search results]"
    lines = ["[Search results — {} result(s)]".format(len(snippets))]
    for i, s in enumerate(snippets, 1):
        title = s.get("title") or ""
        snippet = s.get("snippet") or ""
        url = s.get("url") or ""
        lines.append(f"{i}. {title}\n   {snippet}\n   {url}")
    return "\n".join(lines)


def _format_url_manifest(url_outcomes: list[tuple[str, str]]) -> str:
    if not url_outcomes:
        return "[no URLs attempted]"
    lines = [f"[URL manifest — {len(url_outcomes)} attempted]"]
    for url, status in url_outcomes:
        lines.append(f"- {url} — {status}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Deep research graph — multi-round fetch + synthesize
# ══════════════════════════════════════════════════════════════════════════


# ── session-scoped cross-round URL dedup ─────────────────────────────────
# Keyed on a short random session id generated once per deep_research_graph()
# run, not on anything derived from the query — so two concurrent deep
# research runs never share (or collide on) a dedup set. Swept by TTL as a
# safety net in case a run dies before reaching combine_research_rounds's
# explicit cleanup (e.g. process killed mid-run).
_SESSION_TTL_SECONDS = 6 * 3600
_session_lock = threading.Lock()
_session_seen_urls: dict[str, tuple[float, set[str]]] = {}


def _session_sweep_locked() -> None:
    now = time.monotonic()
    dead = [sid for sid, (ts, _urls) in _session_seen_urls.items() if now - ts > _SESSION_TTL_SECONDS]
    for sid in dead:
        _session_seen_urls.pop(sid, None)


def _session_start() -> str:
    session_id = uuid.uuid4().hex[:16]
    with _session_lock:
        _session_sweep_locked()
        _session_seen_urls[session_id] = (time.monotonic(), set())
    return session_id


def _session_get_seen(session_id: str) -> set[str]:
    with _session_lock:
        entry = _session_seen_urls.get(session_id)
        return set(entry[1]) if entry else set()


def _session_add_seen(session_id: str, urls: set[str]) -> None:
    if not urls:
        return
    with _session_lock:
        entry = _session_seen_urls.get(session_id)
        if entry is None:
            _session_seen_urls[session_id] = (time.monotonic(), set(urls))
        else:
            ts, seen = entry
            seen.update(urls)
            _session_seen_urls[session_id] = (ts, seen)


def _session_end(session_id: str) -> None:
    with _session_lock:
        _session_seen_urls.pop(session_id, None)


def deep_fetch_round(prompt: str = "", num_searches: str = "1", num_fetches: str = "4",
                      session_id: str = "", embedder=None) -> str:
    """Graph tool: one deep_research-style round, with cross-round dedup via
    session_id — URLs fetched in an earlier round of THIS run are excluded
    from later rounds, so an overnight 4-round session doesn't re-fetch the
    same source twice. Only a ~16-char session token travels through graph
    args; the actual URL set lives in the in-process registry above, never
    JSON-encoded into a $result: string."""
    batch_prefetch_fn = functools.partial(_crawl4ai_fetch_many) if RESEARCH_USE_CRAWL4AI else None
    n_searches = int(num_searches) if str(num_searches).lstrip("-").isdigit() else 1
    n_fetches = int(num_fetches) if str(num_fetches).lstrip("-").isdigit() else 4
    exclude_urls = _session_get_seen(session_id) if session_id else None
    bundle, fetched_urls = _deep_search_impl(
        prompt,
        embedder,
        num_searches=n_searches,
        num_fetches=n_fetches,
        max_chars_per_page=DEEP_RESEARCH_MAX_CHARS_PER_PAGE,
        max_workers=DEEP_RESEARCH_MAX_WORKERS,
        exclude_urls=exclude_urls,
        batch_prefetch_fn=batch_prefetch_fn,
        respect_robots=RESEARCH_RESPECT_ROBOTS,
        annotate_agreement=True,
        condense_top_k=RESEARCH_CONDENSE_TOP_K,
        condense_min_score=RESEARCH_CONDENSE_MIN_SCORE,
        condense_max_chunks_to_score=RESEARCH_CONDENSE_MAX_CHUNKS_TO_SCORE,
        expand_sitemap=RESEARCH_SITEMAP_ENABLED,
    )
    if session_id:
        _session_add_seen(session_id, fetched_urls)
    return bundle


def combine_research_rounds(r1: str = "", r2: str = "", r3: str = "", r4: str = "",
                             prompt: str = "", session_id: str = "",
                             client=None, model=None, evidence: str = "",
                             state=None) -> str:
    """Graph tool: filter skipped-round placeholders, combine whatever
    rounds ran, synthesize the final report, then release this run's dedup
    registry entry — this is the normal-path cleanup; the TTL sweep above
    is only a backstop for abnormal exits.

    Supports both:
      - collapsed mode: single ``evidence`` string (synthesized directly)
      - legacy mode: ``r1``-``r4`` combined and then synthesized
    """
    from agentic.toolkit.synthesize import combine_evidence, synthesize_report
    try:
        if evidence:
            return synthesize_report(evidence=evidence, prompt=prompt, style="professional",
                                      client=client, model=model, state=state)
        rounds = [r for r in (r1, r2, r3, r4) if r and not r.strip().startswith("skipped:")]
        if not rounds:
            return "[no research evidence gathered — all rounds skipped or failed]"
        combined = combine_evidence(parts=rounds, separator="\n\n===\n\n")
        return synthesize_report(evidence=combined, prompt=prompt, style="professional",
                                  client=client, model=model, state=state)
    finally:
        if session_id:
            _session_end(session_id)


def _build_deep_research_subgraph(query: str, session_id: str,
                                   max_rounds: int = DEEP_RESEARCH_MAX_ROUNDS,
                                   tool_mode: bool = False) -> PlanGraph:
    """Collapsed to 2 dynamic nodes + 3 static, using loop_to instead of
    an N-node unroll. Fetch → Judge (loops back to Fetch if ESCALATE) →
    Finalize → Report → Learn. In tool_mode, skips report/learn."""
    max_rounds = max(1, min(5, max_rounds))  # Cap at 5 per Grok analysis
    nodes = [
        PlanNode(id="fetch_r", tool="deep_fetch_round",
                 args={"prompt": "$prompt", "num_searches": str(DEEP_RESEARCH_NUM_SEARCHES), "num_fetches": str(DEEP_RESEARCH_NUM_FETCHES),
                       "session_id": session_id}),
        PlanNode(id="judge_r", tool="judge_sufficient", depends_on=("fetch_r",),
                 args={"evidence": "$result:fetch_r", "prompt": "$prompt"},
                 loop_to="fetch_r", loop_condition={"equals": "ESCALATE"},
                 max_visits=max_rounds),
        PlanNode(id="finalize", tool="combine_research_rounds", depends_on=("judge_r",),
                 args={"evidence": "$result:fetch_r", "prompt": "$prompt", "session_id": session_id}),
    ]
    if not tool_mode:
        nodes.extend([
            PlanNode(id="report", tool="write_report", depends_on=("finalize",),
                     args={"title": "$title", "content": "$result:finalize", "report_dir": "reports"}),
            PlanNode(id="learn", tool="learn_report", depends_on=("report",),
                     args={"title": "$title", "text": "$result:finalize", "kind": "self_learned"}),
        ])
    return PlanGraph(id="deep_research", name="Deep research", goal=query, nodes=tuple(nodes))


@tool(
    name="deep_research",
    description="Research tool that fetches and synthesizes full source pages from discovered URLs. Use when the research itself is the deliverable or for deep/thorough self-learning.",
    props={"query": {"type": "string", "description": "The research question. Can be broader/less scoped since the tool refines it internally."}},
    required=["query"],
    domain="research",
    react=True,
    graph=True,
    wiki=True,
)
def deep_research(query: str, embedder=None, client=None, model: str | None = None,
                         max_rounds: int = DEEP_RESEARCH_MAX_ROUNDS, tool_mode: bool = False) -> str:
    """Public entry point — drop-in replacement for research.py's
    deep_research() when called from the graph executor.

    Args:
        tool_mode: If True, skips write_report and learn_report nodes.
                   Use when calling as a sub-tool from an outer playbook
                   that will handle its own report/learn.
    """
    if not query or not query.strip():
        return "[search failed: empty query]"
    session_id = _session_start()
    graph = _build_deep_research_subgraph(query.strip(), session_id, max_rounds=max_rounds, tool_mode=tool_mode)
    try:
        result = execute_graph(graph, embedder=embedder, llm_client=client, llm_model=model)
    except Exception:
        _session_end(session_id)
        raise
    finalize = next((r for r in result.results if r.node_id == "finalize" and r.ok), None)
    return finalize.content if finalize else result.final_answer