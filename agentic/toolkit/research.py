"""
toolkit/research.py

Adaptive search + deep research, expressed as bounded-unroll subgraphs.

Both adaptive_search and deep_research_graph build and run their OWN small
PlanGraph instances via schema.py's execute_graph(). This means they plug
into schema._tool_map() as single opaque tools — so any playbook that
currently calls "deep_search" or "deep_research" can switch to
"adaptive_search" or "deep_research_graph" with a one-line args change.

Sub-modules:
  - adaptive_search:     snippet-only → fetch only if judge says ESCALATE
  - deep_research_graph: multi-round fetch + combine + synthesize
"""
from __future__ import annotations

import functools
import json
import re
import threading
import time
import uuid
from typing import Any

from system.log import get_logger
from agentic.schema import PlanGraph, PlanNode, execute_graph
from agentic.toolkit.websurf import (
    _web_search_raw,
    _deep_search_impl,
    _fetch_and_score_pipeline,
    _finalize_condensed,
    _crawl4ai_fetch_many,
    MAX_RESULTS,
)
from agentic.toolkit.provenance import authority_bonus, query_looks_time_sensitive

log = get_logger(__name__)

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
    results, error = _web_search_raw(prompt, MAX_RESULTS, pageno=1)
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
    deep_search/deep_research already produce, so downstream synthesize/
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


def judge_sufficient(evidence: str = "", prompt: str = "", client=None, model=None) -> str:
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
                             client=client, model=model)
    return "SUFFICIENT" if "SUFFICIENT" in str(out).upper() else "ESCALATE"


def _build_adaptive_search_subgraph(query: str, max_rounds: int | None = None) -> PlanGraph:
    max_rounds = max_rounds if max_rounds is not None else ADAPTIVE_SEARCH_MAX_ROUNDS
    nodes = [
        PlanNode(id="plan", tool="plan_effort", args={"prompt": "$prompt"}),
        PlanNode(id="search_r1", tool="search_and_rank", depends_on=("plan",), args={"prompt": "$prompt"}),
        PlanNode(id="judge_r1", tool="judge_sufficient", depends_on=("search_r1",),
                  args={"evidence": "$result:search_r1", "prompt": "$prompt"}),
        PlanNode(id="fetch_r1", tool="fetch_and_condense_ranked", depends_on=("judge_r1",),
                  run_if={"node": "judge_r1", "equals": "ESCALATE"},
                  args={"candidates_json": "$result:search_r1", "prompt": "$prompt",
                        "num_fetches": "3", "freshness_bias": "false"}),
    ]
    if max_rounds >= 2:
        nodes += [
            PlanNode(id="judge_r2", tool="judge_sufficient", depends_on=("fetch_r1",),
                      run_if={"node": "judge_r1", "equals": "ESCALATE"},
                      args={"evidence": "$result:fetch_r1", "prompt": "$prompt"}),
            PlanNode(id="search_r2", tool="search_and_rank", depends_on=("judge_r2",),
                      run_if={"node": "judge_r2", "equals": "ESCALATE"},
                      args={"prompt": "$prompt"}),
            PlanNode(id="fetch_r2", tool="fetch_and_condense_ranked", depends_on=("search_r2",),
                      run_if={"node": "judge_r2", "equals": "ESCALATE"},
                      args={"candidates_json": "$result:search_r2", "prompt": "$prompt",
                            "num_fetches": "3", "freshness_bias": "false"}),
        ]
    return PlanGraph(id="adaptive_search", name="Adaptive search", goal=query, nodes=tuple(nodes))


def adaptive_search(query: str, embedder=None, client=None, model: str | None = None,
                     max_rounds: int | None = None) -> str:
    if not query or not query.strip():
        return "[search failed: empty query]"
    graph = _build_adaptive_search_subgraph(query.strip(), max_rounds=max_rounds)
    result = execute_graph(graph, embedder=embedder, llm_client=client, llm_model=model)
    # Prefer the last successful fetch/search node's content over the
    # generic no-LLM _synthesize_without_llm() summary, since callers of
    # adaptive_search expect condensed evidence text, not a step log.
    for node_id in ("fetch_r2", "search_r2", "fetch_r1", "search_r1"):
        match = next((r for r in result.results if r.node_id == node_id and r.ok), None)
        if match and match.content and not match.content.startswith("["):
            return match.content
        if match and match.content:
            return match.content
    return result.final_answer

# ══════════════════════════════════════════════════════════════════════════
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
                             client=None, model=None) -> str:
    """Graph tool: filter skipped-round placeholders, combine whatever
    rounds ran, synthesize the final report, then release this run's dedup
    registry entry — this is the normal-path cleanup; the TTL sweep above
    is only a backstop for abnormal exits."""
    from agentic.toolkit.synthesize import combine_evidence, synthesize_report
    try:
        rounds = [r for r in (r1, r2, r3, r4) if r and not r.strip().startswith("skipped:")]
        if not rounds:
            return "[no research evidence gathered — all rounds skipped or failed]"
        combined = combine_evidence(parts=rounds, separator="\n\n===\n\n")
        return synthesize_report(evidence=combined, prompt=prompt, style="professional",
                                  client=client, model=model)
    finally:
        if session_id:
            _session_end(session_id)


def _build_deep_research_subgraph(query: str, session_id: str,
                                   max_rounds: int = DEEP_RESEARCH_MAX_ROUNDS) -> PlanGraph:
    max_rounds = max(1, min(4, max_rounds))
    nodes: list[PlanNode] = []
    prev_judge_id: str | None = None

    for i in range(1, max_rounds + 1):
        fetch_id, judge_id = f"fetch_r{i}", f"judge_r{i}"
        run_if = {"node": prev_judge_id, "equals": "ESCALATE"} if prev_judge_id else None
        depends_fetch = (prev_judge_id,) if prev_judge_id else ()
        nodes.append(PlanNode(id=fetch_id, tool="deep_fetch_round", depends_on=depends_fetch,
                               run_if=run_if,
                               args={"prompt": "$prompt", "num_searches": "1", "num_fetches": "4",
                                     "session_id": session_id}))
        if i < max_rounds:
            nodes.append(PlanNode(id=judge_id, tool="judge_sufficient", depends_on=(fetch_id,),
                                   run_if=run_if, args={"evidence": f"$result:{fetch_id}", "prompt": "$prompt"}))
            prev_judge_id = judge_id

    fetch_ids = [f"fetch_r{i}" for i in range(1, max_rounds + 1)]
    combine_args = {f"r{i+1}": f"$result:{fid}" for i, fid in enumerate(fetch_ids)}
    combine_args["prompt"] = "$prompt"
    combine_args["session_id"] = session_id
    nodes.append(PlanNode(id="finalize", tool="combine_research_rounds",
                           depends_on=tuple(fetch_ids), args=combine_args))
    nodes.append(PlanNode(id="report", tool="write_report", depends_on=("finalize",),
                           args={"title": "$title", "content": "$result:finalize", "report_dir": "reports"}))
    nodes.append(PlanNode(id="learn", tool="learn_report", depends_on=("report",),
                           args={"title": "$title", "text": "$result:finalize", "kind": "self_learned"}))

    return PlanGraph(id="deep_research", name="Deep research", goal=query, nodes=tuple(nodes))


def deep_research(query: str, embedder=None, client=None, model: str | None = None,
                         max_rounds: int = DEEP_RESEARCH_MAX_ROUNDS) -> str:
    """Public entry point — drop-in replacement for research.py's
    deep_research() when called from the graph executor."""
    if not query or not query.strip():
        return "[search failed: empty query]"
    session_id = _session_start()
    graph = _build_deep_research_subgraph(query.strip(), session_id, max_rounds=max_rounds)
    try:
        result = execute_graph(graph, embedder=embedder, llm_client=client, llm_model=model)
    except Exception:
        _session_end(session_id)  # combine_research_rounds's own cleanup never ran
        raise
    finalize = next((r for r in result.results if r.node_id == "finalize" and r.ok), None)
    return finalize.content if finalize else result.final_answer