"""
agentic/graph/job_hunt.py

Self-contained graph module for the job_hunt (Lane D) workflow.

Registers graph tools and provides a PlanGraph builder for the gen_job_post
playbook. Importing this module auto-registers the graph with graph_engine.
"""

from __future__ import annotations

import os
from typing import Any

from agentic.graph_engine import PlanGraph, PlanNode, _gen_job_worker_nodes
from agentic.registry import tool


# ──────────────────────────────────────────────────────────────────────────────
# Graph-tool registration (inline specs — not in tools.yaml)
# ──────────────────────────────────────────────────────────────────────────────
# These tools are imported from agentic.toolkit.job_hunt and registered here
# with graph=True so they appear in registry.get_graph_tool_map() and are
# available to the graph executor. The actual implementations live in
# agentic/toolkit/job_hunt.py — this module only handles graph-layer wiring.

# Import the implementations
from agentic.toolkit.job_hunt import (
    fetch_rss_and_email_into_state as _fetch_rss_and_email_into_state,
    get_next_job as _get_next_job,
    draft_single_job as _draft_single_job,
    save_single_job_draft as _save_single_job_draft,
    check_jobs_remaining as _check_jobs_remaining,
    report_job_run as _report_job_run,
)


# Register each as a graph tool (not react) using inline specs
@tool(
    "fetch_rss_and_email_into_state",
    "Fetch all RSS + email jobs into state for incremental graph processing.",
    props={
        "plan_json": {"type": "string", "description": "JSON plan with max_results, etc."},
    },
    required=["plan_json"],
    domain="jobs",
    react=False,
    graph=True,
)
def fetch_rss_and_email_into_state(plan_json: str, *, state=None) -> str:
    """Fetch all RSS + email jobs into state. Graph-only tool."""
    return _fetch_rss_and_email_into_state(plan_json, state=state)


@tool(
    "get_next_job",
    "Get the next unprocessed job from state.job_all_postings (thread-safe).",
    props={
        "worker_id": {"type": "string", "description": "Worker identifier for logging"},
    },
    required=[],
    domain="jobs",
    react=False,
    graph=True,
)
def get_next_job(state=None, worker_id: str = "0") -> str:
    """Get the next unprocessed job from state. Graph-only tool."""
    return _get_next_job(state=state, worker_id=worker_id)


@tool(
    "draft_single_job",
    "Draft a single job post from one job dict, optionally enriching with LLM.",
    props={
        "job_json": {"type": "string", "description": "JSON from get_next_job with job dict"},
        "template": {"type": "string", "description": "Optional template override"},
    },
    required=["job_json"],
    domain="jobs",
    react=False,
    graph=True,
)
def draft_single_job(
    job_json: str,
    template: str = "",
    *,
    client=None,
    model: str | None = None,
    state=None,
) -> str:
    """Draft a single job post from one job dict. Graph-only tool."""
    return _draft_single_job(job_json, template, client=client, model=model, state=state)


@tool(
    "save_single_job_draft",
    "Save the most recently drafted job to disk as a Threads draft.",
    props={
        "auto_post": {"type": "string", "description": "Auto-post flag (false = draft only)"},
    },
    required=[],
    domain="jobs",
    react=False,
    graph=True,
)
def save_single_job_draft(auto_post: str = "false", *, state=None) -> str:
    """Save the most recently drafted job to disk. Graph-only tool."""
    return _save_single_job_draft(auto_post, state=state)


@tool(
    "check_jobs_remaining",
    "Check if more jobs remain to be processed. Returns 'more' or 'done'.",
    props={},
    required=[],
    domain="jobs",
    react=False,
    graph=True,
)
def check_jobs_remaining(state=None) -> str:
    """Check if more jobs remain. Graph-only tool."""
    return _check_jobs_remaining(state=state)


@tool(
    "report_job_run",
    "Generate an RSS Lane D audit report from accumulated results.",
    props={
        "plan": {"type": "string", "description": "JSON from fetch_rss_and_email_into_state"},
        "search": {"type": "string", "description": "JSON search results"},
        "draft": {"type": "string", "description": "JSON draft results"},
        "save": {"type": "string", "description": "JSON save results"},
    },
    required=[],
    domain="jobs",
    react=False,
    graph=True,
)
def report_job_run(plan: str = "", search: str = "", draft: str = "", save: str = "") -> str:
    """Generate an RSS Lane D audit report. Graph-only tool."""
    return _report_job_run(plan, search, draft, save)


# ──────────────────────────────────────────────────────────────────────────────
# PlanGraph builder
# ──────────────────────────────────────────────────────────────────────────────

def build_gen_job_post_graph(
    max_workers: int | None = None,
    goal: str = "Fetch, draft, and save job listings from configured RSS feeds",
) -> PlanGraph:
    """Build the gen_job_post PlanGraph.

    This is the single source of truth for the graph structure. Both the
    chat path (plan_from_master) and the scheduled path (_run_schedule_graph)
    should call this instead of inline node definitions.

    Args:
        max_workers: Number of parallel worker chains (default from env or 2)
        goal: Goal string for the graph

    Returns:
        PlanGraph ready for execute_graph()
    """
    if max_workers is None:
        env_mw = os.getenv("JOB_HUNT_MAX_WORKERS", "").strip()
        mw = env_mw if env_mw else "2"
        try:
            max_workers = max(1, int(mw))
        except (TypeError, ValueError):
            max_workers = 2

    nodes = _gen_job_worker_nodes(
        fetch_tool="fetch_rss_and_email_into_state",
        check_tool="check_jobs_remaining",
        get_tool="get_next_job",
        draft_tool="draft_single_job",
        save_tool="save_single_job_draft",
        report_tool="report_job_run",
        max_workers=max_workers,
    )

    plan_nodes = []
    for raw in nodes:
        plan_nodes.append(PlanNode(
            id=str(raw["id"]),
            tool=str(raw["tool"]),
            args=dict(raw.get("args", {})),
            depends_on=tuple(str(d) for d in raw.get("depends_on", [])),
            loop_to=str(raw["loop_to"]) if raw.get("loop_to") else None,
            loop_condition=dict(raw["loop_condition"]) if isinstance(raw.get("loop_condition"), dict) else None,
            max_visits=int(raw.get("max_visits", 1) or 1),
        ))

    return PlanGraph(
        id="gen_job_post",
        name="Fetch, draft, and save job listings from configured RSS feeds (parallel workers)",
        goal=goal,
        nodes=tuple(plan_nodes),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Graph registry (auto-populated on import)
# ──────────────────────────────────────────────────────────────────────────────

_GRAPH_REGISTRY: dict[str, PlanGraph] = {}


def register_graph(graph: PlanGraph) -> None:
    """Register a PlanGraph by ID for lookup by graph_engine."""
    _GRAPH_REGISTRY[graph.id] = graph


def get_graph(graph_id: str) -> PlanGraph | None:
    """Retrieve a registered PlanGraph by ID."""
    return _GRAPH_REGISTRY.get(graph_id)


# Auto-register the gen_job_post graph on import
register_graph(build_gen_job_post_graph())


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

__all__ = [
    "build_gen_job_post_graph",
    "register_graph",
    "get_graph",
    "fetch_rss_and_email_into_state",
    "get_next_job",
    "draft_single_job",
    "save_single_job_draft",
    "check_jobs_remaining",
    "report_job_run",
]