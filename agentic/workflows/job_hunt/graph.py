"""
agentic/workflows/job_hunt/graph.py

Graph module for job_hunt (Lane D) workflow.
Registers graph tools and provides a single unified graph that works for both
user prompts and scheduled runs.

Importing this module auto-registers the graph with graph_engine.
"""

from __future__ import annotations

import os
from typing import Any

from agentic.graph_engine import PlanGraph, PlanNode
from agentic.registry import tool


# ──────────────────────────────────────────────────────────────────────────────
# Graph-tool registration (inline specs)
# ──────────────────────────────────────────────────────────────────────────────

try:
    from agentic.workflows.job_hunt.job_hunt import (
        fetch_rss_and_email_into_state as _fetch_rss_and_email_into_state,
        get_next_job as _get_next_job,
        draft_single_job as _draft_single_job,
        save_single_job_draft as _save_single_job_draft,
        check_jobs_remaining as _check_jobs_remaining,
        report_job_run as _report_job_run,
    )
    _IMPORTS_OK = True
except Exception as exc:
    import logging
    logging.getLogger(__name__).warning("job_hunt toolkit import failed: %s", exc)
    _IMPORTS_OK = False
    
    def _fetch_rss_and_email_into_state(plan_json: str, *, state=None) -> str:
        return '{"error": "toolkit not available"}'
    def _get_next_job(state=None, worker_id: str = "0") -> str:
        return '{"error": "toolkit not available"}'
    def _draft_single_job(job_json: str, template: str = "", *, client=None, model: str | None = None, state=None) -> str:
        return '{"error": "toolkit not available"}'
    def _save_single_job_draft(auto_post: str = "false", *, state=None) -> str:
        return '{"error": "toolkit not available"}'
    def _check_jobs_remaining(state=None) -> str:
        return '{"error": "toolkit not available"}'
    def _report_job_run(plan: str = "", search: str = "", draft: str = "", save: str = "") -> str:
        return '{"error": "toolkit not available"}'


# Register each as a graph tool
@tool(
    "fetch_rss_and_email_into_state",
    "Fetch all RSS + email jobs into state for incremental graph processing.",
    props={"plan_json": {"type": "string", "description": "JSON plan with max_results, etc."}},
    required=["plan_json"],
    domain="jobs",
    react=False,
    graph=True,
)
def fetch_rss_and_email_into_state(plan_json: str, *, state=None) -> str:
    return _fetch_rss_and_email_into_state(plan_json, state=state)


@tool(
    "get_next_job",
    "Get the next unprocessed job from state.job_all_postings (thread-safe).",
    props={"worker_id": {"type": "string", "description": "Worker identifier for logging"}},
    required=[],
    domain="jobs",
    react=False,
    graph=True,
)
def get_next_job(state=None, worker_id: str = "0") -> str:
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
    return _draft_single_job(job_json, template, client=client, model=model, state=state)


@tool(
    "save_single_job_draft",
    "Save the most recently drafted job to disk as a Threads draft.",
    props={"auto_post": {"type": "string", "description": "Auto-post flag (false = draft only)"}},
    required=[],
    domain="jobs",
    react=False,
    graph=True,
)
def save_single_job_draft(auto_post: str = "false", *, state=None) -> str:
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
    return _report_job_run(plan, search, draft, save)


# ──────────────────────────────────────────────────────────────────────────────
# Single unified graph: sequential execution
# ──────────────────────────────────────────────────────────────────────────────

def build_gen_job_post_graph(
    goal: str = "Fetch, draft, and save job listings from configured RSS feeds",
) -> PlanGraph:
    """Build the simplified gen_job_post graph.
    
    Single sequential workflow:
      1. Fetch all RSS + email jobs into state
      2. Loop: get next job → draft → save (until no jobs remain)
      3. Report results
    
    Works identically for both user prompt ("draft jobs") and scheduled 11pm run.
    """
    nodes = [
        PlanNode(
            id="fetch_all",
            tool="fetch_rss_and_email_into_state",
            args={"plan_json": '{"max_results": 30}'},
        ),
        PlanNode(
            id="loop_jobs",
            tool="get_next_job",
            args={"worker_id": "main"},
            depends_on=("fetch_all",),
            loop_to="loop_jobs",
            loop_condition={"not": {"contains": '"done": true'}},
            max_visits=100,
        ),
        PlanNode(
            id="draft",
            tool="draft_single_job",
            args={"job_json": "$result:loop_jobs", "template": ""},
            depends_on=("loop_jobs",),
        ),
        PlanNode(
            id="save",
            tool="save_single_job_draft",
            args={"auto_post": "false"},
            depends_on=("draft",),
        ),
        PlanNode(
            id="report",
            tool="report_job_run",
            args={"plan": "$result:fetch_all", "search": "{}", "draft": "{}", "save": "{}"},
            depends_on=("save",),
        ),
    ]

    return PlanGraph(
        id="gen_job_post",
        name="Fetch, draft, and save job listings from configured RSS feeds",
        goal=goal,
        nodes=tuple(nodes),
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


# Auto-register on import
register_graph(build_gen_job_post_graph())


__all__ = [
    "build_gen_job_post_graph",
    "register_graph",
    "get_graph",
]