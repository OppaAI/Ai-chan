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
from agentic.registry import TOOLS, tool


# ──────────────────────────────────────────────────────────────────────────────
# Graph-tool registration (inline specs)
# ──────────────────────────────────────────────────────────────────────────────

try:
    from agentic.workflows.job_hunt.toolset import (
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
@tool(TOOLS["fetch_rss_and_email_into_state"])
def fetch_rss_and_email_into_state(plan_json: str, *, state=None) -> str:
    return _fetch_rss_and_email_into_state(plan_json, state=state)


@tool(TOOLS["get_next_job"])
def get_next_job(state=None, worker_id: str = "0") -> str:
    return _get_next_job(state=state, worker_id=worker_id)


@tool(TOOLS["draft_single_job"])
def draft_single_job(
    job_json: str,
    template: str = "",
    *,
    client=None,
    model: str | None = None,
    state=None,
) -> str:
    return _draft_single_job(job_json, template, client=client, model=model, state=state)


@tool(TOOLS["save_single_job_draft"])
def save_single_job_draft(auto_post: str = "false", *, state=None) -> str:
    return _save_single_job_draft(auto_post, state=state)


@tool(TOOLS["check_jobs_remaining"])
def check_jobs_remaining(state=None) -> str:
    return _check_jobs_remaining(state=state)


@tool(TOOLS["report_job_run"])
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