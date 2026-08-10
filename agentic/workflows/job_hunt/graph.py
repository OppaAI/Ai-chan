"""
agentic/workflows/job_hunt/graph.py

Graph module for job_hunt (Lane D) workflow.
Registers graph tools and provides a single unified graph that works for both
user prompts and scheduled runs.

Importing this module auto-registers the graph with workflows.common.graphs
(and a local alias for backward compatibility).
"""

from __future__ import annotations

from agentic.graph_engine import PlanGraph, PlanNode
from agentic.registry import TOOLS, tool
from agentic.workflows.common.graphs import register_graph as _register_shared
from agentic.workflows.common.graphs import get_graph as get_shared_graph


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


def build_gen_job_post_graph(
    goal: str = "Fetch, draft, and save job listings from configured RSS feeds",
) -> PlanGraph:
    """Build the simplified gen_job_post graph."""
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


# Local alias kept for callers that still import job_hunt.graph.get_graph
def get_graph(graph_id: str):
    return get_shared_graph(graph_id)


def register_graph(graph: PlanGraph) -> None:
    _register_shared(graph)


register_graph(build_gen_job_post_graph())

__all__ = [
    "build_gen_job_post_graph",
    "register_graph",
    "get_graph",
]
