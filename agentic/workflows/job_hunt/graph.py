"""Job hunt Lane D — Layer 2 graph on shared workflow nodes.

Primary graph id remains ``gen_job_post`` (schedule / callers).

Flow:
  ingest_data (adapter:job_hunt → RSS + email via toolset)
  → store_data
  → synthesis_data (post_fields template + optional LLM)
  → verify_results (HITL drafts)
  → output_user_results (no auto Threads; human approves later)

Domain tools (fetch_rss_and_email_into_state, draft_single_job, …) stay
registered for adapters / ReAct / legacy.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

from agentic.graph_engine import PlanGraph, PlanNode
from agentic.registry import TOOLS, tool
from agentic.workflows.common.graphs import get_graph as get_shared_graph
from agentic.workflows.common.graphs import register_graph as _register_shared
from agentic.workflows.common.spec import coerce_config_to_spec
from agentic.workflows.common.spec_graph import build_plan_graph

# Ensure shared Layer-2 nodes (@tool ingest_data / store_data / …) are registered
# before this module's graphs are used. plan_from_master may import this file
# directly without going through common.graphs._ensure_workflow_graphs_loaded().
import agentic.workflows.common.nodes  # noqa: F401

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


def _spec(name: str):
    return TOOLS[name] if name in TOOLS else name


@tool(_spec("fetch_rss_and_email_into_state"))
def fetch_rss_and_email_into_state(plan_json: str, *, state=None) -> str:
    return _fetch_rss_and_email_into_state(plan_json, state=state)


@tool(_spec("get_next_job"))
def get_next_job(state=None, worker_id: str = "0") -> str:
    return _get_next_job(state=state, worker_id=worker_id)


@tool(_spec("draft_single_job"))
def draft_single_job(
    job_json: str,
    template: str = "",
    *,
    client=None,
    model: str | None = None,
    state=None,
) -> str:
    return _draft_single_job(job_json, template, client=client, model=model, state=state)


@tool(_spec("save_single_job_draft"))
def save_single_job_draft(auto_post: str = "false", *, state=None) -> str:
    return _save_single_job_draft(auto_post, state=state)


@tool(_spec("check_jobs_remaining"))
def check_jobs_remaining(state=None) -> str:
    return _check_jobs_remaining(state=state)


@tool(_spec("report_job_run"))
def report_job_run(plan: str = "", search: str = "", draft: str = "", save: str = "") -> str:
    return _report_job_run(plan, search, draft, save)


def build_gen_job_post_graph(
    *,
    goal: str = "Fetch job listings, draft posts, save for human review",
) -> PlanGraph:
    """Layer 3: Spec (or coerced config.json) → shared 5-node PlanGraph.

    Prefer ``spec.json`` when present so Layer-3 Specs are honored. Defaults
    below apply only on the config.json fallback path.
    """
    from agentic.workflows.common.config import load_workflow_config
    from agentic.workflows.common.spec import WorkflowSpec, load_spec

    workflow_dir = Path(__file__).resolve().parent
    spec_path = workflow_dir / "spec.json"
    if spec_path.is_file():
        spec = load_spec(spec_path)
        if goal and goal != spec.goal:
            spec = WorkflowSpec(**{**spec.to_dict(), "goal": goal})
        return build_plan_graph(spec, goal=goal)

    # Prefer the per-user config (<USER_SPACE>/<user_id>/.../job_hunt/config.json)
    # when present; fall back to the repo workflow config.json. _job_config()
    # already implements that user-first priority with repo fallback.
    try:
        from agentic.workflows.job_hunt.toolset import _job_config

        cfg = _job_config() or {}
    except Exception:
        cfg = {}
    if not cfg:
        cfg = load_workflow_config(workflow_dir)
    # Defaults matching Layer 2 behavior when keys are absent (config fallback only)
    if "sources" not in cfg:
        cfg = {
            **cfg,
            "sources": [{"type": "adapter", "id": "job_hunt", "name": "job_hunt"}],
        }
    if "human_in_the_loop" not in cfg:
        cfg = {**cfg, "human_in_the_loop": True}
    if "llm_enriched" not in cfg:
        cfg = {**cfg, "llm_enriched": True}
    if "email" not in cfg:
        cfg = {**cfg, "email": {"enabled": False}}
    if "social" not in cfg:
        cfg = {**cfg, "social": []}
    if "max_items" not in cfg and "max_results" in cfg:
        cfg = {**cfg, "max_items": cfg["max_results"]}
    elif "max_items" not in cfg and "max_results" not in cfg:
        cfg = {**cfg, "max_items": 30}

    spec = coerce_config_to_spec(
        graph_id="gen_job_post",
        name="Job hunt (shared nodes): ingest → store → synth → verify → output",
        goal=goal,
        config=cfg,
        workflow_id="job_hunt",
    )
    return build_plan_graph(spec, goal=goal)


def build_gen_job_post_legacy_graph(
    *,
    goal: str = "Legacy loop: fetch → get_next → draft → save → report",
) -> PlanGraph:
    """Previous loop-based graph; kept for rollback / comparison."""
    nodes = [
        PlanNode(
            id="fetch_all",
            tool="fetch_rss_and_email_into_state",
            args={"plan_json": "{}"},
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
        id="gen_job_post_legacy",
        name="Legacy job hunt loop graph",
        goal=goal,
        nodes=tuple(nodes),
    )


def get_graph(graph_id: str):
    return get_shared_graph(graph_id)


def register_graph(graph: PlanGraph) -> None:
    _register_shared(graph)


register_graph(build_gen_job_post_graph())
register_graph(build_gen_job_post_legacy_graph())

__all__ = [
    "build_gen_job_post_graph",
    "build_gen_job_post_legacy_graph",
    "register_graph",
    "get_graph",
]
