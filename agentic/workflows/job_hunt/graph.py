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

import json
from pathlib import Path

from agentic.graph_engine import PlanGraph, PlanNode
from agentic.registry import TOOLS, tool
from agentic.workflows.common.graphs import get_graph as get_shared_graph
from agentic.workflows.common.graphs import register_graph as _register_shared

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


def _load_config() -> dict:
    path = Path(__file__).resolve().parent / "config.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def build_gen_job_post_graph(
    *,
    goal: str = "Fetch job listings, draft posts, save for human review",
) -> PlanGraph:
    """Layer 2: five shared nodes; domain work lives in adapters + toolset."""
    cfg = _load_config()
    config_json = json.dumps(cfg, ensure_ascii=False)
    sources = cfg.get("sources") or [{"type": "adapter", "id": "job_hunt", "name": "job_hunt"}]
    retain = str(cfg.get("retain_days") or cfg.get("dedup_days") or 3)
    max_items = str(cfg.get("max_results") or 30)
    hitl = "true" if cfg.get("human_in_the_loop", True) else "false"
    llm = "true" if cfg.get("llm_enriched", True) else "false"
    email = cfg.get("email") if isinstance(cfg.get("email"), dict) else {"enabled": False}
    social = cfg.get("social") if isinstance(cfg.get("social"), list) else []

    nodes = [
        PlanNode(
            id="ingest",
            tool="ingest_data",
            args={
                "sources_json": json.dumps(sources),
                "filters_json": json.dumps(cfg.get("filters") or {}),
                "parallel": "true",
                "max_items": max_items,
                "config_json": config_json,
            },
        ),
        PlanNode(
            id="store",
            tool="store_data",
            args={
                "workflow_id": "job_hunt",
                "items_json": "$result:ingest",
                "mode": "append",
                "retain_days": retain,
                "config_json": config_json,
            },
            depends_on=("ingest",),
        ),
        PlanNode(
            id="synth",
            tool="synthesis_data",
            args={
                "items_json": "$result:ingest",
                "template": "",
                "llm_enriched": llm,
                "per_item": "true",
                "config_json": config_json,
            },
            depends_on=("store",),
        ),
        PlanNode(
            id="verify",
            tool="verify_results",
            args={
                "results_json": "$result:synth",
                "human_in_the_loop": hitl,
                "llm_verify": "false",
                "auto_pass_json": "{}",
                "config_json": config_json,
            },
            depends_on=("synth",),
        ),
        PlanNode(
            id="output",
            tool="output_user_results",
            args={
                "results_json": "$result:verify",
                "email_json": json.dumps(email),
                "social_json": json.dumps(social),
                "config_json": config_json,
            },
            depends_on=("verify",),
        ),
    ]
    return PlanGraph(
        id="gen_job_post",
        name="Job hunt (shared nodes): ingest → store → synth → verify → output",
        goal=goal,
        nodes=tuple(nodes),
    )


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
