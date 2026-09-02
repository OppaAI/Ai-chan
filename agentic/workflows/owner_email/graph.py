"""
agentic/workflows/owner_email/graph.py

Layer 1: owner email bridge — poll → reply.

Graph: check_owner_email → reply_owner_email
Runs every 10 minutes via schedule. Each step is idempotent via processed.json.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from agentic.graph_engine import PlanGraph, PlanNode
from agentic.registry import TOOLS, tool
from agentic.workflows.common.config import load_workflow_config
from agentic.workflows.common.graphs import register_graph
from agentic.workflows.common.spec_graph import build_plan_graph

import agentic.workflows.common.nodes  # noqa: F401

log = logging.getLogger(__name__)
_WORKFLOW_DIR = Path(__file__).resolve().parent

try:
    from agentic.workflows.owner_email.toolset import check_owner_email as _check, reply_owner_email as _reply
    _IMPORTS_OK = True
except Exception as exc:
    log.warning("owner_email toolkit import failed: %s", exc)
    _IMPORTS_OK = False
    def _check(*a, **kw): return '{"error":"toolkit not available"}'
    def _reply(*a, **kw): return '{"error":"toolkit not available"}'


@tool(
    TOOLS["check_owner_email"] if "check_owner_email" in TOOLS else "check_owner_email",
    description="Poll ProtonMail inbox for messages FROM AIKO_EMAIL and extract prompts.",
    graph=True,
    react=True,
    domain="email",
)
def check_owner_email(max_results: int = 5, *, state=None) -> str:
    return _check(max_results=max_results, state=state)


@tool(
    TOOLS["reply_owner_email"] if "reply_owner_email" in TOOLS else "reply_owner_email",
    description="Generate LLM replies for owner email prompts and send them back.",
    graph=True,
    react=True,
    domain="email",
)
def reply_owner_email(report_json: str = "", *, state=None) -> str:
    return _reply(report_json, state=state)


def build_owner_email_graph(goal: str = "Poll owner email, reply with LLM answers") -> PlanGraph:
    from agentic.workflows.common.spec import WorkflowSpec, coerce_config_to_spec, load_spec
    spec_path = _WORKFLOW_DIR / "spec.json"
    if spec_path.is_file():
        spec = load_spec(spec_path)
        if goal and goal != spec.goal:
            spec = WorkflowSpec(**{**spec.to_dict(), "goal": goal})
        return build_plan_graph(spec, goal=goal)
    cfg = load_workflow_config(_WORKFLOW_DIR)
    spec = coerce_config_to_spec(
        graph_id="owner_email",
        name="Owner email bridge (poll→reply)",
        goal=goal,
        config=cfg,
        workflow_id="owner_email",
    )
    return build_plan_graph(spec, goal=goal)


register_graph(build_owner_email_graph())

__all__ = ["build_owner_email_graph", "check_owner_email", "reply_owner_email"]
