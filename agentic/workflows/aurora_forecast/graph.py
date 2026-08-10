"""
agentic/workflows/aurora_forecast/graph.py

Layer 1 lane: hourly aurora forecast on shared nodes.

  ingest_data → store_data → synthesis_data → verify_results → output_user_results

Domain check remains available as adapter "aurora" and as tool check_aurora.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from agentic.graph_engine import PlanGraph, PlanNode
from agentic.registry import TOOLS, tool
from agentic.workflows.common.config import load_workflow_config
from agentic.workflows.common.graphs import register_graph

# Ensure shared Layer-1 nodes are registered.
import agentic.workflows.common.nodes  # noqa: F401

log = logging.getLogger(__name__)

_WORKFLOW_DIR = Path(__file__).resolve().parent

try:
    from agentic.workflows.aurora_forecast.toolset import check_aurora as _check_aurora

    _IMPORTS_OK = True
except Exception as exc:
    log.warning("aurora_forecast toolkit import failed: %s", exc)
    _IMPORTS_OK = False

    def _check_aurora(config_path: str = "", *, state=None) -> str:
        return '{"error": "aurora toolkit not available"}'


@tool(
    TOOLS["check_aurora"] if "check_aurora" in TOOLS else "check_aurora",
    description="Fetch NOAA aurora + Kp + cloud cover and score viewability.",
    graph=True,
    react=True,
    domain="weather",
)
def check_aurora(config_path: str = "", *, state=None) -> str:
    return _check_aurora(config_path, state=state)


def _cfg_json() -> str:
    cfg = load_workflow_config(_WORKFLOW_DIR)
    return json.dumps(cfg, ensure_ascii=False)


def build_aurora_forecast_graph(
    goal: str = "Check aurora visibility, store forecast, and notify when warranted",
) -> PlanGraph:
    cfg = load_workflow_config(_WORKFLOW_DIR)
    sources = cfg.get("sources") or [{"type": "adapter", "id": "aurora", "name": "aurora"}]
    template = str(cfg.get("template") or "{summary}")
    retain = str(cfg.get("retain_days") or 3)
    email = cfg.get("email") if isinstance(cfg.get("email"), dict) else {
        "enabled": True,
        "when": "interesting",
    }
    social = cfg.get("social") if isinstance(cfg.get("social"), list) else [
        {"platform": "threads", "when": {"field": "kp_index", "op": ">=", "value": 5.0}}
    ]
    config_json = json.dumps(cfg, ensure_ascii=False)

    nodes = [
        PlanNode(
            id="ingest",
            tool="ingest_data",
            args={
                "sources_json": json.dumps(sources),
                "filters_json": "{}",
                "parallel": "true",
                "max_items": "5",
                "config_json": config_json,
            },
        ),
        PlanNode(
            id="store",
            tool="store_data",
            args={
                "workflow_id": "aurora_forecast",
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
                "template": template,
                "llm_enriched": "false",
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
                "human_in_the_loop": "false",
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
        id="aurora_forecast",
        name="Aurora forecast (shared nodes)",
        goal=goal,
        nodes=tuple(nodes),
    )


register_graph(build_aurora_forecast_graph())

__all__ = ["build_aurora_forecast_graph", "check_aurora"]
