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
from agentic.workflows.common.spec_graph import build_plan_graph

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


def build_aurora_forecast_graph(
    goal: str = "Check aurora visibility, store forecast, and notify when warranted",
) -> PlanGraph:
    """Layer 3: Spec (or coerced config.json) → shared 5-node PlanGraph.

    Prefer ``spec.json`` when present so Layer-3 Specs are honored. Defaults
    below apply only on the config.json fallback path.
    """
    from agentic.workflows.common.spec import WorkflowSpec, coerce_config_to_spec, load_spec

    spec_path = _WORKFLOW_DIR / "spec.json"
    if spec_path.is_file():
        spec = load_spec(spec_path)
        if goal and goal != spec.goal:
            spec = WorkflowSpec(**{**spec.to_dict(), "goal": goal})
        return build_plan_graph(spec, goal=goal)

    cfg = load_workflow_config(_WORKFLOW_DIR)
    # Defaults only when coercing legacy config.json
    if "sources" not in cfg or not isinstance(cfg.get("sources"), list) or not cfg.get("sources"):
        cfg = {
            **cfg,
            "sources": [{"type": "adapter", "id": "aurora", "name": "aurora"}],
        }
    if "max_items" not in cfg and "max_results" not in cfg:
        cfg = {**cfg, "max_items": 5}
    if "email" not in cfg:
        cfg = {
            **cfg,
            "email": {"enabled": True, "when": "interesting"},
        }
    if "social" not in cfg:
        cfg = {
            **cfg,
            "social": [
                {"platform": "threads", "when": {"field": "kp_index", "op": ">=", "value": 5.0}}
            ],
        }
    spec = coerce_config_to_spec(
        graph_id="aurora_forecast",
        name="Aurora forecast (shared nodes)",
        goal=goal,
        config=cfg,
        workflow_id="aurora_forecast",
    )
    return build_plan_graph(spec, goal=goal)


register_graph(build_aurora_forecast_graph())

__all__ = ["build_aurora_forecast_graph", "check_aurora"]
