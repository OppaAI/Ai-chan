"""
agentic/workflows/aurora_forecast/graph.py

Lane: hourly aurora forecast for a configured location.

  check_aurora → store_forecast → notify_aurora

Registered into agentic.workflows.common.graphs for schedule_graphs.json.
"""

from __future__ import annotations

import logging

from agentic.graph_engine import PlanGraph, PlanNode
from agentic.registry import TOOLS, tool
from agentic.workflows.common.graphs import register_graph

log = logging.getLogger(__name__)

try:
    from agentic.workflows.aurora_forecast.toolset import (
        check_aurora as _check_aurora,
        store_aurora_forecast as _store_aurora_forecast,
        notify_aurora as _notify_aurora,
    )
    _IMPORTS_OK = True
except Exception as exc:
    log.warning("aurora_forecast toolkit import failed: %s", exc)
    _IMPORTS_OK = False

    def _check_aurora(config_path: str = "", *, state=None) -> str:
        return '{"error": "aurora toolkit not available"}'

    def _store_aurora_forecast(report_json: str = "", *, state=None) -> str:
        return '{"error": "aurora toolkit not available"}'

    def _notify_aurora(report_json: str = "", *, state=None) -> str:
        return '{"error": "aurora toolkit not available"}'


@tool(TOOLS["check_aurora"] if "check_aurora" in TOOLS else "check_aurora",
      description="Fetch NOAA aurora + Kp + cloud cover and score viewability.",
      graph=True, react=True, domain="weather")
def check_aurora(config_path: str = "", *, state=None) -> str:
    return _check_aurora(config_path, state=state)


@tool(TOOLS["store_aurora_forecast"] if "store_aurora_forecast" in TOOLS else "store_aurora_forecast",
      description="Persist aurora report to the shared workflow store (TTL).",
      graph=True, react=False, domain="weather")
def store_aurora_forecast(report_json: str = "", *, state=None) -> str:
    return _store_aurora_forecast(report_json, state=state)


@tool(TOOLS["notify_aurora"] if "notify_aurora" in TOOLS else "notify_aurora",
      description="Email aurora summary; post to Threads when Kp threshold met.",
      graph=True, react=False, domain="weather")
def notify_aurora(report_json: str = "", *, state=None) -> str:
    return _notify_aurora(report_json, state=state)


def build_aurora_forecast_graph(
    goal: str = "Check aurora visibility, store forecast, and notify when warranted",
) -> PlanGraph:
    nodes = [
        PlanNode(
            id="check",
            tool="check_aurora",
            args={"config_path": ""},
        ),
        PlanNode(
            id="store",
            tool="store_aurora_forecast",
            args={"report_json": "$result:check"},
            depends_on=("check",),
        ),
        PlanNode(
            id="notify",
            tool="notify_aurora",
            args={"report_json": "$result:check"},
            depends_on=("store",),
        ),
    ]
    return PlanGraph(
        id="aurora_forecast",
        name="Aurora forecast (NOAA + clouds)",
        goal=goal,
        nodes=tuple(nodes),
    )


register_graph(build_aurora_forecast_graph())

__all__ = ["build_aurora_forecast_graph", "check_aurora", "store_aurora_forecast", "notify_aurora"]
