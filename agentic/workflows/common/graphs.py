"""Central PlanGraph registry for all workflows.

Schedule and graph_engine resolve graphs by id through this module so
each workflow does not need a bespoke import in system/schedule.py.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentic.graph_engine import PlanGraph

log = logging.getLogger(__name__)

_GRAPH_REGISTRY: dict[str, "PlanGraph"] = {}


def register_graph(graph: "PlanGraph") -> None:
    """Register a PlanGraph by id (last writer wins)."""
    _GRAPH_REGISTRY[graph.id] = graph
    log.debug("workflows.common: registered graph %s", graph.id)


def get_graph(graph_id: str) -> "PlanGraph | None":
    """Lookup a registered graph; ensures workflow modules are imported."""
    if graph_id not in _GRAPH_REGISTRY:
        _ensure_workflow_graphs_loaded()
    return _GRAPH_REGISTRY.get(graph_id)


def list_graphs() -> list[str]:
    _ensure_workflow_graphs_loaded()
    return sorted(_GRAPH_REGISTRY.keys())


def _ensure_workflow_graphs_loaded() -> None:
    """Import workflow graph modules so they self-register."""
    modules = (
        "agentic.workflows.job_hunt.graph",
        "agentic.workflows.aurora_forecast.graph",
    )
    for mod in modules:
        try:
            __import__(mod)
        except Exception as e:
            log.debug("workflows.common: could not import %s: %s", mod, e)
