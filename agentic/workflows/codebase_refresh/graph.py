"""
agentic/workflows/codebase_refresh/graph.py — nightly 22:00 refresh.
"""
from __future__ import annotations

import logging
from pathlib import Path

from agentic.graph_engine import PlanGraph, PlanNode
from agentic.registry import TOOLS, tool
from agentic.workflows.common.graphs import register_graph

log = logging.getLogger(__name__)
_WORKFLOW_DIR = Path(__file__).resolve().parent

try:
    from agentic.workflows.codebase_refresh.toolset import refresh_codebase, codebase_refresh_status
    _IMPORTS_OK = True
except Exception as exc:
    log.warning("codebase_refresh import failed: %s", exc)
    _IMPORTS_OK = False
    def refresh_codebase(*a, **kw): return '{"error":"toolkit not available"}'
    def codebase_refresh_status(*a, **kw): return '{"error":"toolkit not available"}'

@tool(
    TOOLS["refresh_codebase"] if "refresh_codebase" in TOOLS else "refresh_codebase",
    description="Refresh per-user codebase RAG DB (incremental SHA1, prune stale). Nightly 22:00, Jetson-optimized.",
    graph=True, react=True, domain="codebase",
)
def refresh_codebase_tool(*, state=None, **kwargs) -> str:
    return refresh_codebase(state=state, **kwargs)

@tool(
    TOOLS["codebase_refresh_status"] if "codebase_refresh_status" in TOOLS else "codebase_refresh_status",
    description="Check codebase.db existence and stats.",
    graph=True, react=True, domain="codebase",
)
def codebase_refresh_status_tool(*, state=None, **kwargs) -> str:
    return codebase_refresh_status(state=state, **kwargs)

def build_codebase_refresh_graph(goal: str = "Nightly refresh of codebase RAG DB at 22:00") -> PlanGraph:
    nodes = (
        PlanNode(id="status", tool="codebase_refresh_status", args={}),
        PlanNode(id="refresh", tool="refresh_codebase", args={}, depends_on=("status",)),
    )
    return PlanGraph(id="codebase_refresh", name="Codebase Refresh (nightly 22:00)", goal=goal, nodes=nodes)

register_graph(build_codebase_refresh_graph())
__all__ = ["build_codebase_refresh_graph", "refresh_codebase_tool", "codebase_refresh_status_tool"]
