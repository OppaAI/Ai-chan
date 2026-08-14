"""Deprecated shim — implementation lives in graph_export.py (EMC-5)."""
from __future__ import annotations
from interface.webui.studio.memory.ltm.backend.graph_export import (  # noqa: F401
    export_memory_graph,
    list_entity_relations,
    relations_as_graph_edges,
    _legend,
)
__all__ = [
    "export_memory_graph",
    "list_entity_relations",
    "relations_as_graph_edges",
]
