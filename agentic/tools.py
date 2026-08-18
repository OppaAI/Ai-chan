"""
agentic/tools.py

Compatibility facade for Aiko's autonomous toolkit.

This module provides a stable import surface for all tools. Tool metadata is
loaded from config/tools.yaml by agentic.registry, and each toolkit module binds
that metadata to its Python handler with @tool(TOOLS["tool_name"]). Importing
this facade imports those modules so their decorators run.
"""
from __future__ import annotations

# Import all toolkit modules to trigger @tool decorator registration
# Order matters: research depends on websearch, others are independent
from agentic.toolkit import websearch  # noqa: F401
from agentic.toolkit import ingest  # noqa: F401
from agentic.toolkit import synthesize  # noqa: F401

# Re-export helper functions used directly by cognition/think.py
# and other callers that expect a flat import surface from agentic.tools.
from agentic.toolkit.websearch import (
    web_search,
    web_search_context,
    web_fetch,
)
from agentic.toolkit.ingest import fetch_from_url
from agentic.toolkit import plan  # noqa: F401
from agentic.toolkit import organize  # noqa: F401
from agentic.toolkit import photography  # noqa: F401
from agentic.toolkit import self_improve  # noqa: F401
from agentic.toolkit import reports  # noqa: F401
from agentic.toolkit import research  # noqa: F401
from agentic.toolkit import tool_result_cache  # noqa: F401  # unified DAG tool-result cache
from agentic.workflows.job_hunt import toolset  # noqa: F401
from agentic.toolkit import social  # noqa: F401

# Re-export registry for tool registration
from agentic.registry import tool, registry, register_tool_schema

# Auto-generate __all__ from all decorated tools in registry
__all__ = [
    # Registry exports
    "tool",
    "registry",
    "register_tool_schema",
] + sorted(registry.get_all_tool_names())

# Bind registered tool handlers as module-level attributes so
# "from agentic.tools import adaptive_search" works for any tool.
import sys as _sys

for _name in registry.get_all_tool_names():
    _handler = registry.get(_name).handler
    if _handler is not None:
        _sys.modules[__name__].__dict__[_name] = _handler
