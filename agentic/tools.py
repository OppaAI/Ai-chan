"""
agentic/tools.py

Compatibility facade for Aiko's autonomous toolkit.

This module provides a stable import surface for all tools. The tool
definitions live in config/tools.yaml and are loaded at startup. The
@tool decorator in each module registers the handler with the global
registry; this file ensures all toolkit modules are imported so their
@tool decorators run, then loads additional tool metadata from YAML.
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
from agentic.toolkit import job_hunt  # noqa: F401
from agentic.toolkit import social  # noqa: F401

# Re-export registry for tool registration
from agentic.registry import tool, registry, register_tool_schema

# Load tool definitions from centralized YAML config
_TOOLS_YAML = "tools.yaml"
try:
    from system.config import load_tools_from_yaml
    _count = load_tools_from_yaml(_TOOLS_YAML)
    print(f"[tools] Loaded {_count} tool definitions from {_TOOLS_YAML}")
except Exception as e:
    print(f"Warning: Failed to load tools from {_TOOLS_YAML}: {e}")

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