"""
agentic/tools.py

Compatibility facade for Aiko's autonomous toolkit.

This module provides a stable import surface for all tools decorated with
@tool() in the agentic/toolkit/ submodules. The __all__ list is auto-generated
from the registry, so adding a new @tool decorator automatically includes it
here without manual list maintenance.
"""
from __future__ import annotations

# Import all toolkit modules to trigger @tool decorator registration
# Order matters: research depends on websearch, others are independent
from agentic.toolkit import websearch  # noqa: F401
from agentic.toolkit import ingest  # noqa: F401
from agentic.toolkit import synthesize  # noqa: F401
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

# Auto-generate __all__ from all decorated tools in registry
__all__ = [
    # Registry exports
    "tool",
    "registry",
    "register_tool_schema",
] + sorted(registry.get_all_tool_names())
