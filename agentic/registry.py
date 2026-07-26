"""
agentic/registry.py

Centralized tool registry for Aiko's agentic framework.

This module provides a single source of truth for tool definitions, OpenAI schemas,
capability domain tags, always-on flags, and execution handlers across ReAct and
Graph execution loops.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


@dataclass
class ToolSpec:
    name: str
    description: str
    handler: Optional[Callable[..., Any]] = None
    props: Dict[str, Any] = field(default_factory=dict)
    required: List[str] = field(default_factory=list)
    domain: Optional[str] = None
    always_on: bool = False
    graph: bool = True
    react: bool = True

    def to_openai_schema(self) -> dict[str, Any]:
        """Convert ToolSpec to OpenAI function schema format."""
        s = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {"type": "object", "properties": self.props or {}},
            },
        }
        if self.required:
            s["function"]["parameters"]["required"] = list(self.required)
        return s


class ToolRegistry:
    """Singleton registry tracking all declared agentic tools."""

    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}

    def register(
        self,
        name: str,
        description: str,
        handler: Optional[Callable[..., Any]] = None,
        props: Optional[Dict[str, Any]] = None,
        required: Optional[List[str]] = None,
        domain: Optional[str] = None,
        always_on: bool = False,
        graph: bool = True,
        react: bool = True,
    ) -> ToolSpec:
        spec = ToolSpec(
            name=name,
            description=description,
            handler=handler,
            props=props or {},
            required=required or [],
            domain=domain,
            always_on=always_on,
            graph=graph,
            react=react,
        )
        self._tools[name] = spec
        return spec

    def get(self, name: str) -> Optional[ToolSpec]:
        return self._tools.get(name)

    def all_specs(self) -> List[ToolSpec]:
        return list(self._tools.values())

    def get_tool_domains(self) -> Dict[str, str]:
        """Return dict mapping tool_name -> domain tag for capability routing."""
        return {
            spec.name: spec.domain
            for spec in self._tools.values()
            if spec.domain is not None
        }

    def get_always_on_tools(self) -> Set[str]:
        """Return set of tool names marked as always_on."""
        return {
            spec.name
            for spec in self._tools.values()
            if spec.always_on
        }

    def get_react_defs(self) -> List[Tuple[dict[str, Any], Optional[Callable[..., Any]]]]:
        """Return list of (schema, handler) tuples for agentic.py ReAct loop."""
        return [
            (spec.to_openai_schema(), spec.handler)
            for spec in self._tools.values()
            if spec.react
        ]

    def get_graph_tool_map(self) -> Dict[str, Callable[..., Any]]:
        """Return dict mapping tool_name -> handler for graph_engine.py."""
        return {
            spec.name: spec.handler
            for spec in self._tools.values()
            if spec.graph and spec.handler is not None
        }


# Global registry instance
registry = ToolRegistry()


def tool(
    name: str,
    description: str,
    *,
    props: Optional[Dict[str, Any]] = None,
    required: Optional[List[str]] = None,
    domain: Optional[str] = None,
    always_on: bool = False,
    graph: bool = True,
    react: bool = True,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to register a function as an agentic tool."""
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        registry.register(
            name=name,
            description=description,
            handler=fn,
            props=props,
            required=required,
            domain=domain,
            always_on=always_on,
            graph=graph,
            react=react,
        )
        return fn
    return decorator


def register_tool_schema(
    name: str,
    description: str,
    *,
    props: Optional[Dict[str, Any]] = None,
    required: Optional[List[str]] = None,
    domain: Optional[str] = None,
    always_on: bool = False,
    graph: bool = True,
    react: bool = True,
) -> ToolSpec:
    """Register a ReAct/graph tool schema whose execution is handled elsewhere."""
    return registry.register(
        name=name,
        description=description,
        handler=None,
        props=props,
        required=required,
        domain=domain,
        always_on=always_on,
        graph=graph,
        react=react,
    )
