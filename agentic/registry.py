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

from system.config import load_yaml


@dataclass
class ToolSpec:
    name: str
    description: str
    handler: Optional[Callable[..., Any]] = None
    props: Dict[str, Any] = field(default_factory=dict)
    required: List[str] = field(default_factory=list)
    domain: Optional[str] = None  # capability routing (research, scheduling, etc.)
    always_on: bool = False  # always included in tool list regardless of capability match
    # Execution modes - which backends can execute this tool
    react: bool = True   # ReAct loop
    graph: bool = True   # graph_engine playbook
    wiki: bool = False   # wiki workflow
    skill: bool = False  # skill workflow

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
        react: bool = True,
        graph: bool = False,
        wiki: bool = False,
        skill: bool = False,
    ) -> ToolSpec:
        spec = ToolSpec(
            name=name,
            description=description,
            handler=handler,
            props=props or {},
            required=required or [],
            domain=domain,
            always_on=always_on,
            react=react,
            graph=graph,
            wiki=wiki,
            skill=skill,
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

    def get_wiki_tools(self) -> Dict[str, Callable[..., Any]]:
        """Return dict mapping tool_name -> handler for wiki workflow."""
        return {
            spec.name: spec.handler
            for spec in self._tools.values()
            if spec.wiki and spec.handler is not None
        }

    def get_skill_tools(self) -> Dict[str, Callable[..., Any]]:
        """Return dict mapping tool_name -> handler for skill workflow."""
        return {
            spec.name: spec.handler
            for spec in self._tools.values()
            if spec.skill and spec.handler is not None
        }

    def get_all_tool_names(self) -> List[str]:
        """Return list of all registered tool names for auto-export."""
        return list(self._tools.keys())

    def to_catalog(self) -> dict[str, Any]:
        """Return a deterministic, serializable catalog of registered tools.

        The catalog is intended for documentation and drift checks. Runtime
        registration should continue to use decorators/register_tool_schema so
        there is only one source of truth.
        """
        tools: list[dict[str, Any]] = []
        for spec in sorted(self._tools.values(), key=lambda item: item.name):
            entry: dict[str, Any] = {
                "name": spec.name,
                "description": spec.description,
            }
            if spec.handler is not None:
                entry["handler"] = f"{spec.handler.__module__}:{spec.handler.__name__}"
            if spec.props:
                entry["props"] = spec.props
            if spec.required:
                entry["required"] = list(spec.required)
            if spec.domain is not None:
                entry["domain"] = spec.domain
            if spec.always_on:
                entry["always_on"] = spec.always_on
            entry["react"] = spec.react
            entry["graph"] = spec.graph
            if spec.wiki:
                entry["wiki"] = spec.wiki
            if spec.skill:
                entry["skill"] = spec.skill
            tools.append(entry)
        return {"tools": tools}


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
    react: bool = True,    # ReAct is the default execution mode
    graph: bool = False,   # Must opt-in for graph engine
    wiki: bool = False,    # Must opt-in for wiki workflow
    skill: bool = False,   # Must opt-in for skill workflow
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to register a function as an agentic tool.

    Args:
        name: Tool name for OpenAI schema and registry lookup
        description: Human-readable description for LLM tool selection
        props: Parameter schema dict
        required: List of required parameter names
        domain: Capability domain for routing (research, scheduling, etc.)
        always_on: Always include in tool list regardless of capability match
        react: Available in ReAct loop (default: True)
        graph: Available in graph_engine playbook (default: True)
        wiki: Available in wiki workflow (default: False)
        skill: Available in skill workflow (default: False)
    """
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        registry.register(
            name=name,
            description=description,
            handler=fn,
            props=props,
            required=required,
            domain=domain,
            always_on=always_on,
            react=react,
            graph=graph,
            wiki=wiki,
            skill=skill,
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
    react: bool = True,
    graph: bool = True,
    wiki: bool = False,
    skill: bool = False,
) -> ToolSpec:
    """Register a tool schema whose execution is handled elsewhere.

    Use for tools where the handler isn't a Python function (e.g., external
    services, graph nodes, etc.). The registry still tracks the schema and
    execution modes for routing purposes.
    """
    return registry.register(
        name=name,
        description=description,
        handler=None,
        props=props,
        required=required,
        domain=domain,
        always_on=always_on,
        react=react,
        graph=graph,
        wiki=wiki,
        skill=skill,
    )
