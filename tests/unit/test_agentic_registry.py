"""
tests/unit/test_agentic_registry.py

Unit tests for agentic/registry.py and unified tool registration.
"""

from __future__ import annotations

import pytest

from agentic.registry import ToolRegistry, tool
from agentic.capability import filtered_tool_schemas
from agentic.graph_engine import _build_tool_map


def test_registry_registration():
    local_reg = ToolRegistry()

    def dummy_tool(x: int) -> str:
        return f"result_{x}"

    spec = local_reg.register(
        name="test_tool",
        description="A test tool",
        handler=dummy_tool,
        props={"x": {"type": "integer"}},
        required=["x"],
        domain="testing",
        always_on=True,
    )

    assert spec.name == "test_tool"
    assert spec.domain == "testing"
    assert spec.always_on is True
    assert local_reg.get_tool_domains()["test_tool"] == "testing"
    assert "test_tool" in local_reg.get_always_on_tools()

    schema = spec.to_openai_schema()
    assert schema["function"]["name"] == "test_tool"
    assert schema["function"]["parameters"]["required"] == ["x"]


def test_tool_decorator():
    reg = ToolRegistry()

    def custom_tool_decorator(name, desc, **kwargs):
        def dec(fn):
            reg.register(name=name, description=desc, handler=fn, **kwargs)
            return fn
        return dec

    @custom_tool_decorator(
        name="decorated_func",
        desc="Decorated function description",
        props={"query": {"type": "string"}},
        required=["query"],
        domain="custom_domain",
    )
    def my_func(query: str) -> str:
        return f"processed: {query}"

    assert reg.get("decorated_func") is not None
    assert reg.get_tool_domains()["decorated_func"] == "custom_domain"
    assert reg.get_graph_tool_map()["decorated_func"]("hello") == "processed: hello"


def test_capability_integration():
    @tool(
        name="capability_registered_tool",
        description="Tool tested for capability routing",
        props={"param": {"type": "string"}},
        domain="repo",
    )
    def capability_test_tool(param: str) -> str:
        return param

    all_schemas = [
        {"type": "function", "function": {"name": "capability_registered_tool", "description": ""}}
    ]

    # Matching capability "repo" should retain capability_registered_tool
    filtered = filtered_tool_schemas(all_schemas, ["repo"])
    assert any(s["function"]["name"] == "capability_registered_tool" for s in filtered)


def test_graph_engine_integration():
    @tool(
        name="graph_dynamic_tool",
        description="Dynamic graph tool test",
        props={},
    )
    def dynamic_graph_fn() -> str:
        return "graph_ok"

    tool_map = _build_tool_map()
    assert "graph_dynamic_tool" in tool_map
    assert tool_map["graph_dynamic_tool"]() == "graph_ok"
