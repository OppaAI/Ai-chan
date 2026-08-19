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
        graph=True,
    )
    def my_func(query: str) -> str:
        return f"processed: {query}"

    assert reg.get("decorated_func") is not None
    assert reg.get_tool_domains()["decorated_func"] == "custom_domain"
    assert reg.get_graph_tool_map()["decorated_func"]("hello") == "processed: hello"


def test_capability_integration():
    @tool(
        "capability_registered_tool",
        "Tool tested for capability routing",
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
        "graph_dynamic_tool",
        "Dynamic graph tool test",
        props={},
        graph=True,
    )
    def dynamic_graph_fn() -> str:
        return "graph_ok"

    tool_map = _build_tool_map()
    assert "graph_dynamic_tool" in tool_map
    assert tool_map["graph_dynamic_tool"]() == "graph_ok"


def test_pydantic_args_model_generates_schema_and_coerces():
    from pydantic import BaseModel, Field

    class DemoArgs(BaseModel):
        count: int = Field(ge=1)
        label: str

    spec = ToolRegistry().register(
        name="typed_demo",
        description="Typed demo",
        args_model=DemoArgs,
    )

    schema = spec.to_openai_schema()["function"]["parameters"]
    assert schema["properties"]["count"]["type"] == "integer"
    assert "count" in schema["required"]
    assert spec.validate_args({"count": "2", "label": "ok"}) == {"count": 2, "label": "ok"}


def test_pydantic_output_model_validates_json():
    from pydantic import BaseModel

    class FinalShape(BaseModel):
        summary: str
        confidence: float

    spec = ToolRegistry().register(
        name="structured_final",
        description="Structured final answer",
        output_model=FinalShape,
    )

    assert spec.validate_output('{"summary":"done","confidence":0.8}') == {
        "summary": "done",
        "confidence": 0.8,
    }


def test_high_churn_tools_have_pydantic_arg_models():
    from agentic.registry import TOOLS

    for name in [
        "save_note", "schedule_job", "schedule_reminder", "write_report",
        "adaptive_search", "deep_research", "deep_read", "repo_file_tree", "repo_read_file", "repo_search_text",
        "draft_job_post_social", "post_job_post_social", "post_to_social",
        "draft_photo_social", "post_photo_social", "draft_video_social", "post_video_social",
    ]:
        assert TOOLS[name].args_model is not None
        assert TOOLS[name].to_openai_schema()["function"]["parameters"]["type"] == "object"


def test_social_post_tools_require_hitl_approval():
    from agentic.registry import TOOLS

    assert TOOLS["post_job_post_social"].needs_approval is True
    assert TOOLS["post_photo_social"].needs_approval is True
    assert TOOLS["post_video_social"].needs_approval is True
    assert TOOLS["post_to_social"].needs_approval is True


def test_schedule_job_args_fail_fast_for_conditional_fields():
    import pytest
    from agentic.tool_models import ScheduleJobArgs

    with pytest.raises(ValueError):
        ScheduleJobArgs(title="t", task="task", time_of_day="09:00", action="tool")
    with pytest.raises(ValueError):
        ScheduleJobArgs(title="t", task="task", time_of_day="09:00", frequency="custom_weekdays")
