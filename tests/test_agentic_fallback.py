import os
import sys
import types

sys.modules.setdefault("yaml", types.SimpleNamespace(safe_load=lambda *_args, **_kwargs: {}))
sys.modules.setdefault("dotenv", types.SimpleNamespace(load_dotenv=lambda *_args, **_kwargs: None))

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.agentic import (
    TaskState,
    ToolResult,
    _build_incomplete_task_answer,
    _sanitize_user_facing_tool_detail,
)


def test_incomplete_task_answer_reports_successful_evidence_without_lost_apology():
    state = TaskState(goal="search for simple current info")
    state.record(ToolResult(ok=True, tool="web_search", args={"query": "Aiko"}, content="[Web search results]\n1. Useful result"))

    answer = _build_incomplete_task_answer(state)

    assert "I got a bit lost" not in answer
    assert "I completed these step(s):" in answer
    assert "web_search" in answer
    assert "Useful result" in answer


def test_incomplete_task_answer_discloses_blockers():
    state = TaskState(goal="search web")
    state.record(ToolResult(ok=False, tool="web_search", args={"query": "x"}, content="[search failed: connection refused]", error_type="search_failed"))

    answer = _build_incomplete_task_answer(state, "")

    assert "I got a bit lost" not in answer
    assert "could not fully complete" in answer
    assert "web_search" in answer
    assert "connection refused" in answer


def test_incomplete_task_answer_includes_latest_model_draft():
    state = TaskState(goal="draft something")

    answer = _build_incomplete_task_answer(state, "Draft answer from the model")

    assert "Most recent model draft:" in answer
    assert "Draft answer from the model" in answer


def test_incomplete_task_answer_handles_empty_state():
    state = TaskState(goal="do something")

    answer = _build_incomplete_task_answer(state)

    assert "step limit" in answer
    assert "no tool results were recorded" in answer


def test_tool_failure_detail_is_sanitized_for_user_facing_fallback():
    raw = (
        "Traceback (most recent call last):\n"
        "  File \"/workspace/Aiko-chan/core/toolkit/web.py\", line 1\n"
        "Authorization: Bearer abc.def.ghi token=super-secret "
        "http://localhost:8081/search?q=private"
    )

    sanitized = _sanitize_user_facing_tool_detail(raw)

    assert "super-secret" not in sanitized
    assert "abc.def.ghi" not in sanitized
    assert "localhost:8081" not in sanitized
    assert "/workspace/Aiko-chan" not in sanitized
    assert "[redacted]" in sanitized
    assert "[internal-url-redacted]" in sanitized
