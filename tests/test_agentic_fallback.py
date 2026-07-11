import os
import sys
import types

sys.modules.setdefault("yaml", types.SimpleNamespace(safe_load=lambda *_args, **_kwargs: {}))
sys.modules.setdefault("dotenv", types.SimpleNamespace(load_dotenv=lambda *_args, **_kwargs: None))

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skills.agentic import (
    TaskState,
    ToolResult,
    _build_incomplete_task_answer,
    _compact_processed_research_context,
    _has_successful_tool_call,
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
        "  File \"/workspace/Aiko-chan/core/toolkit/researcher.py\", line 1\n"
        "Authorization: Bearer abc.def.ghi token=super-secret "
        "http://localhost:8888/search?q=private"
    )

    sanitized = _sanitize_user_facing_tool_detail(raw)

    assert "super-secret" not in sanitized
    assert "abc.def.ghi" not in sanitized
    assert "localhost:8888" not in sanitized
    assert "/workspace/Aiko-chan" not in sanitized
    assert "[redacted]" in sanitized
    assert "[internal-url-redacted]" in sanitized


def test_processed_deep_search_context_is_compacted_after_use():
    messages = [
        {"role": "tool", "name": "deep_search", "content": "x" * 1200},
        {"role": "assistant", "content": "I processed the evidence and will save a note."},
    ]

    _compact_processed_research_context(messages)

    assert len(messages[0]["content"]) < 800
    assert "research_context_compacted" in messages[0]["content"]


def test_deep_search_limit_only_counts_successful_calls():
    state = TaskState(goal="research with one successful deep search")
    state.record(ToolResult(ok=False, tool="deep_search", args={"query": "bad"}, content="[search failed: timeout]"))

    assert not _has_successful_tool_call(state, "deep_search")

    state.record(ToolResult(ok=True, tool="deep_search", args={"query": "good"}, content="[Web search results]"))

    assert _has_successful_tool_call(state, "deep_search")
