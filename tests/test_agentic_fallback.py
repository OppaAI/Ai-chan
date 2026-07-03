import os
import sys
import types

sys.modules.setdefault("yaml", types.SimpleNamespace(safe_load=lambda *_args, **_kwargs: {}))
sys.modules.setdefault("dotenv", types.SimpleNamespace(load_dotenv=lambda *_args, **_kwargs: None))

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.agentic import TaskState, ToolResult, _build_incomplete_task_answer


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
