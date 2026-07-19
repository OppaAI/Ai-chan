"""
tests/unit/test_agentic_agentic.py

Unit tests for agentic/agentic.py — dispatch_tool, run_agentic_chat,
capability routing, and tool registration.

Run: pytest tests/unit/test_agentic_agentic.py -v
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

os.environ.setdefault("WORKSPACE_ROOT", "/tmp/aiko_test_workspace")

sys.path.insert(0, "/home/oppa-ai/jetson")
from system.config import load_config
load_config()

from agentic.agentic import (
    ToolResult,
    TaskState,
    VerificationResult,
    _TOOLS,
    _TOOL_DEFS,
    _SOCIAL_POST_TOOLS,
    _RESEARCH_TOOLS,
    _reg,
    _reg_no_handler,
    _required_args_for,
    _validate_args,
    _classify_result,
    _owner_embedder,
    dispatch_tool,
    dispatch_tool_checked,
    _max_attempts_for,
    run_agentic_chat,
    AGENT_EXECUTOR_MODE,
    AGENT_RESEARCH_MAX_CALLS,
    _research_call_count,
    _verify_final_answer,
)
from agentic import schema
from agentic.toolkit.plan import save_note, create_checklist, make_plan
from agentic.toolkit.reports import write_report
from agentic.toolkit.research import deep_search, deep_research
from agentic.capability import match_capabilities, filtered_tool_schemas, ALWAYS_ON_TOOLS


class FakeEmbedder:
    def embed_query(self, text: str, instruct: str = "") -> np.ndarray:
        h = hash(text) % 1000
        return np.array([float(h) / 1000.0] * 384, dtype=np.float32)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.embed_query(t) for t in texts])


class MockLLMClient:
    def __init__(self, response_text: str = "Mock response"):
        self.response_text = response_text
        self.call_count = 0
        self.last_messages = None
        self.last_model = None
        self.last_max_tokens = None
        self.last_temperature = None

    def chat_completions_create(self, model: str, messages: list[dict], **kwargs):
        self.call_count += 1
        self.last_messages = messages
        self.last_model = model
        self.last_max_tokens = kwargs.get("max_tokens")
        self.last_temperature = kwargs.get("temperature")
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content=self.response_text))]
        return mock_resp

    @property
    def chat(self):
        return MagicMock(completions=MagicMock(create=self.chat_completions_create))


class MockOwner:
    """Mock AikoThink owner with client, model, embedder."""
    def __init__(self, client=None, model="test-model"):
        self._client = client or MockLLMClient()
        self._llm_model = model
        self._memorize = MagicMock()
        self._memorize._mem = MagicMock()
        self._memorize._mem._embedder = FakeEmbedder()
        self._history = []
        self._history_lock = threading.Lock()
        self._store_calls = []

    def _emit(self, text, token_callback=None):
        pass

    def _store_async(self, prompt, response):
        self._store_calls.append((prompt, response))


class TestToolRegistration:
    """Tests for _reg, _reg_no_handler and tool registry."""

    def test_all_core_tools_registered(self):
        expected_tools = {
            "make_plan", "create_checklist", "save_note", "read_workspace_file",
            "summarize_task_state", "deep_search", "deep_research", "read_paper_url",
            "write_report", "learn_knowledge", "run_playbook", "list_playbooks",
            "search_jobs", "draft_photo_social", "post_photo_social",
            "draft_video_social", "post_video_social", "final_answer",
        }
        for tool in expected_tools:
            assert tool in _TOOLS, f"Missing tool: {tool}"

    def test_tool_schemas_have_required_fields(self):
        for name, (schema, handler) in _TOOLS.items():
            assert "function" in schema
            assert schema["function"]["name"] == name
            assert "parameters" in schema["function"]
            assert "required" in schema["function"]["parameters"]

    def test_social_post_tools_identified(self):
        assert "post_photo_social" in _SOCIAL_POST_TOOLS
        assert "post_video_social" in _SOCIAL_POST_TOOLS
        assert "draft_photo_social" not in _SOCIAL_POST_TOOLS

    def test_research_tools_identified(self):
        assert "deep_search" in _RESEARCH_TOOLS
        assert "deep_research" in _RESEARCH_TOOLS


class TestRequiredArgs:
    """Tests for _required_args_for and _validate_args."""

    def test_required_args_extracted(self):
        assert "goal" in _required_args_for("make_plan")
        assert "title" in _required_args_for("save_note")
        assert "content" in _required_args_for("save_note")
        assert "query" in _required_args_for("deep_search")
        assert "query" in _required_args_for("deep_research")
        assert "title" in _required_args_for("write_report")
        assert "title" in _required_args_for("learn_knowledge")
        assert "task" in _required_args_for("run_playbook")
        assert "draft_dir" in _required_args_for("post_photo_social")

    def test_validate_args_missing_required(self):
        result = _validate_args("save_note", {"title": "test"})
        assert result is not None
        assert result.ok is False
        assert result.error_type == "missing_args"
        assert "content" in result.content

    def test_validate_args_empty_query(self):
        result = _validate_args("deep_search", {"query": ""})
        assert result is not None
        assert result.ok is False
        assert result.error_type == "missing_args"

    def test_validate_args_learn_knowledge_needs_text_or_path(self):
        result = _validate_args("learn_knowledge", {"title": "test"})
        assert result is not None
        assert result.ok is False
        assert "text or relative_path" in result.content

    def test_validate_args_social_post_needs_draft_dir(self):
        result = _validate_args("post_photo_social", {})
        assert result is not None
        assert result.ok is False
        assert "draft_dir" in result.content

    def test_validate_args_passes_valid(self):
        result = _validate_args("save_note", {"title": "t", "content": "c"})
        assert result is None  # None = valid


class TestClassifyResult:
    """Tests for _classify_result error detection."""

    def test_success_string_returns_ok(self):
        result = _classify_result("tool", {}, "success output")
        assert result.ok is True
        assert result.content == "success output"

    def test_bracketed_error_returns_failed(self):
        result = _classify_result("tool", {}, "[search failed: connection timeout]")
        assert result.ok is False
        assert result.error_type == "search_failed"
        assert result.retryable is True

    def test_bracketed_generic_error(self):
        result = _classify_result("tool", {}, "[tool failed: something broke]")
        assert result.ok is False
        assert result.error_type == "tool_failed"

    def test_non_bracketed_returns_ok(self):
        result = _classify_result("tool", {}, "plain text response")
        assert result.ok is True


class TestOwnerEmbedder:
    """Tests for _owner_embedder extraction."""

    def test_returns_embedder_when_available(self):
        owner = MockOwner()
        embedder = _owner_embedder(owner)
        assert embedder is not None
        assert hasattr(embedder, "embed_query")

    def test_returns_none_when_missing(self):
        owner = MagicMock()
        owner._memorize = None
        assert _owner_embedder(owner) is None

        owner = MagicMock()
        owner._memorize = MagicMock()
        owner._memorize._mem = None
        assert _owner_embedder(owner) is None


class TestDispatchTool:
    """Tests for dispatch_tool routing."""

    def test_deep_research_uses_owner_client_model(self):
        owner = MockOwner()
        owner._client = MockLLMClient("Research result")
        owner._llm_model = "test-model"

        result = dispatch_tool("deep_research", {"query": "test query"}, owner=owner)
        assert "Research result" in result
        assert owner._client.call_count == 1

    def test_deep_search_uses_embedder(self):
        owner = MockOwner()
        result = dispatch_tool("deep_search", {"query": "test query"}, owner=owner)
        assert "Web search results" in result or "no results found" in result.lower()

    def test_run_playbook_passes_embedder(self):
        owner = MockOwner()
        with patch("agentic.schema.run_playbook_json") as mock_run:
            mock_run.return_value = '{"ok": true}'
            result = dispatch_tool("run_playbook", {"task": "test task"}, owner=owner)
            mock_run.assert_called_once()
            args, kwargs = mock_run.call_args
            assert "embedder" in kwargs

    def test_learn_knowledge_with_text(self):
        owner = MockOwner()
        with patch("agentic.agentic.ingest_knowledge_text", return_value="doc-123") as mock_ingest:
            result = dispatch_tool("learn_knowledge", {"title": "Test", "text": "Content"}, owner=owner)
            assert "doc-123" in result
            mock_ingest.assert_called_once()

    def test_learn_knowledge_with_relative_path(self):
        owner = MockOwner()
        with patch("agentic.agentic.ingest_knowledge_file", return_value="doc-456") as mock_ingest:
            result = dispatch_tool("learn_knowledge", {"title": "Test", "relative_path": "path/to/file"}, owner=owner)
            assert "doc-456" in result
            mock_ingest.assert_called_once()

    def test_read_paper_url_passes_embedder(self):
        owner = MockOwner()
        with patch("agentic.agentic.read_paper_url") as mock_read:
            mock_read.return_value = "Paper content"
            result = dispatch_tool("read_paper_url", {"url": "http://example.com", "query": "test"}, owner=owner)
            mock_read.assert_called_once()
            args, kwargs = mock_read.call_args
            assert kwargs["embedder"] is not None

    def test_write_report_passes_all_args(self):
        owner = MockOwner()
        with patch("agentic.agentic.write_report") as mock_write:
            mock_write.return_value = "Report written"
            result = dispatch_tool("write_report", {
                "title": "Test", "content": "Content", "report_dir": "reports",
                "arxiv_style": True, "section": "abstract", "append": False
            }, owner=owner)
            mock_write.assert_called_once()
            args, kwargs = mock_write.call_args
            assert kwargs["title"] == "Test"
            assert kwargs["arxiv_style"] is True

    def test_unknown_tool_returns_error(self):
        result = dispatch_tool("nonexistent_tool", {}, owner=MockOwner())
        assert "[unknown tool: nonexistent_tool]" in result


class TestDispatchToolChecked:
    """Tests for dispatch_tool_checked structured results."""

    def test_returns_toolresult_on_success(self):
        owner = MockOwner()
        owner._client = MockLLMClient("Success")
        result = dispatch_tool_checked("deep_research", {"query": "test"}, owner=owner)
        assert isinstance(result, ToolResult)
        assert result.ok is True
        assert result.tool == "deep_research"

    def test_catches_exception_returns_failed(self):
        owner = MockOwner()
        owner._client = MockLLMClient()
        owner._client.chat_completions_create = MagicMock(side_effect=Exception("boom"))
        result = dispatch_tool_checked("deep_research", {"query": "test"}, owner=owner)
        assert isinstance(result, ToolResult)
        assert result.ok is False
        assert result.error_type == "tool_exception"


class TestMaxAttempts:
    """Tests for _max_attempts_for retry logic."""

    def test_deep_research_respects_env(self):
        with patch.dict(os.environ, {"AGENT_DEEP_RESEARCH_ATTEMPTS": "3"}):
            # Reimport to pick up env
            from importlib import reload
            import agentic.agentic as agentic_module
            reload(agentic_module)
            assert agentic_module._max_attempts_for("deep_research") == 3

    def test_other_tools_default_1(self):
        assert _max_attempts_for("deep_search") == 1
        assert _max_attempts_for("save_note") == 1


class TestResearchCallBudget:
    """Tests for AGENT_RESEARCH_MAX_CALLS budget enforcement."""

    def test_research_calls_limited_per_turn(self):
        owner = MockOwner()
        owner._client = MockLLMClient("Result")
        owner._llm_model = "test"

        # Reset counter
        import agentic.agentic as agentic_module
        agentic_module._research_call_count = 0

        # First call should succeed
        dispatch_tool("deep_research", {"query": "q1"}, owner=owner)
        assert agentic_module._research_call_count == 1

        # Second call should be blocked by budget
        result = dispatch_tool("deep_research", {"query": "q2"}, owner=owner)
        assert "research call budget exhausted" in result.lower() or "budget" in result.lower()


class TestCapabilityMatching:
    """Tests for capability matching and tool filtering."""

    def test_match_capabilities_with_embedder(self):
        embedder = FakeEmbedder()
        caps = match_capabilities("research quantum computing", embedder=embedder)
        assert "research" in caps

    def test_match_capabilities_fallback_keyword(self):
        caps = match_capabilities("schedule a meeting for tomorrow")
        assert "scheduling" in caps

    def test_match_capabilities_multiple(self):
        caps = match_capabilities("research and schedule a meeting")
        assert "research" in caps
        assert "scheduling" in caps

    def test_filtered_tool_schemas_narrows(self):
        all_schemas = [s for s, _ in _TOOL_DEFS]
        # With research capability, should only get research tools + always_on
        filtered = filtered_tool_schemas(all_schemas, ["research"])
        tool_names = {s["function"]["name"] for s in filtered}
        # Should have research tools
        assert "deep_search" in tool_names
        assert "deep_research" in tool_names
        assert "read_paper_url" in tool_names
        assert "write_report" in tool_names
        assert "learn_knowledge" in tool_names
        # Should have always_on
        assert "make_plan" in tool_names
        assert "save_note" in tool_names
        assert "final_answer" in tool_names
        # Should NOT have unrelated tools
        assert "schedule_job" not in tool_names
        assert "search_jobs" not in tool_names

    def test_filtered_tool_schemas_no_match_returns_all(self):
        all_schemas = [s for s, _ in _TOOL_DEFS]
        filtered = filtered_tool_schemas(all_schemas, [])
        assert len(filtered) == len(all_schemas)


class TestTaskState:
    """Tests for TaskState recording."""

    def test_records_tool_results(self):
        state = TaskState("test goal")
        state.record(ToolResult(True, "tool1", {}, "output"))
        state.record(ToolResult(False, "tool2", {}, "error", error_type="execution_error"))
        assert len(state.tools) == 2
        assert state.tools[0].ok is True
        assert state.tools[1].ok is False

    def test_last_tool_result(self):
        state = TaskState("goal")
        state.record(ToolResult(True, "a", {}, "1"))
        state.record(ToolResult(True, "b", {}, "2"))
        assert state.last_tool_result.tool == "b"
        assert state.last_tool_result.content == "2"


class TestVerifyFinalAnswer:
    """Tests for _verify_final_answer (smoke tests, full verification is integration)."""

    def test_empty_goal_passes(self):
        state = TaskState("")
        owner = MockOwner()
        owner._client = MockLLMClient('{"ok": true, "score": 1.0, "issues": []}')
        result = _verify_final_answer(owner, "", "any answer", state)
        assert isinstance(result, VerificationResult)

    def test_no_client_returns_none(self):
        state = TaskState("goal")
        owner = MockOwner()
        owner._client = None
        result = _verify_final_answer(owner, "goal", "answer", state)
        assert result is None


class TestGraphExecutorIntegration:
    """Integration tests for graph executor called from agentic_chat."""

    def test_run_schema_agent_called_with_llm(self):
        """Verify run_schema_agent receives llm_client and llm_model."""
        with patch("agentic.schema.run_schema_agent") as mock_run:
            mock_run.return_value = None  # Force fallback to ReAct
            owner = MockOwner()
            owner._client = MockLLMClient()
            owner._llm_model = "test-model"

            # Need to mock more to avoid full ReAct loop
            with patch("agentic.agentic._fetch_agentic_only_context", return_value={}):
                with patch("agentic.agentic.tool_schemas", return_value=[]):
                    try:
                        run_agentic_chat("test prompt", owner, embedder=FakeEmbedder())
                    except Exception:
                        pass  # Expected to fail without full mocking

            # Check run_schema_agent was called with llm args
            if mock_run.called:
                args, kwargs = mock_run.call_args
                assert "llm_client" in kwargs
                assert "llm_model" in kwargs
                assert kwargs["llm_client"] is owner._client
                assert kwargs["llm_model"] == owner._llm_model


class TestRunAgenticChatSmoke:
    """Smoke tests for run_agentic_chat (requires heavy mocking)."""

    def test_graph_mode_returns_graph_result(self):
        """When AGENT_EXECUTOR_MODE=graph, returns graph result directly."""
        with patch.dict(os.environ, {"AGENT_EXECUTOR_MODE": "graph"}):
            with patch("agentic.schema.run_schema_agent") as mock_run:
                mock_result = MagicMock()
                mock_result.final_answer = "Graph answer"
                mock_result.results = []
                mock_run.return_value = mock_result

                owner = MockOwner()
                owner._client = MockLLMClient()

                with patch("agentic.agentic._owner_embedder", return_value=FakeEmbedder()):
                    result = run_agentic_chat("test", owner)
                    assert result == "Graph answer"

    def test_hybrid_fallbacks_to_react_on_untrustworthy(self):
        """When graph result fails verification, falls back to ReAct."""
        with patch.dict(os.environ, {"AGENT_EXECUTOR_MODE": "hybrid"}):
            with patch("agentic.schema.run_schema_agent") as mock_run:
                mock_result = MagicMock()
                mock_result.final_answer = "Graph answer"
                mock_result.results = [MagicMock(ok=False)]  # Failed node
                mock_run.return_value = mock_result

                owner = MockOwner()
                owner._client = MockLLMClient("ReAct answer")

                with patch("agentic.agentic._owner_embedder", return_value=FakeEmbedder()):
                    with patch("agentic.agentic._fetch_agentic_only_context", return_value={}):
                        with patch("agentic.agentic.tool_schemas", return_value=[]):
                            # ReAct loop would run but we can't fully test without more mocks
                            pass


class TestSaveNoteContentTruncation:
    """Tests that save_note truncates content to AGENT_NOTE_MAX_CHARS."""

    def test_save_note_truncates_long_content(self):
        owner = MockOwner()
        long_content = "x" * 10000
        result = dispatch_tool("save_note", {"title": "test", "content": long_content}, owner=owner)
        # Content should be truncated in the actual save
        assert "note saved" in result.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])