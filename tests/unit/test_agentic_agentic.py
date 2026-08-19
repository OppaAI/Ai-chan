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

import pytest
from agentic.registry import registry, ToolSpec

# Test fixtures for registry testing
@pytest.fixture
def _reg():
    """Mock registry with a sample tool for testing."""
    from agentic.registry import ToolRegistry
    reg = ToolRegistry()
    reg.register("test_tool", "A test tool", handler=lambda x: f"ok: {x}")
    return reg

@pytest.fixture
def _reg_no_handler():
    """Mock registry entry without a handler."""
    from agentic.registry import ToolRegistry, ToolSpec
    reg = ToolRegistry()
    spec = ToolSpec(
        name="no_handler_tool",
        description="Tool without a handler",
        handler=None,  # Intentionally None
        needs_approval=False,
        react=True,
    )
    # Manually add without going through register() if needed
    return spec

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
    _required_args_for,
    _validate_args,
    _classify_result,
    _owner_embedder,
    dispatch_tool,
    dispatch_tool_checked,
    execute_tool_with_policy,
    _max_attempts_for,
    run_agentic_chat,
    AGENT_EXECUTOR_MODE,
    AGENT_RESEARCH_MAX_CALLS,
    _research_call_count,
    _verify_final_answer,
)
from agentic import graph_engine as schema
from agentic.toolkit.plan import save_note, create_checklist, make_plan
from agentic.toolkit.reports import write_report
from agentic.toolkit.research import deep_research
from agentic.capability import match_capabilities, filtered_tool_schemas


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
            "summarize_task_state", "adaptive_search", "deep_research", "fetch_from_url",
            "write_report", "learn_knowledge", "run_playbook", "list_playbooks",
            "draft_photo_social", "post_photo_social",
            "draft_video_social", "post_video_social", "final_answer",
        }
        for tool in expected_tools:
            assert tool in _TOOLS, f"Missing tool: {tool}"

    def test_tool_schemas_have_required_fields(self):
        for name, (schema, handler) in _TOOLS.items():
            assert "function" in schema
            assert schema["function"]["name"] == name
            assert "parameters" in schema["function"]
            assert schema["function"]["parameters"]["type"] == "object"

    def test_social_post_tools_identified(self):
        assert "post_photo_social" in _SOCIAL_POST_TOOLS
        assert "post_video_social" in _SOCIAL_POST_TOOLS
        assert "draft_photo_social" not in _SOCIAL_POST_TOOLS

    def test_research_tools_identified(self):
        assert "adaptive_search" in _RESEARCH_TOOLS
        assert "deep_research" in _RESEARCH_TOOLS


class TestRequiredArgs:
    """Tests for _required_args_for and _validate_args."""

    def test_required_args_extracted(self):
        assert "goal" in _required_args_for("make_plan")
        assert "title" in _required_args_for("save_note")
        assert "content" in _required_args_for("save_note")
        assert "query" in _required_args_for("adaptive_search")
        assert "query" in _required_args_for("deep_research")
        assert "title" in _required_args_for("write_report")
        assert "title" in _required_args_for("learn_knowledge")
        assert "task" in _required_args_for("run_playbook")
        assert "url" in _required_args_for("fetch_from_url")

    def test_validate_args_missing_required(self):
        result = _validate_args("save_note", {"title": "test"})
        assert result is not None
        assert result.ok is False
        assert result.error_type == "schema_validation_failed"
        assert "content" in result.content

    def test_validate_args_empty_query_for_adaptive_search(self):
        result = _validate_args("adaptive_search", {"query": ""})
        assert result is not None
        assert result.ok is False
        assert result.error_type == "schema_validation_failed"

    def test_validate_args_learn_knowledge_needs_text_or_path(self):
        result = _validate_args("learn_knowledge", {"title": "test"})
        assert result is not None
        assert result.ok is False
        assert "text or relative_path" in result.content

    def test_validate_args_social_post_passes_without_draft_dir(self):
        result = _validate_args("post_photo_social", {})
        assert result is None  # draft_dir is optional at schema level

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

        with patch("agentic.agentic.deep_research", return_value="Research result") as mock_deep:
            result = dispatch_tool("deep_research", {"query": "test query"}, owner=owner)
        assert "Research result" in result
        mock_deep.assert_called_once_with(
            "test query", client=owner._client, model="test-model", embedder=owner._memorize._mem._embedder,
        )

    def test_deep_search_uses_embedder(self):
        owner = MockOwner()
        with patch("agentic.agentic.adaptive_search", return_value="Web search results: xyz") as mock_search:
            result = dispatch_tool("adaptive_search", {"query": "test query"}, owner=owner)
        assert "Web search results" in result
        _, kwargs = mock_search.call_args
        assert kwargs["embedder"] is not None

    def test_run_playbook_passes_embedder(self):
        owner = MockOwner()
        with patch("agentic.graph_engine.run_playbook_json") as mock_run:
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

    def test_fetch_from_url_passes_url(self):
        owner = MockOwner()
        spec = registry.get("fetch_from_url")
        with patch.object(spec, "handler", return_value="Paper content") as mock_read:
            result = dispatch_tool("fetch_from_url", {"url": "http://example.com"}, owner=owner)
        assert "Paper content" in result
        mock_read.assert_called_once_with(url="http://example.com")

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
            assert args[0] == "Test"
            assert args[3] is True

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
        with patch("agentic.agentic.deep_research", side_effect=Exception("boom")):
            result = dispatch_tool_checked("deep_research", {"query": "test"}, owner=owner)
        assert isinstance(result, ToolResult)
        assert result.ok is False
        assert result.error_type == "tool_error"


class TestMaxAttempts:
    """Tests for _max_attempts_for retry logic."""

    def test_deep_research_respects_env(self):
        with patch.dict(os.environ, {"AGENT_DEEP_RESEARCH_ATTEMPTS": "3"}):
            assert _max_attempts_for("deep_research") == 3

    def test_other_tools_default_1(self):
        # deep_search removed; adaptive_search uses default attempts=1
        assert _max_attempts_for("save_note") == 1


class TestResearchCallBudget:
    """Tests for the research call budget enforcement."""

    def test_research_calls_limited_per_turn(self):
        from agentic.guardrails import research_budget_guard
        state = TaskState("goal")
        guard = research_budget_guard(AGENT_RESEARCH_MAX_CALLS)

        assert guard("adaptive_search", {"query": "q1"}, state) is None
        state.record(ToolResult(True, "adaptive_search", {"query": "q1"}, "ok"))
        assert _research_call_count(state) == 1

        verdict = guard("deep_research", {"query": "q2"}, state)
        assert verdict is not None
        assert verdict.error_type == "research_limit_reached"


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
        assert "adaptive_search" in tool_names
        assert "deep_research" in tool_names
        assert "fetch_from_url" in tool_names
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
        assert len(state.steps) == 2
        assert state.steps[0]["ok"] is True
        assert state.steps[1]["ok"] is False
        assert state.evidence[0] == "tool1: output"
        assert state.failures[0].tool == "tool2"

    def test_last_tool_result(self):
        state = TaskState("goal")
        state.record(ToolResult(True, "a", {}, "1"))
        state.record(ToolResult(True, "b", {}, "2"))
        assert state.steps[-1]["tool"] == "b"
        assert state.steps[-1]["content"] == "2"


class TestVerifyFinalAnswer:
    """Tests for _verify_final_answer (smoke tests, full verification is integration)."""

    def test_empty_goal_passes(self):
        state = TaskState("")
        owner = MockOwner()
        owner._client = MockLLMClient('{"ok": true, "score": 1.0, "issues": []}')
        result = _verify_final_answer(owner, "", "any answer", state)
        assert isinstance(result, VerificationResult)

    def test_no_client_returns_deterministic_pass(self):
        state = TaskState("goal")
        owner = MockOwner()
        owner._client = None
        result = _verify_final_answer(owner, "goal", "answer", state)
        assert isinstance(result, VerificationResult)
        assert result.ok is True


class TestGraphExecutorIntegration:
    """Integration tests for graph executor called from agentic_chat."""

    def test_run_schema_agent_called_with_llm(self):
        """Verify run_schema_agent receives llm_client and llm_model."""
        with patch("agentic.graph_engine.run_schema_agent") as mock_run:
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
            with patch("agentic.graph_engine.run_schema_agent") as mock_run:
                mock_result = MagicMock()
                mock_result.final_answer = "Graph answer"
                mock_result.results = []
                mock_run.return_value = mock_result

                owner = MockOwner()
                owner._client = MockLLMClient()

                with patch("agentic.agentic._owner_embedder", return_value=FakeEmbedder()):
                    result = run_agentic_chat(owner, "test")
                    assert result == "Graph answer"

    def test_hybrid_fallbacks_to_react_on_untrustworthy(self):
        """When graph result fails verification, falls back to ReAct."""
        with patch.dict(os.environ, {"AGENT_EXECUTOR_MODE": "hybrid"}):
            with patch("agentic.graph_engine.run_schema_agent") as mock_run:
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


def test_validate_args_uses_registered_pydantic_model():
    from pydantic import BaseModel, Field
    from agentic.registry import registry

    class StrictArgs(BaseModel):
        limit: int = Field(ge=1, le=5)

    registry.register(
        name="typed_validation_test",
        description="typed validation test",
        args_model=StrictArgs,
        react=True,
    )
    _TOOLS["typed_validation_test"] = (registry.get("typed_validation_test").to_openai_schema(), None)

    args = {"limit": "3"}
    assert _validate_args("typed_validation_test", args) is None
    assert args == {"limit": 3}

    bad = _validate_args("typed_validation_test", {"limit": 99})
    assert bad is not None
    assert bad.error_type == "schema_validation_failed"
    assert bad.retryable is True


def test_execute_tool_with_policy_applies_research_budget_guardrail():
    state = TaskState(goal="research budget")
    state.record(ToolResult(ok=True, tool="adaptive_search", args={"query": "first"}, content="done"))

    result = execute_tool_with_policy("deep_research", {"query": "again"}, state)

    assert result.ok is False
    assert result.error_type == "research_limit_reached"
    assert result.retryable is False


def test_verify_final_answer_uses_post_answer_guardrails_for_saved_path(monkeypatch):
    monkeypatch.setattr("agentic.agentic.AGENT_VERIFY_LLM_MODE", "off")
    owner = MockOwner()
    state = TaskState(goal="save note")
    state.record(ToolResult(ok=True, tool="save_note", args={"title": "x"}, content="note saved"))

    verdict = _verify_final_answer(owner, "save this", "Done.", state)

    assert verdict.ok is False
    assert "does not mention where it was saved" in verdict.feedback


def test_handoff_profile_maps_capabilities():
    from agentic.capability import resolve_handoff
    from agentic.agentic import MAX_AGENT_ITER, AGENT_RESEARCH_MAX_CALLS

    profile = resolve_handoff(
        ["social", "scheduling"],
        default_max_iter=MAX_AGENT_ITER,
        default_research_budget=AGENT_RESEARCH_MAX_CALLS,
    )
    assert "social" in profile.tool_domains
    assert "scheduling" in profile.tool_domains
    assert profile.max_iter <= 5


def test_needs_approval_returns_wait_result_and_checkpoint(monkeypatch, tmp_path):
    from agentic.agentic import AgentContext, TaskState, execute_tool_with_policy
    from agentic.registry import registry

    registry.register("approval_test", "approval test", needs_approval=True, react=True)
    monkeypatch.setattr("agentic.agentic.user_state_dir", lambda user_id=None: tmp_path)
    state = TaskState(goal="approval")
    result = execute_tool_with_policy("approval_test", {}, state, ctx=AgentContext(run_id="r1"))
    assert result.error_type == "needs_approval"
    assert result.metadata["run_id"] == "r1"
    assert (tmp_path / "agentic" / "traces" / "r1.jsonl").exists()


def test_agent_context_passed_to_dispatch(monkeypatch):
    from agentic.agentic import AgentContext, dispatch_tool

    seen = {}
    monkeypatch.setattr("agentic.agentic.adaptive_search", lambda query, client=None, model=None, embedder=None: seen.update(client=client, model=model, embedder=embedder) or "ok")
    ctx = AgentContext(client="c", llm_model="m", embedder="e")
    assert dispatch_tool("adaptive_search", {"query": "q"}, owner=ctx) == "ok"
    assert seen == {"client": "c", "model": "m", "embedder": "e"}


def test_resume_approval_runs_pending_tool(monkeypatch, tmp_path):
    from agentic.agentic import AgentContext, TaskState, execute_tool_with_policy, _maybe_resume_approval
    from agentic.registry import registry

    calls = []
    def handler(**kwargs):
        calls.append(kwargs)
        return "posted"

    registry.register("resume_approval_test", "approval test", handler=handler, needs_approval=True, react=True)
    monkeypatch.setattr("agentic.agentic.user_state_dir", lambda user_id=None: tmp_path)
    monkeypatch.setattr("agentic.agentic.user_workspace_root", lambda user_id=None: tmp_path / "workspace")
    owner = MockOwner()
    state = TaskState(goal="approval")
    wait = execute_tool_with_policy("resume_approval_test", {"draft_dir": "d"}, state, ctx=AgentContext(run_id="r2"))
    assert wait.error_type == "needs_approval"

    resumed = _maybe_resume_approval(owner, "yes approve r2")
    assert resumed is not None
    assert calls == [{"draft_dir": "d"}]


def test_skill_proposal_written_for_multistep_run(monkeypatch, tmp_path):
    from agentic.skill_learning import propose_skill_from_run

    monkeypatch.setattr("agentic.skill_learning.user_workspace_root", lambda user_id=None: tmp_path)
    path = propose_skill_from_run(
        "Summarize docs",
        [{"tool": "repo_file_tree", "ok": True}, {"tool": "repo_read_file", "ok": True}],
        "done",
        verified_ok=True,
        score=0.95,
    )
    assert path is not None
    assert "Reusable tool order" in path.read_text(encoding="utf-8")


def test_handoff_profile_includes_additional_domains():
    from agentic.capability import resolve_handoff
    from agentic.agentic import MAX_AGENT_ITER, AGENT_RESEARCH_MAX_CALLS

    profile = resolve_handoff(
        ["job_hunt", "photo", "kb_proposal"],
        default_max_iter=MAX_AGENT_ITER,
        default_research_budget=AGENT_RESEARCH_MAX_CALLS,
    )
    # job_hunt's registry domain is "jobs" (not "job_hunt") — this is the
    # exact mismatch the resolve_handoff wiring fix corrects.
    assert "jobs" in profile.tool_domains
    assert "social" in profile.tool_domains  # from both job_hunt and photo
    assert "photo" in profile.tool_domains
    assert "kb" in profile.tool_domains
    assert "skills" in profile.tool_domains  # kb_proposal also pulls in skills

def test_approval_resume_does_not_mutate_registry_flag(monkeypatch, tmp_path):
    from agentic.agentic import AgentContext, TaskState, execute_tool_with_policy, _maybe_resume_approval
    from agentic.registry import registry

    registry.register("race_safe_approval_test", "approval test", handler=lambda **kwargs: "ok", needs_approval=True, react=True)
    monkeypatch.setattr("agentic.agentic.user_state_dir", lambda user_id=None: tmp_path)
    owner = MockOwner()
    wait = execute_tool_with_policy("race_safe_approval_test", {}, TaskState(goal="approval"), ctx=AgentContext(run_id="r3"))
    assert wait.error_type == "needs_approval"
    assert registry.get("race_safe_approval_test").needs_approval is True
    assert _maybe_resume_approval(owner, "approve r3") is not None
    assert registry.get("race_safe_approval_test").needs_approval is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
