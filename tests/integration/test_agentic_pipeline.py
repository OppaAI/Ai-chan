"""
tests/integration/test_agentic_pipeline.py

Integration tests for the full agentic pipeline:
- Graph executor → ReAct fallback → Memory/KB persistence
- Tool dispatch → Graph playbook selection → Synthesis → Report writing

Run: pytest tests/integration/test_agentic_pipeline.py -v
"""
from __future__ import annotations

import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

os.environ.setdefault("WORKSPACE_ROOT", "/tmp/aiko_test_workspace")

sys.path.insert(0, "/home/oppa-ai/jetson")
from system.config import load_config
load_config()

from agentic import schema
from agentic.agentic import run_agentic_chat
from agentic.toolkit.synthesize import synthesize_report, kb_search, learn_report
from agentic.toolkit.research import deep_search, deep_research
from agentic.toolkit.plan import save_note, create_checklist


class MockOwner:
    """Mock AikoThink owner for testing."""
    def __init__(self):
        self._client = None
        self._llm_model = "test-model"
        self._history = []
        self._history_lock = MagicMock()
        self._history_lock.__enter__ = MagicMock(return_value=None)
        self._history_lock.__exit__ = MagicMock(return_value=False)
        self.last_prompt_debug = {}
        self.last_usage = {}
        self._memorize = MagicMock()
        self._memorize._mem = MagicMock()
        self._memorize._mem._embedder = None

    def _emit(self, text, token_callback=None):
        pass

    def _store_async(self, *args, **kwargs):
        pass


class FakeEmbedder:
    """Deterministic embedder."""
    def embed_query(self, text: str, instruct: str = "") -> np.ndarray:
        h = hash(text) % 1000
        return np.array([float(h) / 1000.0] * 384, dtype=np.float32)


class MockLLMClient:
    """Mock LLM client for synthesis."""
    def __init__(self, response: str = "Synthesized answer"):
        self.response = response
        self.call_count = 0
        self.last_messages = None

    @property
    def chat(self):
        mock_chat = MagicMock()
        mock_chat.completions = MagicMock()
        mock_chat.completions.create = self._create
        return mock_chat

    def _create(self, model, messages, **kwargs):
        self.call_count += 1
        self.last_messages = messages
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content=self.response))]
        return mock_resp


class TestGraphExecutorIntegration:
    """Integration tests for graph executor with real tool map."""

    def setup_method(self):
        """Clear tool map cache."""
        schema._TOOL_MAP_CACHE = None

    def test_research_playbook_end_to_end(self):
        """research_and_report playbook executes all nodes."""
        owner = MockOwner()
        owner._client = MockLLMClient("Research report synthesized")
        owner._memorize._mem._embedder = FakeEmbedder()

        with patch("agentic.agentic._owner_embedder", return_value=FakeEmbedder()):
            with patch("agentic.agentic._fetch_agentic_only_context", return_value={}):
                with patch("agentic.toolkit.research.deep_research") as mock_deep:
                    mock_deep.return_value = "Deep research results with evidence"
                    with patch("agentic.toolkit.synthesize.kb_search") as mock_kb:
                        mock_kb.return_value = "KB context"
                        with patch("agentic.toolkit.reports.write_report") as mock_write:
                            mock_write.return_value = '{"ok": true, "path": "/tmp/report.md"}'
                            with patch("agentic.toolkit.synthesize.learn_report") as mock_learn:
                                mock_learn.return_value = "doc-123"

                                result = schema.run_schema_agent(
                                    "research quantum computing and write a report",
                                    cap_ids=["research"],
                                    embedder=FakeEmbedder(),
                                    llm_client=owner._client,
                                    llm_model=owner._llm_model,
                                )

        assert result is not None
        assert result.graph.id == "research_and_report"
        assert len(result.results) == 6  # web, kb, merge, draft, report, learn
        assert all(r.ok for r in result.results)

    def test_search_playbook_end_to_end(self):
        """search_kb_and_report playbook executes."""
        owner = MockOwner()
        owner._client = MockLLMClient("Search report synthesized")
        owner._memorize._mem._embedder = FakeEmbedder()

        with patch("agentic.agentic._owner_embedder", return_value=FakeEmbedder()):
            with patch("agentic.agentic._fetch_agentic_only_context", return_value={}):
                with patch("agentic.toolkit.research.deep_search") as mock_search:
                    mock_search.return_value = "Search snippets"
                    with patch("agentic.toolkit.synthesize.kb_search") as mock_kb:
                        mock_kb.return_value = "KB context"
                        with patch("agentic.toolkit.reports.write_report") as mock_write:
                            mock_write.return_value = '{"ok": true}'
                            with patch("agentic.toolkit.synthesize.learn_report") as mock_learn:
                                mock_learn.return_value = "doc-123"

                                result = schema.run_schema_agent(
                                    "search for quantum computing basics",
                                    cap_ids=["research"],
                                    embedder=FakeEmbedder(),
                                    llm_client=owner._client,
                                    llm_model=owner._llm_model,
                                )

        assert result is not None
        assert result.graph.id == "search_kb_and_report"

    def test_compare_playbook_parallel_web_nodes(self):
        """compare_and_report fans out to two parallel deep_research calls."""
        owner = MockOwner()
        owner._client = MockLLMClient("Comparison report")
        owner._memorize._mem._embedder = FakeEmbedder()

        with patch("agentic.agentic._owner_embedder", return_value=FakeEmbedder()):
            with patch("agentic.agentic._fetch_agentic_only_context", return_value={}):
                with patch("agentic.toolkit.research.deep_research") as mock_deep:
                    mock_deep.side_effect = ["Research A results", "Research B results"]
                    with patch("agentic.toolkit.synthesize.kb_search") as mock_kb:
                        mock_kb.return_value = "KB context"
                        with patch("agentic.toolkit.reports.write_report") as mock_write:
                            mock_write.return_value = '{"ok": true}'
                            with patch("agentic.toolkit.synthesize.learn_report") as mock_learn:
                                mock_learn.return_value = "doc-123"

                                result = schema.run_schema_agent(
                                    "compare JAX vs PyTorch for deep learning",
                                    cap_ids=["research"],
                                    embedder=FakeEmbedder(),
                                    llm_client=owner._client,
                                    llm_model=owner._llm_model,
                                )

        assert result is not None
        assert result.graph.id == "compare_and_report"
        # Should have called deep_research twice (parallel)
        assert mock_deep.call_count == 2

    def test_checklist_playbook(self):
        """checklist_and_save creates checklist and saves note."""
        with patch("agentic.agentic._owner_embedder", return_value=FakeEmbedder()):
            with patch("agentic.agentic._fetch_agentic_only_context", return_value={}):
                result = schema.run_schema_agent(
                    "make a checklist for testing and save it",
                                    cap_ids=[],
                                    embedder=FakeEmbedder(),
                )

        assert result is not None
        assert result.graph.id == "checklist_and_save"
        # Should have create_checklist and save_note
        tools = [r.tool for r in result.results]
        assert "create_checklist" in tools
        assert "save_note" in tools

    def test_simple_save_playbook(self):
        """simple_save_note saves the prompt as a note."""
        with patch("agentic.agentic._owner_embedder", return_value=FakeEmbedder()):
            with patch("agentic.agentic._fetch_agentic_only_context", return_value={}):
                result = schema.run_schema_agent(
                    "save this note: remember to buy milk",
                    cap_ids=[],
                    embedder=FakeEmbedder(),
                )

        assert result is not None
        assert result.graph.id == "simple_save_note"
        assert result.results[0].tool == "save_note"


class TestSynthesisIntegration:
    """Tests for synthesis tools integration."""

    def test_synthesize_report_with_llm(self):
        """synthesize_report calls LLM and returns formatted output."""
        client = MockLLMClient("Professional research report on quantum computing.")
        embedder = FakeEmbedder()

        evidence = "Web evidence about quantum computing. KB evidence about qubits."
        result = synthesize_report(evidence, "research quantum computing", client=client, model="m", embedder=embedder)

        assert client.call_count == 1
        assert "quantum computing" in result.lower()

    def test_synthesize_report_no_llm_fallback(self):
        """Without LLM, returns evidence with header."""
        embedder = FakeEmbedder()
        evidence = "Some evidence"
        result = synthesize_report(evidence, "prompt", client=None, model=None, embedder=embedder)

        assert "Aiko Research Report" in result
        assert evidence in result

    def test_kb_search_integration(self):
        """kb_search calls knowledge_context_for and strips XML."""
        with patch("agentic.toolkit.synthesize.knowledge_context_for") as mock_kb:
            mock_kb.return_value = """<knowledge_context>
<knowledge_chunk doc_id="1" title="Test" kind="ingested" source="test" score="0.9">Test content</knowledge_chunk>
</knowledge_context>"""
            result = kb_search("test query", embedder=FakeEmbedder())
            assert "Test content" in result
            assert "<knowledge_context>" not in result

    def test_learn_report_integration(self):
        """learn_report calls ingest_text."""
        with patch("agentic.toolkit.synthesize.ingest_text") as mock_ingest:
            mock_ingest.return_value = "doc-456"
            result = learn_report("Report Title", "Report content", embedder=FakeEmbedder())
            assert result == "doc-456"
            mock_ingest.assert_called_once()


class TestToolDispatchIntegration:
    """Tests for dispatch_tool integration."""

    def test_deep_research_dispatch(self):
        from agentic.agentic import dispatch_tool

        owner = MockOwner()
        owner._client = MockLLMClient("Research done")
        owner._memorize._mem._embedder = FakeEmbedder()

        with patch("agentic.agentic._owner_embedder", return_value=FakeEmbedder()):
            result = dispatch_tool("deep_research", {"query": "test"}, owner=owner})
            assert "Deep research" in result

    def test_deep_search_dispatch(self):
        from agentic.agentic import dispatch_tool

        owner = MockOwner()
        owner._memorize._mem._embedder = FakeEmbedder()

        with patch("agentic.agentic._owner_embedder", return_value=FakeEmbedder()):
            result = dispatch_tool("deep_search", {"query": "test"}, owner)
            assert "Web search results" in result

    def test_write_report_dispatch(self):
        from agentic.agentic import dispatch_tool

        result = dispatch_tool("write_report", {"title": "Test", "content": "Report body"})
        assert "report written" in result.lower()

    def test_learn_knowledge_dispatch(self):
        from agentic.agentic import dispatch_tool

        owner = MockOwner()
        owner._memorize._mem._embedder = FakeEmbedder()

        with patch("agentic.agentic._owner_embedder", return_value=FakeEmbedder()):
            with patch("agentic.agentic.ingest_knowledge_text", return_value="doc-789"):
                result = dispatch_tool("learn_knowledge", {"title": "Test", "text": "Content"}, owner)
                assert "doc-789" in result

    def test_run_playbook_dispatch(self):
        from agentic.agentic import dispatch_tool
        import json

        owner = MockOwner()
        owner._client = MockLLMClient()
        owner._memorize._mem._embedder = FakeEmbedder()

        with patch("agentic.agentic._owner_embedder", return_value=FakeEmbedder()):
            with patch("agentic.schema.run_playbook_json") as mock_run:
                mock_run.return_value = json.dumps({"ok": True, "graph_id": "test"})
                result = dispatch_tool("run_playbook", {"task": "test task"}, owner)
                assert "ok" in result


class TestCapabilityRoutingIntegration:
    """Tests that capability routing works with new tools."""

    def test_research_capability_includes_new_tools(self):
        from agentic.capability import filtered_tool_schemas

        schemas = [
            {"function": {"name": "deep_research"}},
            {"function": {"name": "synthesize_report"}},
            {"function": {"name": "write_report"}},
            {"function": {"name": "learn_report"}},
            {"function": {"name": "kb_search"}},
            {"function": {"name": "make_plan"}},  # always on
        ]

        filtered = filtered_tool_schemas(schemas, ["research"])
        names = {s["function"]["name"] for s in filtered}

        assert "deep_research" in names
        assert "synthesize_report" in names
        assert "write_report" in names
        assert "learn_report" in names
        assert "kb_search" in names
        assert "make_plan" in names  # always on

    def test_scheduling_capability_excludes_research(self):
        from agentic.capability import filtered_tool_schemas

        schemas = [
            {"function": {"name": "deep_research"}},
            {"function": {"name": "schedule_job"}},
            {"function": {"name": "make_plan"}},
        ]

        filtered = filtered_tool_schemas(schemas, ["scheduling"])
        names = {s["function"]["name"] for s in filtered}

        assert "schedule_job" in names
        assert "deep_research" not in names
        assert "make_plan" in names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])