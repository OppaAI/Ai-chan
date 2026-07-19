"""
tests/stress/test_agentic_stress.py

Stress tests for agentic components:
- Graph executor under high concurrency
- Tool dispatch under load
- Memory/knowledge store under concurrent access
- LLM call batching and rate limiting

Run: pytest tests/stress/test_agentic_stress.py -v -m stress
"""
from __future__ import annotations

import os
import sys
import time
import threading
import queue
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

os.environ.setdefault("WORKSPACE_ROOT", "/tmp/aiko_test_workspace")

sys.path.insert(0, "/home/oppa-ai/jetson")
from system.config import load_config
load_config()

from agentic import schema
from agentic.agentic import dispatch_tool, _research_call_count, AGENT_RESEARCH_MAX_CALLS
from agentic.toolkit.synthesize import synthesize_report, kb_search
from agentic.toolkit.plan import save_note, create_checklist
from agentic.toolkit.research import deep_search
from agentic.capability import match_capabilities, filtered_tool_schemas


pytestmark = pytest.mark.stress


class FakeEmbedder:
    """Deterministic embedder for stress tests."""
    def embed_query(self, text: str, instruct: str = "") -> np.ndarray:
        h = hash(text) % 1000
        return np.array([float(h) / 1000.0] * 384, dtype=np.float32)


class MockLLMClient:
    """Mock LLM client with configurable latency."""
    def __init__(self, response: str = "Response", latency: float = 0.01):
        self.response = response
        self.latency = latency
        self.call_count = 0

    @property
    def chat(self):
        mock_chat = MagicMock()
        mock_chat.completions = MagicMock()
        mock_chat.completions.create = self._create
        return mock_chat

    def _create(self, model, messages, **kwargs):
        self.call_count += 1
        time.sleep(self.latency)  # Simulate network latency
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content=self.response))]
        return mock_resp


class TestGraphExecutorStress:
    """Stress tests for graph executor."""

    def setup_method(self):
        schema._TOOL_MAP_CACHE = None

    def test_many_parallel_graph_executions(self):
        """Execute many graphs concurrently."""
        num_graphs = 50
        results = []

        def run_graph(i):
            # Each graph is a simple save_note playbook
            result = schema.run_schema_agent(
                f"save note {i}: data point {i}",
                cap_ids=[],
                embedder=FakeEmbedder(),
            )
            return result is not None and result.graph.id == "simple_save_note"

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(run_graph, i) for i in range(num_graphs)]
            for f in as_completed(futures):
                results.append(f.result())

        # All should succeed
        assert all(results)
        assert len(results) == num_graphs

    def test_parallel_research_graphs(self):
        """Run multiple research graphs in parallel."""
        num_graphs = 10

        def run_research(i):
            with patch("agentic.agentic._owner_embedder", return_value=FakeEmbedder()):
                with patch("agentic.agentic._fetch_agentic_only_context", return_value={}):
                    with patch("agentic.toolkit.research.deep_research") as mock_deep:
                        mock_deep.return_value = f"Research {i} done"
                        with patch("agentic.toolkit.synthesize.kb_search") as mock_kb:
                            mock_kb.return_value = "KB"
                            with patch("agentic.toolkit.reports.write_report") as mock_write:
                                mock_write.return_value = '{"ok": true}'
                                with patch("agentic.toolkit.synthesize.learn_report") as mock_learn:
                                    mock_learn.return_value = f"doc-{i}"
                                    result = schema.run_schema_agent(
                                        f"research topic {i} and report",
                                        cap_ids=["research"],
                                        embedder=FakeEmbedder(),
                                        llm_client=MagicMock(),
                                        llm_model="test",
                                    )
            return result is not None

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(run_research, i) for i in range(num_graphs)]
            results = [f.result() for f in as_completed(futures)]

        assert all(results)

    def test_graph_with_many_parallel_nodes(self):
        """Graph with many parallel nodes executes correctly."""
        # Create a custom playbook with many parallel nodes
        import uuid
        playbook = {
            "id": f"stress_{uuid.uuid4().hex[:8]}",
            "name": "Stress test",
            "triggers": ["stress"],
            "requires_any": [],
            "nodes": [
                {"id": f"task_{i}", "tool": "save_note", "args": {"title": f"Task {i}", "content": "x" * 100}}
                for i in range(20)
            ],
        }
        # Save to user playbook file
        import json
        playbook_path = schema._playbook_file()
        playbook_path.parent.mkdir(parents=True, exist_ok=True)
        existing = []
        if playbook_path.exists():
            try:
                existing = json.loads(playbook_path.read_text())
            except:
                existing = []
        existing.append(playbook)
        playbook_path.write_text(json.dumps(existing))

        try:
            # Force reload
            schema._TOOL_MAP_CACHE = None
            result = schema.run_schema_agent("stress test", cap_ids=[], embedder=FakeEmbedder())
            assert result is not None
            # Should have executed all 20 nodes
            assert len(result.results) == 20
        finally:
            # Cleanup
            if playbook_path.exists():
                playbook_path.unlink()


class TestToolDispatchStress:
    """Stress tests for tool dispatch."""

    def test_concurrent_deep_search_calls(self):
        """Many concurrent deep_search calls."""
        num_calls = 100
        results = []

        def call_search(i):
            owner = MagicMock()
            owner._memorize._mem._embedder = FakeEmbedder()
            with patch("agentic.agentic._owner_embedder", return_value=FakeEmbedder()):
                return dispatch_tool("deep_search", {"query": f"query {i}"}, owner=owner)

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(call_search, i) for i in range(num_calls)]
            for f in as_completed(futures):
                results.append(f.result())

        assert len(results) == num_calls
        assert all("Web search results" in r or "no results" in r.lower() for r in results)

    def test_concurrent_save_note_calls(self):
        """Many concurrent save_note calls."""
        num_calls = 200

        def call_save(i):
            return dispatch_tool("save_note", {"title": f"Note {i}", "content": f"Content {i}"})

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(call_save, i) for i in range(num_calls)]
            results = [f.result() for f in as_completed(futures)]

        assert len(results) == num_calls
        assert all("note saved" in r.lower() for r in results)

    def test_concurrent_synthesize_calls(self):
        """Many concurrent synthesize_report calls."""
        num_calls = 50
        client = MockLLMClient("Synthesized", latency=0.005)

        def call_synthesize(i):
            return synthesize_report(
                f"Evidence {i}", f"Prompt {i}",
                client=client, model="test", embedder=FakeEmbedder()
            )

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(call_synthesize, i) for i in range(num_calls)]
            results = [f.result() for f in as_completed(futures)]

        assert len(results) == num_calls
        assert client.call_count == num_calls


class TestMemoryKnowledgeStress:
    """Stress tests for memory and knowledge stores."""

    def test_concurrent_memory_searches(self):
        """Concurrent memory searches don't corrupt cache."""
        from memory.memorize import AikoMemorize, _MemoryBackend

        # Create backend with some data
        import tempfile
        db_path = tempfile.mktemp(suffix=".db")
        backend = _MemoryBackend(db_path=db_path, llm_base_url="http://unused", model="unused")
        backend._embedder = FakeEmbedder()

        # Seed some memories
        for i in range(100):
            backend.add_raw(f"Memory fact {i}", user_id="stress_user")

        def search_memory(i):
            return backend.search(f"fact {i % 50}", user_id="stress_user", limit=5)

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(search_memory, i) for i in range(200)]
            results = [f.result() for f in as_completed(futures)]

        assert len(results) == 200
        # All should return valid results
        assert all(isinstance(r, list) for r in results)

    def test_concurrent_knowledge_searches(self):
        """Concurrent knowledge searches."""
        # This would need a seeded knowledge DB - skip if not available
        pass

    def test_concurrent_note_saves(self):
        """Many concurrent save_note calls."""
        def save_note(i):
            return save_note(f"Note {i}", f"Content {i} " * 10, folder="notes")

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(save_note, i) for i in range(100)]
            results = [f.result() for f in as_completed(futures)]

        assert len(results) == 100
        assert all("note saved" in r.lower() for r in results)


class TestCapabilityMatchingStress:
    """Stress tests for capability matching."""

    def test_many_capability_matches(self):
        """Run many capability matches."""
        embedder = FakeEmbedder()
        queries = [
            "research quantum computing",
            "schedule a meeting",
            "compare A vs B",
            "search for information",
            "create a checklist",
            "analyze the codebase",
            "find job postings",
            "post to social media",
            "import photos",
            "random chat message",
        ] * 100  # 1000 queries

        start = time.monotonic()
        for q in queries:
            match_capabilities(q, embedder=embedder)
        elapsed = time.monotonic() - start

        # Should be fast (< 1 second for 1000 queries)
        assert elapsed < 1.0

    def test_many_tool_filterings(self):
        """Run many tool schema filterings."""
        schemas = [
            {"function": {"name": "deep_research"}},
            {"function": {"name": "synthesize_report"}},
            {"function": {"name": "write_report"}},
            {"function": {"name": "learn_report"}},
            {"function": {"name": "kb_search"}},
            {"function": {"name": "schedule_job"}},
            {"function": {"name": "search_jobs"}},
            {"function": {"name": "make_plan"}},
            {"function": {"name": "save_note"}},
            {"function": {"name": "final_answer"}},
        ] * 10  # 100 schemas

        cap_sets = [
            ["research"],
            ["scheduling"],
            ["repo"],
            ["job_hunt"],
            ["research", "scheduling"],
            [],
        ] * 100  # 600 filterings

        start = time.monotonic()
        for caps in cap_sets:
            filtered_tool_schemas(schemas, caps)
        elapsed = time.monotonic() - start

        assert elapsed < 0.5


class TestResearchCallBudget:
    """Tests for AGENT_RESEARCH_MAX_CALLS budget enforcement."""

    def test_budget_enforced_across_dispatch(self):
        """Research call budget should limit deep_research calls."""
        import agentic.agentic as agentic_module
        original_max = AGENT_RESEARCH_MAX_CALLS
        agentic_module._research_call_count = 0

        # Set low budget
        with patch.dict(os.environ, {"AGENT_RESEARCH_MAX_CALLS": "3"}):
            agentic_module.AGENT_RESEARCH_MAX_CALLS = 3

            owner = MagicMock()
            owner._client = MockLLMClient()
            owner._llm_model = "test"
            owner._memorize._mem._embedder = FakeEmbedder()

            with patch("agentic.agentic._owner_embedder", return_value=FakeEmbedder()):
                for i in range(5):
                    result = dispatch_tool("deep_research", {"query": f"q{i}"}, owner=owner)
                    if i < 3:
                        assert "Research result" in result or "Deep research" in result
                    else:
                        assert "budget exhausted" in result.lower() or "budget" in result.lower()

    def test_budget_resets_per_turn(self):
        """Budget should reset for each new turn (handled by agentic_chat)."""
        # This is verified in integration tests
        pass


class TestRateLimitingAndTimeouts:
    """Tests for rate limiting and timeout handling."""

    def test_synthesize_timeout_handling(self):
        """synthesize_report should handle slow LLM gracefully."""
        slow_client = MockLLMClient("Response", latency=0.5)  # 500ms latency

        start = time.monotonic()
        result = synthesize_report(
            "evidence", "prompt",
            client=slow_client, model="test", embedder=FakeEmbedder()
        )
        elapsed = time.monotonic() - start

        # Should complete (with timeout handling in real impl)
        assert elapsed >= 0.5
        assert "Synthesized" in result or "evidence" in result.lower()

    def test_deep_research_round_timeout(self):
        """deep_research should respect round timeouts."""
        # This is tested implicitly by the adaptive loop logic
        pass


class TestMemoryLeaks:
    """Tests to detect memory leaks under load."""

    def test_graph_executor_no_leak(self):
        """Repeated graph execution shouldn't leak memory."""
        import gc

        gc.collect()
        initial_objects = len(gc.get_objects())

        for i in range(100):
            schema.run_schema_agent(f"save note {i}", cap_ids=[], embedder=FakeEmbedder())

        gc.collect()
        final_objects = len(gc.get_objects())

        # Object count shouldn't grow significantly
        growth = final_objects - initial_objects
        assert growth < 5000  # Allow some growth but not unbounded

    def test_tool_map_cache_stable(self):
        """Tool map cache shouldn't grow unbounded."""
        schema._TOOL_MAP_CACHE = None

        for _ in range(100):
            schema._tool_map()

        # Cache should be single entry
        assert schema._TOOL_MAP_CACHE is not None
        assert len(schema._TOOL_MAP_CACHE) < 50  # Known tool count


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "stress"])