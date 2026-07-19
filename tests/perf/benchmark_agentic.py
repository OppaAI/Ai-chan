"""
tests/perf/benchmark_agentic.py

Performance benchmarks for agentic components.
Run with: pytest tests/perf/benchmark_agentic.py -m perf --benchmark-only

Results saved to .benchmarks/ for regression tracking.
"""
from __future__ import annotations

import os
import sys
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

os.environ.setdefault("WORKSPACE_ROOT", "/tmp/aiko_test_workspace")

sys.path.insert(0, "/home/oppa-ai/jetson")
from system.config import load_config
load_config()

# ─── Import modules to benchmark ──────────────────────────────────────────────
from agentic import schema
from agentic.capability import match_capabilities, filtered_tool_schemas
from agentic.toolkit.synthesize import synthesize_report, kb_search, combine_evidence, condense_text
from agentic.toolkit.research import deep_search, condense_evidence
from agentic.toolkit.plan import save_note, create_checklist, make_plan
from agentic.toolkit.reports import write_report
from agentic.agentic import _validate_args, _classify_result, _owner_embedder
from memory.knowledge import search_knowledge, knowledge_context_for, ingest_text
from cognition.reason import batch_cosine_scores, keyword_overlap_score


class FakeEmbedder:
    """Deterministic embedder for benchmarks."""
    def __init__(self, dim=384):
        self.dim = dim
        self.call_count = 0

    def embed_query(self, text: str, instruct: str = "") -> np.ndarray:
        self.call_count += 1
        h = hash(text + instruct) % 1000
        return np.array([float(h) / 1000.0] * self.dim, dtype=np.float32)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        self.call_count += len(texts)
        return np.stack([self.embed_query(t) for t in texts])


class MockLLMClient:
    def __init__(self, latency_ms: float = 50):
        self.latency_ms = latency_ms
        self.call_count = 0

    @property
    def chat(self):
        mock_chat = MagicMock()
        mock_chat.completions = MagicMock()
        mock_chat.completions.create = self._create
        return mock_chat

    def _create(self, model, messages, **kwargs):
        self.call_count += 1
        time.sleep(self.latency_ms / 1000.0)
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content="Benchmark response"))]
        return mock_resp


# ─── Fixtures ─────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def embedder():
    return FakeEmbedder()


@pytest.fixture(scope="module")
def llm_client():
    return MockLLMClient(latency_ms=10)  # Fast mock


# ─── Schema/Graph Benchmarks ──────────────────────────────────────────────────
class TestSchemaPerformance:
    """Benchmarks for graph executor and playbook matching."""

    def test_plan_from_master_latency(self, benchmark, embedder):
        """Benchmark playbook selection + graph construction."""
        schema._TOOL_MAP_CACHE = None  # Clear cache for fair measurement

        def _run():
            return schema.plan_from_master(
                "research quantum computing and write a comprehensive report",
                cap_ids=["research"],
                embedder=embedder,
            )

        result = benchmark(_run)
        assert result is not None
        assert result.id == "research_and_report"

    def test_execute_graph_latency(self, benchmark, embedder, llm_client):
        """Benchmark full graph execution for research playbook."""
        graph = schema.plan_from_master(
            "research quantum computing and write a report",
            cap_ids=["research"],
            embedder=embedder,
        )

        def _run():
            return schema.execute_graph(graph, embedder=embedder, llm_client=llm_client, llm_model="test")

        result = benchmark(_run)
        assert isinstance(result, schema.GraphRunResult)
        assert len(result.results) == 6

    def test_playbook_matching_semantic(self, benchmark, embedder):
        """Benchmark semantic playbook matching."""
        def _run():
            return schema._score_plan(
                schema._default_playbooks()[0],  # research_and_report
                "I want a thorough research report on this topic with citations",
                cap_ids=["research"],
                embedder=embedder,
            )

        score = benchmark(_run)
        assert score > 0

    def test_playbook_matching_keyword(self, benchmark):
        """Benchmark keyword-only playbook matching (no embedder)."""
        def _run():
            return schema._score_plan(
                schema._default_playbooks()[0],
                "research this topic and write a report",
                cap_ids=["research"],
                embedder=None,
            )

        score = benchmark(_run)
        assert score > 0


# ─── Capability Routing Benchmarks ────────────────────────────────────────────
class TestCapabilityPerformance:
    """Benchmarks for capability matching and tool filtering."""

    def test_match_capabilities_with_embedder(self, benchmark, embedder):
        """Benchmark capability matching with embedder."""
        def _run():
            return match_capabilities(
                "research quantum computing and schedule a meeting",
                embedder=embedder,
                query_vector=embedder.embed_query("research quantum computing and schedule a meeting", instruct="Which capability/tool domain applies to this task?"),
            )

        caps = benchmark(_run)
        assert isinstance(caps, list)

    def test_match_capabilities_keyword_fallback(self, benchmark):
        """Benchmark keyword-only capability matching."""
        def _run():
            return match_capabilities("schedule a meeting for tomorrow")

        caps = benchmark(_run)
        assert "scheduling" in caps

    def test_filtered_tool_schemas(self, benchmark):
        """Benchmark tool schema filtering."""
        from agentic.agentic import _TOOL_DEFS
        all_schemas = [s for s, _ in _TOOL_DEFS]

        def _run():
            return filtered_tool_schemas(all_schemas, ["research", "scheduling"])

        filtered = benchmark(_run)
        assert len(filtered) > 0


# ─── Synthesis Benchmarks ────────────────────────────────────────────────────
class TestSynthesisPerformance:
    """Benchmarks for synthesis tools."""

    def test_synthesize_report_with_llm(self, benchmark, embedder, llm_client):
        """Benchmark LLM synthesis call."""
        evidence = "Web evidence about quantum computing. " * 50 + "KB evidence about qubits. " * 20

        def _run():
            return synthesize_report(evidence, "research quantum computing", client=llm_client, model="test", embedder=embedder)

        result = benchmark(_run)
        assert llm_client.call_count > 0

    def test_synthesize_report_no_llm(self, benchmark, embedder):
        """Benchmark fallback synthesis without LLM."""
        evidence = "Evidence " * 100

        def _run():
            return synthesize_report(evidence, "prompt", client=None, model=None, embedder=embedder)

        result = benchmark(_run)
        assert "Aiko Research Report" in result

    def test_combine_evidence(self, benchmark):
        """Benchmark evidence combination."""
        parts = [f"Evidence part {i}. " * 20 for i in range(10)]

        def _run():
            return combine_evidence(parts)

        result = benchmark(_run)
        assert len(result) > 0

    def test_condense_text(self, benchmark, embedder):
        """Benchmark semantic condensation."""
        long_text = "Source-1: " + "Quantum computing content. " * 200 + "\n---\nSource-2: " + "More content. " * 200

        def _run():
            return condense_text(long_text, "quantum computing", embedder, max_chars=2000)

        result = benchmark(_run)
        assert len(result) <= 2000


# ─── Research Benchmarks ──────────────────────────────────────────────────────
class TestResearchPerformance:
    """Benchmarks for research tools."""

    def test_deep_search_latency(self, benchmark, embedder):
        """Benchmark deep_search (snippet-only)."""
        with patch("agentic.toolkit.research._web_search_raw") as mock_search:
            mock_search.return_value = ([
                {"title": f"R{i}", "url": f"https://r{i}.com", "content": f"Content {i}"}
                for i in range(10)
            ], None)

            def _run():
                return deep_search("test query", embedder=embedder)

        result = benchmark(_run)
        assert "Web search results" in result

    def test_condense_evidence(self, benchmark, embedder):
        """Benchmark condense_evidence."""
        pages = [(f"https://{i}.com", "Page content about quantum. " * 50) for i in range(5)]

        def _run():
            return condense_evidence(pages, "quantum", embedder=embedder)

        result = benchmark(_run)
        assert "Condensed evidence" in result


# ─── Plan/Note Benchmarks ────────────────────────────────────────────────────
class TestPlanNotePerformance:
    """Benchmarks for planning and note tools."""

    def test_make_plan_latency(self, benchmark):
        """Benchmark make_plan."""
        def _run():
            return make_plan("Build a quantum computer", constraints="budget $1000", max_steps=8)

        result = benchmark(_run)
        assert "plan created" in result

    def test_create_checklist_latency(self, benchmark):
        """Benchmark create_checklist."""
        items = [f"Task {i}" for i in range(20)]

        def _run():
            return create_checklist("Big Checklist", items)

        result = benchmark(_run)
        assert "Big Checklist" in result

    def test_save_note_latency(self, benchmark, tmp_path):
        """Benchmark save_note."""
        import os
        os.environ["WORKSPACE_ROOT"] = str(tmp_path)

        def _run():
            return save_note("Test Note", "Note content " * 10)

        result = benchmark(_run)
        assert "note saved" in result


# ─── Capability/Reasoning Benchmarks ─────────────────────────────────────────
class TestReasoningPerformance:
    """Benchmarks for reasoning utilities."""

    def test_batch_cosine_scores(self, benchmark, embedder):
        """Benchmark batch cosine scoring."""
        query_vec = embedder.embed_query("test query")
        chunk_vecs = np.stack([embedder.embed_query(f"chunk {i}") for i in range(100)])

        def _run():
            return batch_cosine_scores(query_vec, chunk_vecs)

        scores = benchmark(_run)
        assert scores.shape == (100,)

    def test_keyword_overlap_score(self, benchmark):
        """Benchmark keyword overlap scoring."""
        def _run():
            return keyword_overlap_score("quantum computing qubits", "quantum computing uses qubits for computation")

        score = benchmark(_run)
        assert score > 0


# ─── Memory/Knowledge Benchmarks ─────────────────────────────────────────────
class TestMemoryPerformance:
    """Benchmarks for memory/knowledge operations."""

    def test_search_knowledge_latency(self, benchmark, embedder, tmp_path):
        """Benchmark search_knowledge with seeded DB."""
        db_path = tmp_path / "bench.db"
        conn = _connect(str(db_path))

        # Seed 1000 docs
        from memory.knowledge import _knn, _fts, KNOWLEDGE_KNN_LIMIT, KNOWLEDGE_FTS_LIMIT
        import sqlite_vec
        now = "2024-01-01T00:00:00"
        for i in range(1000):
            doc_id = f"doc-{i}"
            conn.execute("INSERT INTO learned_docs (id, user_id, title, source, kind, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                         (doc_id, "bench_user", f"Doc {i}", "bench", "ingested", now))
            text = f"Content about topic {i}. " * 5
            chunk_id = f"{doc_id}-chunk-0"
            vec = embedder.embed_query(text)
            conn.execute("INSERT INTO learned_chunks (id, doc_id, chunk_index, text, created_at) VALUES (?, ?, ?, ?, ?)",
                         (chunk_id, doc_id, 0, text, now))
            conn.execute("INSERT INTO learned_chunks_vec (id, embedding) VALUES (?, ?)",
                         (chunk_id, sqlite_vec.serialize_float32(vec.tolist())))
            conn.execute("INSERT INTO learned_chunks_fts (id, text) VALUES (?, ?)",
                         (chunk_id, text))
        conn.commit()

        def _run():
            with patch("memory.knowledge._connect", return_value=conn):
                return search_knowledge("topic 500", limit=10, embedder=embedder, user_id="bench_user")

        results = benchmark(_run)
        assert len(results) <= 10

    def test_knowledge_context_for_latency(self, benchmark, embedder, tmp_path):
        """Benchmark knowledge_context_for formatting."""
        db_path = tmp_path / "bench.db"
        conn = _connect(str(db_path))
        import sqlite_vec
        now = "2024-01-01T00:00:00"
        for i in range(100):
            doc_id = f"doc-{i}"
            conn.execute("INSERT INTO learned_docs (id, user_id, title, source, kind, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                         (doc_id, "bench_user", f"Doc {i}", "bench", "ingested", now))
            text = f"Content about {i}. " * 10
            chunk_id = f"{doc_id}-chunk-0"
            vec = embedder.embed_query(text)
            conn.execute("INSERT INTO learned_chunks (id, doc_id, chunk_index, text, created_at) VALUES (?, ?, ?, ?, ?)",
                         (chunk_id, doc_id, 0, text, now))
            conn.execute("INSERT INTO learned_chunks_vec (id, embedding) VALUES (?, ?)",
                         (chunk_id, sqlite_vec.serialize_float32(vec.tolist())))
            conn.execute("INSERT INTO learned_chunks_fts (id, text) VALUES (?, ?)",
                         (chunk_id, text))
        conn.commit()

        def _run():
            with patch("memory.knowledge._connect", return_value=conn):
                return knowledge_context_for("topic 50", limit=10, embedder=embedder, user_id="bench_user")

        ctx = benchmark(_run)
        assert "<knowledge_context>" in ctx


# ─── Tool Dispatch Benchmarks ────────────────────────────────────────────────
class TestDispatchPerformance:
    """Benchmarks for tool dispatch."""

    def test_validate_args_latency(self, benchmark):
        """Benchmark argument validation."""
        def _run():
            return _validate_args("save_note", {"title": "t", "content": "c"})

        result = benchmark(_run)
        assert result is None  # Valid = None

    def test_classify_result_latency(self, benchmark):
        """Benchmark result classification."""
        def _run():
            return _classify_result("tool", {}, "success output")

        result = benchmark(_run)
        assert result.ok is True

    def test_owner_embedder_latency(self, benchmark):
        """Benchmark _owner_embedder extraction."""
        class Owner:
            def __init__(self):
                self._memorize = MagicMock()
                self._memorize._mem = MagicMock()
                self._memorize._mem._embedder = FakeEmbedder()

        owner = Owner()

        def _run():
            return _owner_embedder(owner)

        embedder = benchmark(_run)
        assert embedder is not None


# ─── Report Writing Benchmarks ────────────────────────────────────────────────
class TestReportPerformance:
    """Benchmarks for write_report."""

    def test_write_report_latency(self, benchmark, tmp_path):
        """Benchmark write_report."""
        import os
        os.environ["WORKSPACE_ROOT"] = str(tmp_path)

        def _run():
            return write_report("Benchmark Report", "Content " * 100, report_dir="reports")

        result = benchmark(_run)
        assert "report written" in result.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-m", "perf", "--benchmark-only", "-v"])