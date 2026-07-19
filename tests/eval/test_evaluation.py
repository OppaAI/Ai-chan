"""
tests/eval/test_evaluation.py

Evaluation tests for Aiko's agentic system:
- Retrieval accuracy (KB + web)
- Synthesis quality (coherence, factuality, style)
- Playbook selection accuracy
- End-to-end task completion

Run: pytest tests/eval/test_evaluation.py -v
"""
from __future__ import annotations

import os
import sys
import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

os.environ.setdefault("WORKSPACE_ROOT", "/tmp/aiko_test_workspace")

sys.path.insert(0, "/home/oppa-ai/jetson")
from system.config import load_config
load_config()

from agentic import schema
from agentic.toolkit.synthesize import synthesize_report, detect_style, detect_compare, split_subjects
from agentic.toolkit.research import deep_search, deep_research, condense_evidence
from agentic.capability import match_capabilities, filtered_tool_schemas
from agentic.agentic import _verify_final_answer
from agentic.toolkit.reports import write_report
from memory.knowledge import search_knowledge, knowledge_context_for, ingest_text


class FakeEmbedder:
    """Deterministic embedder for evaluations."""
    def embed_query(self, text: str, instruct: str = "") -> np.ndarray:
        h = hash(text + instruct) % 1000
        return np.array([float(h) / 1000.0] * 384, dtype=np.float32)


class MockLLMClient:
    def __init__(self, responses: list[str] = None):
        self.responses = responses or ["Synthesized answer"]
        self.call_count = 0
        self.idx = 0

    @property
    def chat(self):
        mock_chat = MagicMock()
        mock_chat.completions = MagicMock()
        mock_chat.completions.create = self._create
        return mock_chat

    def _create(self, model, messages, **kwargs):
        if self.idx < len(self.responses):
            resp = self.responses[self.idx]
        else:
            resp = self.responses[-1]
        self.idx += 1
        self.call_count += 1
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content=resp))]
        return mock_resp


class MockOwner:
    def __init__(self, client=None):
        self._client = client or MockLLMClient()
        self._llm_model = "test-model"
        self._memorize = MagicMock()
        self._memorize._mem = MagicMock()
        self._memorize._mem._embedder = FakeEmbedder()


# ─── Style Detection Evaluation ────────────────────────────────────────────────
class TestStyleDetectionAccuracy:
    """Evaluate style detection against expected labels."""

    test_cases = [
        ("write a professional report on AI", "professional"),
        ("give me a concise summary", "concise"),
        ("brief overview please", "brief"),
        ("keep it short and sweet", "concise"),
        ("tl;dr version", "concise"),
        ("explain casually like a friend", "casual"),
        ("informal tone please", "casual"),
        ("plain text no formatting", "plain"),
        ("just the facts", "plain"),
        ("comprehensive detailed analysis", "professional"),
        ("in depth study", "professional"),
    ]

    @pytest.mark.parametrize("prompt,expected", test_cases)
    def test_style_detection(self, prompt, expected):
        from agentic.toolkit.synthesize import detect_style
        assert detect_style(prompt) == expected, f"Failed for: '{prompt}' -> got {detect_style(prompt)}, expected {expected}"


# ─── Comparison Detection Evaluation ──────────────────────────────────────────
class TestComparisonDetectionAccuracy:
    """Evaluate comparison subject extraction."""

    test_cases = [
        ("compare JAX vs PyTorch", ("JAX", "PyTorch")),
        ("compare React versus Vue", ("React", "Vue")),
        ("what are the differences between Python and JavaScript", ("Python", "JavaScript")),
        ("contrast TensorFlow and PyTorch", ("TensorFlow", "PyTorch")),
        ("compare A with B", ("A", "B")),
        ("pros and cons of option A vs option B", ("option A", "option B")),
    ]

    @pytest.mark.parametrize("prompt,expected", test_cases)
    def test_detect_compare(self, prompt, expected):
        result = detect_compare(prompt)
        assert result == expected, f"Failed for '{prompt}': got {result}, expected {expected}"

    def test_no_false_positives(self):
        """Non-comparison prompts should not match."""
        non_comparison = [
            "write a report on AI",
            "how does quantum computing work",
            "explain the theory of relativity",
            "search for papers on ML",
        ]
        for prompt in non_comparison:
            assert detect_compare(prompt) is None, f"False positive for: '{prompt}'"

    def test_split_subjects_multi(self):
        """Test multi-subject extraction."""
        subjects = split_subjects("compare A, B, and C")
        assert len(subjects) >= 2
        assert "A" in subjects
        assert "B" in subjects
        assert "C" in subjects


# ─── Synthesis Quality Evaluation ─────────────────────────────────────────────
class TestSynthesisQuality:
    """Evaluate synthesis output quality."""

    def test_synthesis_includes_evidence(self):
        """Synthesized report should reference evidence."""
        client = MockLLMClient(["The evidence shows quantum computing uses qubits."])
        embedder = FakeEmbedder()
        evidence = "Quantum computers use qubits which can be in superposition."

        result = synthesize_report(evidence, "explain quantum computing", client=client, model="m", embedder=embedder)

        # Should contain key terms from evidence
        assert "qubit" in result.lower() or "quantum" in result.lower()

    def test_synthesis_professional_style(self):
        """Default style should be professional."""
        client = MockLLMClient(["A comprehensive analysis reveals..."])
        embedder = FakeEmbedder()
        result = synthesize_report("evidence", "research topic", client=client, model="m", embedder=embedder)

        # Professional style markers
        assert any(marker in result for marker in ["analysis", "comprehensive", "demonstrates", "indicates", "furthermore"])

    def test_synthesis_concise_style(self):
        """Concise style should be brief."""
        client = MockLLMClient(["Quantum computing uses qubits. Key advantage: superposition."])
        embedder = FakeEmbedder()
        result = synthesize_report("evidence", "concise summary of quantum computing", client=client, model="m", embedder=embedder)

        # Should be relatively short
        assert len(result) < 500

    def test_synthesis_no_hallucination(self):
        """Synthesis should not invent facts not in evidence."""
        client = MockLLMClient(["Based on the evidence, quantum computing uses qubits."])
        embedder = FakeEmbedder()
        evidence = "Quantum computers use qubits for computation."

        result = synthesize_report(evidence, "explain quantum computing", client=client, model="m", embedder=embedder)

        # Should not contain specific numbers, dates, or claims not in evidence
        assert "2024" not in result
        assert "99%" not in result

    def test_comparison_synthesis_structure(self):
        """Comparison synthesis should have side-by-side structure."""
        client = MockLLMClient(["Subject A: Feature 1. Subject B: Feature 2. Verdict: A wins on X."])
        embedder = FakeEmbedder()
        evidence = "Evidence about A and B."

        result = synthesize_report(
            evidence, "compare A vs B",
            client=client, model="m", embedder=embedder,
            comparison_subjects=["A", "B"]
        )

        # Should mention both subjects
        assert "A" in result and "B" in result


# ─── Retrieval Accuracy Evaluation ────────────────────────────────────────────
class TestRetrievalAccuracy:
    """Evaluate KB and web search retrieval quality."""

    def setup_method(self):
        self.tmp_path = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp_path, "eval.db")
        self.conn = _connect(self.db_path)
        self._seed_eval_data()

    def teardown_method(self):
        self.conn.close()
        import shutil
        shutil.rmtree(self.tmp_path, ignore_errors=True)

    def _seed_eval_data(self):
        embedder = FakeEmbedder()
        import sqlite_vec
        now = "2024-01-01T00:00:00"
        topics = [
            ("doc-1", "Quantum Computing", "Quantum computing uses qubits. Qubits enable superposition and entanglement."),
            ("doc-2", "Classical Computing", "Classical computers use bits. Bits are 0 or 1. No superposition."),
            ("doc-3", "Machine Learning", "Machine learning trains models on data. Neural networks are common."),
            ("doc-4", "Neural Networks", "Neural networks have layers. Deep learning uses many layers."),
            ("doc-5", "Quantum Algorithms", "Shor's algorithm factors integers. Grover's algorithm searches databases."),
        ]
        for doc_id, title, text in topics:
            self.conn.execute("INSERT INTO learned_docs (id, user_id, title, source, kind, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                              (doc_id, "eval_user", title, "eval", "ingested", now))
            chunks = [text[i:i+80] for i in range(0, len(text), 60)]
            for j, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}-chunk-{j}"
                vec = embedder.embed_query(chunk)
                self.conn.execute("INSERT INTO learned_chunks (id, doc_id, chunk_index, text, created_at) VALUES (?, ?, ?, ?, ?)",
                                  (chunk_id, doc_id, j, chunk, now))
                self.conn.execute("INSERT INTO learned_chunks_vec (id, embedding) VALUES (?, ?)",
                                  (chunk_id, sqlite_vec.serialize_float32(vec.tolist())))
                self.conn.execute("INSERT INTO learned_chunks_fts (id, text) VALUES (?, ?)",
                                  (chunk_id, chunk))
        self.conn.commit()

    def test_search_recalls_relevant_doc(self):
        """Search should find the most relevant document."""
        with patch("memory.knowledge._connect", return_value=self.conn):
            results = search_knowledge("quantum qubits superposition", limit=5, embedder=FakeEmbedder(), user_id="eval_user")

        assert len(results) > 0
        # Top result should be quantum computing doc
        top_text = results[0]["text"].lower()
        assert "quantum" in top_text or "qubit" in top_text

    def test_search_filters_irrelevant(self):
        """Search should not return irrelevant docs in top-k."""
        with patch("memory.knowledge._connect", return_value=self.conn):
            results = search_knowledge("shor algorithm factoring", limit=3, embedder=FakeEmbedder(), user_id="eval_user")

        # Should find quantum algorithms, not classical computing
        for r in results:
            assert "classical" not in r["text"].lower() or r["score"] < 0.5

    def test_knowledge_context_format(self):
        """knowledge_context_for should produce valid XML with scores."""
        with patch("memory.knowledge._connect", return_value=self.conn):
            ctx = knowledge_context_for("quantum algorithms", limit=3, embedder=FakeEmbedder(), user_id="eval_user")

        assert "<knowledge_context>" in ctx
        assert "</knowledge_context>" in ctx
        assert "<knowledge_chunk" in ctx
        assert "doc_id" in ctx
        assert "score=" in ctx


# ─── Playbook Selection Accuracy ──────────────────────────────────────────────
class TestPlaybookSelectionAccuracy:
    """Evaluate playbook selection against ground truth."""

    test_cases = [
        # (prompt, expected_playbook_id)
        ("research quantum computing and write a report", "research_and_report"),
        ("do deep research on AI and save a report", "research_and_report"),
        ("investigate climate change comprehensively", "research_and_report"),
        ("search for quantum computing basics and write summary", "search_kb_and_report"),
        ("look up what is machine learning and save", "search_kb_and_report"),
        ("find information on neural networks and report", "search_kb_and_report"),
        ("compare JAX vs PyTorch for deep learning", "compare_and_report"),
        ("compare React versus Vue pros and cons", "compare_and_report"),
        ("what are the differences between Docker and Podman", "compare_and_report"),
        ("make a checklist for deployment and save it", "checklist_and_save"),
        ("create a todo list for the project and save", "checklist_and_save"),
        ("save this note: meeting at 3pm", "simple_save_note"),
        ("write note: buy groceries", "simple_save_note"),
    ]

    @pytest.mark.parametrize("prompt,expected", test_cases)
    def test_playbook_selection(self, prompt, expected):
        graph = schema.plan_from_master(prompt, cap_ids=["research"])
        assert graph is not None, f"No playbook matched for: '{prompt}'"
        assert graph.id == expected, f"Expected {expected}, got {graph.id} for: '{prompt}'"


# ─── End-to-End Task Completion ───────────────────────────────────────────────
class TestEndToEndTaskCompletion:
    """Evaluate complete task execution."""

    def test_research_task_completes(self):
        """Full research task should execute all nodes successfully."""
        owner = MockOwner()
        owner._client = MockLLMClient(["Research report synthesized with findings."])
        owner._memorize._mem._embedder = FakeEmbedder()

        with patch("agentic.agentic._owner_embedder", return_value=FakeEmbedder()):
            with patch("agentic.agentic._fetch_agentic_only_context", return_value={}):
                with patch("agentic.toolkit.research.deep_research") as mock_deep:
                    mock_deep.return_value = "Deep research results with evidence"
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
        assert all(r.ok for r in result.results)
        assert len(result.results) == 6

    def test_compare_task_completes(self):
        """Comparison task should run parallel research and synthesize."""
        owner = MockOwner()
        owner._client = MockLLMClient(["Comparison report: A has X, B has Y. Verdict: A wins."])
        owner._memorize._mem._embedder = FakeEmbedder()

        with patch("agentic.agentic._owner_embedder", return_value=FakeEmbedder()):
            with patch("agentic.agentic._fetch_agentic_only_context", return_value={}):
                with patch("agentic.toolkit.research.deep_research") as mock_deep:
                    mock_deep.side_effect = ["Research A results", "Research B results"]
                    with patch("agentic.toolkit.reports.write_report") as mock_write:
                        mock_write.return_value = '{"ok": true}'
                        with patch("agentic.toolkit.synthesize.learn_report") as mock_learn:
                            mock_learn.return_value = "doc-456"
                            result = schema.run_schema_agent(
                                "compare TensorFlow vs PyTorch",
                                cap_ids=["research"],
                                embedder=FakeEmbedder(),
                                llm_client=owner._client,
                                llm_model=owner._llm_model,
                            )

        assert result is not None
        assert result.graph.id == "compare_and_report"
        assert mock_deep.call_count == 2  # Parallel execution

    def test_checklist_task_completes(self):
        """Checklist creation should work end-to-end."""
        with patch("agentic.agentic._owner_embedder", return_value=FakeEmbedder()):
            with patch("agentic.agentic._fetch_agentic_only_context", return_value={}):
                result = schema.run_schema_agent(
                    "make a checklist for testing and save it",
                    cap_ids=[],
                    embedder=FakeEmbedder(),
                )

        assert result is not None
        assert result.graph.id == "checklist_and_save"
        assert all(r.ok for r in result.results)


# ─── Verification Evaluation ──────────────────────────────────────────────────
class TestVerificationAccuracy:
    """Evaluate final answer verification."""

    def test_verification_passes_good_answer(self):
        """Good answers should pass verification."""
        from agentic.agentic import TaskState, ToolResult

        state = TaskState("research quantum computing")
        state.record(ToolResult(True, "deep_research", {"query": "q"}, "Found: quantum uses qubits"))
        state.record(ToolResult(True, "write_report", {"title": "R"}, "Report written"))

        owner = MockOwner()
        owner._client = MockLLMClient(['{"ok": true, "score": 0.9, "issues": []}'])

        result = _verify_final_answer(owner, "research quantum computing", "Quantum computing uses qubits for computation.", state)

        assert result.ok is True
        assert result.score >= 0.8

    def test_verification_catches_bad_answer(self):
        """Unsubstantiated answers should fail."""
        from agentic.agentic import TaskState, ToolResult

        state = TaskState("research quantum computing")
        # No tool results recorded

        owner = MockOwner()
        owner._client = MockLLMClient(['{"ok": false, "score": 0.2, "issues": ["no evidence cited"]}'])

        result = _verify_final_answer(owner, "research quantum computing", "Quantum computing is magic.", state)

        assert result.ok is False
        assert result.score < 0.5


# ─── Report Writing Evaluation ────────────────────────────────────────────────
class TestReportWritingQuality:
    """Evaluate report output quality."""

    def test_write_report_creates_markdown(self, tmp_path):
        """write_report should create valid markdown."""
        import os
        os.environ["WORKSPACE_ROOT"] = str(tmp_path)

        result = write_report("Test Report", "# Heading\n\nContent here.", report_dir="reports")
        assert "report written" in result.lower()

        # Check file exists and has content
        report_files = list(tmp_path.glob("reports/*.md"))
        assert len(report_files) == 1
        content = report_files[0].read_text()
        assert "# Heading" in content
        assert "Content here" in content

    def test_write_report_arxiv_style(self, tmp_path):
        """arxiv_style should create structured sections."""
        import os
        os.environ["WORKSPACE_ROOT"] = str(tmp_path)

        write_report("Paper", "", report_dir="reports", arxiv_style=True, section="abstract", append=False)
        write_report("Paper", "Abstract content", report_dir="reports", arxiv_style=True, section="abstract", append=True)
        write_report("Paper", "Intro content", report_dir="reports", arxiv_style=True, section="introduction", append=True)

        report_files = list(tmp_path.glob("reports/*.md"))
        assert len(report_files) == 1
        content = report_files[0].read_text()
        assert "## Abstract" in content
        assert "## 1. Introduction" in content
        assert "Abstract content" in content
        assert "Intro content" in content


# ─── Capability Routing Evaluation ────────────────────────────────────────────
class TestCapabilityRoutingAccuracy:
    """Evaluate capability matching and tool filtering."""

    def test_research_capability_includes_right_tools(self):
        from agentic.agentic import _TOOL_DEFS
        all_schemas = [s for s, _ in _TOOL_DEFS]
        filtered = filtered_tool_schemas(all_schemas, ["research"])
        names = {s["function"]["name"] for s in filtered}

        # Research domain
        assert "deep_search" in names
        assert "deep_research" in names
        assert "read_paper_url" in names
        # KB domain
        assert "learn_knowledge" in names
        # Reports domain
        assert "write_report" in names
        # Always on
        assert "make_plan" in names
        assert "save_note" in names
        # Not included
        assert "schedule_job" not in names

    def test_multiple_capabilities_union(self):
        from agentic.agentic import _TOOL_DEFS
        all_schemas = [s for s, _ in _TOOL_DEFS]
        filtered = filtered_tool_schemas(all_schemas, ["research", "scheduling"])
        names = {s["function"]["name"] for s in filtered}

        assert "deep_search" in names
        assert "schedule_job" in names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])