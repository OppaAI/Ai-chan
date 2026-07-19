"""
tests/unit/test_agentic_synthesize.py

Unit tests for agentic/toolkit/synthesize.py — graph-level LLM synthesis tools.

Run: pytest tests/unit/test_agentic_synthesize.py -v
"""
from __future__ import annotations

import json
import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

os.environ.setdefault("WORKSPACE_ROOT", "/tmp/aiko_test_workspace")

sys.path.insert(0, "/home/oppa-ai/jetson")
from system.config import load_config
load_config()

from agentic.toolkit.synthesize import (
    detect_style,
    detect_compare,
    split_subjects,
    combine_evidence,
    condense_text,
    kb_search,
    learn_report,
    _STYLE_INSTRUCTIONS,
    _format_comparison_block,
    synthesize_report,
    polish_text,
)


class FakeEmbedder:
    """Deterministic embedder for tests."""
    def embed_query(self, text: str, instruct: str = "") -> np.ndarray:
        h = hash(text) % 1000
        return np.array([float(h) / 1000.0] * 384, dtype=np.float32)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.embed_query(t) for t in texts])


class MockLLMClient:
    """Mock OpenAI-compatible client."""
    def __init__(self, response_text: str = "Synthesized response"):
        self.response_text = response_text
        self.call_count = 0
        self.last_messages = None

    def chat_completions_create(self, model: str, messages: list[dict], **kwargs):
        self.call_count += 1
        self.last_messages = messages
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content=self.response_text))]
        return mock_resp

    # Compatibility with openai client structure
    @property
    def chat(self):
        return MagicMock(completions=MagicMock(create=self.chat_completions_create))


class TestStyleDetection:
    """Tests for detect_style heuristic."""

    def test_default_professional(self):
        assert detect_style("write a report on quantum computing") == "professional"

    def test_concise_keywords(self):
        assert detect_style("give me a concise summary") == "concise"
        assert detect_style("keep it short and sweet") == "concise"
        assert detect_style("in brief") == "concise"
        assert detect_style("tl;dr") == "concise"

    def test_brief_keywords(self):
        assert detect_style("brief overview please") == "brief"
        assert detect_style("quick summary") == "brief"

    def test_casual_keywords(self):
        assert detect_style("chill, just explain it casually") == "casual"
        assert detect_style("informal tone please") == "casual"

    def test_plain_keywords(self):
        assert detect_style("plain text no fluff") == "plain"
        assert detect_style("just the facts") == "plain"

    def test_first_match_wins(self):
        # "concise" should win over "professional" if both present
        assert detect_style("professional but concise report") == "concise"
        assert detect_style("concise professional report") == "concise"

    def test_informal_maps_to_casual(self):
        assert detect_style("make it informal") == "casual"  # informal in casual keywords


class TestComparisonDetection:
    """Tests for detect_compare and split_subjects."""

    def test_detect_vs_pattern(self):
        result = detect_compare("compare JAX vs PyTorch")
        assert result == ("JAX", "PyTorch")

    def test_detect_versus_pattern(self):
        result = detect_compare("what are the differences between React versus Vue")
        assert result == ("React", "Vue")

    def test_detect_compared_to(self):
        result = detect_compare("TensorFlow compared to JAX")
        assert result == ("TensorFlow", "JAX")

    def test_detect_contrast(self):
        result = detect_compare("contrast Django and Flask")
        assert result == ("Django", "Flask")

    def test_no_match_returns_none(self):
        assert detect_compare("write a report on AI") is None
        assert detect_compare("what is quantum computing") is None
        assert detect_compare("") is None

    def test_split_subjects_from_vs(self):
        subjects = split_subjects("compare JAX vs PyTorch vs TensorFlow")
        assert subjects == ["JAX", "PyTorch", "TensorFlow"]

    def test_split_subjects_from_comma(self):
        subjects = split_subjects("compare A, B, and C")
        assert len(subjects) >= 2

    def test_split_subjects_requires_compare_keyword(self):
        # Should not split a random list
        subjects = split_subjects("apples, oranges, bananas")
        assert subjects == []

    def test_split_subjects_strips_compare_words(self):
        subjects = split_subjects("compare the differences between React and Vue")
        assert "React" in subjects
        assert "Vue" in subjects


class TestCombineEvidence:
    """Tests for combine_evidence."""

    def test_combines_multiple_parts(self):
        parts = ["web evidence", "kb evidence", "prior round"]
        result = combine_evidence(parts)
        assert "web evidence" in result
        assert "kb evidence" in result
        assert "prior round" in result
        assert "---" in result  # separator

    def test_filters_empty_parts(self):
        parts = ["content", "", "  ", None, "more content"]
        result = combine_evidence(parts)
        assert result == "content\n\n---\n\nmore content"

    def test_single_part_returns_as_is(self):
        result = combine_evidence(["single piece"])
        assert result == "single piece"

    def test_empty_list_returns_empty(self):
        assert combine_evidence([]) == ""


class TestCondenseText:
    """Tests for condense_text semantic condensation."""

    def test_short_text_returns_as_is(self):
        embedder = FakeEmbedder()
        text = "short text"
        result = condense_text(text, "query", embedder, max_chars=1000)
        assert result == text

    def test_long_text_condensed(self):
        embedder = FakeEmbedder()
        # Create text longer than max_chars
        long_text = "source-1: This is a test document. " * 100  # ~4000 chars
        result = condense_text(long_text, "test query", embedder, max_chars=500)
        assert len(result) <= 500
        assert "source-" in result  # preserves source headers

    def test_empty_text_returns_empty(self):
        embedder = FakeEmbedder()
        result = condense_text("", "query", embedder)
        assert result == ""

    def test_embedder_failure_fallbacks_to_truncate(self):
        """When embedder fails, should head-truncate."""
        class BadEmbedder:
            def embed_query(self, *a, **k):
                raise RuntimeError("embedder broken")

        embedder = BadEmbedder()
        long_text = "x" * 5000
        result = condense_text(long_text, "query", embedder, max_chars=100)
        assert len(result) == 100
        assert result == long_text[:100]


class TestKBSearch:
    """Tests for kb_search wrapper."""

    def test_returns_no_matching_when_empty(self):
        with patch("agentic.toolkit.synthesize.knowledge_context_for", return_value="<knowledge_context>\nNo matching learned knowledge found.\n</knowledge_context>"):
            result = kb_search("test query", embedder=FakeEmbedder())
            assert result == "[no matching learned knowledge]"

    def test_strips_xml_wrapper(self):
        kb_context = """<knowledge_context>
<knowledge_chunk doc_id="1" title="Test" kind="ingested" source="test" score="0.9">Chunk content here</knowledge_chunk>
</knowledge_context>"""
        with patch("agentic.toolkit.synthesize.knowledge_context_for", return_value=kb_context):
            result = kb_search("test", embedder=FakeEmbedder())
            assert "Chunk content here" in result
            assert "<knowledge_context>" not in result

    def test_handles_exception_gracefully(self):
        with patch("agentic.toolkit.synthesize.knowledge_context_for", side_effect=Exception("DB error")):
            result = kb_search("test", embedder=FakeEmbedder())
            assert result == "[no matching learned knowledge]"


class TestLearnReport:
    """Tests for learn_report ingestion."""

    def test_returns_doc_id_on_success(self):
        with patch("agentic.toolkit.synthesize.ingest_text", return_value="doc-123"):
            result = learn_report("Test Report", "Report content", embedder=FakeEmbedder())
            assert result == "doc-123"

    def test_empty_text_skips(self):
        result = learn_report("Test", "", embedder=FakeEmbedder())
        assert result == "[learn skipped: empty report]"

    def test_exception_returns_error_sentinel(self):
        with patch("agentic.toolkit.synthesize.ingest_text", side_effect=Exception("ingest failed")):
            result = learn_report("Test", "content", embedder=FakeEmbedder())
            assert result.startswith("[learn failed:")


class TestFormatComparisonBlock:
    """Tests for _format_comparison_block."""

    def test_two_subjects(self):
        result = _format_comparison_block(["A", "B"])
        assert "Subject A: A" in result
        assert "Subject B: B" in result
        assert "side-by-side" in result

    def test_three_plus_subjects(self):
        result = _format_comparison_block(["A", "B", "C"])
        assert "3 subjects" in result
        assert "multi-way" in result

    def test_empty_list(self):
        assert _format_comparison_block([]) == ""


class TestSynthesizeReport:
    """Tests for synthesize_report main LLM synthesis function."""

    def test_no_llm_returns_evidence_with_header(self):
        """When no client/model, returns header + evidence."""
        evidence = "Combined web + KB evidence"
        result = synthesize_report(evidence, "test prompt", client=None, model=None)
        assert "Aiko Research Report" in result
        assert "test prompt" in result
        assert evidence in result

    def test_condenses_long_evidence(self):
        """Long evidence should be condensed before LLM call."""
        long_evidence = "x" * 10000
        client = MockLLMClient("OK")
        embedder = FakeEmbedder()
        result = synthesize_report(long_evidence, "test", client=client, model="m", embedder=embedder)
        # Should have called LLM with condensed evidence
        assert client.call_count == 1

    def test_llm_call_with_correct_messages(self):
        client = MockLLMClient("Synthesized answer")
        embedder = FakeEmbedder()
        result = synthesize_report("evidence", "user question", client=client, model="m", embedder=embedder)
        assert client.call_count == 1
        assert client.last_messages is not None
        # Check system message has style instructions
        sys_msg = client.last_messages[0]
        assert sys_msg["role"] == "system"
        assert "professional" in sys_msg["content"].lower() or "precise" in sys_msg["content"].lower()

    def test_comparison_subjects_in_prompt(self):
        client = MockLLMClient("Comparison result")
        embedder = FakeEmbedder()
        result = synthesize_report(
            "evidence", "compare A vs B",
            client=client, model="m", embedder=embedder,
            comparison_subjects=["A", "B"]
        )
        # Check comparison block was included in system prompt
        sys_content = client.last_messages[0]["content"]
        assert "Subject A: A" in sys_content
        assert "Subject B: B" in sys_content

    def test_style_instructions_applied(self):
        client = MockLLMClient("Styled result")
        embedder = FakeEmbedder()
        for style in ["professional", "concise", "brief", "casual", "plain"]:
            client.call_count = 0
            result = synthesize_report("evidence", "prompt", client=client, model="m", style=style, embedder=embedder)
            sys_content = client.last_messages[0]["content"]
            assert _STYLE_INSTRUCTIONS[style] in sys_content

    def test_auto_style_uses_detect_style(self):
        client = MockLLMClient("Result")
        embedder = FakeEmbedder()
        # "concise" should trigger concise style
        result = synthesize_report("evidence", "give me a concise summary", client=client, model="m", style="auto", embedder=embedder)
        sys_content = client.last_messages[0]["content"]
        assert _STYLE_INSTRUCTIONS["concise"] in sys_content

    def test_llm_failure_fallbacks_to_header_plus_evidence(self):
        """When LLM call fails, returns degraded but usable output."""
        client = MockLLMClient()
        client.chat_completions_create = MagicMock(side_effect=Exception("LLM down"))
        embedder = FakeEmbedder()
        result = synthesize_report("evidence", "prompt", client=client, model="m", embedder=embedder)
        assert "automatic synthesis was unavailable" in result.lower() or "evidence" in result.lower()


class TestPolishText:
    """Tests for polish_text style rewrite."""

    def test_passthrough_for_known_styles(self):
        """Known styles return input unchanged (synthesize already applied style)."""
        for style in ["professional", "concise", "brief", "casual", "plain", "informal"]:
            result = polish_text("draft text", client=None, model=None, style=style)
            assert result == "draft text"

    def test_unknown_style_calls_llm(self):
        client = MockLLMClient("Polished version")
        result = polish_text("draft", client=client, model="m", style="shakespearean")
        assert result == "Polished version"
        assert client.call_count == 1

    def test_no_llm_returns_original(self):
        result = polish_text("draft", client=None, model=None, style="shakespearean")
        assert result == "draft"

    def test_empty_input_returns_empty(self):
        assert polish_text("", client=MockLLMClient(), model="m", style="x") == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])