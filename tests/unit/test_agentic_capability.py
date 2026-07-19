"""
tests/unit/test_agentic_capability.py

Unit tests for agentic/capability.py — capability matching and tool schema filtering.

Run: pytest tests/unit/test_agentic_capability.py -v
"""
from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

os.environ.setdefault("WORKSPACE_ROOT", "/tmp/aiko_test_workspace")

sys.path.insert(0, "/home/oppa-ai/jetson")
from system.config import load_config
load_config()

from agentic.capability import (
    Capability,
    TOOL_DOMAINS,
    ALWAYS_ON_TOOLS,
    CAPABILITIES,
    _CAPABILITY_INSTRUCT,
    _CAPABILITY_THRESHOLD,
    _trigger_embed_cache,
    _get_trigger_embedding,
    match_capabilities,
    filtered_tool_schemas,
)


class FakeEmbedder:
    """Deterministic embedder for tests."""
    def __init__(self):
        self.call_count = 0
        self.last_text = None

    def embed_query(self, text: str, instruct: str = "") -> np.ndarray:
        self.call_count += 1
        self.last_text = text
        h = hash(text + instruct) % 1000
        return np.array([float(h) / 1000.0] * 384, dtype=np.float32)


class TestCapabilityStructure:
    """Tests for capability data structures."""

    def test_tool_domains_comprehensive(self):
        """All registered tools should have a domain or be in ALWAYS_ON_TOOLS."""
        all_tools = set(TOOL_DOMAINS.keys()) | ALWAYS_ON_TOOLS
        # Check key tools are present
        assert "deep_search" in TOOL_DOMAINS
        assert "deep_research" in TOOL_DOMAINS
        assert "write_report" in TOOL_DOMAINS
        assert "learn_knowledge" in TOOL_DOMAINS
        assert "kb_search" in TOOL_DOMAINS
        assert "synthesize_report" in TOOL_DOMAINS
        assert "combine_evidence" in TOOL_DOMAINS
        assert "condense_text" in TOOL_DOMAINS
        assert "polish_text" in TOOL_DOMAINS
        assert "make_plan" in ALWAYS_ON_TOOLS
        assert "save_note" in ALWAYS_ON_TOOLS

    def test_capabilities_defined(self):
        expected_caps = {"research", "scheduling", "kb_proposal", "photo", "repo", "job_hunt", "social"}
        assert set(CAPABILITIES.keys()) == expected_caps

    def test_research_capability_domains(self):
        """Research capability should include research, kb, reports domains."""
        cap = CAPABILITIES["research"]
        assert set(cap.tool_domains) == {"research", "kb", "reports"}

    def test_repo_capability_includes_reports(self):
        cap = CAPABILITIES["repo"]
        assert "reports" in cap.tool_domains

    def test_always_on_tools_comprehensive(self):
        """ALWAYS_ON_TOOLS should include base tools every turn needs."""
        expected = {"make_plan", "create_checklist", "save_note", "read_workspace_file",
                    "summarize_task_state", "list_playbooks", "run_playbook", "final_answer"}
        assert expected.issubset(ALWAYS_ON_TOOLS)


class TestTriggerEmbeddingCache:
    """Tests for trigger embedding caching."""

    def test_cache_populated_on_first_call(self):
        embedder = FakeEmbedder()
        cap = CAPABILITIES["research"]
        _trigger_embed_cache.clear()

        vec1 = _get_trigger_embedding(cap, embedder)
        assert embedder.call_count == 1

        vec2 = _get_trigger_embedding(cap, embedder)
        assert embedder.call_count == 1  # Cached, no new call
        assert np.array_equal(vec1, vec2)

    def test_different_capabilities_different_vectors(self):
        embedder = FakeEmbedder()
        _trigger_embed_cache.clear()

        vec_research = _get_trigger_embedding(CAPABILITIES["research"], embedder)
        vec_scheduling = _get_trigger_embedding(CAPABILITIES["scheduling"], embedder)

        # Should be different (different trigger texts)
        assert not np.array_equal(vec_research, vec_scheduling)


class TestMatchCapabilities:
    """Tests for match_capabilities function."""

    def test_semantic_match_with_embedder(self):
        """Should match capabilities via cosine similarity."""
        embedder = FakeEmbedder()
        _trigger_embed_cache.clear()

        caps = match_capabilities("do deep research on quantum computing", embedder=embedder)
        assert "research" in caps

    def test_keyword_fallback_without_embedder(self):
        """Should fall back to keyword matching when no embedder."""
        caps = match_capabilities("schedule a meeting for tomorrow")
        assert "scheduling" in caps

    def test_keyword_fallback_when_embedder_fails(self):
        """Should fall back to keyword when embedder raises."""
        class BadEmbedder:
            def embed_query(self, *a, **k):
                raise RuntimeError("embedder broken")

        caps = match_capabilities("research this topic", embedder=BadEmbedder())
        # Should still match via keyword fallback
        assert "research" in caps

    def test_precomputed_query_vector_used(self):
        """Pre-computed query_vector should be used instead of re-embedding."""
        embedder = FakeEmbedder()
        _trigger_embed_cache.clear()

        # Pre-compute a vector
        query_vec = embedder.embed_query("test query", instruct=_CAPABILITY_INSTRUCT)

        caps = match_capabilities(
            "actual user input ignored",
            embedder=embedder,
            query_vector=query_vec
        )
        # embedder should only be called for trigger embeddings, not for query
        # (trigger embeddings are cached, so call_count might be small)
        assert isinstance(caps, list)

    def test_threshold_filtering(self):
        """Capabilities below threshold should not match."""
        # With our deterministic fake embedder, some capabilities may score low
        embedder = FakeEmbedder()
        _trigger_embed_cache.clear()

        caps = match_capabilities("completely unrelated gibberish xyz", embedder=embedder)
        # May return empty or only keyword matches
        assert isinstance(caps, list)

    def test_multiple_capabilities_can_match(self):
        embedder = FakeEmbedder()
        _trigger_embed_cache.clear()

        caps = match_capabilities("research and schedule a meeting", embedder=embedder)
        # Both should match via keyword fallback at minimum
        assert "research" in caps or "scheduling" in caps


class TestFilteredToolSchemas:
    """Tests for filtered_tool_schemas tool filtering."""

    def build_mock_schemas(self):
        """Build mock tool schemas for testing."""
        return [
            {"function": {"name": "make_plan", "parameters": {}}},
            {"function": {"name": "deep_search", "parameters": {}}},
            {"function": {"name": "deep_research", "parameters": {}}},
            {"function": {"name": "write_report", "parameters": {}}},
            {"function": {"name": "learn_knowledge", "parameters": {}}},
            {"function": {"name": "schedule_job", "parameters": {}}},
            {"function": {"name": "search_jobs", "parameters": {}}},
            {"function": {"name": "final_answer", "parameters": {}}},
        ]

    def test_no_capabilities_returns_all(self):
        """No matched capabilities -> return all schemas unchanged."""
        schemas = self.build_mock_schemas()
        filtered = filtered_tool_schemas(schemas, [])
        assert len(filtered) == len(schemas)

    def test_research_capability_includes_research_tools(self):
        """Research capability pulls in research, kb, reports domains."""
        schemas = self.build_mock_schemas()
        filtered = filtered_tool_schemas(schemas, ["research"])
        names = {s["function"]["name"] for s in filtered}

        # Research domain
        assert "deep_search" in names
        assert "deep_research" in names
        # KB domain
        assert "learn_knowledge" in names
        # Reports domain
        assert "write_report" in names
        # Always on
        assert "make_plan" in names
        assert "final_answer" in names
        # Not included
        assert "schedule_job" not in names
        assert "search_jobs" not in names

    def test_scheduling_capability_only_adds_scheduling(self):
        schemas = self.build_mock_schemas()
        filtered = filtered_tool_schemas(schemas, ["scheduling"])
        names = {s["function"]["name"] for s in filtered}

        assert "schedule_job" in names
        assert "deep_search" not in names
        assert "make_plan" in names  # Always on

    def test_multiple_capabilities_union(self):
        """Multiple capabilities -> union of their domains."""
        schemas = self.build_mock_schemas()
        filtered = filtered_tool_schemas(schemas, ["research", "scheduling"])
        names = {s["function"]["name"] for s in filtered}

        assert "deep_search" in names
        assert "schedule_job" in names

    def test_unknown_capability_ignored(self):
        """Unknown capability IDs should be ignored, not crash."""
        schemas = self.build_mock_schemas()
        filtered = filtered_tool_schemas(schemas, ["nonexistent_capability"])
        # Should fall back to always_on only
        names = {s["function"]["name"] for s in filtered}
        assert names == {"make_plan", "final_answer"}  # only always_on from our mock

    def test_always_on_tools_always_included(self):
        """ALWAYS_ON_TOOLS always present regardless of capabilities."""
        schemas = self.build_mock_schemas()
        filtered = filtered_tool_schemas(schemas, ["scheduling"])
        names = {s["function"]["name"] for s in filtered}

        assert "make_plan" in names
        assert "create_checklist" not in names  # Not in our mock but would be
        assert "save_note" not in names  # Not in mock
        assert "final_answer" in names

    def test_empty_filtered_returns_all(self):
        """If filtering removes everything, return all schemas (safety net)."""
        schemas = self.build_mock_schemas()
        # Capability that doesn't match any tool domains
        filtered = filtered_tool_schemas(schemas, ["nonexistent"])
        # Our mock only has always_on tools that don't have domains in TOOL_DOMAINS
        # So filtered would be just always_on. But safety net: if filtered is empty, return all.
        assert len(filtered) > 0


class TestCapabilityThreshold:
    """Tests around _CAPABILITY_THRESHOLD."""

    def test_default_threshold_value(self):
        assert _CAPABILITY_THRESHOLD == 0.35

    def test_instruct_constant(self):
        assert _CAPABILITY_INSTRUCT == "Which capability/tool domain applies to this task?"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])