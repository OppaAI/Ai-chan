"""
tests/unit/test_agentic_research.py

Unit tests for web primitives (websurf) and research graphs (research_graph).

Run: pytest tests/unit/test_agentic_research.py -v
"""
from __future__ import annotations

import json
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

from agentic.toolkit.websurf import (
    web_search,
    web_fetch,
    fetch_search_results,
    _stream_download,
    _check_host_ssrf,
    web_search_context,
)
from agentic.toolkit.ingest import (
    _extract_with_markitdown,
    _sniff_content_type,
)
from agentic.toolkit.research import (
    condense_evidence,
    _score_url_chunks,
    _finalize_condensed,
    _apply_corroboration_bonus,
    _fetch_and_score_pipeline,
    _deep_search_impl,
    deep_research,
    _build_deep_research_subgraph,
    DEEP_RESEARCH_NUM_FETCHES,
    DEEP_RESEARCH_NUM_SEARCHES,
    DEEP_RESEARCH_MAX_CHARS_PER_PAGE,
    DEEP_RESEARCH_MAX_ROUNDS,
    CONDENSE_CHUNK_CHARS,
    CONDENSE_TOP_K,
    CONDENSE_MIN_SCORE,
)
from agentic.graph_engine import PlanGraph, PlanNode


class FakeEmbedder:
    """Deterministic bag-of-words embedder for tests.
    
    Same words → similar vectors; different words → dissimilar vectors.
    """
    def embed_query(self, text: str, instruct: str = "") -> np.ndarray:
        words = text.lower().split()
        vec = np.zeros(384, dtype=np.float32)
        for word in words:
            idx = hash(word) % 384
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.embed_query(t) for t in texts])


class MockSearXNG:
    """Mock SearXNG responses."""
    def __init__(self, results=None, error=None):
        self.results = results or [
            {"title": "Result 1", "url": "https://example.com/1", "content": "Content about quantum computing"},
            {"title": "Result 2", "url": "https://example.com/2", "content": "More quantum content"},
        ]
        self.error = error
        self.call_count = 0

    def search(self, query, max_results, pageno=1):
        self.call_count += 1
        if self.error:
            return None, self.error
        return self.results[:max_results], None


class TestFetchSearchResults:
    """Tests for fetch_search_results low-level SearXNG call."""

    def test_successful_search(self):
        # Patch _SEARCH_CACHE (module-level, all caps), importlib.util.find_spec and importlib.import_module
        with patch("agentic.toolkit.websurf._SEARCH_CACHE.get") as mock_cache_get:
            mock_cache_get.return_value = None  # cache miss

            with patch("importlib.util.find_spec") as mock_find_spec:
                mock_find_spec.return_value = True  # requests is available

                with patch("importlib.import_module") as mock_import:
                    mock_requests = MagicMock()
                    mock_get = MagicMock()
                    mock_requests.get.return_value = mock_get
                    mock_get.status_code = 200
                    mock_get.json.return_value = {
                        "results": [
                            {"title": "Test", "url": "https://example.com", "content": "test content"}
                        ]
                    }
                    mock_import.return_value = mock_requests

                    results, error = fetch_search_results("test query", 5)
                    assert error is None
                    assert len(results) == 1
                    assert results[0]["title"] == "Test"

    def test_rate_limit_retry(self):
        with patch("agentic.toolkit.websurf._SEARCH_CACHE.get") as mock_cache_get:
            mock_cache_get.return_value = None  # cache miss

            with patch("importlib.util.find_spec") as mock_find_spec:
                mock_find_spec.return_value = True  # requests is available

                with patch("importlib.import_module") as mock_import:
                    mock_requests = MagicMock()
                    mock_get_429 = MagicMock()
                    mock_get_429.status_code = 429
                    mock_get_ok = MagicMock()
                    mock_get_ok.status_code = 200
                    mock_get_ok.json.return_value = {"results": [{"title": "OK", "url": "https://ok.com", "content": "ok"}]}
                    mock_requests.get.side_effect = [mock_get_429, mock_get_429, mock_get_ok]
                    mock_import.return_value = mock_requests

                    results, error = fetch_search_results("test", 5)
                    assert error is None
                    assert len(results) == 1
                    assert mock_requests.get.call_count == 3

    def test_connection_error_retry(self):
        import requests
        with patch("agentic.toolkit.websurf._SEARCH_CACHE.get") as mock_cache_get:
            mock_cache_get.return_value = None  # cache miss
            with patch("importlib.util.find_spec") as mock_find_spec:
                mock_find_spec.return_value = True  # requests is available

                # Patch requests.get directly
                call_count = [0]
                def mock_get(*args, **kwargs):
                    call_count[0] += 1
                    if call_count[0] == 1:
                        raise requests.exceptions.ConnectionError("conn error")
                    resp = MagicMock()
                    resp.status_code = 200
                    resp.json.return_value = {"results": []}
                    return resp

                with patch("requests.get", side_effect=mock_get):
                    results, error = fetch_search_results("test", 5)
                    assert error is None
                    assert call_count[0] == 2

    def test_invalid_json(self):
        import requests
        with patch("agentic.toolkit.websurf._SEARCH_CACHE.get") as mock_cache_get:
            mock_cache_get.return_value = None  # cache miss
            with patch("importlib.util.find_spec") as mock_find_spec:
                mock_find_spec.return_value = True  # requests is available

                # Mock the response with invalid JSON
                mock_get = MagicMock()
                mock_get.status_code = 200
                mock_get.json.side_effect = ValueError("bad json")

                with patch("requests.get", return_value=mock_get):
                    results, error = fetch_search_results("test", 5)
                    assert "invalid JSON" in error


class TestWebSearch:
    """Tests for web_search public function."""

    def test_formats_results(self):
        with patch("agentic.toolkit.websurf.fetch_search_results") as mock_raw:
            mock_raw.return_value = (
                [{"title": "T", "url": "https://u.com", "content": "c"}], None
            )
            result = web_search("query", 3)
            assert "Web search results for: query" in result
            assert "1. T" in result
            assert "https://u.com" in result

    def test_no_results(self):
        with patch("agentic.toolkit.websurf.fetch_search_results") as mock_raw:
            mock_raw.return_value = ([], None)
            result = web_search("nothing")
            assert "no results found" in result.lower()

    def test_search_failure_propagates(self):
        with patch("agentic.toolkit.websurf.fetch_search_results") as mock_raw:
            mock_raw.return_value = (None, "[search failed: connection failed]")
            result = web_search("query")
            assert "search failed" in result.lower()


class TestWebFetch:
    """Tests for web_fetch."""

    def test_rejects_non_http(self):
        result = web_fetch("ftp://example.com")
        assert "fetch failed" in result.lower()

    def test_rejects_private_ips(self):
        result = web_fetch("http://192.168.1.1")
        assert "not allowed" in result.lower()

    def test_rejects_localhost(self):
        result = web_fetch("http://localhost:8080")
        assert "not allowed" in result.lower()


class TestDeepResearch:
    """Tests for deep_research graph-based research."""

    def test_empty_query(self):
        result = deep_research("", embedder=FakeEmbedder())
        assert "search failed" in result.lower()
        result = deep_research("   ", embedder=FakeEmbedder())
        assert "search failed" in result.lower()

    def test_returns_finalize_content(self):
        mock_result = MagicMock()
        mock_result.results = (
            MagicMock(node_id="fetch_r", ok=True, content="evidence"),
            MagicMock(node_id="judge_r", ok=True, content="SUFFICIENT"),
            MagicMock(node_id="finalize", ok=True, content="Final synthesized answer"),
        )
        mock_result.final_answer = "fallback"

        with patch("agentic.toolkit.research.execute_graph", return_value=mock_result) as mock_exec:
            result = deep_research("quantum computing", embedder=FakeEmbedder())

        assert result == "Final synthesized answer"
        mock_exec.assert_called_once()
        graph = mock_exec.call_args[0][0]
        assert isinstance(graph, PlanGraph)
        assert graph.goal == "quantum computing"

    def test_falls_back_to_final_answer(self):
        mock_result = MagicMock()
        mock_result.results = (
            MagicMock(node_id="fetch_r", ok=True, content="evidence"),
        )
        mock_result.final_answer = "fallback answer"

        with patch("agentic.toolkit.research.execute_graph", return_value=mock_result):
            result = deep_research("query", embedder=FakeEmbedder())
        assert result == "fallback answer"

    def test_skips_failed_finalize(self):
        mock_result = MagicMock()
        mock_result.results = (
            MagicMock(node_id="finalize", ok=False, content="failed"),
        )
        mock_result.final_answer = "actual answer"

        with patch("agentic.toolkit.research.execute_graph", return_value=mock_result):
            result = deep_research("query", embedder=FakeEmbedder())
        assert result == "actual answer"

    def test_exception_during_execution(self):
        with patch("agentic.toolkit.research.execute_graph", side_effect=RuntimeError("graph error")):
            with pytest.raises(RuntimeError, match="graph error"):
                deep_research("query", embedder=FakeEmbedder())

    def test_tool_mode_omits_report_learn(self):
        mock_result = MagicMock()
        mock_result.results = (
            MagicMock(node_id="finalize", ok=True, content="result"),
        )
        mock_result.final_answer = ""

        with patch("agentic.toolkit.research.execute_graph", return_value=mock_result) as mock_exec:
            deep_research("query", embedder=FakeEmbedder(), tool_mode=True)

        graph = mock_exec.call_args[0][0]
        node_ids = [n.id for n in graph.nodes]
        assert "report" not in node_ids
        assert "learn" not in node_ids

    def test_default_includes_report_learn(self):
        mock_result = MagicMock()
        mock_result.results = (
            MagicMock(node_id="finalize", ok=True, content="result"),
        )
        mock_result.final_answer = ""

        with patch("agentic.toolkit.research.execute_graph", return_value=mock_result) as mock_exec:
            deep_research("query", embedder=FakeEmbedder())

        graph = mock_exec.call_args[0][0]
        node_ids = [n.id for n in graph.nodes]
        assert "report" in node_ids
        assert "learn" in node_ids

    def test_build_deep_research_subgraph_structure(self):
        graph = _build_deep_research_subgraph("test query", "sess123", max_rounds=5)
        assert isinstance(graph, PlanGraph)
        assert graph.id == "deep_research"
        assert graph.goal == "test query"
        node_ids = [n.id for n in graph.nodes]
        assert "fetch_r" in node_ids
        assert "judge_r" in node_ids
        assert "finalize" in node_ids
        assert "report" in node_ids
        assert "learn" in node_ids

    def test_build_deep_research_subgraph_tool_mode(self):
        graph = _build_deep_research_subgraph("q", "sess456", tool_mode=True)
        node_ids = [n.id for n in graph.nodes]
        assert "report" not in node_ids
        assert "learn" not in node_ids

    def test_build_deep_research_subgraph_caps_max_rounds(self):
        graph = _build_deep_research_subgraph("q", "sess789", max_rounds=999)
        judge = next(n for n in graph.nodes if n.id == "judge_r")
        assert judge.max_visits <= 5


class TestCondenseEvidence:
    """Tests for condense_evidence and _finalize_condensed."""

    def test_condense_evidence_basic(self):
        embedder = FakeEmbedder()
        pages = [("https://a.com", "This is a long page about quantum computing. " * 50)]
        result = condense_evidence(pages, "quantum", embedder=embedder)
        assert "Condensed evidence for: quantum" in result
        assert "relevant excerpt" in result.lower()

    def test_no_relevant_content(self):
        embedder = FakeEmbedder()
        pages = [("https://a.com", "Unrelated content about cats. " * 50)]
        result = condense_evidence(pages, "quantum", embedder=embedder)
        assert "no relevant content found" in result.lower()

    def test_corroboration_bonus(self):
        """_apply_corroboration_bonus boosts cross-domain agreement."""
        scored = [
            (0.5, "https://a.com", "quantum computing uses qubits"),
            (0.5, "https://b.org", "quantum computing uses qubits"),
            (0.3, "https://a.com", "different content"),
        ]
        boosted = _apply_corroboration_bonus(scored)
        # First two from different domains, similar content -> boosted
        assert boosted[0][0] > 0.5  # score increased
        assert boosted[1][0] > 0.5
        assert boosted[0][3] >= 2  # corroboration count
        # Third from same domain as first -> no boost
        assert boosted[2][0] == 0.3

    def test_deduplication_by_hash(self):
        """_finalize_condensed deduplicates by content hash."""
        scored = [
            (0.8, "https://a.com", "same content"),
            (0.7, "https://b.com", "same content"),  # duplicate
            (0.6, "https://c.com", "different content"),
        ]
        result = _finalize_condensed(scored, "query", top_k=5, min_score=0.1)
        # Should only have 2 unique entries
        assert result.count("same content") == 1
        assert "different content" in result

    def test_min_score_filter(self):
        """Chunks below min_score are dropped."""
        scored = [
            (0.05, "https://a.com", "low relevance"),
            (0.5, "https://b.com", "high relevance"),
        ]
        result = _finalize_condensed(scored, "query", min_score=0.1)
        assert "low relevance" not in result
        assert "high relevance" in result

    def test_top_k_limit(self):
        """Only top_k chunks returned."""
        scored = [(0.9 - i * 0.1, f"https://{i}.com", f"content {i}") for i in range(10)]
        result = _finalize_condensed(scored, "query", top_k=3)
        # Count source lines
        source_lines = [l for l in result.split("\n") if l.startswith("[source:")]
        assert len(source_lines) == 3


class TestFetchAndScorePipeline:
    """Tests for _fetch_and_score_pipeline."""

    def test_parallel_fetch(self):
        """Multiple URLs fetched in parallel."""
        urls = [f"https://example.com/{i}" for i in range(4)]

        def mock_fetch(url, max_chars=4000):
            time.sleep(0.01)  # Simulate network
            return f"Content from {url}"

        with patch("agentic.toolkit.websurf.web_fetch", side_effect=mock_fetch) as mock_fetch_fn:
            start = time.monotonic()
            scored, pages, outcomes = _fetch_and_score_pipeline(
                urls, "query", FakeEmbedder(), 1000, max_workers=4, fetch_fn=mock_fetch_fn
            )
            elapsed = time.monotonic() - start

        # With 4 workers, should take ~10ms not 40ms
        assert elapsed < 0.05
        assert len(pages) == 4
        assert len(scored) > 0

    def test_batch_prefetch_used(self):
        """batch_prefetch_fn called for Crawl4AI batch fetch."""
        urls = ["https://a.com", "https://b.com"]
        prefetched = {"https://a.com": "prefetched content"}

        def batch_fn(url_list, max_chars):
            return {u: prefetched.get(u, "") for u in url_list}

        with patch("agentic.toolkit.websurf.web_fetch", return_value="fallback content") as mock_fetch:
            scored, pages, outcomes = _fetch_and_score_pipeline(
                urls, "query", FakeEmbedder(), 1000, batch_prefetch_fn=batch_fn,
                fetch_fn=mock_fetch
            )
        # a.com should use prefetched, b.com fallback
        assert any("prefetched" in p[1] for p in pages)

    def test_failed_fetches_excluded(self):
        """Failed fetches don't produce scored chunks."""
        urls = ["https://good.com", "https://bad.com"]

        def mock_fetch(url, max_chars=4000):
            if "bad" in url:
                return "[fetch failed: connection error]"
            return "Good content here"

        with patch("agentic.toolkit.websurf.web_fetch", side_effect=mock_fetch) as mock_fetch_fn:
            scored, pages, outcomes = _fetch_and_score_pipeline(
                urls, "query", FakeEmbedder(), 1000, fetch_fn=mock_fetch_fn
            )

        # Only good.com should be in pages
        assert len(pages) == 1
        assert pages[0][0] == "https://good.com"


class TestScoreUrlChunks:
    """Tests for _score_url_chunks embedding vs keyword fallback."""

    def test_uses_embedder_when_available(self):
        embedder = FakeEmbedder()
        chunks = [("https://a.com", "quantum computing content")]
        scored = _score_url_chunks(chunks, "quantum", embedder, 10)
        assert len(scored) == 1
        assert scored[0][0] >= 0  # cosine similarity score

    def test_falls_back_to_keyword_overlap(self):
        """When embedder fails, uses keyword overlap."""
        class BadEmbedder:
            def embed_query(self, *a, **k):
                raise RuntimeError("embedder broken")

        chunks = [("https://a.com", "quantum computing uses qubits")]
        scored = _score_url_chunks(chunks, "quantum", BadEmbedder(), 10)
        assert len(scored) == 1
        # Keyword overlap should give some score
        assert scored[0][0] > 0

    def test_batch_embedding(self):
        """Multiple chunks embedded in single batch."""
        embedder = FakeEmbedder()
        chunks = [(f"https://{i}.com", f"content {i}") for i in range(5)]
        scored = _score_url_chunks(chunks, "query", embedder, 100)
        assert len(scored) == 5


class TestResearchEnvConfig:
    """Tests that env vars are read correctly."""

    def test_deep_research_defaults(self):
        assert DEEP_RESEARCH_NUM_SEARCHES >= 1
        assert DEEP_RESEARCH_NUM_FETCHES >= 1
        assert DEEP_RESEARCH_MAX_ROUNDS >= 1

    def test_condense_defaults(self):
        assert CONDENSE_CHUNK_CHARS == 500
        assert CONDENSE_TOP_K == 8
        assert CONDENSE_MIN_SCORE == 0.15


class TestIntegrationScenarios:
    """Integration-style tests with mocked external deps."""

    def test_deep_research_full_flow(self):
        """End-to-end deep_research with mocked graph execution."""
        mock_result = MagicMock()
        mock_result.results = (
            MagicMock(node_id="fetch_r", ok=True, content="Round evidence here"),
            MagicMock(node_id="judge_r", ok=True, content="SUFFICIENT"),
            MagicMock(node_id="finalize", ok=True, content="Final synthesized answer"),
        )
        mock_result.final_answer = "fallback"

        with patch("agentic.toolkit.research.execute_graph", return_value=mock_result):
            result = deep_research(
                "how does quantum computing work",
                embedder=FakeEmbedder(),
            )

        assert "Final synthesized answer" in result

if __name__ == "__main__":
    pytest.main([__file__, "-v"])