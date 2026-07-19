"""
tests/unit/test_agentic_research.py

Unit tests for agentic/toolkit/research.py — web search, deep_search, deep_research.

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

from agentic.toolkit.research import (
    web_search,
    web_fetch,
    deep_search,
    deep_research,
    condense_evidence,
    read_paper_url,
    _web_search_raw,
    _score_url_chunks,
    _finalize_condensed,
    _apply_corroboration_bonus,
    _fetch_and_score_pipeline,
    _deep_search_impl,
    _ask_llm_json,
    DEEP_SEARCH_NUM_FETCHES,
    DEEP_SEARCH_NUM_SEARCHES,
    DEEP_SEARCH_MAX_CHARS_PER_PAGE,
    DEEP_RESEARCH_NUM_FETCHES,
    DEEP_RESEARCH_NUM_SEARCHES,
    DEEP_RESEARCH_MAX_CHARS_PER_PAGE,
    DEEP_RESEARCH_MAX_ROUNDS,
    CONDENSE_CHUNK_CHARS,
    CONDENSE_TOP_K,
    CONDENSE_MIN_SCORE,
)


class FakeEmbedder:
    """Deterministic embedder for tests."""
    def embed_query(self, text: str, instruct: str = "") -> np.ndarray:
        h = hash(text) % 1000
        return np.array([float(h) / 1000.0] * 384, dtype=np.float32)

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


class TestWebSearchRaw:
    """Tests for _web_search_raw low-level SearXNG call."""

    def test_successful_search(self):
        with patch("agentic.toolkit.research.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "results": [
                    {"title": "Test", "url": "https://example.com", "content": "test content"}
                ]
            }
            mock_get.return_value = mock_resp

            results, error = _web_search_raw("test query", 5)
            assert error is None
            assert len(results) == 1
            assert results[0]["title"] == "Test"

    def test_rate_limit_retry(self):
        with patch("agentic.toolkit.research.requests.get") as mock_get:
            # First two calls rate limited, third succeeds
            mock_resp_429 = MagicMock()
            mock_resp_429.status_code = 429
            mock_resp_ok = MagicMock()
            mock_resp_ok.status_code = 200
            mock_resp_ok.json.return_value = {"results": [{"title": "OK", "url": "https://ok.com", "content": "ok"}]}
            mock_get.side_effect = [mock_resp_429, mock_resp_429, mock_resp_ok]

            results, error = _web_search_raw("test", 5)
            assert error is None
            assert len(results) == 1
            assert mock_get.call_count == 3

    def test_connection_error_retry(self):
        import requests
        with patch("agentic.toolkit.research.requests.get") as mock_get:
            mock_get.side_effect = [
                requests.exceptions.ConnectionError("conn error"),
                MagicMock(status_code=200, json=lambda: {"results": []}),
            ]
            results, error = _web_search_raw("test", 5)
            assert error is None
            assert mock_get.call_count == 2

    def test_invalid_json(self):
        with patch("agentic.toolkit.research.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.side_effect = ValueError("bad json")
            mock_get.return_value = mock_resp

            results, error = _web_search_raw("test", 5)
            assert "invalid JSON" in error


class TestWebSearch:
    """Tests for web_search public function."""

    def test_formats_results(self):
        with patch("agentic.toolkit.research._web_search_raw") as mock_raw:
            mock_raw.return_value = (
                [{"title": "T", "url": "https://u.com", "content": "c"}], None
            )
            result = web_search("query", 3)
            assert "Web search results for: query" in result
            assert "1. T" in result
            assert "https://u.com" in result

    def test_no_results(self):
        with patch("agentic.toolkit.research._web_search_raw") as mock_raw:
            mock_raw.return_value = ([], None)
            result = web_search("nothing")
            assert "no results found" in result.lower()

    def test_search_failure_propagates(self):
        with patch("agentic.toolkit.research._web_search_raw") as mock_raw:
            mock_raw.return_value = (None, "connection failed")
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


class TestDeepSearch:
    """Tests for deep_search snippet-only search."""

    def test_snippet_only_by_default(self):
        """DEEP_SEARCH_NUM_FETCHES=0 means snippet-only."""
        assert DEEP_SEARCH_NUM_FETCHES == 0

        with patch("agentic.toolkit.research._web_search_raw") as mock_raw:
            mock_raw.return_value = ([{"title": "T", "url": "https://u.com", "content": "c"}], None)
            result = deep_search("query", embedder=FakeEmbedder())
            assert "Web search results for: query" in result
            # Should NOT have manifest or condensed sections
            assert "URL manifest" not in result
            assert "Condensed evidence" not in result

    def test_with_fetches_if_configured(self):
        """If DEEP_SEARCH_NUM_FETCHES > 0, fetches pages."""
        with patch.dict("os.environ", {"DEEP_SEARCH_NUM_FETCHES": "2"}):
            # Need to reload module for env change - skip for now
            pass


class TestDeepResearch:
    """Tests for deep_research multi-round adaptive research."""

    def test_requires_client_and_model_for_adaptive(self):
        """Without client/model, runs single round only."""
        with patch("agentic.toolkit.research._deep_search_impl") as mock_impl:
            mock_impl.return_value = ("results", set())
            result = deep_research("query", client=None, model=None, embedder=FakeEmbedder())
            # Should only call once (max_rounds=1 when not adaptive)
            assert mock_impl.call_count == 1

    def test_adaptive_continues_with_client(self):
        """With client/model, runs adaptive rounds."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            # Decision: continue
            MagicMock(choices=[MagicMock(message=MagicMock(content='{"continue": true, "next_query": "refined query", "reason": "need more"}'))]),
            # Decision: stop
            MagicMock(choices=[MagicMock(message=MagicMock(content='{"continue": false, "next_query": "", "reason": "enough"}'))]),
        ]

        with patch("agentic.toolkit.research._deep_search_impl") as mock_impl:
            mock_impl.return_value = ("round results", {"https://example.com"})
            result = deep_research("query", client=mock_client, model="test", embedder=FakeEmbedder(), max_rounds=3)
            # Should have called _deep_search_impl twice (2 rounds)
            assert mock_impl.call_count == 2

    def test_synthesis_called_when_evidence_exists(self):
        """When adaptive and has evidence, calls LLM for synthesis."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            # Decision: stop after 1 round
            MagicMock(choices=[MagicMock(message=MagicMock(content='{"continue": false, "next_query": "", "reason": "done"}'))]),
            # Synthesis
            MagicMock(choices=[MagicMock(message=MagicMock(content="Synthesized answer"))]),
        ]

        with patch("agentic.toolkit.research._deep_search_impl") as mock_impl:
            mock_impl.return_value = ("good evidence here", {"https://example.com"})
            result = deep_research("query", client=mock_client, model="test", embedder=FakeEmbedder())
            assert "Synthesized answer" in result
            assert "[Synthesis]" in result

    def test_synthesis_failure_fallbacks_to_raw(self):
        """If synthesis LLM call fails, returns raw rounds."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            # Decision: stop
            MagicMock(choices=[MagicMock(message=MagicMock(content='{"continue": false}'))]),
            # Synthesis: fails
            Exception("LLM error"),
        ]

        with patch("agentic.toolkit.research._deep_search_impl") as mock_impl:
            mock_impl.return_value = ("evidence", {"https://example.com"})
            result = deep_research("query", client=mock_client, model="test", embedder=FakeEmbedder())
            # Should return raw rounds log without synthesis
            assert "[Synthesis]" not in result
            assert "Round 1" in result

    def test_empty_results_handled(self):
        with patch("agentic.toolkit.research._deep_search_impl") as mock_impl:
            mock_impl.return_value = ("[search failed: ...]", set())
            result = deep_research("query", embedder=FakeEmbedder())
            assert "no results found" in result.lower()


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

        with patch("agentic.toolkit.research.web_fetch", side_effect=mock_fetch):
            start = time.monotonic()
            scored, pages, outcomes = _fetch_and_score_pipeline(
                urls, "query", FakeEmbedder(), 1000, max_workers=4
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

        with patch("agentic.toolkit.research.web_fetch") as mock_fetch:
            mock_fetch.return_value = "fallback content"
            scored, pages, outcomes = _fetch_and_score_pipeline(
                urls, "query", FakeEmbedder(), 1000, batch_prefetch_fn=batch_fn
            )
        # a.com should use prefetched, b.com fallback
        assert any("prefetched" in p[1] for p in pages)

    def test_failed_fetches_excluded(self):
        """Failed fetches don't produce scored chunks."""
        urls = ["https://good.com", "https://bad.com"]

        def mock_fetch(url, **kwargs):
            if "bad" in url:
                return "[fetch failed: connection error]"
            return "Good content here"

        with patch("agentic.toolkit.research.web_fetch", side_effect=mock_fetch):
            scored, pages, outcomes = _fetch_and_score_pipeline(
                urls, "query", FakeEmbedder(), 1000
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


class TestReadPaperUrl:
    """Tests for read_paper_url."""

    def test_without_query_returns_full(self):
        with patch("agentic.toolkit.research.web_fetch") as mock_fetch:
            mock_fetch.return_value = "Full paper content here"
            result = read_paper_url("https://arxiv.org/abs/1234.5678")
            assert "Full paper content here" in result

    def test_with_query_condenses(self):
        with patch("agentic.toolkit.research.web_fetch") as mock_fetch:
            mock_fetch.return_value = "Long paper about quantum. " * 100
            with patch("agentic.toolkit.research.condense_evidence") as mock_condense:
                mock_condense.return_value = "Condensed for query"
                result = read_paper_url("https://arxiv.org/abs/1234.5678", query="quantum")
                assert "Condensed for query" in result


class TestResearchEnvConfig:
    """Tests that env vars are read correctly."""

    def test_deep_search_defaults(self):
        assert DEEP_SEARCH_NUM_SEARCHES == 1
        assert DEEP_SEARCH_NUM_FETCHES == 0
        assert DEEP_SEARCH_MAX_CHARS_PER_PAGE == 2000

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
        """End-to-end deep_research with mocked SearXNG and fetch."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            # Decision: stop after 1 round
            MagicMock(choices=[MagicMock(message=MagicMock(content='{"continue": false}'))]),
            # Synthesis
            MagicMock(choices=[MagicMock(message=MagicMock(content="Final synthesized answer"))]),
        ]

        with patch("agentic.toolkit.research._web_search_raw") as mock_search:
            mock_search.return_value = ([
                {"title": "Q Computing", "url": "https://qc.com", "content": "Quantum computing uses qubits"}
            ], None)

            with patch("agentic.toolkit.research.web_fetch") as mock_fetch:
                mock_fetch.return_value = "Full page: quantum computing uses qubits for superposition."

                result = deep_research(
                    "how does quantum computing work",
                    client=mock_client,
                    model="test",
                    embedder=FakeEmbedder(),
                    max_rounds=2,
                )

        assert "Final synthesized answer" in result
        assert "[Synthesis]" in result
        assert "Round 1" in result

    def test_deep_search_snippet_only(self):
        """deep_search returns only snippets by default."""
        with patch("agentic.toolkit.research._web_search_raw") as mock_search:
            mock_search.return_value = ([
                {"title": "R1", "url": "https://r1.com", "content": "content 1"},
                {"title": "R2", "url": "https://r2.com", "content": "content 2"},
            ], None)

            result = deep_search("test query", embedder=FakeEmbedder())
            assert "Web search results for: test query" in result
            assert "1. R1" in result
            assert "2. R2" in result
            # No fetch/condense sections
            assert "URL manifest" not in result
            assert "Condensed evidence" not in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])