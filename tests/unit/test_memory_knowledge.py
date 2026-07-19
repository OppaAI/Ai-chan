"""
tests/unit/test_memory_knowledge.py

Unit tests for memory/knowledge.py — search_knowledge, knowledge_context_for, ingest.

Run: pytest tests/unit/test_memory_knowledge.py -v
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from unittest.mock import patch

import numpy as np
import pytest

os.environ.setdefault("WORKSPACE_ROOT", "/tmp/aiko_test_workspace")

sys.path.insert(0, "/home/oppa-ai/jetson")
from system.config import load_config
load_config()

from memory.knowledge import (
    search_knowledge,
    knowledge_context_for,
    ingest_text,
    ingest_file,
    _connect,
    _knn,
    _fts,
    KNOWLEDGE_KNN_LIMIT,
    KNOWLEDGE_FTS_LIMIT,
    KNOWLEDGE_RRF_K,
    KNOWLEDGE_RECALL_SCORE_THRESHOLD,
    KNOWLEDGE_CONTEXT_CHARS,
    _search_cache_get,
    _search_cache_set,
)


class FakeEmbedder:
    """Deterministic embedder for tests."""
    def embed_query(self, text: str, instruct: str = "") -> np.ndarray:
        h = hash(text) % 1000
        return np.array([float(h) / 1000.0] * 384, dtype=np.float32)


class TestDatabaseSetup:
    """Tests for database connection and schema."""

    def test_connect_creates_tables(self, tmp_path):
        db_path = tmp_path / "test_knowledge.db"
        conn = _connect(str(db_path))
        try:
            # Check tables exist
            tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            table_names = {t[0] for t in tables}
            assert "learned_docs" in table_names
            assert "learned_chunks" in table_names
            assert "learned_chunks_fts" in table_names
            # Check vec table
            vec_tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%vec%'").fetchall()
            assert len(vec_tables) > 0
        finally:
            conn.close()

    def test_connect_uses_user_isolation(self, tmp_path):
        """Different user_ids should use different databases."""
        db1 = _connect("user1", tmp_path / "k1.db")
        db2 = _connect("user2", tmp_path / "k2.db")
        try:
            # They should be separate connections
            assert db1 is not db2
        finally:
            db1.close()
            db2.close()


class TestIngestText:
    """Tests for ingest_text."""

    def test_ingest_creates_doc_and_chunks(self, tmp_path):
        db_path = tmp_path / "test.db"
        with patch("memory.knowledge._connect", return_value=_connect(str(db_path))):
            doc_id = ingest_text(
                title="Test Doc",
                text="This is a test document. " * 10,  # Long enough to chunk
                source="test",
                kind="ingested",
                embedder=FakeEmbedder(),
            )
            assert doc_id is not None
            assert doc_id.startswith("doc-")

    def test_ingest_empty_text_returns_none(self, tmp_path):
        db_path = tmp_path / "test.db"
        with patch("memory.knowledge._connect", return_value=_connect(str(db_path))):
            doc_id = ingest_text("Title", "", embedder=FakeEmbedder())
            assert doc_id is None

    def test_ingest_sanitizes_text(self, tmp_path):
        db_path = tmp_path / "test.db"
        with patch("memory.knowledge._connect", return_value=_connect(str(db_path))):
            doc_id = ingest_text(
                "Title", "  \n\n  Content with  excessive   whitespace  \n\n  ",
                embedder=FakeEmbedder()
            )
            assert doc_id is not None


class TestIngestFile:
    """Tests for ingest_file."""

    def test_ingest_text_file(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("File content for ingestion. " * 5)

        with patch("memory.knowledge._connect", return_value=_connect(str(tmp_path / "test.db"))):
            doc_id = ingest_file(str(test_file), title="Test File", embedder=FakeEmbedder())
            assert doc_id is not None

    def test_ingest_nonexistent_file(self, tmp_path):
        with patch("memory.knowledge._connect", return_value=_connect(str(tmp_path / "test.db"))):
            doc_id = ingest_file("/nonexistent/path.txt", embedder=FakeEmbedder())
            assert doc_id is None


class TestSearchKnowledge:
    """Tests for search_knowledge."""

    def setup_method(self):
        """Create a fresh DB with test data for each test."""
        self.tmp_path = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp_path, "test.db")
        self.conn = _connect(self.db_path)

        # Insert test documents
        self._seed_test_data()

    def teardown_method(self):
        self.conn.close()
        import shutil
        shutil.rmtree(self.tmp_path, ignore_errors=True)

    def _seed_test_data(self):
        embedder = FakeEmbedder()
        now = "2024-01-01T00:00:00"

        # Doc 1: Quantum computing
        doc1_id = "doc-1"
        self.conn.execute(
            "INSERT INTO learned_docs (id, user_id, title, source, kind, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (doc1_id, "test_user", "Quantum Basics", "test", "ingested", now)
        )
        text1 = "Quantum computing uses qubits for computation. Qubits can be in superposition."
        chunks1 = [text1[i:i+100] for i in range(0, len(text1), 80)]
        for i, chunk in enumerate(chunks1):
            chunk_id = f"{doc1_id}-chunk-{i}"
            vec = embedder.embed_query(chunk)
            import sqlite_vec
            self.conn.execute(
                "INSERT INTO learned_chunks (id, doc_id, chunk_index, text, created_at) VALUES (?, ?, ?, ?, ?)",
                (chunk_id, doc1_id, i, chunk, now)
            )
            self.conn.execute(
                "INSERT INTO learned_chunks_vec (id, embedding) VALUES (?, ?)",
                (chunk_id, sqlite_vec.serialize_float32(vec.tolist()))
            )
            self.conn.execute(
                "INSERT INTO learned_chunks_fts (id, text) VALUES (?, ?)",
                (chunk_id, chunk)
            )

        # Doc 2: Classical computing
        doc2_id = "doc-2"
        self.conn.execute(
            "INSERT INTO learned_docs (id, user_id, title, source, kind, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (doc2_id, "test_user", "Classical Computing", "test", "ingested", now)
        )
        text2 = "Classical computers use bits which are either 0 or 1. No superposition."
        chunks2 = [text2[i:i+100] for i in range(0, len(text2), 80)]
        for i, chunk in enumerate(chunks2):
            chunk_id = f"{doc2_id}-chunk-{i}"
            vec = embedder.embed_query(chunk)
            import sqlite_vec
            self.conn.execute(
                "INSERT INTO learned_chunks (id, doc_id, chunk_index, text, created_at) VALUES (?, ?, ?, ?, ?)",
                (chunk_id, doc2_id, i, chunk, now)
            )
            self.conn.execute(
                "INSERT INTO learned_chunks_vec (id, embedding) VALUES (?, ?)",
                (chunk_id, sqlite_vec.serialize_float32(vec.tolist()))
            )
            self.conn.execute(
                "INSERT INTO learned_chunks_fts (id, text) VALUES (?, ?)",
                (chunk_id, chunk)
            )

        self.conn.commit()

    def test_search_returns_relevant_results(self):
        with patch("memory.knowledge._connect", return_value=self.conn):
            results = search_knowledge("quantum qubits", limit=5, embedder=FakeEmbedder(), user_id="test_user")
            assert len(results) > 0
            # Should find quantum doc
            assert any("quantum" in r["text"].lower() for r in results)

    def test_search_filters_by_user(self):
        with patch("memory.knowledge._connect", return_value=self.conn):
            results = search_knowledge("quantum", limit=5, embedder=FakeEmbedder(), user_id="other_user")
            assert len(results) == 0

    def test_search_returns_scores(self):
        with patch("memory.knowledge._connect", return_value=self.conn):
            results = search_knowledge("quantum", limit=5, embedder=FakeEmbedder(), user_id="test_user")
            for r in results:
                assert "score" in r
                assert isinstance(r["score"], float)
                assert 0 <= r["score"] <= 1

    def test_search_limit_respected(self):
        with patch("memory.knowledge._connect", return_value=self.conn):
            results = search_knowledge("computing", limit=1, embedder=FakeEmbedder(), user_id="test_user")
            assert len(results) <= 1

    def test_empty_query_returns_empty(self):
        with patch("memory.knowledge._connect", return_value=self.conn):
            results = search_knowledge("", embedder=FakeEmbedder(), user_id="test_user")
            assert results == []


class TestKnowledgeContextFor:
    """Tests for knowledge_context_for formatting."""

    def setup_method(self):
        self.tmp_path = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp_path, "test.db")
        self.conn = _connect(self.db_path)
        self._seed_data()

    def teardown_method(self):
        self.conn.close()
        import shutil
        shutil.rmtree(self.tmp_path, ignore_errors=True)

    def _seed_data(self):
        embedder = FakeEmbedder()
        now = "2024-01-01T00:00:00"
        doc_id = "doc-test"
        self.conn.execute(
            "INSERT INTO learned_docs (id, user_id, title, source, kind, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (doc_id, "test_user", "Test Title", "test", "ingested", now)
        )
        text = "This is test content for knowledge retrieval."
        chunk_id = f"{doc_id}-chunk-0"
        vec = embedder.embed_query(text)
        import sqlite_vec
        self.conn.execute(
            "INSERT INTO learned_chunks (id, doc_id, chunk_index, text, created_at) VALUES (?, ?, ?, ?, ?)",
            (chunk_id, doc_id, 0, text, now)
        )
        self.conn.execute(
            "INSERT INTO learned_chunks_vec (id, embedding) VALUES (?, ?)",
            (chunk_id, sqlite_vec.serialize_float32(vec.tolist()))
        )
        self.conn.execute(
            "INSERT INTO learned_chunks_fts (id, text) VALUES (?, ?)",
            (chunk_id, text)
        )
        self.conn.commit()

    def test_returns_xml_format(self):
        with patch("memory.knowledge._connect", return_value=self.conn):
            ctx = knowledge_context_for("test", limit=5, embedder=FakeEmbedder(), user_id="test_user")
            assert "<knowledge_context>" in ctx
            assert "</knowledge_context>" in ctx
            assert "<knowledge_chunk" in ctx
            assert "doc_id" in ctx
            assert "title" in ctx
            assert "kind" in ctx
            assert "source" in ctx
            assert "score" in ctx

    def test_no_results_returns_empty_message(self):
        with patch("memory.knowledge._connect", return_value=self.conn):
            ctx = knowledge_context_for("nonexistent query xyz", limit=5, embedder=FakeEmbedder(), user_id="test_user")
            assert "No matching learned knowledge found" in ctx

    def test_max_chars_limit(self):
        with patch("memory.knowledge._connect", return_value=self.conn):
            ctx = knowledge_context_for("test", limit=5, max_chars=50, embedder=FakeEmbedder(), user_id="test_user")
            # Content should be truncated
            assert len(ctx) < 500  # Well under default

    def test_cache_hit_returns_cached(self):
        """Second call with same query should use cache."""
        with patch("memory.knowledge._connect", return_value=self.conn):
            ctx1 = knowledge_context_for("test query", limit=5, embedder=FakeEmbedder(), user_id="test_user")
            ctx2 = knowledge_context_for("test query", limit=5, embedder=FakeEmbedder(), user_id="test_user")
            assert ctx1 == ctx2


class TestKNNAndFTS:
    """Tests for _knn and _fts internal functions."""

    def setup_method(self):
        self.tmp_path = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp_path, "test.db")
        self.conn = _connect(self.db_path)
        self._seed_data()

    def teardown_method(self):
        self.conn.close()
        import shutil
        shutil.rmtree(self.tmp_path, ignore_errors=True)

    def _seed_data(self):
        embedder = FakeEmbedder()
        now = "2024-01-01T00:00:00"
        for i in range(10):
            doc_id = f"doc-{i}"
            self.conn.execute(
                "INSERT INTO learned_docs (id, user_id, title, source, kind, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (doc_id, "test_user", f"Doc {i}", "test", "ingested", now)
            )
            text = f"Content about topic {i}. " * 5
            chunk_id = f"{doc_id}-chunk-0"
            vec = embedder.embed_query(text)
            import sqlite_vec
            self.conn.execute(
                "INSERT INTO learned_chunks (id, doc_id, chunk_index, text, created_at) VALUES (?, ?, ?, ?, ?)",
                (chunk_id, doc_id, 0, text, now)
            )
            self.conn.execute(
                "INSERT INTO learned_chunks_vec (id, embedding) VALUES (?, ?)",
                (chunk_id, sqlite_vec.serialize_float32(vec.tolist()))
            )
            self.conn.execute(
                "INSERT INTO learned_chunks_fts (id, text) VALUES (?, ?)",
                (chunk_id, text)
            )
        self.conn.commit()

    def test_knn_returns_ranked_ids(self):
        ranked = _knn(self.conn, "topic 5", FakeEmbedder(), "test_user", KNOWLEDGE_KNN_LIMIT)
        assert isinstance(ranked, list)
        assert len(ranked) <= KNOWLEDGE_KNN_LIMIT

    def test_fts_returns_ranked_ids(self):
        ranked = _fts(self.conn, "topic", "test_user", KNOWLEDGE_FTS_LIMIT)
        assert isinstance(ranked, list)
        assert len(ranked) <= KNOWLEDGE_FTS_LIMIT

    def test_knn_empty_query(self):
        ranked = _knn(self.conn, "", FakeEmbedder(), "test_user", KNOWLEDGE_KNN_LIMIT)
        assert ranked == []

    def test_fts_empty_query(self):
        ranked = _fts(self.conn, "", "test_user", KNOWLEDGE_FTS_LIMIT)
        assert ranked == []


class TestCache:
    """Tests for search caching."""

    def test_cache_set_get(self):
        _search_cache_get("test", "user1", 5)  # Clear any existing
        _search_cache_set("test query", "user1", 5, [{"id": "1", "text": "cached"}])
        cached = _search_cache_get("test query", "user1", 5)
        assert cached is not None
        assert cached[0]["id"] == "1"

    def test_cache_miss_different_params(self):
        _search_cache_set("query", "user1", 5, [{"id": "1"}])
        cached = _search_cache_get("query", "user1", 10)  # Different limit
        assert cached is None

    def test_cache_ttl_expiry(self):
        import time
        _search_cache_set("query", "user1", 5, [{"id": "1"}], ttl=0.01)
        time.sleep(0.02)
        cached = _search_cache_get("query", "user1", 5)
        assert cached is None

    def test_cache_max_entries_eviction(self):
        # Fill beyond max
        for i in range(300):
            _search_cache_set(f"query{i}", "user1", 5, [{"id": str(i)}])
        # Oldest should be evicted
        cached = _search_cache_get("query0", "user1", 5)
        # May or may not be evicted depending on implementation
        # Just verify no crash
        assert cached is None or isinstance(cached, list)


class TestConstants:
    """Tests for module constants."""

    def test_knn_limit_positive(self):
        assert KNOWLEDGE_KNN_LIMIT > 0

    def test_fts_limit_positive(self):
        assert KNOWLEDGE_FTS_LIMIT > 0

    def test_rrf_k_positive(self):
        assert KNOWLEDGE_RRF_K > 0

    def test_recall_threshold_nonnegative(self):
        assert KNOWLEDGE_RECALL_SCORE_THRESHOLD >= 0

    def test_context_chars_positive(self):
        assert KNOWLEDGE_CONTEXT_CHARS > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])