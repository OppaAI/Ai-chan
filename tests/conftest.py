"""
tests/conftest.py

Shared pytest fixtures and configuration.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Ensure config is loaded before any module imports
os.environ.setdefault("WORKSPACE_ROOT", "/tmp/aiko_test_workspace")

sys.path.insert(0, "/home/oppa-ai/jetson")
from system.config import load_config
load_config()


@pytest.fixture(scope="session", autouse=True)
def setup_test_env():
    """Session-wide test environment setup."""
    # Create temp workspace
    workspace = Path(tempfile.mkdtemp(prefix="aiko_test_"))
    os.environ["WORKSPACE_ROOT"] = str(workspace)
    yield workspace
    # Cleanup
    import shutil
    shutil.rmtree(workspace, ignore_errors=True)


@pytest.fixture
def fake_embedder():
    """Deterministic embedder for tests."""
    class FakeEmbedder:
        def __init__(self, dim=384):
            self.dim = dim

        def embed_query(self, text: str, instruct: str = "") -> np.ndarray:
            h = hash(text + instruct) % 1000
            return np.array([float(h) / 1000.0] * self.dim, dtype=np.float32)

        def embed_batch(self, texts: list[str]) -> np.ndarray:
            return np.stack([self.embed_query(t) for t in texts])

    return FakeEmbedder()


@pytest.fixture
def mock_llm_client():
    """Mock LLM client with configurable responses."""
    class MockLLMClient:
        def __init__(self, responses: list[str] = None, latency_ms: float = 10):
            self.responses = responses or ["Mock response"]
            self.latency_ms = latency_ms
            self.call_count = 0
            self.idx = 0
            self.last_messages = None

        @property
        def chat(self):
            mock_chat = MagicMock()
            mock_chat.completions = MagicMock()
            mock_chat.completions.create = self._create
            return mock_chat

        def _create(self, model, messages, **kwargs):
            import time
            time.sleep(self.latency_ms / 1000.0)
            if self.idx < len(self.responses):
                resp = self.responses[self.idx]
            else:
                resp = self.responses[-1]
            self.idx += 1
            self.call_count += 1
            self.last_messages = messages
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock(message=MagicMock(content=resp))]
            return mock_resp

    return MockLLMClient


@pytest.fixture
def mock_owner(mock_llm_client, fake_embedder):
    """Mock AikoThink owner."""
    owner = MagicMock()
    owner._client = mock_llm_client()
    owner._llm_model = "test-model"
    owner._history = []
    owner._history_lock = MagicMock()
    owner._history_lock.__enter__ = MagicMock(return_value=None)
    owner._history_lock.__exit__ = MagicMock(return_value=False)
    owner._memorize = MagicMock()
    owner._memorize._mem = MagicMock()
    owner._memorize._mem._embedder = fake_embedder
    owner._store_async = MagicMock()
    owner._emit = MagicMock()
    return owner


@pytest.fixture
def temp_workspace(tmp_path):
    """Temporary workspace directory."""
    os.environ["WORKSPACE_ROOT"] = str(tmp_path)
    return tmp_path


# ─── Test Data Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def seeded_knowledge_db(tmp_path, fake_embedder):
    """Create a knowledge DB with test data."""
    db_path = tmp_path / "knowledge.db"
    conn = _connect(str(db_path))

    import sqlite_vec
    now = "2024-01-01T00:00:00"

    # Insert test documents
    docs = [
        ("doc-1", "Quantum Computing", "Quantum computing uses qubits. Qubits can be in superposition and entanglement."),
        ("doc-2", "Classical Computing", "Classical computers use bits. Bits are either 0 or 1. No superposition."),
        ("doc-3", "Machine Learning", "Machine learning trains models on data. Neural networks are a common approach."),
        ("doc-4", "Quantum Algorithms", "Shor's algorithm factors integers. Grover's algorithm searches unstructured databases."),
    ]

    for doc_id, title, text in docs:
        conn.execute(
            "INSERT INTO learned_docs (id, user_id, title, source, kind, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (doc_id, "test_user", title, "test", "ingested", now)
        )
        chunks = [text[i:i+100] for i in range(0, len(text), 80)]
        for j, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}-chunk-{j}"
            vec = fake_embedder.embed_query(chunk)
            conn.execute(
                "INSERT INTO learned_chunks (id, doc_id, chunk_index, text, created_at) VALUES (?, ?, ?, ?, ?)",
                (chunk_id, doc_id, j, chunk, now)
            )
            conn.execute(
                "INSERT INTO learned_chunks_vec (id, embedding) VALUES (?, ?)",
                (chunk_id, sqlite_vec.serialize_float32(vec.tolist()))
            )
            conn.execute(
                "INSERT INTO learned_chunks_fts (id, text) VALUES (?, ?)",
                (chunk_id, chunk)
            )

    conn.commit()
    yield conn
    conn.close()


# ─── Common Mock Patches ─────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def mock_external_services():
    """Auto-mock external services to prevent network calls in tests."""
    # Mock requests module at the system level so any import gets the mock
    import requests
    import sys
    original_requests = sys.modules.get('requests')
    sys.modules['requests'] = MagicMock()
    sys.modules['requests'].get.return_value = MagicMock(
        status_code=200,
        json=lambda: {"results": [{"title": "Test", "url": "https://example.com", "content": "Test"}]}
    )
    yield
    if original_requests:
        sys.modules['requests'] = original_requests
    else:
        del sys.modules['requests']


# ─── Import helpers (avoid circular imports) ──────────────────────────────────

def _connect(db_path: str):
    """Import connect function at runtime to avoid circular imports."""
    from memory.knowledge import _connect as _connect_fn
    return _connect_fn(db_path)