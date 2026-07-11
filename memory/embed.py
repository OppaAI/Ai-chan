"""
embed.py
───────────
llama.cpp-backed text embedder for harrier-oss-v1-270m (GGUF, via llama-server).

Replaces the previous ONNX Runtime implementation. The ONNX export of Harrier
uses the GroupQueryAttention contrib op, which has no CUDA kernel in onnxruntime
(confirmed via verbose EP logging) and no entry in the onnx-tensorrt operator
support matrix either — so every embed call fell back to CPU for attention,
dominating latency (~300ms/call on Jetson Orin Nano).

llama.cpp has native GQA kernels (same family used for Ministral/smollm in this
project) and runs Harrier via its own server with --pooling-type last, which
matches the model's required last-token pooling. This wrapper talks to that
server over HTTP instead of loading an ONNX graph directly.

Small iterable interface (unchanged from the ONNX version):
    embedder = HarrierEmbedder()
    vectors = list(embedder.embed(["text one", "text two"]))
    # or
    vectors = embedder.embed_batch(["text one", "text two"])  # returns np.ndarray (N, 640)

Query-side instruction prefix (set EMBED_QUERY_INSTRUCT in .env):
    Use embedder.embed_query("your query") for search queries.
    Use embedder.embed() / embed_batch() for document/memory storage (no prefix).

Used by memory.memorize and util.migrate_embeddings.

Requires the harrier llama-server instance to be running with:
    embedding = true
    pooling-type = last
"""

import os
import struct
from typing import Generator, Iterable

import numpy as np
import requests

# ── config from env ───────────────────────────────────────────────────────────
_EMBED_BASE_URL   = os.getenv("EMBED_BASE_URL", "http://127.0.0.1:8080")
_EMBED_MODEL      = os.getenv("EMBED_MODEL", "harrier")
_EMBED_DIMS       = int(os.getenv("EMBED_DIMS", "640"))
_BATCH_SIZE       = int(os.getenv("EMBED_BATCH_SIZE", "32"))
_EMBED_TIMEOUT    = float(os.getenv("EMBED_TIMEOUT_S", "30"))
_QUERY_INSTRUCT   = os.getenv(
    "EMBED_QUERY_INSTRUCT",
    "Retrieve relevant memories that answer the query",
)


class HarrierEmbedder:
    """
    HTTP-based text embedder for harrier-oss-v1-270m via llama-server.

    Talks to a running llama-server instance (started with `embedding = true`,
    `pooling-type = last`) over its /embedding endpoint. Connection is lazy —
    the first call just hits the HTTP endpoint, no local model loading happens
    in this process.
    """

    def __init__(
        self,
        base_url: str   = _EMBED_BASE_URL,
        model: str      = _EMBED_MODEL,
        dims: int       = _EMBED_DIMS,
        batch_size: int = _BATCH_SIZE,
        timeout: float  = _EMBED_TIMEOUT,
    ) -> None:
        self.base_url   = base_url.rstrip("/")
        self.model      = model
        self.dims       = dims
        self.batch_size = batch_size
        self.timeout    = timeout
        self._session   = requests.Session()

        # Concurrent calls hit this session from multiple threads at once
        # (route()'s ternary-routing embed_query call + the two CONTEXT_POOL
        # futures in _fetch_memory_and_knowledge). Default urllib3 pool size
        # is too small for that and transient RemoteDisconnected errors from
        # the backend aren't retried at all by default — add both.
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=10,
            max_retries=requests.adapters.Retry(
                total=2,
                connect=2,
                read=2,
                backoff_factor=0.2,
                status_forcelist=[502, 503, 504],
                allowed_methods=frozenset(["GET", "POST"]),
            ),
        )
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

    # ── core inference ────────────────────────────────────────────────────────

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of raw texts via llama-server's /embedding endpoint.
        Returns np.ndarray of shape (len(texts), dims), L2-normalised.
        """
        resp = self._session.post(
            f"{self.base_url}/embedding",
            json={"model": self.model, "content": texts},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        # llama-server returns a list of {"embedding": [...]} or {"embedding": [[...]]}
        # depending on version — handle both a flat vector and a nested batch-of-1 shape.
        vecs = []
        for item in data:
            emb = item["embedding"]
            if isinstance(emb[0], list):
                emb = emb[0]  # unwrap nested batch dimension some versions return
            vecs.append(emb)

        arr = np.asarray(vecs, dtype=np.float32)

        # L2 normalise defensively — llama-server's --pooling-type last gives the
        # last-token hidden state, but normalisation behaviour has varied across
        # versions, so we enforce it here to match the ONNX path's guarantee.
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return arr / norms

    # ── public API (unchanged from ONNX version) ────────────────────────────

    def embed(self, texts: Iterable[str]) -> Generator[np.ndarray, None, None]:
        """
        Embed documents (no instruction prefix).
        Yields one np.ndarray(dim,) per text.
        """
        texts = list(texts)
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            vecs  = self._embed_texts(batch)
            for v in vecs:
                yield v

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """
        Embed documents and return all vectors as np.ndarray (N, dims).
        """
        all_vecs = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            all_vecs.append(self._embed_texts(batch))
        return np.vstack(all_vecs)

    def embed_query(self, query: str, instruct: str = _QUERY_INSTRUCT) -> np.ndarray:
        """
        Embed a single search query with the instruction prefix.
        Returns np.ndarray(dims,).

        harrier query format (from model card):
            "Instruct: <task>\\nQuery: <query>"
        """
        prefixed = f"Instruct: {instruct}\nQuery: {query}"
        return self._embed_texts([prefixed])[0]

    def embed_queries(self, queries: list[str], instruct: str = _QUERY_INSTRUCT) -> np.ndarray:
        """
        Embed multiple search queries with the instruction prefix.
        Returns np.ndarray (N, dims).
        """
        prefixed = [f"Instruct: {instruct}\nQuery: {q}" for q in queries]
        return self.embed_batch(prefixed)

    # ── sqlite-vec serialisation helper ──────────────────────────────────────

    @staticmethod
    def serialize(vector: np.ndarray) -> bytes:
        """Serialise a float32 vector for sqlite-vec INSERT."""
        v = vector.astype(np.float32)
        return struct.pack(f"{len(v)}f", *v)
