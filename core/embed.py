"""
embed.py
───────────
Drop-in ONNX embedder for ferrisS/harrier-oss-v1-270m-fastembed.

Replaces fastembed.TextEmbedding for harrier because fastembed's custom
model API only supports MEAN/CLS pooling. Harrier is a decoder-only
Gemma3 model that requires last-token pooling.  This wrapper uses
onnxruntime + tokenizers directly and applies correct last-token pooling
+ L2 normalisation.

Small iterable interface:
    embedder = HarrierEmbedder()
    vectors = list(embedder.embed(["text one", "text two"]))
    # or
    vectors = embedder.embed_batch(["text one", "text two"])  # returns np.ndarray (N, 640)

Query-side instruction prefix (set EMBED_QUERY_INSTRUCT in .env):
    Use embedder.embed_query("your query") for search queries.
    Use embedder.embed() / embed_batch() for document/memory storage (no prefix).

Used by core.memorize and util.migrate_embeddings.
"""

import os
import struct
from pathlib import Path
from typing import Generator, Iterable

import numpy as np

# ── config from env ───────────────────────────────────────────────────────────
_EMBED_CACHE      = os.getenv("EMBED_CACHE_PATH") or os.getenv("FASTEMBED_CACHE_PATH") or str(Path.home() / ".cache" / "huggingface" / "hub")
_MODEL_ID         = os.getenv("EMBED_MODEL", "ferrisS/harrier-oss-v1-270m-fastembed")
_MODEL_FILE       = os.getenv("EMBED_MODEL_FILE", "model_quantized.onnx")
_EMBED_DIMS       = int(os.getenv("EMBED_DIMS", "640"))
_BATCH_SIZE       = int(os.getenv("EMBED_BATCH_SIZE", "64"))
_QUERY_INSTRUCT   = os.getenv(
    "EMBED_QUERY_INSTRUCT",
    "Retrieve relevant memories that answer the query",
)

# ── snapshot resolution ───────────────────────────────────────────────────────

def _find_snapshot(cache_dir: str, model_id: str) -> Path:
    """
    Resolve model snapshot — checks HF hub cache first, then the configured cache.
    HF hub layout: ~/.cache/huggingface/hub/models--<org>--<name>/snapshots/<hash>/
    legacy cache layout: ~/.cache/fastembed/models--<org>--<name>/snapshots/<hash>/
    """
    folder = "models--" + model_id.replace("/", "--")

    # prefer HF hub cache (where hf download put it)
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    for base in [hf_cache, Path(cache_dir)]:
        snapshots = base / folder / "snapshots"
        if snapshots.exists():
            hashes = sorted(snapshots.iterdir())
            if hashes:
                return hashes[0]

    raise FileNotFoundError(
        f"Harrier snapshot not found in HF hub cache ({hf_cache / folder}) "
        f"or configured embedding cache ({Path(cache_dir) / folder}). "
        f"Run: hf download {model_id} model_quantized.onnx model_quantized.onnx_data tokenizer.json tokenizer_config.json special_tokens_map.json config.json"
    )


# ── main embedder ─────────────────────────────────────────────────────────────

class HarrierEmbedder:
    """
    ONNX-based text embedder for harrier-oss-v1-270m with last-token pooling.

    Loads the model once on first use (lazy init) and caches the session
    for the lifetime of the object — safe to reuse across calls.
    """

    def __init__(
        self,
        cache_dir: str  = _EMBED_CACHE,
        model_id: str   = _MODEL_ID,
        model_file: str = _MODEL_FILE,
        dims: int       = _EMBED_DIMS,
        batch_size: int = _BATCH_SIZE,
    ) -> None:
        self._cache_dir   = cache_dir
        self._model_id    = model_id
        self._model_file  = model_file
        self.dims         = dims
        self.batch_size   = batch_size
        self._session     = None   # lazy — loaded on first embed call
        self._tokenizer   = None   # lazy — loaded on first embed call

    # ── lazy init ─────────────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        """Load ONNX session and tokenizer on first use."""
        if self._session is not None:
            return

        import onnxruntime as ort
        from tokenizers import Tokenizer

        snapshot = _find_snapshot(self._cache_dir, self._model_id)
        onnx_path     = snapshot / self._model_file
        tokenizer_path = snapshot / "tokenizer.json"

        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

        # prefer CUDA EP if available, fall back to CPU
        available = ort.get_available_providers()
        providers  = ["CUDAExecutionProvider"] if "CUDAExecutionProvider" in available else ["CPUExecutionProvider"]

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session   = ort.InferenceSession(str(onnx_path), sess_opts, providers=providers)
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))

        # enable padding + truncation for stable batched ONNX inputs
        from tokenizers import processors
        self._tokenizer.enable_padding(pad_id=0, pad_token="<pad>")
        self._tokenizer.enable_truncation(max_length=32768)

    # ── pooling ───────────────────────────────────────────────────────────────

    @staticmethod
    def _last_token_pool(hidden_states: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """
        Extract the last non-padding token embedding per sequence and L2-normalise.

        hidden_states : (batch, seq_len, hidden_dim)
        attention_mask: (batch, seq_len)  — 1 = real token, 0 = pad
        returns       : (batch, hidden_dim)  L2-normalised
        """
        batch_size = hidden_states.shape[0]
        # index of last real token per sequence
        seq_lengths = attention_mask.sum(axis=1) - 1          # (batch,)
        pooled = hidden_states[np.arange(batch_size), seq_lengths]  # (batch, dim)
        # L2 normalise
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)               # avoid div-by-zero
        return pooled / norms

    # ── core inference ────────────────────────────────────────────────────────

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of raw texts (no prefix).
        Returns np.ndarray of shape (len(texts), dims).
        """
        self._ensure_loaded()

        encodings = self._tokenizer.encode_batch(texts)
        input_ids      = np.array([e.ids      for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)

        # some harrier ONNX exports expect token_type_ids
        inputs = {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
        }
        input_names = {inp.name for inp in self._session.get_inputs()}
        if "token_type_ids" in input_names:
            inputs["token_type_ids"] = np.zeros_like(input_ids)

        outputs = self._session.run(None, inputs)
        # sentence_embedding is already pooled + L2-normalised: (batch, 640)
        return outputs[0]

    # ── public API ────────────────────────────────────────────────────────────

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
        Convenience method — avoids materialising a list of 1-D arrays.
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