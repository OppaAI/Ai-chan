"""core/reason.py

Shared numpy-vectorized embedding utilities for semantic retrieval and
condensation. Centralizes what used to be four separate per-module
implementations (tools.py web-evidence condensation, knowledge.py KB
ranking, skills.py skill ranking, agentic.py policy filtering) into one
batched-matmul scoring primitive instead of four Python-loop cosine calls.

Every scoring function here degrades gracefully to keyword overlap when no
embedder is supplied or an embed call raises — semantic scoring is strictly
additive, never a hard dependency.
"""

from __future__ import annotations

import re
from typing import Protocol

import numpy as np

_WORD_RE = re.compile(r"[a-z0-9_./-]+", re.IGNORECASE)

# Shared stopword list — same rationale everywhere it's used: without
# filtering, common words incidentally match inside unrelated text and
# inflate keyword-fallback scores.
STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "how", "what", "who", "when", "where", "why",
    "we", "you", "i", "he", "she", "they", "this", "that", "these",
    "those", "some", "any", "all", "each", "can", "could", "will",
    "would", "should", "shall", "may", "might", "must", "to", "of", "in",
    "on", "at", "for", "with", "and", "or", "not", "no", "yes", "make",
    "made", "get", "got", "go", "going", "let", "lets", "want", "wants",
    "just", "so", "up", "down", "out", "about", "if", "then", "than",
})


class Embedder(Protocol):
    def embed_query(self, text: str, instruct: str = "") -> object: ...
    def embed_queries(self, texts: list[str], instruct: str = "") -> object: ...


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize each row of a 2D array; zero rows are left untouched
    (dividing by 1.0 instead of 0) so a degenerate embedding doesn't NaN
    out the whole batch."""
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return matrix / norms


def normalize_vec(vector: np.ndarray) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    return arr / norm if norm > 1e-12 else arr


def batch_cosine_scores(query_vec, item_vecs: np.ndarray) -> np.ndarray:
    """Cosine similarity of one query vector against N item vectors as a
    single vectorized matmul, replacing what used to be a per-item Python
    loop calling a scalar _cosine() each time."""
    item_vecs = np.asarray(item_vecs, dtype=np.float32)
    if item_vecs.size == 0:
        return np.array([], dtype=np.float32)
    q = normalize_vec(np.asarray(query_vec, dtype=np.float32))
    m = normalize_rows(item_vecs)
    return m @ q


def embed_batch_or_none(embedder: Embedder, texts: list[str]) -> np.ndarray | None:
    """Best-effort batched embedding. Probes conventional batch method
    names first (embed_queries/embed_documents/embed_batch/embed); falls
    back to per-text embed_query calls stacked into a matrix; returns None
    only if embedding fails outright, so the caller can fall back to
    keyword scoring."""
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    for method_name in ("embed_queries", "embed_documents", "embed_batch", "embed"):
        method = getattr(embedder, method_name, None)
        if callable(method):
            try:
                result = method(texts)
                if result is not None and len(result) == len(texts):
                    return np.asarray(result, dtype=np.float32)
            except Exception:
                continue
    try:
        vecs = [np.asarray(embedder.embed_query(t), dtype=np.float32) for t in texts]
        return np.stack(vecs) if vecs else np.empty((0, 0), dtype=np.float32)
    except Exception:
        return None


def keyword_overlap_score(query: str, text: str) -> float:
    """Fallback relevance score when no embedder is available. Literal
    substring/term overlap only — misses paraphrases and synonyms, which is
    exactly why this is a fallback and not the primary scoring path."""
    q_terms = {t for t in _WORD_RE.findall(query.lower()) if len(t) > 2 and t not in STOPWORDS}
    if not q_terms:
        return 0.0
    t_terms = {t for t in _WORD_RE.findall(text.lower()) if len(t) > 2}
    if not t_terms:
        return 0.0
    return len(q_terms & t_terms) / len(q_terms)


def chunk_text(text: str, chunk_chars: int) -> list[str]:
    """Split text into non-overlapping stripped chunks of at most
    chunk_chars, dropping empty pieces."""
    if not text:
        return []
    chunks = []
    for i in range(0, len(text), chunk_chars):
        piece = text[i:i + chunk_chars].strip()
        if piece:
            chunks.append(piece)
    return chunks


def select_relevant_chunks(
    query: str,
    chunks: list[str],
    embedder: Embedder | None,
    top_k: int,
    min_score: float,
    instruct: str = "",
) -> list[tuple[float, str]]:
    """The shared RAG-selection primitive: given a list of text chunks,
    return up to top_k (score, chunk) pairs scoring >= min_score.

    Uses one batched embed call + one vectorized matmul when an embedder is
    available; falls back to per-chunk keyword overlap otherwise. This is
    what lets knowledge.py/skills.py/agentic.py inject only the relevant
    slice of a document instead of the whole file.
    """
    if not chunks:
        return []

    if embedder is not None and hasattr(embedder, "embed_query"):
        try:
            query_vec = (
                embedder.embed_query(query, instruct=instruct) if instruct
                else embedder.embed_query(query)
            )
            chunk_vecs = embed_batch_or_none(embedder, chunks)
            if chunk_vecs is not None and chunk_vecs.shape[0] == len(chunks):
                scores = batch_cosine_scores(query_vec, chunk_vecs)
                order = np.argsort(-scores)
                out: list[tuple[float, str]] = []
                for idx in order:
                    score = float(scores[idx])
                    if score < min_score:
                        break
                    out.append((score, chunks[idx]))
                    if len(out) >= top_k:
                        break
                return out
        except Exception:
            pass  # fall through to keyword scoring below

    scored = [(keyword_overlap_score(query, c), c) for c in chunks]
    scored = [(s, c) for s, c in scored if s > 0]
    scored.sort(key=lambda pair: -pair[0])
    return scored[:top_k]
