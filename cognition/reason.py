"""cognition/reason.py

Shared numpy-vectorized embedding utilities for semantic retrieval and
condensation. Centralizes what used to be four separate per-module
implementations (tools.py web-evidence condensation, knowledge.py KB
ranking, skills.py skill ranking, agentic.py policy filtering) into one
batched-matmul scoring primitive instead of four Python-loop cosine calls.

Also centralizes the close-vector label-scoring primitive used by
think.py's semantic intent router (top-k mean cosine per label against a
static example corpus) — previously a second, duplicate implementation of
the same normalize/matmul math lived in think.py itself.

Every scoring function here degrades gracefully to keyword overlap when no
embedder is supplied or an embed call raises — semantic scoring is strictly
additive, never a hard dependency.
"""

from __future__ import annotations

import re
from collections import defaultdict
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

def cosine_similarity(vec_a, vec_b) -> float:
    a = np.asarray(vec_a, dtype=float)
    b = np.asarray(vec_b, dtype=float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def block_relevance_score(embedder, query: str, text: str, instruct: str | None = None) -> float:
    """Post-hoc relevance score for an already-assembled context block —
    used for budget arbitration when the block's own retrieval score
    isn't threaded through by its source module (wiki/skill/experience)."""
    if embedder is None or not query or not text:
        return 0.0
    try:
        q_vec = embedder.embed_query(query, instruct=instruct)
        b_vec = embedder.embed_query(text[:1500], instruct=instruct)
    except Exception:
        return 0.0
    return cosine_similarity(q_vec, b_vec)


def batch_block_relevance_scores(
    embedder, query: str, texts: list[str], instruct: str | None = None,
) -> list[float]:
    """Score multiple context blocks against one query in a single batch
    embedding call. Embeds the query once and all texts in one batch,
    then returns a list of cosine scores (same order as `texts`).

    Uses embed_queries when available (batched HTTP) instead of N
    individual embed_query calls, cutting per-block latency from 2 HTTP
    round-trips to 2 total.
    """
    if embedder is None or not query or not texts:
        return [0.0] * len(texts) if texts else []
    try:
        truncated = [t[:1500] for t in texts]
        all_texts = [query] + truncated
        batch = embedder.embed_queries(all_texts, instruct=instruct)
        if batch is None or len(batch) != len(all_texts):
            return [block_relevance_score(embedder, query, t, instruct=instruct) for t in texts]
        q_vec = np.asarray(batch[0], dtype=np.float32)
        b_vecs = np.asarray(batch[1:], dtype=np.float32)
        scores = batch_cosine_scores(q_vec, b_vecs)
        return [float(s) for s in scores]
    except Exception:
        return [block_relevance_score(embedder, query, t, instruct=instruct) for t in texts]


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


# ── close-vector label scoring ───────────────────────────────────────────────
# Shared primitive for classifying one query against a static, labeled
# example corpus (think.py's semantic intent router: agentic/webchat/
# localchat). Kept here rather than in think.py so the normalize+matmul
# math has exactly one implementation instead of two.

def embed_example_matrix(
    embedder: Embedder,
    examples_by_label: dict[str, list[str] | tuple[str, ...]],
    instruct: str = "",
) -> tuple[list[str], np.ndarray]:
    """Embed a {label: [examples...]} corpus into one aligned
    (labels, matrix) pair — labels[i] is the label for matrix row i.

    This always re-embeds; it does not cache. Callers with a static
    example corpus (e.g. router examples that never change at runtime)
    should cache the returned (labels, matrix) pair themselves, keyed on
    corpus identity + instruct string, rather than paying the embed cost
    on every call.
    """
    labels: list[str] = []
    prompts: list[str] = []
    for label, examples in examples_by_label.items():
        labels.extend([label] * len(examples))
        prompts.extend(examples)

    if not prompts:
        return [], np.empty((0, 0), dtype=np.float32)

    raw = embedder.embed_queries(prompts, instruct=instruct) if instruct else embedder.embed_queries(prompts)
    matrix = normalize_rows(np.asarray(raw, dtype=np.float32))
    return labels, matrix


def label_scores_topk(
    query_vec,
    labels: list[str],
    example_vecs: np.ndarray,
    top_k: int = 3,
) -> dict[str, float]:
    """Mean of the top-k cosine scores per label, for close-vector
    classification against a static example corpus (intent routing, tagging,
    etc). `labels` and `example_vecs` must be row-aligned, as returned by
    embed_example_matrix. `query_vec` need not be pre-normalized —
    batch_cosine_scores normalizes internally.

    Returns {} if example_vecs is empty (nothing to score against).
    """
    if example_vecs.size == 0:
        return {}
    scores = batch_cosine_scores(query_vec, example_vecs)
    by_label: dict[str, list[float]] = defaultdict(list)
    for label, score in zip(labels, scores):
        by_label[label].append(float(score))
    k = max(1, top_k)
    return {
        label: sum(sorted(values, reverse=True)[:k]) / min(k, len(values))
        for label, values in by_label.items()
    }
