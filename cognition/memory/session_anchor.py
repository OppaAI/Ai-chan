"""
Phase 17 — session-level dynamic anchor for personal memory recall.

Maintains a per-user ring buffer of recent query embeddings for this process.
Rank boosts memories similar to the session mean ("what's hot in this chat")
without changing monthly consolidation novelty.
"""
from __future__ import annotations

import math
import os
import threading
from collections import defaultdict, deque
from typing import Deque

def _env_flag(name: str, default: str = "1") -> bool:
    return str(os.getenv(name, default)).strip().lower() not in ("0", "false", "no", "off", "")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


MEMORY_SESSION_ANCHOR_ENABLED = _env_flag("MEMORY_SESSION_ANCHOR_ENABLED", "1")
MEMORY_SESSION_ANCHOR_K = max(1, _env_int("MEMORY_SESSION_ANCHOR_K", 8))
# Mild boost; same order as MEMORY_RANK_RECENCY_WEIGHT (~0.004)
MEMORY_SESSION_ANCHOR_WEIGHT = _env_float("MEMORY_SESSION_ANCHOR_WEIGHT", 0.006)

_lock = threading.RLock()
_buffers: dict[str, Deque[list[float]]] = defaultdict(
    lambda: deque(maxlen=MEMORY_SESSION_ANCHOR_K)
)


def _normalize(vec: list[float]) -> list[float] | None:
    if not vec:
        return None
    s = math.sqrt(sum(float(x) * float(x) for x in vec))
    if s <= 1e-12:
        return None
    return [float(x) / s for x in vec]


def push_query_vector(user_id: str, vector: list[float] | None) -> None:
    """Record this turn's query embedding for the session mean."""
    if not MEMORY_SESSION_ANCHOR_ENABLED or not user_id or not vector:
        return
    normed = _normalize(list(vector))
    if normed is None:
        return
    with _lock:
        buf = _buffers[user_id]
        # Refresh maxlen if K changed at runtime (rare)
        if buf.maxlen != MEMORY_SESSION_ANCHOR_K:
            _buffers[user_id] = deque(buf, maxlen=MEMORY_SESSION_ANCHOR_K)
            buf = _buffers[user_id]
        buf.append(normed)


def session_mean(user_id: str) -> list[float] | None:
    """L2-normalized mean of the last K query vectors, or None if empty."""
    if not MEMORY_SESSION_ANCHOR_ENABLED or not user_id:
        return None
    with _lock:
        buf = _buffers.get(user_id)
        if not buf:
            return None
        dim = len(buf[0])
        acc = [0.0] * dim
        n = 0
        for v in buf:
            if len(v) != dim:
                continue
            for i, x in enumerate(v):
                acc[i] += x
            n += 1
        if n <= 0:
            return None
        acc = [x / n for x in acc]
    return _normalize(acc)


def cosine(a: list[float] | None, b: list[float] | None) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return float(sum(x * y for x, y in zip(a, b)))


def clear_session(user_id: str | None = None) -> None:
    """Drop buffer(s) — tests or explicit session end."""
    with _lock:
        if user_id is None:
            _buffers.clear()
        else:
            _buffers.pop(user_id, None)


def load_memory_vectors(conn, ids: list[str]) -> dict[str, list[float]]:
    """Fetch embeddings for candidate ids from memories_vec (best-effort)."""
    out: dict[str, list[float]] = {}
    if not ids:
        return out
    try:
        placeholders = ",".join("?" * len(ids))
        rows = conn.execute(
            f"SELECT id, embedding FROM memories_vec WHERE id IN ({placeholders})",
            list(ids),
        ).fetchall()
        for row in rows:
            mid = row["id"] if hasattr(row, "keys") else row[0]
            emb = row["embedding"] if hasattr(row, "keys") else row[1]
            if emb is None:
                continue
            if isinstance(emb, (bytes, bytearray)):
                continue  # unexpected format
            try:
                vec = list(emb) if not isinstance(emb, list) else emb
                normed = _normalize([float(x) for x in vec])
                if normed:
                    out[str(mid)] = normed
            except Exception:
                continue
    except Exception:
        return {}
    return out


def session_boost_for(mid: str, mean: list[float] | None, vecs: dict[str, list[float]]) -> float:
    if mean is None or MEMORY_SESSION_ANCHOR_WEIGHT <= 0:
        return 0.0
    v = vecs.get(str(mid))
    if not v:
        return 0.0
    # Only reward alignment with current chat topic (not anti-topic)
    return MEMORY_SESSION_ANCHOR_WEIGHT * max(0.0, cosine(v, mean))
