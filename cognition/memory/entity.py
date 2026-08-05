"""Phase 3: entity importance I_e and supersession chain helpers.

I_e = (1-α)·centrality + α·recency
No LLM. Used by monthly consolidation and mild recall boost.
"""
from __future__ import annotations

import math
import os
import re
from datetime import datetime, timezone

from system.log import get_logger

log = get_logger(__name__)

ENTITY_IMPORTANCE_ALPHA = float(os.getenv("ENTITY_IMPORTANCE_ALPHA", "0.4"))
ENTITY_IMPORTANCE_BETA = float(os.getenv("ENTITY_IMPORTANCE_BETA", "0.05"))
MEMORY_RANK_ENTITY_IMPORTANCE_WEIGHT = float(os.getenv("MEMORY_RANK_ENTITY_IMPORTANCE_WEIGHT", "0.008"))
MEMORY_SUPERSESSION_CHAIN_EXPAND = os.getenv("MEMORY_SUPERSESSION_CHAIN_EXPAND", "1").lower() in {"1", "true", "yes", "on"}
MEMORY_SUPERSESSION_CHAIN_KINDS = {
    k.strip().lower()
    for k in os.getenv("MEMORY_SUPERSESSION_CHAIN_KINDS", "identity,preference,plan").split(",")
    if k.strip()
}
_REFLECTIVE_RE = re.compile(
    r"\b(used to|changed|before|previously|history|what changed|"
    r"remember when|used to be|no longer|switched from)\b",
    re.IGNORECASE,
)

MEMORY_SPREADING_ENABLED = os.getenv("MEMORY_SPREADING_ENABLED", "1").lower() in {"1", "true", "yes", "on"}
MEMORY_SPREADING_MAX_DEPTH = max(1, int(os.getenv("MEMORY_SPREADING_MAX_DEPTH", "2")))
MEMORY_SPREADING_DECAY = float(os.getenv("MEMORY_SPREADING_DECAY", "0.6"))
MEMORY_SPREADING_MIN_STRENGTH = float(os.getenv("MEMORY_SPREADING_MIN_STRENGTH", "0.15"))


def spread_activation(
    seed_entities: list[str],
    edges: list[tuple[str, str, float]],
    *,
    max_depth: int | None = None,
    decay: float | None = None,
    min_strength: float | None = None,
) -> dict[str, float]:
    """BFS-style activation over undirected co-mention edges.

    seed_entities: casefolded entity strings from entry-hit memories / query
    edges: list of (entity_a, entity_b, weight) already casefolded
    returns: entity -> activation strength in [0, 1+]
    """
    if not MEMORY_SPREADING_ENABLED or not seed_entities:
        return {}
    max_depth = max_depth if max_depth is not None else MEMORY_SPREADING_MAX_DEPTH
    decay = decay if decay is not None else MEMORY_SPREADING_DECAY
    min_strength = min_strength if min_strength is not None else MEMORY_SPREADING_MIN_STRENGTH
    decay = max(0.0, min(1.0, float(decay)))

    # adjacency
    adj: dict[str, list[tuple[str, float]]] = {}
    for a, b, w in edges:
        a, b = a.casefold(), b.casefold()
        if not a or not b or a == b:
            continue
        ww = max(0.0, float(w or 0.0))
        adj.setdefault(a, []).append((b, ww))
        adj.setdefault(b, []).append((a, ww))

    strength: dict[str, float] = {}
    for e in seed_entities:
        k = e.casefold()
        if k:
            strength[k] = max(strength.get(k, 0.0), 1.0)

    frontier = set(strength)
    for _ in range(max_depth):
        nxt: set[str] = set()
        for node in frontier:
            s0 = strength.get(node, 0.0)
            if s0 < min_strength:
                continue
            for nb, w in adj.get(node, []):
                # normalize soft weight: treat weight as relative, clamp
                hop = s0 * decay * min(1.0, max(0.0, w))
                if hop < min_strength:
                    continue
                if hop > strength.get(nb, 0.0):
                    strength[nb] = hop
                    nxt.add(nb)
        frontier = nxt
        if not frontier:
            break
    return {e: s for e, s in strength.items() if s >= min_strength}


def memory_max_activation(row, activation: dict[str, float]) -> float:
    if not activation:
        return 0.0
    ents = entities_from_json_safe(
        row["entities"] if hasattr(row, "keys") and "entities" in row.keys() else row.get("entities")
    )
    if not ents:
        return 0.0
    return max((activation.get(e.casefold(), 0.0) for e in ents), default=0.0)
def entities_from_json_safe(raw) -> list[str]:
    try:
        from cognition.memory.memorize import entities_from_json
        return entities_from_json(raw)
    except Exception:
        return []


def compute_entity_importance_map(memorize_or_backend, user_id: str) -> dict[str, float]:
    """I_e = (1-α)·centrality + α·recency per entity (casefolded)."""
    try:
        conn = getattr(memorize_or_backend, "_conn", None)
        if conn is None:
            mem = getattr(memorize_or_backend, "_mem", None)
            conn = getattr(mem, "_conn", None) if mem is not None else None
        if conn is None:
            return {}
        lock = getattr(getattr(memorize_or_backend, "_mem", memorize_or_backend), "_db_lock", None)

        def _read():
            try:
                rows = conn.execute(
                    "SELECT entity_a, entity_b, weight FROM entity_relations WHERE user_id = ?",
                    (user_id,),
                ).fetchall()
            except Exception:
                return {}, {}
            degree: dict[str, float] = {}
            for row in rows:
                a = str(row["entity_a"] or "").casefold()
                b = str(row["entity_b"] or "").casefold()
                w = float(row["weight"] or 0.0)
                if a:
                    degree[a] = degree.get(a, 0.0) + w
                if b:
                    degree[b] = degree.get(b, 0.0) + w
            last_touch: dict[str, str] = {}
            try:
                mem_rows = conn.execute(
                    """
                    SELECT entities, last_accessed_at, created_at FROM memories
                    WHERE user_id = ? AND (status = 'active' OR status IS NULL)
                    """,
                    (user_id,),
                ).fetchall()
            except Exception:
                mem_rows = []
            for mr in mem_rows:
                ents = entities_from_json_safe(mr["entities"] if "entities" in mr.keys() else "[]")
                ts = mr["last_accessed_at"] or mr["created_at"] or ""
                for e in ents:
                    key = e.casefold()
                    prev = last_touch.get(key, "")
                    if ts and (not prev or str(ts) > prev):
                        last_touch[key] = str(ts)
            return degree, last_touch

        if lock is not None:
            with lock:
                degree, last_touch = _read()
        else:
            degree, last_touch = _read()

        if not degree and not last_touch:
            return {}
        max_deg = max(degree.values(), default=1.0) or 1.0
        alpha = max(0.0, min(1.0, ENTITY_IMPORTANCE_ALPHA))
        beta = max(0.0, ENTITY_IMPORTANCE_BETA)
        now = datetime.now(timezone.utc)
        out: dict[str, float] = {}
        for e in set(degree) | set(last_touch):
            c = (degree.get(e, 0.0) / max_deg) if max_deg else 0.0
            r = 0.0
            ts = last_touch.get(e) or ""
            if ts and ts != "never":
                try:
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    days = max(0.0, (now - dt).total_seconds() / 86400.0)
                    r = float(math.exp(-beta * days))
                except Exception:
                    r = 0.0
            out[e] = (1.0 - alpha) * c + alpha * r
        return out
    except Exception as exc:
        log.debug("compute_entity_importance_map failed: %s", exc)
        return {}


def memory_max_entity_importance(row, importance_map: dict[str, float]) -> float:
    if not importance_map:
        return 0.0
    try:
        raw = row["entities"] if hasattr(row, "keys") and "entities" in row.keys() else row.get("entities")
        ents = entities_from_json_safe(raw)
    except Exception:
        ents = []
    if not ents:
        return 0.0
    return max((importance_map.get(e.casefold(), 0.0) for e in ents), default=0.0)


def should_expand_supersession_chain(query: str, row) -> bool:
    if not MEMORY_SUPERSESSION_CHAIN_EXPAND:
        return False
    if _REFLECTIVE_RE.search(query or ""):
        return True
    try:
        kind = (row["kind"] if hasattr(row, "keys") else row.get("kind")) or ""
    except Exception:
        kind = ""
    return str(kind).lower() in MEMORY_SUPERSESSION_CHAIN_KINDS


def walk_supersession_chain(conn, mem_id: str, user_id: str, max_depth: int = 12) -> list[dict]:
    try:
        row = conn.execute(
            "SELECT * FROM memories WHERE id = ? AND user_id = ?",
            (mem_id, user_id),
        ).fetchone()
        if row is None:
            return []
        chain_ids: list[str] = [mem_id]
        seen = {mem_id}
        cur = row
        for _ in range(max_depth):
            sid = cur["supersedes_id"] if "supersedes_id" in cur.keys() else None
            if not sid or sid in seen:
                break
            prev = conn.execute(
                "SELECT * FROM memories WHERE id = ? AND user_id = ?",
                (sid, user_id),
            ).fetchone()
            if prev is None:
                break
            chain_ids.append(sid)
            seen.add(sid)
            cur = prev
        chain_ids.reverse()
        tip = chain_ids[-1]
        for _ in range(max_depth):
            nxt = conn.execute(
                """
                SELECT * FROM memories
                WHERE user_id = ? AND supersedes_id = ?
                ORDER BY created_at ASC LIMIT 1
                """,
                (user_id, tip),
            ).fetchone()
            if nxt is None:
                break
            nid = str(nxt["id"])
            if nid in seen:
                break
            chain_ids.append(nid)
            seen.add(nid)
            tip = nid
        out: list[dict] = []
        for mid in chain_ids:
            r = conn.execute("SELECT * FROM memories WHERE id = ?", (mid,)).fetchone()
            if r is not None:
                out.append(dict(r))
        return out
    except Exception as exc:
        log.debug("supersession chain walk failed for %s: %s", mem_id, exc)
        return []


# Entity extraction/classification helpers still live with the write backend while
# the backend is being decomposed; expose lazy wrappers here so entity-related
# imports have a single home without forcing backend/numpy imports at module load.
def extract_entities(text: str):
    from .backend import extract_entities as _impl
    return _impl(text)


def entities_to_json(entities):
    from .backend import entities_to_json as _impl
    return _impl(entities)


def entities_from_json(raw):
    from .backend import entities_from_json as _impl
    return _impl(raw)


def entity_overlap_score(a, b) -> float:
    from .backend import entity_overlap_score as _impl
    return _impl(a, b)


def backfill_entities(conn) -> int:
    from .backend import backfill_entities as _impl
    return _impl(conn)


__all__ = [
    "ENTITY_IMPORTANCE_ALPHA",
    "ENTITY_IMPORTANCE_BETA",
    "MEMORY_RANK_ENTITY_IMPORTANCE_WEIGHT",
    "MEMORY_SUPERSESSION_CHAIN_EXPAND",
    "MEMORY_SUPERSESSION_CHAIN_KINDS",
    "MEMORY_SPREADING_ENABLED",
    "MEMORY_SPREADING_MAX_DEPTH",
    "MEMORY_SPREADING_DECAY",
    "MEMORY_SPREADING_MIN_STRENGTH",
    "backfill_entities",
    "compute_entity_importance_map",
    "entities_from_json",
    "entities_to_json",
    "entity_overlap_score",
    "extract_entities",
    "memory_max_activation",
    "memory_max_entity_importance",
    "should_expand_supersession_chain",
    "spread_activation",
    "walk_supersession_chain",
]
