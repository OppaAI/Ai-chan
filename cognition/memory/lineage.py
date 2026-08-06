"""Phase 19 — supersession lineage walk for Studio / GET /api/memory/{id}/lineage."""
from __future__ import annotations

import os
from typing import Any, Protocol


def _max_depth() -> int:
    try:
        return max(1, int(os.getenv("MEMORY_LINEAGE_MAX_DEPTH", "32")))
    except ValueError:
        return 32


class _Store(Protocol):
    def get_by_id(self, mem_id: str, user_id: str | None = None) -> dict | None: ...
    # Optional: implement with raw SQL if get_by_id does not exist yet.


def _row_public(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "id", "memory", "created_at", "status", "supersedes_id",
        "pinned", "kind", "source", "valence_score", "arousal_score",
        "entities", "access_count",
    )
    return {k: row.get(k) for k in keys if k in row or row.get(k) is not None}


def walk_supersession_lineage(
    store: Any,
    mem_id: str,
    *,
    user_id: str | None = None,
    max_depth: int | None = None,
) -> dict[str, Any]:
    """Return ordered supersession chain around ``mem_id``.

    - ``chain``: oldest → newest (root predecessor … current … successors)
    - ``center_id``: requested id
    - Walks ``supersedes_id`` backward (this → older) then forward (who points here)

    Requires the store to support loading a row by id for ``user_id``.
    If ``get_by_id`` is missing, implement via:

        SELECT * FROM memories WHERE id=? AND user_id=?
    """
    depth = max_depth if max_depth is not None else _max_depth()
    uid = user_id

    def load(mid: str) -> dict | None:
        if hasattr(store, "get_by_id"):
            return store.get_by_id(mid, user_id=uid)
        # Fallback: AikoMemorize-style connection helper expected on backend
        get = getattr(store, "_get_memory_row", None)
        if callable(get):
            return get(mid, user_id=uid)
        raise AttributeError("store must provide get_by_id or _get_memory_row")

    center = load(mem_id)
    if not center:
        return {"center_id": mem_id, "chain": [], "error": "not_found"}

    # Backward: follow supersedes_id toward older roots
    backward: list[dict] = []
    cur = center
    seen: set[str] = {str(center.get("id") or mem_id)}
    for _ in range(depth):
        sid = cur.get("supersedes_id")
        if not sid or sid in seen:
            break
        prev = load(str(sid))
        if not prev:
            break
        backward.append(prev)
        seen.add(str(prev.get("id")))
        cur = prev
    backward.reverse()  # oldest first

    # Forward: memories that supersede the current node (one hop fan-in at a time)
    # Prefer a store method if present; else single-step SQL via store helper.
    forward: list[dict] = []
    frontier = [str(center.get("id") or mem_id)]
    for _ in range(depth):
        if not frontier:
            break
        nxt_ids: list[str] = []
        for fid in frontier:
            children = _find_superseding(store, fid, user_id=uid)
            for ch in children:
                cid = str(ch.get("id") or "")
                if not cid or cid in seen:
                    continue
                seen.add(cid)
                forward.append(ch)
                nxt_ids.append(cid)
                if len(backward) + 1 + len(forward) >= depth:
                    break
            if len(backward) + 1 + len(forward) >= depth:
                break
        frontier = nxt_ids

    chain = [_row_public(r) for r in backward] + [_row_public(center)] + [_row_public(r) for r in forward]
    return {
        "center_id": str(center.get("id") or mem_id),
        "chain": chain[:depth],
        "depth": len(chain[:depth]),
    }


def _find_superseding(store: Any, mem_id: str, user_id: str | None) -> list[dict]:
    """Return rows whose supersedes_id == mem_id."""
    if hasattr(store, "find_by_supersedes"):
        return list(store.find_by_supersedes(mem_id, user_id=user_id) or [])
    # Raw SQL fallback on backend connection
    conn = getattr(store, "_conn", None)
    lock = getattr(store, "_db_lock", None)
    if conn is None:
        return []
    sql = """
        SELECT id, memory, created_at, status, supersedes_id, pinned,
               kind, source, valence_score, arousal_score, entities, access_count
        FROM memories
        WHERE supersedes_id = ? AND (? IS NULL OR user_id = ?)
        ORDER BY created_at ASC
        LIMIT 16
    """
    args = (mem_id, user_id, user_id)
    if lock is not None:
        with lock:
            rows = conn.execute(sql, args).fetchall()
    else:
        rows = conn.execute(sql, args).fetchall()
    return [dict(r) for r in rows]
