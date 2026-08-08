"""
memory/studio/ltm/backend/search_memory.py

Phase B unified recall facade over personal memory + learned knowledge.

Studio-facing read helper (user search across mem + KB). Keeps each store's
own ranking; merges results with a simple interleave by normalized score.
No second embedding model — callers may pass a shared query_vector for
personal memory and an embedder for knowledge.
"""
from __future__ import annotations

import os
from typing import Any, Sequence

from system.log import get_logger
from system.userspace import current_user_id

log = get_logger(__name__)

# Soft boost when query text mentions tagged entities on a personal memory.
MEMORY_ENTITY_BOOST = float(os.getenv("MEMORY_ENTITY_BOOST", "0.003"))


def _escape_like(value: str) -> str:
    """Escape SQL LIKE wildcards so entity tokens match literally."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _normalize_personal_hit(row: dict, query: str = "") -> dict[str, Any]:
    from cognition.memory.memorize import entities_from_json, entity_overlap_score

    entities = entities_from_json(row.get("entities"))
    base = float(row.get("_recall_score") or row.get("score") or 0.0)
    boost = MEMORY_ENTITY_BOOST * entity_overlap_score(query, entities)
    return {
        "store": "personal",
        "id": str(row.get("id") or ""),
        "text": row.get("memory") or row.get("text") or "",
        "score": base + boost,
        "created_at": row.get("created_at"),
        "kind": row.get("kind") or "fact",
        "source": row.get("source") or "",
        "status": row.get("status") or "active",
        "entities": entities,
        "pinned": bool(row.get("pinned")),
        "title": "",
    }


def _normalize_knowledge_hit(row: dict) -> dict[str, Any]:
    return {
        "store": "knowledge",
        "id": str(row.get("id") or ""),
        "text": row.get("text") or "",
        "score": float(row.get("score") or 0.0),
        "created_at": row.get("created_at"),
        "kind": row.get("kind") or "ingested",
        "source": row.get("source") or "",
        "status": "active",
        "entities": [],
        "pinned": False,
        "title": row.get("title") or "",
        "doc_id": row.get("doc_id"),
    }


def search_personal(
    query: str,
    *,
    limit: int = 5,
    memorize: Any = None,
    user_id: str | None = None,
    query_vector: list[float] | None = None,
    include_history: bool = False,
) -> list[dict[str, Any]]:
    """Search personal memory store; apply entity overlap boost."""
    if memorize is None:
        return []
    try:
        rows = memorize.search(
            query,
            user_id=user_id,
            limit=limit,
            query_vector=query_vector,
            include_history=include_history,
        )
    except TypeError:
        # Pre-Phase-A search signature
        rows = memorize.search(query, user_id=user_id, limit=limit, query_vector=query_vector)
    except Exception as e:
        log.warning("search_personal failed: %s", e)
        return []
    hits = [_normalize_personal_hit(r, query) for r in (rows or [])]
    hits.sort(key=lambda h: h["score"], reverse=True)
    return hits[:limit]


def search_knowledge_store(
    query: str,
    *,
    limit: int = 5,
    embedder: Any = None,
    user_id: str | None = None,
) -> list[dict[str, Any]]:
    try:
        from cognition.knowledge import search_knowledge
    except Exception as e:
        log.debug("knowledge store unavailable: %s", e)
        return []
    try:
        rows = search_knowledge(query, limit=limit, embedder=embedder, user_id=user_id)
    except Exception as e:
        log.warning("search_knowledge failed: %s", e)
        return []
    return [_normalize_knowledge_hit(r) for r in (rows or [])]


def search_memory(
    query: str,
    *,
    limit: int = 5,
    stores: Sequence[str] = ("personal", "knowledge"),
    memorize: Any = None,
    embedder: Any = None,
    user_id: str | None = None,
    query_vector: list[float] | None = None,
    include_history: bool = False,
    per_store_limit: int | None = None,
) -> list[dict[str, Any]]:
    """Unified multi-store recall.

    Returns a list of normalized hits::

        {
          "store": "personal" | "knowledge",
          "id": str,
          "text": str,
          "score": float,
          "kind": str,
          "entities": list[str],
          ...
        }

    Each store is queried independently (cheap parallel-ready structure);
    results are merged by score descending. Defaults keep total work low:
    ``per_store_limit`` defaults to ``limit``.
    """
    q = (query or "").strip()
    if not q:
        return []
    uid = user_id or current_user_id()
    sub = int(per_store_limit) if per_store_limit is not None else int(limit)
    sub = max(1, sub)

    merged: list[dict[str, Any]] = []
    wanted = {s.strip().lower() for s in stores}

    if "personal" in wanted:
        merged.extend(
            search_personal(
                q,
                limit=sub,
                memorize=memorize,
                user_id=uid,
                query_vector=query_vector,
                include_history=include_history,
            )
        )
    if "knowledge" in wanted:
        # Prefer embedder from memorize backend when not provided
        emb = embedder
        if emb is None and memorize is not None:
            try:
                emb = memorize._mem._embedder
            except Exception:
                emb = None
        merged.extend(
            search_knowledge_store(q, limit=sub, embedder=emb, user_id=uid)
        )

    merged.sort(key=lambda h: float(h.get("score") or 0.0), reverse=True)
    return merged[: int(limit)]


def memories_for_entity(
    entity: str,
    *,
    memorize: Any = None,
    user_id: str | None = None,
    limit: int = 50,
    include_history: bool = False,
) -> list[dict[str, Any]]:
    """List personal memories whose entities JSON contains ``entity``.

    SQL LIKE over the JSON text — fine at personal-memory scale; no FTS index
    required for Phase B. Used by future graph studio / debug tools.
    """
    if memorize is None or not (entity or "").strip():
        return []
    uid = user_id or current_user_id()
    needle = entity.strip()
    try:
        conn = memorize._conn
        lock = memorize._mem._db_lock
    except Exception:
        return []
    # status_sql is one of two hardcoded literals (not user input).
    status_clause = "" if include_history else "AND (status = 'active' OR status IS NULL)"
    # Match `"needle"` as a full JSON array element, i.e. followed by a `,`
    # (more elements) or a `]` (last element). The entities column stores a
    # JSON array like ["a","b"], so a bare LIKE `%"needle"%` would also match
    # `"needle-adjacent"`. Split into two predicates because SQLite LIKE
    # cannot express `]` inside a character class.
    needle_esc = _escape_like(needle)
    like_next = f'%"{needle_esc}",%'
    like_end = f'%"{needle_esc}"]%'
    sql = (
        "SELECT * FROM memories "
        "WHERE user_id = ? "
        "AND (entities LIKE ? ESCAPE '\\' OR entities LIKE ? ESCAPE '\\') "
        f"{status_clause} "
        "ORDER BY created_at DESC "
        "LIMIT ?"
    )
    with lock:
        rows = conn.execute(sql, (uid, like_next, like_end, int(limit))).fetchall()
    return [_normalize_personal_hit(dict(r), needle) for r in rows]
