"""Knowledge search, context formatting, and search cache."""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from html import escape
import sqlite3

from cognition.memory.vecstore import rank_by_id, rrf_score, user_scoped_fts_search, user_scoped_vec_knn, utc_now_iso
from cognition.memory.memorize import entities_from_json, entity_overlap_score
from system.log import get_logger
from system.userspace import current_user_id

from .schema import (
    Embedder,
    KNOWLEDGE_CONTEXT_CHARS,
    KNOWLEDGE_ENTITY_BOOST,
    KNOWLEDGE_FTS_LIMIT,
    KNOWLEDGE_KNN_LIMIT,
    KNOWLEDGE_KNN_MIN_SIMILARITY,
    KNOWLEDGE_QUERY_INSTRUCT,
    KNOWLEDGE_RECALL_SCORE_THRESHOLD,
    KNOWLEDGE_RRF_K,
    KNOWLEDGE_SPREADING_ENABLED,
    KNOWLEDGE_SPREADING_MAX_EXTRA,
    KNOWLEDGE_SPREADING_SCORE_WEIGHT,
    KnowledgeSchema,
    connect,
)

log = get_logger(__name__)

# ── search cache (mirrors memory/memorize.py's pattern) ─────────────────────

_KNOWLEDGE_SEARCH_CACHE: OrderedDict[
    tuple[str, str, int, str], tuple[float, list[dict]]
] = OrderedDict()
_KNOWLEDGE_SEARCH_CACHE_LOCK = threading.RLock()
_KNOWLEDGE_SEARCH_CACHE_TTL: float = 20.0
_KNOWLEDGE_SEARCH_CACHE_MAX: int = 128

def _cache_key(query: str, user_id: str, limit: int, embedder_id: str) -> tuple[str, str, int, str]:
    return (user_id, query or "", limit, embedder_id)


def _search_cache_get(query: str, user_id: str, limit: int, embedder_id: str) -> list[dict] | None:
    key = _cache_key(query, user_id, limit, embedder_id)
    now = time.monotonic()
    with _KNOWLEDGE_SEARCH_CACHE_LOCK:
        cached = _KNOWLEDGE_SEARCH_CACHE.get(key)
        if cached is not None and now - cached[0] <= _KNOWLEDGE_SEARCH_CACHE_TTL:
            _KNOWLEDGE_SEARCH_CACHE.move_to_end(key)
            return [dict(r) for r in cached[1]]
        if cached:
            _KNOWLEDGE_SEARCH_CACHE.pop(key, None)
    return None


def _search_cache_set(query: str, user_id: str, limit: int, embedder_id: str, results: list[dict]) -> None:
    key = _cache_key(query, user_id, limit, embedder_id)
    now = time.monotonic()
    with _KNOWLEDGE_SEARCH_CACHE_LOCK:
        _KNOWLEDGE_SEARCH_CACHE[key] = (now, [dict(r) for r in results])
        while len(_KNOWLEDGE_SEARCH_CACHE) > _KNOWLEDGE_SEARCH_CACHE_MAX:
            _KNOWLEDGE_SEARCH_CACHE.popitem(last=False)


def maybe_clear_knowledge_cache() -> None:
    """Clear the cache after every successful write to ensure fresh data."""
    with _KNOWLEDGE_SEARCH_CACHE_LOCK:
        _KNOWLEDGE_SEARCH_CACHE.clear()



class KnowledgeSearch:
    """Owns hybrid retrieval, spreading, context formatting, and access tracking."""

    def __init__(self, schema: KnowledgeSchema | None = None, embedder: Embedder | None = None):
        self.schema = schema or KnowledgeSchema()
        self.embedder = embedder

    def search(
        self,
        query: str,
        limit: int = 5,
        *,
        embedder: Embedder | None = None,
        user_id: str | None = None,
    ) -> list[dict]:
        return search_knowledge(
            query,
            limit=limit,
            embedder=embedder if embedder is not None else self.embedder,
            user_id=user_id,
        )

    def context_for(
        self,
        query: str,
        limit: int = 5,
        max_chars: int | None = None,
        embedder: Embedder | None = None,
        user_id: str | None = None,
    ) -> str:
        return knowledge_context_for(
            query,
            limit=limit,
            max_chars=max_chars,
            embedder=embedder if embedder is not None else self.embedder,
            user_id=user_id,
        )


def _knn(conn: sqlite3.Connection, query: str, embedder: Embedder | None, uid: str, limit: int) -> list[sqlite3.Row]:
    if embedder is None or not (query or "").strip():
        return []
    vector = embedder.embed_query(query, instruct=KNOWLEDGE_QUERY_INSTRUCT)
    return user_scoped_vec_knn(
        conn,
        vec_table="learned_chunks_vec",
        owner_table="learned_chunks",
        owner_alias="c",
        vector=vector,
        user_id=uid,
        limit=limit,
        threshold=KNOWLEDGE_KNN_MIN_SIMILARITY,
    )


def _fts(conn: sqlite3.Connection, query: str, uid: str, limit: int) -> list[sqlite3.Row]:
    return user_scoped_fts_search(
        conn,
        fts_table="learned_chunks_fts",
        owner_table="learned_chunks",
        owner_alias="c",
        query=query,
        user_id=uid,
        limit=limit,
    )



def _knowledge_spread_extra(conn: sqlite3.Connection, uid: str, hits: list[dict], limit: int) -> list[dict]:
    """Phase 18 optional: pull active chunks sharing an entity with hits."""
    if not KNOWLEDGE_SPREADING_ENABLED or limit <= 0 or not hits:
        return []
    seen = {str(h.get("id") or "") for h in hits}
    entities: set[str] = set()
    for h in hits:
        try:
            for e in entities_from_json(h.get("entities") or "[]"):
                if e:
                    entities.add(str(e).strip().lower())
        except Exception:
            continue
    if not entities:
        return []
    extra: list[dict] = []
    try:
        rows = conn.execute(
            "SELECT id, text, chunk_index, created_at, entities, status FROM learned_chunks "
            "WHERE user_id = ? AND (status = 'active' OR status IS NULL) LIMIT 200",
            (uid,),
        ).fetchall()
        for row in rows:
            rid = str(row["id"])
            if rid in seen:
                continue
            try:
                ents = {str(e).strip().lower() for e in entities_from_json(row["entities"] or "[]") if e}
            except Exception:
                continue
            if len(ents & entities) < 2:
                continue
            d = dict(row)
            d["score"] = float(KNOWLEDGE_SPREADING_SCORE_WEIGHT)
            d["_from_spreading"] = True
            extra.append(d)
            seen.add(rid)
            if len(extra) >= limit:
                break
    except Exception as exc:
        log.debug("knowledge spreading skipped: %s", exc)
    return extra


def search_knowledge(
    query: str,
    limit: int = 5,
    *,
    embedder: Embedder | None = None,
    user_id: str | None = None,
) -> list[dict]:
    """Legacy free-function shim: sole implementation that :class:`KnowledgeSearch`
    delegates to. Kept for the historical import path."""
    uid = user_id or current_user_id()
    conn = connect(uid)
    try:
        rank_knn = rank_by_id(_knn(conn, query, embedder, uid, KNOWLEDGE_KNN_LIMIT))
        rank_fts = rank_by_id(_fts(conn, query, uid, KNOWLEDGE_FTS_LIMIT))
        ids = set(rank_knn) | set(rank_fts)
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        rows = conn.execute(
            f"""
            SELECT c.id, c.text, c.chunk_index, c.created_at, c.entities, c.status,
                d.title, d.source, d.kind, d.id AS doc_id
            FROM learned_chunks c
            JOIN learned_docs d ON d.id = c.doc_id
            WHERE c.id IN ({placeholders})
                AND (c.status = 'active' OR c.status IS NULL)
            """,
            list(ids),
        ).fetchall()
        by_id = {row["id"]: row for row in rows}
        scored: list[tuple[float, str]] = []
        for cid in ids:
            score = rrf_score(cid, rank_knn, rank_fts, k=KNOWLEDGE_RRF_K)
            row = by_id.get(cid)
            if row is None:
                continue
            ents = entities_from_json(row["entities"] if "entities" in row.keys() else "[]")
            score += KNOWLEDGE_ENTITY_BOOST * entity_overlap_score(query, ents)
            if score >= KNOWLEDGE_RECALL_SCORE_THRESHOLD:
                scored.append((score, cid))
        scored.sort(key=lambda pair: (-pair[0], by_id[pair[1]]["created_at"]))
        results = [dict(by_id[cid]) | {"score": score} for score, cid in scored[:limit]]
        if KNOWLEDGE_SPREADING_ENABLED and KNOWLEDGE_SPREADING_MAX_EXTRA > 0:
            results.extend(_knowledge_spread_extra(conn, uid, results, KNOWLEDGE_SPREADING_MAX_EXTRA))
        return results
    except Exception as exc:
        log.warning("Knowledge search failed: %s", exc)
        return []
    finally:
        conn.close()

def _attr(value: object) -> str:
    return escape(str(value or ""), quote=True)


def knowledge_context_for(
    query: str,
    limit: int = 5,
    max_chars: int | None = None,
    embedder: Embedder | None = None,
    user_id: str | None = None,
) -> str:
    """Legacy free-function shim: sole implementation that
    :class:`KnowledgeSearch.context_for` delegates to. Kept for the historical
    import path.

    Retrieve knowledge context for a query, tracking access counts.

    Args:
        query: Search query
        limit: Max number of results
        max_chars: Max total characters in returned context (default: KNOWLEDGE_CONTEXT_CHARS)
        embedder: Optional embedder for vector search
        user_id: Optional user ID (defaults to current_user_id)
    """
    uid = user_id or current_user_id()
    remaining = KNOWLEDGE_CONTEXT_CHARS if max_chars is None else max_chars

    # Generate embedder ID for cache key
    embedder_id = str(id(embedder)) if embedder is not None else "default"

    cached = _search_cache_get(query, uid, limit, embedder_id)
    if cached is not None:
        # Track access for cached results too
        chunk_ids = [r["id"] for r in cached]
        _increment_access_count(chunk_ids, uid)
        return _format_knowledge_context(cached, remaining)

    hits = search_knowledge(query, limit=limit, embedder=embedder, user_id=uid)
    if hits:
        _search_cache_set(query, uid, limit, embedder_id, hits)

    if not hits:
        return "<knowledge_context>\nNo matching learned knowledge found.\n</knowledge_context>"

    # Track access for non-cached results
    chunk_ids = [h["id"] for h in hits]
    _increment_access_count(chunk_ids, uid)

    return _format_knowledge_context(hits, remaining)



def _increment_access_count(chunk_ids: list[str], user_id: str | None = None) -> None:
    """Increment access count and update last_accessed for given chunk IDs."""
    if not chunk_ids:
        return
    uid = user_id or current_user_id()
    conn = connect(uid)
    try:
        now = utc_now_iso()
        placeholders = ",".join("?" * len(chunk_ids))
        conn.execute(
            f"UPDATE learned_chunks SET access_count = access_count + 1, last_accessed = ? WHERE id IN ({placeholders})",
            [now] + chunk_ids,
        )
        conn.commit()
    except Exception as exc:
        log.warning("increment access count failed: %s", exc)
    finally:
        conn.close()


def _format_knowledge_context(results: list[dict], max_chars: int) -> str:
    remaining = max_chars
    blocks = []
    for row in results:
        if remaining <= 0:
            break
        body = row["text"][:remaining]
        blocks.append(
            f'<knowledge_chunk doc_id="{_attr(row.get("doc_id", ""))}" title="{_attr(row.get("title", ""))}" '
            f'kind="{_attr(row.get("kind", ""))}" source="{_attr(row.get("source", ""))}" score="{row.get("score", row.get("recall_score", 0)):.4f}">\n'
            f'{body}\n</knowledge_chunk>'
        )
        remaining -= len(body)
    return "<knowledge_context>\n" + "\n\n".join(blocks) + "\n</knowledge_context>"

