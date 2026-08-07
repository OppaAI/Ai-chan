"""Experience search, context formatting, and spreading."""
from __future__ import annotations

import json
import sqlite3
from html import escape

from cognition.memory.memorize import entities_from_json, entity_overlap_score
from cognition.memory.vecstore import rank_by_id, rrf_score, user_scoped_fts_search, user_scoped_vec_knn
from system.log import get_logger
from system.userspace import current_user_id

from .schema import (
    Embedder,
    EXPERIENCE_CONTEXT_CHARS,
    EXPERIENCE_ENTITY_BOOST,
    ExperienceSchema,
    EXPERIENCE_FTS_LIMIT,
    EXPERIENCE_KNN_LIMIT,
    EXPERIENCE_QUERY_INSTRUCT,
    EXPERIENCE_RECALL_SCORE_THRESHOLD,
    EXPERIENCE_RRF_K,
    EXPERIENCE_SPREADING_ENABLED,
    EXPERIENCE_SPREADING_MAX_EXTRA,
    EXPERIENCE_SPREADING_SCORE_WEIGHT,
    connect,
)

log = get_logger(__name__)


class ExperienceSearch:
    """Owns hybrid retrieval (RRF of KNN + FTS), spreading, and context formatting."""

    def __init__(self, schema: ExperienceSchema | None = None):
        self.schema = schema or ExperienceSchema()

    def search(self, query: str, limit: int = 3, embedder: Embedder | None = None, user_id: str | None = None) -> list[dict]:
        return search_experience(query, limit=limit, embedder=embedder, user_id=user_id)

    def context_for(self, query: str, limit: int = 3, embedder: Embedder | None = None) -> str:
        return experience_context_for(query, limit=limit, embedder=embedder)


def search_experience(query: str, limit: int = 3, embedder=None, user_id: str | None = None) -> list[dict]:
    """Legacy free-function shim: :class:`ExperienceSearch.search` delegates here."""
    uid = user_id or current_user_id()
    conn = connect(uid)
    try:
        rank_knn = rank_by_id(_knn(conn, query, embedder, uid, EXPERIENCE_KNN_LIMIT))
        rank_fts = rank_by_id(_fts(conn, query, uid, EXPERIENCE_FTS_LIMIT))
        ids = set(rank_knn) | set(rank_fts)
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        rows = conn.execute(f"SELECT * FROM experiences WHERE id IN ({placeholders})", list(ids)).fetchall()
        by_id = {row["id"]: row for row in rows}
        scored = []
        for cid in ids:
            score = rrf_score(cid, rank_knn, rank_fts, k=EXPERIENCE_RRF_K)
            row = by_id.get(cid)
            if row is None:
                continue
            try:
                st = (row["status"] if "status" in row.keys() else "active") or "active"
                if str(st).strip().lower() == "superseded":
                    continue
            except Exception:
                pass
            try:
                ents = entities_from_json(row["entities"] if "entities" in row.keys() else "[]")
            except Exception:
                ents = []
            score += EXPERIENCE_ENTITY_BOOST * entity_overlap_score(query, ents)
            if score >= EXPERIENCE_RECALL_SCORE_THRESHOLD:
                scored.append((score, cid))
        scored.sort(key=lambda pair: (-pair[0], by_id[pair[1]]["created_at"]))
        results = [dict(by_id[eid]) | {"recall_score": score} for score, eid in scored[:limit]]
        if EXPERIENCE_SPREADING_ENABLED and EXPERIENCE_SPREADING_MAX_EXTRA > 0 and results:
            results.extend(_experience_spread_extra(conn, uid, results, EXPERIENCE_SPREADING_MAX_EXTRA))
        return results
    except Exception as exc:
        log.warning("Experience search failed: %s", exc)
        return []
    finally:
        conn.close()


def _knn(conn: sqlite3.Connection, query: str, embedder, uid: str, limit: int) -> list[sqlite3.Row]:
    if embedder is None:
        return []
    vector = embedder.embed_query(query, instruct=EXPERIENCE_QUERY_INSTRUCT)
    return user_scoped_vec_knn(
        conn,
        vec_table="experiences_vec",
        owner_table="experiences",
        owner_alias="e",
        vector=vector,
        user_id=uid,
        limit=limit,
    )


def _fts(conn: sqlite3.Connection, query: str, uid: str, limit: int) -> list[sqlite3.Row]:
    return user_scoped_fts_search(
        conn,
        fts_table="experiences_fts",
        owner_table="experiences",
        owner_alias="e",
        query=query,
        user_id=uid,
        limit=limit,
    )


def _experience_spread_extra(conn: sqlite3.Connection, uid: str, hits: list[dict], limit: int) -> list[dict]:
    """Phase 18 optional: pull related engrams via engram_relations."""
    if limit <= 0 or not hits:
        return []
    seen = {str(h.get("id") or "") for h in hits}
    extra: list[dict] = []
    try:
        for h in hits:
            hid = str(h.get("id") or "")
            if not hid:
                continue
            rels = conn.execute(
                "SELECT to_engram AS oid FROM engram_relations WHERE from_engram = ? "
                "UNION SELECT from_engram AS oid FROM engram_relations WHERE to_engram = ?",
                (hid, hid),
            ).fetchall()
            for rel in rels:
                oid = str(rel["oid"] if hasattr(rel, "keys") else rel[0])
                if not oid or oid in seen:
                    continue
                row = conn.execute(
                    "SELECT * FROM experiences WHERE id = ? AND user_id = ?",
                    (oid, uid),
                ).fetchone()
                if row is None:
                    continue
                try:
                    st = (row["status"] if "status" in row.keys() else "active") or "active"
                    if str(st).strip().lower() == "superseded":
                        continue
                except Exception:
                    pass
                d = dict(row)
                d["recall_score"] = float(EXPERIENCE_SPREADING_SCORE_WEIGHT)
                d["_from_spreading"] = True
                extra.append(d)
                seen.add(oid)
                if len(extra) >= limit:
                    return extra
    except Exception as exc:
        log.debug("experience spreading skipped: %s", exc)
    return extra


def _attr(value: object) -> str:
    return escape(str(value or ""), quote=True)


def experience_context_for(query: str, limit: int = 3, embedder=None) -> str:
    """Legacy free-function shim: :class:`ExperienceSearch.context_for` delegates here."""
    hits = search_experience(query, limit=limit, embedder=embedder)
    if not hits:
        return "<experience_context>\nNo similar past task found.\n</experience_context>"
    remaining = EXPERIENCE_CONTEXT_CHARS
    blocks = []
    for hit in hits:
        if remaining <= 0:
            break
        steps = json.loads(hit["steps_json"] or "[]")
        step_line = ", ".join(f"{s['tool']}[{'ok' if s['ok'] else s.get('error_type') or 'fail'}]" for s in steps)
        body = f"goal: {hit['goal']}\nsteps: {step_line}\nresult: {hit['answer_excerpt']}"[:remaining]
        blocks.append(f'<past_task outcome="{_attr(hit["outcome"])}" verifier_score="{float(hit["score"]):.2f}" recall_score="{hit["recall_score"]:.4f}">\n{body}\n</past_task>')
        remaining -= len(body)
    return "<experience_context>\n" + "\n\n".join(blocks) + "\n</experience_context>"