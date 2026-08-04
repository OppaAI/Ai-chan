"""
interface/webui/studio/memory/backend/graph_export.py

Export personal memory as a node/edge graph for Memory Graph Studio.

Phase C: memory + entity nodes, supersedes / mentions edges.
Phase D: entity co-mention edges from entity_relations.
Phase 10: retain tendency, rim scores, valence, I_e → size + scores on nodes.

Read-only. Tolerates pre-Phase-A DBs (missing status/entities columns).
"""
from __future__ import annotations

import json
import math
import os
import re
import sqlite3
from pathlib import Path
from typing import Any

from system.log import get_logger
from system.userspace import current_user_id

log = get_logger(__name__)

_SPACING_SAT = max(1, int(os.getenv("MONTHLY_CONSOLIDATION_SPACING_SATURATION", "5")))
_DAILY_RE = re.compile(r"^\[\d{4}-\d{2}-\d{2}\]\s")
_MONTHLY_RE = re.compile(r"^\[\d{4}-\d{2}\]\s")

def _env_int(name: str, default: int, *, floor: int = 0) -> int:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return max(floor, default)
    try:
        return max(floor, int(str(raw).strip()))
    except (TypeError, ValueError):
        log.warning("Invalid %s=%r; using default %s", name, raw, default)
        return max(floor, default)

_MAX_MEMORIES = _env_int("MEMORY_STUDIO_MAX_MEMORIES", 400, floor=1)
_MAX_ENTITIES = _env_int("MEMORY_STUDIO_MAX_ENTITIES", 120, floor=0)
_MAX_EDGES = _env_int("MEMORY_STUDIO_MAX_EDGES", 200, floor=0)


def _entities_from_raw(raw: Any) -> list[str]:
    if raw is None or raw == "":
        return []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, json.JSONDecodeError):
        return []
    if not isinstance(data, list):
        return []
    return [str(x).strip() for x in data if str(x).strip()]


def _table_columns(conn: sqlite3.Connection, table: str = "memories") -> set[str]:
    try:
        return {str(r[1]) for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    except sqlite3.Error:
        return set()


def _resolve_db_path(user_id: str) -> Path:
    import os
    from memory.vecstore import resolve_user_db_path

    env = os.getenv("SQLITE_MEMORY_PATH", "").strip()
    if env:
        return Path(env).expanduser()
    return resolve_user_db_path("memory/memory.db", user_id=user_id)


def _valence_score(tag: Any) -> float:
    t = (str(tag or "neutral")).strip().lower()
    if t == "neg":
        return 0.85
    if t == "pos":
        return 0.65
    return 0.25


def _salience_score(text: str, stored_hit: Any) -> float:
    if stored_hit is not None and str(stored_hit) != "":
        try:
            return 1.0 if int(stored_hit) else 0.3
        except (TypeError, ValueError):
            pass
    try:
        from memory.memorize import SALIENCE_POLICY_RE
        return 1.0 if SALIENCE_POLICY_RE.search(text or "") else 0.3
    except Exception:
        return 0.3


def _spacing_score(access_day_count: int, access_count: int) -> float:
    day_count = int(access_day_count or 0)
    if day_count <= 0:
        day_count = 1 if int(access_count or 0) > 0 else 0
    return min(1.0, day_count / float(_SPACING_SAT))


def _access_score(access_count: int) -> float:
    ac = int(access_count or 0)
    return min(1.0, math.log1p(ac) / math.log1p(50))


def _memory_scores(
    *,
    text: str,
    status: str,
    pinned: bool,
    access_count: int,
    access_day_count: int,
    valence_tag: Any,
    salience_hit: Any,
    entities: list[str],
    entity_importance: dict[str, float],
) -> dict[str, float]:
    """Rim arcs + retain tendency (node size). Not full monthly R (no anchors)."""
    if str(status).strip().lower() == "superseded":
        retain = 0.15
    else:
        sal = _salience_score(text, salience_hit)
        sp = _spacing_score(access_day_count, access_count)
        val = _valence_score(valence_tag)
        acc = _access_score(access_count)
        ie = 0.0
        if entities and entity_importance:
            ie = max(
                (entity_importance.get(e.casefold(), 0.0) for e in entities),
                default=0.0,
            )
        pin = 1.0 if pinned else 0.0
        if _MONTHLY_RE.match(text or ""):
            pin = max(pin, 0.95)
        retain = (
            0.28 * sal
            + 0.18 * sp
            + 0.12 * val
            + 0.12 * acc
            + 0.20 * ie
            + 0.25 * pin
        )
        retain = float(max(0.05, min(1.0, retain)))

    sal = _salience_score(text, salience_hit)
    sp = _spacing_score(access_day_count, access_count)
    val = _valence_score(valence_tag)
    acc = _access_score(access_count)
    ie = 0.0
    if entities and entity_importance:
        ie = max(
            (entity_importance.get(e.casefold(), 0.0) for e in entities),
            default=0.0,
        )

    return {
        "retain": round(retain, 4),
        "salience": round(sal, 4),
        "spacing": round(sp, 4),
        "connectivity": round(ie, 4),
        "valence": round(val, 4),
        "access": round(acc, 4),
    }


def export_memory_graph(
    *,
    user_id: str | None = None,
    limit: int = 200,
    include_history: bool = True,
    include_entities: bool = True,
    conn: sqlite3.Connection | None = None,
) -> dict[str, Any]:
    """Build a graph dict: {nodes, edges, meta, legend}.

    Node types:
      - memory: one per fact row (scores + size = retain tendency)
      - entity: hub nodes (size = I_e when available)

    Edge types:
      - supersedes: newer memory → older memory it replaced
      - mentions: memory → entity
      - co_mentions / related_to: entity → entity (entity_relations)
    """
    uid = user_id or current_user_id()
    owns_conn = conn is None
    if conn is None:
        from memory.vecstore import initialize_store_db

        db_path = _resolve_db_path(uid)
        if not db_path.exists() and str(db_path) != ":memory:":
            return {
                "nodes": [],
                "edges": [],
                "meta": {"user_id": uid, "count": 0, "note": "db missing"},
                "legend": _legend(),
            }
        try:
            conn = initialize_store_db(str(db_path), "PRAGMA journal_mode = WAL;", user_id=uid, vector=True)
        except Exception as e:
            log.warning("graph_export: open failed: %s", e)
            return {"nodes": [], "edges": [], "meta": {"user_id": uid, "error": str(e)}, "legend": _legend()}

    try:
        cols = _table_columns(conn)
        if "id" not in cols:
            return {
                "nodes": [],
                "edges": [],
                "meta": {"user_id": uid, "count": 0, "note": "no memories table"},
                "legend": _legend(),
            }

        has_status = "status" in cols
        has_entities = "entities" in cols
        has_kind = "kind" in cols
        has_source = "source" in cols
        has_supersedes = "supersedes_id" in cols
        has_valence = "valence_tag" in cols
        has_salience = "salience_hit" in cols
        has_day_count = "access_day_count" in cols

        select_cols = ["id", "memory", "created_at", "pinned", "access_count"]
        if has_status:
            select_cols.append("status")
        if has_entities:
            select_cols.append("entities")
        if has_kind:
            select_cols.append("kind")
        if has_source:
            select_cols.append("source")
        if has_supersedes:
            select_cols.append("supersedes_id")
        if has_valence:
            select_cols.append("valence_tag")
        if has_salience:
            select_cols.append("salience_hit")
        if has_day_count:
            select_cols.append("access_day_count")

        sql = f"SELECT {', '.join(select_cols)} FROM memories WHERE user_id = ?"
        params: list[Any] = [uid]
        if has_status and not include_history:
            sql += " AND (status = 'active' OR status IS NULL)"
        sql += " ORDER BY created_at DESC"
        effective_limit = min(int(limit or 200), _MAX_MEMORIES) if limit else _MAX_MEMORIES
        if effective_limit > 0:
            sql += " LIMIT ?"
            params.append(int(effective_limit))

        rows = conn.execute(sql, params).fetchall()

        entity_importance: dict[str, float] = {}
        try:
            from memory.entity_importance import compute_entity_importance_map
            from types import SimpleNamespace

            # Reuse export conn + uid (no new AikoMemorize / write worker).
            entity_importance = compute_entity_importance_map(
                SimpleNamespace(_conn=conn, _db_lock=None),
                uid,
            ) or {}
        except Exception as ex:
            log.debug("graph_export: I_e map skipped: %s", ex)

        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        entity_ids: set[str] = set()
        mem_ids: set[str] = set()

        for row in rows:
            mid = str(row["id"])
            mem_ids.add(mid)
            text = (row["memory"] or "").strip()
            label = text if len(text) <= 80 else text[:77] + "…"
            status = str(row["status"]) if has_status and row["status"] is not None else "active"
            kind = str(row["kind"]) if has_kind and row["kind"] else "fact"
            if _MONTHLY_RE.match(text):
                kind = "monthly"
            elif _DAILY_RE.match(text) and kind == "fact":
                kind = "daily"
            source = str(row["source"]) if has_source and row["source"] else ""
            ents = _entities_from_raw(row["entities"]) if has_entities else []
            supersedes_id = (
                str(row["supersedes_id"]) if has_supersedes and row["supersedes_id"] else None
            )
            valence_tag = (
                str(row["valence_tag"]) if has_valence and row["valence_tag"] is not None else "neutral"
            )
            salience_hit = row["salience_hit"] if has_salience else None
            access_day_count = int(row["access_day_count"] or 0) if has_day_count else 0
            access_count = int(row["access_count"] or 0)
            pinned = bool(row["pinned"])

            scores = _memory_scores(
                text=text,
                status=status,
                pinned=pinned,
                access_count=access_count,
                access_day_count=access_day_count,
                valence_tag=valence_tag,
                salience_hit=salience_hit,
                entities=ents,
                entity_importance=entity_importance,
            )
            size = round(0.35 + 0.9 * scores["retain"], 4)

            nodes.append({
                "id": mid,
                "type": "memory",
                "label": label,
                "text": text,
                "status": status,
                "kind": kind,
                "source": source,
                "pinned": pinned,
                "access_count": access_count,
                "access_day_count": access_day_count,
                "created_at": row["created_at"],
                "entities": ents,
                "supersedes_id": supersedes_id,
                "valence_tag": valence_tag,
                "scores": scores,
                "size": size,
            })

            if supersedes_id:
                edges.append({
                    "id": f"sup:{mid}->{supersedes_id}",
                    "source": mid,
                    "target": supersedes_id,
                    "type": "supersedes",
                })

            if include_entities:
                for ent in ents:
                    eid = f"ent:{ent.casefold()}"
                    if eid not in entity_ids:
                        entity_ids.add(eid)
                        ie = float(entity_importance.get(ent.casefold(), 0.0))
                        nodes.append({
                            "id": eid,
                            "type": "entity",
                            "label": ent,
                            "text": ent,
                            "status": "active",
                            "kind": "entity",
                            "source": "",
                            "pinned": False,
                            "access_count": 0,
                            "created_at": None,
                            "entities": [],
                            "supersedes_id": None,
                            "valence_tag": "neutral",
                            "scores": {
                                "retain": round(ie, 4),
                                "importance": round(ie, 4),
                                "salience": 0.0,
                                "spacing": 0.0,
                                "connectivity": round(ie, 4),
                                "valence": 0.25,
                                "access": 0.0,
                            },
                            "size": round(0.25 + 0.85 * ie, 4),
                        })
                    edges.append({
                        "id": f"men:{mid}->{eid}",
                        "source": mid,
                        "target": eid,
                        "type": "mentions",
                    })

        edges = [
            e for e in edges
            if e["type"] != "supersedes" or e["target"] in mem_ids
        ]

        if include_entities:
            try:
                from memory.memorize import ensure_entity_relations_schema

                ensure_entity_relations_schema(conn)
                for e in relations_as_graph_edges(
                    conn, user_id=uid, limit=max(int(limit) * 2, 500)
                ):
                    for endpoint in (e["source"], e["target"]):
                        if endpoint not in entity_ids and endpoint not in mem_ids:
                            label = endpoint.removeprefix("ent:")
                            entity_ids.add(endpoint)
                            ie = float(entity_importance.get(label.casefold(), 0.0))
                            nodes.append({
                                "id": endpoint,
                                "type": "entity",
                                "label": label,
                                "text": label,
                                "status": "active",
                                "kind": "entity",
                                "source": "",
                                "pinned": False,
                                "access_count": 0,
                                "created_at": None,
                                "entities": [],
                                "supersedes_id": None,
                                "valence_tag": "neutral",
                                "scores": {
                                    "retain": round(ie, 4),
                                    "importance": round(ie, 4),
                                    "salience": 0.0,
                                    "spacing": 0.0,
                                    "connectivity": round(ie, 4),
                                    "valence": 0.25,
                                    "access": 0.0,
                                },
                                "size": round(0.25 + 0.85 * ie, 4),
                            })
                    edges.append(e)
            except Exception as ex:
                log.debug("graph_export: entity_relations skipped: %s", ex)

        mem_nodes = [n for n in nodes if n.get("type") == "memory"]
        ent_nodes = [n for n in nodes if n.get("type") == "entity"]
        mem_nodes.sort(
            key=lambda n: float((n.get("scores") or {}).get("retain") or n.get("size") or 0),
            reverse=True,
        )
        ent_nodes.sort(key=lambda n: float(n.get("size") or 0), reverse=True)
        mem_nodes = mem_nodes[:_MAX_MEMORIES]
        ent_nodes = ent_nodes[:_MAX_ENTITIES]
        keep_ids = {n["id"] for n in mem_nodes} | {n["id"] for n in ent_nodes}
        nodes = mem_nodes + ent_nodes
        edges = [e for e in edges if e.get("source") in keep_ids and e.get("target") in keep_ids]
        edges.sort(
            key=lambda e: (0 if e.get("type") == "supersedes" else 1, -float(e.get("weight") or 0)),
        )
        edges = edges[:_MAX_EDGES]
        
        return {
            "nodes": nodes,
            "edges": edges,
            "meta": {
                "user_id": uid,
                "memory_count": len(mem_nodes),
                "entity_count": len(ent_nodes),
                "edge_count": len(edges),
                "include_history": include_history,
                "include_entities": include_entities,
                "limit": limit,
                "theme": "galaxy",
                "max_memories": _MAX_MEMORIES,
                "max_entities": _MAX_ENTITIES,
                "max_edges": _MAX_EDGES,
            },
            "legend": _legend(),
        }
    finally:
        if owns_conn and conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def _legend() -> dict[str, Any]:
    return {
        "size": "retain tendency (memories) / I_e (entities)",
        "rim_arcs": ["salience", "spacing", "connectivity", "valence", "access"],
        "colors": {
            "neg": "#ff6b6b",
            "neutral": "#c0c8d8",
            "pos": "#7ad7f0",
            "entity": "#c9a0ff",
            "monthly": "#ffd27a",
            "pinned": "#51d4c8",
            "superseded": "#4a3a6a",
        },
    }


def list_entity_relations(
    conn: sqlite3.Connection,
    *,
    user_id: str,
    limit: int = 500,
) -> list[dict[str, Any]]:
    from memory.memorize import ensure_entity_relations_schema

    ensure_entity_relations_schema(conn)
    rows = conn.execute(
        """
        SELECT entity_a, entity_b, relation, weight, memory_id, updated_at
        FROM entity_relations
        WHERE user_id = ?
        ORDER BY weight DESC, updated_at DESC
        LIMIT ?
        """,
        (user_id, int(limit)),
    ).fetchall()
    return [dict(r) for r in rows]


def relations_as_graph_edges(
    conn: sqlite3.Connection,
    *,
    user_id: str,
    limit: int = 500,
) -> list[dict[str, Any]]:
    """Emit edges compatible with Memory Graph Studio (entity node ids)."""
    edges = []
    for rel in list_entity_relations(conn, user_id=user_id, limit=limit):
        a = rel["entity_a"]
        b = rel["entity_b"]
        edges.append({
            "id": f"rel:{a.casefold()}::{b.casefold()}::{rel['relation']}",
            "source": f"ent:{a.casefold()}",
            "target": f"ent:{b.casefold()}",
            "type": rel["relation"] or "related_to",
            "weight": float(rel.get("weight") or 1.0),
        })
    return edges
