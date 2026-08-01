"""
memory/graph_export.py

Phase C: export personal memory as a node/edge graph for visualization.
Phase D polish: merge entity co-mention edges from entity_relations.

Read-only. Tolerates pre-Phase-A DBs (missing status/entities columns).
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from system.log import get_logger
from system.userspace import current_user_id

log = get_logger(__name__)


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


def export_memory_graph(
    *,
    user_id: str | None = None,
    limit: int = 200,
    include_history: bool = True,
    include_entities: bool = True,
    conn: sqlite3.Connection | None = None,
) -> dict[str, Any]:
    """Build a graph dict: {nodes, edges, meta}.

    Node types:
      - memory: one per fact row
      - entity: hub nodes shared across facts (when include_entities)

    Edge types:
      - supersedes: newer memory → older memory it replaced
      - mentions: memory → entity
      - co_mentions: entity → entity (Phase D entity_relations)
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
            }
        # Minimal DDL — do not recreate schema; just open
        try:
            conn = initialize_store_db(str(db_path), "PRAGMA journal_mode = WAL;", user_id=uid, vector=True)
        except Exception as e:
            log.warning("graph_export: open failed: %s", e)
            return {"nodes": [], "edges": [], "meta": {"user_id": uid, "error": str(e)}}

    try:
        cols = _table_columns(conn)
        if "id" not in cols:
            return {"nodes": [], "edges": [], "meta": {"user_id": uid, "count": 0, "note": "no memories table"}}

        has_status = "status" in cols
        has_entities = "entities" in cols
        has_kind = "kind" in cols
        has_source = "source" in cols
        has_supersedes = "supersedes_id" in cols

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

        sql = f"SELECT {', '.join(select_cols)} FROM memories WHERE user_id = ?"
        params: list[Any] = [uid]
        if has_status and not include_history:
            sql += " AND (status = 'active' OR status IS NULL)"
        sql += " ORDER BY created_at DESC"
        if limit and limit > 0:
            sql += " LIMIT ?"
            params.append(int(limit))

        rows = conn.execute(sql, params).fetchall()

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
            source = str(row["source"]) if has_source and row["source"] else ""
            ents = _entities_from_raw(row["entities"]) if has_entities else []
            supersedes_id = (
                str(row["supersedes_id"]) if has_supersedes and row["supersedes_id"] else None
            )

            nodes.append({
                "id": mid,
                "type": "memory",
                "label": label,
                "text": text,
                "status": status,
                "kind": kind,
                "source": source,
                "pinned": bool(row["pinned"]),
                "access_count": int(row["access_count"] or 0),
                "created_at": row["created_at"],
                "entities": ents,
                "supersedes_id": supersedes_id,
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
                        })
                    edges.append({
                        "id": f"men:{mid}->{eid}",
                        "source": mid,
                        "target": eid,
                        "type": "mentions",
                    })

        # Drop supersedes edges whose target fell outside the limit window
        edges = [
            e for e in edges
            if e["type"] != "supersedes" or e["target"] in mem_ids
        ]

        # Phase D: entity ↔ entity co-mentions
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
                            })
                    edges.append(e)
            except Exception as ex:
                log.debug("graph_export: entity_relations skipped: %s", ex)

        return {
            "nodes": nodes,
            "edges": edges,
            "meta": {
                "user_id": uid,
                "memory_count": len(mem_ids),
                "entity_count": len(entity_ids),
                "edge_count": len(edges),
                "include_history": include_history,
                "include_entities": include_entities,
                "limit": limit,
            },
        }
    finally:
        if owns_conn and conn is not None:
            try:
                conn.close()
            except Exception:
                pass


# ── entity relation reads (Studio-facing) ────────────────────────────────────
# The write side (upsert_co_mentions, schema) lives in memory/memorize.py;
# these read-only helpers serve the Studio's edge export.

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
