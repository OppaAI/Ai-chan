"""
memory/entity_graph.py

Phase D: thin entity-relation layer (not a full graph database).

Stores co-mention / explicit links between entity labels so multi-hop style
queries and the Memory Graph Studio can show entity–entity edges without
Graphiti-scale infrastructure.

Design:
  - One SQLite table ``entity_relations`` (user-scoped)
  - Built primarily from co-occurrence on the same memory fact
  - Idempotent schema ensure; no vector changes; no LLM
"""
from __future__ import annotations

import json
import sqlite3
from itertools import combinations
from typing import Any, Iterable

from system.log import get_logger

log = get_logger(__name__)

RELATION_CO_MENTION = "co_mentions"
RELATION_RELATED = "related_to"

_DDL = """
CREATE TABLE IF NOT EXISTS entity_relations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     TEXT NOT NULL,
    entity_a    TEXT NOT NULL,
    entity_b    TEXT NOT NULL,
    relation    TEXT NOT NULL DEFAULT 'co_mentions',
    weight      REAL NOT NULL DEFAULT 1.0,
    memory_id   TEXT,
    updated_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_entity_rel_user ON entity_relations(user_id);
CREATE INDEX IF NOT EXISTS idx_entity_rel_a ON entity_relations(user_id, entity_a);
CREATE INDEX IF NOT EXISTS idx_entity_rel_b ON entity_relations(user_id, entity_b);
CREATE UNIQUE INDEX IF NOT EXISTS idx_entity_rel_pair
    ON entity_relations(user_id, entity_a, entity_b, relation);
"""


def _norm_entity(e: str) -> str:
    return (e or "").strip()


def _ordered_pair(a: str, b: str) -> tuple[str, str]:
    """Canonical unordered pair key (casefold order, display preserves first-seen casing via callers)."""
    aa, bb = _norm_entity(a), _norm_entity(b)
    if aa.casefold() <= bb.casefold():
        return aa, bb
    return bb, aa


def ensure_entity_relations_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_DDL)
    conn.commit()


def _entities_from_row(raw: Any) -> list[str]:
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


def upsert_co_mentions(
    conn: sqlite3.Connection,
    *,
    user_id: str,
    entities: Iterable[str],
    memory_id: str | None = None,
    updated_at: str | None = None,
) -> int:
    """Record co-mention pairs for entities on one memory. Returns pairs touched."""
    from memory.vecstore import utc_now_iso

    ents = []
    seen: set[str] = set()
    for e in entities:
        n = _norm_entity(e)
        if not n:
            continue
        key = n.casefold()
        if key in seen:
            continue
        seen.add(key)
        ents.append(n)
    if len(ents) < 2:
        return 0

    ensure_entity_relations_schema(conn)
    now = updated_at or utc_now_iso()
    touched = 0
    for a, b in combinations(ents, 2):
        ea, eb = _ordered_pair(a, b)
        if ea.casefold() == eb.casefold():
            continue
        conn.execute(
            """
            INSERT INTO entity_relations (user_id, entity_a, entity_b, relation, weight, memory_id, updated_at)
            VALUES (?, ?, ?, ?, 1.0, ?, ?)
            ON CONFLICT(user_id, entity_a, entity_b, relation) DO UPDATE SET
                weight = entity_relations.weight + 1.0,
                memory_id = COALESCE(excluded.memory_id, entity_relations.memory_id),
                updated_at = excluded.updated_at
            """,
            (user_id, ea, eb, RELATION_CO_MENTION, memory_id, now),
        )
        touched += 1
    return touched


def rebuild_entity_relations(
    conn: sqlite3.Connection,
    *,
    user_id: str,
    clear: bool = True,
) -> dict[str, int]:
    """Rebuild co-mention edges from memories.entities JSON for one user."""
    ensure_entity_relations_schema(conn)
    cols = {str(r[1]) for r in conn.execute("PRAGMA table_info(memories)").fetchall()}
    if "entities" not in cols:
        return {"pairs": 0, "memories": 0, "note": 1}

    if clear:
        conn.execute(
            "DELETE FROM entity_relations WHERE user_id = ? AND relation = ?",
            (user_id, RELATION_CO_MENTION),
        )

    rows = conn.execute(
        """
        SELECT id, entities FROM memories
        WHERE user_id = ?
          AND (status IS NULL OR status = 'active')
        """,
        (user_id,),
    ).fetchall()

    pairs = 0
    for row in rows:
        ents = _entities_from_row(row["entities"])
        pairs += upsert_co_mentions(
            conn, user_id=user_id, entities=ents, memory_id=str(row["id"])
        )
    conn.commit()
    log.info("entity_relations rebuild user=%s memories=%d pairs=%d", user_id, len(rows), pairs)
    return {"pairs": pairs, "memories": len(rows)}


def list_entity_relations(
    conn: sqlite3.Connection,
    *,
    user_id: str,
    limit: int = 500,
) -> list[dict[str, Any]]:
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
            "type": rel["relation"] or RELATION_RELATED,
            "weight": float(rel.get("weight") or 1.0),
        })
    return edges
