"""
Knowledge-only graph export for Knowledge Graph Studio.

Nodes: learned_chunks (type=knowledge) + entity hubs.
Edges: about (chunk → entity), same_doc (chunks sharing doc_id).

Size/brightness driven by importance score (access + recency + degree).
No personal memory or experience nodes.
"""
from __future__ import annotations

import math
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any

from system.log import get_logger

log = get_logger(__name__)

# Prefer knowledge module path helpers when available
try:
    from cognition.knowledge import KNOWLEDGE_DB_PATH, _connect as knowledge_connect
except Exception:
    KNOWLEDGE_DB_PATH = os.getenv("KNOWLEDGE_DB_PATH", "knowledge/knowledge.db")
    knowledge_connect = None  # type: ignore

try:
    from memory.memorize import entities_from_json
except Exception:  # pragma: no cover
    import json

    def entities_from_json(raw: Any) -> list[str]:
        if not raw:
            return []
        if isinstance(raw, list):
            return [str(x).strip() for x in raw if str(x).strip()]
        try:
            data = json.loads(raw) if isinstance(raw, str) else raw
            if isinstance(data, list):
                return [str(x).strip() for x in data if str(x).strip()]
        except Exception:
            pass
        return []


def _env_int(name: str, default: int, *, floor: int = 0) -> int:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return max(floor, default)
    try:
        return max(floor, int(str(raw).strip()))
    except (TypeError, ValueError):
        log.warning("Invalid %s=%r; using %s", name, raw, default)
        return max(floor, default)


_MAX_CHUNKS = _env_int("KNOWLEDGE_STUDIO_MAX_CHUNKS", 200, floor=1)
_MAX_ENTITIES = _env_int("KNOWLEDGE_STUDIO_MAX_ENTITIES", 120, floor=0)
_MAX_EDGES = _env_int("KNOWLEDGE_STUDIO_MAX_EDGES", 300, floor=0)


def _parse_ts(raw: Any) -> datetime | None:
    if not raw or str(raw).strip().lower() in ("", "never", "none"):
        return None
    try:
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _recency_01(raw: Any, *, half_life_days: float = 45.0) -> float:
    dt = _parse_ts(raw)
    if dt is None:
        return 0.25
    now = datetime.now(timezone.utc)
    days = max(0.0, (now - dt).total_seconds() / 86400.0)
    return float(max(0.05, min(1.0, 0.5 ** (days / max(half_life_days, 1.0)))))


def _chunk_importance(
    *,
    access_count: int,
    last_accessed: Any,
    created_at: Any,
    entity_count: int,
    status: str,
) -> dict[str, float]:
    """0..1 factors + composite importance for size/brightness."""
    ac = max(0, int(access_count or 0))
    access = min(1.0, math.log1p(ac) / math.log1p(40.0))
    rec = _recency_01(last_accessed or created_at)
    connectivity = min(1.0, entity_count / 6.0)
    status_l = (status or "active").strip().lower()
    if status_l not in ("active", "", "none"):
        access *= 0.35
        rec *= 0.35

    # Weighted composite (knowledge has no personal retain R)
    importance = (
        0.45 * access
        + 0.30 * rec
        + 0.25 * connectivity
    )
    importance = float(max(0.05, min(1.0, importance)))
    size = round(0.20 + 1.10 * (importance ** 1.25), 4)
    return {
        "access": round(access, 4),
        "recency": round(rec, 4),
        "connectivity": round(connectivity, 4),
        "importance": round(importance, 4),
        "retain": round(importance, 4),  # alias so Memory UI helpers can reuse
        "size": size,
    }


def _open_conn(user_id: str | None) -> sqlite3.Connection:
    if knowledge_connect is not None:
        return knowledge_connect(user_id)
    path = KNOWLEDGE_DB_PATH
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def export_knowledge_graph(
    *,
    user_id: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Return {nodes, edges, meta} for knowledge-only studio."""
    try:
        from system.userspace import current_user_id
        uid = user_id or current_user_id()
    except Exception:
        uid = user_id or "default"

    nodes: list[dict] = []
    edges: list[dict] = []
    meta: dict[str, Any] = {
        "user_id": uid,
        "store": "knowledge",
        "include_memory": False,
        "include_experience": False,
    }

    try:
        conn = _open_conn(uid)
    except Exception as exc:
        log.warning("knowledge graph: open failed: %s", exc)
        meta["error"] = "open failed"
        return {"nodes": [], "edges": [], "meta": meta}

    try:
        # Discover columns
        try:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(learned_chunks)").fetchall()}
        except sqlite3.OperationalError:
            meta["error"] = "no learned_chunks table"
            return {"nodes": [], "edges": [], "meta": meta}

        has_entities = "entities" in cols
        has_status = "status" in cols
        has_access = "access_count" in cols
        has_last = "last_accessed" in cols

        select = [
            "c.id", "c.text", "c.chunk_index", "c.created_at", "c.doc_id",
            "d.title AS doc_title", "d.source AS doc_source", "d.kind AS doc_kind",
        ]
        if has_entities:
            select.append("c.entities")
        if has_status:
            select.append("c.status")
        if has_access:
            select.append("c.access_count")
        if has_last:
            select.append("c.last_accessed")

        try:
            req = int(limit) if limit is not None else _MAX_CHUNKS
        except (TypeError, ValueError):
            req = _MAX_CHUNKS
        if req < 1:
            req = _MAX_CHUNKS
        fetch_n = min(max(req * 2, req), _MAX_CHUNKS * 2)

        sql = f"""
            SELECT {", ".join(select)}
            FROM learned_chunks c
            LEFT JOIN learned_docs d ON d.id = c.doc_id
            WHERE c.user_id = ?
        """
        params: list[Any] = [uid]
        if has_status:
            sql += " AND (c.status = 'active' OR c.status IS NULL OR c.status = '')"
        sql += " ORDER BY c.created_at DESC LIMIT ?"
        params.append(int(fetch_n))

        rows = conn.execute(sql, params).fetchall()
    except Exception as exc:
        log.warning("knowledge graph: query failed: %s", exc)
        meta["error"] = str(exc)
        try:
            conn.close()
        except Exception:
            pass
        return {"nodes": [], "edges": [], "meta": meta}

    # Build chunk nodes + entity degree
    entity_degree: dict[str, int] = {}
    entity_label: dict[str, str] = {}  # casefold key → display string

    chunk_nodes: list[dict] = []
    doc_chunks: dict[str, list[str]] = {}

    for row in rows:
        rid = dict(row) if not isinstance(row, dict) else row
        cid = str(rid.get("id") or "")
        if not cid:
            continue
        text = (rid.get("text") or "")[:500]
        ents = entities_from_json(rid.get("entities") if has_entities else "[]")
        for e in ents:
            k = e.casefold()
            entity_degree[k] = entity_degree.get(k, 0) + 1
            # prefer first non-empty original form
            if k not in entity_label and e.strip():
                entity_label[k] = e.strip()
        status = str(rid.get("status") or "active") if has_status else "active"
        ac = int(rid.get("access_count") or 0) if has_access else 0
        scores = _chunk_importance(
            access_count=ac,
            last_accessed=rid.get("last_accessed") if has_last else None,
            created_at=rid.get("created_at"),
            entity_count=len(ents),
            status=status,
        )
        label = (rid.get("doc_title") or text or cid)[:80]
        node = {
            "id": cid,
            "type": "knowledge",
            "label": label,
            "text": text,
            "status": status,
            "doc_id": rid.get("doc_id"),
            "doc_title": rid.get("doc_title") or "",
            "doc_source": rid.get("doc_source") or "",
            "doc_kind": rid.get("doc_kind") or "",
            "chunk_index": rid.get("chunk_index"),
            "created_at": rid.get("created_at"),
            "access_count": ac,
            "entities": ents,
            "scores": scores,
            "size": scores["size"],
            "valence_tag": "neutral",
        }
        chunk_nodes.append(node)
        doc_id = str(rid.get("doc_id") or "")
        if doc_id:
            doc_chunks.setdefault(doc_id, []).append(cid)

    # Prefer high importance within window
    chunk_nodes.sort(key=lambda n: float((n.get("scores") or {}).get("importance") or 0), reverse=True)
    chunk_nodes = chunk_nodes[: min(req, _MAX_CHUNKS)]
    keep_chunk_ids = {n["id"] for n in chunk_nodes}
    nodes.extend(chunk_nodes)

    # Entity hubs (top by degree among visible chunks)
    ent_list = sorted(entity_degree.items(), key=lambda kv: -kv[1])[:_MAX_ENTITIES]
    max_deg = max((d for _, d in ent_list), default=1) or 1
    for name, deg in ent_list:
        # Only include entities mentioned by kept chunks
        mentioned = False
        for n in chunk_nodes:
            if any(e.casefold() == name for e in (n.get("entities") or [])):
                mentioned = True
                break
        if not mentioned:
            continue
        imp = min(1.0, deg / max_deg)
        size = round(0.25 + 0.9 * (imp ** 1.1), 4)
        eid = f"ent:{name}"
        label = entity_label.get(name, name)
        nodes.append({
            "id": eid,
            "type": "entity",
            "label": label,
            "text": label,
            "degree": deg,
            "size": size,
            "scores": {
                "importance": round(imp, 4),
                "retain": round(imp, 4),
                "connectivity": round(imp, 4),
                "access": 0.0,
                "recency": 0.5,
            },
            "valence_tag": "neutral",
        })
        keep_chunk_ids.add(eid)

    entity_ids = {n["id"] for n in nodes if n["type"] == "entity"}

    # about edges: chunk → entity
    for n in chunk_nodes:
        for e in n.get("entities") or []:
            eid = f"ent:{e.casefold()}"
            if eid not in entity_ids:
                continue
            edges.append({
                "source": n["id"],
                "target": eid,
                "type": "about",
                "weight": 1.0,
            })

    # same_doc edges (light): consecutive chunks in a doc among kept set
    for _doc, cids in doc_chunks.items():
        kept = [c for c in cids if c in keep_chunk_ids]
        for a, b in zip(kept, kept[1:]):
            edges.append({
                "source": a,
                "target": b,
                "type": "same_doc",
                "weight": 0.5,
            })

    # Cap edges
    edges.sort(key=lambda e: (0 if e.get("type") == "about" else 1, -float(e.get("weight") or 0)))
    edges = edges[:_MAX_EDGES]

    # Drop edges to missing nodes
    id_set = {n["id"] for n in nodes}
    edges = [e for e in edges if e["source"] in id_set and e["target"] in id_set]

    meta.update({
        "chunk_count": len([n for n in nodes if n["type"] == "knowledge"]),
        "entity_count": len([n for n in nodes if n["type"] == "entity"]),
        "edge_count": len(edges),
        "max_chunks": _MAX_CHUNKS,
        "max_entities": _MAX_ENTITIES,
        "max_edges": _MAX_EDGES,
    })

    try:
        conn.close()
    except Exception:
        pass

    return {"nodes": nodes, "edges": edges, "meta": meta}
