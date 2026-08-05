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
_MAX_KNOWLEDGE = _env_int("MEMORY_STUDIO_MAX_KNOWLEDGE", 80, floor=0)
_MAX_EXPERIENCE = _env_int("MEMORY_STUDIO_MAX_EXPERIENCE", 40, floor=0)
_INCLUDE_KNOWLEDGE = os.getenv("MEMORY_STUDIO_INCLUDE_KNOWLEDGE", "1").lower() in {"1", "true", "yes", "on"}
_INCLUDE_EXPERIENCE = os.getenv("MEMORY_STUDIO_INCLUDE_EXPERIENCE", "1").lower() in {"1", "true", "yes", "on"}


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


def _norm_date(value: Any, *, end: bool) -> str | None:
    """Normalize a YYYY-MM-DD or full ISO timestamp for created_at comparisons.

    Bare dates are padded to start-of-day (end=False) or end-of-day (end=True)
    so string comparison against ISO-8601 created_at stays correct.
    """
    v = str(value or "").strip()
    if not v:
        return None
    v = v[:19]
    if len(v) == 10:
        return v + ("T23:59:59.999999" if end else "T00:00:00")
    return v


_FULL_DATE_RE = re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)")
_MONTH_ONLY_RE = re.compile(r"(?<!\d)(\d{4})-(\d{2})(?!\d)")


def _memory_date_span(text: Any, created_at: Any = None) -> tuple[str, str] | None:
    """Return the (start_iso, end_iso) span of a memory's original date.

    Precedence:
      - a full date in the text ([YYYY-MM-DD] or bare YYYY-MM-DD, including
        the "Daily journal of YYYY-MM-DD:" blobs) → that exact day;
      - a month tag ([YYYY-MM] / bare YYYY-MM) → the full month;
      - otherwise the created_at timestamp.

    Consolidation rewrites created_at (scene re-summary, monthly/journal
    add_raw) but never rewrites the text tag, so the tag is the stable
    original date. Month-tagged memories match any filter day inside the
    month.
    """
    import calendar

    t = str(text or "")
    m = _FULL_DATE_RE.search(t)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= mo <= 12 and 1 <= d <= 31:
            return (
                f"{y:04d}-{mo:02d}-{d:02d}T00:00:00",
                f"{y:04d}-{mo:02d}-{d:02d}T23:59:59.999999",
            )
    m = _MONTH_ONLY_RE.search(t)
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
        if 1 <= mo <= 12:
            last = calendar.monthrange(y, mo)[1]
            return (
                f"{y:04d}-{mo:02d}-01T00:00:00",
                f"{y:04d}-{mo:02d}-{last:02d}T23:59:59.999999",
            )
    s = str(created_at or "").strip()[:19]
    if s:
        return (s, s)
    return None


def _date_spans_overlap(span: tuple[str, str] | None, from_dt: str | None, to_dt: str | None) -> bool:
    """True when a memory's [start, end] overlaps the filter window [from, to]."""
    if span is None:
        return from_dt is None and to_dt is None
    start, end = span
    if from_dt and end < from_dt:
        return False
    if to_dt and start > to_dt:
        return False
    return True


def _resolve_db_path(user_id: str) -> Path:
    import os
    from memory.vecstore import resolve_user_db_path

    env = os.getenv("SQLITE_MEMORY_PATH", "").strip()
    if env:
        return Path(env).expanduser()
    return resolve_user_db_path("memory/memory.db", user_id=user_id)


def _valence_score(tag: Any, score: Any = None) -> float:
    """0..1 rim. Prefer 5-pt score when present."""
    if score is not None and str(score).strip() != "":
        try:
            s = max(-2, min(2, int(score)))
            return round(0.25 + 0.30 * abs(s), 4)
        except (TypeError, ValueError):
            pass
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
    valence_score: Any = None,
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
        val = _valence_score(valence_tag, valence_score)
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
    val = _valence_score(valence_tag, valence_score)
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
    include_knowledge: bool | None = None,
    include_experience: bool | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
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
    if include_knowledge is None:
        include_knowledge = _INCLUDE_KNOWLEDGE
    if include_experience is None:
        include_experience = _INCLUDE_EXPERIENCE
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
        has_valence_score = "valence_score" in cols
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
        if has_valence_score:
            select_cols.append("valence_score")
        if has_salience:
            select_cols.append("salience_hit")
        if has_day_count:
            select_cols.append("access_day_count")

        sql = f"SELECT {', '.join(select_cols)} FROM memories WHERE user_id = ?"
        params: list[Any] = [uid]
        if has_status and not include_history:
            sql += " AND (status = 'active' OR status IS NULL)"
        from_dt = _norm_date(date_from, end=False)
        to_dt = _norm_date(date_to, end=True)

        try:
            req_limit = int(limit) if limit is not None else 200
        except (TypeError, ValueError):
            req_limit = 200
        if req_limit < 1:
            req_limit = 200
        effective_limit = min(req_limit, _MAX_MEMORIES)
        # Over-fetch so retain ranking can prefer strong older rows in a wider window
        fetch_n = min(max(effective_limit * 3, effective_limit), max(_MAX_MEMORIES * 3, effective_limit))

        sql += " ORDER BY created_at DESC"
        # Date filter runs in Python against the stable text date tag
        # (created_at is rewritten by consolidation). When filtering, fetch
        # all rows so in-range tagged memories aren't cut off by the LIMIT.
        if not (from_dt or to_dt) and fetch_n > 0:
            sql += " LIMIT ?"
            params.append(int(fetch_n))

        rows = conn.execute(sql, params).fetchall()

        # Apply the date filter on the memory's original date (text tag) with
        # created_at fallback, month tags matching any day within the month.
        if from_dt or to_dt:
            rows = [
                r for r in rows
                if _date_spans_overlap(_memory_date_span(r["memory"], r["created_at"]), from_dt, to_dt)
            ]

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
            imprint = bool(_MONTHLY_RE.match(text))
            if imprint:
                kind = "imprint"
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
            valence_score = None
            if has_valence_score and row["valence_score"] is not None:
                try:
                    valence_score = int(row["valence_score"])
                except (TypeError, ValueError):
                    valence_score = None
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
                valence_score=valence_score,
                salience_hit=salience_hit,
                entities=ents,
                entity_importance=entity_importance,
            )
            # Wider spread: weak ≈ 0.2, strong ≈ 1.3
            r = float(scores["retain"])
            size = round(0.20 + 1.10 * (r ** 1.25), 4)

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
                    "valence_score": valence_score,
                    "imprint": imprint,
                    "size": size,
                    "scores": scores,
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
                        "weight": 0.9,
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
                    conn, user_id=uid, limit=max(int(fetch_n) * 2, 500)
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


        # Phase 13b: knowledge + experience layers (entity overlap)
        if include_knowledge and _MAX_KNOWLEDGE > 0:
            try:
                _add_knowledge_layer(conn, uid, nodes, edges, entity_ids, mem_ids, date_from=from_dt, date_to=to_dt)
            except Exception as ex:
                log.debug("graph_export: knowledge layer skipped: %s", ex)
        if include_experience and _MAX_EXPERIENCE > 0:
            try:
                _add_experience_layer(conn, uid, nodes, edges, entity_ids, mem_ids, date_from=from_dt, date_to=to_dt)
            except Exception as ex:
                log.debug("graph_export: experience layer skipped: %s", ex)

        mem_nodes = [n for n in nodes if n.get("type") == "memory"]
        ent_nodes = [n for n in nodes if n.get("type") == "entity"]
        kb_nodes = [n for n in nodes if n.get("type") == "knowledge"]
        exp_nodes = [n for n in nodes if n.get("type") == "experience"]
        mem_nodes.sort(
            key=lambda n: float((n.get("scores") or {}).get("retain") or n.get("size") or 0),
            reverse=True,
        )
        # Prefer high retain among the over-fetched window
        mem_nodes = mem_nodes[:effective_limit]
        ent_nodes.sort(key=lambda n: float(n.get("size") or 0), reverse=True)
        ent_nodes = ent_nodes[:_MAX_ENTITIES]
        kb_nodes = kb_nodes[:_MAX_KNOWLEDGE]
        exp_nodes = exp_nodes[:_MAX_EXPERIENCE]
        keep_ids = (
            {n["id"] for n in mem_nodes}
            | {n["id"] for n in ent_nodes}
            | {n["id"] for n in kb_nodes}
            | {n["id"] for n in exp_nodes}
        )
        nodes = mem_nodes + ent_nodes + kb_nodes + exp_nodes
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
                "max_knowledge": _MAX_KNOWLEDGE,
                "max_experience": _MAX_EXPERIENCE,
                "include_knowledge": include_knowledge,
                "include_experience": include_experience,
                "date_from": from_dt,
                "date_to": to_dt,
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
            "neg": "#3de0ff",
            "neutral": "#8a9bb8",
            "pos": "#f0c14a",
            "entity": "#b794f6",
            "imprint": "#c651a8",
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


def _add_knowledge_layer(
    conn: sqlite3.Connection,
    uid: str,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    entity_ids: set[str],
    mem_ids: set[str],
    date_from: str | None = None,
    date_to: str | None = None,
) -> None:
    """Add learned_chunks as knowledge nodes + about / grounded_in edges."""
    kb_conn = conn
    owns = False
    try:
        cols = {r[1] for r in kb_conn.execute("PRAGMA table_info(learned_chunks)").fetchall()}
    except Exception:
        cols = set()
    if not cols or "id" not in cols:
        try:
            from memory.knowledge import _connect as kb_connect
            kb_conn = kb_connect(uid)
            owns = True
            cols = {r[1] for r in kb_conn.execute("PRAGMA table_info(learned_chunks)").fetchall()}
        except Exception as ex:
            log.debug("knowledge layer open failed: %s", ex)
            return
    if not cols or "id" not in cols:
        if owns:
            try:
                kb_conn.close()
            except Exception:
                pass
        return
    has_ent = "entities" in cols
    has_status = "status" in cols
    sql = "SELECT id, text, created_at"
    if has_ent:
        sql += ", entities"
    if has_status:
        sql += ", status"
    sql += " FROM learned_chunks WHERE user_id = ?"
    kb_params: list[Any] = [uid]
    if has_status:
        sql += " AND (status = 'active' OR status IS NULL)"
    if date_from:
        sql += " AND created_at >= ?"
        kb_params.append(date_from)
    if date_to:
        sql += " AND created_at <= ?"
        kb_params.append(date_to)
    sql += " ORDER BY created_at DESC LIMIT ?"
    kb_params.append(_MAX_KNOWLEDGE)
    try:
        rows = kb_conn.execute(sql, kb_params).fetchall()
    except Exception as ex:
        log.debug("knowledge layer query failed: %s", ex)
        if owns:
            try:
                kb_conn.close()
            except Exception:
                pass
        return

    mem_ents: dict[str, set[str]] = {}
    for e in edges:
        if e.get("type") == "mentions":
            mid, eid = e.get("source"), e.get("target")
            if mid in mem_ids and isinstance(eid, str) and eid.startswith("ent:"):
                mem_ents.setdefault(str(mid), set()).add(eid.removeprefix("ent:").casefold())

    try:
        grounded_seen: set[str] = set()
        for row in rows:
            kid = f"kb:{row['id']}"
            text = (row["text"] or "").strip()
            label = text if len(text) <= 72 else text[:69] + "…"
            ents = _entities_from_raw(row["entities"]) if has_ent else []
            nodes.append({
                "id": kid,
                "type": "knowledge",
                "label": label,
                "text": text,
                "status": "active",
                "kind": "knowledge",
                "source": "learned_chunks",
                "pinned": False,
                "access_count": 0,
                "created_at": row["created_at"] if "created_at" in row.keys() else None,
                "entities": ents,
                "supersedes_id": None,
                "valence_tag": "neutral",
                "scores": {
                    "retain": 0.55, "salience": 0.4, "spacing": 0.0,
                    "connectivity": 0.0, "valence": 0.25, "access": 0.0,
                },
                "size": 0.45,
            })
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
                        "valence_tag": "neutral",
                        "scores": {
                            "retain": 0.3, "importance": 0.3, "salience": 0.0,
                            "spacing": 0.0, "connectivity": 0.3, "valence": 0.25, "access": 0.0,
                        },
                        "size": 0.35,
                    })
                edges.append({
                    "id": f"about:{kid}->{eid}",
                    "source": kid,
                    "target": eid,
                    "type": "about",
                    "weight": 1.0,
                })
                ent_cf = ent.casefold()
                for mid, mes in mem_ents.items():
                    if ent_cf in mes:
                        gid = f"grounded:{mid}->{kid}"
                        if gid not in grounded_seen:
                            grounded_seen.add(gid)
                            edges.append({
                                "id": gid,
                                "source": mid,
                                "target": kid,
                                "type": "grounded_in",
                                "weight": 1.0,
                            })
    finally:
        if owns:
            try:
                kb_conn.close()
            except Exception:
                pass


def _add_experience_layer(
    conn: sqlite3.Connection,
    uid: str,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    entity_ids: set[str],
    mem_ids: set[str],
    date_from: str | None = None,
    date_to: str | None = None,
) -> None:
    """Add experiences as nodes + about / practiced_in edges (separate DB)."""
    exp_conn = None
    owns = False
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(experiences)").fetchall()}
        exp_conn = conn
    except Exception:
        cols = set()
    if not cols or "id" not in cols:
        try:
            from agentic.experience import _connect as exp_connect
            exp_conn = exp_connect(uid)
            owns = True
            cols = {r[1] for r in exp_conn.execute("PRAGMA table_info(experiences)").fetchall()}
        except Exception as ex:
            log.debug("experience layer open failed: %s", ex)
            return
    if not cols or "id" not in cols or exp_conn is None:
        return
    has_ent = "entities" in cols
    sql = "SELECT id, goal, record_text, outcome, answer_excerpt, created_at"
    if has_ent:
        sql += ", entities"
    sql += " FROM experiences WHERE user_id = ?"
    exp_params: list[Any] = [uid]
    if date_from:
        sql += " AND created_at >= ?"
        exp_params.append(date_from)
    if date_to:
        sql += " AND created_at <= ?"
        exp_params.append(date_to)
    sql += " ORDER BY created_at DESC LIMIT ?"
    exp_params.append(_MAX_EXPERIENCE)
    try:
        rows = exp_conn.execute(sql, exp_params).fetchall()
    except Exception as ex:
        log.debug("experience layer query failed: %s", ex)
        if owns:
            try:
                exp_conn.close()
            except Exception:
                pass
        return

    mem_ents: dict[str, set[str]] = {}
    for e in edges:
        if e.get("type") == "mentions":
            mid, eid = e.get("source"), e.get("target")
            if mid in mem_ids and isinstance(eid, str) and eid.startswith("ent:"):
                mem_ents.setdefault(str(mid), set()).add(eid.removeprefix("ent:").casefold())

    try:
        practiced_seen: set[str] = set()
        for row in rows:
            xid = f"exp:{row['id']}"
            text = (row["record_text"] or row["goal"] or row["answer_excerpt"] or "").strip()
            goal = (row["goal"] or text)[:72]
            label = goal + ("…" if len(goal) >= 72 else "")
            ents = _entities_from_raw(row["entities"]) if has_ent else []
            nodes.append({
                "id": xid,
                "type": "experience",
                "label": label,
                "text": text,
                "status": "active",
                "kind": "experience",
                "source": "experiences",
                "pinned": False,
                "access_count": 0,
                "created_at": row["created_at"] if "created_at" in row.keys() else None,
                "entities": ents,
                "supersedes_id": None,
                "valence_tag": "neutral",
                "outcome": row["outcome"] if "outcome" in row.keys() else "",
                "scores": {
                    "retain": 0.5, "salience": 0.35, "spacing": 0.0,
                    "connectivity": 0.0, "valence": 0.25, "access": 0.0,
                },
                "size": 0.42,
            })
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
                        "valence_tag": "neutral",
                        "scores": {
                            "retain": 0.3, "importance": 0.3, "salience": 0.0,
                            "spacing": 0.0, "connectivity": 0.3, "valence": 0.25, "access": 0.0,
                        },
                        "size": 0.35,
                    })
                edges.append({
                    "id": f"about:{xid}->{eid}",
                    "source": xid,
                    "target": eid,
                    "type": "about",
                    "weight": 1.0,
                })
                ent_cf = ent.casefold()
                for mid, mes in mem_ents.items():
                    if ent_cf in mes:
                        pid = f"practiced:{mid}->{xid}"
                        if pid not in practiced_seen:
                            practiced_seen.add(pid)
                            edges.append({
                                "id": pid,
                                "source": mid,
                                "target": xid,
                                "type": "practiced_in",
                                "weight": 1.0,
                            })
    finally:
        if owns and exp_conn is not None:
            try:
                exp_conn.close()
            except Exception:
                pass
