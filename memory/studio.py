"""
memory/studio.py

Phase 10 — Memory Graph Studio data layer (read-only).

Builds galaxy-friendly graph payloads from personal memory:
  - memory nodes sized by retain tendency
  - rim score arcs (salience, spacing, connectivity proxy, valence, access)
  - entity nodes sized by I_e
  - co-mention edges from entity_relations
  - supersession chains

No writes. Safe for WebUI / CLI inspection.
"""
from __future__ import annotations

import math
import os
import re
from typing import Any

from system.log import get_logger
from system.userspace import current_user_id

log = get_logger(__name__)

STUDIO_MAX_MEMORIES = max(50, int(os.getenv("MEMORY_STUDIO_MAX_MEMORIES", "400")))
STUDIO_MAX_ENTITIES = max(20, int(os.getenv("MEMORY_STUDIO_MAX_ENTITIES", "120")))
STUDIO_MAX_EDGES = max(20, int(os.getenv("MEMORY_STUDIO_MAX_EDGES", "200")))

_DAILY_RE = re.compile(r"^\[\d{4}-\d{2}-\d{2}\]\s")
_MONTHLY_RE = re.compile(r"^\[\d{4}-\d{2}\]\s")
_SPACING_SAT = max(1, int(os.getenv("MONTHLY_CONSOLIDATION_SPACING_SATURATION", "5")))


def _entities_list(raw: Any) -> list[str]:
    try:
        from memory.memorize import entities_from_json
        return list(entities_from_json(raw) or [])
    except Exception:
        if isinstance(raw, list):
            return [str(x) for x in raw if x]
        if isinstance(raw, str) and raw.strip().startswith("["):
            import json
            try:
                data = json.loads(raw)
                if isinstance(data, list):
                    return [str(x) for x in data if x]
            except Exception:
                pass
        return []


def _valence_score(tag: Any) -> float:
    t = (str(tag or "neutral")).strip().lower()
    if t == "neg":
        return 0.85
    if t == "pos":
        return 0.65
    return 0.25


def _salience_score(row: dict) -> float:
    hit = row.get("salience_hit")
    if hit is not None and str(hit) != "":
        return 1.0 if int(hit) else 0.3
    try:
        from memory.memorize import SALIENCE_POLICY_RE
        text = row.get("memory") or row.get("text") or ""
        return 1.0 if SALIENCE_POLICY_RE.search(text) else 0.3
    except Exception:
        return 0.3


def _spacing_score(row: dict) -> float:
    day_count = int(row.get("access_day_count") or 0)
    if day_count <= 0:
        day_count = 1 if int(row.get("access_count") or 0) > 0 else 0
    return min(1.0, day_count / float(_SPACING_SAT))


def _access_score(row: dict) -> float:
    ac = int(row.get("access_count") or 0)
    return min(1.0, math.log1p(ac) / math.log1p(50))


def _retain_proxy(row: dict, entity_importance: dict[str, float]) -> float:
    """Cheap retain tendency for node size (not full monthly R with anchors)."""
    text = (row.get("memory") or "").strip()
    pinned = 1.0 if int(row.get("pinned") or 0) else 0.0
    if _MONTHLY_RE.match(text):
        pinned = max(pinned, 0.95)
    status = (row.get("status") or "active")
    if str(status).strip().lower() == "superseded":
        return 0.15

    sal = _salience_score(row)
    sp = _spacing_score(row)
    val = _valence_score(row.get("valence_tag"))
    acc = _access_score(row)

    ents = _entities_list(row.get("entities"))
    ie = 0.0
    if ents and entity_importance:
        ie = max((entity_importance.get(e.casefold(), 0.0) for e in ents), default=0.0)

    # Weighted proxy — size encodes keep-likelihood
    r = (
        0.28 * sal
        + 0.18 * sp
        + 0.12 * val
        + 0.12 * acc
        + 0.20 * ie
        + 0.25 * pinned
    )
    return float(max(0.05, min(1.0, r)))


def _layout_polar(n: int, radius: float, seed: int = 0) -> list[tuple[float, float]]:
    """Deterministic spiral galaxy layout in [-1,1] plane."""
    pts: list[tuple[float, float]] = []
    if n <= 0:
        return pts
    golden = math.pi * (3 - math.sqrt(5))
    for i in range(n):
        t = i + seed * 0.17
        r = radius * math.sqrt((i + 1) / n)
        theta = t * golden
        pts.append((r * math.cos(theta), r * math.sin(theta)))
    return pts


def build_galaxy_graph(
    memorize=None,
    *,
    user_id: str | None = None,
    limit: int | None = None,
    include_superseded: bool = True,
) -> dict:
    """Return galaxy graph JSON: nodes, edges, legend."""
    uid = user_id or current_user_id()
    limit = limit or STUDIO_MAX_MEMORIES

    if memorize is None:
        from memory.memorize import AikoMemorize
        memorize = AikoMemorize(silent=True)

    try:
        rows = list(memorize.get_all(user_id=uid) or [])
    except Exception as exc:
        log.warning("studio get_all failed: %s", exc)
        rows = []

    entity_importance: dict[str, float] = {}
    try:
        from memory.entity_importance import compute_entity_importance_map
        entity_importance = compute_entity_importance_map(memorize, uid) or {}
    except Exception as exc:
        log.debug("studio I_e map skipped: %s", exc)

    mem_nodes: list[dict] = []
    for m in rows:
        text = (m.get("memory") or "").strip()
        if not text:
            continue
        status = (m.get("status") or "active")
        if not include_superseded and str(status).strip().lower() == "superseded":
            continue
        sal = _salience_score(m)
        sp = _spacing_score(m)
        val = _valence_score(m.get("valence_tag"))
        acc = _access_score(m)
        ents = _entities_list(m.get("entities"))
        ie = 0.0
        if ents and entity_importance:
            ie = max((entity_importance.get(e.casefold(), 0.0) for e in ents), default=0.0)
        retain = _retain_proxy(m, entity_importance)
        kind = "monthly" if _MONTHLY_RE.match(text) else ("daily" if _DAILY_RE.match(text) else "fact")
        mem_nodes.append({
            "id": str(m.get("id") or ""),
            "type": "memory",
            "label": text[:80],
            "text": text,
            "kind": kind,
            "status": status,
            "pinned": int(m.get("pinned") or 0),
            "entities": ents,
            "supersedes_id": m.get("supersedes_id"),
            "scores": {
                "retain": round(retain, 4),
                "salience": round(sal, 4),
                "spacing": round(sp, 4),
                "connectivity": round(ie, 4),
                "valence": round(val, 4),
                "access": round(acc, 4),
            },
            "size": round(0.35 + 0.9 * retain, 4),
            "valence_tag": (m.get("valence_tag") or "neutral"),
            "created_at": m.get("created_at"),
            "last_accessed_at": m.get("last_accessed_at"),
            "access_count": int(m.get("access_count") or 0),
            "access_day_count": int(m.get("access_day_count") or 0),
        })

    mem_nodes.sort(key=lambda n: n["scores"]["retain"], reverse=True)
    mem_nodes = mem_nodes[:limit]

    # Entity nodes from importance + presence on kept memories
    seen_ents: dict[str, dict] = {}
    for n in mem_nodes:
        for e in n["entities"]:
            k = e.casefold()
            if not k:
                continue
            if k not in seen_ents:
                ie = float(entity_importance.get(k, 0.0))
                seen_ents[k] = {
                    "id": f"ent:{k}",
                    "type": "entity",
                    "label": e,
                    "scores": {
                        "importance": round(ie, 4),
                        "retain": round(ie, 4),
                    },
                    "size": round(0.25 + 0.85 * ie, 4),
                    "mention_count": 0,
                }
            seen_ents[k]["mention_count"] += 1

    ent_list = sorted(seen_ents.values(), key=lambda x: x["scores"]["importance"], reverse=True)
    ent_list = ent_list[:STUDIO_MAX_ENTITIES]

    # Co-mention edges
    edges: list[dict] = []
    try:
        lock = getattr(getattr(memorize, "_mem", None), "_db_lock", None)
        conn = getattr(memorize, "_conn", None) or getattr(getattr(memorize, "_mem", None), "_conn", None)
        if conn is not None:
            def _fetch():
                return conn.execute(
                    "SELECT entity_a, entity_b, weight FROM entity_relations WHERE user_id = ?",
                    (uid,),
                ).fetchall()
            if lock is not None:
                with lock:
                    erows = _fetch()
            else:
                erows = _fetch()
            ent_ids = {e["label"].casefold() for e in ent_list}
            for r in erows:
                a = str(r["entity_a"] or "").casefold()
                b = str(r["entity_b"] or "").casefold()
                if a not in ent_ids or b not in ent_ids or a == b:
                    continue
                w = float(r["weight"] or 0.0)
                edges.append({
                    "source": f"ent:{a}",
                    "target": f"ent:{b}",
                    "weight": w,
                    "kind": "co_mention",
                })
            edges.sort(key=lambda e: e["weight"], reverse=True)
            edges = edges[:STUDIO_MAX_EDGES]
    except Exception as exc:
        log.debug("studio entity_relations skipped: %s", exc)

    # Memory–entity soft edges (for galaxy filaments)
    mem_ent_edges: list[dict] = []
    ent_id_set = {e["id"] for e in ent_list}
    for n in mem_nodes[: min(80, len(mem_nodes))]:
        for e in n["entities"]:
            eid = f"ent:{e.casefold()}"
            if eid in ent_id_set:
                mem_ent_edges.append({
                    "source": n["id"],
                    "target": eid,
                    "weight": 0.3,
                    "kind": "mentions",
                })
    mem_ent_edges = mem_ent_edges[:STUDIO_MAX_EDGES]

    # Layout
    mpos = _layout_polar(len(mem_nodes), 0.85, seed=1)
    for i, n in enumerate(mem_nodes):
        n["x"], n["y"] = (round(mpos[i][0], 5), round(mpos[i][1], 5)) if i < len(mpos) else (0.0, 0.0)
    epos = _layout_polar(len(ent_list), 0.45, seed=7)
    for i, n in enumerate(ent_list):
        n["x"], n["y"] = (round(epos[i][0], 5), round(epos[i][1], 5)) if i < len(epos) else (0.0, 0.0)

    return {
        "theme": "galaxy",
        "user_id": uid,
        "nodes": mem_nodes + ent_list,
        "edges": edges + mem_ent_edges,
        "legend": {
            "size": "retain tendency (memories) / I_e (entities)",
            "rim_arcs": ["salience", "spacing", "connectivity", "valence", "access"],
            "colors": {
                "neg": "#ff6b6b",
                "neutral": "#c0c8d8",
                "pos": "#7ad7f0",
                "entity": "#c9a0ff",
                "monthly": "#ffd27a",
            },
        },
        "stats": {
            "memories": len(mem_nodes),
            "entities": len(ent_list),
            "edges": len(edges) + len(mem_ent_edges),
        },
    }


def supersession_chain(memorize, memory_id: str, *, user_id: str | None = None) -> list[dict]:
    """Oldest → newest chain via supersedes_id."""
    uid = user_id or current_user_id()
    try:
        from memory.entity_importance import walk_supersession_chain
        ids = walk_supersession_chain(memorize, memory_id, user_id=uid)
    except Exception:
        ids = [memory_id]
    out: list[dict] = []
    for mid in ids:
        try:
            rows = [m for m in (memorize.get_all(user_id=uid) or []) if str(m.get("id")) == str(mid)]
            if rows:
                m = rows[0]
                out.append({
                    "id": mid,
                    "text": (m.get("memory") or "")[:200],
                    "status": m.get("status"),
                    "supersedes_id": m.get("supersedes_id"),
                })
            else:
                out.append({"id": mid})
        except Exception:
            out.append({"id": mid})
    return out


def list_memories(
    memorize=None,
    *,
    user_id: str | None = None,
    limit: int = 100,
    entity: str | None = None,
    valence: str | None = None,
    pinned: int | None = None,
    status: str | None = None,
    q: str | None = None,
) -> list[dict]:
    """Filtered memory list for Studio table view."""
    g = build_galaxy_graph(memorize, user_id=user_id, limit=max(limit, STUDIO_MAX_MEMORIES))
    rows = [n for n in g["nodes"] if n.get("type") == "memory"]
    if entity:
        ek = entity.casefold()
        rows = [r for r in rows if any(e.casefold() == ek for e in r.get("entities") or [])]
    if valence:
        rows = [r for r in rows if (r.get("valence_tag") or "").lower() == valence.lower()]
    if pinned is not None:
        rows = [r for r in rows if int(r.get("pinned") or 0) == int(pinned)]
    if status:
        rows = [r for r in rows if str(r.get("status") or "").lower() == status.lower()]
    if q:
        ql = q.casefold()
        rows = [r for r in rows if ql in (r.get("text") or "").casefold()]
    return rows[:limit]
