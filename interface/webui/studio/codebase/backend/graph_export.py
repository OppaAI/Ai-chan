"""
interface/webui/studio/codebase/backend/graph_export.py

Figure-shaped codebase graph — maps repo modules to body parts.
Sharp silhouette, not blob. Each node is a file chunk, grouped by body part.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from system.userspace import current_user_id, user_state_path
import sqlite3

# Body part mapping — file path regex → body part
_BODY_MAP = [
    # Head / Brain
    (r"^cognition/(memory|knowledge|subliminal|attention|think|reason|consolidate)", "head", "brain", "#ff6b6b"),
    (r"^cognition/", "head", "brain", "#ff8e8e"),
    # Eyes (vision)
    (r"^sensory/(vision|video)|^interface/webui|^training/persona", "head_eyes", "eyes", "#4ecdc4"),
    # Ears (listen)
    (r"^sensory/(listen|audio)|^sensory/speak.*listen|sherpa", "head_ears", "ears", "#ffe66d"),
    # Mouth (speak)
    (r"^sensory/speak|interface/adapter|agentic/toolkit/social", "head_mouth", "mouth", "#ff9f43"),
    # Heart / Chest (core state)
    (r"^system/(userspace|config|secure|bioclock|log)|^main\.py", "chest", "heart", "#a29bfe"),
    # Arms / Hands (tools)
    (r"^agentic/toolkit|^agentic/workflows|^agentic/graph_engine|^agentic/registry", "arms", "hands", "#6c5ce7"),
    # Legs / Mobility (orchestration)
    (r"^system/(orchestrate|schedule|wakeup|prepare|turngate)", "legs", "legs", "#00b894"),
    # Spine / Backbone
    (r"^agentic/mcp|^interface/mcp|^interface/webui|^backend", "spine", "spine", "#636e72"),
    # Tail / Misc
    (r".*", "tail", "other", "#b2bec3"),
]

# Figure silhouette anchors (normalized 0..1 within 400x600 viewBox)
_BODY_ANCHORS = {
    "head": (0.5, 0.15),
    "head_eyes": (0.5, 0.12),
    "head_ears": (0.5, 0.15),
    "head_mouth": (0.5, 0.20),
    "chest": (0.5, 0.35),
    "arms": (0.5, 0.38),
    "spine": (0.5, 0.50),
    "legs": (0.5, 0.75),
    "tail": (0.5, 0.90),
}

def _body_for_path(rel: str) -> tuple[str, str, str]:
    for pat, anchor, label, color in _BODY_MAP:
        if re.search(pat, rel):
            return anchor, label, color
    return "tail", "other", "#b2bec3"

def export_codebase_graph(user_id: str | None = None, limit: int = 400) -> dict:
    uid = user_id or current_user_id()
    p = user_state_path("knowledge/codebase.db", user_id=uid)
    if not p.exists():
        return {"nodes": [], "edges": [], "meta": {"user_id": uid, "exists": False, "path": str(p)}}
    try:
        conn = sqlite3.connect(str(p))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
    except Exception as e:
        return {"nodes": [], "edges": [], "meta": {"error": str(e)}}
    try:
        # Fetch docs + sample chunks
        docs = conn.execute("SELECT id, path, title FROM codebase_docs WHERE user_id=? LIMIT ?", (uid, limit)).fetchall()
        nodes = []
        body_counts: dict[str, int] = {}
        for row in docs:
            rel = row["path"]
            anchor, label, color = _body_for_path(rel)
            body_counts[label] = body_counts.get(label, 0) + 1
            ax, ay = _BODY_ANCHORS.get(anchor, (0.5, 0.5))
            # jitter within body region
            import random
            random.seed(hash(rel) % (2**32))
            jx = (random.random() - 0.5) * 0.18
            jy = (random.random() - 0.5) * 0.08
            nodes.append({
                "id": row["id"],
                "path": rel,
                "title": row["title"],
                "body_part": label,
                "anchor": anchor,
                "color": color,
                "x": max(0.05, min(0.95, ax + jx)),
                "y": max(0.05, min(0.95, ay + jy)),
            })
        # Edges: co-located body parts + import-like co-occurrence (simple: same anchor = edge)
        edges = []
        # Group by anchor
        from collections import defaultdict
        groups: dict[str, list[str]] = defaultdict(list)
        for n in nodes:
            groups[n["anchor"]].append(n["id"])
        for anchor, ids in groups.items():
            if len(ids) < 2:
                continue
            # star within group
            center = ids[0]
            for other in ids[1:5]:
                edges.append({"source": center, "target": other, "kind": "body_co_located"})
        return {
            "nodes": nodes,
            "edges": edges,
            "meta": {
                "user_id": uid,
                "path": str(p),
                "exists": True,
                "count": len(nodes),
                "body_counts": body_counts,
                "limit": limit,
            },
            "anchors": {k: {"x": v[0], "y": v[1]} for k, v in _BODY_ANCHORS.items()},
            "body_map": [{"label": lbl, "anchor": anc, "color": col} for _, anc, lbl, col in _BODY_MAP[:9]],
        }
    finally:
        try: conn.close()
        except: pass
