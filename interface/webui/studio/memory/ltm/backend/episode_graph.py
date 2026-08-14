"""EMC-5: episode nodes for LTM Graph Studio (emc_storage → type=episode)."""
from __future__ import annotations

import json
import os
import sqlite3
from typing import Any

from system.log import get_logger

log = get_logger(__name__)


def _env_int(name: str, default: int, *, floor: int = 0) -> int:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return max(floor, default)
    try:
        return max(floor, int(str(raw).strip()))
    except (TypeError, ValueError):
        return max(floor, default)


MAX_EPISODES = _env_int("MEMORY_STUDIO_MAX_EPISODES", 80, floor=0)
INCLUDE_EPISODES = os.getenv("MEMORY_STUDIO_INCLUDE_EPISODES", "1").lower() in {
    "1", "true", "yes", "on",
}


def _add_episode_layer(
    conn: sqlite3.Connection,
    uid: str,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    entity_ids: set[str],
    date_from: str | None = None,
    date_to: str | None = None,
) -> None:
    """Add emc_storage rows as episode nodes + about edges to entities.

    Missing human-EM fields stay absent (never invent). Same DB as memories (Option A).
    """
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(emc_storage)").fetchall()}
    except Exception:
        cols = set()
    if not cols or "id" not in cols or "trace" not in cols:
        return

    select = ["id", "timestamp", "date", "trace"]
    if "created_at" in cols:
        select.append("created_at")
    optional = [
        "valence_tag",
        "arousal_score",
        "salience_score",
        "entities",
        "source",
        "session_id",
        "recall_count",
        "last_recalled_at",
        "distilled_at",
        "superseded_by",
    ]
    for c in optional:
        if c in cols:
            select.append(c)

    sql = f"SELECT {', '.join(select)} FROM emc_storage WHERE user_id = ?"
    params: list[Any] = [uid]
    if "superseded_by" in cols:
        sql += " AND (superseded_by IS NULL)"
    if date_from:
        if "date" in cols:
            sql += " AND date >= ?"
            params.append(date_from[:10])
        elif "timestamp" in cols:
            sql += " AND timestamp >= ?"
            params.append(date_from)
    if date_to:
        if "date" in cols:
            sql += " AND date <= ?"
            params.append(date_to[:10])
        elif "timestamp" in cols:
            sql += " AND timestamp <= ?"
            params.append(date_to)
    order_bits = []
    if "salience_score" in cols:
        order_bits.append("COALESCE(salience_score, 0) DESC")
    if "timestamp" in cols:
        order_bits.append("timestamp DESC")
    elif "created_at" in cols:
        order_bits.append("created_at DESC")
    if order_bits:
        sql += " ORDER BY " + ", ".join(order_bits)
    sql += " LIMIT ?"
    params.append(MAX_EPISODES)

    try:
        rows = conn.execute(sql, params).fetchall()
    except Exception as ex:
        log.debug("episode layer query failed: %s", ex)
        return

    for row in rows:
        r = dict(row)
        eid = f"ep:{r['id']}"
        trace = (r.get("trace") or "").strip()
        if not trace:
            continue
        label = trace.replace("\n", " ").strip()
        if len(label) > 72:
            label = label[:69] + "…"

        ents_raw = r.get("entities")
        ents: list[str] = []
        if ents_raw:
            try:
                parsed = json.loads(ents_raw) if isinstance(ents_raw, str) else ents_raw
                if isinstance(parsed, list):
                    ents = [str(x).strip() for x in parsed if str(x).strip()]
            except Exception:
                ents = []

        sal = r.get("salience_score")
        try:
            sal_f = float(sal) if sal is not None else 0.35
        except (TypeError, ValueError):
            sal_f = 0.35
        sal_f = max(0.0, min(1.0, sal_f if sal_f <= 1.0 else sal_f / 10.0))

        recall = int(r.get("recall_count") or 0)
        distilled = bool(r.get("distilled_at"))
        valence = (r.get("valence_tag") or "neutral") or "neutral"

        nodes.append({
            "id": eid,
            "type": "episode",
            "label": label,
            "text": trace,
            "status": "distilled" if distilled else "active",
            "kind": "episode",
            "source": r.get("source") or "chat",
            "pinned": False,
            "access_count": recall,
            "created_at": r.get("timestamp") or r.get("created_at"),
            "date": r.get("date"),
            "entities": ents,
            "supersedes_id": None,
            "valence_tag": valence,
            "arousal_score": r.get("arousal_score"),
            "salience_score": r.get("salience_score"),
            "session_id": r.get("session_id"),
            "distilled_at": r.get("distilled_at"),
            "last_recalled_at": r.get("last_recalled_at"),
            "scores": {
                "retain": 0.4 + 0.35 * sal_f,
                "salience": sal_f,
                "spacing": 0.0,
                "connectivity": min(1.0, 0.15 * len(ents)),
                "valence": 0.25,
                "access": min(1.0, recall / 10.0),
            },
            "size": 0.35 + 0.4 * sal_f,
        })

        for ent in ents:
            ent_id = f"ent:{ent.casefold()}"
            if ent_id not in entity_ids:
                entity_ids.add(ent_id)
                nodes.append({
                    "id": ent_id,
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
                        "retain": 0.3,
                        "importance": 0.3,
                        "salience": 0.0,
                        "spacing": 0.0,
                        "connectivity": 0.3,
                        "valence": 0.25,
                        "access": 0.0,
                    },
                    "size": 0.35,
                })
            edges.append({
                "id": f"about:{eid}->{ent_id}",
                "source": eid,
                "target": ent_id,
                "type": "about",
                "weight": 1.0,
            })
