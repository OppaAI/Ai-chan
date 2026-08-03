"""Phase 3: entity importance I_e and supersession chain helpers.

I_e = (1-α)·centrality + α·recency
No LLM. Used by monthly consolidation and mild recall boost.
"""
from __future__ import annotations

import math
import os
import re
from datetime import datetime, timezone

from system.log import get_logger

log = get_logger(__name__)

ENTITY_IMPORTANCE_ALPHA = float(os.getenv("ENTITY_IMPORTANCE_ALPHA", "0.4"))
ENTITY_IMPORTANCE_BETA = float(os.getenv("ENTITY_IMPORTANCE_BETA", "0.05"))
MEMORY_RANK_ENTITY_IMPORTANCE_WEIGHT = float(os.getenv("MEMORY_RANK_ENTITY_IMPORTANCE_WEIGHT", "0.008"))
MEMORY_SUPERSESSION_CHAIN_EXPAND = os.getenv("MEMORY_SUPERSESSION_CHAIN_EXPAND", "1").lower() in {"1", "true", "yes", "on"}
MEMORY_SUPERSESSION_CHAIN_KINDS = {
    k.strip().lower()
    for k in os.getenv("MEMORY_SUPERSESSION_CHAIN_KINDS", "identity,preference,plan").split(",")
    if k.strip()
}
_REFLECTIVE_RE = re.compile(
    r"\b(used to|changed|before|previously|history|what changed|"
    r"remember when|used to be|no longer|switched from)\b",
    re.IGNORECASE,
)


def entities_from_json_safe(raw) -> list[str]:
    try:
        from memory.memorize import entities_from_json
        return entities_from_json(raw)
    except Exception:
        return []


def compute_entity_importance_map(memorize_or_backend, user_id: str) -> dict[str, float]:
    """I_e = (1-α)·centrality + α·recency per entity (casefolded)."""
    try:
        conn = getattr(memorize_or_backend, "_conn", None)
        if conn is None:
            mem = getattr(memorize_or_backend, "_mem", None)
            conn = getattr(mem, "_conn", None) if mem is not None else None
        if conn is None:
            return {}
        lock = getattr(getattr(memorize_or_backend, "_mem", memorize_or_backend), "_db_lock", None)

        def _read():
            try:
                rows = conn.execute(
                    "SELECT entity_a, entity_b, weight FROM entity_relations WHERE user_id = ?",
                    (user_id,),
                ).fetchall()
            except Exception:
                return {}, {}
            degree: dict[str, float] = {}
            for row in rows:
                a = str(row["entity_a"] or "").casefold()
                b = str(row["entity_b"] or "").casefold()
                w = float(row["weight"] or 0.0)
                if a:
                    degree[a] = degree.get(a, 0.0) + w
                if b:
                    degree[b] = degree.get(b, 0.0) + w
            last_touch: dict[str, str] = {}
            try:
                mem_rows = conn.execute(
                    """
                    SELECT entities, last_accessed_at, created_at FROM memories
                    WHERE user_id = ? AND (status = 'active' OR status IS NULL)
                    """,
                    (user_id,),
                ).fetchall()
            except Exception:
                mem_rows = []
            for mr in mem_rows:
                ents = entities_from_json_safe(mr["entities"] if "entities" in mr.keys() else "[]")
                ts = mr["last_accessed_at"] or mr["created_at"] or ""
                for e in ents:
                    key = e.casefold()
                    prev = last_touch.get(key, "")
                    if ts and (not prev or str(ts) > prev):
                        last_touch[key] = str(ts)
            return degree, last_touch

        if lock is not None:
            with lock:
                degree, last_touch = _read()
        else:
            degree, last_touch = _read()

        if not degree and not last_touch:
            return {}
        max_deg = max(degree.values(), default=1.0) or 1.0
        alpha = max(0.0, min(1.0, ENTITY_IMPORTANCE_ALPHA))
        beta = max(0.0, ENTITY_IMPORTANCE_BETA)
        now = datetime.now(timezone.utc)
        out: dict[str, float] = {}
        for e in set(degree) | set(last_touch):
            c = (degree.get(e, 0.0) / max_deg) if max_deg else 0.0
            r = 0.0
            ts = last_touch.get(e) or ""
            if ts and ts != "never":
                try:
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    days = max(0.0, (now - dt).total_seconds() / 86400.0)
                    r = float(math.exp(-beta * days))
                except Exception:
                    r = 0.0
            out[e] = (1.0 - alpha) * c + alpha * r
        return out
    except Exception as exc:
        log.debug("compute_entity_importance_map failed: %s", exc)
        return {}


def memory_max_entity_importance(row, importance_map: dict[str, float]) -> float:
    if not importance_map:
        return 0.0
    try:
        raw = row["entities"] if hasattr(row, "keys") and "entities" in row.keys() else row.get("entities")
        ents = entities_from_json_safe(raw)
    except Exception:
        ents = []
    if not ents:
        return 0.0
    return max((importance_map.get(e.casefold(), 0.0) for e in ents), default=0.0)


def should_expand_supersession_chain(query: str, row) -> bool:
    if not MEMORY_SUPERSESSION_CHAIN_EXPAND:
        return False
    if _REFLECTIVE_RE.search(query or ""):
        return True
    try:
        kind = (row["kind"] if hasattr(row, "keys") else row.get("kind")) or ""
    except Exception:
        kind = ""
    return str(kind).lower() in MEMORY_SUPERSESSION_CHAIN_KINDS


def walk_supersession_chain(conn, mem_id: str, user_id: str, max_depth: int = 12) -> list[dict]:
    try:
        row = conn.execute(
            "SELECT * FROM memories WHERE id = ? AND user_id = ?",
            (mem_id, user_id),
        ).fetchone()
        if row is None:
            return []
        chain_ids: list[str] = [mem_id]
        seen = {mem_id}
        cur = row
        for _ in range(max_depth):
            sid = cur["supersedes_id"] if "supersedes_id" in cur.keys() else None
            if not sid or sid in seen:
                break
            prev = conn.execute(
                "SELECT * FROM memories WHERE id = ? AND user_id = ?",
                (sid, user_id),
            ).fetchone()
            if prev is None:
                break
            chain_ids.append(sid)
            seen.add(sid)
            cur = prev
        chain_ids.reverse()
        tip = chain_ids[-1]
        for _ in range(max_depth):
            nxt = conn.execute(
                """
                SELECT * FROM memories
                WHERE user_id = ? AND supersedes_id = ?
                ORDER BY created_at ASC LIMIT 1
                """,
                (user_id, tip),
            ).fetchone()
            if nxt is None:
                break
            nid = str(nxt["id"])
            if nid in seen:
                break
            chain_ids.append(nid)
            seen.add(nid)
            tip = nid
        out: list[dict] = []
        for mid in chain_ids:
            r = conn.execute("SELECT * FROM memories WHERE id = ?", (mid,)).fetchone()
            if r is not None:
                out.append(dict(r))
        return out
    except Exception as exc:
        log.debug("supersession chain walk failed for %s: %s", mem_id, exc)
        return []
