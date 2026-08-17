"""
EMC-3: episodic recall helpers attached onto EpisodicStore at boot.

Kept separate so EMC-3 can land without rewriting the large episode.py body.
apply_emc2_hooks() calls attach_recall_to_store().
"""
from __future__ import annotations

import json
import time
from collections import OrderedDict
from typing import Any

import sqlite_vec

from system.log import get_logger
from cognition.memory.search import _sanitize_fts_query
from cognition.memory.env import env_bool, env_int, env_float
from cognition.memory.vecstore import KNN_MATCH_K_MIN, KNN_MATCH_OVERSCAN

log = get_logger(__name__)

# Mirrors episode.py / memory.yaml (read env at call time via these constants)
EMC_RECALL_ENABLED = env_bool("EMC_RECALL_ENABLED", "1")
EMC_RECALL_LIMIT = max(0, env_int("EMC_RECALL_LIMIT", 2))
EMC_KNN_LIMIT = max(1, env_int("EMC_KNN_LIMIT", 12))
EMC_FTS_LIMIT = max(1, env_int("EMC_FTS_LIMIT", 12))
EMC_RRF_K = max(1, env_int("EMC_RRF_K", 60))
EMC_CONTEXT_CHARS = max(100, env_int("EMC_CONTEXT_CHARS", 600))
EMC_CONTEXT_EPISODE_CHARS = max(40, env_int("EMC_CONTEXT_EPISODE_CHARS", 280))
EMC_JOINT_BUDGET = env_bool("EMC_JOINT_BUDGET", "1")
# Episodic recall result cache. Without this, format_for_context runs a full
# EMC KNN+FTS+R RF pass (plus a per-id touch loop) on every non-greeting turn,
# even when the semantic memory search already served from its own cache. A
# short TTL makes repeat turns near-free; hits still touch recall_count so
# recency/importance stay fresh.
EMC_RECALL_CACHE_SIZE = max(1, env_int("EMC_RECALL_CACHE_SIZE", 128))
EMC_RECALL_CACHE_TTL = max(1.0, env_float("EMC_RECALL_CACHE_TTL", 20.0))


def _utc_now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def search(
    self,
    query: str,
    *,
    limit: int | None = None,
    query_vector: list[float] | None = None,
    user_id: str | None = None,
) -> list[dict]:
    """KNN + FTS5 → RRF over emc_storage. Returns payload dicts."""
    from cognition.memory.episode import EMC_ENABLED

    if not EMC_ENABLED or not EMC_RECALL_ENABLED:
        return []
    top_k = EMC_RECALL_LIMIT if limit is None else max(0, int(limit))
    if top_k <= 0:
        return []
    uid = user_id or self._user_id
    q = (query or "").strip()
    if not q:
        return []

    # ── recall result cache ───────────────────────────────────────────────────
    # Short-TTL per (user, query, limit) cache so every non-greeting turn
    # doesn't re-run KNN+FTS+RRF against emc_vec/emc_fts.
    cache_key = (uid, " ".join(q.casefold().split()), int(top_k))
    now = time.monotonic()
    with self._recall_cache_lock:
        cached = self._recall_cache.get(cache_key)
        if cached and now - cached[0] <= EMC_RECALL_CACHE_TTL:
            self._recall_cache.move_to_end(cache_key)
            hits = [dict(r) for r in cached[1]]
            try:
                _touch_episodes(self, [r["id"] for r in hits])
            except Exception:
                pass
            return hits
        if cached:
            self._recall_cache.pop(cache_key, None)

    vector = None
    try:
        if query_vector is not None:
            vector = list(query_vector)
        else:
            emb = self._embedder
            if hasattr(emb, "embed_query"):
                vector = list(emb.embed_query(q))
            else:
                vector = list(emb.embed([q]))[0]
    except Exception as e:
        log.debug("EMC search embed failed: %s", e)

    fts_query = _sanitize_fts_query(q)

    with self._lock:
        rank_knn: dict[int, int] = {}
        rank_fts: dict[int, int] = {}
        if vector is not None:
            try:
                vec_blob = sqlite_vec.serialize_float32(vector)
                k = max(int(EMC_KNN_LIMIT) * KNN_MATCH_OVERSCAN, KNN_MATCH_K_MIN)
                knn_rows = self._conn.execute(
                    """
                    SELECT v.rowid AS id, v.distance AS dist
                    FROM emc_vec v
                    JOIN emc_storage s ON s.id = v.rowid
                    WHERE v.embedding MATCH ?
                      AND v.k = ?
                      AND s.user_id = ?
                      AND (s.superseded_by IS NULL)
                    ORDER BY v.distance ASC
                    LIMIT ?
                    """,
                    (vec_blob, k, uid, EMC_KNN_LIMIT),
                ).fetchall()
                rank_knn = {int(r[0]): i + 1 for i, r in enumerate(knn_rows)}
            except Exception as e:
                log.debug("EMC KNN failed: %s", e)

        if fts_query:
            try:
                fts_rows = self._conn.execute(
                    """
                    SELECT s.id
                    FROM emc_fts
                    JOIN emc_storage s ON s.id = emc_fts.rowid
                    WHERE emc_fts MATCH ?
                      AND s.user_id = ?
                      AND (s.superseded_by IS NULL)
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (fts_query, uid, EMC_FTS_LIMIT),
                ).fetchall()
                rank_fts = {int(r[0]): i + 1 for i, r in enumerate(fts_rows)}
            except Exception as e:
                log.debug("EMC FTS failed: %s", e)

        candidate_ids = set(rank_knn) | set(rank_fts)
        if not candidate_ids:
            return []

        scores: dict[int, float] = {}
        for eid in candidate_ids:
            s = 0.0
            if eid in rank_knn:
                s += 1.0 / (EMC_RRF_K + rank_knn[eid])
            if eid in rank_fts:
                s += 1.0 / (EMC_RRF_K + rank_fts[eid])
            scores[eid] = s

        ordered = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)[:top_k]
        if not ordered:
            return []

        placeholders = ",".join("?" * len(ordered))
        rows = self._conn.execute(
            f"""
            SELECT id, timestamp, date, trace, valence_tag, arousal_score,
                   salience_score, entities, source, session_id, recall_count, cognitive_json
            FROM emc_storage
            WHERE id IN ({placeholders})
            """,
            ordered,
        ).fetchall()
        by_id = {int(r[0]): r for r in rows}

        results: list[dict] = []
        for eid in ordered:
            row = by_id.get(eid)
            if not row:
                continue
            entities = None
            if row[7]:
                try:
                    entities = json.loads(row[7])
                except Exception:
                    entities = None
            results.append({
                "id": int(row[0]),
                "timestamp": row[1],
                "date": row[2],
                "trace": row[3],
                "memory": row[3],
                "valence_tag": row[4],
                "arousal_score": row[5],
                "salience_score": row[6],
                "entities": entities,
                "source": row[8],
                "session_id": row[9],
                "recall_count": int(row[10] or 0),
                "cognitive_state": json.loads(row[11]) if row[11] else None,
                "_recall_score": scores.get(eid, 0.0),
                "_emc": True,
            })

        _touch_episodes(self, [r["id"] for r in results])
        with self._recall_cache_lock:
            self._recall_cache[cache_key] = (now, [dict(r) for r in results])
            while len(self._recall_cache) > EMC_RECALL_CACHE_SIZE:
                self._recall_cache.popitem(last=False)
        return results


def _touch_episodes(self, ids: list[int]) -> None:
    if not ids:
        return
    now = _utc_now_iso()
    try:
        placeholders = ",".join("?" * len(ids))
        self._conn.execute(
            f"""
            UPDATE emc_storage
            SET recall_count = recall_count + 1,
                last_recalled_at = ?
            WHERE id IN ({placeholders})
            """,
            [now] + ids,
        )
        self._conn.commit()
    except Exception as e:
        log.debug("EMC touch failed: %s", e)


def format_for_context(self, episodes: list[dict], *, max_chars: int | None = None) -> str | None:
    """Format episodic hits as a compact <episodic_context> block."""
    if not episodes:
        return None
    budget = EMC_CONTEXT_CHARS if max_chars is None else max(40, int(max_chars))
    lines = [
        "<episodic_context>",
        "Past conversation moments (what happened), not durable facts. "
        "Use silently for continuity. Never quote this block or claim total recall.",
        "",
    ]
    kept = False
    for ep in episodes:
        trace = (ep.get("trace") or ep.get("memory") or "").strip()
        if not trace:
            continue
        kept = True
        if len(trace) > EMC_CONTEXT_EPISODE_CHARS:
            trace = trace[:EMC_CONTEXT_EPISODE_CHARS].rstrip() + "…"
        when = ep.get("date") or (ep.get("timestamp") or "")[:10] or ""
        if when:
            lines.append(f"  - [{when}] {trace}")
        else:
            lines.append(f"  - {trace}")
    if not kept:
        return None
    lines.append("</episodic_context>")
    block = "\n".join(lines)
    closing_tag = "\n</episodic_context>"
    if len(block) > budget:
        block = block[:budget - len(closing_tag)].rstrip() + closing_tag
    return block


def attach_recall_to_store() -> None:
    """Bind search / format_for_context onto EpisodicStore (idempotent)."""
    from cognition.memory.episode import EpisodicStore

    if getattr(EpisodicStore, "_emc3_recall_attached", False):
        return
    EpisodicStore.search = search  # type: ignore[method-assign]
    EpisodicStore.format_for_context = format_for_context  # type: ignore[method-assign]
    EpisodicStore._emc3_recall_attached = True  # type: ignore[attr-defined]
    log.debug("EMC-3 recall methods attached to EpisodicStore")
