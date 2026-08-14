"""
EMC-3: episodic recall helpers attached onto EpisodicStore at boot.

Kept separate so EMC-3 can land without rewriting the large episode.py body.
apply_emc2_hooks() calls attach_recall_to_store().
"""
from __future__ import annotations

import json
from typing import Any

import sqlite_vec

from system.log import get_logger
from cognition.memory.search import _sanitize_fts_query
from cognition.memory.env import env_bool, env_int

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
        return []

    fts_query = _sanitize_fts_query(q)

    with self._lock:
        rank_knn: dict[int, int] = {}
        rank_fts: dict[int, int] = {}
        try:
            vec_blob = sqlite_vec.serialize_float32(vector)
            knn_rows = self._conn.execute(
                """
                SELECT v.rowid AS id, vec_distance_cosine(v.embedding, ?) AS dist
                FROM emc_vec v
                JOIN emc_storage s ON s.id = v.rowid
                WHERE s.user_id = ?
                  AND (s.superseded_by IS NULL)
                ORDER BY dist ASC
                LIMIT ?
                """,
                (vec_blob, uid, EMC_KNN_LIMIT),
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
                   salience_score, entities, source, session_id, recall_count
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
                "_recall_score": scores.get(eid, 0.0),
                "_emc": True,
            })

        _touch_episodes(self, [r["id"] for r in results])
        return results


def _touch_episodes(self, ids: list[int]) -> None:
    if not ids:
        return
    now = _utc_now_iso()
    try:
        for eid in ids:
            self._conn.execute(
                """
                UPDATE emc_storage
                SET recall_count = recall_count + 1,
                    last_recalled_at = ?
                WHERE id = ?
                """,
                (now, eid),
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
    if len(block) > budget:
        block = block[:budget].rstrip() + "\n</episodic_context>"
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
