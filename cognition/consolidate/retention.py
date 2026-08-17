"""Retention scoring and gate helpers for monthly consolidation.

Mirrors :mod:`cognition.knowledge` structure: this module owns the retention
gate, anchor build, and row scoring used by the consolidation lifecycle.
"""
from __future__ import annotations

from datetime import datetime, timezone
import numpy as np

from system.log import get_logger
from cognition.memory.memorize import classify_kind, entities_from_json, SALIENCE_POLICY_RE

from .schema import (
    _DYNAMIC_ANCHOR_DAYS,
    _DYNAMIC_ANCHOR_LIMIT,
    _MONTHLY_FACT_TAG_RE,
    _MUST_KEEP_KEYWORDS,
    _NOVELTY_W_DYNAMIC,
    _NOVELTY_W_STATIC,
    _RETENTION_SPACING_SATURATION,
    _RETENTION_W_CONNECTIVITY,
    _RETENTION_W_NOVELTY,
    _RETENTION_W_SALIENCE,
    _RETENTION_W_SPACING,
    _RETENTION_W_VALENCE,
    CONSOLIDATION_ANCHOR_K,
    CONSOLIDATION_ANCHOR_LOOKBACK,
    CONSOLIDATION_MAX_MONTH,
    CONSOLIDATION_MIN_MONTH,
    CONSOLIDATION_SOFT_THRESHOLD,
)

log = get_logger(__name__)

__all__ = [
    "apply_retention_gate",
    "build_dynamic_anchors",
    "build_static_anchors",
    "entity_connectivity_weights",
    "is_must_keep",
    "score_daily_row",
]


def is_must_keep(text: str) -> bool:
    kind = classify_kind(text, default="fact")
    if kind in ("event", "plan"):
        return True
    low = (text or "").casefold()
    return any(k in low for k in _MUST_KEEP_KEYWORDS)


def entity_connectivity_weights(memorize, user_id: str) -> dict[str, float]:
    try:
        lock = getattr(getattr(memorize, "_mem", None), "_db_lock", None)
        if lock is not None:
            with lock:
                rows = memorize._conn.execute(
                    "SELECT entity_a, entity_b, weight FROM entity_relations WHERE user_id = ?",
                    (user_id,),
                ).fetchall()
        else:
            rows = memorize._conn.execute(
                "SELECT entity_a, entity_b, weight FROM entity_relations WHERE user_id = ?",
                (user_id,),
            ).fetchall()
    except Exception as exc:
        log.debug("Connectivity weights: entity_relations read skipped: %s", exc)
        return {}

    weights: dict[str, float] = {}
    for row in rows:
        a = str(row["entity_a"] or "")
        b = str(row["entity_b"] or "")
        w = float(row["weight"] or 0.0)
        if a:
            weights[a] = weights.get(a, 0.0) + w
        if b:
            weights[b] = weights.get(b, 0.0) + w
    return weights


def build_static_anchors(memorize, user_id: str) -> "np.ndarray | None":
    # Phase 6: get_all + filter is OK monthly; SQL limit if this ever shows up in profiles.
    try:
        all_mems = memorize.get_all(user_id=user_id)
    except Exception as exc:
        log.warning("Static anchor build: failed to fetch memories: %s", exc)
        return None

    monthly_texts = [
        (m.get("memory") or "").strip()
        for m in sorted(all_mems, key=lambda m: m.get("created_at", "") or "", reverse=True)
        if _MONTHLY_FACT_TAG_RE.match((m.get("memory") or "").strip())
    ][:CONSOLIDATION_ANCHOR_LOOKBACK]

    if len(monthly_texts) < 2:
        return None

    try:
        vectors = np.array(memorize.embed_texts(monthly_texts, query=False))
    except Exception as exc:
        log.warning("Static anchor build: embedding failed: %s", exc)
        return None

    if vectors.size == 0:
        return None

    k = max(1, min(CONSOLIDATION_ANCHOR_K, len(monthly_texts) // 3 or 1))
    if k >= 2:
        try:
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=k, random_state=42, n_init=5)
            km.fit(vectors)
            return km.cluster_centers_
        except Exception as exc:
            log.debug(
                "Static anchor build: kmeans unavailable/failed (%s); "
                "falling back to a single mean-vector anchor.", exc,
            )

    return np.mean(vectors, axis=0, keepdims=True)


def build_dynamic_anchors(memorize, user_id: str) -> "np.ndarray | None":
    """Mean vector of recently active memories — 'what has been active lately'.

    Used at monthly gate time (not turn-level WMC). Prefers rows with recent
    last_accessed_at / created_at within DYNAMIC_ANCHOR_DAYS, capped at LIMIT.
    Returns shape (1, dim) or None when insufficient data.
    """
    try:
        all_mems = memorize.get_all(user_id=user_id)
    except Exception as exc:
        log.warning("Dynamic anchor build: failed to fetch memories: %s", exc)
        return None
    if not all_mems:
        return None

    from datetime import timedelta
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=_DYNAMIC_ANCHOR_DAYS)

    def _ts(m):
        # Prefer last_accessed; treat "never" as missing so created_at can win.
        raw = m.get("last_accessed_at") or ""
        if not raw or str(raw).strip().lower() == "never":
            raw = m.get("created_at") or ""
        if not raw or str(raw).strip().lower() == "never":
            return None
        try:
            dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            return None

    scored = []
    for m in all_mems:
        status = m.get("status")
        if status is not None and str(status).strip().lower() not in ("active", ""):
            continue

        text = (m.get("memory") or "").strip()
        if not text:
            continue
        if _MONTHLY_FACT_TAG_RE.match(text):
            continue

        dt = _ts(m)
        if dt is None or dt < cutoff:
            continue

        ac = int(m.get("access_count") or 0)
        scored.append((dt, ac, text))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    texts = [t for _, _, t in scored[:_DYNAMIC_ANCHOR_LIMIT]]
    if len(texts) < 2:
        return None
    try:
        vectors = np.array(memorize.embed_texts(texts, query=False))
    except Exception as exc:
        log.warning("Dynamic anchor build: embedding failed: %s", exc)
        return None
    if vectors.size == 0:
        return None
    return np.mean(vectors, axis=0, keepdims=True)


def score_daily_row(
    row: dict,
    *,
    static_anchors: "np.ndarray | None",
    dynamic_anchors: "np.ndarray | None",
    row_vector: "np.ndarray | None",
    entity_weights: dict[str, float],
    entity_weight_cap: float,
    entity_importance: dict[str, float] | None = None,
) -> float:
    text = row.get("_text", "") or ""
    entities = entities_from_json(row.get("entities"))

    # Phase 4: prefer stored turn tags; fall back to text scan for legacy rows.
    stored_hit = row.get("salience_hit")
    if stored_hit is not None and str(stored_hit) != "":
        salience = 1.0 if int(stored_hit) else 0.3
    else:
        salience = 1.0 if SALIENCE_POLICY_RE.search(text) else 0.3

    vs = row.get("valence_score")
    if vs is not None and str(vs).strip() != "":
        try:
            s = max(-2, min(2, int(vs)))
            valence = 0.25 + 0.30 * abs(s)
        except (TypeError, ValueError):
            vs = None
    if vs is None or str(vs).strip() == "":
        v_raw = (row.get("valence_tag") or "neutral")
        if isinstance(v_raw, str):
            v_raw = v_raw.strip().lower()
        else:
            v_raw = "neutral"
        if v_raw == "neg":
            valence = 0.85
        elif v_raw == "pos":
            valence = 0.65
        else:
            valence = 0.25

    # Phase 2: distinct recall days (access_day_count). Fallback for pre-Phase-2 rows.
    day_count = int(row.get("access_day_count") or 0)
    if day_count <= 0:
        day_count = 1 if int(row.get("access_count") or 0) > 0 else 0
    spacing = min(1.0, day_count / float(_RETENTION_SPACING_SATURATION))

    # Phase 3: blend edge connectivity with entity importance I_e.
    edge_conn = 0.0
    if entities and entity_weights:
        raw = [entity_weights.get(e.casefold(), 0.0) for e in entities]
        edge_conn = min(1.0, (sum(raw) / len(raw)) / max(entity_weight_cap, 1e-6))
    ie = 0.0
    if entities and entity_importance:
        vals = [entity_importance.get(e.casefold(), 0.0) for e in entities]
        ie = max(vals) if vals else 0.0
    connectivity = 0.5 * edge_conn + 0.5 * ie if (entities and (entity_weights or entity_importance)) else 0.0

    # Phase 6: novelty = blend of distance-to-static and distance-to-dynamic anchors.
    def _one_minus_max_sim(anchors, vec):
        if anchors is None or not len(anchors) or vec is None:
            return None
        try:
            norms = np.linalg.norm(anchors, axis=1) * np.linalg.norm(vec) + 1e-9
            sims = (anchors @ vec) / norms
            return float(max(0.0, min(1.0, 1.0 - float(np.max(sims)))))
        except Exception:
            return None

    n_static = _one_minus_max_sim(static_anchors, row_vector)
    n_dynamic = _one_minus_max_sim(dynamic_anchors, row_vector)
    w_s = max(0.0, _NOVELTY_W_STATIC)
    w_d = max(0.0, _NOVELTY_W_DYNAMIC)
    w_sum = w_s + w_d
    if w_sum <= 0:
        w_s, w_d, w_sum = 1.0, 0.0, 1.0
    w_s, w_d = w_s / w_sum, w_d / w_sum
    if n_static is None and n_dynamic is None:
        novelty = 0.5
    elif n_static is None:
        novelty = n_dynamic if n_dynamic is not None else 0.5
    elif n_dynamic is None:
        novelty = n_static
    else:
        novelty = w_s * n_static + w_d * n_dynamic

    return (
        _RETENTION_W_SALIENCE * salience
        + _RETENTION_W_NOVELTY * novelty
        + _RETENTION_W_SPACING * spacing
        + _RETENTION_W_CONNECTIVITY * connectivity
        + _RETENTION_W_VALENCE * valence
    )


def apply_retention_gate(
    memorize,
    user_id: str,
    daily_rows: list[dict],
) -> tuple[list[dict], dict]:
    must_keep_rows: list[dict] = []
    candidate_rows: list[dict] = []

    for row in daily_rows:
        text = row.get("_text", "") or ""
        if row.get("_cognitive_lesson") or is_must_keep(text):
            must_keep_rows.append(row)
        else:
            candidate_rows.append(row)

    if not candidate_rows:
        return must_keep_rows, {
            "must_keep": len(must_keep_rows),
            "candidates": 0,
            "kept_candidates": 0,
            "dropped_candidates": 0,
        }

    static_anchors = build_static_anchors(memorize, user_id)
    dynamic_anchors = build_dynamic_anchors(memorize, user_id)
    entity_weights = entity_connectivity_weights(memorize, user_id)
    entity_weight_cap = max(entity_weights.values(), default=1.0) or 1.0
    entity_importance: dict[str, float] = {}
    try:
        from cognition.memory.entity import compute_entity_importance_map
        entity_importance = compute_entity_importance_map(memorize, user_id) or {}
    except Exception as exc:
        log.debug("Entity importance map skipped: %s", exc)

    candidate_texts = [row.get("_text", "") or "" for row in candidate_rows]
    candidate_vectors = None
    try:
        candidate_vectors = np.array(memorize.embed_texts(candidate_texts, query=False))
    except Exception as exc:
        log.warning(
            "Retention gate: failed to embed %d candidate row(s), "
            "novelty will default to neutral for this run: %s",
            len(candidate_texts), exc,
        )

    scored: list[tuple[float, dict]] = []
    for i, row in enumerate(candidate_rows):
        vec = candidate_vectors[i] if candidate_vectors is not None else None
        score = score_daily_row(
            row,
            static_anchors=static_anchors,
            dynamic_anchors=dynamic_anchors,
            row_vector=vec,
            entity_weights=entity_weights,
            entity_weight_cap=entity_weight_cap,
            entity_importance=entity_importance,
        )
        scored.append((score, row))

    scored.sort(key=lambda pair: pair[0], reverse=True)

    above_threshold = sum(1 for score, _ in scored if score >= CONSOLIDATION_SOFT_THRESHOLD)
    target_count = max(CONSOLIDATION_MIN_MONTH, min(CONSOLIDATION_MAX_MONTH, above_threshold))
    target_count = min(target_count, len(scored))

    kept_candidates = [row for _, row in scored[:target_count]]

    log.info(
        "Retention gate: %d must_keep, %d candidates scored (threshold=%.2f "
        "above=%d), target_count=%d -> kept=%d dropped=%d",
        len(must_keep_rows), len(scored), CONSOLIDATION_SOFT_THRESHOLD,
        above_threshold, target_count, len(kept_candidates), len(scored) - target_count,
    )

    return must_keep_rows + kept_candidates, {
        "must_keep": len(must_keep_rows),
        "candidates": len(scored),
        "kept_candidates": len(kept_candidates),
        "dropped_candidates": len(scored) - target_count,
    }