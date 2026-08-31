"""
memory/forget.py
Ebbinghaus-style exponential decay scoring for memory lifecycle management.

Core formula (inspired by the forgetting curve):
    strength = 1 + log1p(min(access_count, ACCESS_COUNT_CAP))
    weighted_score = strength * 0.5^(days_since_last_access / H_eff)

Where:
    H_eff = HALF_LIFE_DAYS
            * (1 + γ * intensity(valence))
            * (1 + SPACING_GAMMA * log1p(access_day_count))
            * mood_factor

Intensity prefers valence_score (−2…+2) → |score|/2; falls back to valence_tag.
Mood factor uses query_valence when provided; offline cleanup falls back to the
user's ambient edge-cognitive affect so mood-dependent forgetting is not dead.

Access strength uses log1p so frequent recall still helps retention without
letting raw access_count dominate both ranking *and* survival (the old linear
multiplier made heavily-touched memories nearly undeletable).

Called by memorize.py — no I/O on the pure math path; ambient mood lookup is
best-effort and never raises.
"""
from __future__ import annotations

from datetime import datetime, timezone
import math
import os

# ── mood-dependent forgetting ──────────────────────────────────────────────
FORGET_MOOD_MATCH_ENABLED = os.getenv("FORGET_MOOD_MATCH_ENABLED", "1").lower() in {"1", "true", "yes", "on"}
FORGET_MOOD_MATCH_SLOWDOWN = float(os.getenv("FORGET_MOOD_MATCH_SLOWDOWN", "1.3"))
FORGET_MOOD_MISMATCH_ACCELERATION = float(os.getenv("FORGET_MOOD_MISMATCH_ACCELERATION", "1.3"))
# When query/ambient valence matches memory valence (both non-zero): H_eff × slowdown.
# When mismatched: H_eff ÷ acceleration. Neutral/zero → no mood modulation.

# ── tunable parameters ────────────────────────────────────────────────────────
HALF_LIFE_DAYS    = float(os.getenv("FORGET_HALF_LIFE_DAYS",    "21.0"))
CLEANUP_THRESHOLD = float(os.getenv("FORGET_CLEANUP_THRESHOLD", "0.02"))
ACCESS_COUNT_CAP  = int(  os.getenv("FORGET_ACCESS_COUNT_CAP",  "255"))
GRACE_PERIOD_DAYS = int(  os.getenv("FORGET_GRACE_PERIOD_DAYS", "35"))

# Phase 5: emotion imprint — lengthens effective half-life for high-intensity valence.
EMOTION_GAMMA = float(os.getenv("FORGET_EMOTION_GAMMA", "0.5"))
_INTENSITY = {
    "neg": float(os.getenv("FORGET_INTENSITY_NEG", "1.0")),
    "pos": float(os.getenv("FORGET_INTENSITY_POS", "0.4")),
    "neutral": float(os.getenv("FORGET_INTENSITY_NEUTRAL", "0.0")),
}

# Spaced-repetition stretch from distinct local days of access.
# access_day_count was tracked but unused in live decay; fold it into H_eff.
SPACING_GAMMA = float(os.getenv("FORGET_SPACING_GAMMA", "0.35"))

# Proportional negative-recall soft penalty (replaces flat −0.015).
NEG_PENALTY_BASE = float(os.getenv("MEMORY_NEG_RECALL_AVOID_WEIGHT", "0.015"))
NEG_PENALTY_SCALE = float(os.getenv("MEMORY_NEG_PENALTY_SCALE", "1.0"))


def _valence_intensity(
    valence_tag: str | None = None,
    valence_score: int | float | None = None,
) -> float:
    """0..1 intensity for half-life stretch. Prefer 5-pt score when present."""
    if valence_score is not None and str(valence_score).strip() != "":
        try:
            s = float(valence_score)
            s = max(-2, min(2, s))
            return min(1.0, abs(s) / 2.0)
        except (TypeError, ValueError):
            pass
    key = (valence_tag or "neutral").strip().lower()
    try:
        v = float(_INTENSITY.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0
    if v != v or v in (float("inf"), float("-inf")):
        return 0.0
    return max(0.0, v)


def _memory_valence_sign(
    valence_tag: str | None = None,
    valence_score: int | float | None = None,
) -> int:
    """Return −1 / 0 / +1 for mood match logic."""
    if valence_score is not None and str(valence_score).strip() != "":
        try:
            s = int(float(valence_score))
            if s > 0:
                return 1
            if s < 0:
                return -1
        except (TypeError, ValueError):
            pass
    key = (valence_tag or "neutral").strip().lower()
    if key in ("pos", "positive"):
        return 1
    if key in ("neg", "negative"):
        return -1
    return 0


def _access_strength(access_count: int) -> float:
    """Log-compressed strength so access helps retention without linear runaway.

    Old: strength = min(ac, 255)  → one heavily recalled memory dwarfs all else.
    New: strength = 1 + log1p(min(ac, 255))  → diminishing returns, still monotonic.
    """
    try:
        ac = max(0, int(access_count or 0))
    except (TypeError, ValueError):
        return 0.0
    if ac <= 0:
        return 0.0
    return 1.0 + math.log1p(min(ac, ACCESS_COUNT_CAP))


def resolve_ambient_valence(user_id: str | None = None) -> int | None:
    """Best-effort ambient mood from edge cognitive state (offline cleanup).

    Maps affect ∈ [−1, 1] to a coarse valence sign used as query_valence.
    Returns None when unavailable so callers keep neutral behavior.
    """
    if not FORGET_MOOD_MATCH_ENABLED:
        return None
    try:
        from cognition.attention import for_identity
        from system.userspace import current_user_id

        uid = user_id or current_user_id()
        if not uid or uid in ("default", "guest"):
            return None
        snap = for_identity(uid).snapshot()
        affect = float(snap.get("affect") or 0.0)
        if affect > 0.2:
            return 1
        if affect < -0.2:
            return -1
        return 0
    except Exception:
        return None


def compute_weighted_score(
    access_count: int,
    last_accessed_iso: str,
    valence_tag: str | None = None,
    valence_score: int | float | None = None,
    query_valence: int | None = None,
    access_day_count: int | None = None,
) -> float:
    """Compute exponential decay score for a memory entry.

    Score = (1 + log1p(min(ac, cap))) × 0.5^(days / H_eff)

    H_eff = HALF_LIFE_DAYS
            × (1 + γ × intensity)
            × (1 + SPACING_GAMMA × log1p(access_day_count))
            × mood_factor

    Mood factor:
      • query_valence matches memory valence (both non-zero) → × slowdown
      • query_valence mismatches memory valence → ÷ acceleration
      • query_valence is None or zero → no mood modulation
    """
    strength = _access_strength(access_count)
    if strength <= 0.0 or not last_accessed_iso or last_accessed_iso == "never":
        return 0.0

    try:
        ts = last_accessed_iso.replace("Z", "+00:00")
        last_dt = datetime.fromisoformat(ts)
        if last_dt.tzinfo is None:
            last_dt = last_dt.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        days = max(0.0, (now - last_dt).total_seconds() / 86400.0)
        intensity = _valence_intensity(valence_tag, valence_score)
        h_eff = HALF_LIFE_DAYS * (1.0 + max(0.0, EMOTION_GAMMA) * intensity)

        # Spaced-repetition: distinct access days stretch half-life.
        try:
            day_n = max(0, int(access_day_count or 0))
        except (TypeError, ValueError):
            day_n = 0
        if day_n > 0 and SPACING_GAMMA > 0:
            h_eff *= 1.0 + SPACING_GAMMA * math.log1p(day_n)

        # Mood-dependent forgetting (query or ambient sign).
        if FORGET_MOOD_MATCH_ENABLED and query_valence is not None and query_valence != 0:
            mem_sign = _memory_valence_sign(valence_tag, valence_score)
            q_sign = 1 if query_valence > 0 else (-1 if query_valence < 0 else 0)
            if mem_sign != 0 and q_sign != 0 and mem_sign == q_sign:
                h_eff *= FORGET_MOOD_MATCH_SLOWDOWN
            elif mem_sign != 0 and q_sign != 0 and mem_sign != q_sign:
                h_eff /= FORGET_MOOD_MISMATCH_ACCELERATION

        h_eff = max(h_eff, 1e-6)
        return float(strength) * (0.5 ** (days / h_eff))
    except Exception:
        return 0.0


def is_grace_protected(created_at_iso: str) -> bool:
    """True while memory is inside the post-write grace window."""
    if not created_at_iso or created_at_iso == "never":
        return False

    try:
        ts = created_at_iso.replace("Z", "+00:00")
        created_dt = datetime.fromisoformat(ts)
        if created_dt.tzinfo is None:
            created_dt = created_dt.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        days = max(0, (now - created_dt).total_seconds() / 86400)
        return days < GRACE_PERIOD_DAYS
    except Exception:
        return False


def should_cleanup(
    access_count: int,
    last_accessed_iso: str,
    created_at_iso: str,
    valence_tag: str | None = None,
    valence_score: int | float | None = None,
    query_valence: int | None = None,
    access_day_count: int | None = None,
) -> bool:
    """Return True if a memory is a deletion candidate."""
    if is_grace_protected(created_at_iso):
        return False
    return (
        compute_weighted_score(
            access_count,
            last_accessed_iso,
            valence_tag=valence_tag,
            valence_score=valence_score,
            query_valence=query_valence,
            access_day_count=access_day_count,
        )
        < CLEANUP_THRESHOLD
    )


def negative_recall_penalty(
    valence_tag: str | None = None,
    valence_score: int | float | None = None,
) -> float:
    """Soft proportional penalty for negative memories at recall time.

    Replaces a flat −NEG_PENALTY_BASE with intensity scaling:
        penalty = NEG_PENALTY_BASE * NEG_PENALTY_SCALE * intensity
    so mildly negative facts are only lightly suppressed while strong
    negatives still get pushed down unless the query engages emotion.
    Returns 0.0 for non-negative memories.
    """
    if _memory_valence_sign(valence_tag, valence_score) >= 0:
        return 0.0
    intensity = max(_valence_intensity(valence_tag, valence_score), 0.25)
    return float(NEG_PENALTY_BASE) * max(0.0, NEG_PENALTY_SCALE) * intensity


def salience_score(
    *,
    text: str = "",
    access_count: int = 0,
    access_day_count: int = 0,
    valence_tag: str | None = None,
    valence_score: int | float | None = None,
    salience_hit: int | None = None,
    age_days: float | None = None,
) -> float:
    """Composite 0..1 salience for dream-boost decisions (not pure boolean OR).

    Combines stored salience flag, emotional intensity, access strength,
    spacing, and a mild recency prior. Callers can threshold (e.g. ≥ 0.35)
    instead of treating every matching heuristic as equally important.
    """
    score = 0.0
    if salience_hit:
        try:
            if int(salience_hit):
                score += 0.45
        except (TypeError, ValueError):
            pass

    intensity = _valence_intensity(valence_tag, valence_score)
    score += 0.30 * intensity

    try:
        ac = max(0, int(access_count or 0))
    except (TypeError, ValueError):
        ac = 0
    if ac >= 3:
        score += min(0.25, 0.08 * math.log1p(ac))

    try:
        days = max(0, int(access_day_count or 0))
    except (TypeError, ValueError):
        days = 0
    if days >= 2:
        score += min(0.15, 0.06 * math.log1p(days))

    if age_days is not None:
        try:
            age = max(0.0, float(age_days))
            if age <= 7.0:
                score += 0.15 * (1.0 - age / 7.0)
        except (TypeError, ValueError):
            pass

    # Light text prior when no structured signals fired.
    if score < 0.2 and text:
        try:
            from cognition.memory.entity import SALIENCE_POLICY_RE
            if SALIENCE_POLICY_RE.search(text or ""):
                score += 0.35
        except Exception:
            pass

    return max(0.0, min(1.0, score))
