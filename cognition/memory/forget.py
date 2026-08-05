"""
memory/forget.py
Ebbinghaus-style exponential decay scoring for memory lifecycle management.

Core formula (inspired by the forgetting curve):
    weighted_score = min(access_count, ACCESS_COUNT_CAP) * 0.5^(days_since_last_access / H_eff)

Where H_eff = HALF_LIFE_DAYS * (1 + γ * intensity(valence))  (Phase 5 + 12R).
Intensity prefers valence_score (−2…+2) → |score|/2; falls back to valence_tag.

Called by memorize.py — no I/O, pure math only.
"""
from __future__ import annotations

from datetime import datetime, timezone
import os

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


def _valence_intensity(
    valence_tag: str | None = None,
    valence_score: int | float | None = None,
) -> float:
    """0..1 intensity for half-life stretch. Prefer 5-pt score when present."""
    if valence_score is not None and str(valence_score).strip() != "":
        try:
            s = int(valence_score)
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


def compute_weighted_score(
    access_count: int,
    last_accessed_iso: str,
    valence_tag: str | None = None,
    valence_score: int | float | None = None,
) -> float:
    """Compute exponential decay score for a memory entry.

    Score = min(access_count, cap) × 0.5^(days / H_eff)

    H_eff = HALF_LIFE_DAYS × (1 + γ × intensity).
    High emotional intensity (esp. |score|=2 / neg) decays slower.
    FORGET_EMOTION_GAMMA=0 disables emotion modification.
    """
    if not access_count or not last_accessed_iso or last_accessed_iso == "never":
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
        h_eff = max(h_eff, 1e-6)
        return float(min(access_count, ACCESS_COUNT_CAP)) * (0.5 ** (days / h_eff))
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
        )
        < CLEANUP_THRESHOLD
    )
