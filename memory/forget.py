"""
memory/forget.py
Ebbinghaus-style exponential decay scoring for memory lifecycle management.

Core formula (inspired by the forgetting curve):
    weighted_score = min(access_count, ACCESS_COUNT_CAP) * 0.5^(days_since_last_access / HALF_LIFE_DAYS)

This mirrors how biological memory consolidation works:
  - Frequently accessed memories persist longer   (reinforced neural pathways)
  - Unused memories naturally decay               (pruned rarely-used connections)
  - New memories get a protection window          (working memory consolidation)
  - Decay is gradual, not abrupt                  (unlike hard TTL which kills instantly)

Called by memorize.py — no I/O, pure math only.
"""
from __future__ import annotations

from datetime import datetime, timezone
import os

# ── tunable parameters ────────────────────────────────────────────────────────
HALF_LIFE_DAYS    = float(os.getenv("FORGET_HALF_LIFE_DAYS",    7.0))
CLEANUP_THRESHOLD = float(os.getenv("FORGET_CLEANUP_THRESHOLD", 0.05))
ACCESS_COUNT_CAP  = int(  os.getenv("FORGET_ACCESS_COUNT_CAP",  255))
GRACE_PERIOD_DAYS = int(  os.getenv("FORGET_GRACE_PERIOD_DAYS", 14))

# ── scoring ───────────────────────────────────────────────────────────────────

def compute_weighted_score(access_count: int, last_accessed_iso: str) -> float:
    """Compute exponential decay score for a memory entry.

    Score = min(access_count, 255) × 0.5^(days_elapsed / HALF_LIFE_DAYS)

    access_count is capped at ACCESS_COUNT_CAP so even a memory retrieved
    thousands of times has a bounded score — combined with half-life decay,
    old memories reliably fade toward zero.

    Timestamp parsing handles UTC, timezone-aware, naive, Z-suffix, and
    negative-offset ISO strings via fromisoformat + tzinfo coercion.
    On parse failure returns 0.0 so broken timestamps expire immediately
    rather than becoming immortal zombies.

    Examples (HALF_LIFE_DAYS=7):
      - 3 accesses, accessed today          → 3.0
      - 10 accesses, last seen 21 days ago  → 1.25
      - 1 access,  last seen 33 days ago    → ~0.05  (at cleanup threshold)
    """
    if not access_count or not last_accessed_iso or last_accessed_iso == "never":
        return 0.0

    try:
        ts      = last_accessed_iso.replace("Z", "+00:00")
        last_dt = datetime.fromisoformat(ts)
        if last_dt.tzinfo is None:
            last_dt = last_dt.replace(tzinfo=timezone.utc)
        now  = datetime.now(timezone.utc)
        days = max(0, (now - last_dt).total_seconds() / 86400)
        return float(min(access_count, ACCESS_COUNT_CAP)) * (0.5 ** (days / HALF_LIFE_DAYS))
    except Exception:
        # Timestamp parse failure → treat as expired; never leave immortal zombies
        return 0.0


def is_grace_protected(created_at_iso: str) -> bool:
    """Return True if a memory is still within the grace period protection window.

    Grace period prevents freshly created memories from being swept by cleanup()
    before they accumulate enough access_count to score above threshold — mirrors
    how working memory consolidation protects new engrams before LTP sets in.

    On parse failure returns False — unknown creation time gets no protection.
    """
    if not created_at_iso or created_at_iso == "never":
        return False

    try:
        ts         = created_at_iso.replace("Z", "+00:00")
        created_dt = datetime.fromisoformat(ts)
        if created_dt.tzinfo is None:
            created_dt = created_dt.replace(tzinfo=timezone.utc)
        now  = datetime.now(timezone.utc)
        days = max(0, (now - created_dt).total_seconds() / 86400)
        return days < GRACE_PERIOD_DAYS
    except Exception:
        return False

# ── lifecycle gate ─────────────────────────────────────────────────────────────

def should_cleanup(access_count: int, last_accessed_iso: str, created_at_iso: str) -> bool:
    """Return True if a memory is a deletion candidate.

    A memory is prunable only when both conditions hold:
      1. Grace period has expired    (age > GRACE_PERIOD_DAYS)
      2. Decay score is below threshold (weighted_score < CLEANUP_THRESHOLD)

    Called per-memory inside memorize.cleanup() and memorize.dream() prune stage.
    """
    if is_grace_protected(created_at_iso):
        return False

    return compute_weighted_score(access_count, last_accessed_iso) < CLEANUP_THRESHOLD