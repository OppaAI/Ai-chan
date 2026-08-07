"""Experience engram graph/lifecycle: explicit relations between experiences."""
from __future__ import annotations

from cognition.memory.vecstore import utc_now_iso
from system.log import get_logger
from system.userspace import current_user_id

from .schema import ExperienceSchema, connect

log = get_logger(__name__)

# ── Engram relations ───────────────────────────────────────────────────────────
# Explicit links between experiences: continuation, contradiction, refines, synthesizes

RELATION_TYPES = ("continuation", "contradiction", "refines", "synthesizes")


class ExperienceLifecycle:
    """Owns the engram-relation graph (explicit links between experiences)."""

    def __init__(self, schema: ExperienceSchema | None = None):
        self.schema = schema or ExperienceSchema()

    def record_relation(
        self,
        from_engram: str,
        to_engram: str,
        relation_type: str,
        confidence: float = 1.0,
        user_id: str | None = None,
    ) -> bool:
        return record_engram_relation(from_engram, to_engram, relation_type, confidence, user_id=user_id)

    def get_relations(self, engram_id: str, direction: str = "both", user_id: str | None = None) -> list[dict]:
        return get_engram_relations(engram_id, direction=direction, user_id=user_id)


def record_engram_relation(from_engram: str, to_engram: str, relation_type: str, confidence: float = 1.0, user_id: str | None = None) -> bool:
    """Record an explicit relation between two experiences.

    Args:
        from_engram: Source experience ID
        to_engram: Target experience ID
        relation_type: One of 'continuation', 'contradiction', 'refines', 'synthesizes'
        confidence: 0.0-1.0 confidence score
        user_id: Optional user ID (defaults to current)

    Returns:
        True if recorded, False if invalid relation type or DB error
    """
    if relation_type not in RELATION_TYPES:
        log.warning("Invalid relation_type %r; must be one of %s", relation_type, RELATION_TYPES)
        return False
    uid = user_id or current_user_id()
    conn = connect(uid)
    try:
        conn.execute(
            """INSERT OR REPLACE INTO engram_relations
               (from_engram, to_engram, relation_type, confidence, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (from_engram, to_engram, relation_type, max(0.0, min(1.0, confidence)), utc_now_iso())
        )
        conn.commit()
        return True
    except Exception as exc:
        log.warning("record_engram_relation failed: %s", exc)
        return False
    finally:
        conn.close()


def get_engram_relations(engram_id: str, direction: str = "both", user_id: str | None = None) -> list[dict]:
    """Fetch relations for an engram.

    Args:
        engram_id: Experience ID to query
        direction: 'outgoing' (from), 'incoming' (to), or 'both'
        user_id: Optional user ID (defaults to current)

    Returns:
        List of relation dicts with keys: from_engram, to_engram, relation_type, confidence, created_at
    """
    uid = user_id or current_user_id()
    conn = connect(uid)
    try:
        if direction == "outgoing":
            rows = conn.execute(
                "SELECT from_engram, to_engram, relation_type, confidence, created_at FROM engram_relations WHERE from_engram=?",
                (engram_id,)
            ).fetchall()
        elif direction == "incoming":
            rows = conn.execute(
                "SELECT from_engram, to_engram, relation_type, confidence, created_at FROM engram_relations WHERE to_engram=?",
                (engram_id,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT from_engram, to_engram, relation_type, confidence, created_at FROM engram_relations WHERE from_engram=? OR to_engram=?",
                (engram_id, engram_id)
            ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()