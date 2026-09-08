"""
Learn / Review helpers on top of the shared vocab pool + per-user SRS.

Mounted onto the main lingo router from __init__.py so we do not rewrite
router.py in this PR.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import Depends, HTTPException
from pydantic import BaseModel, Field

from .srs import LingoSRS
from . import spawn

log = logging.getLogger(__name__)


class LearnItem(BaseModel):
    pool_id: Optional[int] = None
    front: str
    back: str
    reading: str = ""
    kind: str = "kanji"


class LearnSession(BaseModel):
    items: List[LearnItem]
    pending_in_pool: int = 0


class MarkLearnedRequest(BaseModel):
    items: List[LearnItem] = Field(default_factory=list)


class MarkLearnedResponse(BaseModel):
    learned: int
    xp: int = 0


def attach_learn_routes(router, get_lingo_session, award_xp=None):
    """Register /learn/* routes on an existing APIRouter."""

    @router.get("/learn/new", response_model=LearnSession)
    async def learn_new(session: dict = Depends(get_lingo_session)):
        """Unlearnt shared-pool items (no LLM wait). Caps at 10."""
        uid = session["user_id"]
        # A+B: if pool is low, top up in background — never block the response
        if spawn.pool_count() < spawn.POOL_MIN:
            spawn.top_up_pool_background(level="beginner")
        items = spawn.pool_take_unlearned(uid, n=spawn.LEARN_SESSION_N)
        return LearnSession(
            items=[LearnItem(**it) for it in items],
            pending_in_pool=max(0, spawn.pool_count() - len(items)),
        )

    @router.post("/learn/mark", response_model=MarkLearnedResponse)
    async def learn_mark(body: MarkLearnedRequest, session: dict = Depends(get_lingo_session)):
        """Mark items as learnt → copy into per-user SRS + small XP."""
        uid = session["user_id"]
        n = spawn.mark_learned(uid, [it.model_dump() for it in body.items])
        xp_total = 0
        if award_xp and n:
            try:
                xp_total = award_xp(uid, n * 5)  # 5 XP per newly learnt card
            except Exception:
                log.warning("XP award on learn_mark failed", exc_info=True)
        return MarkLearnedResponse(learned=n, xp=xp_total)

    @router.get("/learn/status")
    async def learn_status(session: dict = Depends(get_lingo_session)):
        uid = session["user_id"]
        srs = LingoSRS(uid)
        stats = srs.get_stats()
        return {
            "pool_size": spawn.pool_count(),
            "user_cards": stats.get("total_cards", 0),
            "reviews_today": stats.get("reviews_today", 0),
            "learned_today": stats.get("learned_today", 0),
            "review_session_size": spawn.REVIEW_SESSION_N,
            "learn_session_size": spawn.LEARN_SESSION_N,
        }
