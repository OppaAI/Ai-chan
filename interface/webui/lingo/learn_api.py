"""
Learn helpers on shared vocab pool + per-user SRS.

Mutating routes require a real authenticated session (no owner fallback).
"""
from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import Depends, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from .srs import LingoSRS
from . import spawn
from .lingo_store import JLPT_ORDER, normalize_level, init_user_db

log = logging.getLogger(__name__)

# Seed pool so first-ever Learn open is instant (no LLM wait).
_INSTANT_SEED = [
    {"front": "ねこ", "back": "cat", "reading": "ねこ", "kind": "kanji"},
    {"front": "水", "back": "water", "reading": "みず", "kind": "kanji"},
    {"front": "ありがとう", "back": "thank you", "reading": "ありがとう", "kind": "phrase"},
    {"front": "おはよう", "back": "good morning", "reading": "おはよう", "kind": "phrase"},
    {"front": "すき", "back": "liking", "reading": "すき", "kind": "kanji"},
]


def _user_level(uid: str) -> str:
    try:
        import sqlite3
        path = init_user_db(uid)
        con = sqlite3.connect(str(path))
        try:
            row = con.execute("SELECT level FROM user_level WHERE id=1").fetchone()
            if row and row[0]:
                return normalize_level(row[0])
        finally:
            con.close()
    except Exception:
        pass
    return "N5"


class LearnItem(BaseModel):
    pool_id: Optional[int] = None
    front: str = ""
    back: str = ""
    reading: str = ""
    kind: str = "kanji"


class LearnSession(BaseModel):
    items: List[LearnItem]
    pending_in_pool: int = 0


class MarkLearnedRequest(BaseModel):
    items: List[LearnItem] = Field(default_factory=list)

    @field_validator("items")
    @classmethod
    def _cap_items(cls, v: list) -> list:
        if len(v) > spawn.LEARN_SESSION_N:
            return v[: spawn.LEARN_SESSION_N]
        return v


class MarkLearnedResponse(BaseModel):
    learned: int
    xp: int = 0


async def require_lingo_user(request: Request) -> dict:
    """Authenticated session only — no AIKO_USER_ID fallback (for mutating routes)."""
    from interface.webui import auth
    try:
        session = await auth.require_session(request)
        return await auth.require_accepted_session(session)
    except HTTPException:
        raise
    except Exception as e:
        log.warning("Lingo auth required failed: %s", e)
        raise HTTPException(status_code=401, detail="Authentication required")


def _allowed_levels(uid: str) -> list:
    """Equal-or-lower JLPT only: N5 sees N5; N4 sees N5+N4."""
    lvl = _user_level(uid)
    idx = JLPT_ORDER.index(lvl) if lvl in JLPT_ORDER else 0
    return JLPT_ORDER[:idx + 1]


def attach_learn_routes(router, get_lingo_session, award_xp=None):
    """Register /learn/* routes on an existing APIRouter."""

    @router.get("/learn/new", response_model=LearnSession)
    async def learn_new(session: dict = Depends(get_lingo_session)):
        """Unlearnt shared-pool items at/below user level (never blocks on LLM)."""
        from fastapi import Query
        uid = session["user_id"]
        level = _user_level(uid)
        allowed = _allowed_levels(uid)
        if spawn.pool_count(level) == 0:
            # Instant seed so first open never waits; LLM top-up in background.
            try:
                spawn.pool_add(_INSTANT_SEED, level=level)
            except Exception:
                pass
        if spawn.pool_count(level) < spawn.POOL_MIN:
            spawn.top_up_pool_background(level=level)
        items = spawn.pool_take_unlearned(uid, n=spawn.LEARN_SESSION_N, levels=allowed)
        if not items:
            # Fall back to any level so users with skewed pools still study.
            items = spawn.pool_take_unlearned(uid, n=spawn.LEARN_SESSION_N)
        unlearned_total = spawn.pool_count_unlearned(uid, levels=allowed)
        return LearnSession(
            items=[LearnItem(**it) for it in items],
            pending_in_pool=max(0, unlearned_total - len(items)),
        )

    @router.post("/learn/mark", response_model=MarkLearnedResponse)
    async def learn_mark(
        body: MarkLearnedRequest,
        session: dict = Depends(require_lingo_user),
    ):
        """Mark pool items as learnt → SRS + XP. Requires real auth; uses pool_id only."""
        uid = session["user_id"]
        # Only pool_ids are trusted; client front/back are ignored in mark_learned
        payload = [{"pool_id": it.pool_id} for it in body.items if it.pool_id is not None]
        if not payload:
            raise HTTPException(status_code=400, detail="Each item needs a valid pool_id")
        n = spawn.mark_learned(uid, payload)
        xp_total = 0
        if award_xp and n:
            try:
                xp_total = award_xp(uid, n * 5)
            except Exception:
                log.warning("XP award on learn_mark failed", exc_info=True)
        return MarkLearnedResponse(learned=n, xp=xp_total)

    @router.get("/learn/status")
    async def learn_status(session: dict = Depends(get_lingo_session)):
        uid = session["user_id"]
        srs = LingoSRS(uid)
        stats = srs.get_stats()
        level = _user_level(uid)
        allowed = _allowed_levels(uid)
        return {
            "level": level,
            "levels": JLPT_ORDER,
            "pool_size_at_level": spawn.pool_count_unlearned(uid, levels=allowed),
            "pool_size_total": spawn.pool_count(),
            "user_cards": stats.get("total_cards", 0),
            "reviews_today": stats.get("reviews_today", 0),
            "learned_today": stats.get("learned_today", 0),
            "review_session_size": spawn.REVIEW_SESSION_N,
            "learn_session_size": spawn.LEARN_SESSION_N,
        }
