"""Learn / courses / grammar routes on the shared pool + per-user SRS.

Mutating routes require real auth (no owner fallback).
"""
from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import Depends, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from .srs import LingoSRS
from . import spawn
from .levels import get_user_level, set_user_level, JLPT_LEVELS, normalize_level
from . import courses as course_mod

log = logging.getLogger(__name__)


class LearnItem(BaseModel):
    pool_id: Optional[int] = None
    front: str = ""
    back: str = ""
    reading: str = ""
    kind: str = "kanji"
    level: str = "N5"


class LearnSession(BaseModel):
    items: List[LearnItem]
    level: str
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


class SetLevelRequest(BaseModel):
    level: str


class DeckMeta(BaseModel):
    id: str
    title: str
    level: str
    kind: str
    card_count: int
    lesson: Optional[int] = None


class DeckCard(BaseModel):
    front: str
    back: str
    reading: str = ""
    note: str = ""


class DeckDetail(BaseModel):
    id: str
    title: str
    level: str
    kind: str
    cards: List[DeckCard]


async def require_lingo_user(request: Request) -> dict:
    """Authenticated session only — no AIKO_USER_ID fallback."""
    from interface.webui import auth
    try:
        session = await auth.require_session(request)
        return await auth.require_accepted_session(session)
    except HTTPException:
        raise
    except Exception as e:
        log.warning("Lingo auth required failed: %s", e)
        raise HTTPException(status_code=401, detail="Authentication required")


def attach_learn_routes(router, get_lingo_session, award_xp=None):
    @router.get("/learn/new", response_model=LearnSession)
    async def learn_new(session: dict = Depends(get_lingo_session)):
        uid = session["user_id"]
        level = get_user_level(uid)
        if spawn.pool_count(level) < spawn.POOL_MIN:
            spawn.top_up_user_level_background(uid)
        items = spawn.pool_take_unlearned(uid, n=spawn.LEARN_SESSION_N, level=level)
        unlearned_total = spawn.pool_count_unlearned(uid, level=level)
        return LearnSession(
            items=[LearnItem(**it) for it in items],
            level=level,
            pending_in_pool=max(0, unlearned_total - len(items)),
        )

    @router.post("/learn/mark", response_model=MarkLearnedResponse)
    async def learn_mark(
        body: MarkLearnedRequest,
        session: dict = Depends(require_lingo_user),
    ):
        uid = session["user_id"]
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
        level = get_user_level(uid)
        srs = LingoSRS(uid)
        stats = srs.get_stats()
        return {
            "level": level,
            "levels": list(JLPT_LEVELS),
            "pool_size_at_level": spawn.pool_count(level),
            "pool_unlearned_at_level": spawn.pool_count_unlearned(uid, level=level),
            "pool_size_total": spawn.pool_count(),
            "user_cards": stats.get("total_cards", 0),
            "reviews_today": stats.get("reviews_today", 0),
            "learned_today": stats.get("learned_today", 0),
            "review_session_size": spawn.REVIEW_SESSION_N,
            "learn_session_size": spawn.LEARN_SESSION_N,
        }

    @router.post("/level")
    async def set_level(
        body: SetLevelRequest,
        session: dict = Depends(require_lingo_user),
    ):
        uid = session["user_id"]
        level = set_user_level(uid, body.level)
        return {"level": level}

    @router.get("/level")
    async def get_level(session: dict = Depends(get_lingo_session)):
        uid = session["user_id"]
        return {"level": get_user_level(uid), "levels": list(JLPT_LEVELS)}

    @router.get("/courses", response_model=List[DeckMeta])
    async def list_courses(session: dict = Depends(get_lingo_session), all_levels: bool = False):
        uid = session["user_id"]
        level = None if all_levels else get_user_level(uid)
        return [DeckMeta(**c) for c in course_mod.list_courses(level=level)]

    @router.get("/courses/{course_id}", response_model=DeckDetail)
    async def get_course(course_id: str, session: dict = Depends(get_lingo_session)):
        c = course_mod.get_course(course_id)
        if not c:
            raise HTTPException(404, f"Unknown course: {course_id}")
        return DeckDetail(
            id=c["id"], title=c["title"], level=c["level"], kind=c["kind"],
            cards=[DeckCard(**card) for card in c["cards"]],
        )

    @router.get("/grammar", response_model=List[DeckMeta])
    async def list_grammar(session: dict = Depends(get_lingo_session), all_levels: bool = False):
        uid = session["user_id"]
        level = None if all_levels else get_user_level(uid)
        return [DeckMeta(**g) for g in course_mod.list_grammar(level=level)]

    @router.get("/grammar/{grammar_id}", response_model=DeckDetail)
    async def get_grammar(grammar_id: str, session: dict = Depends(get_lingo_session)):
        g = course_mod.get_grammar(grammar_id)
        if not g:
            raise HTTPException(404, f"Unknown grammar deck: {grammar_id}")
        return DeckDetail(
            id=g["id"], title=g["title"], level=g["level"], kind=g["kind"],
            cards=[DeckCard(**card) for card in g["cards"]],
        )
