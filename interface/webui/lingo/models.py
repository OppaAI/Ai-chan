"""
Lingo data models (request/response schemas).

Pulled out of the old lingo.py into their own module per lingo_architecture.md
so router.py, srs.py, and vocab.py can each depend only on the schemas they
actually need, instead of everything living in one 600-line file.
"""
from typing import Optional, List
from pydantic import BaseModel, Field


class TranslateRequest(BaseModel):
    text: str = Field(..., max_length=500)


class StartRequest(BaseModel):
    # Pydantic v2 uses `pattern=`, not `regex=`. `regex=` raises a
    # TypeError/PydanticUserError at class-definition time on v2, which
    # would take the whole module (and router) down on import.
    level: str = Field(..., pattern="^(beginner|intermediate|advanced)$")


class DialogueHistoryEntry(BaseModel):
    speaker: str
    text: str


class RespondRequest(BaseModel):
    text: str = Field(..., max_length=500)
    history: Optional[List[DialogueHistoryEntry]] = []


class TranslationResult(BaseModel):
    register: str
    text: str
    audioUrl: Optional[str] = None


class TranslateResponse(BaseModel):
    translations: List[TranslationResult]


class Toast(BaseModel):
    type: str  # "success", "feedback", "info", "warning", "error"
    message: str


class ConversationResponse(BaseModel):
    japaneseText: str
    englishTranslation: str
    audioUrl: Optional[str] = None
    isFinished: bool = False
    isCorrect: bool = True
    feedback: Optional[str] = None
    suggestion: Optional[str] = None
    explanation: Optional[str] = None
    toast: Optional[Toast] = None


class ReviewCard(BaseModel):
    card_id: int
    hiragana: str
    meaning: str
    context: str


class ReviewSessionResponse(BaseModel):
    cards_due: int
    first_card: ReviewCard


class ReviewResponseRequest(BaseModel):
    card_id: int
    response: str
    grade: int = 0


class UpdatedCard(BaseModel):
    id: int
    interval: int
    ease: float


class ReviewResponseData(BaseModel):
    updated_card: UpdatedCard
    next_card: Optional[ReviewCard] = None
    cards_remaining: int
    # Optional "🌟 word = meaning" style toast surfaced on an EASY grade.
    # Was previously dropped on the floor by the Android client because
    # ReviewResponseData.kt didn't declare the field (lingo_audit.md #5).
    toast: Optional[Toast] = None


class StatsResponse(BaseModel):
    total_cards: int
    learned_today: int
    reviews_today: int
    avg_ease: float
    cards_due: int
    streak: dict
    xp: int
    level: int
    last_level: str
    # Cards with a high review-failure rate, for the dashboard's "Practice
    # These" widget. Defaults to [] so existing clients that don't know
    # about this field are unaffected.
    weak_vocab: List[ReviewCard] = []


class XPResponse(BaseModel):
    xp: int
    level: int
    next_level_xp: int
