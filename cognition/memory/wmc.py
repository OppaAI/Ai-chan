"""cognition/memory/wmc.py - Working Memory Cortex (WMC) framework for Aiko.

Fast, capacity-limited active buffer for the *current conversational focus*.
Miller 7±2 soft slots + token-budget dual guard. All scoring is pure Python
(no embedder, no LLM, no disk on the hot path).

Lifecycle: Induction -> Filling -> Sustaining -> Receding -> Evicting
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable

# -- config helpers -----------------------------------------------------------

def _env_flag(name: str, default: str = "1") -> bool:
    return str(os.getenv(name, default)).strip().lower() not in ("0", "false", "no", "off", "")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


# -- public knobs -------------------------------------------------------------

WMC_ENABLED = _env_flag("WMC_ENABLED", "1")

WMC_MILLER_MIN = max(1, _env_int("WMC_MILLER_MIN", 5))
WMC_MILLER_MAX = max(WMC_MILLER_MIN, _env_int("WMC_MILLER_MAX", 9))
WMC_MILLER_CENTER = max(WMC_MILLER_MIN, min(WMC_MILLER_MAX, _env_int("WMC_MILLER_CENTER", 7)))
WMC_TOKEN_BUDGET = max(0, _env_int("WMC_TOKEN_BUDGET", 0))

# Scoring weights (renormalized at runtime). Defaults sum to 1.0.
WMC_W_EMOTION = _env_float("WMC_W_EMOTION", 0.15)
WMC_W_IMPORTANCE = _env_float("WMC_W_IMPORTANCE", 0.18)
WMC_W_RECENCY = _env_float("WMC_W_RECENCY", 0.15)
WMC_W_RELEVANCE = _env_float("WMC_W_RELEVANCE", 0.12)
WMC_W_NOVELTY = _env_float("WMC_W_NOVELTY", 0.12)
WMC_W_QUESTION = _env_float("WMC_W_QUESTION", 0.10)
WMC_W_ENTITY = _env_float("WMC_W_ENTITY", 0.08)
WMC_W_RECALL_FREQ = _env_float("WMC_W_RECALL_FREQ", 0.10)

WMC_RECENCY_HALF_LIFE = max(1.0, _env_float("WMC_RECENCY_HALF_LIFE", 4.0))
WMC_RECALL_FREQ_CAP = max(1, _env_int("WMC_RECALL_FREQ_CAP", 6))

_CHARS_PER_TOKEN = 4.0

# -- lexicons / patterns ------------------------------------------------------

_POS_WORDS = re.compile(
    r"\b(love|great|awesome|amazing|happy|glad|thanks|thank you|yay|nice|good|wonderful|excellent|perfect)\b",
    re.I,
)
_NEG_WORDS = re.compile(
    r"\b(hate|awful|terrible|sad|angry|mad|upset|annoyed|frustrated|bad|worst|sucks|furious)\b",
    re.I,
)
_IMPORTANCE_RE = re.compile(
    r"\b(remember|pin|important|always|never|prefer|favorite|favourite|my name|i am|i'm|birthday|"
    r"allergy|allergic|don't forget|do not forget|please note|key point|from now on)\b",
    re.I,
)
_QUESTION_RE = re.compile(
    r"\?|\b(what|why|how|when|where|who|which|can you|could you|would you|do you|is there)\b",
    re.I,
)
_NAME_LIKE_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")
_NUMBER_RE = re.compile(r"\b\d+(?:[.:]\d+)?\b")
_TOKEN_RE = re.compile(r"[a-z0-9_]{3,}")

_POS_EMOJI = set("😀😃😄😁🙂😊😍🥰❤️👍🎉✨🔥")
_NEG_EMOJI = set("😞😢😭😠😡💔👎😱")


# -- data model ---------------------------------------------------------------

@dataclass
class WMTurn:
    """One sustained turn-pair (user + assistant) in the active buffer."""

    user: str
    assistant: str
    tokens: int
    emotion: float = 0.0
    importance: float = 0.0
    relevance: float = 0.0
    novelty: float = 0.0
    question: float = 0.0
    entity: float = 0.0
    recall_count: int = 0
    created_turn: int = 0
    created_at: float = field(default_factory=time.time)
    score: float = 0.0
    _token_set: set[str] = field(default_factory=set, repr=False)

    def text(self) -> str:
        return f"User: {self.user}\nAssistant: {self.assistant}"

    def factor_breakdown(self) -> dict[str, float]:
        return {
            "emotion": (self.emotion + 1.0) * 0.5,
            "importance": self.importance,
            "recency": 0.0,
            "relevance": self.relevance,
            "novelty": self.novelty,
            "question": self.question,
            "entity": self.entity,
            "recall_freq": min(1.0, self.recall_count / max(1, WMC_RECALL_FREQ_CAP)),
            "score": self.score,
        }


# -- scoring primitives -------------------------------------------------------

def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, int(len(text) / _CHARS_PER_TOKEN))


def _content_tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def score_emotion(user: str, assistant: str) -> float:
    text = f"{user} {assistant}"
    pos = 1 if _POS_WORDS.search(text) else 0
    neg = 1 if _NEG_WORDS.search(text) else 0
    for ch in text:
        if ch in _POS_EMOJI:
            pos += 1
        elif ch in _NEG_EMOJI:
            neg += 1
    if pos == neg == 0:
        return 0.0
    return max(-1.0, min(1.0, (pos - neg) / max(1, pos + neg)))


def score_importance(user: str, assistant: str) -> float:
    text = f"{user}\n{assistant}"
    score = 0.0
    if _IMPORTANCE_RE.search(text):
        score += 0.45
    length = len(text)
    if length > 400:
        score += 0.15
    elif length > 150:
        score += 0.08
    return max(0.0, min(1.0, score))


def score_question(user: str) -> float:
    if not user:
        return 0.0
    hits = len(_QUESTION_RE.findall(user))
    return max(0.0, min(1.0, hits * 0.35))


def score_entity(user: str, assistant: str) -> float:
    text = f"{user} {assistant}"
    names = len(_NAME_LIKE_RE.findall(text))
    nums = len(_NUMBER_RE.findall(text))
    return max(0.0, min(1.0, names * 0.12 + nums * 0.08))


def score_recency(created_turn: int, current_turn: int, half_life: float = WMC_RECENCY_HALF_LIFE) -> float:
    age = max(0, current_turn - created_turn)
    if half_life <= 0:
        return 1.0 if age == 0 else 0.0
    return 0.5 ** (age / half_life)


def score_relevance_to_anchor(token_set: set[str], anchor_tokens: set[str] | None) -> float:
    if not anchor_tokens or not token_set:
        return 0.0
    overlap = len(token_set & anchor_tokens)
    return max(0.0, min(1.0, overlap / max(3, len(token_set) ** 0.5)))


def score_novelty(token_set: set[str], others: list[set[str]]) -> float:
    if not token_set or not others:
        return 1.0
    max_j = 0.0
    for o in others:
        if not o:
            continue
        inter = len(token_set & o)
        union = len(token_set | o)
        if union:
            max_j = max(max_j, inter / union)
    return max(0.0, min(1.0, 1.0 - max_j))


def score_recall_freq(recall_count: int, cap: int = WMC_RECALL_FREQ_CAP) -> float:
    return max(0.0, min(1.0, recall_count / max(1, cap)))


def compute_score(turn: "WMTurn", current_turn: int) -> float:
    emo = (turn.emotion + 1.0) * 0.5
    rec = score_recency(turn.created_turn, current_turn)
    rf = score_recall_freq(turn.recall_count)
    weights = {
        "emotion": WMC_W_EMOTION,
        "importance": WMC_W_IMPORTANCE,
        "recency": WMC_W_RECENCY,
        "relevance": WMC_W_RELEVANCE,
        "novelty": WMC_W_NOVELTY,
        "question": WMC_W_QUESTION,
        "entity": WMC_W_ENTITY,
        "recall_freq": WMC_W_RECALL_FREQ,
    }
    total_w = sum(weights.values())
    if total_w <= 0:
        return 0.0
    return (
        weights["emotion"] * emo
        + weights["importance"] * turn.importance
        + weights["recency"] * rec
        + weights["relevance"] * turn.relevance
        + weights["novelty"] * turn.novelty
        + weights["question"] * turn.question
        + weights["entity"] * turn.entity
        + weights["recall_freq"] * rf
    ) / total_w


# -- Working Memory Cortex ----------------------------------------------------

class WorkingMemoryCortex:
    """Capacity-limited active buffer of recent turn-pairs.

    Single-threaded by design (caller holds the turn lock).
    """

    def __init__(
        self,
        *,
        miller_min: int = WMC_MILLER_MIN,
        miller_max: int = WMC_MILLER_MAX,
        miller_center: int = WMC_MILLER_CENTER,
        token_budget: int = WMC_TOKEN_BUDGET,
        static_anchor_tokens: set[str] | None = None,
        on_evict: Callable[[WMTurn], None] | None = None,
    ) -> None:
        self.miller_min = max(1, miller_min)
        self.miller_max = max(self.miller_min, miller_max)
        self.miller_center = max(self.miller_min, min(self.miller_max, miller_center))
        self.token_budget = max(0, token_budget)
        self.static_anchor_tokens = static_anchor_tokens or set()
        self.on_evict = on_evict
        self._slots: list[WMTurn] = []
        self._turn_counter = 0

    @property
    def size(self) -> int:
        return len(self._slots)

    @property
    def total_tokens(self) -> int:
        return sum(t.tokens for t in self._slots)

    def clear(self) -> None:
        self._slots.clear()
        self._turn_counter = 0

    def set_static_anchor(self, tokens: Iterable[str]) -> None:
        self.static_anchor_tokens = {t.lower() for t in tokens if t}

    def fill(self, user: str, assistant: str, *, token_budget: int | None = None) -> list[WMTurn]:
        if not WMC_ENABLED:
            return []

        user = (user or "").strip()
        assistant = (assistant or "").strip()
        if not user and not assistant:
            return []

        self._turn_counter += 1
        text = f"User: {user}\nAssistant: {assistant}"
        tokens = estimate_tokens(text)
        token_set = _content_tokens(text)

        turn = WMTurn(
            user=user,
            assistant=assistant,
            tokens=tokens,
            emotion=score_emotion(user, assistant),
            importance=score_importance(user, assistant),
            question=score_question(user),
            entity=score_entity(user, assistant),
            relevance=score_relevance_to_anchor(token_set, self.static_anchor_tokens),
            created_turn=self._turn_counter,
            _token_set=token_set,
        )
        others = [s._token_set for s in self._slots]
        turn.novelty = score_novelty(token_set, others)
        turn.score = compute_score(turn, self._turn_counter)

        self._slots.append(turn)
        self._rescore()

        budget = self.token_budget if token_budget is None else max(0, token_budget)
        return self._enforce_capacity(budget)

    def touch(self, n: int | None = None) -> None:
        targets = self._slots if n is None else self._slots[: max(0, n)]
        for t in targets:
            t.recall_count += 1

    def get_context_block(self, *, max_tokens: int | None = None, touch: bool = True) -> str:
        if not self._slots:
            return ""

        lines: list[str] = []
        used = 0
        limit = max_tokens if max_tokens is not None else 0
        included: list[WMTurn] = []

        for t in self._slots:
            if limit and used + t.tokens > limit:
                break
            lines.append(t.text())
            used += t.tokens
            included.append(t)

        if touch:
            for t in included:
                t.recall_count += 1

        if not lines:
            return ""
        return (
            "<working_memory>\n"
            "Current conversational focus (most salient first):\n\n"
            + "\n\n".join(lines)
            + "\n</working_memory>"
        )

    def snapshot(self) -> list[WMTurn]:
        return list(self._slots)

    def studio_state(self) -> dict:
        return {
            "turn_counter": self._turn_counter,
            "size": self.size,
            "total_tokens": self.total_tokens,
            "miller": {
                "min": self.miller_min,
                "center": self.miller_center,
                "max": self.miller_max,
            },
            "token_budget": self.token_budget,
            "anchor_size": len(self.static_anchor_tokens),
            "slots": [
                {
                    "user": t.user[:120],
                    "assistant": t.assistant[:120],
                    "tokens": t.tokens,
                    "created_turn": t.created_turn,
                    "recall_count": t.recall_count,
                    "factors": {
                        **t.factor_breakdown(),
                        "recency": score_recency(t.created_turn, self._turn_counter),
                    },
                    "score": t.score,
                }
                for t in self._slots
            ],
        }

    def _rescore(self) -> None:
        for i, t in enumerate(self._slots):
            others = [s._token_set for j, s in enumerate(self._slots) if j != i]
            t.novelty = score_novelty(t._token_set, others)
            t.score = compute_score(t, self._turn_counter)
        self._slots.sort(key=lambda t: t.score, reverse=True)

    def _enforce_capacity(self, token_budget: int) -> list[WMTurn]:
        evicted: list[WMTurn] = []

        while len(self._slots) > self.miller_max:
            victim = self._slots.pop()
            evicted.append(victim)
            self._fire_evict(victim)

        while len(self._slots) > self.miller_center and self._should_trim_to_center():
            victim = self._slots.pop()
            evicted.append(victim)
            self._fire_evict(victim)

        if token_budget > 0:
            while self._slots and self.total_tokens > token_budget:
                victim = self._slots.pop()
                evicted.append(victim)
                self._fire_evict(victim)

        return evicted

    def _should_trim_to_center(self) -> bool:
        if not self._slots:
            return False
        return self._slots[-1].score < 0.35

    def _fire_evict(self, turn: WMTurn) -> None:
        if self.on_evict:
            try:
                self.on_evict(turn)
            except Exception:
                pass


def build_wmc(
    *,
    static_anchor_tokens: set[str] | None = None,
    on_evict: Callable[[WMTurn], None] | None = None,
) -> WorkingMemoryCortex:
    return WorkingMemoryCortex(
        miller_min=WMC_MILLER_MIN,
        miller_max=WMC_MILLER_MAX,
        miller_center=WMC_MILLER_CENTER,
        token_budget=WMC_TOKEN_BUDGET,
        static_anchor_tokens=static_anchor_tokens,
        on_evict=on_evict,
    )
