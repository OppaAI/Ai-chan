"""Small, zero-I/O cognitive state for edge deployments.

This bounded per-user layer bridges working and long-term memory: it keeps
salient recent turns, open questions/tasks, and a coarse affect cue. It uses no LLM, embeddings, database, or worker thread on the hot path.

This is deliberately one bounded module rather than a collection of tiny
"emotion", "goals", and "working memory" stores. It is the fast internal
state layer between the current turn and long-term memory.
"""
from __future__ import annotations

import re
import threading
from collections import OrderedDict, deque
from dataclasses import dataclass

from .env import env_flag, env_int

EDGE_COGNITION_ENABLED = env_flag("EDGE_COGNITION_ENABLED", "1")
EDGE_COGNITION_MAX_TURNS = max(1, env_int("EDGE_COGNITION_MAX_TURNS", 7))
EDGE_COGNITION_MAX_CHARS = max(240, env_int("EDGE_COGNITION_MAX_CHARS", 1200))
EDGE_COGNITION_MAX_OPEN_LOOPS = max(1, env_int("EDGE_COGNITION_MAX_OPEN_LOOPS", 3))
EDGE_COGNITION_MAX_GOALS = max(1, env_int("EDGE_COGNITION_MAX_GOALS", 5))
EDGE_COGNITION_MAX_IDENTITIES = max(1, env_int("EDGE_COGNITION_MAX_IDENTITIES", 16))
_WORD_RE = re.compile(r"[a-z0-9_]{3,}")
_QUESTION_RE = re.compile(r"\?|\b(what|why|how|when|where|who|which|remember|can you|could you|need to|todo)\b", re.I)
_COMMITMENT_RE = re.compile(r"\b(i will|i'll|we will|we'll|need to|remember to|don't forget|next step|todo)\b", re.I)
_GOAL_RE = re.compile(r"\b(?:i want to|i need to|we need to|let\x27s|lets|goal is to|trying to|working on)\s+(.{3,180})", re.I)
_DONE_RE = re.compile(r"\b(done|finished|completed|fixed|solved|never mind|forget it)\b", re.I)
_UNCERTAIN_RE = re.compile(r"\b(i don\x27t know|not sure|unclear|maybe|might|probably|could be|i think)\b", re.I)
_ENERGY_LOW_RE = re.compile(r"\b(tired|exhausted|sleepy|drained|burned out|can\x27t focus|cannot focus)\b", re.I)
_ENERGY_HIGH_RE = re.compile(r"\b(excited|energized|motivated|let\x27s go|can\x27t wait)\b", re.I)
_STOP = {"the", "and", "that", "this", "with", "you", "are", "for", "have", "from", "about"}


@dataclass(slots=True)
class _Event:
    user: str
    assistant: str
    tokens: frozenset[str]


@dataclass(slots=True)
class _Goal:
    text: str
    progress: str = "active"


def _tokens(text: str) -> frozenset[str]:
    return frozenset(w for w in _WORD_RE.findall((text or "").lower()) if w not in _STOP)


def _affect(text: str) -> float:
    lower = (text or "").lower()
    pos = sum(lower.count(w) for w in ("love", "great", "thanks", "happy", "good", "excited"))
    neg = sum(lower.count(w) for w in ("hate", "sad", "angry", "bad", "frustrated", "worried"))
    return max(-1.0, min(1.0, (pos - neg) / max(1, pos + neg)))


class EdgeCognitiveState:
    """Bounded per-identity state; all operations are lock-protected."""

    def __init__(self) -> None:
        self._events: deque[_Event] = deque(maxlen=EDGE_COGNITION_MAX_TURNS)
        self._open_loops: deque[str] = deque(maxlen=EDGE_COGNITION_MAX_OPEN_LOOPS)
        self._affect = 0.0
        self._goals: deque[_Goal] = deque(maxlen=EDGE_COGNITION_MAX_GOALS)
        self._energy = 0.5
        self._uncertainty = 0.0
        self._attention = ""
        self._lock = threading.RLock()

    def record(self, user: str, assistant: str) -> None:
        if not EDGE_COGNITION_ENABLED:
            return
        user = " ".join((user or "").split())[:360]
        assistant = " ".join((assistant or "").split())[:360]
        if not user and not assistant:
            return
        with self._lock:
            self._events.append(_Event(user, assistant, _tokens(user + " " + assistant)))
            combined = user + " " + assistant
            self._affect = max(-1.0, min(1.0, self._affect * 0.7 + _affect(combined) * 0.3))
            self._energy = max(0.0, min(1.0, self._energy * 0.8 + self._energy_signal(combined) * 0.2))
            self._uncertainty = max(0.0, min(1.0, self._uncertainty * 0.7 + self._uncertainty_signal(combined) * 0.3))
            self._attention = self._attention_for(user)
            if _QUESTION_RE.search(user) or _COMMITMENT_RE.search(user):
                loop = user[:220]
                if loop and loop not in self._open_loops:
                    self._open_loops.appendleft(loop)
            self._capture_goal(user)
            if _DONE_RE.search(user):
                self._close_matching_goal(user)

    def clear(self) -> None:
        with self._lock:
            self._events.clear(); self._open_loops.clear(); self._goals.clear(); self._affect = 0.0; self._energy = 0.5; self._uncertainty = 0.0; self._attention = ""


    @staticmethod
    def _energy_signal(text: str) -> float:
        if _ENERGY_LOW_RE.search(text):
            return 0.2
        if _ENERGY_HIGH_RE.search(text):
            return 0.8
        return 0.5


    @staticmethod
    def _uncertainty_signal(text: str) -> float:
        return 0.8 if _UNCERTAIN_RE.search(text) else 0.0


    @staticmethod
    def _attention_for(user: str) -> str:
        words = _tokens(user)
        return " ".join(sorted(words, key=lambda w: (-len(w), w))[:6])

    def _capture_goal(self, user: str) -> None:
        match = _GOAL_RE.search(user)
        if not match: return
        text = " ".join(match.group(1).split()).rstrip(".!?")
        if not text: return
        self._goals = deque((g for g in self._goals if g.text.casefold() != text.casefold()), maxlen=EDGE_COGNITION_MAX_GOALS)
        self._goals.appendleft(_Goal(text=text))

    def _close_matching_goal(self, user: str) -> None:
        words = _tokens(user)
        if not words:
            return
        for goal in self._goals:
            goal_words = _tokens(goal.text)
            overlap = len(words & goal_words)
            if overlap >= 2 or (goal_words and overlap / len(goal_words) >= 0.6):
                goal.progress = "completed"



    @staticmethod
    def _bounded(text: str, limit: int) -> str:
        return text if len(text) <= limit else text[: max(0, limit - 1)].rstrip() + "…"

    def snapshot(self) -> dict:
        """Return a compact diagnostic snapshot without exposing mutable state."""
        with self._lock:
            mood = "positive" if self._affect > 0.2 else "negative" if self._affect < -0.2 else "neutral"
            return {"mood": mood, "affect": round(self._affect, 3), "energy": round(self._energy, 3), "uncertainty": round(self._uncertainty, 3), "attention": self._attention, "open_loops": list(self._open_loops), "goals": [g.text for g in self._goals if g.progress == "active"]}

    def context(self, query: str = "") -> str:
        if not EDGE_COGNITION_ENABLED:
            return ""
        query_tokens = _tokens(query)
        with self._lock:
            events, loops, affect = list(self._events), list(self._open_loops), self._affect
            goals = [g.text for g in self._goals if g.progress == "active"]
            energy, uncertainty, attention = self._energy, self._uncertainty, self._attention
        if not events and not loops and not goals:
            return ""
        ranked = sorted(enumerate(events), key=lambda p: (len(p[1].tokens & query_tokens) * 3 + p[0], p[0]), reverse=True)
        lines, used = [], 0
        for _, event in ranked:
            line = f"User: {event.user}\nAiko: {event.assistant}"
            if used + len(line) > EDGE_COGNITION_MAX_CHARS:
                continue
            lines.append(line); used += len(line)
        if loops:
            lines.append("Open loops: " + " | ".join(loops))
        if goals:
            lines.append("Active goals: " + " | ".join(goals))
        mood = "positive" if affect > 0.2 else "negative" if affect < -0.2 else "neutral"
        energy_label = "low" if energy < 0.35 else "high" if energy > 0.65 else "steady"
        uncertainty_label = "elevated" if uncertainty > 0.35 else "ordinary"
        lines.append(f"Internal state: mood={mood}; energy={energy_label}; uncertainty={uncertainty_label}")
        if attention: lines.append(f"Recent attention: {attention}")
        body = "\n\n".join(lines)
        return "<edge_cognitive_state>\n" + self._bounded(body, EDGE_COGNITION_MAX_CHARS) + "\n</edge_cognitive_state>"


_states: OrderedDict[str, EdgeCognitiveState] = OrderedDict()
_states_lock = threading.Lock()


def for_identity(identity: str) -> EdgeCognitiveState:
    with _states_lock:
        key = identity or "default"
        state = _states.pop(key, None) or EdgeCognitiveState()
        _states[key] = state
        while len(_states) > EDGE_COGNITION_MAX_IDENTITIES:
            _states.popitem(last=False)
        return state


__all__ = ["EdgeCognitiveState", "for_identity"]
