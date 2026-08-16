"""Small, zero-I/O cognitive state for edge deployments.

This bounded per-user layer bridges working and long-term memory: it keeps
salient recent turns, open questions/tasks, and a coarse affect cue. It uses
no LLM, embeddings, database, or worker thread on the hot path.
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
EDGE_COGNITION_MAX_IDENTITIES = max(1, env_int("EDGE_COGNITION_MAX_IDENTITIES", 16))
_WORD_RE = re.compile(r"[a-z0-9_]{3,}")
_QUESTION_RE = re.compile(r"\?|\b(what|why|how|when|where|who|which|can you|could you|need to|todo)\b", re.I)
_COMMITMENT_RE = re.compile(r"\b(i will|i'll|we will|we'll|need to|remember to|don't forget|next step|todo)\b", re.I)
_STOP = {"the", "and", "that", "this", "with", "you", "are", "for", "have", "from", "about"}


@dataclass(slots=True)
class _Event:
    user: str
    assistant: str
    tokens: frozenset[str]


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
            self._affect = max(-1.0, min(1.0, self._affect * 0.7 + _affect(user + " " + assistant) * 0.3))
            if _QUESTION_RE.search(user) or _COMMITMENT_RE.search(user):
                loop = user[:220]
                if loop and loop not in self._open_loops:
                    self._open_loops.appendleft(loop)

    def clear(self) -> None:
        with self._lock:
            self._events.clear(); self._open_loops.clear(); self._affect = 0.0

    def context(self, query: str = "") -> str:
        if not EDGE_COGNITION_ENABLED:
            return ""
        query_tokens = _tokens(query)
        with self._lock:
            events, loops, affect = list(self._events), list(self._open_loops), self._affect
        if not events and not loops:
            return ""
        ranked = sorted(enumerate(events), key=lambda p: (len(p[1].tokens & query_tokens) * 3 + p[0], p[0]), reverse=True)
        lines, used = [], 0
        for _, event in ranked:
            line = f"User: {event.user}\nAiko: {event.assistant}"
            if used + len(line) > EDGE_COGNITION_MAX_CHARS:
                continue
            lines.append(line); used += len(line)
        if loops:
            lines.append("Open loops: " + " | ".join(loops)[:max(0, EDGE_COGNITION_MAX_CHARS - used - 14)])
        mood = "positive" if affect > 0.2 else "negative" if affect < -0.2 else "neutral"
        lines.append(f"Affect cue: {mood}")
        return "<edge_cognitive_state>\n" + "\n\n".join(lines) + "\n</edge_cognitive_state>"


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
