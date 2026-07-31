"""
memory/entities.py

Phase B: lightweight entity tagging for personal memories.

Pure regex / heuristics — no LLM, no NER model. Runs on already-extracted
fact strings at write time so the hot path stays Jetson-friendly.
"""
from __future__ import annotations

import re
from typing import Iterable

# Multi-word Proper Case spans: "Hugging Face", "San Francisco"
_PROPER_SPAN_RE = re.compile(
    r"\b([A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+){0,4})\b"
)
# Short ALLCAPS tokens often used as project codes: GRACE, ROS2, API
_ALLCAPS_RE = re.compile(r"\b([A-Z]{2,12}[0-9]*)\b")
# Quoted names: "Max", 'Aiko'
_QUOTED_RE = re.compile(r"[\"']([^\"']{2,40})[\"']")
# Explicit name patterns: called X, named X, project X
_CALLED_RE = re.compile(
    r"\b(?:called|named|project|robot|dog|cat|company|team)\s+([A-Z][\w.-]{1,40})",
    re.IGNORECASE,
)

_STOP_ENTITIES = frozenset({
    "the", "a", "an", "and", "or", "but", "for", "with", "from", "into",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
    "today", "yesterday", "tomorrow", "user", "assistant", "aiko",
    "he", "she", "they", "his", "her", "their", "this", "that",
})

# Kind heuristics (keyword → kind). First match wins.
_KIND_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("identity", ("name is", "birthday", "lives in", "nationality", "age is", "is from ")),
    ("preference", ("likes", "loves", "hates", "dislikes", "prefers", "favorite", "favourite")),
    ("plan", ("deadline", "due ", "will ", "going to", "plans to", "wants to")),
    ("event", ("hackathon", "interview", "meeting", "appointment", "lost ", "joined")),
)


def _clean_entity(raw: str) -> str | None:
    s = (raw or "").strip(" .,;:!?()[]{}").strip()
    if len(s) < 2 or len(s) > 80:
        return None
    if s.casefold() in _STOP_ENTITIES:
        return None
    # Drop pure numbers
    if s.isdigit():
        return None
    return s


def extract_entities(text: str, *, max_entities: int = 12) -> list[str]:
    """Extract entity-like tokens from a memory fact string.

    Deterministic and cheap. Prefer precision over recall — empty is fine.
    """
    if not (text or "").strip():
        return []

    found: list[str] = []
    seen: set[str] = set()

    def _add(raw: str) -> None:
        ent = _clean_entity(raw)
        if not ent:
            return
        key = ent.casefold()
        if key in seen:
            return
        seen.add(key)
        found.append(ent)

    for m in _QUOTED_RE.finditer(text):
        _add(m.group(1))
    for m in _CALLED_RE.finditer(text):
        _add(m.group(1))
    for m in _PROPER_SPAN_RE.finditer(text):
        span = m.group(1)
        # Sentence-initial single word is usually grammar capitalization, not an entity.
        if m.start() == 0 and " " not in span:
            continue
        _add(span)
    for m in _ALLCAPS_RE.finditer(text):
        _add(m.group(1))

    return found[:max_entities]


def classify_kind(text: str, default: str = "fact") -> str:
    """Heuristic memory kind from fact text. No LLM."""
    low = (text or "").casefold()
    for kind, needles in _KIND_RULES:
        if any(n in low for n in needles):
            return kind
    return default


def entity_overlap_score(query: str, entities: Iterable[str]) -> float:
    """Return 0..1 fraction of entities mentioned in the query (casefold)."""
    ents = [e for e in entities if e]
    if not ents:
        return 0.0
    q = (query or "").casefold()
    hits = sum(1 for e in ents if e.casefold() in q)
    return hits / len(ents)
