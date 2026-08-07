"""Search-domain pure helpers for memory recall.

Candidate computation (_sqlite_knn_search etc.) stays in the memorize engine;
this module holds the stateless lexical/trivial helpers recall calls.
"""
from __future__ import annotations

import os
import re


AI_NAME = os.getenv("AI_NAME", "Aiko").strip().lower()

_FILLER_WORDS = (
    "hi", "hey", "hello", "ok", "okay", "thanks", "thank you",
    "yes", "no", "yeah", "nah", "lol", "sure", "bye",
)

# Social/wellbeing phrases that carry no retrievable intent on their own —
# distinct from _FILLER_WORDS since these are multi-word and never a
# stand-alone ack.
_GREETING_PHRASES = (
    "how are you", "how are you doing", "hows it going", "how's it going",
    "how are things", "how you doing", "whats up", "what's up",
)

_name_alt = re.escape(AI_NAME) if AI_NAME else ""

# Combined trivial vocabulary: filler acks + greeting/wellbeing phrases.
# Sorted longest-first so e.g. "how are you doing" matches before the
# shorter "how are you" prefix inside the alternation.
_TRIVIAL_PHRASES = sorted(_FILLER_WORDS + _GREETING_PHRASES, key=len, reverse=True)
_trivial_alt = "|".join(re.escape(p) for p in _TRIVIAL_PHRASES)
_CLAUSE_SPLIT_RE = re.compile(r"[,.!?]+")

def _is_trivial_input(text: str) -> bool:
    """
    True when every clause of the message (split on , . ! ?) is filler,
    a greeting/wellbeing phrase, or the wake-word alone — i.e. no clause
    carries retrievable intent.

    Replaces the old single-anchor _TRIVIAL_INPUT_RE, which could only
    match one-or-two-token messages and had no concept of multi-word
    social phrases like "how are you doing". Splitting into clauses also
    handles ragged ASR transcripts like "Hi, I. How are you doing." —
    each clause is checked independently rather than requiring the whole
    string to match one rigid pattern.

    Any clause that doesn't fully match the trivial vocabulary (a real
    question, name, or request) makes the whole input non-trivial, so
    "hi aiko, what's the weather" still searches normally.
    """
    clauses = [c.strip().lower() for c in _CLAUSE_SPLIT_RE.split(text or "") if c.strip()]
    if not clauses:
        return True
    for clause in clauses:
        if _name_alt and re.fullmatch(_name_alt, clause, re.IGNORECASE):
            continue
        if re.fullmatch(_trivial_alt, clause, re.IGNORECASE):
            continue
        # stray ASR fragments (bare pronouns/fillers with no verb) carry no
        # retrievable intent on their own -- e.g. "I" from "Hi, I. How are..."
        if len(clause.split()) == 1 and len(clause) <= 2:
            continue
        return False
    return True

_BROAD_RECALL_RE = re.compile(
    r"\b(what|anything|things|facts|memories?|remember|recall)\b.*\b(about me|about oppa|you remember|past|before)\b"
    r"|\b(remember|recall)\b.*\b(me|oppa)\b",
    re.IGNORECASE,
)

def _sanitize_fts_query(query: str) -> str | None:
    """
    Strip characters that break FTS5 query parsing.
    FTS5 treats , " ( ) * ^ : - ' as syntax tokens — remove them all.
    Returns None when nothing usable remains (caller should skip the FTS5
    lookup entirely — a bare '*' is not a valid FTS5 "match everything"
    query and raises `sqlite3.OperationalError: unknown special query:`).
    """
    cleaned = re.sub(r'[^\w\s]', ' ', query or "")
    cleaned = ' '.join(cleaned.split())
    return cleaned or None

def _normalize_memory_text(text: str) -> str:
    """
    Normalize memory text for exact-duplicate comparison at recall time.
    Lowercased, whitespace-collapsed. Intentionally cheap/exact (not
    fuzzy) — recall-time dedup targets true copies (e.g. the same
    daily-record string inserted multiple times via add_raw), not
    semantic near-duplicates. Semantic near-duplicates are dream()'s job.
    """
    return " ".join((text or "").split()).lower()


__all__ = [
    "AI_NAME",
    "_BROAD_RECALL_RE",
    "_CLAUSE_SPLIT_RE",
    "_FILLER_WORDS",
    "_GREETING_PHRASES",
    "_TRIVIAL_PHRASES",
    "_is_trivial_input",
    "_name_alt",
    "_normalize_memory_text",
    "_sanitize_fts_query",
    "_trivial_alt",
]

