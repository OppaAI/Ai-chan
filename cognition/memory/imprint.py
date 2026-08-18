"""Write-path pure helpers for memory fact extraction and persistence.

The stateful write orchestration (add/add_raw/_insert_row/_maybe_supersede_neighbor)
lives on _MemoryBackend in the memorize engine; this module holds the prompt and
stateless pre/post-processing helpers the write path uses.
"""
from __future__ import annotations

import os
import re


_EXTRACT_MIN_CHARS = int(os.getenv("MEMORY_EXTRACT_MIN_CHARS", 80))
_EXTRACT_MAX_TOKENS = int(os.getenv("MEMORY_EXTRACT_MAX_TOKENS", 512))
_EXTRACT_TIMEOUT = float(os.getenv("MEMORY_EXTRACT_TIMEOUT", 18))

_HEDGE_SIGNALS = frozenset([
    "might", "probably", "seems", "i think", "perhaps", "maybe",
    "appears", "possibly", "could be", "not sure", "i believe",
    "it sounds like", "it seems like",
])

_HEDGE_RE = re.compile(
    r'\b(?:' + '|'.join(re.escape(h) for h in _HEDGE_SIGNALS) + r')\b',
    re.IGNORECASE,
)

# Frozen relative-time / date-check facts that leak into stored memories
# ("Oppa today is Monday, August 10, 2026", "Aiko checks the date of July 3
# (day after tomorrow)", "[2026-08-11] OppaAI asks Aiko to verify the date
# of July 3"). When recalled these masquerade as the current date and
# contaminate Aiko's sense of "now" — the <current_datetime> block is the
# only authoritative clock. memorize.format_for_context drops such memories
# from context; util/scrub_stale_memory_dates.py deletes them from the DB.
_TEMPORAL_CONTAMINATION_RE = re.compile(
    r"(?:\b(?:today|tomorrow|yesterday|tonight|right now|currently|as of now|"
    r"next day|day after tomorrow|last (?:week|month|night))\s+"
    r"(?:is|was|will be|refers? to|means?|equals?)\b"
    r"|\b(?:as today|as of today|referred? to [^.\n]{0,40} as today)\b)",
    re.IGNORECASE,
)
_DATE_CHECK_RE = re.compile(
    r"\b(?:verif(?:y|ied|ying)|check(?:es|ed|s)?|confirm(?:ed|s)?|ask(?:ed|s)?|request(?:ed|s)?)\b"
    r"[^.\n]{0,60}\b(?:the\s+)?(?:date|day of)\b",
    re.IGNORECASE,
)

def _is_stale_temporal_fact(text: str) -> bool:
    """True when a stored memory is a frozen date/time fact whose wording
    would masquerade as the current date if injected into context."""
    return bool(_TEMPORAL_CONTAMINATION_RE.search(text) or _DATE_CHECK_RE.search(text))

def _force_subject_name(text: str, subject: str, user_name: str) -> str:
    t = (text or "").strip()
    if not t:
        return t
    name = "Aiko" if subject == "assistant" else user_name
    low = t.casefold()
    # strip a leading wrong/right name token once
    for prefix in (user_name, "Aiko", "User", "Assistant"):
        if prefix and low.startswith(prefix.casefold()):
            parts = t.split(None, 1)
            t = parts[1] if len(parts) > 1 else ""
            low = t.casefold()
            break
    if not t:
        return name
    if not t.casefold().startswith(name.casefold()):
        body = t[0].lower() + t[1:] if t else t
        t = f"{name} {body}"
    return t.strip()

def _valence_from_llm() -> bool:
    """Whether extract-provided valence_score overrides lexical inference.

    Read from the same env/yaml source as the other MEMORY_* keys. Defaults
    to ON (matches MEMORY_VALENCE_FROM_LLM: "1" in config/memory.yaml).
    """
    import os
    return os.getenv("MEMORY_VALENCE_FROM_LLM", "1").strip().lower() not in ("0", "false", "no", "off")

_EXTRACT_PROMPT = """\
Extract memorable facts from this conversation.
{user_name} is the user. Aiko is the assistant. Attribute each fact to the correct person.

Rules:
- Only include facts the speaker stated explicitly. Never infer or assume.
- Each line is prefixed with the speaker: "{user_name}:" or "Aiko:".
- NEVER misattribute: what Aiko says is a fact about Aiko ("Aiko ..."). What {user_name} says is a fact about {user_name} ("{user_name} ..."). Never turn one speaker's statement into a fact about the other.
- Prefer memorable facts about {user_name}. Also keep Aiko's own explicit statements about herself (identity, limits, preferences, plans) — written as "Aiko ...", third person.
- Write facts as short, direct, self-contained statements in third person.
- No uncertain language: never use might, probably, seems, maybe, perhaps, appears.
- NEVER store relative-time facts: no "today", "tomorrow", "yesterday", "day after tomorrow", "next day", "last week", "this week", "currently", "now". These go stale the moment they are stored and confuse later conversations. If a fact genuinely depends on a date, state the concrete calendar date instead.
- NEVER extract facts from a date/time-checking exchange ("what is today's date", "verify the date of X", "what day is the day after tomorrow", "check the date"). These are ephemeral lookups, not memorable facts. If the only substance of the turn is a date lookup, return: []
- If nothing is worth remembering, return: []

Return ONLY a JSON array of objects. No markdown. No explanation.
Each object:
{{"fact": "<third-person sentence>", "subject": "user"|"assistant", "valence_score": <int>}}

subject MUST match the speaker line:
- "{user_name}: ..." → "user" → fact starts with "{user_name}"
- "Aiko: ..." → "assistant" → fact starts with "Aiko"
- User giving the assistant rules ("follow my rules") → subject "assistant",
  e.g. "Aiko should follow {user_name}'s rules"
valence_score is -2..+2 (user feeling: -2 strong neg … 0 neutral/technical … +2 strong pos).
Use 0 when there is no clear emotion.

Speaker attribution (critical):
- 'Aiko: I am off limits to others' → {{"fact": "Aiko is off limits to others"}}. NEVER "{user_name} is off limits..." or "{user_name} says he is off limits...".
- 'Aiko: I dislike being treated as human-like' → {{"fact": "Aiko dislikes being treated as human-like"}}. NEVER "{user_name} dislikes being human-like".
- '{user_name}: follow my rules' / rules for the assistant → {{"fact": "Aiko should follow {user_name}'s rules"}}. NEVER "{user_name} needs to follow {user_name}'s rules".
- '{user_name}: I prefer dark mode' → {{"fact": "{user_name} prefers dark mode"}}.

Good examples:
[{{"fact": "{user_name} prefers dark mode", "subject": "user", "valence_score": 0}}, {{"fact": "{user_name} is building a robot called GRACE", "subject": "user", "valence_score": 1}}, {{"fact": "{user_name} joined the Hugging Face Hackathon", "subject": "user", "valence_score": 1}}, {{"fact": "{user_name} lost his wallet", "subject": "user", "valence_score": -2}}, {{"fact": "{user_name} has a deadline on Friday", "subject": "user", "valence_score": -1}}, {{"fact": "{user_name} dislikes mushrooms", "subject": "user", "valence_score": -1}}, {{"fact": "Aiko is off limits to others", "subject": "assistant", "valence_score": 0}}, {{"fact": "Aiko dislikes being treated as human-like", "subject": "assistant", "valence_score": -1}}, {{"fact": "Aiko should follow {user_name}'s rules", "subject": "assistant", "valence_score": 0}}]

Bad examples (do not produce these):
[{{"fact": "{user_name} might like cats", "subject": "user", "valence_score": 0}}, {{"fact": "It seems {user_name} is tired", "subject": "user", "valence_score": 0}}, {{"fact": "Aiko should remember this", "subject": "assistant", "valence_score": 0}}, {{"fact": "{user_name} says he is off limits to others", "subject": "user", "valence_score": 0}}, {{"fact": "{user_name} dislikes being human-like", "subject": "user", "valence_score": -1}}, {{"fact": "{user_name} needs to follow {user_name}'s rules", "subject": "user", "valence_score": 0}}, {{"fact": "{user_name} today is Monday, August 10, 2026", "subject": "user", "valence_score": 0}}, {{"fact": "{user_name}'s next day is July 1, 2026", "subject": "user", "valence_score": 0}}, {{"fact": "Aiko checks the date of July 3 (day after tomorrow)", "subject": "assistant", "valence_score": 0}}]

Conversation:
{conversation}"""


__all__ = [
    "_DATE_CHECK_RE",
    "_EXTRACT_MAX_TOKENS",
    "_EXTRACT_MIN_CHARS",
    "_EXTRACT_PROMPT",
    "_EXTRACT_TIMEOUT",
    "_HEDGE_RE",
    "_HEDGE_SIGNALS",
    "_TEMPORAL_CONTAMINATION_RE",
    "_force_subject_name",
    "_is_stale_temporal_fact",
    "_valence_from_llm",
]

