"""
memory/fact_identity.py

Fix user vs assistant mis-attribution in extracted memory facts.

- User display name (e.g. Oppa) vs assistant (Aiko)
- "I" in user messages → user name
- "I" in assistant messages → Aiko
- Cheap post-filters for known bad patterns after LLM extract
"""

from __future__ import annotations

import re
from typing import Iterable

DEFAULT_ASSISTANT_NAME = "Aiko"
DEFAULT_USER_NAME = "Oppa"

# Patterns: (compiled regex, replacement callable or string template)
def _user_re(name: str) -> str:
    return re.escape(name.strip())


def fix_fact_identity(
    text: str,
    *,
    user_name: str | None = None,
    assistant_name: str | None = None,
) -> str:
    """
    Apply cheap repairs for common Oppa/Aiko swaps.
    Does not invent new facts; returns text unchanged if no rule matches.
    """
    t = (text or "").strip()
    if not t:
        return t

    u = (user_name or DEFAULT_USER_NAME).strip() or DEFAULT_USER_NAME
    a = (assistant_name or DEFAULT_ASSISTANT_NAME).strip() or DEFAULT_ASSISTANT_NAME
    if u.casefold() == a.casefold():
        return t

    ue = _user_re(u)

    # 1) "{User} dislikes/hates being human-like" → Aiko (persona self-talk)
    t2 = re.sub(
        rf"^({ue})\s+(dislikes|doesn't like|does not like|hates|dislike)\s+"
        rf"(to be |being )?(portrayed as )?human[- ]?like\b",
        lambda m: f"{a} {m.group(2)} being human-like",
        t,
        count=1,
        flags=re.IGNORECASE,
    )
    if t2 != t:
        t = t2

    # 2) "{User} needs/must/should follow {User}'s rules" → Aiko follows user's rules
    t2 = re.sub(
        rf"^{ue}\s+(needs to|must|should|has to)\s+follow\s+{ue}'s\s+rules\b",
        lambda m: f"{a} should follow {u}'s rules",
        t,
        count=1,
        flags=re.IGNORECASE,
    )
    if t2 != t:
        t = t2

    # 3) "{User} must/should obey {User}'s instructions" (same idea)
    t2 = re.sub(
        rf"^{ue}\s+(needs to|must|should|has to)\s+(obey|follow)\s+{ue}'s\s+"
        rf"(instructions|commands|orders)\b",
        lambda m: f"{a} should follow {u}'s {m.group(3)}",
        t,
        count=1,
        flags=re.IGNORECASE,
    )
    if t2 != t:
        t = t2

    # 4) "{User} is an AI / assistant / language model" → Aiko
    t2 = re.sub(
        rf"^{ue}\s+(is|as)\s+(an?\s+)?(AI|assistant|language model|LLM)\b",
        lambda m: f"{a} is {m.group(2) or ''}{m.group(3)}",
        t,
        count=1,
        flags=re.IGNORECASE,
    )
    if t2 != t:
        t = t2

    return t.strip()


def should_skip_misattributed_fact(
    text: str,
    *,
    user_name: str | None = None,
    assistant_name: str | None = None,
) -> bool:
    """
    True if fact still looks like assistant-persona pinned on the user
    after fix_fact_identity — safer to skip than store wrong.
    """
    t = (text or "").strip()
    if not t:
        return True
    u = (user_name or DEFAULT_USER_NAME).strip() or DEFAULT_USER_NAME
    ue = _user_re(u)
    # User-name subject + strong assistant-only cues
    if re.match(
        rf"^{ue}\s+.*\b(human[- ]?like|as an AI|language model|my persona)\b",
        t,
        re.IGNORECASE,
    ):
        return True
    return False


def sanitize_extracted_facts(
    facts: Iterable[str],
    *,
    user_name: str | None = None,
    assistant_name: str | None = None,
) -> list[str]:
    """Fix identity then drop remaining misattributed persona facts."""
    out: list[str] = []
    u = user_name or DEFAULT_USER_NAME
    a = assistant_name or DEFAULT_ASSISTANT_NAME
    for raw in facts:
        if not isinstance(raw, str):
            continue
        fixed = fix_fact_identity(raw, user_name=u, assistant_name=a)
        if should_skip_misattributed_fact(fixed, user_name=u, assistant_name=a):
            continue
        if fixed:
            out.append(fixed)
    return out


def sanitize_fact_score_pairs(
    pairs: Iterable[tuple[str, int | None]],
    *,
    user_name: str | None = None,
    assistant_name: str | None = None,
) -> list[tuple[str, int | None]]:
    """Same as sanitize_extracted_facts for (text, valence_score) pairs."""
    out: list[tuple[str, int | None]] = []
    u = user_name or DEFAULT_USER_NAME
    a = assistant_name or DEFAULT_ASSISTANT_NAME
    for item in pairs:
        if not item:
            continue
        raw, score = item[0], item[1] if len(item) > 1 else None
        if not isinstance(raw, str):
            continue
        fixed = fix_fact_identity(raw, user_name=u, assistant_name=a)
        if should_skip_misattributed_fact(fixed, user_name=u, assistant_name=a):
            continue
        if fixed:
            out.append((fixed, score))
    return out


# --- Prompt fragment (paste into extract system prompt) ---

IDENTITY_PROMPT_RULES = """
IDENTITY (critical — never get this wrong):
- The USER is named {user_name}. The ASSISTANT is {assistant_name}.
- "I/me/my" in a USER message → fact subject is {user_name}.
- "I/me/my" in an ASSISTANT message → fact subject is {assistant_name}.
- Do NOT write {assistant_name}'s preferences, personality, or self-rules as {user_name}'s.
- Do NOT write {user_name}'s preferences as {assistant_name}'s.
- Assistant talking about being human-like, persona, or "my rules" → subject is {assistant_name}.
- User commands/rules for the assistant → "{assistant_name} should follow {user_name}'s rules"
  NOT "{user_name} needs to follow {user_name}'s rules".

Examples:
Good: "{assistant_name} dislikes being portrayed as human-like."
Bad:  "{user_name} dislikes being portrayed as human-like." (when the assistant said it about herself)
Good: "{assistant_name} should follow {user_name}'s rules."
Bad:  "{user_name} needs to follow {user_name}'s rules."
Good: "{user_name} prefers dark mode."
Bad:  "{assistant_name} prefers dark mode." (when the user said it)
""".strip()


def format_identity_prompt_rules(
    *,
    user_name: str | None = None,
    assistant_name: str | None = None,
) -> str:
    u = (user_name or DEFAULT_USER_NAME).strip() or DEFAULT_USER_NAME
    a = (assistant_name or DEFAULT_ASSISTANT_NAME).strip() or DEFAULT_ASSISTANT_NAME
    return IDENTITY_PROMPT_RULES.format(user_name=u, assistant_name=a)
