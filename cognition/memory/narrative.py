"""Phase 16 — supersession narrative helpers ("I used to think…").

Also hosts context-formatting helpers that were previously in format.py,
so that the stable import path ``cognition.memory.format`` continues to work
while the implementation lives here.
"""

from __future__ import annotations

from typing import Any


def _text(row: dict[str, Any] | Any) -> str:
    if isinstance(row, dict):
        return (row.get("memory") or row.get("text") or "").strip()
    return (getattr(row, "memory", None) or getattr(row, "text", None) or "").strip()


def format_supersession_narrative(
    chain: list[dict[str, Any]],
    *,
    max_chars: int = 220,
) -> str | None:
    """
    chain: oldest → newest memory rows (length >= 2).
    Returns a short "Previously held / Current" line, or None.
    """
    if not chain or len(chain) < 2:
        return None
    old = _text(chain[0])[:120]
    new = _text(chain[-1])[:120]
    if not old or not new or old == new:
        return None
    line = f'Previously held: "{old}". Current: "{new}".'
    if len(line) > max_chars:
        line = line[: max_chars - 1] + "…"
    return line


def query_wants_emotion(query: str) -> bool:
    """True when neg-avoid should be relaxed (emotional / reflective query)."""
    q = (query or "").lower()
    keys = (
        "feel",
        "felt",
        "upset",
        "happy",
        "sad",
        "angry",
        "love",
        "hate",
        "used to",
        "remember when",
        "why did i",
        "emotion",
        "mood",
        "i used to think",
        "changed my mind",
    )
    return any(k in q for k in keys)


# ── context-formatting helpers (moved from format.py) ────────────────────

def format_for_context(
    facts: list[dict[str, str]],
    *,
    max_chars: int = 1200,
    max_facts: int | None = None,
) -> str:
    """
    Format a list of memory facts into a context string injected into the
    query context window.  Each fact dict should have ``"memory"`` or
    ``"text"`` keys.

    Returns the formatted string, trimmed to ``max_chars`` characters.
    """
    parts: list[str] = []
    for i, fact in enumerate(facts):
        text = (fact.get("memory") or fact.get("text") or "").strip()
        if not text:
            continue
        parts.append(text)
        if max_facts is not None and len(parts) >= max_facts:
            break
    result = " ".join(parts)
    return result[:max_chars]


def scene_context(
    limit: int = 5, user_id: str | None = None
) -> str | None:
    """
    Format a scene-level context summary for the current user, if one exists.
    Returns ``None`` when no user is set or no scene memories are available.
    """
    from cognition.memory.memorize import AikoMemorize
    memorize = AikoMemorize()
    if user_id is None:
        user_id = memorize.get_user_id()
    if not user_id:
        return None
    try:
        ctx = memorize.scene_context(limit=limit, user_id=user_id)
        return ctx
    except Exception:
        return None


def persona_context(user_id: str | None = None) -> str | None:
    """
    Format a persona-level context summary for the current user, if one exists.
    Returns ``None`` when no user is set or no persona facts are available.
    """
    from cognition.memory.memorize import AikoMemorize
    memorize = AikoMemorize()
    if user_id is None:
        user_id = memorize.get_user_id()
    if not user_id:
        return None
    try:
        ctx = memorize.persona_context(user_id=user_id)
        return ctx
    except Exception:
        return None
