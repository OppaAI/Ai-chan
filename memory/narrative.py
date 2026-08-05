"""Phase 16 — supersession narrative helpers ("I used to think…")."""

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
