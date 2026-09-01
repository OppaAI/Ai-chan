"""Context formatting helpers for memory results.

Re-exported from ``cognition.memory.narrative`` to preserve the stable
import path after the merge. Also re-exports the cross-store related-
context fetchers/formatters so callers can keep using
``cognition.memory.format`` as a single import point.
"""
from __future__ import annotations

from .narrative import (
    format_for_context,
    persona_context,
    scene_context,
    fetch_related_for_memories,
    format_related_blocks,
    related_knowledge,
    related_experience,
)

__all__ = [
    "format_for_context",
    "persona_context",
    "scene_context",
    "fetch_related_for_memories",
    "format_related_blocks",
    "related_knowledge",
    "related_experience",
]