"""Context formatting helpers for memory results.

Re-exported from ``cognition.memory.narrative`` to preserve the stable
import path after the merge.
"""

from __future__ import annotations

from .narrative import (
    format_for_context,
    persona_context,
    scene_context,
)

__all__ = ["format_for_context", "persona_context", "scene_context"]