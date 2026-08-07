"""Context formatting helpers for memory results.

Re-exported from the memory engine hub (memorize) to preserve the stable
import path.
"""

from __future__ import annotations

from .memorize import AikoMemorize

format_for_context = AikoMemorize.format_for_context
scene_context = AikoMemorize.scene_context
persona_context = AikoMemorize.persona_context

__all__ = ["format_for_context", "persona_context", "scene_context"]