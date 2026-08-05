"""Context formatting helpers for memory results."""

from __future__ import annotations

from .backend import AikoMemorize

format_for_context = AikoMemorize.format_for_context
scene_context = AikoMemorize.scene_context
persona_context = AikoMemorize.persona_context

__all__ = ["format_for_context", "persona_context", "scene_context"]
