"""Lifecycle, cleanup, and dream-pruning helpers for memory stores."""

from __future__ import annotations

from .backend import AikoMemorize, vacuum_memory_db

touch_memories = AikoMemorize._touch_memories
dream = AikoMemorize.dream
dream_boost = AikoMemorize._dream_boost
dream_merge = AikoMemorize._dream_merge
cleanup = AikoMemorize.cleanup
optimize = AikoMemorize.optimize

__all__ = [
    "cleanup",
    "dream",
    "dream_boost",
    "dream_merge",
    "optimize",
    "touch_memories",
    "vacuum_memory_db",
]
