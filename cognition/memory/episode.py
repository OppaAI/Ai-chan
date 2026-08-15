"""
cognition/memory/episode.py

Episodic Memory Cortex (EMC) — true episodic store for Aiko.

Stores first-person, time-stamped episodes that come from working-memory
eviction. Separate from the existing memories / knowledge / experience tables.

Design rules:
  - Same per-user SQLite file as the rest of memory (Option A).
  - Same sqlite-vec + FTS5 technology.
  - Missing human-EM fields stay NULL / empty — never invent values.

EMC-1: storage + bind/flush API
EMC-2: turn ingest + buffer drain (eviction path)
EMC-6: coherent episode formation — group related staging rows on flush

Public surface:
    from cognition.memory.episode import EpisodicStore, ensure_episode_schema
"""
from __future__ import annotations

import json
import os
import sqlite3
import struct
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from system.log import get_logger
from system.userspace import current_user_id
from cognition.memory.vecstore import HarrierEmbedder, initialize_store_db
from cognition.memory.env import env_bool, env_int, env_float
from cognition.memory.episode_group import (
    EMC_GROUP_ENABLED,
    group_staging_rows as _group_staging_rows,
    merge_staging_group as _merge_staging_group,
)

log = get_logger(__name__)

# ── tunables (also mirrored in config/memory.yaml) ────────────────────────────

EMC_ENABLED = env_bool("EMC_ENABLED", "1")
EMC_EMBED_ON_FLUSH = env_bool("EMC_EMBED_ON_FLUSH", "1")
EMC_FLUSH_BATCH = max(1, env_int("EMC_FLUSH_BATCH", 32))
EMC_STAGING_MAX = max(10, env_int("EMC_STAGING_MAX", 200))

# EMC-2: eviction / buffer drain
EMC_EVICT_ENABLED = env_bool("EMC_EVICT_ENABLED", "1")
EMC_EVICT_MIN_CHARS = max(1, env_int("EMC_EVICT_MIN_CHARS", 40))
EMC_FLUSH_EVERY_TURNS = max(0, env_int("EMC_FLUSH_EVERY_TURNS", 8))
EMC_FLUSH_ON_STAGING = max(1, env_int("EMC_FLUSH_ON_STAGING", 24))

EMBED_DIMS = int(os.getenv("EMBED_DIMS", "640"))
