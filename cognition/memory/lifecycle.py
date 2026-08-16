"""Lifecycle-domain pure helpers for memory consolidation (dream pass).

The dream()/cleanup() orchestration methods live on AikoMemorize in the memorize
engine; this module holds the tunables the dream pass consumes.
"""
from __future__ import annotations

import os
import re


DREAM_MERGE_THRESHOLD = float(os.getenv("DREAM_MERGE_THRESHOLD", 0.88))
WRITE_DEDUP_THRESHOLD = float(os.getenv("WRITE_DEDUP_THRESHOLD", 0.95))

# access_count boost applied to salient memories during dream pass.
DREAM_BOOST_AMOUNT = int(os.getenv("DREAM_BOOST_AMOUNT", 2))

# Phase 21: schema abstraction in the dream pass. A cluster of active memories
# sharing an entity with a dominant valence sign is abstracted into ONE
# generalized kind='schema' memory (a gist/schema). These knobs bound how many
# clusters per run and how strongly valenced the cluster must be to count.
DREAM_SCHEMA_ENABLED = os.getenv("DREAM_SCHEMA_ENABLED", "1").lower() in {"1", "true", "yes", "on"}
DREAM_SCHEMA_MIN_MEMBERS = int(os.getenv("DREAM_SCHEMA_MIN_MEMBERS", "3"))
DREAM_SCHEMA_MAX_CLUSTERS = int(os.getenv("DREAM_SCHEMA_MAX_CLUSTERS", "12"))
DREAM_SCHEMA_VALENCE_MAJORITY = float(os.getenv("DREAM_SCHEMA_VALENCE_MAJORITY", "0.6"))
MEMORY_WM_CAPACITY = max(1, int(os.getenv("MEMORY_WM_CAPACITY", "7")))

_SALIENCE_KEYWORDS = frozenset([
    "name", "called", "likes", "loves", "hates", "dislikes", "always", "never",
    "important", "remember", "favourite", "favorite", "birthday", "works",
    "lives", "studying", "job", "afraid", "dream", "goal",
    "deadline", "due", "appointment", "event", "hackathon", "wallet",
    "lost", "passport", "license", "meeting", "interview", "project",
])

_SALIENCE_RE = re.compile(
    r'\b(?:' + '|'.join(re.escape(k) for k in _SALIENCE_KEYWORDS) + r')\b',
    re.IGNORECASE,
)


__all__ = [
    "DREAM_BOOST_AMOUNT",
    "DREAM_MERGE_THRESHOLD",
    "DREAM_SCHEMA_ENABLED",
    "DREAM_SCHEMA_MAX_CLUSTERS",
    "DREAM_SCHEMA_MIN_MEMBERS",
    "DREAM_SCHEMA_VALENCE_MAJORITY",
    "MEMORY_WM_CAPACITY",
    "WRITE_DEDUP_THRESHOLD",
    "_SALIENCE_KEYWORDS",
    "_SALIENCE_RE",
]

