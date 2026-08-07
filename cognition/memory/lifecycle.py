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
    "WRITE_DEDUP_THRESHOLD",
    "_SALIENCE_KEYWORDS",
    "_SALIENCE_RE",
]

