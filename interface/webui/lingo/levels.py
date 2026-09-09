"""JLPT level helpers for Lingo.

Official scale (easiest → hardest): N5, N4, N3, N2, N1.
Rough study targets (not official fixed lists):
  N5 ~800 words / ~100 kanji — basic phrases
  N4 ~1,500 words / ~300 kanji — elementary everyday
  N3 ~3,500 words / ~650 kanji — intermediate bridge
  N2 ~6,000 words / ~1,000 kanji — upper-intermediate / workplace
  N1 ~10,000+ words / ~2,000 kanji — advanced / abstract

User level is stored in data/levels/{uid}.json with key "level".
Legacy beginner/intermediate/advanced map onto N5/N3/N1.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

JLPT_LEVELS = ("N5", "N4", "N3", "N2", "N1")

_LEGACY_MAP = {
    "beginner": "N5",
    "elementary": "N4",
    "intermediate": "N3",
    "upper": "N2",
    "upper-intermediate": "N2",
    "advanced": "N1",
}


def normalize_level(level: Optional[str], default: str = "N5") -> str:
    if not level:
        return default
    s = str(level).strip().upper().replace(" ", "")
    if s in JLPT_LEVELS:
        return s
    mapped = _LEGACY_MAP.get(str(level).strip().lower())
    if mapped:
        return mapped
    # accept "n1" style already uppercased fail
    if s.startswith("N") and s[1:].isdigit():
        cand = f"N{s[1:]}"
        if cand in JLPT_LEVELS:
            return cand
    return default


def level_file(uid: str) -> Path:
    p = Path(f"data/levels/{uid}.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def get_user_level(uid: str, default: str = "N5") -> str:
    try:
        path = level_file(uid)
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            return normalize_level(data.get("level"), default=default)
    except Exception:
        log.warning("get_user_level failed for %s", uid, exc_info=True)
    return default


def set_user_level(uid: str, level: str) -> str:
    level = normalize_level(level)
    path = level_file(uid)
    path.write_text(json.dumps({"level": level, "scale": "JLPT"}, indent=2), encoding="utf-8")
    return level


def ensure_owner_n1() -> None:
    """Pin the box owner to N1 so Learn/Review filter at advanced level."""
    import os
    owner = (os.getenv("AIKO_USER_ID") or "OppaAI").strip() or "OppaAI"
    try:
        set_user_level(owner, "N1")
        log.info("Owner %s JLPT level set to N1", owner)
    except Exception:
        log.warning("ensure_owner_n1 failed", exc_info=True)
