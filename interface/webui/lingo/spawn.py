"""
Shared Lingo vocab spawn pool + hourly scheduler handler.

  * Content: interface/webui/lingo/vocab_pool.db (shared, JLPT-tagged)
  * Progress: USER_SPACE_ROOT/<uid>/agentic/lingo/vocab.db (per-user SRS)

Hourly: 2 items × N5..N1 = 10. Learn only shows unlearnt items at user level.
Includes integrity fixes: pool_id-only mark, unlearned counts, top-up inflight guard.
"""
from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, List, Optional

from .levels import JLPT_LEVELS, get_user_level, normalize_level

log = logging.getLogger(__name__)

POOL_DB = Path(__file__).parent / "vocab_pool.db"
SPAWN_PER_LEVEL = int(os.getenv("LINGO_SPAWN_PER_LEVEL", "2"))
POOL_MIN = int(os.getenv("LINGO_POOL_MIN", "20"))
LEARN_SESSION_N = 10
REVIEW_SESSION_N = 10

ITEM_KINDS = ("hiragana", "katakana", "kanji", "phrase", "sentence")

_pool_lock = threading.Lock()
_topup_inflight: set[str] = set()
_inflight_lock = threading.Lock()


def _conn() -> sqlite3.Connection:
    POOL_DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(POOL_DB), timeout=30)
    con.execute(
        """CREATE TABLE IF NOT EXISTS vocab_pool (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            front TEXT NOT NULL,
            back TEXT NOT NULL,
            reading TEXT NOT NULL DEFAULT '',
            kind TEXT NOT NULL DEFAULT 'kanji',
            level TEXT NOT NULL DEFAULT 'N5',
            used_count INTEGER NOT NULL DEFAULT 0,
            created_at REAL,
            UNIQUE(front, back)
        )"""
    )
    return con


def pool_get(pool_id: int) -> Optional[dict]:
    con = _conn()
    try:
        row = con.execute(
            "SELECT id, front, back, reading, kind, level FROM vocab_pool WHERE id = ?",
            (int(pool_id),),
        ).fetchone()
        if not row:
            return None
        return {
            "pool_id": row[0],
            "front": row[1],
            "back": row[2],
            "reading": row[3],
            "kind": row[4],
            "level": row[5],
        }
    finally:
        con.close()


def pool_count(level: Optional[str] = None) -> int:
    con = _conn()
    try:
        if level:
            level = normalize_level(level)
            return con.execute(
                "SELECT COUNT(*) FROM vocab_pool WHERE level = ?", (level,)
            ).fetchone()[0]
        return con.execute("SELECT COUNT(*) FROM vocab_pool").fetchone()[0]
    finally:
        con.close()


def _learned_keys(uid: str) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    try:
        from .srs import LingoSRS
        srs = LingoSRS(uid)
        con_u = sqlite3.connect(str(srs.db_path))
        try:
            rows = con_u.execute(
                "SELECT hiragana, meaning FROM lingo_vocab_cards WHERE user_id = ?",
                (uid,),
            ).fetchall()
            keys = {(r[0] or "", r[1] or "") for r in rows}
        finally:
            con_u.close()
    except Exception:
        log.warning("Could not load user learnt set for %s", uid, exp_info=True)
    return keys
