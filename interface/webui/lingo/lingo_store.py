"""
Lingo centralized storage.

Global (shared, versioned with code):
    interface/webui/lingo/materials.db
    - jlpt_cards / courses / grammar_decks, read-only at runtime.
    - One copy for all users, no per-user download.

Per-user (isolated, backed up per user):
    <USER_SPACE_ROOT>/<uid>/agentic/lingo.db
    - user_level / xp_ledger / streaks / srs_cards / review_logs
      / material_state / jlpt_progress
    - Resolved via system.userspace.user_state_path().
"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

MATERIALS_DB = Path(__file__).parent / "materials.db"
MATERIALS_VERSION = "2026.09.09-openjlpt-v2"
USER_DB_REL = "agentic/lingo.db"

LEGACY_LEVEL_MAP = {"beginner": "N5", "intermediate": "N3", "advanced": "N1"}
JLPT_ORDER = ["N5", "N4", "N3", "N2", "N1"]


def normalize_level(level: str | None) -> str:
    lvl = (level or "").strip()
    if lvl in JLPT_ORDER:
        return lvl
    return LEGACY_LEVEL_MAP.get(lvl, "N5")


def global_materials_db() -> Path:
    return MATERIALS_DB


def user_lingo_db_path(uid: str) -> Path:
    try:
        from system.userspace import user_state_path
        p = user_state_path(USER_DB_REL, uid)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    except Exception:
        p = Path.home() / ".aiko" / uid / USER_DB_REL
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path))
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")
    return con


def init_materials_db(seed: bool = True) -> Path:
    con = _connect(MATERIALS_DB)
    try:
        con.execute("BEGIN")
        con.execute("""CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY, value TEXT)""")
        con.execute("""CREATE TABLE IF NOT EXISTS jlpt_cards (
            id TEXT PRIMARY KEY, level TEXT NOT NULL, kind TEXT NOT NULL,
            front TEXT NOT NULL, reading TEXT NOT NULL DEFAULT '',
            back TEXT NOT NULL, note TEXT NOT NULL DEFAULT '',
            source TEXT NOT NULL DEFAULT 'seed')""")
        con.execute("""CREATE TABLE IF NOT EXISTS courses (
            id TEXT PRIMARY KEY, title TEXT NOT NULL,
            level TEXT NOT NULL DEFAULT 'N5', kind TEXT NOT NULL DEFAULT '',
            cards_json TEXT NOT NULL DEFAULT '[]')""")
        con.execute("""CREATE TABLE IF NOT EXISTS grammar_decks (
            id TEXT PRIMARY KEY, title TEXT NOT NULL,
            level TEXT NOT NULL DEFAULT 'N5', kind TEXT NOT NULL DEFAULT '',
            cards_json TEXT NOT NULL DEFAULT '[]')""")
        con.execute("""CREATE INDEX IF NOT EXISTS idx_jlpt_level_kind
            ON jlpt_cards(level, kind)""")
        con.execute("""CREATE TABLE IF NOT EXISTS vocab_pool (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            front TEXT NOT NULL, back TEXT NOT NULL,
            reading TEXT NOT NULL DEFAULT '', kind TEXT NOT NULL DEFAULT 'kanji',
            level TEXT NOT NULL DEFAULT 'N5', used_count INTEGER NOT NULL DEFAULT 0,
            created_at REAL, source TEXT NOT NULL DEFAULT 'spawn',
            UNIQUE(front, back))""")
        con.execute("""CREATE TABLE IF NOT EXISTS lesson_pool (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            front TEXT NOT NULL, back TEXT NOT NULL,
            reading TEXT NOT NULL DEFAULT '', level TEXT NOT NULL DEFAULT 'N5',
            used_count INTEGER NOT NULL DEFAULT 0, created_at REAL,
            UNIQUE(front, back))""")
        ver = con.execute("SELECT value FROM meta WHERE key='version'").fetchone()
        if seed and (ver is None or ver[0] != MATERIALS_VERSION):
            _seed_materials(con)
            _migrate_legacy_pools(con)
            con.execute(
                "INSERT OR REPLACE INTO meta(key,value) VALUES('version',?)",
                (MATERIALS_VERSION,),
            )
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()
    return MATERIALS_DB


def _seed_materials(con: sqlite3.Connection) -> None:
    """Seed OpenJLPT (verified) first; optional small curated fallback."""
    try:
        from .import_openjlpt import import_openjlpt_into
        stats = import_openjlpt_into(con)
        log.info("OpenJLPT seed: %s", stats)
        print(f"[lingo] OpenJLPT import: {stats}")
        if stats.get("vocab_pool", 0) > 0 and not stats.get("fetch_limited"):
            return
    except sqlite3.Error:
        raise
    except Exception:
        log.warning("OpenJLPT seed failed", exc_info=True)
        print("[lingo] OpenJLPT seed failed; trying curated packs")
    from .import_bank import import_curated_into
    stats = import_curated_into(con)
    log.info("curated fallback seed: %s", stats)
    print(f"[lingo] curated import: {stats}")


def materials_count() -> dict:
    init_materials_db(seed=False)
    con = sqlite3.connect(str(MATERIALS_DB))
    try:
        return {
            "jlpt_cards": con.execute("SELECT COUNT(*) FROM jlpt_cards").fetchone()[0],
            "courses": con.execute("SELECT COUNT(*) FROM courses").fetchone()[0],
            "grammar_decks": con.execute("SELECT COUNT(*) FROM grammar_decks").fetchone()[0],
            "vocab_pool": con.execute("SELECT COUNT(*) FROM vocab_pool").fetchone()[0],
            "lesson_pool": con.execute("SELECT COUNT(*) FROM lesson_pool").fetchone()[0],
        }
    finally:
        con.close()


def _migrate_legacy_pools(con: sqlite3.Connection) -> dict:
    moved = {"vocab_pool": 0, "lesson_pool": 0}
    here = Path(__file__).parent
    for src_name, table, has_kind in (
        ("vocab_pool.db", "vocab_pool", True),
        ("lesson_pool.db", "lesson_pool", False),
    ):
        src = here / src_name
        if not src.exists():
            continue
        try:
            old = sqlite3.connect(str(src))
            try:
                if has_kind:
                    rows = old.execute(
                        "SELECT front,back,reading,kind,level,used_count,created_at FROM vocab_pool"
                    ).fetchall()
                    for front, back, reading, kind, level, used, ts in rows:
                        con.execute(
                            "INSERT OR IGNORE INTO vocab_pool"
                            "(front,back,reading,kind,level,used_count,created_at)"
                            " VALUES(?,?,?,?,?,?,?)",
                            (front, back, reading or "", kind or "kanji",
                             normalize_level(level), used or 0, ts),
                        )
                        if con.total_changes:
                            moved["vocab_pool"] += 1
                else:
                    rows = old.execute(
                        "SELECT front,back,reading,level,used_count,created_at FROM lesson_pool"
                    ).fetchall()
                    for front, back, reading, level, used, ts in rows:
                        con.execute(
                            "INSERT OR IGNORE INTO lesson_pool"
                            "(front,back,reading,level,used_count,created_at)"
                            " VALUES(?,?,?,?,?,?)",
                            (front, back, reading or "", normalize_level(level),
                             used or 0, ts),
                        )
                        if con.total_changes:
                            moved["lesson_pool"] += 1
            finally:
                old.close()
        except Exception:
            continue
    return moved


def init_user_db(uid: str) -> Path:
    path = user_lingo_db_path(uid)
    con = _connect(path)
    try:
        con.execute("""CREATE TABLE IF NOT EXISTS user_level (
            id INTEGER PRIMARY KEY CHECK(id=1), level TEXT NOT NULL DEFAULT 'N5',
            updated_at TEXT)""")
        con.execute("""CREATE TABLE IF NOT EXISTS xp_ledger (
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL,
            amount INTEGER NOT NULL, reason TEXT NOT NULL DEFAULT '')""")
        con.execute("""CREATE TABLE IF NOT EXISTS streaks (
            id INTEGER PRIMARY KEY CHECK(id=1), last_date TEXT,
            current INTEGER NOT NULL DEFAULT 0, total INTEGER NOT NULL DEFAULT 0)""")
        con.execute("""CREATE TABLE IF NOT EXISTS srs_cards (
            id INTEGER PRIMARY KEY AUTOINCREMENT, material_id TEXT,
            kanji TEXT NOT NULL DEFAULT '', hiragana TEXT NOT NULL,
            romaji TEXT NOT NULL DEFAULT '', meaning TEXT NOT NULL,
            pos TEXT NOT NULL DEFAULT '', context TEXT NOT NULL DEFAULT '',
            interval INTEGER NOT NULL DEFAULT 0, ease_factor REAL NOT NULL DEFAULT 2.5,
            review_count INTEGER NOT NULL DEFAULT 0, consecutive_correct INTEGER NOT NULL DEFAULT 0,
            last_review TEXT, next_review TEXT, created_at TEXT NOT NULL,
            source_context TEXT NOT NULL DEFAULT '')""")
        con.execute("""CREATE TABLE IF NOT EXISTS review_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT, card_id INTEGER NOT NULL,
            grade INTEGER NOT NULL, review_date TEXT NOT NULL,
            before_interval INTEGER NOT NULL DEFAULT 0, after_interval INTEGER NOT NULL DEFAULT 0,
            before_ease REAL NOT NULL DEFAULT 2.5, after_ease REAL NOT NULL DEFAULT 2.5,
            response_time_ms INTEGER NOT NULL DEFAULT 0)""")
        con.execute("""CREATE TABLE IF NOT EXISTS material_state (
            material_id TEXT PRIMARY KEY, used_count INTEGER NOT NULL DEFAULT 0,
            learned_at TEXT)""")
        con.execute("""CREATE TABLE IF NOT EXISTS jlpt_progress (
            level TEXT PRIMARY KEY, pool_left INTEGER NOT NULL DEFAULT 0,
            due INTEGER NOT NULL DEFAULT 0, grammar_pass INTEGER NOT NULL DEFAULT 0,
            unlocked INTEGER NOT NULL DEFAULT 0)""")
        con.execute("""CREATE TABLE IF NOT EXISTS lesson_progress (
            track TEXT NOT NULL, level TEXT NOT NULL,
            current INTEGER NOT NULL DEFAULT 1,
            updated_at TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (track, level))""")
        con.execute("INSERT OR IGNORE INTO user_level(id,level) VALUES(1,'N5')")
        con.execute("INSERT OR IGNORE INTO streaks(id) VALUES(1)")
        con.commit()
    finally:
        con.close()
    return path


def migrate_user_legacy(uid: str) -> dict:
    summary = {"level": None, "xp": 0, "streak": None, "cards": 0}
    init_user_db(uid)
    return summary


# ---------------------------------------------------------------------------
# Lesson/test progression: one lesson at a time per (track, level).
# `current` is 1-based; current > lessons_total means every lesson passed and
# the level final test is unlocked.
# ---------------------------------------------------------------------------
VALID_TRACKS = ("vocab", "grammar")


def get_lesson_progress(uid: str, track: str, level: str) -> int:
    path = init_user_db(uid)
    con = sqlite3.connect(str(path))
    try:
        row = con.execute(
            "SELECT current FROM lesson_progress WHERE track=? AND level=?",
            (track, normalize_level(level)),
        ).fetchone()
        return max(1, int(row[0])) if row else 1
    finally:
        con.close()


def set_lesson_progress(uid: str, track: str, level: str, current: int) -> int:
    path = init_user_db(uid)
    con = sqlite3.connect(str(path))
    try:
        con.execute(
            "INSERT INTO lesson_progress(track, level, current, updated_at)"
            " VALUES(?,?,?,?)"
            " ON CONFLICT(track, level) DO UPDATE SET current=excluded.current,"
            " updated_at=excluded.updated_at",
            (track, normalize_level(level), max(1, int(current)),
             datetime.now(timezone.utc).isoformat()),
        )
        con.commit()
        return max(1, int(current))
    finally:
        con.close()
