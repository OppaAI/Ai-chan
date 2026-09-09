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
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

MATERIALS_DB = Path(__file__).parent / "materials.db"
MATERIALS_VERSION = "2026.09.01-n5seed"
USER_DB_REL = "agentic/lingo.db"


def global_materials_db() -> Path:
    return MATERIALS_DB


def user_lingo_db_path(uid: str) -> Path:
    try:
        from system.userspace import user_state_path
        p = user_state_path(USER_DB_REL, uid)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    except Exception:
        # Fallback for standalone tests: ~/.aiko/<uid>/agentic/lingo.db
        p = Path.home() / ".aiko" / uid / USER_DB_REL
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path))
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")
    return con


# ---------------------------------------------------------------- global ---
def init_materials_db(seed: bool = True) -> Path:
    con = _connect(MATERIALS_DB)
    try:
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
        ver = con.execute("SELECT value FROM meta WHERE key='version'").fetchone()
        if seed and (ver is None or ver[0] != MATERIALS_VERSION):
            _seed_materials(con)
            con.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('version',?)",
                        (MATERIALS_VERSION,))
        con.commit()
    finally:
        con.close()
    return MATERIALS_DB


def _seed_materials(con: sqlite3.Connection) -> None:
    """Minimal N5-start seed (original examples). Full 800+ lists append later."""
    kana = []
    for ch, roma in [("あ", "a"), ("い", "i"), ("う", "u"), ("え", "e"), ("お", "o"),
                     ("か", "ka"), ("き", "ki"), ("く", "ku"), ("け", "ke"), ("こ", "ko")]:
        kana.append((f"KANA-h-{ch}", "N5", "kana", ch, ch, roma, "hiragana", "seed"))
    starter = [
        ("N5-v-0001", "N5", "vocab", "ねこ", "ねこ", "cat", "daily noun", "seed"),
        ("N5-v-0002", "N5", "vocab", "水", "みず", "water", "daily noun", "seed"),
        ("N5-g-0001", "N5", "grammar", "〜ます", "", "polite present", "classroom sentences", "seed"),
        ("N4-v-0001", "N4", "vocab", "約束", "やくそく", "promise", "familiar topic", "seed"),
    ]
    con.executemany(
        "INSERT OR IGNORE INTO jlpt_cards(id,level,kind,front,reading,back,note,source)"
        " VALUES(?,?,?,?,?,?,?,?)", kana + starter)


def materials_count() -> dict:
    init_materials_db(seed=False)
    con = sqlite3.connect(str(MATERIALS_DB))
    try:
        return {
            "jlpt_cards": con.execute("SELECT COUNT(*) FROM jlpt_cards").fetchone()[0],
            "courses": con.execute("SELECT COUNT(*) FROM courses").fetchone()[0],
            "grammar_decks": con.execute("SELECT COUNT(*) FROM grammar_decks").fetchone()[0],
        }
    finally:
        con.close()


# --------------------------------------------------------------- per-user ---
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
            last_review TEXT, next_review TEXT, created_at TEXT NOT NULL, source_context TEXT NOT NULL DEFAULT '')""")
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
        con.execute("INSERT OR IGNORE INTO user_level(id,level) VALUES(1,'N5')")
        con.execute("INSERT OR IGNORE INTO streaks(id) VALUES(1)")
        con.commit()
    finally:
        con.close()
    return path


def migrate_user_legacy(uid: str) -> dict:
    """One-shot move: data/{levels,xp,streaks}/{uid}.json + lingo_vocab.db rows."""
    summary = {"level": None, "xp": 0, "streak": None, "cards": 0}
    path = init_user_db(uid)
    con = _connect(path)
    try:
        base = Path("data")
        for name in ("levels", "xp", "streaks"):
            f = base / name / f"{uid}.json"
            if not f.exists():
                # also try absolute backend data dir
                f = Path(__file__).parent.parent.parent.parent / "data" / name / f"{uid}.json"
            if f.exists():
                try:
                    data = json.loads(f.read_text())
                    if name == "levels" and data.get("level"):
                        lvl = data["level"]
                        # map old 3-track to JLPT start (N5 lowest)
                        lvl = {"beginner": "N5", "intermediate": "N3", "advanced": "N1"}.get(lvl, lvl)
                        con.execute("UPDATE user_level SET level=?,updated_at=? WHERE id=1",
                                    (lvl, datetime.now(timezone.utc).isoformat()))
                        summary["level"] = lvl
                    elif name == "xp":
                        xp = int(data.get("xp", 0))
                        if xp > 0:
                            con.execute("INSERT INTO xp_ledger(ts,amount,reason) VALUES(?,?,?)",
                                        (datetime.now(timezone.utc).isoformat(), xp, "legacy"))
                            summary["xp"] = xp
                    elif name == "streaks":
                        con.execute("UPDATE streaks SET last_date=?,current=?,total=? WHERE id=1",
                                    (data.get("last_date"), int(data.get("current_streak", 0)),
                                     int(data.get("total_sessions", 0))))
                        summary["streak"] = data
                except Exception:
                    continue
        # legacy SRS cards
        old_db = Path(__file__).parent / "lingo_vocab.db"
        if old_db.exists():
            try:
                old = sqlite3.connect(str(old_db))
                rows = old.execute(
                    "SELECT kanji,hiragana,romaji,meaning,pos,context,interval,ease_factor,"
                    "review_count,consecutive_correct,last_review,next_review,created_at,source_context"
                    " FROM lingo_vocab_cards WHERE user_id=?", (uid,)).fetchall()
                old.close()
                for r in rows:
                    con.execute(
                        "INSERT INTO srs_cards(kanji,hiragana,romaji,meaning,pos,context,interval,"
                        "ease_factor,review_count,consecutive_correct,last_review,next_review,"
                        "created_at,source_context) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)", r)
                summary["cards"] = len(rows)
            except Exception:
                pass
        con.commit()
    finally:
        con.close()
    return summary


__all__ = ["global_materials_db", "user_lingo_db_path", "init_materials_db",
           "init_user_db", "migrate_user_legacy", "materials_count", "MATERIALS_VERSION"]
