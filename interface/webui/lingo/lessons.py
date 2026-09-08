"""
Lingo lesson decks (static curated content for Learn mode).

Unlike SRS cards (auto-extracted from conversation into the per-user DB),
these decks are fixed study material: kana charts, starter words, daily
phrases, and N5 kanji. Served read-only via GET /lessons and
GET /lessons/{deck_id} — no SRS writes, no auth side effects.
"""
from typing import Dict, List

# Each card: (front, back, reading). Reading feeds the app's speak button.
_DECKS: Dict[str, dict] = {}


def _kana_deck(deck_id: str, title: str, subtitle: str, rows: List[tuple]) -> None:
    cards = [{"front": kana, "back": roma, "reading": kana} for kana, roma in rows]
    _DECKS[deck_id] = {
        "id": deck_id, "title": title, "subtitle": subtitle,
        "kind": "kana", "cards": cards,
    }


_kana_deck("hiragana", "Hiragana", "46 basic characters", [
    ("あ", "a"), ("い", "i"), ("う", "u"), ("え", "e"), ("お", "o"),
    ("か", "ka"), ("き", "ki"), ("く", "ku"), ("け", "ke"), ("こ", "ko"),
    ("さ", "sa"), ("し", "shi"), ("す", "su"), ("せ", "se"), ("そ", "so"),
    ("た", "ta"), ("ち", "chi"), ("つ", "tsu"), ("て", "te"), ("と", "to"),
    ("な", "na"), ("に", "ni"), ("ぬ", "nu"), ("ね", "ne"), ("の", "no"),
    ("は", "ha"), ("ひ", "hi"), ("ふ", "fu"), ("へ", "he"), ("ほ", "ho"),
    ("ま", "ma"), ("み", "mi"), ("む", "mu"), ("め", "me"), ("も", "mo"),
    ("や", "ya"), ("ゆ", "yu"), ("よ", "yo"),
    ("ら", "ra"), ("り", "ri"), ("る", "ru"), ("れ", "re"), ("ろ", "ro"),
    ("わ", "wa"), ("を", "wo"), ("ん", "n"),
])

_kana_deck("katakana", "Katakana", "46 basic characters", [
    ("ア", "a"), ("イ", "i"), ("ウ", "u"), ("エ", "e"), ("オ", "o"),
    ("カ", "ka"), ("キ", "ki"), ("ク", "ku"), ("ケ", "ke"), ("コ", "ko"),
    ("サ", "sa"), ("シ", "shi"), ("ス", "su"), ("セ", "se"), ("ソ", "so"),
    ("タ", "ta"), ("チ", "chi"), ("ツ", "tsu"), ("テ", "te"), ("ト", "to"),
    ("ナ", "na"), ("ニ", "ni"), ("ヌ", "nu"), ("ネ", "ne"), ("ノ", "no"),
    ("ハ", "ha"), ("ヒ", "hi"), ("フ", "fu"), ("ヘ", "he"), ("ホ", "ho"),
    ("マ", "ma"), ("ミ", "mi"), ("ム", "mu"), ("メ", "me"), ("モ", "mo"),
    ("ヤ", "ya"), ("ユ", "yu"), ("ヨ", "yo"),
    ("ラ", "ra"), ("リ", "ri"), ("ル", "ru"), ("レ", "re"), ("ロ", "ro"),
    ("ワ", "wa"), ("ヲ", "wo"), ("ン", "n"),
])


# ============================================================================
# Pregenerated LLM word/phrase pool (SQLite-backed)
# ============================================================================
# The Words & Phrases deck is LLM-generated but NOT generated live per
# request: a background-topped pool keeps N fresh items ready per level so
# opens are instant and repeat visits rotate stock. Served items graduate
# into the user's SRS queue (router does the insert at serve time).
import sqlite3 as _sqlite3
import time as _time
from pathlib import Path as _Path

def list_decks() -> List[dict]:
    """Deck metadata without cards (for the lesson picker)."""
    return [
        {"id": d["id"], "title": d["title"], "subtitle": d["subtitle"],
         "kind": d["kind"], "card_count": len(d["cards"])}
        for d in _DECKS.values()
    ]


def get_deck(deck_id: str) -> dict | None:
    """Full deck with cards, or None for unknown ids."""
    return _DECKS.get(deck_id)


POOL_DB = _Path(__file__).parent / "lesson_pool.db"
POOL_MIN = 10      # top up when a level's pool drops below this
POOL_TOPUP = 15    # fresh items generated per top-up
POOL_SERVE_N = 10  # cards per deck open


def _pool_conn():
    con = _sqlite3.connect(POOL_DB)
    con.execute(
        """CREATE TABLE IF NOT EXISTS lesson_pool (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            front TEXT NOT NULL, back TEXT NOT NULL,
            reading TEXT NOT NULL DEFAULT '',
            level TEXT NOT NULL DEFAULT 'beginner',
            used_count INTEGER NOT NULL DEFAULT 0,
            created_at REAL,
            UNIQUE(front, back))"""
    )
    return con


def pool_count(level: str) -> int:
    con = _pool_conn()
    try:
        return con.execute(
            "SELECT COUNT(*) FROM lesson_pool WHERE level = ?", (level,)
        ).fetchone()[0]
    finally:
        con.close()


def pool_add(words: list, level: str) -> int:
    """Insert pregenerated words; duplicates ignored. Returns rows added."""
    con = _pool_conn()
    try:
        added = 0
        for w in words:
            try:
                con.execute(
                    "INSERT OR IGNORE INTO lesson_pool "
                    "(front, back, reading, level, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (w["front"], w["back"], w.get("reading", ""),
                     level, _time.time()),
                )
                added += con.total_changes and 1 or 0
            except Exception:
                continue
        con.commit()
        return added
    finally:
        con.close()


def pool_take(level: str, n: int = POOL_SERVE_N) -> list:
    """Take the least-used words for a level and bump their use counts."""
    con = _pool_conn()
    try:
        rows = con.execute(
            "SELECT id, front, back, reading FROM lesson_pool "
            "WHERE level = ? ORDER BY used_count ASC, RANDOM() LIMIT ?",
            (level, n),
        ).fetchall()
        ids = [r[0] for r in rows]
        if ids:
            con.execute(
                f"UPDATE lesson_pool SET used_count = used_count + 1 "
                f"WHERE id IN ({','.join('?' * len(ids))})", ids
            )
            con.commit()
        return [
            {"front": r[1], "back": r[2], "reading": r[3]} for r in rows
        ]
    finally:
        con.close()
