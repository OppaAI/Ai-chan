"""
Lingo lesson decks (static curated content for Learn mode).

Static kana decks stay here. Legacy words-phrases pool remains on the
previous shared path when present, with package-local fallback.
"""
from typing import Dict, List

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


def list_decks() -> List[dict]:
    return [
        {"id": d["id"], "title": d["title"], "subtitle": d["subtitle"],
         "kind": d["kind"], "card_count": len(d["cards"])}
        for d in _DECKS.values()
    ]


def get_deck(deck_id: str) -> dict | None:
    return _DECKS.get(deck_id)


# ---------------------------------------------------------------------------
# Consolidated lesson_pool in materials.db (single global file).
# ---------------------------------------------------------------------------
import sqlite3 as _sqlite3
import time as _time
from pathlib import Path as _Path


def _pool_db_path() -> _Path:
    from .lingo_store import MATERIALS_DB, init_materials_db
    init_materials_db(seed=True)
    return MATERIALS_DB


POOL_DB = _Path(__file__).parent / "materials.db"
POOL_MIN = 10
POOL_TOPUP = 15
POOL_SERVE_N = 10


def _pool_conn():
    from .lingo_store import normalize_level
    path = _pool_db_path()
    con = _sqlite3.connect(str(path), timeout=30)
    con.execute(
        """CREATE TABLE IF NOT EXISTS lesson_pool (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            front TEXT NOT NULL, back TEXT NOT NULL,
            reading TEXT NOT NULL DEFAULT '',
            level TEXT NOT NULL DEFAULT 'N5',
            used_count INTEGER NOT NULL DEFAULT 0,
            created_at REAL,
            UNIQUE(front, back))"""
    )
    return con


def pool_count(level: str) -> int:
    from .lingo_store import normalize_level
    con = _pool_conn()
    try:
        return con.execute(
            "SELECT COUNT(*) FROM lesson_pool WHERE level = ?", (normalize_level(level),)
        ).fetchone()[0]
    finally:
        con.close()


def pool_add(words: list, level: str) -> int:
    from .lingo_store import normalize_level
    level = normalize_level(level)
    con = _pool_conn()
    try:
        added = 0
        for w in words:
            try:
                # total_changes is cumulative on the connection -- diff it to
                # tell a real insert apart from an IGNORED duplicate.
                before = con.total_changes
                con.execute(
                    "INSERT OR IGNORE INTO lesson_pool "
                    "(front, back, reading, level, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (w["front"], w["back"], w.get("reading", ""),
                     level, _time.time()),
                )
                if con.total_changes > before:
                    added += 1
            except Exception:
                continue
        con.commit()
        return added
    finally:
        con.close()


def pool_take(level: str, n: int = POOL_SERVE_N) -> list:
    from .lingo_store import normalize_level
    level = normalize_level(level)
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
            {"front": r[1], "back": r[2], "reading": r[3], "level": level} for r in rows
        ]
    finally:
        con.close()


def pool_take_levels(levels: list, n: int = POOL_SERVE_N) -> list:
    """Round-robin across equal-or-lower JLPT levels (N4 sees N5+N4)."""
    from .lingo_store import normalize_level
    levels = [normalize_level(l) for l in (levels or ["N5"])]
    if not levels:
        levels = ["N5"]
    per, rem = divmod(n, len(levels))
    out = []
    for i, lvl in enumerate(levels):
        out += pool_take(lvl, per + (1 if i < rem else 0))
    return out
