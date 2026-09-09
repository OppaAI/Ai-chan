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
# Legacy lesson_pool for /lessons/words-phrases (router still uses this).
# Prefer existing USER_SPACE_ROOT/_shared/agentic/lingo/lesson_pool.db so
# prior data is not orphaned; fall back to package-local path.
# ---------------------------------------------------------------------------
import sqlite3 as _sqlite3
import time as _time
from pathlib import Path as _Path


def _legacy_pool_db_path() -> _Path:
    candidates = []
    try:
        from system.userspace import _user_state_root_value
        shared = (
            _Path(_user_state_root_value()).expanduser()
            / "_shared" / "agentic" / "lingo" / "lesson_pool.db"
        )
        candidates.append(shared)
    except Exception:
        pass
    candidates.append(_Path(__file__).parent / "lesson_pool.db")
    for p in candidates:
        if p.is_file():
            return p
    # Prefer shared path for new writes when userspace works
    try:
        from system.userspace import _user_state_root_value
        p = (
            _Path(_user_state_root_value()).expanduser()
            / "_shared" / "agentic" / "lingo" / "lesson_pool.db"
        )
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    except Exception:
        return _Path(__file__).parent / "lesson_pool.db"


POOL_DB = _legacy_pool_db_path()
POOL_MIN = 10
POOL_TOPUP = 15
POOL_SERVE_N = 10


def _pool_conn():
    path = _legacy_pool_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    con = _sqlite3.connect(path)
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
