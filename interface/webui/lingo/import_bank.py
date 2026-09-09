"""Curated JLPT bank import into materials.db."""
from __future__ import annotations

import json
import logging
import sqlite3
import time

from .bank_data import BANKS
from .bank_rest import GRAMMAR

log = logging.getLogger(__name__)


def _courses_from_banks() -> list[dict]:
    out = []
    for level, items in BANKS.items():
        chunk = 12
        for i in range(0, len(items), chunk):
            part = items[i:i + chunk]
            n = i // chunk + 1
            cards = [
                {"front": f, "back": b, "reading": r, "note": note}
                for f, r, b, kind, note in part
            ]
            out.append({
                "id": f"{level.lower()}-lesson-{n}",
                "title": f"{level} Lesson {n}",
                "level": level,
                "lesson": n,
                "kind": "course",
                "cards": cards,
            })
    return out


def import_curated_into(con: sqlite3.Connection) -> dict[str, int]:
    stats = {"jlpt_cards": 0, "vocab_pool": 0, "courses": 0, "grammar": 0}
    cols = {r[1] for r in con.execute("PRAGMA table_info(vocab_pool)").fetchall()}
    if "source" not in cols:
        try:
            con.execute(
                "ALTER TABLE vocab_pool ADD COLUMN source TEXT NOT NULL DEFAULT 'spawn'"
            )
            cols.add("source")
        except sqlite3.Error:
            pass

    kinds_ok = {"hiragana", "katakana", "kanji", "phrase", "sentence"}
    for level, items in BANKS.items():
        for i, (front, reading, back, kind, note) in enumerate(items, 1):
            front, back, reading = front.strip(), back.strip(), (reading or front).strip()
            if not front or not back:
                continue
            cid = f"{level}-c-{i:04d}"
            try:
                before = con.total_changes
                con.execute(
                    "INSERT OR IGNORE INTO jlpt_cards"
                    "(id,level,kind,front,reading,back,note,source) VALUES(?,?,?,?,?,?,?,?)",
                    (cid, level, "vocab" if kind == "kanji" else kind,
                     front, reading, back, note, "curated"),
                )
                if con.total_changes > before:
                    stats["jlpt_cards"] += 1
            except sqlite3.Error:
                pass
            pool_kind = kind if kind in kinds_ok else "kanji"
            try:
                if "source" in cols:
                    cur = con.execute(
                        "INSERT OR IGNORE INTO vocab_pool"
                        "(front,back,reading,kind,level,used_count,created_at,source)"
                        " VALUES(?,?,?,?,?,0,?,?)",
                        (front, back, reading, pool_kind, level, time.time(), "curated"),
                    )
                else:
                    cur = con.execute(
                        "INSERT OR IGNORE INTO vocab_pool"
                        "(front,back,reading,kind,level,used_count,created_at)"
                        " VALUES(?,?,?,?,?,0,?)",
                        (front, back, reading, pool_kind, level, time.time()),
                    )
                if cur.rowcount:
                    stats["vocab_pool"] += 1
            except sqlite3.Error:
                try:
                    cur = con.execute(
                        "INSERT OR IGNORE INTO vocab_pool"
                        "(front,back,reading,kind,level,used_count,created_at)"
                        " VALUES(?,?,?,?,?,0,?)",
                        (front, back, reading, pool_kind, level, time.time()),
                    )
                    if cur.rowcount:
                        stats["vocab_pool"] += 1
                except sqlite3.Error:
                    pass

    for c in _courses_from_banks():
        try:
            con.execute(
                "INSERT OR REPLACE INTO courses(id,title,level,kind,cards_json)"
                " VALUES(?,?,?,?,?)",
                (c["id"], c["title"], c["level"], c["kind"],
                 json.dumps(c["cards"], ensure_ascii=False)),
            )
            stats["courses"] += 1
        except sqlite3.Error:
            pass

    for g in GRAMMAR:
        try:
            con.execute(
                "INSERT OR REPLACE INTO grammar_decks(id,title,level,kind,cards_json)"
                " VALUES(?,?,?,?,?)",
                (g["id"], g["title"], g["level"], g.get("kind") or "grammar",
                 json.dumps(g.get("cards") or [], ensure_ascii=False)),
            )
            stats["grammar"] += 1
        except sqlite3.Error:
            pass

    log.info("Curated import: %s", stats)
    return stats
