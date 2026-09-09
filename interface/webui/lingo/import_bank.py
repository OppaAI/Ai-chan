"""Curated JLPT bank import into materials.db."""
from __future__ import annotations

import json
import logging
import sqlite3
import time

from .content_packs import BANK_CSV, GRAMMAR_JSON

log = logging.getLogger(__name__)


def _parse_csv_banks() -> dict[str, list]:
    out: dict[str, list] = {}
    for level, blob in BANK_CSV.items():
        items = []
        for line in blob.strip().splitlines():
            parts = line.split("|")
            if len(parts) < 3:
                continue
            front, reading, back = parts[0], parts[1], parts[2]
            kind = parts[3] if len(parts) > 3 else "kanji"
            note = parts[4] if len(parts) > 4 else ""
            items.append({
                "front": front,
                "reading": reading or front,
                "back": back,
                "kind": kind or "kanji",
                "note": note,
            })
        out[level] = items
    return out


def _courses_from_banks(banks: dict) -> list[dict]:
    out = []
    for level, items in banks.items():
        chunk = 12
        for i in range(0, len(items), chunk):
            part = items[i:i + chunk]
            n = i // chunk + 1
            cards = [
                {
                    "front": it["front"],
                    "back": it["back"],
                    "reading": it["reading"],
                    "note": it.get("note", ""),
                }
                for it in part
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
        con.execute(
            "ALTER TABLE vocab_pool ADD COLUMN source TEXT NOT NULL DEFAULT 'spawn'"
        )

    kinds_ok = {"hiragana", "katakana", "kanji", "phrase", "sentence"}
    banks = _parse_csv_banks()
    for level, items in banks.items():
        for i, it in enumerate(items, 1):
            front = it["front"].strip()
            back = it["back"].strip()
            reading = (it.get("reading") or front).strip()
            kind = (it.get("kind") or "kanji").strip().lower()
            note = it.get("note") or ""
            if not front or not back:
                continue
            cid = f"{level}-c-{i:04d}"
            before = con.total_changes
            con.execute(
                "INSERT OR IGNORE INTO jlpt_cards"
                "(id,level,kind,front,reading,back,note,source) VALUES(?,?,?,?,?,?,?,?)",
                (
                    cid, level,
                    "vocab" if kind == "kanji" else kind,
                    front, reading, back, note, "curated",
                ),
            )
            if con.total_changes > before:
                stats["jlpt_cards"] += 1
            pool_kind = kind if kind in kinds_ok else "kanji"
            cur = con.execute(
                "INSERT INTO vocab_pool"
                "(front,back,reading,kind,level,used_count,created_at,source)"
                " VALUES(?,?,?,?,?,0,?,?)"
                " ON CONFLICT(front,back) DO UPDATE SET"
                " level=excluded.level, source='curated'",
                (front, back, reading, pool_kind, level, time.time(), "curated"),
            )
            if cur.rowcount:
                stats["vocab_pool"] += 1

    for c in _courses_from_banks(banks):
        con.execute(
            "INSERT OR REPLACE INTO courses(id,title,level,kind,cards_json)"
            " VALUES(?,?,?,?,?)",
            (c["id"], c["title"], c["level"], c["kind"],
             json.dumps(c["cards"], ensure_ascii=False)),
        )
        stats["courses"] += 1

    grammar = json.loads(GRAMMAR_JSON)
    for g in grammar:
        con.execute(
            "INSERT OR REPLACE INTO grammar_decks(id,title,level,kind,cards_json)"
            " VALUES(?,?,?,?,?)",
            (g["id"], g.get("title") or g["id"], g.get("level") or "N5",
             g.get("kind") or "grammar",
             json.dumps(g.get("cards") or [], ensure_ascii=False)),
        )
        stats["grammar"] += 1

    log.info("Curated import: %s", stats)
    return stats
