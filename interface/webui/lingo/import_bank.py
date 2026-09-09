"""Curated JLPT bank import into materials.db from content/*.json."""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path

log = logging.getLogger(__name__)
CONTENT_DIR = Path(__file__).parent / "content"


def _load_json(name: str, default):
    path = CONTENT_DIR / name
    if not path.is_file():
        log.warning("Missing content pack %s", path)
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        log.warning("Failed to read %s", path, exc_info=True)
        return default


def _load_banks() -> dict[str, list]:
    out = {}
    for level in ("N5", "N4", "N3", "N2", "N1"):
        items = _load_json(f"{level.lower()}.json", [])
        out[level] = items if isinstance(items, list) else []
    return out


def _courses_from_banks(banks: dict) -> list[dict]:
    out = []
    for level, items in banks.items():
        chunk = 12
        for i in range(0, len(items), chunk):
            part = items[i:i + chunk]
            n = i // chunk + 1
            cards = []
            for it in part:
                cards.append({
                    "front": str(it.get("front") or ""),
                    "back": str(it.get("back") or ""),
                    "reading": str(it.get("reading") or it.get("front") or ""),
                    "note": str(it.get("note") or ""),
                })
            out.append({
                "id": f"{level.lower()}-lesson-{n}",
                "title": f"{level} Lesson {n}",
                "level": level,
                "lesson": n,
                "kind": "course",
                "cards": cards,
            })
    # Prefer explicit courses.json if present
    explicit = _load_json("courses.json", None)
    if explicit:
        return explicit
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
    banks = _load_banks()
    for level, items in banks.items():
        for i, it in enumerate(items, 1):
            front = str(it.get("front") or "").strip()
            back = str(it.get("back") or "").strip()
            reading = str(it.get("reading") or front).strip()
            kind = str(it.get("kind") or "kanji").strip().lower()
            note = str(it.get("note") or "")
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

    for c in _courses_from_banks(banks):
        try:
            con.execute(
                "INSERT OR REPLACE INTO courses(id,title,level,kind,cards_json)"
                " VALUES(?,?,?,?,?)",
                (c["id"], c.get("title") or c["id"], c.get("level") or "N5",
                 c.get("kind") or "course",
                 json.dumps(c.get("cards") or [], ensure_ascii=False)),
            )
            stats["courses"] += 1
        except sqlite3.Error:
            pass

    for g in _load_json("grammar.json", []):
        try:
            con.execute(
                "INSERT OR REPLACE INTO grammar_decks(id,title,level,kind,cards_json)"
                " VALUES(?,?,?,?,?)",
                (g["id"], g.get("title") or g["id"], g.get("level") or "N5",
                 g.get("kind") or "grammar",
                 json.dumps(g.get("cards") or [], ensure_ascii=False)),
            )
            stats["grammar"] += 1
        except sqlite3.Error:
            pass

    log.info("Curated import: %s", stats)
    return stats
