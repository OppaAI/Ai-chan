"""
Shared Lingo vocab spawn pool + hourly scheduler handler.

  * Content: interface/webui/lingo/vocab_pool.db (shared, tagged with JLPT level)
  * Progress: USER_SPACE_ROOT/<uid>/agentic/lingo/vocab.db (per-user SRS)

Hourly job spawns 2 items per JLPT level N5..N1 (10 total), each tagged
with its level. Learn/Review only surface items matching the user's level.
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
SPAWN_PER_LEVEL = int(os.getenv("LINGO_SPAWN_PER_LEVEL", "2"))  # 2 × 5 = 10/hour
POOL_MIN = int(os.getenv("LINGO_POOL_MIN", "20"))
LEARN_SESSION_N = 10
REVIEW_SESSION_N = 10

ITEM_KINDS = ("hiragana", "katakana", "kanji", "phrase", "sentence")

_pool_lock = threading.Lock()


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


def pool_add(items: list, level: str = "N5") -> int:
    level = normalize_level(level)
    con = _conn()
    added = 0
    try:
        for it in items:
            front = str(it.get("front") or "").strip()
            back = str(it.get("back") or "").strip()
            reading = str(it.get("reading") or front).strip()
            kind = str(it.get("kind") or "kanji").strip().lower()
            item_level = normalize_level(it.get("level") or level)
            if kind not in ITEM_KINDS:
                kind = "kanji"
            if not front or not back:
                continue
            try:
                cur = con.execute(
                    "INSERT OR IGNORE INTO vocab_pool "
                    "(front, back, reading, kind, level, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (front, back, reading, kind, item_level, time.time()),
                )
                if cur.rowcount:
                    added += 1
            except sqlite3.Error:
                continue
        con.commit()
        return added
    finally:
        con.close()


def pool_take_unlearned(
    uid: str,
    n: int = LEARN_SESSION_N,
    level: Optional[str] = None,
) -> List[dict]:
    """Pool items for the user's JLPT level that are not yet in their SRS."""
    level = normalize_level(level or get_user_level(uid))
    learned_keys: set[tuple[str, str]] = set()
    try:
        from .srs import LingoSRS
        import sqlite3 as _sq
        srs = LingoSRS(uid)
        con_u = _sq.connect(str(srs.db_path))
        try:
            rows = con_u.execute(
                "SELECT hiragana, meaning FROM lingo_vocab_cards WHERE user_id = ?",
                (uid,),
            ).fetchall()
            learned_keys = {(r[0] or "", r[1] or "") for r in rows}
        finally:
            con_u.close()
    except Exception:
        log.warning("Could not load user learnt set for %s", uid, exc_info=True)

    con = _conn()
    try:
        candidates = con.execute(
            "SELECT id, front, back, reading, kind, level FROM vocab_pool "
            "WHERE level = ? ORDER BY used_count ASC, RANDOM() LIMIT ?",
            (level, max(n * 5, 50)),
        ).fetchall()
        out = []
        taken_ids = []
        for row in candidates:
            pid, front, back, reading, kind, lv = row
            if (reading or front, back) in learned_keys or (front, back) in learned_keys:
                continue
            out.append({
                "pool_id": pid,
                "front": front,
                "back": back,
                "reading": reading,
                "kind": kind,
                "level": lv,
            })
            taken_ids.append(pid)
            if len(out) >= n:
                break
        if taken_ids:
            con.execute(
                f"UPDATE vocab_pool SET used_count = used_count + 1 "
                f"WHERE id IN ({','.join('?' * len(taken_ids))})",
                taken_ids,
            )
            con.commit()
        return out
    finally:
        con.close()


def mark_learned(uid: str, items: List[dict]) -> int:
    from .srs import LingoSRS, LingoVocabCard
    srs = LingoSRS(uid)
    n = 0
    for it in items:
        front = str(it.get("front") or "").strip()
        back = str(it.get("back") or "").strip()
        reading = str(it.get("reading") or front).strip()
        kind = str(it.get("kind") or "kanji")
        level = normalize_level(it.get("level") or get_user_level(uid))
        if not front or not back:
            continue
        try:
            srs.add_card(LingoVocabCard(
                kanji=front if kind in ("kanji", "phrase", "sentence") else "",
                hiragana=reading,
                meaning=back,
                pos=f"{kind}|{level}",
                context=f"spawn:{level}",
                source_context="hourly_spawn",
            ))
            n += 1
        except Exception:
            log.warning("mark_learned failed for %s", front, exc_info=True)
    return n


def _llm_spawn_for_level(level: str, count: int = SPAWN_PER_LEVEL) -> List[dict]:
    try:
        from interface.webui import auth
        if not auth.aiko_web_instance or not auth.aiko_web_instance._think:
            return []
        think = auth.aiko_web_instance._think
        level = normalize_level(level)
        guidance = {
            "N5": "basic greetings, numbers, family, food; mostly kana; very simple words",
            "N4": "everyday school/work/shopping; common verbs and adjectives",
            "N3": "daily life plus some abstract nouns; newspaper headlines level",
            "N2": "news, workplace, formal written Japanese",
            "N1": "abstract, academic, literary, nuanced formal vocabulary",
        }.get(level, "appropriate difficulty")
        response = think._client.chat.completions.create(
            model=think._llm_model,
            messages=[
                {"role": "system", "content": (
                    f"You are a JLPT {level} Japanese teacher. Produce exactly {count} study items "
                    f"at JLPT {level} ({guidance}). Mix kinds: hiragana, katakana, kanji, phrase, sentence when suitable. "
                    "Each item: natural Japanese, hiragana reading, short English meaning. "
                    "Output ONLY JSON: "
                    '{"items":[{"kind":"kanji","japanese":"...","hiragana":"...","meaning":"..."}]}'
                )},
                {"role": "user", "content": f"Spawn {count} fresh JLPT {level} items."},
            ],
            response_format={"type": "json_object"},
            timeout=90.0,
        )
        import json, re
        raw = response.choices[0].message.content or ""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            data = json.loads(m.group(0)) if m else {}
        out = []
        for w in (data.get("items") or data.get("words") or [])[:count]:
            surf = str(w.get("japanese") or w.get("front") or "").strip()
            hira = str(w.get("hiragana") or w.get("reading") or surf).strip()
            mean = str(w.get("meaning") or w.get("back") or "").strip()
            kind = str(w.get("kind") or "kanji").strip().lower()
            if kind not in ITEM_KINDS:
                kind = "kanji"
            if not surf or not mean or len(surf) > 40:
                continue
            out.append({
                "front": surf, "back": mean, "reading": hira,
                "kind": kind, "level": level,
            })
        return out
    except Exception:
        log.warning("LLM spawn for %s failed", level, exc_info=True)
        return []


def top_up_all_levels(force: bool = False) -> int:
    """Spawn SPAWN_PER_LEVEL items for each N5..N1 (default 2×5=10)."""
    total = 0
    with _pool_lock:
        for level in JLPT_LEVELS:
            if not force and pool_count(level) >= POOL_MIN:
                continue
            items = _llm_spawn_for_level(level, SPAWN_PER_LEVEL)
            if items:
                total += pool_add(items, level=level)
    return total


def top_up_user_level_background(uid: str) -> None:
    level = get_user_level(uid)

    def _run():
        try:
            if pool_count(level) < POOL_MIN:
                items = _llm_spawn_for_level(level, max(SPAWN_PER_LEVEL, 5))
                if items:
                    pool_add(items, level=level)
        except Exception:
            log.warning("background top-up failed", exc_info=True)

    threading.Thread(target=_run, daemon=True).start()


def handle_lingo_spawn_vocab(memorize: Any = None) -> str:
    n = top_up_all_levels(force=True)
    msg = f"Lingo spawn: added {n} items (2× N5–N1) to shared pool"
    log.info(msg)
    return msg


LINGO_SPAWN_JOB_TITLE = "lingo_vocab_spawn"


def ensure_lingo_spawn_job(timezone: str | None = None, user_id: str | None = None) -> None:
    try:
        from system import schedule as sched
        existing = {job.get("title") for job in sched._read_all(user_id=user_id)}
        if LINGO_SPAWN_JOB_TITLE in existing:
            return
        sched.schedule_job_record(
            title=LINGO_SPAWN_JOB_TITLE,
            task="Spawn JLPT N5–N1 vocab (2 each) into shared Lingo pool",
            time_of_day="00:00",
            frequency="hourly",
            timezone=timezone,
            action="agentic",
            handler="lingo_spawn_vocab",
            user_id=user_id,
        )
        log.info("Seeded hourly lingo vocab spawn job")
    except Exception:
        log.warning("ensure_lingo_spawn_job failed", exc_info=True)


def register_lingo_spawn_handler(seed_jobs: bool = False, timezone: str | None = None,
                                 user_id: str | None = None) -> None:
    from system.schedule import register_system_handler
    register_system_handler(
        "lingo_spawn_vocab",
        lambda memorize: handle_lingo_spawn_vocab(memorize),
    )
    if seed_jobs:
        ensure_lingo_spawn_job(timezone=timezone, user_id=user_id)


def warm_pools_on_startup() -> None:
    def _run():
        try:
            top_up_all_levels(force=False)
        except Exception:
            log.warning("startup pool warm failed", exc_info=True)
    threading.Thread(target=_run, daemon=True).start()
