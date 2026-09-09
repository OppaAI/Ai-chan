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
        log.warning("Could not load user learnt set for %s", uid, exc_info=True)
    return keys


def _is_learned(front: str, back: str, reading: str, learned: set[tuple[str, str]]) -> bool:
    return (reading or front, back) in learned or (front, back) in learned


def pool_count_unlearned(uid: str, level: Optional[str] = None) -> int:
    """Count pool rows not yet in this user's SRS (same rules as take)."""
    if level:
        level = normalize_level(level)
    learned = _learned_keys(uid)
    con = _conn()
    try:
        if level:
            rows = con.execute(
                "SELECT front, back, reading FROM vocab_pool WHERE level = ?",
                (level,),
            ).fetchall()
        else:
            rows = con.execute("SELECT front, back, reading FROM vocab_pool").fetchall()
        return sum(
            1 for front, back, reading in rows
            if not _is_learned(front or "", back or "", reading or "", learned)
        )
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
    level = normalize_level(level or get_user_level(uid))
    learned_keys = _learned_keys(uid)
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
            if _is_learned(front, back, reading or "", learned_keys):
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
    """Graduate by pool_id only; count only newly inserted SRS rows."""
    from .srs import LingoSRS, LingoVocabCard
    srs = LingoSRS(uid)
    n = 0
    for it in items[:LEARN_SESSION_N]:
        pool_id = it.get("pool_id")
        if pool_id is None:
            continue
        stored = pool_get(int(pool_id))
        if not stored:
            log.warning("mark_learned: unknown pool_id %s", pool_id)
            continue
        front = stored["front"]
        back = stored["back"]
        reading = stored["reading"] or front
        kind = stored["kind"] or "kanji"
        level = normalize_level(stored.get("level") or get_user_level(uid))
        try:
            before = srs.get_stats().get("total_cards", 0)
            srs.add_card(LingoVocabCard(
                kanji=front if kind in ("kanji", "phrase", "sentence") else "",
                hiragana=reading,
                meaning=back,
                pos=f"{kind}|{level}",
                context=f"spawn:{level}",
                source_context="hourly_spawn",
            ))
            after = srs.get_stats().get("total_cards", 0)
            if after > before:
                n += 1
        except Exception:
            log.warning("mark_learned failed for pool_id=%s", pool_id, exc_info=True)
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

    with _inflight_lock:
        if level in _topup_inflight:
            return
        _topup_inflight.add(level)

    def _run():
        try:
            if pool_count(level) < POOL_MIN:
                items = _llm_spawn_for_level(level, max(SPAWN_PER_LEVEL, 5))
                if items:
                    pool_add(items, level=level)
        except Exception:
            log.warning("background top-up failed", exc_info=True)
        finally:
            with _inflight_lock:
                _topup_inflight.discard(level)

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
            log.warning("startup pool warm failed", exp_info=True)
    threading.Thread(target=_run, daemon=True).start()
