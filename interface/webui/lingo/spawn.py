"""
Shared Lingo vocab spawn pool + hourly scheduler handler.

Architecture (single-user friendly, multi-user ready):

  * Content (spawned JP + EN items) lives in a SHARED SQLite DB next to
    this package: interface/webui/lingo/vocab_pool.db — every process can
    read it; hourly job writes ~20 mixed items (hiragana / katakana /
    kanji / phrase / sentence).

  * Progress (learnt vs not, SM-2 schedule, review logs) stays per-user
    under USER_SPACE_ROOT/<uid>/agentic/lingo/vocab.db via LingoSRS.

  * XP / streaks stay in the existing per-user files (data/xp, data/streaks).

Learn serves pool rows the user has not yet inserted into their SRS.
Review serves the user's SRS due/learnt cards (cap 10).
"""
from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, List, Optional

log = logging.getLogger(__name__)

POOL_DB = Path(__file__).parent / "vocab_pool.db"
SPAWN_BATCH = int(os.getenv("LINGO_SPAWN_BATCH", "20"))
POOL_MIN = int(os.getenv("LINGO_POOL_MIN", "30"))
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
            level TEXT NOT NULL DEFAULT 'beginner',
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
            return con.execute(
                "SELECT COUNT(*) FROM vocab_pool WHERE level = ?", (level,)
            ).fetchone()[0]
        return con.execute("SELECT COUNT(*) FROM vocab_pool").fetchone()[0]
    finally:
        con.close()


def pool_add(items: list, level: str = "beginner") -> int:
    """Insert spawned items. Returns rows actually added."""
    con = _conn()
    added = 0
    try:
        for it in items:
            front = str(it.get("front") or "").strip()
            back = str(it.get("back") or "").strip()
            reading = str(it.get("reading") or front).strip()
            kind = str(it.get("kind") or "kanji").strip().lower()
            if kind not in ITEM_KINDS:
                kind = "kanji"
            if not front or not back:
                continue
            try:
                cur = con.execute(
                    "INSERT OR IGNORE INTO vocab_pool "
                    "(front, back, reading, kind, level, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (front, back, reading, kind, level, time.time()),
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
    """Return up to n pool items the user has not yet put in their SRS."""
    learned_keys: set[tuple[str, str]] = set()
    try:
        from .srs import LingoSRS
        srs = LingoSRS(uid)
        # Lightweight: all meanings+hiragana the user already has
        import sqlite3 as _sq
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
        if level:
            candidates = con.execute(
                "SELECT id, front, back, reading, kind FROM vocab_pool "
                "WHERE level = ? ORDER BY used_count ASC, RANDOM() LIMIT ?",
                (level, max(n * 5, 50)),
            ).fetchall()
        else:
            candidates = con.execute(
                "SELECT id, front, back, reading, kind FROM vocab_pool "
                "ORDER BY used_count ASC, RANDOM() LIMIT ?",
                (max(n * 5, 50),),
            ).fetchall()
        out = []
        taken_ids = []
        for row in candidates:
            pid, front, back, reading, kind = row
            if (reading or front, back) in learned_keys:
                continue
            if (front, back) in learned_keys:
                continue
            out.append({
                "pool_id": pid,
                "front": front,
                "back": back,
                "reading": reading,
                "kind": kind,
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
    """Graduate pool items into the user's SRS (now Reviewable). Returns count."""
    from .srs import LingoSRS, LingoVocabCard
    srs = LingoSRS(uid)
    n = 0
    for it in items:
        front = str(it.get("front") or "").strip()
        back = str(it.get("back") or "").strip()
        reading = str(it.get("reading") or front).strip()
        kind = str(it.get("kind") or "kanji")
        if not front or not back:
            continue
        try:
            srs.add_card(LingoVocabCard(
                kanji=front if kind in ("kanji", "phrase", "sentence") else "",
                hiragana=reading,
                meaning=back,
                pos=kind,
                context="spawn",
                source_context="hourly_spawn",
            ))
            n += 1
        except Exception:
            log.warning("mark_learned failed for %s", front, exc_info=True)
    return n


def _llm_spawn_batch(count: int = SPAWN_BATCH, level: str = "beginner") -> List[dict]:
    """Ask the tutor LLM for a mixed batch. Empty if brain offline."""
    try:
        from interface.webui import auth
        if not auth.aiko_web_instance or not auth.aiko_web_instance._think:
            return []
        think = auth.aiko_web_instance._think
        per = max(1, count // len(ITEM_KINDS))
        response = think._client.chat.completions.create(
            model=think._llm_model,
            messages=[
                {"role": "system", "content": (
                    "You are a Japanese teacher building a mixed study list. "
                    f"Produce exactly {count} items for a {level} learner. "
                    f"Roughly {per} of each kind: hiragana, katakana, kanji, phrase, sentence. "
                    "Each item needs natural Japanese, hiragana reading, and a short English meaning. "
                    "For hiragana/katakana kinds use only that script on the front. "
                    "For sentences keep them short (under 20 characters Japanese). "
                    "Output ONLY valid JSON: "
                    '{"items":[{"kind":"kanji","japanese":"...","hiragana":"...","meaning":"..."}]}'
                )},
                {"role": "user", "content": "Spawn a fresh mixed vocab batch."},
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
            out.append({"front": surf, "back": mean, "reading": hira, "kind": kind})
        return out
    except Exception:
        log.warning("LLM spawn batch failed", exc_info=True)
        return []


def top_up_pool(level: str = "beginner", force: bool = False) -> int:
    """If pool is low (or force), generate SPAWN_BATCH items into shared pool."""
    with _pool_lock:
        if not force and pool_count(level) >= POOL_MIN:
            return 0
        items = _llm_spawn_batch(SPAWN_BATCH, level=level)
        if not items:
            return 0
        return pool_add(items, level=level)


def top_up_pool_background(level: str = "beginner") -> None:
    threading.Thread(target=top_up_pool, args=(level,), daemon=True).start()


def handle_lingo_spawn_vocab(memorize: Any = None) -> str:
    """Scheduler handler: spawn ~20 mixed items into the shared pool."""
    levels = ("beginner", "intermediate", "advanced")
    total = 0
    for level in levels:
        # One batch per level keeps intermediate/advanced stocked too
        try:
            total += top_up_pool(level=level, force=False)
        except Exception:
            log.warning("spawn failed for level %s", level, exc_info=True)
    msg = f"Lingo spawn: added {total} items to shared pool"
    log.info(msg)
    return msg


LINGO_SPAWN_JOB_TITLE = "lingo_vocab_spawn"


def ensure_lingo_spawn_job(timezone: str | None = None, user_id: str | None = None) -> None:
    """Idempotently seed hourly shared-pool top-up in schedule.json."""
    try:
        from system import schedule as sched
        existing = {job.get("title") for job in sched._read_all(user_id=user_id)}
        if LINGO_SPAWN_JOB_TITLE in existing:
            return
        sched.schedule_job_record(
            title=LINGO_SPAWN_JOB_TITLE,
            task="Spawn mixed Japanese vocab into the shared Lingo pool",
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
    """A+B: non-blocking warm for beginner pool at process start."""
    def _run():
        try:
            top_up_pool(level="beginner", force=False)
        except Exception:
            log.warning("startup pool warm failed", exc_info=True)
    threading.Thread(target=_run, daemon=True).start()
