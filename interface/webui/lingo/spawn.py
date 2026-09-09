"""
Shared Lingo vocab spawn pool + hourly scheduler handler.

  * Content: interface/webui/lingo/vocab_pool.db (shared)
  * Progress: USER_SPACE_ROOT/<uid>/agentic/lingo/vocab.db (per-user SRS)
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
            level TEXT NOT NULL DEFAULT 'beginner',
            used_count INTEGER NOT NULL DEFAULT 0,
            created_at REAL,
            UNIQUE(front, back)
        )"""
    )
    return con


def pool_get(pool_id: int) -> Optional[dict]:
    """Fetch one pool row by id, or None."""
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
    """Count pool rows not yet in this user's SRS (same match rules as take)."""
    learned = _learned_keys(uid)
    con = _conn()
    try:
        if level:
            rows = con.execute(
                "SELECT front, back, reading FROM vocab_pool WHERE level = ?",
                (level,),
            ).fetchall()
        else:
            rows = con.execute(
                "SELECT front, back, reading FROM vocab_pool"
            ).fetchall()
        n = 0
        for front, back, reading in rows:
            if not _is_learned(front or "", back or "", reading or "", learned):
                n += 1
        return n
    finally:
        con.close()


def pool_add(items: list, level: str = "beginner") -> int:
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
    learned_keys = _learned_keys(uid)
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
            if _is_learned(front, back, reading or "", learned_keys):
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
    """Graduate pool items into user SRS. Resolves by pool_id only; counts new rows."""
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
        try:
            before = srs.get_stats().get("total_cards", 0)
            card = srs.add_card(LingoVocabCard(
                kanji=front if kind in ("kanji", "phrase", "sentence") else "",
                hiragana=reading,
                meaning=back,
                pos=kind,
                context="spawn",
                source_context="hourly_spawn",
            ))
            # Only count if this is a newly inserted row (new id and total grew),
            # or if add_card returned a card that was just created (review_count 0
            # and created this call). Safest: compare totals.
            after = srs.get_stats().get("total_cards", 0)
            if after > before:
                n += 1
            elif card and getattr(card, "id", None) and getattr(card, "review_count", 0) == 0:
                # Existing unreviewed card — do not re-award XP
                pass
        except Exception:
            log.warning("mark_learned failed for pool_id=%s", pool_id, exc_info=True)
    return n


def _llm_spawn_batch(count: int = SPAWN_BATCH, level: str = "beginner") -> List[dict]:
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
    with _pool_lock:
        if not force and pool_count(level) >= POOL_MIN:
            return 0
        items = _llm_spawn_batch(SPAWN_BATCH, level=level)
        if not items:
            return 0
        return pool_add(items, level=level)


def top_up_pool_background(level: str = "beginner") -> None:
    """One in-flight top-up per level — avoids thread pile-up under concurrent Learn."""
    with _inflight_lock:
        if level in _topup_inflight:
            return
        _topup_inflight.add(level)

    def _run() -> None:
        try:
            top_up_pool(level=level)
        finally:
            with _inflight_lock:
                _topup_inflight.discard(level)

    threading.Thread(target=_run, daemon=True).start()


def handle_lingo_spawn_vocab(memorize: Any = None) -> str:
    levels = ("beginner", "intermediate", "advanced")
    total = 0
    for level in levels:
        try:
            total += top_up_pool(level=level, force=False)
        except Exception:
            log.warning("spawn failed for level %s", level, exc_info=True)
    msg = f"Lingo spawn: added {total} items to shared pool"
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
    def _run():
        try:
            top_up_pool(level="beginner", force=False)
        except Exception:
            log.warning("startup pool warm failed", exc_info=True)
    threading.Thread(target=_run, daemon=True).start()
