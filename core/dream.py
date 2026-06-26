"""
core/dream.py
Schedules Aiko's daily dream() consolidation pass after the configured local time.

Usage — call start() once during Aiko startup (e.g. in UI):
    from core.dream import start as start_dream_scheduler
    start_dream_scheduler(memorize_instance)

The scheduler runs in a background daemon thread and fires once per local date after the daily window opens; missed runs fire on the next boot after that window.
It does NOT block startup or conversation flow.

 VRAM safety:
     memorize.dream() itself does zero LLM calls — only sqlite-vec ops and memory deletes.
     This scheduler also runs reflection/monthly consolidation, which may call the LLM;
     avoid configuring the window during expected active conversation time.
     A _dream_lock flag is checked to avoid overlapping passes.
"""

from datetime import datetime, timedelta, timezone
import json
import os
import threading
import time
from pathlib import Path

from core.log import get_logger
from core.reflect import generate_and_post
from core.monthly import maybe_run_monthly_consolidation

log = get_logger(__name__)

# Prevent overlapping dream passes (e.g. if system clock jumps or scheduler
# fires twice due to a suspend/resume cycle).
_dream_lock = threading.Lock()

_DREAM_STATE_PATH = Path(os.getenv("DREAM_STATE_PATH", str(Path.home() / ".aiko" / "dream_state.json")))
_DREAM_POLL_SECONDS = max(60, int(os.getenv("DREAM_POLL_SECONDS", "300")))
_DREAM_RUN_AFTER_HOUR = int(os.getenv("DREAM_RUN_AFTER_HOUR", "0"))
_DREAM_RUN_AFTER_MINUTE = int(os.getenv("DREAM_RUN_AFTER_MINUTE", "0"))


def _load_last_run_date() -> str | None:
    try:
        data = json.loads(_DREAM_STATE_PATH.read_text(encoding="utf-8"))
        return data.get("last_run_date")
    except Exception:
        return None


def _save_last_run_date(day: str) -> None:
    _DREAM_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _DREAM_STATE_PATH.write_text(json.dumps({"last_run_date": day}, ensure_ascii=False), encoding="utf-8")


def _after_daily_window(now: datetime) -> bool:
    return (now.hour, now.minute) >= (_DREAM_RUN_AFTER_HOUR, _DREAM_RUN_AFTER_MINUTE)


def _run_dream_once(memorize, run_day: str) -> None:
    if not _dream_lock.acquire(blocking=False):
        log.warning("Pass already running — skipping.")
        return

    try:
        log.info("Firing dream() for %s at %s", run_day, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        result = memorize.dream()
        log.info(f"Pass complete: {result}")

        # anchor: yesterday 00:00 UTC
        yesterday_start = (datetime.now(timezone.utc) - timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )

        # get yesterday's memories first, sorted newest-first
        memories = memorize.get_since(yesterday_start)

        # backfill with older memories if yesterday was quiet
        if len(memories) < int(os.getenv("REFLECT_MAX_MEMS", 20)):
            seen_ids  = {m["id"] for m in memories}
            all_mems  = sorted(
                memorize.get_all(),
                key=lambda m: m.get("created_at", ""),
                reverse=True,
            )
            for m in all_mems:
                if m["id"] not in seen_ids:
                    memories.append(m)
                if len(memories) >= int(os.getenv("REFLECT_MAX_MEMS", 20)):
                    break

        generate_and_post(memories, date=yesterday_start, memorize=memorize)
        _save_last_run_date(run_day)
    except Exception as e:
        log.error(f"dream() raised: {e}")
    finally:
        _dream_lock.release()


def _dream_loop(memorize) -> None:
    """Run once per local date after the configured daily window opens."""
    while True:
        now = datetime.now()
        today = now.date().isoformat()
        last_run = _load_last_run_date()
        if _after_daily_window(now) and last_run != today:
            _run_dream_once(memorize, today)
        if _after_daily_window(now):
            try:
                monthly = maybe_run_monthly_consolidation(memorize, now=now)
                if monthly.get("ran"):
                    log.info("Monthly consolidation result: %s", monthly)
            except Exception as e:
                log.error("monthly consolidation raised: %s", e)
        time.sleep(_DREAM_POLL_SECONDS)

def start(memorize) -> threading.Thread:
    """
    Start the nightly dream scheduler as a daemon background thread.

    Args:
        memorize: An initialised AikoMemorize instance.

    Returns the Thread (rarely needed, but useful for testing).
    """
    t = threading.Thread(
        target=_dream_loop,
        args=(memorize,),
        daemon=True,
        name="dream-scheduler",
    )
    t.start()
    log.info(
        "Started — will consolidate once daily after %02d:%02d; missed runs fire on next boot.",
        _DREAM_RUN_AFTER_HOUR,
        _DREAM_RUN_AFTER_MINUTE,
    )
    return t
