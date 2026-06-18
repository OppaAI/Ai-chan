"""
core/dream.py
Schedules Aiko's nightly dream() consolidation pass at 00:00 local time.

Usage — call start() once during Aiko startup (e.g. in UI):
    from core.dream import start as start_dream_scheduler
    start_dream_scheduler(memorize_instance)

The scheduler runs in a background daemon thread and fires dream() at midnight.
It does NOT block startup or conversation flow.

 VRAM safety:
     dream() does zero LLM calls — only sqlite-vec ops and memory deletes.
     No Ollama contention. Safe to fire even if a conversation is mid-flight,
     though a _dream_lock flag is checked to avoid overlapping passes.
"""

from datetime import datetime, timedelta, timezone
import os
import threading
import time

from core.log import get_logger
from core.reflect import generate_and_post

log = get_logger(__name__)

# Prevent overlapping dream passes (e.g. if system clock jumps or scheduler
# fires twice due to a suspend/resume cycle).
_dream_lock = threading.Lock()


def _seconds_until_midnight() -> float:
    """Seconds from now until the next local 00:00:00."""
    now       = datetime.now()
    tomorrow  = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return max((tomorrow - now).total_seconds(), 0.0)

def _dream_loop(memorize) -> None:
    """
    Background loop: sleep until midnight, fire dream(), repeat.
    Runs as a daemon thread — exits automatically when the main process ends.
    """
    while True:
        wait = _seconds_until_midnight()
        log.info(f"Next consolidation pass in {wait / 3600:.1f}h (at midnight).")
        time.sleep(wait)

        if not _dream_lock.acquire(blocking=False):
            log.warning("Pass already running — skipping.")
            # Sleep a bit to avoid tight loop if something is wrong
            time.sleep(60)
            continue

        try:
            log.info(f"Firing dream() at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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
        except Exception as e:
            log.error(f"dream() raised: {e}")
        finally:
            _dream_lock.release()

        # Sleep 90 seconds after firing so we don't re-trigger at 00:00:00
        # on the same second due to scheduler drift.
        time.sleep(90)

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
    log.info("Started — will consolidate memories nightly at midnight.")
    return t
