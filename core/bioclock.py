"""
core/bioclock.py

Single source of truth for "what time is it right now" across every module
that needs to reason about dates or wall-clock time — chat/webchat/agentic
system prompts, proactive check-ins, and core/schedule.py's job timing.

Centralized here so every module resolves timezone the same way instead of
each rolling its own ZoneInfo lookup with its own fallback quirks. Config
lives in config/bioclock.yaml (TIMEZONE key); core.config.load_config() has
already populated it into the process environment by the time this module
is imported, same as every other module's config block.

Callers that need a *different* timezone than the app default for one
specific record (e.g. a schedule.json job saved with its own "timezone"
field) can pass an explicit override to any function here — the override
always wins, the config default is only the fallback when none is given.
"""

import os
import time
import threading

from core.config import load_config
load_config()
from datetime import datetime, timezone

from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from core.log import get_logger

log = get_logger(__name__)

DEFAULT_TIMEZONE = os.getenv("TIMEZONE", "UTC")


def timezone_name(name: str | None = None) -> str:
    """Resolve the effective timezone name: an explicit override (e.g. a
    job's own "timezone" field) takes precedence, otherwise the app-wide
    default from config/bioclock.yaml."""
    name = (name or "").strip()
    return name or DEFAULT_TIMEZONE


def get_timezone(name: str | None = None) -> ZoneInfo:
    """Return a ZoneInfo for the resolved timezone, falling back to UTC
    when the name is invalid/unknown."""
    resolved = timezone_name(name)
    try:
        return ZoneInfo(resolved)
    except ZoneInfoNotFoundError:
        log.warning("[bioclock] unknown timezone %r, falling back to UTC", resolved)
        return ZoneInfo("UTC")


def local_now(name: str | None = None) -> datetime:
    """Timezone-aware 'now' for the resolved timezone (app default unless
    an override is given)."""
    return datetime.now(get_timezone(name))


def current_datetime_block(name: str | None = None) -> str:
    """Rendered <current_datetime> block for injection into any system prompt."""
    now = local_now(name)
    return (
        "<current_datetime>\n"
        f"Now: {now.strftime('%A, %B %d, %Y, %I:%M %p')} ({timezone_name(name)})\n"
        f"UTC: {utc_now().isoformat()}\n"
        "</current_datetime>"
    )


def utc_now() -> datetime:
    """Timezone-aware UTC now for persisted timestamps."""
    return datetime.now(timezone.utc)


def monotonic_now() -> float:
    """Monotonic seconds for durations, cooldowns, and polling intervals."""
    return time.monotonic()


def sleep_seconds(seconds: float) -> None:
    """Sleep for a duration; centralized so ticker/polling loops share one clock module."""
    time.sleep(max(0.0, float(seconds)))


def wait_seconds(event: threading.Event, seconds: float) -> bool:
    """Wait on an event for a duration using the centralized clock API."""
    return event.wait(timeout=max(0.0, float(seconds)))
