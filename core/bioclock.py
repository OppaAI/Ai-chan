"""
core/bioclock.py

Single source of truth for "what time is it right now" across every module
that needs to reason about dates — chat/webchat/proactive system prompts,
the agentic task loop, and (eventually) reflect.py/consolidate.py's nightly
boundary logic. Centralized here so all of them resolve the same timezone
the same way instead of each rolling its own ZoneInfo lookup.
"""

import os
from datetime import datetime

try:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
except ImportError:  # pragma: no cover - py<3.9 fallback, shouldn't happen here
    ZoneInfo = None
    ZoneInfoNotFoundError = Exception

from core.log import get_logger

log = get_logger(__name__)


def timezone_name() -> str:
    """Same fallback chain proactive.yaml already relies on: prefer a
    dedicated PROACTIVE_TIMEZONE override, fall back to the app-wide
    TIMEZONE, default to UTC."""
    return os.getenv("PROACTIVE_TIMEZONE", "").strip() or os.getenv("TIMEZONE", "UTC")


def local_now() -> datetime:
    """Tz-aware 'now', or naive local time if the configured tz string is
    bad/missing zoneinfo data. Callers should not assume tzinfo is set."""
    if ZoneInfo is None:
        return datetime.now()
    try:
        return datetime.now(ZoneInfo(timezone_name()))
    except ZoneInfoNotFoundError:
        log.warning("[bioclock] unknown timezone %r, falling back to naive local time", timezone_name())
        return datetime.now()


def current_datetime_block() -> str:
    """Rendered <current_datetime> block for injection into any system prompt."""
    now = local_now()
    return (
        "<current_datetime>\n"
        f"Now: {now.strftime('%A, %B %d, %Y, %I:%M %p')} ({timezone_name()})\n"
        "</current_datetime>"
    )
