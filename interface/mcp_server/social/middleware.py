from __future__ import annotations

import time
from functools import wraps
from typing import Any, Callable

from social.db import get_db

# ── Rate limit configuration (per-platform, with breathing room) ─────────────
# YouTube is quota-limited (6 videos/day on default 10k quota)
# Others have generous API limits; we cap well below to avoid bans
RATE_LIMITS: dict[str, dict[str, int]] = {
    "post_x": {"per_hour": 50, "per_day": 100},
    "post_threads": {"per_hour": 50, "per_day": 100},
    "post_youtube": {"per_hour": 5, "per_day": 10},  # Quota-limited
    "post_reddit": {"per_hour": 30, "per_day": 50},
    "post_bluesky": {"per_hour": 100, "per_day": 500},
    "post_mastodon": {"per_hour": 30, "per_day": 100},
    "post_pixelset": {"per_hour": 30, "per_day": 100},
    "post_discord": {"per_hour": 30, "per_day": 200},
    "post_social": {"per_hour": 30, "per_day": 100},
    "send_email": {"per_hour": 30, "per_day": 100},
    "read_emails": {"per_hour": 30, "per_day": 50},
}


def _get_limits(tool_name: str) -> tuple[int, int]:
    """Get per_hour and per_day limits for a tool."""
    limits = RATE_LIMITS.get(tool_name, {"per_hour": 30, "per_day": 100})
    return limits["per_hour"], limits["per_day"]


def wrap_tool(tool_name: str, fn: Callable[..., dict]) -> Callable[..., dict]:
    """
    Wrap tool with rate limiting and audit logging.
    
    Docstring: Enforce per-service rate limits (hour/day quotas),
    log all invocations to audit trail, measure execution time.
    
    Inline: Check quotas, execute, increment counter, return result.
    """
    per_hour, per_day = _get_limits(tool_name)

    @wraps(fn)
    def wrapped(**kwargs: Any) -> dict:
        db = get_db()
        t0 = time.time()
        
        # Extract service name from tool_name (e.g., "post_x" → "x")
        service = tool_name.replace("post_", "").replace("send_", "").replace("read_", "")

        # Rate limit check: block if over quota
        allowed, msg = db.check_rate_limit(service, per_hour, per_day)
        if not allowed:
            result = {"ok": False, "error": msg, "rate_limited": True}
            elapsed = (time.time() - t0) * 1000
            db.log_tool_call(tool_name, kwargs, result, elapsed)
            return result

        # Execute tool, catch exceptions
        try:
            result = fn(**kwargs)
        except Exception as e:
            result = {"ok": False, "error": str(e)}

        elapsed = (time.time() - t0) * 1000

        # Log call to audit trail
        db.log_tool_call(tool_name, kwargs, result, elapsed)

        # Increment counter only on success
        if result.get("ok"):
            db.increment_rate_limit(service)

        return result

    return wrapped