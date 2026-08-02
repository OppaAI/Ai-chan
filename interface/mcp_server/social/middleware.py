from __future__ import annotations

import time
from functools import wraps
from typing import Any, Callable

from social.db import get_db

# ── Rate limit configuration ───────────────────────────────────────────────

LOW_TRAFFIC_LIMIT = {"per_hour": 6, "per_day": 10}
RATE_LIMITS: dict[str, dict[str, int]] = {
    name: LOW_TRAFFIC_LIMIT
    for name in (
        "post_x", "post_threads", "post_youtube", "post_reddit",
        "post_bluesky", "post_mastodon", "post_pixelset", "post_discord",
        "post_social", "send_email", "read_emails",
    )
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