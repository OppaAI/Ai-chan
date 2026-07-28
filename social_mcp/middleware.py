from __future__ import annotations

import time
from functools import wraps
from typing import Any, Callable

from social_mcp.db import get_db

RATE_LIMITS: dict[str, dict[str, int]] = {
    "post_x": {"per_hour": 50, "per_day": 300},
    "post_threads": {"per_hour": 24, "per_day": 100},
    "post_instagram": {"per_hour": 10, "per_day": 50},
    "post_youtube": {"per_hour": 6, "per_day": 10},
    "post_reddit": {"per_hour": 6, "per_day": 30},
    "post_bluesky": {"per_hour": 48, "per_day": 300},
    "post_mastodon": {"per_hour": 30, "per_day": 100},
    "post_discord": {"per_hour": 30, "per_day": 200},
    "post_telegram": {"per_hour": 30, "per_day": 200},
    "post_slack": {"per_hour": 60, "per_day": 500},
    "post_linkedin": {"per_hour": 10, "per_day": 50},
    "post_facebook": {"per_hour": 10, "per_day": 50},
    "send_email": {"per_hour": 50, "per_day": 300},
    "read_emails": {"per_hour": 60, "per_day": 500},
}


def _get_limits(tool_name: str) -> tuple[int, int]:
    limits = RATE_LIMITS.get(tool_name, {"per_hour": 30, "per_day": 100})
    return limits["per_hour"], limits["per_day"]


def wrap_tool(tool_name: str, fn: Callable[..., dict]) -> Callable[..., dict]:
    per_hour, per_day = _get_limits(tool_name)

    @wraps(fn)
    def wrapped(**kwargs: Any) -> dict:
        db = get_db()

        t0 = time.time()
        service = tool_name.replace("post_", "").replace("send_", "").replace("read_", "")

        # Idempotency check
        cached = db.get_idempotent_result(tool_name, kwargs)
        if cached is not None:
            elapsed = (time.time() - t0) * 1000
            db.log_tool_call(tool_name, kwargs, cached, elapsed)
            return cached

        # Rate limit check
        allowed, msg = db.check_rate_limit(service, per_hour, per_day)
        if not allowed:
            result = {"ok": False, "error": msg, "rate_limited": True}
            elapsed = (time.time() - t0) * 1000
            db.log_tool_call(tool_name, kwargs, result, elapsed)
            db.set_idempotent_result(tool_name, kwargs, result, ttl_hours=1)
            return result

        # Execute
        try:
            result = fn(**kwargs)
        except Exception as e:
            result = {"ok": False, "error": str(e)}

        elapsed = (time.time() - t0) * 1000

        # Log and cache
        db.log_tool_call(tool_name, kwargs, result, elapsed)
        if result.get("ok"):
            db.increment_rate_limit(service)
            db.set_idempotent_result(tool_name, kwargs, result, ttl_hours=24)
        else:
            # Cache failures for shorter time to allow retry
            db.set_idempotent_result(tool_name, kwargs, result, ttl_hours=1)

        return result

    return wrapped
