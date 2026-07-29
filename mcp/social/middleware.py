from __future__ import annotations

import time
from functools import wraps
from typing import Any, Callable

from social.db import get_db

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
