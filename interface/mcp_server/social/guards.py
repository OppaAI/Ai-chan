from __future__ import annotations

import asyncio
import inspect
import time
from functools import wraps
from typing import Any, Callable, get_type_hints

from social.state import get_db

# ── Rate limit configuration (per-platform, with breathing room) ─────────────
RATE_LIMITS: dict[str, dict[str, int]] = {
    "post_threads": {"per_hour": 50, "per_day": 100},
    "post_youtube": {"per_hour": 5, "per_day": 10},
    "post_bluesky": {"per_hour": 100, "per_day": 500},
    "post_mastodon": {"per_hour": 30, "per_day": 100},
    "post_pixelfed": {"per_hour": 30, "per_day": 100},
    "post_discord": {"per_hour": 30, "per_day": 200},
    "post_social": {"per_hour": 30, "per_day": 100},
    "read_protonmail": {"per_hour": 30, "per_day": 100},
    "send_protonmail": {"per_hour": 20, "per_day": 50},
}


def _get_limits(tool_name: str) -> tuple[int, int]:
    """Get per_hour and per_day limits for a tool."""
    limits = RATE_LIMITS.get(tool_name, {"per_hour": 30, "per_day": 100})
    return limits["per_hour"], limits["per_day"]


_SKIP_IDEMPOTENCY = frozenset({
    "read_protonmail",
})


def wrap_tool(tool_name: str, fn: Callable[..., dict]) -> Callable[..., dict]:
    """Wrap tool with rate limiting, idempotency cache, and audit logging."""
    per_hour, per_day = _get_limits(tool_name)
    fn_is_coro = inspect.iscoroutinefunction(fn)
    fn_sig = inspect.signature(fn)

    @wraps(fn)
    async def wrapped(*args: Any, **kwargs: Any) -> dict:
        db = get_db()
        t0 = time.time()

        service = tool_name
        for prefix in ("post_", "send_", "read_", "search_", "delete_"):
            if service.startswith(prefix):
                service = service[len(prefix):]
                break

        if tool_name not in _SKIP_IDEMPOTENCY:
            cached = db.get_idempotent_result(tool_name, kwargs)
            if cached is not None:
                elapsed = (time.time() - t0) * 1000
                with db.transaction():
                    db.log_tool_call(tool_name, kwargs, cached, elapsed)
                return cached

        allowed, msg = db.check_rate_limit(service, per_hour, per_day)
        if not allowed:
            result = {"ok": False, "error": msg, "rate_limited": True}
            elapsed = (time.time() - t0) * 1000
            with db.transaction():
                db.log_tool_call(tool_name, kwargs, result, elapsed)
                db.set_idempotent_result(tool_name, kwargs, result, ttl_hours=1)
            return result

        try:
            if fn_is_coro:
                result = await fn(*args, **kwargs)
            else:
                result = await asyncio.to_thread(fn, *args, **kwargs)
        except Exception as e:
            result = {"ok": False, "error": str(e)}

        elapsed = (time.time() - t0) * 1000

        with db.transaction():
            db.log_tool_call(tool_name, kwargs, result, elapsed)

            if result.get("ok"):
                db.increment_rate_limit(service)
                if tool_name not in _SKIP_IDEMPOTENCY:
                    db.set_idempotent_result(tool_name, kwargs, result, ttl_hours=24)
            else:
                if tool_name not in _SKIP_IDEMPOTENCY and not tool_name.startswith("post_"):
                    db.set_idempotent_result(tool_name, kwargs, result, ttl_hours=1)

        return result

    try:
        resolved = get_type_hints(fn)
    except Exception:
        resolved = {}

    if resolved:
        params = [
            p.replace(annotation=resolved.get(p.name, p.annotation))
            for p in fn_sig.parameters.values()
        ]
        wrapped.__signature__ = fn_sig.replace(
            parameters=params, return_annotation=resolved.get("return", fn_sig.return_annotation)
        )
        wrapped.__annotations__ = resolved
    else:
        wrapped.__signature__ = fn_sig
    return wrapped
