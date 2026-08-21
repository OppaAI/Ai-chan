from __future__ import annotations

import asyncio
import inspect
import time
from functools import wraps
from typing import Any, Callable

from social.state import get_db

# ── Rate limit configuration (per-platform, with breathing room) ─────────────
# YouTube is quota-limited (6 videos/day on default 10k quota)
# Others have generous API limits; we cap well below to avoid bans
RATE_LIMITS: dict[str, dict[str, int]] = {
    "post_x": {"per_hour": 50, "per_day": 100},
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


# Tools that must always hit live data — never serve cached results.
# Read/search tools need fresh inbox state; serving a 24h-old snapshot
# means Aiko misses new emails entirely until the cache expires.
_SKIP_IDEMPOTENCY = frozenset({
    "read_protonmail",
})


def wrap_tool(tool_name: str, fn: Callable[..., dict]) -> Callable[..., dict]:
    """
    Wrap tool with rate limiting, idempotency cache, and audit logging.

    Docstring: Enforce per-service rate limits (hour/day quotas),
    cache results to prevent duplicate API calls on retry, log all
    invocations to audit trail, measure execution time.

    Inline: Check idempotency cache first (return cached result if hit),
    then check rate limits, execute tool, log result, cache on success.
    Read/search tools bypass the idempotency cache entirely so they always
    return live data.
    """
    per_hour, per_day = _get_limits(tool_name)
    fn_is_coro = inspect.iscoroutinefunction(fn)
    fn_sig = inspect.signature(fn)

    @wraps(fn)
    async def wrapped(*args: Any, **kwargs: Any) -> dict:
        db = get_db()
        t0 = time.time()

        # Extract service name from tool_name (e.g., "post_x" → "x", "search_protonmail" → "protonmail")
        service = tool_name
        for prefix in ("post_", "send_", "read_", "search_", "delete_"):
            if service.startswith(prefix):
                service = service[len(prefix):]
                break

        # ── Idempotency check ─────────────────────────────────────────────
        # If same tool + same arguments called recently, return cached result.
        # Prevents duplicate posts if Aiko crashes mid-call and retries.
        # Read/search tools are excluded — they must always return live data.
        if tool_name not in _SKIP_IDEMPOTENCY:
            cached = db.get_idempotent_result(tool_name, kwargs)
            if cached is not None:
                elapsed = (time.time() - t0) * 1000
                with db.transaction():
                    db.log_tool_call(tool_name, kwargs, cached, elapsed)
                return cached

        # ── Rate limit check ──────────────────────────────────────────────
        # Block if over quota (hour or day).
        allowed, msg = db.check_rate_limit(service, per_hour, per_day)
        if not allowed:
            result = {"ok": False, "error": msg, "rate_limited": True}
            elapsed = (time.time() - t0) * 1000
            with db.transaction():
                db.log_tool_call(tool_name, kwargs, result, elapsed)
                # Cache rate limit response for 1 hour (allow retry later)
                db.set_idempotent_result(tool_name, kwargs, result, ttl_hours=1)
            return result

        # ── Execute tool ──────────────────────────────────────────────────
        try:
            if fn_is_coro:
                result = await fn(*args, **kwargs)
            else:
                result = await asyncio.to_thread(fn, *args, **kwargs)
        except Exception as e:
            result = {"ok": False, "error": str(e)}

        elapsed = (time.time() - t0) * 1000

        # ── Log and cache ─────────────────────────────────────────────────
        with db.transaction():
            db.log_tool_call(tool_name, kwargs, result, elapsed)

            if result.get("ok"):
                # Success: increment rate limit, cache result for 24 hours
                db.increment_rate_limit(service)
                # Skip idempotency caching for read/search tools (live-data tools).
                if tool_name not in _SKIP_IDEMPOTENCY:
                    db.set_idempotent_result(tool_name, kwargs, result, ttl_hours=24)
            else:
                # Failure: cache for shorter time (1 hour) to allow retry
                # Never cache failed posting attempts. A transient provider,
                # connection, or worker error must be retryable immediately;
                # replaying it can hide a repaired service for an hour.
                if tool_name not in _SKIP_IDEMPOTENCY and not tool_name.startswith("post_"):
                    db.set_idempotent_result(tool_name, kwargs, result, ttl_hours=1)

        return result

    # Preserve the original function's signature for FastMCP schema introspection
    wrapped.__signature__ = fn_sig
    return wrapped
