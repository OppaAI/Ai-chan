"""Shared notification helpers (email + optional Threads).

Best-effort: uses registered tools when present; never raises to callers.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from typing import Any

log = logging.getLogger(__name__)


def notify_email(subject: str, body: str, *, to: str | None = None) -> dict[str, Any]:
    """Send an email via a registered mail tool if available.

    Tries common tool names: send_email, send_protonmail. The recipient
    defaults to the AIKO_EMAIL env var so aurora/job-alert notifications
    reach the owner even when the caller doesn't pass `to` explicitly.
    """
    import os
    if not to:
        to = os.getenv("AIKO_EMAIL", "").strip() or None

    try:
        from agentic.registry import registry
    except Exception as e:
        return {"ok": False, "error": f"registry unavailable: {e}"}

    for name in ("send_email", "send_protonmail", "compose_email"):
        spec = registry.get(name)
        if spec is None or spec.handler is None:
            continue
        try:
            # Introspect handler signature to determine compatible arguments
            sig = inspect.signature(spec.handler)
            params = sig.parameters
            args: dict[str, Any] = {}

            # Always include subject and body if parameters accept them
            if "subject" in params:
                args["subject"] = subject
            if "body" in params:
                args["body"] = body
            # Recipients: map `to` (single address) to whichever plural or
            # singular param the tool exposes (recipients / to / address).
            if to:
                if "recipients" in params:
                    args["recipients"] = [to]
                elif "to" in params:
                    args["to"] = to
                elif "address" in params:
                    args["address"] = to
            if not to:
                # No recipient resolved and the tool needs one; skip this
                # tool rather than calling it with an empty recipient.
                continue

            if inspect.iscoroutinefunction(spec.handler):
                try:
                    asyncio.get_running_loop()
                except RuntimeError:
                    # No running loop: safe to use asyncio.run
                    result = asyncio.run(spec.handler(**args))
                else:
                    # Already in a loop: isolate in a dedicated thread
                    import concurrent.futures

                    def _run_isolated():  # type: ignore[no-untyped-def]
                        return asyncio.run(spec.handler(**args))

                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                        result = pool.submit(_run_isolated).result(timeout=60)
            else:
                result = spec.handler(**args)
            if isinstance(result, dict):
                return {"ok": bool(result.get("ok", True)), "tool": name, "result": result}
            return {"ok": True, "tool": name, "result": result}
        except Exception as e:
            log.warning("notify_email: %s failed: %s", name, e)
            return {"ok": False, "tool": name, "error": str(e)}

    log.info("notify_email: no send_email tool registered; subject=%s", subject[:80])
    return {"ok": False, "error": "no_email_tool", "subject": subject, "body_preview": body[:200]}


def maybe_post_threads(
    text: str,
    *,
    enabled: bool = True,
    reason: str = "",
) -> dict[str, Any]:
    """Optionally post text to Threads via registered social tools.

    Only attempts when ``enabled`` is True (caller applies thresholds e.g. KP>5).
    Uses post_to_social when available; does not bypass approval policy.
    """
    if not enabled:
        return {"ok": False, "skipped": True, "reason": reason or "threshold_not_met"}

    text = (text or "").strip()
    if not text:
        return {"ok": False, "error": "empty_text"}

    try:
        from agentic.registry import registry
    except Exception as e:
        return {"ok": False, "error": f"registry unavailable: {e}"}

    for name in ("post_to_social", "post_job_post_social"):
        spec = registry.get(name)
        if spec is None or spec.handler is None:
            continue
        try:
            # Introspect handler signature to determine compatible arguments
            sig = inspect.signature(spec.handler)
            params = sig.parameters
            args: dict[str, Any] = {}

            # Check which parameter name the handler expects
            if "text" in params:
                args["text"] = text
            elif "message" in params:
                args["message"] = text

            if inspect.iscoroutinefunction(spec.handler):
                try:
                    asyncio.get_running_loop()
                except RuntimeError:
                    # No running loop: safe to use asyncio.run
                    result = asyncio.run(spec.handler(**args))
                else:
                    # Already in a loop: isolate in a dedicated thread
                    import concurrent.futures

                    def _run_isolated():  # type: ignore[no-untyped-def]
                        return asyncio.run(spec.handler(**args))

                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                        result = pool.submit(_run_isolated).result(timeout=60)
            else:
                result = spec.handler(**args)
            # Check if handler returned a dict with ok field (Finding 5)
            if isinstance(result, dict):
                ok = bool(result.get("ok", True))
                return {"ok": ok, "tool": name, "result": result, "reason": reason}
            # Non-dict results are treated as success if truthy
            return {"ok": True, "tool": name, "result": result, "reason": reason}
        except Exception as e:
            log.warning("maybe_post_threads: %s failed: %s", name, e)
            return {"ok": False, "tool": name, "error": str(e)}

    return {
        "ok": False,
        "error": "no_threads_tool",
        "reason": reason,
        "text_preview": text[:200],
    }


def dumps_result(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)
