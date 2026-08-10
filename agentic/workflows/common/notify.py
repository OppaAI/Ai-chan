"""Shared notification helpers (email + optional Threads).

Best-effort: uses registered tools when present; never raises to callers.
"""

from __future__ import annotations

import json
import logging
from typing import Any

log = logging.getLogger(__name__)


def notify_email(subject: str, body: str, *, to: str | None = None) -> dict[str, Any]:
    """Send an email via a registered mail tool if available.

    Tries common tool names: send_email, send_protonmail.
    """
    try:
        from agentic.registry import registry
    except Exception as e:
        return {"ok": False, "error": f"registry unavailable: {e}"}

    for name in ("send_email", "send_protonmail", "compose_email"):
        spec = registry.get(name)
        if spec is None or spec.handler is None:
            continue
        try:
            args: dict[str, Any] = {"subject": subject, "body": body}
            if to:
                args["to"] = to
            result = spec.handler(**args)
            if isinstance(result, dict):
                return {"ok": bool(result.get("ok", True)), "tool": name, "result": result}
            return {"ok": True, "tool": name, "result": result}
        except TypeError:
            # Try alternate signatures
            try:
                result = spec.handler(subject=subject, body=body)
                return {"ok": True, "tool": name, "result": result}
            except Exception as e2:
                log.debug("notify_email: %s failed: %s", name, e2)
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
            result = spec.handler(text=text)
            return {"ok": True, "tool": name, "result": result, "reason": reason}
        except TypeError:
            try:
                result = spec.handler(message=text)
                return {"ok": True, "tool": name, "result": result, "reason": reason}
            except Exception as e2:
                log.debug("maybe_post_threads: %s failed: %s", name, e2)
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
