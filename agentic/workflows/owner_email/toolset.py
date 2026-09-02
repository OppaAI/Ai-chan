"""
agentic/workflows/owner_email/toolset.py

Owner email bridge: poll ProtonMail inbox for messages FROM the owner's
address (AIKO_EMAIL), treat the email body as a prompt, generate a
reply via Aiko's LLM, and email the response back.

Polling design (anti-ban):
  - Interval: 10 minutes (6 polls/hour) — well under ProtonMail guard
    per_hour=30 / per_day=100 and matches industrial IMAP polling
    convention (Gmail 10 min, Outlook 10-15 min). Never <2 min.
  - Batch: max 5 messages per poll, list_only=False for full bodies.
  - Idempotency: processed message IDs stored in
    user_state_path("owner_email/processed.json") — skip already replied.
  - Loop guard: never reply to own sent messages; skip if subject starts
    with Re: and body contains our own footer marker.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from system.log import get_logger
from system.userspace import user_state_path

log = get_logger(__name__)

WORKFLOW_ID = "owner_email"
_PROCESSED_FILE = Path(user_state_path("owner_email/processed.json"))
_PROCESSED_IDS: set[str] = set()
_LAST_LOAD = 0.0
_MARKER = "— Aiko via owner-email bridge"


def _owner_email() -> str:
    return (os.getenv("AIKO_EMAIL") or "").strip().lower()


def _load_processed() -> set[str]:
    global _PROCESSED_IDS, _LAST_LOAD
    now = time.monotonic()
    if now - _LAST_LOAD < 30 and _PROCESSED_IDS:
        return _PROCESSED_IDS
    try:
        if _PROCESSED_FILE.is_file():
            data = json.loads(_PROCESSED_FILE.read_text(encoding="utf-8"))
            if isinstance(data, list):
                _PROCESSED_IDS = set(str(x) for x in data)
    except Exception:
        pass
    _LAST_LOAD = now
    return _PROCESSED_IDS


def _save_processed(ids: set[str]) -> None:
    try:
        _PROCESSED_FILE.parent.mkdir(parents=True, exist_ok=True)
        # keep last 500
        trimmed = sorted(ids)[-500:]
        _PROCESSED_FILE.write_text(json.dumps(trimmed, ensure_ascii=False, indent=2), encoding="utf-8")
        _PROCESSED_IDS.clear()
        _PROCESSED_IDS.update(trimmed)
    except Exception as e:
        log.warning("owner_email save processed failed: %s", e)


def _is_from_owner(msg_from: str) -> bool:
    owner = _owner_email()
    if not owner:
        return False
    return owner in (msg_from or "").strip().lower()


def _extract_prompt(msg: dict[str, Any]) -> str:
    # Prefer full body, fallback snippet+subject
    body = (msg.get("body") or msg.get("snippet") or "").strip()
    subject = (msg.get("subject") or "").strip()
    # Strip quoted history and marker to avoid loops
    if _MARKER in body:
        body = body.split(_MARKER)[0].strip()
    # If body is HTML, strip tags lightly
    if "<html" in body.lower() or "<body" in body.lower():
        body = re.sub(r"<[^>]+>", " ", body)
        body = re.sub(r"\s+", " ", body).strip()
    # Use body as prompt; prefix subject if body empty
    if not body and subject:
        return subject
    if subject and subject.lower() not in body.lower()[:80]:
        return f"Subject: {subject}\n\n{body}"
    return body


def check_owner_email(max_results: int = 5, *, state=None) -> str:
    """Graph step: poll inbox for owner's unread commands. Returns JSON."""
    owner = _owner_email()
    if not owner:
        return json.dumps({"ok": False, "error": "AIKO_EMAIL not set"})
    try:
        from agentic.registry import registry
        spec = registry.get("read_protonmail")
        if spec is None or spec.handler is None:
            return json.dumps({"ok": False, "error": "read_protonmail not registered"})
        # list_only False to get bodies; query owner email to reduce noise
        # Some providers ignore query, so filter after.
        result = spec.handler(max_results=max_results, list_only=False, query=owner)
        if not isinstance(result, dict) or not result.get("ok"):
            # fallback: list_only True + query
            result = spec.handler(max_results=max_results, list_only=True, query=owner)
        msgs = result.get("messages") or []
        processed = _load_processed()
        owner_msgs = [m for m in msgs if _is_from_owner(str(m.get("from") or "")) and str(m.get("id") or "") not in processed]
        # Also need full bodies for list_only case
        if owner_msgs and not owner_msgs[0].get("body"):
            full_msgs = []
            for m in owner_msgs:
                mid = str(m.get("id") or "")
                try:
                    full = spec.handler(message_id=mid)
                    if isinstance(full, dict) and full.get("ok") and full.get("body"):
                        # merge
                        m = {**m, "body": full.get("body"), "subject": full.get("subject") or m.get("subject")}
                    full_msgs.append(m)
                except Exception:
                    full_msgs.append(m)
            owner_msgs = full_msgs
        payload = {
            "ok": True,
            "owner": owner,
            "count": len(owner_msgs),
            "messages": [
                {"id": m.get("id"), "from": m.get("from"), "subject": m.get("subject"), "prompt": _extract_prompt(m)[:2000]}
                for m in owner_msgs
            ],
        }
        if state is not None and hasattr(state, "data"):
            state.data["owner_email_batch"] = payload
            state.data["owner_email_messages"] = owner_msgs
        return json.dumps(payload, ensure_ascii=False)
    except Exception as e:
        log.warning("check_owner_email failed: %s", e)
        return json.dumps({"ok": False, "error": str(e)})


def reply_owner_email(report_json: str = "", *, state=None) -> str:
    """Graph step: for each owner message, generate reply via LLM and email back."""
    owner = _owner_email()
    if not owner:
        return json.dumps({"ok": False, "error": "AIKO_EMAIL not set"})
    # Retrieve batch from state or report_json
    batch: dict[str, Any] = {}
    if report_json:
        try:
            batch = json.loads(report_json)
        except Exception:
            batch = {}
    if not batch and state is not None and hasattr(state, "data"):
        batch = dict(state.data.get("owner_email_batch") or {})
        msgs = state.data.get("owner_email_messages") or []
    else:
        msgs = batch.get("messages") or []
        # need full bodies - re-fetch if needed
        if msgs and not any(m.get("body") for m in msgs):
            msgs = batch.get("messages") or []
    # Fallback: if msgs empty but batch has messages with prompt, reconstruct
    if not msgs:
        # try to use prompt field as body
        for m in batch.get("messages") or []:
            msgs.append({"id": m.get("id"), "from": owner, "subject": m.get("subject"), "body": m.get("prompt")})

    if not msgs:
        return json.dumps({"ok": True, "replied": 0, "reason": "no_owner_messages"})

    # Resolve think/LLM for generation
    think = None
    try:
        # try to get think from schedule's memorize reference via state
        if state is not None and hasattr(state, "data"):
            # state may carry memorize
            mem = state.data.get("_memorize") if isinstance(state.data, dict) else None
            if mem is not None and hasattr(mem, "_think"):
                think = getattr(mem, "_think", None)
    except Exception:
        pass
    # fallback: try global AikoThink singleton via cognition
    if think is None:
        try:
            from cognition.think import AikoThink  # type: ignore
            # Not instantiated; will fallback to direct OpenAI
            think = None
        except Exception:
            think = None

    from agentic.registry import registry
    send_spec = registry.get("send_protonmail")
    if send_spec is None or send_spec.handler is None:
        return json.dumps({"ok": False, "error": "send_protonmail not registered"})

    processed = _load_processed()
    replied = 0
    errors = []
    for m in msgs:
        mid = str(m.get("id") or "")
        if not mid or mid in processed:
            continue
        prompt = _extract_prompt(m) if "body" in m else str(m.get("prompt") or "")
        if not prompt or len(prompt.strip()) < 2:
            continue
        # Guard: skip our own auto-replies (subject Re: + marker)
        subj = str(m.get("subject") or "")
        if subj.lower().startswith("re:") and _MARKER in prompt:
            processed.add(mid)
            continue
        try:
            # Generate reply
            answer = ""
            if think is not None and hasattr(think, "chat"):
                try:
                    answer = think.chat(prompt, skip_memory=False, store_turn=True)  # type: ignore
                except Exception as e:
                    log.warning("owner_email think.chat failed: %s", e)
                    answer = ""
            if not answer:
                # Fallback: direct OpenAI call
                from openai import OpenAI
                import os as _os
                base = _os.getenv("LLM_BASE_URL", "http://127.0.0.1:11434/v1")
                model = _os.getenv("LLM_MODEL", "qwen2.5:7b")
                client = OpenAI(base_url=base, api_key="not-needed")
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=800,
                    temperature=0.7,
                )
                answer = (resp.choices[0].message.content or "").strip() if resp.choices else ""
            if not answer:
                answer = "(No response generated.)"
            # Append marker to avoid loops
            body = answer + f"\n\n{_MARKER}\nOriginal subject: {subj[:120]}"
            subject = f"Re: {subj[:120]}" if subj else "Re: your message to Aiko"
            if len(subject) > 120:
                subject = subject[:120]
            res = send_spec.handler(recipients=[owner], subject=subject, body=body)
            if isinstance(res, dict) and res.get("ok") is False:
                errors.append(str(res.get("error") or res))
                continue
            processed.add(mid)
            replied += 1
            _save_processed(processed)
            # Small delay to avoid burst
            time.sleep(0.5)
        except Exception as e:
            errors.append(str(e))
            log.warning("reply_owner_email msg %s failed: %s", mid[:8], e)

    _save_processed(processed)
    return json.dumps({"ok": True, "replied": replied, "errors": errors[:3]}, ensure_ascii=False)


__all__ = ["check_owner_email", "reply_owner_email"]
