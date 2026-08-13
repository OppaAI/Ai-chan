"""HTTP client for Bright Data Scraper Studio Collection + Self-Healing APIs.

Collection (get data):
  POST /dca/trigger?collector=c_...
  GET  /dca/dataset?id=j_...   (poll until JSON array)

Self-healing (fix scraper in place, same collector id):
  POST /dca/collectors/{c_*}/refactor_template
  GET  /dca/collectors/{c_*}/refactor_template/progress
  POST /dca/collectors/{c_*}/resume_automation_job

Self-healing is NOT automatic when a site changes — you (or Aiko) must
trigger it with a plain-language prompt when extraction goes empty/wrong.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import requests

BASE = os.getenv("BRIGHT_DATA_API_BASE", "https://api.brightdata.com").rstrip("/")


def _validate_collector_id(cid: str) -> str:
    """Validate and return a safe collector ID.

    Requires c_ prefix and rejects path/query/fragment separators to prevent URL injection.
    """
    cid = cid.strip()
    if not cid:
        raise ValueError("collector_id required (or set BRIGHT_DATA_COLLECTOR_ID)")
    if not cid.startswith("c_"):
        raise ValueError("collector_id must start with 'c_' prefix")
    # Reject path separators, query strings, fragments, and encoded variants
    dangerous_chars = ['/', '?', '#', '%2F', '%2f', '%3F', '%3f', '%23']
    for char in dangerous_chars:
        if char in cid:
            raise ValueError(f"collector_id contains invalid character(s): {char}")
    return cid


def _token() -> str:
    return (
        os.getenv("BRIGHT_DATA_API_TOKEN", "").strip()
        or os.getenv("BRIGHTDATA_API_TOKEN", "").strip()
        or os.getenv("BRIGHT_DATA_API_KEY", "").strip()
    )


def _headers() -> dict[str, str]:
    tok = _token()
    if not tok:
        raise ValueError(
            "BRIGHT_DATA_API_TOKEN is not set — create a key in Bright Data "
            "account settings and export it before calling this MCP server"
        )
    return {
        "Authorization": f"Bearer {tok}",
        "Content-Type": "application/json",
    }


def default_collector_id() -> str:
    return os.getenv("BRIGHT_DATA_COLLECTOR_ID", "").strip()


def trigger_collect(
    collector_id: str,
    inputs: list[dict[str, Any]],
    *,
    timeout: float = 60.0,
) -> dict[str, Any]:
    """Queue a batch collection run. Returns collection_id / snapshot id."""
    cid = (collector_id or default_collector_id()).strip()
    if not cid:
        raise ValueError("collector_id required (or set BRIGHT_DATA_COLLECTOR_ID)")
    if not inputs:
        raise ValueError("inputs must be a non-empty list of objects")

    url = f"{BASE}/dca/trigger"
    resp = requests.post(
        url,
        params={"collector": cid},
        headers=_headers(),
        json=inputs,
        timeout=timeout,
    )
    if not (200 <= resp.status_code < 300):
        raise RuntimeError(f"trigger failed HTTP {resp.status_code}: {resp.text[:800]}")
    data = resp.json() if resp.content else {}
    if not isinstance(data, dict):
        raise RuntimeError(f"unexpected trigger response: {data!r}")
    return data


def get_dataset(
    snapshot_id: str,
    *,
    timeout: float = 60.0,
) -> Any:
    """Fetch batch results. While running, API returns an object; when ready, a JSON array."""
    sid = (snapshot_id or "").strip()
    if not sid:
        raise ValueError("snapshot_id / collection_id required")
    url = f"{BASE}/dca/dataset"
    resp = requests.get(
        url,
        params={"id": sid},
        headers=_headers(),
        timeout=timeout,
    )
    if not (200 <= resp.status_code < 300):
        raise RuntimeError(f"dataset failed HTTP {resp.status_code}: {resp.text[:800]}")
    try:
        return resp.json()
    except json.JSONDecodeError:
        return {"raw": resp.text[:4000]}


def run_collect(
    collector_id: str,
    inputs: list[dict[str, Any]],
    *,
    poll_seconds: float = 5.0,
    max_wait_seconds: float = 300.0,
) -> dict[str, Any]:
    """Trigger + poll until results are a list (or timeout)."""
    trigger = trigger_collect(collector_id, inputs)
    snapshot_id = str(
        trigger.get("collection_id")
        or trigger.get("snapshot_id")
        or trigger.get("id")
        or ""
    ).strip()
    if not snapshot_id:
        return {"ok": False, "error": "no collection_id in trigger response", "trigger": trigger}

    deadline = time.monotonic() + max(10.0, max_wait_seconds)
    last: Any = None
    while time.monotonic() < deadline:
        last = get_dataset(snapshot_id)
        if isinstance(last, list):
            return {
                "ok": True,
                "collection_id": snapshot_id,
                "collector_id": collector_id or default_collector_id(),
                "row_count": len(last),
                "rows": last,
            }
        # still running / status object
        time.sleep(max(1.0, poll_seconds))

    return {
        "ok": False,
        "error": "timed out waiting for dataset",
        "collection_id": snapshot_id,
        "last_status": last,
    }


def trigger_self_heal(
    collector_id: str,
    prompt: str,
    *,
    custom_input: list[dict[str, Any]] | None = None,
    timeout: float = 60.0,
) -> dict[str, Any]:
    """Start Self-Healing refactor (plain-language fix, same collector id)."""
    cid = _validate_collector_id(collector_id or default_collector_id())
    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("prompt required (max 1000 chars)")
    if len(prompt) > 1000:
        raise ValueError(f"prompt exceeds maximum length of 1000 characters (got {len(prompt)})")

    url = f"{BASE}/dca/collectors/{cid}/refactor_template"
    body: dict[str, Any] = {"prompt": prompt}
    if custom_input is not None:
        body["custom_input"] = custom_input
    else:
        body["custom_input"] = [{}]

    resp = requests.post(url, headers=_headers(), json=body, timeout=timeout)
    if not (200 <= resp.status_code < 300):
        raise RuntimeError(f"self-heal trigger failed HTTP {resp.status_code}: {resp.text[:800]}")
    data = resp.json() if resp.content else {}
    return data if isinstance(data, dict) else {"raw": data}


def self_heal_progress(
    collector_id: str,
    *,
    timeout: float = 60.0,
) -> dict[str, Any]:
    """Poll Self-Healing job progress until ready / pending_answer / error."""
    cid = _validate_collector_id(collector_id or default_collector_id())
    url = f"{BASE}/dca/collectors/{cid}/refactor_template/progress"
    resp = requests.get(url, headers=_headers(), timeout=timeout)
    if not (200 <= resp.status_code < 300):
        raise RuntimeError(f"self-heal progress failed HTTP {resp.status_code}: {resp.text[:800]}")
    data = resp.json() if resp.content else {}
    return data if isinstance(data, dict) else {"raw": data}


def resume_self_heal(
    collector_id: str,
    *,
    approve: bool = True,
    auto_save: bool = True,
    timeout: float = 60.0,
) -> dict[str, Any]:
    """Approve or reject a pending Self-Healing diff."""
    cid = _validate_collector_id(collector_id or default_collector_id())
    url = f"{BASE}/dca/collectors/{cid}/resume_automation_job"
    body = {"message": bool(approve), "auto_save": bool(auto_save) and bool(approve)}
    resp = requests.post(url, headers=_headers(), json=body, timeout=timeout)
    if not (200 <= resp.status_code < 300):
        raise RuntimeError(f"resume self-heal failed HTTP {resp.status_code}: {resp.text[:800]}")
    data = resp.json() if resp.content else {}
    return data if isinstance(data, dict) else {"raw": data}


def run_self_heal(
    collector_id: str,
    prompt: str,
    *,
    custom_input: list[dict[str, Any]] | None = None,
    auto_approve: bool = False,
    poll_seconds: float = 10.0,
    max_wait_seconds: float = 900.0,
) -> dict[str, Any]:
    """Full self-heal loop: trigger → poll → optional auto-approve.

    Self-heal can take up to ~15 minutes on Bright Data side.
    Default is HITL: stop at pending_answer so a human (or agent) reviews.
    Set auto_approve=True only when you trust unattended commits.
    """
    started = trigger_self_heal(collector_id, prompt, custom_input=custom_input)
    deadline = time.monotonic() + max(30.0, max_wait_seconds)
    last: dict[str, Any] = {}

    while time.monotonic() < deadline:
        last = self_heal_progress(collector_id)
        status = str(last.get("status") or last.get("state") or "").lower()
        if status in {"pending_answer", "awaiting_approval", "waiting_for_approval"}:
            if auto_approve:
                resume = resume_self_heal(collector_id, approve=True, auto_save=True)
                # Continue polling until completion after auto-approval
                while time.monotonic() < deadline:
                    last = self_heal_progress(collector_id)
                    status = str(last.get("status") or last.get("state") or "").lower()
                    if status in {"done", "completed", "success", "ready", "finished"}:
                        return {
                            "ok": True,
                            "phase": "completed",
                            "collector_id": collector_id or default_collector_id(),
                            "trigger": started,
                            "progress": last,
                            "resume": resume,
                        }
                    if status in {"error", "failed", "canceled", "cancelled"}:
                        return {
                            "ok": False,
                            "phase": "failed",
                            "collector_id": collector_id or default_collector_id(),
                            "trigger": started,
                            "progress": last,
                            "resume": resume,
                        }
                    time.sleep(max(2.0, poll_seconds))
                # Timeout after approval
                return {
                    "ok": False,
                    "phase": "timeout",
                    "collector_id": collector_id or default_collector_id(),
                    "trigger": started,
                    "progress": last,
                    "resume": resume,
                    "error": "timed out waiting for self-heal completion after approval",
                }
            return {
                "ok": True,
                "phase": "awaiting_approval",
                "collector_id": collector_id or default_collector_id(),
                "message": "Review proposed diff, then call bd_self_heal_approve",
                "trigger": started,
                "progress": last,
            }
        if status in {"done", "completed", "success", "ready", "finished"}:
            return {
                "ok": True,
                "phase": "completed",
                "collector_id": collector_id or default_collector_id(),
                "trigger": started,
                "progress": last,
            }
        if status in {"error", "failed", "canceled", "cancelled"}:
            return {
                "ok": False,
                "phase": "failed",
                "collector_id": collector_id or default_collector_id(),
                "trigger": started,
                "progress": last,
            }
        time.sleep(max(2.0, poll_seconds))

    return {
        "ok": False,
        "phase": "timeout",
        "collector_id": collector_id or default_collector_id(),
        "trigger": started,
        "progress": last,
        "error": "timed out waiting for self-heal job",
    }
