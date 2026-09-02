"""Bright Data Scraper Studio MCP server (stdio).

Removable package for Aiko research / Scrape-Verse experiments.
Spawn:
  python -m interface.mcp_server.brightdata.server

Env:
  BRIGHT_DATA_API_TOKEN   — required for live calls
  BRIGHT_DATA_COLLECTOR_ID — default c_* collector
  BRIGHT_DATA_API_BASE    — optional (default https://api.brightdata.com)
"""

from __future__ import annotations

import json
import os
from typing import Any

from fastmcp import FastMCP

from interface.mcp_server.brightdata import api_client as bd

mcp = FastMCP("Aiko Bright Data Scraper Studio MCP")


def _ok(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


def _err(exc: Exception) -> str:
    return json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False)


def _parse_inputs(inputs_json: str) -> list[dict[str, Any]]:
    raw = (inputs_json or "").strip()
    if not raw:
        return [{"url": ""}]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"inputs_json must be valid JSON: {e}") from e
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        out = [x for x in data if isinstance(x, dict)]
        if not out:
            raise ValueError("inputs_json list must contain objects")
        return out
    raise ValueError("inputs_json must be an object or array of objects")


@mcp.tool()
def bd_trigger_collect(collector_id: str = "", inputs_json: str = "[{\"url\": \"\"}]") -> str:
    """Queue a Bright Data Scraper Studio batch collection run.

    collector_id: c_* id (default BRIGHT_DATA_COLLECTOR_ID).
    inputs_json: JSON array of input objects matching the collector schema
      (usually [{"url": "https://..."}, ...]).

    Returns collection_id for bd_get_results.
    """
    try:
        inputs = _parse_inputs(inputs_json)
        data = bd.trigger_collect(collector_id, inputs)
        return _ok({"ok": True, **data})
    except Exception as e:
        return _err(e)


@mcp.tool()
def bd_get_results(collection_id: str) -> str:
    """Fetch or poll status for a collection/snapshot id from bd_trigger_collect.

    When ready, Bright Data returns a JSON array of rows; while running, an object.
    """
    try:
        data = bd.get_dataset(collection_id)
        if isinstance(data, list):
            return _ok({"ok": True, "ready": True, "row_count": len(data), "rows": data})
        return _ok({"ok": True, "ready": False, "status": data})
    except Exception as e:
        return _err(e)


@mcp.tool()
def bd_run_collect(
    collector_id: str = "",
    inputs_json: str = "[{\"url\": \"\"}]",
    max_wait_seconds: float = 300.0,
) -> str:
    """Trigger a collector and wait until structured rows are ready (or timeout).

    Preferred one-shot tool for deep research / agent use.
    """
    try:
        inputs = _parse_inputs(inputs_json)
        result = bd.run_collect(
            collector_id,
            inputs,
            max_wait_seconds=float(max_wait_seconds),
        )
        return _ok(result)
    except Exception as e:
        return _err(e)


@mcp.tool()
def bd_self_heal(
    prompt: str,
    collector_id: str = "",
    sample_url: str = "",
    auto_approve: bool = False,
    max_wait_seconds: float = 900.0,
) -> str:
    """Trigger Bright Data Self-Healing when extraction breaks after a site change.

    Self-healing is Bright Data's AI refactor: you describe the break in plain
    language; they rewrite scraper code in place. Collector ID stays the same.

    prompt: e.g. "price returns null after redesign — capture span.price-now"
    sample_url: optional URL to anchor the heal (passed as custom_input).
    auto_approve: if false (default), stops at awaiting_approval for HITL.
      If true, auto-approves, auto-saves, and waits for completion (unattended).

    Returns phase=completed when self-healing is done. After completion,
    re-run bd_run_collect with the same collector_id.
    """
    try:
        custom = None
        if (sample_url or "").strip():
            custom = [{"url": sample_url.strip()}]
        result = bd.run_self_heal(
            collector_id,
            prompt,
            custom_input=custom,
            auto_approve=bool(auto_approve),
            max_wait_seconds=float(max_wait_seconds),
        )
        return _ok(result)
    except Exception as e:
        return _err(e)


@mcp.tool()
def bd_self_heal_progress(collector_id: str = "") -> str:
    """Poll Self-Healing job status for a collector.

    Use after bd_self_heal_approve (or during an in-flight heal) until progress
    status is a terminal value (done/completed/failed). Do not treat approve
    alone as completion.
    """
    try:
        data = bd.self_heal_progress(collector_id)
        status = str(
            (data or {}).get("status") or (data or {}).get("state") or ""
        ).lower() if isinstance(data, dict) else ""
        terminal_ok = status in {"done", "completed", "success", "ready", "finished"}
        terminal_bad = status in {"error", "failed", "canceled", "cancelled"}
        return _ok({
            "ok": True,
            "progress": data,
            "status": status or None,
            "completed": terminal_ok,
            "failed": terminal_bad,
            "terminal": terminal_ok or terminal_bad,
        })
    except Exception as e:
        return _err(e)


@mcp.tool()
def bd_self_heal_approve(
    collector_id: str = "",
    approve: bool = True,
    auto_save: bool = True,
) -> str:
    """Approve or reject a pending Self-Healing diff (HITL gate).

    Call after bd_self_heal returns phase=awaiting_approval.

    This only *submits* the decision. It does NOT mean the heal job is finished.
    Next steps:
      1. Poll bd_self_heal_progress until completed=true (or failed=true)
      2. Then call bd_run_collect with the same collector_id
    """
    try:
        cid = (collector_id or bd.default_collector_id()).strip()
        data = bd.resume_self_heal(
            collector_id,
            approve=bool(approve),
            auto_save=bool(auto_save),
        )
        return _ok({
            "ok": True,
            "phase": "decision_submitted",
            "decision_submitted": True,
            "approval_submitted": bool(approve),
            "completed": False,
            "collector_id": cid or None,
            "resume": data,
            "next_action": (
                "Poll bd_self_heal_progress until completed=true before bd_run_collect"
                if approve
                else "Diff rejected; no further collect expected for this heal"
            ),
        })
    except Exception as e:
        return _err(e)


def main() -> None:
    if not bd._token():
        # Still start so list_tools works; tool calls will error clearly.
        pass
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
