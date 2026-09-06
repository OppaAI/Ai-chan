"""
agentic/toolkit/needle_team.py

Needle2 multi-agent delegation as first-class graph/ReAct tools.

Aiko's rule (unchanged): Needle proposes, Aiko disposes. Every Needle
response is validated against the capability-filtered tool subset and
executed only through Aiko's registry + approval gates. Needle never
runs tools directly.

Jetson Orin Nano 8GB tuning:
  - Default max 2 workers (NEEDLE_MAX_WORKERS caps at 4; on 8GB keep 2).
  - 15s per-worker timeout, 0.85 confidence threshold — low-confidence
    or unavailable workers fall back to the main LLM with a clear note.
  - Prompts are short (task capped at 1500 chars) so small local Needle
    servers stay responsive.

Tools:
  needle_team_run     Fan out one task to all configured workers, merge.
  needle_team_status  Show configured workers (roles + tools, no URLs/keys).
"""

from __future__ import annotations

import os

from agentic.registry import TOOLS, tool
from agentic.toolkit.common import json_block
from system.log import get_logger

log = get_logger(__name__)


def _spec(name: str, description: str):
    return TOOLS[name] if name in TOOLS else name


def _worker_summary(w) -> dict:
    return {"id": w.id, "role": w.role, "allowed_tools": list(w.allowed_tools),
            "confidence_threshold": w.confidence_threshold, "timeout": w.timeout}


@tool(
    _spec("needle_team_status", "Show Needle2 worker config (roles/tools, no secrets)."),
    description="Show Needle2 worker config (roles/tools, no secrets).",
    graph=True,
    react=True,
    domain="multi_agent",
)
def needle_team_status() -> str:
    """Read-only config probe. Never leaks base_urls."""
    try:
        from agentic.needle_orchestrator import load_needle_workers
        from agentic.needle import NeedleError

        raw = os.getenv("NEEDLE_WORKERS", "")
        if not raw.strip():
            return json_block("needle_team_status", {
                "ok": True, "configured": False, "workers": [],
                "hint": 'Set NEEDLE_WORKERS JSON array, e.g. [{"id":"research","role":"researcher",'
                        '"base_url":"http://127.0.0.1:8082","allowed_tools":["adaptive_search"]}]',
                "backend": os.getenv("AGENT_REACT_BACKEND", "openai"),
            })
        try:
            max_w = int(os.getenv("NEEDLE_MAX_WORKERS", "4"))
        except ValueError:
            max_w = 4
        workers = load_needle_workers(raw, default_timeout=float(os.getenv("NEEDLE_TIMEOUT", "15")),
                                      default_confidence_threshold=float(os.getenv("NEEDLE_CONFIDENCE_THRESHOLD", "0.85")),
                                      max_workers=max_w)
        return json_block("needle_team_status", {
            "ok": True, "configured": True, "count": len(workers),
            "workers": [_worker_summary(w) for w in workers],
            "backend": os.getenv("AGENT_REACT_BACKEND", "openai"),
            "note": "Needle proposes tool calls only; Aiko validates + executes.",
        })
    except Exception as e:
        return json_block("needle_team_status", {"ok": False, "error": str(e)[:300]})


@tool(
    _spec("needle_team_run", "Fan out a task to Needle2 workers and merge proposals."),
    description="Fan out a task to Needle2 workers and merge proposals.",
    graph=True,
    react=True,
    domain="multi_agent",
)
def needle_team_run(task: str = "", tools_hint: str = "") -> str:
    """One task -> all workers concurrently -> merged proposal list.

    tools_hint: optional comma-separated tool names to further restrict
    the subset handed to workers (intersected with each worker's
    allow-list AND the caller-provided subset when invoked from ReAct).
    From a graph node, pass "" to use each worker's full allow-list.

    Returns per-worker {response|error} — the CALLER (ReAct/graph) must
    still validate + execute via dispatch_tool. This tool itself never
    executes anything.
    """
    try:
        from agentic.needle import NeedleError
        from agentic.needle_orchestrator import NeedleOrchestrator, load_needle_workers
        from agentic.registry import registry

        task = (task or "").strip()[:1500]
        if not task:
            return json_block("needle_team_run", {"ok": False, "error": "task required"})
        raw = os.getenv("NEEDLE_WORKERS", "")
        if not raw.strip():
            return json_block("needle_team_run", {
                "ok": False, "error": "no Needle workers configured",
                "hint": "Set NEEDLE_WORKERS or call needle_team_status for the format.",
                "fallback": "Use the main LLM (AGENT_REACT_BACKEND=openai) for this turn.",
            })
        try:
            max_w = int(os.getenv("NEEDLE_MAX_WORKERS", "4"))
            timeout = float(os.getenv("NEEDLE_TIMEOUT", "15"))
            conf = float(os.getenv("NEEDLE_CONFIDENCE_THRESHOLD", "0.85"))
        except ValueError as e:
            return json_block("needle_team_run", {"ok": False, "error": f"bad Needle env: {e}"})
        try:
            workers = load_needle_workers(raw, default_timeout=timeout,
                                          default_confidence_threshold=conf, max_workers=max_w)
        except NeedleError as e:
            return json_block("needle_team_run", {"ok": False, "error": str(e)[:300]})

        # Build the capability-aware schema subset: all react tools, then
        # optionally narrow by tools_hint.
        schemas = [spec.to_openai_schema() for spec in registry.all_specs() if spec.react]
        hint = {t.strip() for t in (tools_hint or "").split(",") if t.strip()}
        if hint:
            schemas = [s for s in schemas if s.get("function", {}).get("name") in hint]

        orchestrator = NeedleOrchestrator(workers)
        try:
            results = orchestrator.complete(task, schemas)
        except NeedleError as e:
            return json_block("needle_team_run", {"ok": False, "error": str(e)[:400],
                                                  "fallback": "Use the main LLM for this turn."})
        merged = []
        for r in results:
            if r.response is not None:
                merged.append({"worker_id": r.worker_id, "role": r.role, "ok": True,
                               "kind": r.response.kind, "confidence": r.response.confidence,
                               "reasoning": (r.response.reasoning or "")[:500],
                               "content": (r.response.content or "")[:1500],
                               "calls": [{"name": c.name, "arguments": c.arguments} for c in r.response.calls]})
            else:
                merged.append({"worker_id": r.worker_id, "role": r.role, "ok": False, "error": r.error[:300]})
        ok_any = any(m.get("ok") for m in merged)
        return json_block("needle_team_run", {"ok": ok_any, "task": task[:300], "workers": merged,
                                              "note": "Proposals only — validate + execute via Aiko's registry."})
    except Exception as e:
        log.warning("needle_team_run failed: %s", e)
        return json_block("needle_team_run", {"ok": False, "error": str(e)[:300]})
    

__all__ = ["needle_team_status", "needle_team_run"]
