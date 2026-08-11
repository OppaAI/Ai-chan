"""Register Layer-1 shared workflow nodes as graph tools.

Importing this module binds handlers into the registry using catalog
metadata from config/tools.yaml (with string fallback).

Also hosts small Layer-1 adapters (e.g. aurora) so domain checks can run
through ingest_data without bloating execution.py.
"""

from __future__ import annotations

import json
from typing import Any

from agentic.registry import TOOLS, tool
from agentic.workflows.common import execution as _ex


def _spec(name: str, description: str):
    if name in TOOLS:
        return TOOLS[name]
    return name


def _loads(raw: str | dict | list | None, default: Any = None) -> Any:
    if raw is None or raw == "":
        return default if default is not None else {}
    if isinstance(raw, (dict, list)):
        return raw
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return default if default is not None else {}


def _expand_aurora_sources(sources_json: str, state=None) -> tuple[str, list[dict[str, Any]]]:
    """Resolve adapter:aurora sources into concrete items; leave other sources."""
    sources = _loads(sources_json, [])
    if isinstance(sources, dict):
        sources = sources.get("sources") or []
    if not isinstance(sources, list):
        sources = []

    remaining: list[dict[str, Any]] = []
    pre_items: list[dict[str, Any]] = []

    for src in sources:
        if not isinstance(src, dict):
            continue
        stype = str(src.get("type") or "").strip().lower()
        name = str(src.get("name") or src.get("adapter") or src.get("id") or "").strip().lower()
        if stype == "adapter" and name in {"aurora", "aurora_forecast"}:
            try:
                from agentic.workflows.aurora_forecast.toolset import check_aurora

                raw = check_aurora(str(src.get("config_path") or ""), state=state)
                payload = json.loads(raw) if isinstance(raw, str) else raw
                if isinstance(payload, dict) and not (
                    payload.get("error") and "location_name" not in payload
                ):
                    item = dict(payload)
                    sid = str(src.get("id") or "aurora")
                    item.setdefault("id", sid)
                    item.setdefault("source", sid)
                    item.setdefault("type", "adapter")
                    item.setdefault("text", str(payload.get("summary") or ""))
                    pre_items.append(item)
                else:
                    remaining.append(src)
            except Exception:
                remaining.append(src)
        else:
            remaining.append(src)

    return json.dumps(remaining, ensure_ascii=False), pre_items


def _merge_pre_items(result_json: str, pre_items: list[dict[str, Any]], state=None, max_items: int = 50) -> str:
    if not pre_items:
        return result_json
    data = _loads(result_json, {})
    if not isinstance(data, dict):
        data = {"ok": True, "items": [], "meta": {}}
    items = list(data.get("items") or [])
    items = pre_items + items
    # Enforce max_items limit after prepending
    items = items[:max_items]
    data["items"] = items
    data["ok"] = True if items else data.get("ok", True)
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    meta["pre_expanded"] = ["aurora"]
    data["meta"] = meta
    if state is not None and hasattr(state, "data") and isinstance(state.data, dict):
        state.data["ingest_items"] = items
    return json.dumps(data, ensure_ascii=False)


def _promote_synth_fields(result_json: str, state=None) -> str:
    """Copy level/kp/viewable onto each result for verify/output rules."""
    data = _loads(result_json, {})
    if not isinstance(data, dict):
        return result_json
    results = data.get("results")
    if not isinstance(results, list):
        return result_json
    keys = (
        "level",
        "kp_index",
        "viewable",
        "summary",
        "location_name",
        "aurora_probability_pct",
        "cloud_cover_pct",
        "is_night",
    )
    out = []
    for r in results:
        if not isinstance(r, dict):
            out.append(r)
            continue
        row = dict(r)
        item = row.get("item") if isinstance(row.get("item"), dict) else {}
        for k in keys:
            if k not in row and k in item:
                row[k] = item[k]
        out.append(row)
    data["results"] = out
    if state is not None and hasattr(state, "data") and isinstance(state.data, dict):
        state.data["synth_results"] = out
    return json.dumps(data, ensure_ascii=False)


@tool(
    _spec("ingest_data", "Layer-1 shared node: ingest sources into items[]"),
    description="Layer-1 shared node: ingest sources into items[].",
    graph=True,
    react=False,
    domain="workflow",
)
def ingest_data(
    sources_json: str = "[]",
    filters_json: str = "{}",
    parallel: str = "true",
    max_items: str = "50",
    config_json: str = "{}",
    *,
    state=None,
) -> str:
    remaining_json, pre_items = _expand_aurora_sources(sources_json, state=state)
    result = _ex.ingest_data(
        sources_json=remaining_json,
        filters_json=filters_json,
        parallel=parallel,
        max_items=max_items,
        config_json=config_json,
        state=state,
    )
    try:
        limit = max(1, int(max_items or 50))
    except (TypeError, ValueError):
        limit = 50
    return _merge_pre_items(result, pre_items, state=state, max_items=limit)


@tool(
    _spec("store_data", "Layer-1 shared node: persist items to TTL store"),
    description="Layer-1 shared node: persist items to TTL store.",
    graph=True,
    react=False,
    domain="workflow",
)
def store_data(
    workflow_id: str = "",
    items_json: str = "",
    mode: str = "append",
    retain_days: str = "3",
    config_json: str = "{}",
    *,
    state=None,
) -> str:
    return _ex.store_data(
        workflow_id=workflow_id,
        items_json=items_json,
        mode=mode,
        retain_days=retain_days,
        config_json=config_json,
        state=state,
    )


@tool(
    _spec("synthesis_data", "Layer-1 shared node: template fill from items"),
    description="Layer-1 shared node: template fill from items.",
    graph=True,
    react=False,
    domain="workflow",
)
def synthesis_data(
    items_json: str = "",
    template: str = "",
    llm_enriched: str = "false",
    per_item: str = "true",
    config_json: str = "{}",
    *,
    client=None,
    model: str | None = None,
    state=None,
) -> str:
    result = _ex.synthesis_data(
        items_json=items_json,
        template=template,
        llm_enriched=llm_enriched,
        per_item=per_item,
        config_json=config_json,
        client=client,
        model=model,
        state=state,
    )
    return _promote_synth_fields(result, state=state)


@tool(
    _spec("verify_results", "Layer-1 shared node: HITL or auto-pass verify"),
    description="Layer-1 shared node: HITL or auto-pass verify.",
    graph=True,
    react=False,
    domain="workflow",
)
def verify_results(
    results_json: str = "",
    human_in_the_loop: str = "false",
    llm_verify: str = "false",
    auto_pass_json: str = "{}",
    config_json: str = "{}",
    *,
    state=None,
) -> str:
    return _ex.verify_results(
        results_json=results_json,
        human_in_the_loop=human_in_the_loop,
        llm_verify=llm_verify,
        auto_pass_json=auto_pass_json,
        config_json=config_json,
        state=state,
    )


@tool(
    _spec("output_user_results", "Layer-1 shared node: email / social output"),
    description="Layer-1 shared node: email / social output.",
    graph=True,
    react=False,
    domain="workflow",
)
def output_user_results(
    results_json: str = "",
    email_json: str = "{}",
    social_json: str = "[]",
    config_json: str = "{}",
    *,
    state=None,
) -> str:
    return _ex.output_user_results(
        results_json=results_json,
        email_json=email_json,
        social_json=social_json,
        config_json=config_json,
        state=state,
    )


__all__ = [
    "ingest_data",
    "store_data",
    "synthesis_data",
    "verify_results",
    "output_user_results",
]
