"""Register Layer-1 shared workflow nodes as graph tools.

Importing this module binds handlers from execution.py into the registry
using catalog metadata from config/tools.yaml (with string fallback).
"""

from __future__ import annotations

from agentic.registry import TOOLS, tool
from agentic.workflows.common import execution as _ex


def _spec(name: str, description: str):
    if name in TOOLS:
        return TOOLS[name]
    return name


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
    return _ex.ingest_data(
        sources_json=sources_json,
        filters_json=filters_json,
        parallel=parallel,
        max_items=max_items,
        config_json=config_json,
        state=state,
    )


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
    return _ex.synthesis_data(
        items_json=items_json,
        template=template,
        llm_enriched=llm_enriched,
        per_item=per_item,
        config_json=config_json,
        client=client,
        model=model,
        state=state,
    )


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
