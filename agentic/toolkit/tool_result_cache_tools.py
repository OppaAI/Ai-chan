"""Graph-registered wrappers for unified tool-result cache."""
from __future__ import annotations

from agentic.registry import tool
from agentic.toolkit.tool_result_cache import (
    SELECT_DEFAULT_LIMIT,
    DEFAULT_RETENTION_RUNS,
    cache_write,
    cache_select,
    cache_read,
    cache_gc,
)


@tool(
    "cache_write",
    description=(
        "Append tool results to the unified on-disk JSONL tool-result cache. "
        "Use after fetch/search so full payloads stay out of the LLM context. "
        "Pass items directly or from_state to read a GraphState key."
    ),
    props={
        "items": {"type": "array", "description": "List of result objects or strings"},
        "workflow": {"type": "string", "description": "Workflow id, e.g. lane_d_job_hunt"},
        "source": {"type": "string", "description": "rss|email|web|tool"},
        "run_id": {"type": "string"},
        "from_state": {"type": "string", "description": "GraphState key holding items"},
    },
    required=["workflow"],
    domain="cache",
    react=False,
    graph=True,
)
def cache_write_tool(
    workflow: str,
    items: list | dict | str | None = None,
    source: str = "tool",
    run_id: str | None = None,
    from_state: str | None = None,
    state=None,
) -> dict:
    return cache_write(
        items if items is not None else [],
        workflow=workflow,
        source=source,
        run_id=run_id,
        state=state,
        from_state=from_state,
    )


@tool(
    "cache_select",
    description=(
        "Select a compact ranked subset from the tool-result cache for LLM synthesis. "
        "Only this slice should enter context — not the full JSONL."
    ),
    props={
        "workflow": {"type": "string"},
        "source": {"type": "string"},
        "run_id": {"type": "string"},
        "keywords": {"type": "array", "items": {"type": "string"}},
        "matched_only": {"type": "boolean"},
        "limit": {"type": "integer"},
        "to_state": {"type": "string"},
    },
    required=["workflow"],
    domain="cache",
    react=False,
    graph=True,
)
def cache_select_tool(
    workflow: str,
    source: str | None = None,
    run_id: str | None = None,
    keywords: list | str | None = None,
    matched_only: bool = True,
    limit: int = SELECT_DEFAULT_LIMIT,
    to_state: str = "selection",
    state=None,
) -> dict:
    return cache_select(
        workflow=workflow,
        source=source,
        run_id=run_id,
        keywords=keywords,
        matched_only=matched_only,
        limit=limit,
        state=state,
        to_state=to_state,
    )


@tool(
    "cache_read",
    description="Read records from the tool-result cache (debug / drill-down).",
    props={
        "workflow": {"type": "string"},
        "source": {"type": "string"},
        "run_id": {"type": "string"},
        "limit": {"type": "integer"},
    },
    required=["workflow"],
    domain="cache",
    react=False,
    graph=True,
)
def cache_read_tool(
    workflow: str,
    source: str | None = None,
    run_id: str | None = None,
    limit: int = 500,
) -> dict:
    return cache_read(workflow=workflow, source=source, run_id=run_id, limit=limit)


@tool(
    "cache_gc",
    description="Drop old tool-result cache runs for a workflow (retention).",
    props={
        "workflow": {"type": "string"},
        "keep_runs": {"type": "integer"},
    },
    required=["workflow"],
    domain="cache",
    react=False,
    graph=True,
)
def cache_gc_tool(workflow: str, keep_runs: int = DEFAULT_RETENTION_RUNS) -> dict:
    return cache_gc(workflow=workflow, keep_runs=keep_runs)
