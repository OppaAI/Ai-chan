"""Graph-registered wrappers for unified tool-result cache."""
from __future__ import annotations

import json
import os
from typing import Any

from agentic.registry import tool
from agentic.toolkit.tool_result_cache import (
    SELECT_DEFAULT_LIMIT,
    DEFAULT_RETENTION_RUNS,
    cache_write,
    cache_select,
    cache_read,
    cache_gc,
)

# Must stay <= graph_engine._substitute $result: slice (currently 4000).
# Prefer a shared env so both sides can be raised together later.
GRAPH_RESULT_SUBSTITUTE_MAX_CHARS = int(
    os.getenv("GRAPH_RESULT_SUBSTITUTE_MAX_CHARS", "4000")
)


def _selection_to_evidence(selection: list[dict[str, Any]]) -> str:
    """Format compact selection as plain text for synthesize_report evidence.

    Tracks cumulative character budget and only includes whole records that fit
    within GRAPH_RESULT_SUBSTITUTE_MAX_CHARS so graph_engine._substitute does not
    silently truncate mid-set. Preserves record boundaries and ranking order.
    """
    blocks: list[str] = []
    total_chars = 0
    separator_chars = 2  # "\n\n" between blocks
    budget = max(1, GRAPH_RESULT_SUBSTITUTE_MAX_CHARS)

    for i, r in enumerate(selection or [], 1):
        title = (r.get("title") or "").strip()
        body = (r.get("body") or "").strip()
        url = (r.get("url") or "").strip()
        head = f"{i}. {title}" if title else f"{i}."
        parts = [head]
        if body:
            parts.append(body)
        if url:
            parts.append(url)

        block = "\n".join(parts)
        block_chars = len(block)
        needed_chars = block_chars + (separator_chars if blocks else 0)
        if total_chars + needed_chars > budget:
            break

        blocks.append(block)
        total_chars += needed_chars

    return "\n\n".join(blocks)


@tool(
    "cache_write",
    description=(
        "Append tool results to the unified on-disk JSONL tool-result cache. "
        "Use after fetch/search so full payloads stay out of the LLM context. "
        "Pass items (string or list) or from_state to read a GraphState key."
    ),
    props={
        "items": {"description": "Result object(s) or text from prior node ($result:...)"},
        "workflow": {"type": "string", "description": "Workflow id, e.g. research_and_report"},
        "source": {"type": "string", "description": "rss|email|web|kb|tool"},
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
        "Returns plain-text evidence (not full JSONL) for synthesize_report."
    ),
    props={
        "workflow": {"type": "string"},
        "source": {"type": "string"},
        "run_id": {"type": "string"},
        "keywords": {"type": "array", "items": {"type": "string"}},
        "keywords_env": {"type": "string", "description": "Environment variable name containing comma-separated keywords"},
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
    keywords_env: str | None = None,
    matched_only: bool = True,
    limit: int = SELECT_DEFAULT_LIMIT,
    to_state: str = "selection",
    state=None,
) -> str:
    # Resolve keywords from environment variable if keywords_env is provided
    resolved_keywords = keywords
    if keywords_env:
        env_value = os.getenv(keywords_env, "").strip()
        if env_value:
            resolved_keywords = env_value

    out = cache_select(
        workflow=workflow,
        source=source,
        run_id=run_id,
        keywords=resolved_keywords,
        matched_only=matched_only,
        limit=limit,
        state=state,
        to_state=to_state,
    )
    return _selection_to_evidence(list(out.get("selection") or []))


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
) -> str:
    out = cache_read(workflow=workflow, source=source, run_id=run_id, limit=limit)
    return json.dumps({"count": out.get("count"), "records": out.get("records")}, ensure_ascii=False)


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
def cache_gc_tool(workflow: str, keep_runs: int = DEFAULT_RETENTION_RUNS) -> str:
    out = cache_gc(workflow=workflow, keep_runs=keep_runs)
    return json.dumps(out, ensure_ascii=False)
