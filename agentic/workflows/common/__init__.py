"""Shared primitives for agentic workflows (job_hunt, aurora_forecast, …).

Layer 0 stack:
  Spec → Graph → Nodes (execution.py) → Toolsets (registry/MCP)

Helpers: config, store, notify, graph registry.
"""

from agentic.workflows.common.config import load_workflow_config, resolve_config_value
from agentic.workflows.common.execution import (
    ingest_data,
    output_user_results,
    store_data,
    synthesis_data,
    verify_results,
)
from agentic.workflows.common.graphs import get_graph, list_graphs, register_graph
from agentic.workflows.common.notify import maybe_post_threads, notify_email
from agentic.workflows.common.store import (
    append_record,
    load_records,
    prune_records,
    workflow_data_dir,
)

__all__ = [
    "load_workflow_config",
    "resolve_config_value",
    "register_graph",
    "get_graph",
    "list_graphs",
    "append_record",
    "load_records",
    "prune_records",
    "workflow_data_dir",
    "notify_email",
    "maybe_post_threads",
    "ingest_data",
    "store_data",
    "synthesis_data",
    "verify_results",
    "output_user_results",
]
