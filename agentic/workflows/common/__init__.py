"""Shared primitives for agentic workflows (job_hunt, aurora_forecast, …).

Common stages across lanes:
  trigger → ingest → synthesize → store → notify/post

This package holds the cross-cutting pieces so each workflow only owns
its domain-specific fetch/score logic.
"""

from agentic.workflows.common.config import load_workflow_config, resolve_config_value
from agentic.workflows.common.graphs import get_graph, list_graphs, register_graph
from agentic.workflows.common.notify import notify_email, maybe_post_threads
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
]
