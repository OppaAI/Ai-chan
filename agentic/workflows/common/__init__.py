"""Shared primitives for agentic workflows (job_hunt, aurora_forecast, …).

Stack:
  Spec → Graph → Nodes (execution + nodes.py registration) → Toolsets

Layer 1 registers the five shared nodes as graph tools.
Layer 3 compiles WorkflowSpec → PlanGraph (shared_5 pipeline).
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
from agentic.workflows.common.spec import (
    SPEC_VERSION,
    SpecError,
    WorkflowSpec,
    coerce_config_to_spec,
    load_spec,
    load_spec_for_workflow,
    validate_spec,
)
from agentic.workflows.common.spec_graph import build_plan_graph
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
    "SPEC_VERSION",
    "SpecError",
    "WorkflowSpec",
    "validate_spec",
    "load_spec",
    "coerce_config_to_spec",
    "load_spec_for_workflow",
    "build_plan_graph",
    "list_graphs",
    "ingest_data",
    "store_data",
    "synthesis_data",
    "verify_results",
    "output_user_results",
    "append_record",
    "load_records",
    "prune_records",
    "workflow_data_dir",
    "notify_email",
    "maybe_post_threads",
]
