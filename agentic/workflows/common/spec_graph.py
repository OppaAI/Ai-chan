"""Layer 3 — build a PlanGraph from a WorkflowSpec.

Currently supports pipeline ``shared_5``:

  ingest_data → store_data → synthesis_data → verify_results → output_user_results
"""

from __future__ import annotations

import json
import logging
from typing import Any

from agentic.graph_engine import PlanGraph, PlanNode
from agentic.workflows.common.spec import SpecError, WorkflowSpec

log = logging.getLogger(__name__)


def build_plan_graph(spec: WorkflowSpec, *, goal: str | None = None) -> PlanGraph:
    """Compile a WorkflowSpec into a PlanGraph.

    ``goal`` overrides ``spec.goal`` when provided (runtime user prompt).
    """
    if spec.pipeline != "shared_5":
        raise SpecError(f"cannot build pipeline {spec.pipeline!r}")

    run_goal = (goal or spec.goal or spec.name or spec.id).strip()
    # Merge lifted fields back into the domain config bag so node handlers
    # that only read config_json still see sources / HITL / email / etc.
    domain: dict[str, Any] = dict(spec.config or {})
    domain.setdefault("workflow_id", spec.workflow_id or spec.id)
    domain.setdefault("sources", spec.sources)
    domain.setdefault("filters", spec.filters)
    domain.setdefault("retain_days", spec.retain_days)
    domain.setdefault("max_items", spec.max_items)
    domain.setdefault("template", spec.template)
    domain.setdefault("llm_enriched", spec.llm_enriched)
    domain.setdefault("human_in_the_loop", spec.human_in_the_loop)
    # Overwrite with validated spec values (not setdefault) for lifted fields
    domain["per_item"] = spec.per_item
    domain["parallel"] = spec.parallel
    domain["auto_pass_if"] = spec.auto_pass_if
    domain["email"] = spec.email
    domain["social"] = spec.social

    config_json = json.dumps(domain, ensure_ascii=False)
    sources = spec.sources or domain.get("sources") or []
    filters = spec.filters or {}
    retain = str(spec.retain_days)
    max_items = str(spec.max_items)
    parallel = "true" if spec.parallel else "false"
    template = spec.template or ""
    llm = "true" if spec.llm_enriched else "false"
    per_item = "true" if spec.per_item else "false"
    hitl = "true" if spec.human_in_the_loop else "false"
    auto_pass = json.dumps(spec.auto_pass_if or {}, ensure_ascii=False)
    email = spec.email or domain.get("email") or {}
    social = spec.social if spec.social else (domain.get("social") or [])
    workflow_id = spec.workflow_id or spec.id

    nodes = [
        PlanNode(
            id="ingest",
            tool="ingest_data",
            args={
                "sources_json": json.dumps(sources, ensure_ascii=False),
                "filters_json": json.dumps(filters, ensure_ascii=False),
                "parallel": parallel,
                "max_items": max_items,
                "config_json": config_json,
            },
        ),
        PlanNode(
            id="store",
            tool="store_data",
            args={
                "workflow_id": workflow_id,
                "items_json": "$result:ingest",
                "mode": "append",
                "retain_days": retain,
                "config_json": config_json,
            },
            depends_on=("ingest",),
        ),
        PlanNode(
            id="synth",
            tool="synthesis_data",
            args={
                "items_json": "$result:ingest",
                "template": template,
                "llm_enriched": llm,
                "per_item": per_item,
                "config_json": config_json,
            },
            depends_on=("store",),
        ),
        PlanNode(
            id="verify",
            tool="verify_results",
            args={
                "results_json": "$result:synth",
                "human_in_the_loop": hitl,
                "llm_verify": "false",
                "auto_pass_json": auto_pass,
                "config_json": config_json,
            },
            depends_on=("synth",),
        ),
        PlanNode(
            id="output",
            tool="output_user_results",
            args={
                "results_json": "$result:verify",
                "email_json": json.dumps(email, ensure_ascii=False),
                "social_json": json.dumps(social, ensure_ascii=False),
                "config_json": config_json,
            },
            depends_on=("verify",),
        ),
    ]

    return PlanGraph(
        id=spec.id,
        name=spec.name or spec.id,
        goal=run_goal,
        nodes=tuple(nodes),
        source="spec",
    )


__all__ = ["build_plan_graph"]
