"""Layer 3 — Workflow Spec schema, load/validate, and coerce from config.json.

A Spec is a versioned JSON document that fully describes a shared-pipeline
workflow. The engine builds a PlanGraph from it (see ``build_plan_graph``).

Existing ``config.json`` files remain valid: ``coerce_config_to_spec`` lifts
known keys into Spec fields and keeps the rest under ``config`` for domain
adapters (post_fields, latitude, RSS feeds, …).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

SPEC_VERSION = "1"
SUPPORTED_PIPELINES = frozenset({"shared_5"})


@dataclass
class WorkflowSpec:
    """Version-1 workflow Spec (Layer 3)."""

    spec_version: str = SPEC_VERSION
    id: str = ""
    name: str = ""
    goal: str = ""
    pipeline: str = "shared_5"
    workflow_id: str = ""
    sources: list[Any] = field(default_factory=list)
    filters: dict[str, Any] = field(default_factory=dict)
    max_items: int = 50
    retain_days: int = 3
    parallel: bool = True
    template: str = "{summary}"
    llm_enriched: bool = False
    per_item: bool = True
    human_in_the_loop: bool = False
    auto_pass_if: dict[str, Any] | None = None
    email: dict[str, Any] = field(default_factory=dict)
    social: list[Any] = field(default_factory=list)
    # Full domain bag (RSS feeds, post_fields, lat/lon, …) passed as config_json
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SpecError(ValueError):
    """Invalid or unsupported Spec."""


def validate_spec(raw: dict[str, Any]) -> WorkflowSpec:
    """Validate a Spec dict and return a WorkflowSpec. Raises SpecError."""
    if not isinstance(raw, dict):
        raise SpecError("spec must be a JSON object")

    version = str(raw.get("spec_version") or SPEC_VERSION).strip() or SPEC_VERSION
    if version != SPEC_VERSION:
        raise SpecError(f"unsupported spec_version {version!r}; want {SPEC_VERSION!r}")

    pipeline = str(raw.get("pipeline") or "shared_5").strip() or "shared_5"
    if pipeline not in SUPPORTED_PIPELINES:
        raise SpecError(
            f"unsupported pipeline {pipeline!r}; supported: {sorted(SUPPORTED_PIPELINES)}"
        )

    sid = str(raw.get("id") or raw.get("workflow_id") or "").strip()
    if not sid:
        raise SpecError("spec requires id (or workflow_id)")

    workflow_id = str(raw.get("workflow_id") or sid).strip()
    name = str(raw.get("name") or sid).strip()
    goal = str(raw.get("goal") or name).strip()

    sources = raw.get("sources")
    if sources is None:
        sources = []
    if not isinstance(sources, list):
        raise SpecError("sources must be a list")

    filters = raw.get("filters") if isinstance(raw.get("filters"), dict) else {}

    def _int(key: str, default: int) -> int:
        try:
            return max(1, int(raw.get(key, default)))
        except (TypeError, ValueError):
            return default

    max_items = _int("max_items", 50)
    retain_days = _int("retain_days", 3)

    parallel = raw.get("parallel", True)
    if isinstance(parallel, str):
        parallel = parallel.strip().lower() in {"1", "true", "yes", "on"}
    else:
        parallel = bool(parallel)

    template = str(raw.get("template") if raw.get("template") is not None else "{summary}")

    llm = raw.get("llm_enriched", False)
    if isinstance(llm, str):
        llm = llm.strip().lower() in {"1", "true", "yes", "on"}
    else:
        llm = bool(llm)

    per_item = raw.get("per_item", True)
    if isinstance(per_item, str):
        per_item = per_item.strip().lower() in {"1", "true", "yes", "on"}
    else:
        per_item = bool(per_item)

    hitl = raw.get("human_in_the_loop", False)
    if isinstance(hitl, str):
        hitl = hitl.strip().lower() in {"1", "true", "yes", "on"}
    else:
        hitl = bool(hitl)

    auto_pass = raw.get("auto_pass_if")
    if auto_pass is not None and not isinstance(auto_pass, dict):
        raise SpecError("auto_pass_if must be an object or null")

    email = raw.get("email") if isinstance(raw.get("email"), dict) else {}
    social = raw.get("social") if isinstance(raw.get("social"), list) else []

    config = raw.get("config")
    if config is None:
        config = {}
    if not isinstance(config, dict):
        raise SpecError("config must be an object")

    return WorkflowSpec(
        spec_version=version,
        id=sid,
        name=name,
        goal=goal,
        pipeline=pipeline,
        workflow_id=workflow_id,
        sources=list(sources),
        filters=dict(filters),
        max_items=max_items,
        retain_days=retain_days,
        parallel=bool(parallel),
        template=template,
        llm_enriched=bool(llm),
        per_item=bool(per_item),
        human_in_the_loop=bool(hitl),
        auto_pass_if=dict(auto_pass) if isinstance(auto_pass, dict) else None,
        email=dict(email),
        social=list(social),
        config=dict(config),
    )


def load_spec(path: str | Path) -> WorkflowSpec:
    """Load and validate a Spec JSON file."""
    p = Path(path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SpecError(f"failed to load spec from {p}: {exc}") from exc
    return validate_spec(data if isinstance(data, dict) else {})


# Keys lifted from legacy config.json onto Spec fields (not only left in config bag).
_SPEC_LIFT_KEYS = frozenset({
    "sources", "filters", "max_items", "retain_days", "parallel",
    "template", "llm_enriched", "per_item", "human_in_the_loop",
    "auto_pass_if", "email", "social", "workflow_id",
    "max_results",  # job_hunt alias for max_items
    "dedup_days",   # job_hunt alias for retain_days
})


def coerce_config_to_spec(
    *,
    graph_id: str,
    name: str,
    goal: str,
    config: dict[str, Any],
    workflow_id: str | None = None,
) -> WorkflowSpec:
    """Build a Spec from a legacy workflow config.json (+ graph metadata).

    Domain-only keys stay under ``config`` so adapters keep working unchanged.
    """
    cfg = dict(config or {})
    wid = workflow_id or str(cfg.get("workflow_id") or graph_id)

    max_items = cfg.get("max_items", cfg.get("max_results", 50))
    retain = cfg.get("retain_days", cfg.get("dedup_days", 3))

    sources = cfg.get("sources")
    if not isinstance(sources, list) or not sources:
        # Sensible defaults when config has no explicit sources list
        if wid in {"job_hunt", "gen_job_post"} or graph_id == "gen_job_post":
            sources = [{"type": "adapter", "id": "job_hunt", "name": "job_hunt"}]
        elif wid in {"aurora_forecast", "aurora"} or "aurora" in graph_id:
            sources = [{"type": "adapter", "id": "aurora", "name": "aurora"}]
        else:
            sources = []

    raw: dict[str, Any] = {
        "spec_version": SPEC_VERSION,
        "id": graph_id,
        "name": name,
        "goal": goal,
        "pipeline": "shared_5",
        "workflow_id": wid,
        "sources": sources,
        "filters": cfg.get("filters") if isinstance(cfg.get("filters"), dict) else {},
        "max_items": max_items,
        "retain_days": retain,
        "parallel": cfg.get("parallel", True),
        "template": cfg.get("template", "{summary}"),
        "llm_enriched": cfg.get("llm_enriched", False),
        "per_item": cfg.get("per_item", True),
        "human_in_the_loop": cfg.get("human_in_the_loop", False),
        "auto_pass_if": cfg.get("auto_pass_if"),
        "email": cfg.get("email") if isinstance(cfg.get("email"), dict) else {},
        "social": cfg.get("social") if isinstance(cfg.get("social"), list) else [],
        "config": cfg,  # full original config for domain adapters
    }
    return validate_spec(raw)


def load_spec_for_workflow(
    workflow_dir: Path,
    *,
    graph_id: str,
    name: str,
    goal: str,
    workflow_id: str | None = None,
) -> WorkflowSpec:
    """Load Spec from ``spec.json`` if present, else coerce ``config.json``."""
    from agentic.workflows.common.config import load_workflow_config

    spec_path = workflow_dir / "spec.json"
    if spec_path.is_file():
        try:
            spec = load_spec(spec_path)
            # Allow goal override from caller (runtime prompt)
            if goal and goal != spec.goal:
                spec = WorkflowSpec(**{**spec.to_dict(), "goal": goal})
            return spec
        except SpecError as exc:
            log.warning("invalid spec.json in %s (%s); falling back to config.json", workflow_dir, exc)

    cfg = load_workflow_config(workflow_dir)
    return coerce_config_to_spec(
        graph_id=graph_id,
        name=name,
        goal=goal,
        config=cfg,
        workflow_id=workflow_id,
    )


__all__ = [
    "SPEC_VERSION",
    "SUPPORTED_PIPELINES",
    "SpecError",
    "WorkflowSpec",
    "coerce_config_to_spec",
    "load_spec",
    "load_spec_for_workflow",
    "validate_spec",
]
