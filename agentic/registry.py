"""
agentic/registry.py

Centralized tool registry for Aiko's agentic framework.

This module provides a single source of truth for tool definitions, OpenAI schemas,
capability domain tags, always-on flags, and execution handlers across ReAct and
Graph execution loops.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type

try:
    from pydantic import BaseModel, ValidationError
except Exception:  # pragma: no cover - pydantic is optional at import time
    BaseModel = None  # type: ignore[assignment]
    ValidationError = None  # type: ignore[assignment]

from system.config import load_yaml

log = logging.getLogger(__name__)


@dataclass
class ToolSpec:
    name: str
    description: str
    handler: Optional[Callable[..., Any]] = None
    props: Dict[str, Any] = field(default_factory=dict)
    required: List[str] = field(default_factory=list)
    args_model: Type[BaseModel] | None = None
    output_model: Type[BaseModel] | None = None
    needs_approval: bool = False
    domain: Optional[str] = None  # capability routing (research, scheduling, etc.)
    always_on: bool = False  # always included in tool list regardless of capability match
    # Execution modes - which backends can execute this tool
    react: bool = True   # ReAct loop
    graph: bool = True   # graph_engine playbook
    wiki: bool = False   # wiki workflow
    skill: bool = False  # skill workflow

    def _model_json_schema(self, model: Type[BaseModel] | None) -> dict[str, Any] | None:
        if model is None:
            return None
        if BaseModel is None:
            raise RuntimeError("pydantic is required for ToolSpec args_model/output_model")
        schema = model.model_json_schema()
        schema.setdefault("type", "object")
        return schema


    def validate_args(self, args: dict[str, Any]) -> dict[str, Any]:
        """Validate/coerce tool arguments using args_model when configured."""
        if self.args_model is None:
            return args
        if BaseModel is None:
            raise RuntimeError("pydantic is required for ToolSpec args_model validation")
        model = self.args_model.model_validate(args)
        return model.model_dump(mode="json", exclude_none=False)

    def validate_output(self, output: Any) -> Any:
        """Validate/coerce a structured tool/final output when configured."""
        if self.output_model is None:
            return output
        if BaseModel is None:
            raise RuntimeError("pydantic is required for ToolSpec output_model validation")
        if isinstance(output, str):
            model = self.output_model.model_validate_json(output)
        else:
            model = self.output_model.model_validate(output)
        return model.model_dump(mode="json", exclude_none=False)

    def to_openai_schema(self) -> dict[str, Any]:
        """Convert ToolSpec to OpenAI function schema format."""
        parameters = self._model_json_schema(self.args_model) or {"type": "object", "properties": self.props or {}}
        s = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
        }
        if self.required and not self.args_model:
            s["function"]["parameters"]["required"] = list(self.required)
        return s


def load_tool_catalog(path: str = "tools.yaml") -> Dict[str, ToolSpec]:
    """Load declarative tool metadata keyed by tool name from config/tools.yaml."""
    data = load_yaml(path)
    raw_tools = data.get("tools", [])
    if not isinstance(raw_tools, list):
        raise ValueError("tools.yaml must contain a 'tools' list")
    catalog: Dict[str, ToolSpec] = {}
    for item in raw_tools:
        if not isinstance(item, dict) or not item.get("name"):
            log.warning("Skipping invalid tools.yaml entry: %r", item)
            continue
        tool_data = {k: v for k, v in item.items() if k != "handler"}
        spec = ToolSpec(**tool_data)
        catalog[spec.name] = spec
    return catalog


TOOLS = load_tool_catalog()


def _attach_builtin_arg_models() -> None:
    """Attach concrete Pydantic schemas while keeping tools.yaml as fallback metadata."""
    try:
        from agentic.tool_models import (
            DirectSocialPostArgs, DraftJobPostSocialArgs, DraftPhotoSocialArgs, DraftVideoSocialArgs,
            LearnKnowledgeArgs, PostPhotoSocialArgs, PostSocialDraftArgs, PostVideoSocialArgs,
            SaveNoteArgs, ScheduleJobArgs, ScheduleReminderArgs, WriteReportArgs,
        )
    except Exception as exc:  # pragma: no cover - import-time optional dependency fallback
        log.warning("Pydantic tool argument models unavailable; using tools.yaml schemas: %s", exc)
        return
    mapping = {
        "save_note": SaveNoteArgs,
        "schedule_job": ScheduleJobArgs,
        "schedule_reminder": ScheduleReminderArgs,
        "learn_knowledge": LearnKnowledgeArgs,
        "write_report": WriteReportArgs,
        "draft_job_post_social": DraftJobPostSocialArgs,
        "post_job_post_social": PostSocialDraftArgs,
        "post_to_social": DirectSocialPostArgs,
        "draft_photo_social": DraftPhotoSocialArgs,
        "post_photo_social": PostPhotoSocialArgs,
        "draft_video_social": DraftVideoSocialArgs,
        "post_video_social": PostVideoSocialArgs,
    }
    for name, model in mapping.items():
        if name in TOOLS:
            TOOLS[name].args_model = model


_attach_builtin_arg_models()


class ToolRegistry:
    """Singleton registry tracking all declared agentic tools."""

    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}

    def register(
        self,
        name: str,
        description: str,
        handler: Optional[Callable[..., Any]] = None,
        props: Optional[Dict[str, Any]] = None,
        required: Optional[List[str]] = None,
        domain: Optional[str] = None,
        always_on: bool = False,
        react: bool = True,
        graph: bool = False,
        wiki: bool = False,
        skill: bool = False,
        args_model: Type[BaseModel] | None = None,
        output_model: Type[BaseModel] | None = None,
        needs_approval: bool = False,
    ) -> ToolSpec:
        spec = ToolSpec(
            name=name,
            description=description,
            handler=handler,
            props=props or {},
            required=required or [],
            domain=domain,
            always_on=always_on,
            react=react,
            graph=graph,
            wiki=wiki,
            skill=skill,
            args_model=args_model,
            output_model=output_model,
            needs_approval=needs_approval,
        )
        self._tools[name] = spec
        return spec

    def get(self, name: str) -> Optional[ToolSpec]:
        return self._tools.get(name)

    def all_specs(self) -> List[ToolSpec]:
        return list(self._tools.values())

    def get_tool_domains(self) -> Dict[str, str]:
        """Return dict mapping tool_name -> domain tag for capability routing."""
        return {
            spec.name: spec.domain
            for spec in self._tools.values()
            if spec.domain is not None
        }

    def get_always_on_tools(self) -> Set[str]:
        """Return set of tool names marked as always_on."""
        return {
            spec.name
            for spec in self._tools.values()
            if spec.always_on
        }

    def get_react_defs(self) -> List[Tuple[dict[str, Any], Optional[Callable[..., Any]]]]:
        """Return list of (schema, handler) tuples for agentic.py ReAct loop."""
        return [
            (spec.to_openai_schema(), spec.handler)
            for spec in self._tools.values()
            if spec.react
        ]

    def get_graph_tool_map(self) -> Dict[str, Callable[..., Any]]:
        """Return dict mapping tool_name -> handler for graph_engine.py."""
        return {
            spec.name: spec.handler
            for spec in self._tools.values()
            if spec.graph and spec.handler is not None
        }

    def get_wiki_tools(self) -> Dict[str, Callable[..., Any]]:
        """Return dict mapping tool_name -> handler for wiki workflow."""
        return {
            spec.name: spec.handler
            for spec in self._tools.values()
            if spec.wiki and spec.handler is not None
        }

    def get_skill_tools(self) -> Dict[str, Callable[..., Any]]:
        """Return dict mapping tool_name -> handler for skill workflow."""
        return {
            spec.name: spec.handler
            for spec in self._tools.values()
            if spec.skill and spec.handler is not None
        }

    def get_all_tool_names(self) -> List[str]:
        """Return list of all registered tool names for auto-export."""
        return list(self._tools.keys())

    def to_catalog(self) -> dict[str, Any]:
        """Return a deterministic, serializable catalog of registered tools.

        The catalog is intended for documentation and drift checks. Runtime
        registration should continue to use decorators/register_tool_schema so
        there is only one source of truth.
        """
        tools: list[dict[str, Any]] = []
        for spec in sorted(self._tools.values(), key=lambda item: item.name):
            entry: dict[str, Any] = {
                "name": spec.name,
                "description": spec.description,
            }
            if spec.handler is not None:
                entry["handler"] = f"{spec.handler.__module__}:{spec.handler.__name__}"
            if spec.props:
                entry["props"] = spec.props
            if spec.required:
                entry["required"] = list(spec.required)
            if spec.domain is not None:
                entry["domain"] = spec.domain
            if spec.always_on:
                entry["always_on"] = spec.always_on
            entry["react"] = spec.react
            entry["graph"] = spec.graph
            if spec.wiki:
                entry["wiki"] = spec.wiki
            if spec.skill:
                entry["skill"] = spec.skill
            if spec.args_model is not None:
                entry["args_model"] = f"{spec.args_model.__module__}:{spec.args_model.__name__}"
            if spec.output_model is not None:
                entry["output_model"] = f"{spec.output_model.__module__}:{spec.output_model.__name__}"
            if spec.needs_approval:
                entry["needs_approval"] = spec.needs_approval
            tools.append(entry)
        return {"tools": tools}


# Global registry instance
registry = ToolRegistry()


def tool(
    spec_or_name: ToolSpec | str,
    description: Optional[str] = None,
    *,
    props: Optional[Dict[str, Any]] = None,
    required: Optional[List[str]] = None,
    domain: Optional[str] = None,
    always_on: bool = False,
    react: bool = True,    # ReAct is the default execution mode
    graph: bool = False,   # Must opt-in for graph engine
    wiki: bool = False,    # Must opt-in for wiki workflow
    skill: bool = False,   # Must opt-in for skill workflow
    args_model: Type[BaseModel] | None = None,
    output_model: Type[BaseModel] | None = None,
    needs_approval: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to register a function as an agentic tool.

    Prefer passing a catalog spec loaded once from config/tools.yaml:

        @tool(TOOLS["repo_file_tree"])
        def repo_file_tree(...): ...

    The legacy ``@tool(name, description, ...)`` form is still supported for
    non-catalog callers and tests.
    """
    if isinstance(spec_or_name, ToolSpec):
        spec = spec_or_name
    else:
        if description is None:
            raise TypeError("description is required when registering by name")
        spec = ToolSpec(
            name=spec_or_name,
            description=description,
            props=props or {},
            required=required or [],
            domain=domain,
            always_on=always_on,
            react=react,
            graph=graph,
            wiki=wiki,
            skill=skill,
            args_model=args_model,
            output_model=output_model,
            needs_approval=needs_approval,
        )

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        registry.register(
            name=spec.name,
            description=spec.description,
            handler=fn,
            props=spec.props,
            required=spec.required,
            domain=spec.domain,
            always_on=spec.always_on,
            react=spec.react,
            graph=spec.graph,
            wiki=spec.wiki,
            skill=spec.skill,
            args_model=spec.args_model,
            output_model=spec.output_model,
            needs_approval=spec.needs_approval,
        )
        return fn
    return decorator


def register_tool_schema(
    name: str,
    description: str,
    *,
    props: Optional[Dict[str, Any]] = None,
    required: Optional[List[str]] = None,
    domain: Optional[str] = None,
    always_on: bool = False,
    react: bool = True,
    graph: bool = True,
    wiki: bool = False,
    skill: bool = False,
    args_model: Type[BaseModel] | None = None,
    output_model: Type[BaseModel] | None = None,
    needs_approval: bool = False,
) -> ToolSpec:
    """Register a tool schema whose execution is handled elsewhere.

    Use for tools where the handler isn't a Python function (e.g., external
    services, graph nodes, etc.). The registry still tracks the schema and
    execution modes for routing purposes.
    """
    return registry.register(
        name=name,
        description=description,
        handler=None,
        props=props,
        required=required,
        domain=domain,
        always_on=always_on,
        react=react,
        graph=graph,
        wiki=wiki,
        skill=skill,
        args_model=args_model,
        output_model=output_model,
        needs_approval=needs_approval,
    )
