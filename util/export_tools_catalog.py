#!/usr/bin/env python3
"""Normalize config/tools.yaml from the live tool registry.

Source of truth flow:
  config/tools.yaml --(load)--> agentic.registry.TOOLS --(@tool)--> registry
  registry --(this script)--> config/tools.yaml (normalized)

Run after adding/renaming a tool so the catalog stays sorted and complete:
  uv run python util/export_tools_catalog.py
  uv run python util/export_tools_catalog.py --check   # CI: exit 1 if stale

The script imports agentic.tools (all toolkit modules) plus the graph/skill
modules that register extra tools via the legacy @tool("name") form, then
writes every registered spec back sorted by name. Tools present in the YAML
but with no registered handler are preserved (schema-only entries).
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from system.config import load_config

load_config()

HEADER = """# Declarative tool catalog for Aiko's agentic toolkit.
# Runtime source of truth for @tool decorator metadata.
# Normalize with: uv run python util/export_tools_catalog.py
"""

# Modules whose import side-effect registers tools beyond agentic.tools.
_EXTRA_MODULES = [
    "agentic.skills",
    "agentic.agentic",  # final_answer
    "agentic.graph_engine",  # list_playbooks
    "agentic.workflows.common.nodes",
    "agentic.workflows.job_hunt.graph",
    "agentic.workflows.aurora_forecast.graph",
    "agentic.workflows.owner_email.graph",
    "agentic.workflows.codebase_refresh.graph",
]


def _import_all() -> None:
    import agentic.tools  # noqa: F401 — triggers all toolkit @tool registrations

    for name in _EXTRA_MODULES:
        try:
            importlib.import_module(name)
        except Exception as exc:  # optional lanes may fail on Jetson; keep going
            print(f"warn: could not import {name}: {exc}", file=sys.stderr)


def _handler_path(fn: object) -> str | None:
    module = getattr(fn, "__module__", None)
    qualname = getattr(fn, "__qualname__", None) or getattr(fn, "__name__", None)
    if not module or not qualname:
        return None
    return f"{module}:{qualname.split('.')[-1]}"


def _spec_to_entry(name: str, spec: object, fallback_handler: str | None) -> dict:
    fn = getattr(spec, "handler", None)
    handler = _handler_path(fn) or fallback_handler or ""
    entry: dict = {
        "handler": handler,
        "name": name,
        "description": getattr(spec, "description", ""),
        "props": getattr(spec, "props", {}) or {},
    }
    required = getattr(spec, "required", []) or []
    if required:
        entry["required"] = list(required)
    domain = getattr(spec, "domain", None)
    if domain is not None:
        entry["domain"] = domain
    if getattr(spec, "always_on", False):
        entry["always_on"] = True
    entry["react"] = bool(getattr(spec, "react", True))
    entry["graph"] = bool(getattr(spec, "graph", False))
    entry["wiki"] = bool(getattr(spec, "wiki", False))
    entry["skill"] = bool(getattr(spec, "skill", False))
    return entry


def build_catalog() -> list[dict]:
    from agentic.registry import registry

    _import_all()
    raw = _read_raw_tools()
    raw_handler = {
        item["name"]: item.get("handler", "")
        for item in raw
        if isinstance(item, dict) and item.get("name")
    }
    entries: list[dict] = []
    for name in sorted(registry.get_all_tool_names()):
        spec = registry.get(name)
        entries.append(_spec_to_entry(name, spec, raw_handler.get(name)))

    # Preserve schema-only YAML entries with no registered handler.
    registered = set(registry.get_all_tool_names())
    for item in raw:
        if isinstance(item, dict) and item.get("name") and item["name"] not in registered:
            entries.append(item)
    entries.sort(key=lambda e: e.get("name", ""))
    return entries


def _read_raw_tools() -> list:
    from system.config import load_yaml

    try:
        data = load_yaml("tools.yaml")
    except Exception:
        return []
    tools = data.get("tools", [])
    return tools if isinstance(tools, list) else []


def render(entries: list[dict]) -> str:
    body = json.dumps({"tools": entries}, ensure_ascii=False, indent=2)
    return HEADER + body + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default="config/tools.yaml", help="Catalog path")
    parser.add_argument("--check", action="store_true", help="Exit 1 if file differs, without writing")
    args = parser.parse_args()
    path = Path(args.path)
    if not path.is_absolute():
        path = ROOT / path
    entries = build_catalog()
    rendered = render(entries)
    if args.check:
        current = path.read_text(encoding="utf-8") if path.exists() else ""
        if current != rendered:
            print(f"{path} is stale ({len(entries)} tools in registry). Run util/export_tools_catalog.py")
            return 1
        print(f"{path} is up to date ({len(entries)} tools)")
        return 0
    path.write_text(rendered, encoding="utf-8")
    print(f"Normalized {len(entries)} tools -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
