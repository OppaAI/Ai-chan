"""Normalize declarative tool metadata from config/tools.yaml."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def discover_tools() -> dict[str, Any]:
    """Load and sort the declarative tool catalog.

    Tool metadata now lives in config/tools.yaml and toolkit decorators consume
    that metadata with @tool(TOOLS["tool_name"]). This utility remains useful
    as a normalizer/drift-safe formatter after editing the YAML by hand.
    """
    from system.config import load_yaml

    data = load_yaml("tools.yaml")
    tools = data.get("tools", [])
    if not isinstance(tools, list):
        raise ValueError("config/tools.yaml must contain a 'tools' list")
    normalized: list[dict[str, Any]] = []
    for entry in tools:
        if not isinstance(entry, dict) or not entry.get("name"):
            continue
        item = dict(entry)
        item.setdefault("react", True)
        item.setdefault("graph", False)
        item.setdefault("wiki", False)
        item.setdefault("skill", False)
        normalized.append(item)
    normalized.sort(key=lambda item: item["name"])
    return {"tools": normalized}


def _dump(data: dict[str, Any]) -> str:
    header = (
        "# Declarative tool catalog for Aiko's agentic toolkit.\n"
        "# Runtime source of truth for @tool decorator metadata.\n"
        "# Normalize with: python util/export_tools_catalog.py\n"
    )
    if yaml is None:
        return header + json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    return header + yaml.safe_dump(data, sort_keys=False, allow_unicode=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "config" / "tools.yaml")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rendered = _dump(discover_tools())
    if args.check:
        existing = args.output.read_text(encoding="utf-8") if args.output.exists() else ""
        if existing != rendered:
            print(f"{args.output} is out of date; run python util/export_tools_catalog.py")
            return 1
        print(f"{args.output} is up to date")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
