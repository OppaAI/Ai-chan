"""Export decorated tool metadata to config/tools.yaml without importing runtime deps."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

ROOT = Path(__file__).resolve().parents[1]


def _literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except Exception:
        return ast.unparse(node)


def _decorator_name(dec: ast.AST) -> str | None:
    if isinstance(dec, ast.Call):
        dec = dec.func
    if isinstance(dec, ast.Name):
        return dec.id
    if isinstance(dec, ast.Attribute):
        return dec.attr
    return None


def _module_path(path: Path) -> str:
    return ".".join(path.relative_to(ROOT).with_suffix("").parts)


def discover_tools() -> dict[str, Any]:
    tools: list[dict[str, Any]] = []
    for path in sorted((ROOT / "agentic").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        module = _module_path(path)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for dec in node.decorator_list:
                if not isinstance(dec, ast.Call) or _decorator_name(dec) != "tool":
                    continue
                entry: dict[str, Any] = {"handler": f"{module}:{node.name}"}
                positional = list(dec.args)
                if positional:
                    entry["name"] = _literal(positional.pop(0))
                if positional:
                    entry["description"] = _literal(positional.pop(0))
                for kw in dec.keywords:
                    if kw.arg is not None:
                        entry[kw.arg] = _literal(kw.value)
                entry.setdefault("react", True)
                entry.setdefault("graph", False)
                entry.setdefault("wiki", False)
                entry.setdefault("skill", False)
                entry.setdefault("name", node.name)
                tools.append(entry)
    tools.sort(key=lambda item: item["name"])
    return {"tools": tools}


def _dump(data: dict[str, Any]) -> str:
    header = (
        "# Generated tool catalog for Aiko's agentic toolkit.\n"
        "# Runtime source of truth: @tool decorators.\n"
        "# Regenerate with: python util/export_tools_catalog.py\n"
    )
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required to generate config/tools.yaml. "
            "Install with: pip install pyyaml"
        )
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
