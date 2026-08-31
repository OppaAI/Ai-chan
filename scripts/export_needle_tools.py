#!/usr/bin/env python3
"""Export Aiko's registry schemas in Needle 2's startup catalogue format."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic.agentic import tool_schemas
from agentic.needle import needle_tools


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default="needle-tools.json", help="Output JSON path")
    args = parser.parse_args()
    path = Path(args.path)
    path.write_text(json.dumps(needle_tools(tool_schemas()), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {len(tool_schemas())} Aiko tool schemas to {path}")


if __name__ == "__main__":
    main()
