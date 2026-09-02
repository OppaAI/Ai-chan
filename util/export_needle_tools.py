#!/usr/bin/env python3
"""Export Aiko's registry schemas in Needle 2's startup catalogue format."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from system.config import load_config

load_config()

from agentic.agentic import tool_schemas
from agentic.needle import needle_tools


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default="needle-tools.json", help="Output JSON path")
    args = parser.parse_args()
    path = Path(args.path)
    if not path.is_absolute():
        path = ROOT / path
    schemas = tool_schemas()
    path.write_text(json.dumps(needle_tools(schemas), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {len(schemas)} Aiko tool schemas to {path}")


if __name__ == "__main__":
    main()
