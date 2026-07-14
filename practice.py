#!/usr/bin/env python3
"""Practice and promote schema-driven graph workflows without booting the chat LLM.

Examples:
  uv run python practice.py --task "make a deployment checklist and save it" \
    --tools create_checklist save_note --promote

  uv run python practice.py --task "research X and save a note" \
    --steps '[{"tool":"deep_search","ok":true,"args":{"query":"$prompt"}},{"tool":"save_note","ok":true,"args":{"title":"$title","content":"$result:step_1"}}]' \
    --promote
"""
from __future__ import annotations

import argparse
import json
from typing import Any

from system.config import load_config
load_config()

from skills import experience, schema


def _steps_from_tools(tools: list[str]) -> list[dict[str, Any]]:
    return [{"tool": t, "ok": True, "args": {}} for t in tools]


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed Aiko experience and master-plan workflows from practice examples.")
    parser.add_argument("--task", required=True, help="Example task prompt Aiko should learn/practice.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--tools", nargs="+", help="Ordered tool names for the workflow.")
    group.add_argument("--steps", help="JSON list of step objects with tool/ok/args fields.")
    parser.add_argument("--answer", default="practice workflow recorded", help="Outcome excerpt to store with the experience.")
    parser.add_argument("--promote", action="store_true", help="Append the practiced sequence to the graph master-plan JSON.")
    parser.add_argument("--name", help="Human-readable name for the promoted master plan.")
    args = parser.parse_args()

    if args.steps:
        steps = json.loads(args.steps)
        if not isinstance(steps, list):
            raise SystemExit("--steps must decode to a JSON list")
    else:
        steps = _steps_from_tools(args.tools or [])

    exp_id = experience.record_practice_experience(args.task, steps, args.answer, verified_ok=True, score=1.0)
    print(f"recorded_experience={exp_id}")

    if args.promote:
        path = schema.append_master_plan_from_experience(args.task, steps, name=args.name)
        print(f"promoted_master_plan={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
