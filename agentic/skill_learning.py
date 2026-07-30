"""Machine-written skill proposal helpers for observe -> distill -> reuse loops.

Trusted skillsets remain human-owned. This module only writes reviewable drafts
under the active user's workspace so successful/failing workflows can be
promoted deliberately later.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from system.userspace import user_workspace_root

_SLUG_RE = re.compile(r"[^a-z0-9]+")

# Per-step args are capped independently of final_preview, since a step's
# args can carry a full tool-call payload (report bodies, social post text,
# ingested document text). Without this cap, repeated observed runs against
# the same goal slug make the proposal file grow without bound.
_STEP_ARGS_MAX_CHARS = 500


def _slug(text: str) -> str:
    return _SLUG_RE.sub("-", (text or "skill").lower()).strip("-")[:80] or "skill"


def skill_proposal_dir(user_id: str | None = None) -> Path:
    path = user_workspace_root(user_id) / "skill_proposals"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _truncated_step(step: dict[str, Any]) -> dict[str, Any]:
    """Copy a step record with its args stringified and capped so a large
    tool-call payload can't blow up the proposal file size."""
    return {**step, "args": str(step.get("args"))[:_STEP_ARGS_MAX_CHARS]}


def propose_skill_from_run(goal: str, steps: list[dict[str, Any]], final_text: str, *, verified_ok: bool, score: float, user_id: str | None = None) -> Path | None:
    """Draft or refine a reviewable skill proposal from an agentic run.

    Successes capture reusable tool order. Failures append avoid-notes to the
    same proposal path so repeated mistakes refine the draft without modifying
    trusted `agentic/skillsets/` files.
    """
    if len(steps or []) < 2:
        return None

    path = skill_proposal_dir(user_id) / f"{_slug(goal)}.md"
    tools = [str(s.get("tool")) for s in steps if s.get("tool")]
    status = "success" if verified_ok and score >= 0.7 else "failure"

    header = f"# Skill Proposal: {goal[:120]}\n\n"
    if not path.exists():
        path.write_text(header + "Status: draft_review_required\n\n", encoding="utf-8")

    block = {
        "ts": time.time(),
        "status": status,
        "score": score,
        "tools": tools,
        "steps": [_truncated_step(s) for s in steps],
        "final_preview": (final_text or "")[:1000],
    }

    with path.open("a", encoding="utf-8") as f:
        f.write(f"\n## Observed run ({status})\n\n")
        if status == "success":
            f.write("Reusable tool order: " + " -> ".join(tools) + "\n\n")
        else:
            f.write("Avoid / failed approach notes: review this trace before promotion.\n\n")
        f.write("```json\n" + json.dumps(block, ensure_ascii=False, indent=2, default=str) + "\n```\n")

    return path
