"""Planning, note, and workspace file tools."""

from __future__ import annotations

import textwrap
from datetime import datetime, timezone

from core.toolkit.common import MAX_READ_CHARS, MAX_WRITE_CHARS, NOTES_DIR, json_block, now_stamp, safe_path, slugify


def make_plan(goal: str, constraints: str = "", max_steps: int = 8) -> str:
    """Create a pragmatic step-by-step plan for a real-world or digital task."""
    max_steps = max(3, min(max_steps, 12))
    generic_steps = [
        "Clarify the desired outcome and success criteria.",
        "List known facts, constraints, deadlines, and missing information.",
        "Gather the minimum information needed before acting.",
        "Break the work into small reversible actions.",
        "Do the highest-impact safe action first.",
        "Check the result against the success criteria.",
        "Adjust the plan if new information changes the situation.",
        "Summarize what was done, what remains, and the next best action.",
    ][:max_steps]
    return json_block("plan created", {
        "goal": goal,
        "constraints": constraints or "none stated",
        "created_at": now_stamp(),
        "steps": generic_steps,
    })


def create_checklist(title: str, items: list[str] | str) -> str:
    """Build a markdown checklist from a list or newline-separated string."""
    if isinstance(items, str):
        item_list = [line.strip(" -\t") for line in items.splitlines() if line.strip()]
    else:
        item_list = [str(item).strip() for item in items if str(item).strip()]
    if not item_list:
        item_list = ["Define the first concrete action."]
    markdown = [f"# {title}", "", f"Created: {now_stamp()}", ""]
    markdown.extend(f"- [ ] {item}" for item in item_list)
    return "\n".join(markdown)


def save_note(title: str, content: str, folder: str = "notes") -> str:
    """Save a note, plan, draft, or task artifact under WORKSPACE_ROOT."""
    base = NOTES_DIR if folder == "notes" else safe_path(folder)
    base.mkdir(parents=True, exist_ok=True)
    filename = f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{slugify(title)}.md"
    path = base / filename
    body = content[:MAX_WRITE_CHARS]
    path.write_text(body, encoding="utf-8")
    return json_block("note saved", {"path": str(path), "chars": len(body)})


def read_workspace_file(relative_path: str, max_chars: int = MAX_READ_CHARS) -> str:
    """Read a text file from WORKSPACE_ROOT for continuation or review."""
    try:
        path = safe_path(relative_path)
        if not path.exists() or not path.is_file():
            return f"[read failed: file not found: {relative_path}]"
        return path.read_text(encoding="utf-8", errors="replace")[:max_chars]
    except Exception as e:
        return f"[read failed: {e}]"


def summarize_task_state(goal: str, done: str = "", next_action: str = "", risks: str = "") -> str:
    """Produce a compact task-state snapshot."""
    return textwrap.dedent(f"""
        # Task State

        **Goal:** {goal}
        **Updated:** {now_stamp()}

        ## Done
        {done or 'Nothing completed yet.'}

        ## Next Action
        {next_action or 'Choose the smallest safe next step.'}

        ## Risks / Unknowns
        {risks or 'No specific risks recorded.'}
    """).strip()
