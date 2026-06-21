"""
core/tools.py

Aiko's tool belt for autonomous work.

The functions in this module are intentionally small, explicit, and boring:
plain Python callables that return strings safe to inject back into an LLM
context.  `think.py` decides *when* to use a tool; this module decides *how*
the work is performed and how failures are reported.

Capabilities:
  - Web search through SearXNG.
  - Main-text extraction from individual pages.
  - Deep-search bundles for research tasks.
  - Lightweight planning, checklist, note, and file helpers for practical
    multi-step tasks.
  - Persistent local scheduled jobs for reminders, wake-up alarms, and recurring tasks.

Safety model:
  - File helpers are restricted to AIKO_WORKSPACE_ROOT (default: ./workspace).
  - Shell execution is intentionally not exposed here.  Aiko should explain
    commands to run, not run arbitrary system commands on behalf of the user.
"""

from __future__ import annotations

import json
import os
import re
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    import requests
except ImportError:  # pragma: no cover - dependency may be absent in text-only installs
    requests = None

try:
    import trafilatura
except ImportError:  # pragma: no cover - dependency may be absent in text-only installs
    trafilatura = None

from core.log import get_logger
from core.schedule import (
    cancel_reminder_record,
    cancel_schedule_record,
    list_reminder_records,
    list_schedule_records,
    schedule_job_record,
    schedule_reminder_record,
)

log = get_logger(__name__)

# ── config ────────────────────────────────────────────────────────────────────

SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:8081")
MAX_RESULTS = int(os.getenv("SEARXNG_MAX_RESULTS", 5))
WORKSPACE_ROOT = Path(os.getenv("WORKSPACE_ROOT", "workspace")).resolve()
NOTES_DIR = WORKSPACE_ROOT / "notes"
MAX_WRITE_CHARS = int(os.getenv("MAX_WRITE_CHARS", 20_000))
MAX_READ_CHARS = int(os.getenv("MAX_READ_CHARS", 12_000))

# ── formatting helpers ────────────────────────────────────────────────────────

def _now_stamp() -> str:
    """Return a compact UTC timestamp for generated notes and plans."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _slugify(text: str, fallback: str = "task") -> str:
    """Create a stable lowercase file slug from arbitrary user text."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return (slug or fallback)[:80]


def _safe_path(relative_path: str) -> Path:
    """Resolve a user path under WORKSPACE_ROOT, rejecting path traversal."""
    cleaned = relative_path.strip().lstrip("/\\")
    path = (WORKSPACE_ROOT / cleaned).resolve()
    if path != WORKSPACE_ROOT and WORKSPACE_ROOT not in path.parents:
        raise ValueError(f"path escapes workspace: {relative_path}")
    return path


def _json_block(title: str, payload: dict[str, Any]) -> str:
    """Render machine-readable tool output with a short human title."""
    return f"[{title}]\n" + json.dumps(payload, ensure_ascii=False, indent=2)


# ── web search ────────────────────────────────────────────────────────────────

def web_search(query: str, max_results: int = MAX_RESULTS) -> str:
    """
    Search the web via SearXNG and return compact numbered results.

    Returns a readable string rather than raw JSON because local models follow
    that format more reliably during ReAct-style loops.  Failures are encoded as
    bracketed messages (`[search failed: ...]`) so callers can continue safely.
    """
    if requests is None:
        return "[search failed: requests is not installed]"
    try:
        response = requests.get(
            f"{SEARXNG_URL}/search",
            params={"q": query, "format": "json"},
            timeout=8,
        )
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        return f"[search failed: {e}]"
    except ValueError:
        return "[search failed: invalid JSON response]"

    results = data.get("results", [])[:max_results]
    if not results:
        return f"[no results found for: {query}]"

    lines = [f"[Web search results for: {query}]"]
    for i, result in enumerate(results, 1):
        title = result.get("title", "").strip()
        url = result.get("url", "").strip()
        content = result.get("content", "").strip()
        lines.append(f"{i}. {title}\n   {url}\n   {content}")

    return "\n\n".join(lines)


def fetch_and_extract(url: str, max_chars: int = 4000) -> str:
    """
    Fetch a URL and extract its main article/body text with trafilatura.

    The result is truncated to `max_chars` to protect the context window.  Only
    HTTP(S) URLs are accepted; unsupported schemes return a failure string.
    """
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return f"[fetch failed: unsupported URL scheme: {parsed.scheme or 'none'}]"
    if trafilatura is None:
        return "[fetch failed: trafilatura is not installed]"
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return "[fetch failed: empty response]"
        text = trafilatura.extract(downloaded, include_links=False, include_tables=False) or ""
        return text[:max_chars] if text else "[fetch failed: no extractable text]"
    except Exception as e:
        return f"[fetch failed: {e}]"


def deep_search(query: str, max_results: int = 3, fetch_top: int = 2) -> str:
    """
    Search, fetch the top pages, and return one compact research bundle.

    This is the preferred tool for questions where snippets may be too shallow:
    comparisons, fact checks, current events, and practical how-to tasks.
    """
    raw_results = web_search(query, max_results)
    if raw_results.startswith("[search failed") or raw_results.startswith("[no results"):
        return raw_results

    bundle = [raw_results]
    result_blocks = raw_results.split("\n\n")[1:]

    for i, result in enumerate(result_blocks[:fetch_top], 1):
        url = next((line.strip() for line in result.splitlines() if line.strip().startswith("http")), None)
        if not url:
            continue
        page = fetch_and_extract(url)
        if not page.startswith("[fetch failed"):
            bundle.append(f"\n[Full page {i}: {url}]\n{page[:2000]}")

    return "\n\n".join(bundle)


def web_search_context(query: str, max_results: int = MAX_RESULTS) -> str | None:
    """Run web_search and wrap successful results as context for chat mode."""
    results = web_search(query, max_results)
    if results.startswith("[search failed") or results.startswith("[no results"):
        return None
    return f"{results}\n\nUser asked: {query}"


# ── autonomous planning tools ─────────────────────────────────────────────────

def make_plan(goal: str, constraints: str = "", max_steps: int = 8) -> str:
    """
    Create a pragmatic step-by-step plan for a real-world or digital task.

    The plan is heuristic rather than LLM-generated so it can be used as a tool
    observation inside an agent loop.  Aiko can then refine it in language.
    """
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
    payload = {
        "goal": goal,
        "constraints": constraints or "none stated",
        "created_at": _now_stamp(),
        "steps": generic_steps,
    }
    return _json_block("plan created", payload)


def create_checklist(title: str, items: list[str] | str) -> str:
    """
    Build a markdown checklist from a list or newline-separated string.

    Useful when the user wants Aiko to help execute a complicated workflow over
    multiple turns: moving, studying, debugging, shopping, launching a project,
    or planning a trip.
    """
    if isinstance(items, str):
        item_list = [line.strip(" -\t") for line in items.splitlines() if line.strip()]
    else:
        item_list = [str(item).strip() for item in items if str(item).strip()]
    if not item_list:
        item_list = ["Define the first concrete action."]
    markdown = [f"# {title}", "", f"Created: {_now_stamp()}", ""]
    markdown.extend(f"- [ ] {item}" for item in item_list)
    return "\n".join(markdown)


def save_note(title: str, content: str, folder: str = "notes") -> str:
    """
    Save a note, plan, draft, or task artifact under AIKO_WORKSPACE_ROOT.

    The default location is `workspace/notes`.  Content is truncated at
    MAX_WRITE_CHARS to avoid accidental huge writes.
    """
    base = NOTES_DIR if folder == "notes" else _safe_path(folder)
    base.mkdir(parents=True, exist_ok=True)
    filename = f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{_slugify(title)}.md"
    path = base / filename
    body = content[:MAX_WRITE_CHARS]
    path.write_text(body, encoding="utf-8")
    return _json_block("note saved", {"path": str(path), "chars": len(body)})


def read_workspace_file(relative_path: str, max_chars: int = MAX_READ_CHARS) -> str:
    """Read a text file from AIKO_WORKSPACE_ROOT for continuation or review."""
    try:
        path = _safe_path(relative_path)
        if not path.exists() or not path.is_file():
            return f"[read failed: file not found: {relative_path}]"
        return path.read_text(encoding="utf-8", errors="replace")[:max_chars]
    except Exception as e:
        return f"[read failed: {e}]"


def summarize_task_state(goal: str, done: str = "", next_action: str = "", risks: str = "") -> str:
    """
    Produce a compact task-state snapshot that can be saved or spoken aloud.

    This helps Aiko behave like a persistent autonomous entity: she can state the
    current objective, completed work, next action, and risks without pretending
    she has performed actions outside her tools.
    """
    summary = textwrap.dedent(f"""
        # Task State

        **Goal:** {goal}
        **Updated:** {_now_stamp()}

        ## Done
        {done or 'Nothing completed yet.'}

        ## Next Action
        {next_action or 'Choose the smallest safe next step.'}

        ## Risks / Unknowns
        {risks or 'No specific risks recorded.'}
    """).strip()
    return summary

# ── schedule tools ─────────────────────────────────────────────────────────────

def schedule_job(
    title: str,
    task: str,
    time_of_day: str,
    frequency: str = "daily",
    timezone: str | None = None,
    days_of_week: list[str] | str | None = None,
    action: str = "agentic",
) -> str:
    """Schedule a local recurring job while Aiko is running."""
    try:
        job = schedule_job_record(title, task, time_of_day, frequency, timezone, days_of_week, action)
        return _json_block("scheduled job created", job)
    except Exception as e:
        return f"[schedule failed: {e}]"


def list_schedule(include_disabled: bool = False) -> str:
    """List local scheduled jobs from Aiko's schedule file."""
    jobs = list_schedule_records(include_disabled=include_disabled)
    return _json_block("schedule", {"count": len(jobs), "items": jobs})


def cancel_schedule(job_id: str) -> str:
    """Cancel/disable a local scheduled job by id."""
    if cancel_schedule_record(job_id):
        return _json_block("scheduled job cancelled", {"id": job_id})
    return f"[scheduled job not found: {job_id}]"


# ── reminder compatibility tools ──────────────────────────────────────────────

def schedule_reminder(
    title: str,
    message: str,
    time_of_day: str,
    repeat: str = "daily",
    timezone: str | None = None,
) -> str:
    """Schedule a local reminder/alarm while Aiko is running."""
    try:
        reminder = schedule_reminder_record(title, message, time_of_day, repeat, timezone)
        return _json_block("reminder scheduled", reminder)
    except Exception as e:
        return f"[reminder failed: {e}]"


def list_reminders(include_disabled: bool = False) -> str:
    """List reminders stored in Aiko's local reminder file."""
    reminders = list_reminder_records(include_disabled=include_disabled)
    return _json_block("reminders", {"count": len(reminders), "items": reminders})


def cancel_reminder(reminder_id: str) -> str:
    """Cancel/disable a local reminder by id."""
    if cancel_reminder_record(reminder_id):
        return _json_block("reminder cancelled", {"id": reminder_id})
    return f"[reminder not found: {reminder_id}]"
