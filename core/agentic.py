"""
core/agentic.py

Aiko's task-mode loop: tool schemas, ReAct-style dispatch, and final response
handling. Pure tool implementations stay in core/tools.py; chat facade, TTS,
history, and memory queue ownership stay in core/think.py.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from core.log import get_logger
from core.skills import list_skillsets, load_skillset, load_skills, search_skillsets_json, skill_context_for
from core.knowledge import knowledge_context_for, wiki_context_for
from core.tools import (
    fetch_and_extract,
    deep_search,
    web_search,
    make_plan,
    create_checklist,
    save_note,
    read_workspace_file,
    summarize_task_state,
    schedule_job,
    list_schedule,
    cancel_schedule,
    schedule_reminder,
    list_reminders,
    cancel_reminder,
    scan_photo_workspace,
    propose_photo_ingestion,
    write_photo_ingestion_report,
    repo_file_tree,
    repo_read_file,
    repo_search_text,
    search_jobs,
)

log = get_logger(__name__)

MAX_AGENT_ITER = int(os.getenv("MAX_AGENT_ITER", 8))
AGENT_MAX_TOKENS = int(os.getenv("AGENT_MAX_TOKENS", os.getenv("LLM_MAX_TOKENS", 512)))
# Belt-and-suspenders budget check: even with better relevance scoring in
# knowledge.py/skills.py, a coincidental match can still slip through. This
# is a rough chars/4 ≈ tokens estimate (no tokenizer call here), used only
# to decide whether to drop the lowest-priority context blocks before ever
# reaching llama.cpp — a soft degradation instead of a hard 400 mid-task.
LLM_CTX_SIZE = int(os.getenv("LLM_CTX_SIZE", 12288))
AGENT_CONTEXT_BUDGET_RATIO = float(os.getenv("AGENT_CONTEXT_BUDGET_RATIO", 0.65))
AGENT_MEMORY_DRAIN_TIMEOUT = float(os.getenv("AGENT_MEMORY_DRAIN_TIMEOUT", os.getenv("MEMORY_AGENT_DRAIN_TIMEOUT", 0.25)))
AGENT_MEMORY_RECALL_LIMIT = int(os.getenv("AGENT_MEMORY_RECALL_LIMIT", min(int(os.getenv("MEMORY_RECALL_LIMIT", 3)), 2)))
AGENT_NOTE_MAX_CHARS = int(os.getenv("AGENT_NOTE_MAX_CHARS", 1500))
AGENT_TOOL_RESULT_MAX_CHARS = int(os.getenv("AGENT_TOOL_RESULT_MAX_CHARS", 3000))
AGENT_VERIFY_FINAL = os.getenv("AGENT_VERIFY_FINAL", "1").lower() in {"1", "true", "yes", "on"}
AGENT_VERIFY_LLM = os.getenv("AGENT_VERIFY_LLM", "1").lower() in {"1", "true", "yes", "on"}
AGENT_MAX_FINAL_REPAIRS = int(os.getenv("AGENT_MAX_FINAL_REPAIRS", 2))
AGENT_VERIFY_MIN_SCORE = float(os.getenv("AGENT_VERIFY_MIN_SCORE", "0.70"))
AGENT_TOOL_RETRY_BACKOFF = float(os.getenv("AGENT_TOOL_RETRY_BACKOFF", 0.4))

_REPO_ROOT = Path(__file__).resolve().parent.parent
_AGENTIC_POLICY_PATHS = (
    _REPO_ROOT / "persona" / "skills.md",
    _REPO_ROOT / "persona" / "schedule.md",
)


def _agentic_policy_context() -> str:
    """Load task-only persona policy for the agent loop.

    Normal chat intentionally excludes these files to keep casual turns light;
    task mode injects them explicitly so scheduling and tool discipline remain
    available before the first tool-choice call.
    """
    blocks: list[str] = []
    for path in _AGENTIC_POLICY_PATHS:
        text = load_skills(path).strip()
        if text:
            rel = path.relative_to(_REPO_ROOT)
            blocks.append(f'<agentic_policy path="{rel}">\n{text}\n</agentic_policy>')
    if not blocks:
        return "<agentic_policy_context>\nNo task policy files found.\n</agentic_policy_context>"
    return "<agentic_policy_context>\n" + "\n\n".join(blocks) + "\n</agentic_policy_context>"


_ERROR_PREFIX_RE = re.compile(r"^\[(?P<label>[^\]:]+)(?::\s*(?P<detail>.*))?\]$", re.DOTALL)
_DISCLOSURE_RE = re.compile(
    r"\b(couldn'?t|cannot|can't|failed|unavailable|not available|limitation|"
    r"could not|wasn'?t able|unable|unverified|not verified|partial)\b",
    re.IGNORECASE,
)
_EXTERNAL_ACTION_RE = re.compile(r"\b(send|sent|email|post|posted|buy|bought|book|booked|order|ordered|delete|deleted)\b", re.IGNORECASE)
_LOCAL_ARTIFACT_RE = re.compile(r"\b(saved|created|scheduled|cancelled|path|id|draft|note|workspace)\b", re.IGNORECASE)
_RESEARCH_CONTEXT_TOOLS = {"deep_search", "fetch_page"}


def _tool(schema: dict):
    """Decorator used to keep schemas and dispatch handlers in one registry."""
    def decorator(func):
        _TOOLS[schema["function"]["name"]] = (schema, func)
        return func
    return decorator


_TOOLS: dict[str, tuple[dict, object]] = {}


@dataclass
class ToolResult:
    """Structured outcome for one tool call attempt."""

    ok: bool
    tool: str
    args: dict
    content: str
    error_type: str | None = None
    retryable: bool = False
    attempts: int = 1
    metadata: dict = field(default_factory=dict)

    def observation(self) -> str:
        """Render a compact machine-readable observation for the next LLM step."""
        payload = {
            "ok": self.ok,
            "tool": self.tool,
            "attempts": self.attempts,
            "retryable": self.retryable,
            "error_type": self.error_type,
            "args": self.args,
            "content": self.content[:AGENT_TOOL_RESULT_MAX_CHARS],
        }
        if self.metadata:
            payload["metadata"] = self.metadata
        return json.dumps(payload, ensure_ascii=False, indent=2)


@dataclass
class TaskState:
    """Runtime ledger of actions, evidence, and unresolved failures."""

    goal: str
    steps: list[dict] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    failures: list[ToolResult] = field(default_factory=list)

    def record(self, result: ToolResult) -> None:
        self.steps.append({
            "tool": result.tool,
            "ok": result.ok,
            "attempts": result.attempts,
            "error_type": result.error_type,
            "args": result.args,
        })
        if result.ok:
            self.evidence.append(f"{result.tool}: {result.content[:500]}")
        else:
            self.failures.append(result)

    def summary(self) -> str:
        payload = {
            "goal": self.goal,
            "completed_tools": [s for s in self.steps if s["ok"]],
            "failed_tools": [s for s in self.steps if not s["ok"]],
            "evidence_count": len(self.evidence),
            "unresolved_failures": [
                {
                    "tool": f.tool,
                    "error_type": f.error_type,
                    "content": f.content[:300],
                    "args": f.args,
                }
                for f in self.failures
            ],
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)


@dataclass
class VerificationResult:
    """Final-answer verification verdict."""

    ok: bool
    feedback: str
    score: float = 1.0


def tool_schemas() -> list[dict]:
    """Return OpenAI-compatible tool schemas for autonomous task mode."""
    return [schema for schema, _handler in _TOOLS.values()]


_TOOL_SCHEMAS = [
        {"type": "function", "function": {
            "name": "web_search",
            "description": "Snippet-only web search for discovering candidate sources. Does not fetch page text.",
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string", "description": "The search query."}},
                "required": ["query"]}}},
        {"type": "function", "function": {
            "name": "deep_search",
            "description": "Agentic research fetch: search once, fetch the top pages, and return a compact evidence bundle. Use at most once per workflow, then plan/summarize/save from that evidence instead of calling more web tools.",
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string", "description": "The focused research query to search and fetch."}},
                "required": ["query"]}}},
        {"type": "function", "function": {
            "name": "fetch_page", "description": "Fetch one explicitly supplied URL when the user provides it or a prior snippet result must be verified. Prefer deep_search for research workflows; do not use fetch_page to bulk-fetch search results.",
            "parameters": {"type": "object", "properties": {
                "url": {"type": "string", "description": "The URL to fetch."}},
                "required": ["url"]}}},
        {"type": "function", "function": {
            "name": "make_plan", "description": "Make plan.",
            "parameters": {"type": "object", "properties": {
                "goal": {"type": "string"},
                "constraints": {"type": "string"},
                "max_steps": {"type": "integer"}},
                "required": ["goal"]}}},
        {"type": "function", "function": {
            "name": "create_checklist", "description": "Make checklist.",
            "parameters": {"type": "object", "properties": {
                "title": {"type": "string"},
                "items": {"type": "string", "description": "Newline-separated checklist items."}},
                "required": ["title", "items"]}}},
        {"type": "function", "function": {
            "name": "save_note",
            "description": (
                "Save a note to a workspace file. "
                "content MUST be plain text only, under 400 characters. "
                "No markdown tables, no bullet lists, no backticks, no quotes. "
                "Write a brief plain-text summary only."
            ),
            "parameters": {"type": "object", "properties": {
                "title": {"type": "string", "description": "Short filename title."},
                "content": {"type": "string", "description": "Plain text only. Max 400 chars. No markdown."},
                "folder": {"type": "string", "description": "Subfolder, default: notes"}},
                "required": ["title", "content"]}}},
        {"type": "function", "function": {
            "name": "read_workspace_file", "description": "Read workspace file.",
            "parameters": {"type": "object", "properties": {
                "relative_path": {"type": "string"}},
                "required": ["relative_path"]}}},
        {"type": "function", "function": {
            "name": "summarize_task_state", "description": "Summarize task state.",
            "parameters": {"type": "object", "properties": {
                "goal": {"type": "string"}, "done": {"type": "string"},
                "next_action": {"type": "string"}, "risks": {"type": "string"}},
                "required": ["goal"]}}},
        {"type": "function", "function": {
            "name": "schedule_job", "description": "Schedule local job/alarm. HH:MM. Frequencies: once,hourly,daily,weekdays,weekly,biweekly,monthly,custom_weekdays. Supports relative_days for today/tomorrow/day-after-tomorrow offsets.",
            "parameters": {"type": "object", "properties": {
                "title": {"type": "string"}, "task": {"type": "string"},
                "time_of_day": {"type": "string", "description": "24-hour local time, e.g. 06:00"},
                "frequency": {"type": "string", "enum": ["once", "hourly", "daily", "weekdays", "weekly", "biweekly", "monthly", "custom_weekdays"]},
                "timezone": {"type": "string"},
                "days_of_week": {"type": "string", "description": "Optional weekdays, e.g. Monday Wednesday Friday"},
                "relative_days": {"type": "string", "description": "Optional day offset/phrase for the first due date, e.g. 0/today, 1/tomorrow, 2/day after tomorrow"},
                "action": {"type": "string", "enum": ["announce", "agentic"], "description": "announce only, or agentic to let Aiko perform a local autonomous task"}},
                "required": ["title", "task", "time_of_day"]}}},
        {"type": "function", "function": {
            "name": "list_schedule", "description": "List schedule.",
            "parameters": {"type": "object", "properties": {
                "include_disabled": {"type": "boolean"}}}}},
        {"type": "function", "function": {
            "name": "cancel_schedule", "description": "Cancel schedule item.",
            "parameters": {"type": "object", "properties": {
                "job_id": {"type": "string"}},
                "required": ["job_id"]}}},
        {"type": "function", "function": {
            "name": "schedule_reminder", "description": "Simple once/daily reminder.",
            "parameters": {"type": "object", "properties": {
                "title": {"type": "string"}, "message": {"type": "string"},
                "time_of_day": {"type": "string"},
                "repeat": {"type": "string", "enum": ["once", "daily"]},
                "timezone": {"type": "string"}},
                "required": ["title", "message", "time_of_day"]}}},
        {"type": "function", "function": {
            "name": "list_reminders", "description": "List reminders.",
            "parameters": {"type": "object", "properties": {
                "include_disabled": {"type": "boolean"}}}}},
        {"type": "function", "function": {
            "name": "cancel_reminder", "description": "Cancel reminder by id.",
            "parameters": {"type": "object", "properties": {
                "reminder_id": {"type": "string"}},
                "required": ["reminder_id"]}}},
        {"type": "function", "function": {
            "name": "list_skillsets", "description": "List Aiko's predefined local workflow skillsets.",
            "parameters": {"type": "object", "properties": {}}}},
        {"type": "function", "function": {
            "name": "search_skillsets", "description": "Search Aiko's predefined workflow skillsets by task/query.",
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer"}},
                "required": ["query"]}}},
        {"type": "function", "function": {
            "name": "load_skillset", "description": "Load the full markdown instructions for one predefined skillset by id.",
            "parameters": {"type": "object", "properties": {
                "skill_id": {"type": "string"}},
                "required": ["skill_id"]}}},
        {"type": "function", "function": {
            "name": "scan_photo_workspace", "description": "Scan a workspace photo inbox for wildlife/nature/astro image files.",
            "parameters": {"type": "object", "properties": {
                "inbox": {"type": "string", "description": "Workspace-relative inbox path, default photos/inbox."},
                "limit": {"type": "integer"}}}}},
        {"type": "function", "function": {
            "name": "propose_photo_ingestion", "description": "Create a safe dry-run ingestion plan for photo files without moving or editing metadata.",
            "parameters": {"type": "object", "properties": {
                "inbox": {"type": "string"},
                "library_root": {"type": "string"},
                "rating_rule": {"type": "string"}}}}},
        {"type": "function", "function": {
            "name": "write_photo_ingestion_report", "description": "Write a photo workflow report under the workspace reports folder.",
            "parameters": {"type": "object", "properties": {
                "title": {"type": "string"},
                "content": {"type": "string"},
                "report_dir": {"type": "string"}}}}},
        {"type": "function", "function": {
            "name": "repo_file_tree", "description": "List repository text files for Aiko architecture/code navigation.",
            "parameters": {"type": "object", "properties": {
                "prefix": {"type": "string"},
                "limit": {"type": "integer"}}}}},
        {"type": "function", "function": {
            "name": "repo_read_file", "description": "Read one repository text file for architecture/code work.",
            "parameters": {"type": "object", "properties": {
                "relative_path": {"type": "string"},
                "max_chars": {"type": "integer"}},
                "required": ["relative_path"]}}},
        {"type": "function", "function": {
            "name": "repo_search_text", "description": "Search repository text files with simple substring matching.",
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string"},
                "prefix": {"type": "string"},
                "limit": {"type": "integer"}},
                "required": ["query"]}}},
        {"type": "function", "function": {
            "name": "search_jobs", "description": "Search configured job boards for a role. If location is omitted, uses the job_hunt skill default location. Deduped automatically.",
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string"},
                "location": {"type": "string", "description": "Optional override. Defaults to the job_hunt skill location."},
                "max_results": {"type": "integer"},
                "max_age_days": {"type": "integer"},
                "job_type": {"type": "string", "description": "Optional employment type filter from the user prompt, e.g. full-time, contract, remote."}},
                "required": ["query"]}}},
        {"type": "function", "function": {
            "name": "final_answer", "description": "Final answer.",
            "parameters": {"type": "object", "properties": {
                "answer": {"type": "string", "description": "The final answer text."}},
                "required": ["answer"]}}},
    ]


def _register_tools() -> None:
    handlers = {
        "web_search": lambda args: web_search(args.get("query", "")),
        "deep_search": lambda args: deep_search(args.get("query", "")),
        "fetch_page": lambda args: fetch_and_extract(args.get("url", "")),
        "make_plan": lambda args: make_plan(args.get("goal", ""), args.get("constraints", ""), int(args.get("max_steps", 8) or 8)),
        "create_checklist": lambda args: create_checklist(args.get("title", "Checklist"), args.get("items", "")),
        "save_note": lambda args: save_note(args.get("title", "Aiko note"), args.get("content", ""), args.get("folder", "notes")),
        "read_workspace_file": lambda args: read_workspace_file(args.get("relative_path", "")),
        "summarize_task_state": lambda args: summarize_task_state(args.get("goal", ""), args.get("done", ""), args.get("next_action", ""), args.get("risks", "")),
        "schedule_job": lambda args: schedule_job(args.get("title", "Scheduled job"), args.get("task", "Scheduled job"), args.get("time_of_day", "06:00"), args.get("frequency", "daily"), args.get("timezone"), args.get("days_of_week"), args.get("action", "agentic"), args.get("relative_days")),
        "list_schedule": lambda args: list_schedule(bool(args.get("include_disabled", False))),
        "cancel_schedule": lambda args: cancel_schedule(args.get("job_id", "")),
        "schedule_reminder": lambda args: schedule_reminder(args.get("title", "Reminder"), args.get("message", "Reminder"), args.get("time_of_day", "06:00"), args.get("repeat", "daily"), args.get("timezone")),
        "list_reminders": lambda args: list_reminders(bool(args.get("include_disabled", False))),
        "cancel_reminder": lambda args: cancel_reminder(args.get("reminder_id", "")),
        "list_skillsets": lambda args: list_skillsets(),
        "search_skillsets": lambda args: search_skillsets_json(args.get("query", ""), int(args.get("limit", 3) or 3)),
        "load_skillset": lambda args: load_skillset(args.get("skill_id", "")),
        "scan_photo_workspace": lambda args: scan_photo_workspace(args.get("inbox", "photos/inbox"), int(args.get("limit", 100) or 100)),
        "propose_photo_ingestion": lambda args: propose_photo_ingestion(args.get("inbox", "photos/inbox"), args.get("library_root", "photos/library"), args.get("rating_rule", "manual-review-first")),
        "write_photo_ingestion_report": lambda args: write_photo_ingestion_report(args.get("title", "photo-ingestion"), args.get("content", ""), args.get("report_dir", "photos/reports")),
        "repo_file_tree": lambda args: repo_file_tree(args.get("prefix", ""), int(args.get("limit", 200) or 200)),
        "repo_read_file": lambda args: repo_read_file(args.get("relative_path", ""), int(args.get("max_chars", 20000) or 20000)),
        "repo_search_text": lambda args: repo_search_text(args.get("query", ""), args.get("prefix", ""), int(args.get("limit", 50) or 50)),
        "search_jobs": lambda args: json.dumps(
            search_jobs(
                args.get("query", ""),
                args.get("location", ""),
                int(args["max_results"]) if args.get("max_results") not in (None, "") else None,
                int(args["max_age_days"]) if args.get("max_age_days") not in (None, "") else None,
                args.get("job_type", ""),
            ),
            ensure_ascii=False,
        ),
    }
    for schema in _TOOL_SCHEMAS:
        name = schema["function"]["name"]
        _TOOLS[name] = (schema, handlers.get(name, lambda _args, n=name: f"[unknown tool: {n}]"))


_register_tools()


def _required_args_for(name: str) -> list[str]:
    entry = _TOOLS.get(name)
    if not entry:
        return []
    return list(entry[0].get("function", {}).get("parameters", {}).get("required", []))


def _validate_args(name: str, args: object) -> ToolResult | None:
    """Return a validation error result, or None when args are safe to dispatch."""
    if name == "final_answer":
        return None
    if not isinstance(args, dict):
        return ToolResult(
            ok=False, tool=name, args={},
            content="Tool arguments must be a JSON object. Reissue the call with valid JSON.",
            error_type="invalid_args", retryable=True,
        )
    missing = [
        key for key in _required_args_for(name)
        if args.get(key) is None or str(args.get(key)).strip() == ""
    ]
    if missing:
        return ToolResult(
            ok=False, tool=name, args=args,
            content=f"Missing required argument(s): {', '.join(missing)}. Reissue the tool call with complete arguments.",
            error_type="missing_args", retryable=True,
        )

    # Guard against blank strings for search/fetch
    if name == "web_search" and not (args.get("query") or "").strip():
        return ToolResult(
            ok=False, tool=name, args=args,
            content="Missing required argument: query must be a non-empty string. Reissue with a specific search query.",
            error_type="missing_args", retryable=True,
        )
    if name == "deep_search" and not (args.get("query") or "").strip():
        return ToolResult(
            ok=False, tool=name, args=args,
            content="Missing required argument: query must be a non-empty string. Reissue with a focused research query.",
            error_type="missing_args", retryable=True,
        )
    if name == "fetch_page" and not (args.get("url") or "").strip():
        return ToolResult(
            ok=False, tool=name, args=args,
            content="Missing required argument: url must be a non-empty string.",
            error_type="missing_args", retryable=True,
        )

    return None


def _classify_result(name: str, args: dict, content: str, attempts: int = 1) -> ToolResult:
    """Convert legacy string tool output into a structured result."""
    text = content or ""
    stripped = text.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        match = _ERROR_PREFIX_RE.match(stripped)
        label = (match.group("label") if match else "tool failed").lower()
        detail = match.group("detail") if match else stripped.strip("[]")
        retryable = any(marker in label for marker in ("search failed", "fetch failed"))
        retryable = retryable or any(marker in (detail or "").lower() for marker in ("timeout", "connection", "empty response"))
        return ToolResult(
            ok=False, tool=name, args=args, content=stripped,
            error_type=label.replace(" ", "_"),
            retryable=retryable,
            attempts=attempts,
            metadata={"detail": detail or label},
        )
    return ToolResult(ok=True, tool=name, args=args, content=text, attempts=attempts)


def dispatch_tool(name: str, args: dict) -> str:
    """Run one named tool with already-decoded JSON args."""
    entry = _TOOLS.get(name)
    if not entry:
        return f"[unknown tool: {name}]"
    if name == "save_note":
        args["content"] = args.get("content", "")[:AGENT_NOTE_MAX_CHARS]
        args["title"] = args.get("title", "aiko-note")
    return entry[1](args)


def dispatch_tool_checked(name: str, args: dict) -> ToolResult:
    """Run a tool and return a structured result, catching unexpected exceptions."""
    try:
        content = dispatch_tool(name, args)
    except Exception as e:
        log.exception("Tool %s raised unexpectedly", name)
        return ToolResult(
            ok=False, tool=name, args=args,
            content=f"[tool exception: {e}]",
            error_type="tool_exception",
            retryable=False,
        )
    return _classify_result(name, args, str(content))


def _max_attempts_for(name: str) -> int:
    if name in {"web_search", "deep_search", "fetch_page"}:
        return max(1, int(os.getenv("AGENT_WEB_TOOL_ATTEMPTS", 2)))
    if name in {"save_note", "schedule_job", "schedule_reminder"}:
        return max(1, int(os.getenv("AGENT_LOCAL_TOOL_ATTEMPTS", 1)))
    return 1


def execute_tool_with_policy(name: str, args: dict, state: TaskState) -> ToolResult:
    """Validate, run, retry, and ledger one tool call."""
    validation = _validate_args(name, args)
    if validation is not None:
        state.record(validation)
        return validation

    last = ToolResult(ok=False, tool=name, args=args, content="[tool did not run]", error_type="not_run")
    for attempt in range(1, _max_attempts_for(name) + 1):
        last = dispatch_tool_checked(name, dict(args))
        last.attempts = attempt
        if last.ok or not last.retryable:
            break
        if attempt < _max_attempts_for(name):
            time.sleep(AGENT_TOOL_RETRY_BACKOFF * attempt)

    state.record(last)
    return last


def _has_successful_tool_call(state: TaskState, tool_name: str) -> bool:
    """Return True only when a prior call to a tool completed successfully."""
    return any(step["tool"] == tool_name and step["ok"] for step in state.steps)


def _compact_processed_research_context(messages: list[dict]) -> None:
    """Replace already-consumed fetched-page observations with tiny placeholders.

    The model has processed tool observations once a later assistant message has
    arrived. Keeping full fetched pages in every subsequent request bloats small
    context windows, so task mode preserves only the fact that research evidence
    was consumed. The durable ledger keeps a short evidence preview separately.
    """
    for message in messages:
        if message.get("role") != "tool" or message.get("name") not in _RESEARCH_CONTEXT_TOOLS:
            continue
        content = str(message.get("content") or "")
        if '"research_context_compacted"' in content:
            continue
        if len(content) < 800:
            continue
        message["content"] = json.dumps(
            {
                "ok": True,
                "tool": message.get("name"),
                "research_context_compacted": True,
                "content": (
                    "Fetched research evidence was provided to and consumed by "
                    "the previous reasoning step; use the derived plan/summary/"
                    "artifact from subsequent context instead of re-reading raw pages."
                ),
            },
            ensure_ascii=False,
            indent=2,
        )



def _sanitize_user_facing_tool_detail(detail: str, max_chars: int = 300) -> str:
    """Redact sensitive/internal-looking details before surfacing blockers."""
    text = (detail or "").strip()
    if not text:
        return "unknown tool failure"
    text = re.sub(
        r"(?i)(api[_-]?key|token|secret|password)(\s*[:=]\s*)([^\s,;]+)",
        r"\1\2[redacted]",
        text,
    )
    text = re.sub(r"(?i)(authorization\s*:\s*bearer\s+)[A-Za-z0-9._~+/=-]+", r"\1[redacted]", text)
    text = re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+", r"\1[redacted]", text)
    text = re.sub(r"(?i)(https?://)(localhost|127\.0\.0\.1|0\.0\.0\.0|[^\s/]+\.local)([^\s)]*)", r"\1[internal-url-redacted]", text)
    text = re.sub(r"(?m)^\s*File \"[^\n]+", "File [internal path redacted]", text)
    text = re.sub(r"(?m)^\s*(Traceback \(most recent call last\):|During handling of the above exception.*)$", "[stack trace redacted]", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars] or "unknown tool failure"

def _build_incomplete_task_answer(state: TaskState, last_content: str = "") -> str:
    """Create a useful final response when the model never emits final_answer.

    This is the last-resort path for small/local models that keep looping,
    produce invalid tool calls, or spend the whole iteration budget repairing a
    final answer. It should disclose blockers without using the generic
    "got lost" apology that makes successful partial work look like a total
    failure.
    """
    lines: list[str] = []
    if state.evidence:
        lines.append("I completed these step(s):")
        for item in state.evidence[-5:]:
            lines.append(f"- {item[:600]}")
    if state.failures:
        lines.append("I could not fully complete the task because of these blocker(s):")
        for failure in state.failures[-3:]:
            detail = _sanitize_user_facing_tool_detail(failure.content or failure.error_type or "")
            lines.append(f"- {failure.tool}: {detail}")
    if last_content.strip():
        lines.append("Most recent model draft:")
        lines.append(last_content.strip())
    if not lines:
        lines.append(
            "I could not complete the task before the agent loop reached its step limit, "
            "and no tool results were recorded."
        )
    return "\n".join(lines)

def _coerce_verifier_bool(value) -> bool:
    """Parse verifier booleans without treating non-empty strings as True."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "pass", "passed"}
    return bool(value)


def _verify_final_answer(owner, user_input: str, answer: str, state: TaskState) -> VerificationResult:
    """Check answer completeness and evidence support before Aiko speaks it."""
    issues: list[str] = []
    stripped = (answer or "").strip()
    lowered = stripped.lower()

    if not stripped:
        issues.append("The final answer is empty.")

    if state.failures and not _DISCLOSURE_RE.search(stripped):
        failed = ", ".join(f.tool for f in state.failures[-3:])
        issues.append(f"Unresolved tool failure(s) were not disclosed: {failed}.")

    if any(step["tool"] == "save_note" and step["ok"] for step in state.steps):
        if "path" not in lowered and "workspace" not in lowered and ".md" not in lowered:
            issues.append("A saved note was created, but the final answer does not mention where it was saved.")

    if any(step["tool"] in {"schedule_job", "schedule_reminder"} and step["ok"] for step in state.steps):
        if "scheduled" not in lowered and "reminder" not in lowered and "alarm" not in lowered:
            issues.append("A schedule/reminder tool succeeded, but the final answer does not confirm it.")

    if _EXTERNAL_ACTION_RE.search(user_input) and not _LOCAL_ARTIFACT_RE.search(stripped):
        issues.append("The answer may imply an unsupported external action instead of a local draft/staged artifact.")

    if issues or not AGENT_VERIFY_LLM:
        return VerificationResult(ok=not issues, feedback="\n".join(issues) or "Verified by deterministic checks.", score=0.0 if issues else 1.0)

    prompt = (
        "You are Aiko's final-answer verifier. This is NOT just a JSON schema check. "
        "Judge whether the candidate answer is accurate, complete, and supported by "
        "the task ledger/tool evidence. Do not use outside knowledge to bless facts that "
        "are missing from the ledger. Fail answers that invent unsupported details, hide "
        "tool failures, imply external actions that were not performed, omit required paths "
        "or confirmations, or do not answer the user's request. Return ONLY compact JSON "
        "with keys: pass (boolean), score (0-1), feedback (string). Do not add markdown.\n\n"
        f"User request:\n{user_input}\n\n"
        f"Task ledger/tool evidence:\n{state.summary()}\n\n"
        f"Candidate answer:\n{stripped}"
    )
    try:
        resp = owner._client.chat.completions.create(
            model=owner._llm_model,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            max_tokens=160,
            temperature=0.0,
        )
        raw = (resp.choices[0].message.content or "").strip()
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        data = json.loads(match.group(0) if match else raw)
        ok = _coerce_verifier_bool(data.get("pass"))
        raw_score = data.get("score", 1.0 if ok else 0.0)
        feedback = str(data.get("feedback") or ("Verifier passed." if ok else "Verifier failed."))
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            score = 0.0
            ok = False
            feedback = "Verifier returned an invalid score."
        if not math.isfinite(score) or score < 0.0 or score > 1.0:
            ok = False
            feedback = f"Verifier returned an out-of-range score: {raw_score!r}."
            score = 0.0
        if score < AGENT_VERIFY_MIN_SCORE:
            ok = False
            if feedback == "Verifier passed.":
                feedback = f"Verifier score {score:.2f} below threshold {AGENT_VERIFY_MIN_SCORE:.2f}."
        return VerificationResult(ok=ok, feedback=feedback, score=score)
    except Exception as e:
        log.warning("Agent verifier failed; falling back to deterministic pass: %s", e)
        return VerificationResult(ok=True, feedback="Verifier unavailable; deterministic checks passed.", score=0.75)


def _estimate_tokens(text: str) -> int:
    """Rough chars/4 token estimate — good enough for a budget guard, not
    for billing/accounting (those use the real /tokenize endpoint elsewhere)."""
    return max(1, len(text) // 4)


def _enforce_agentic_context_budget(
    persona: str,
    agentic_policy_context: str,
    memory_context: str,
    user_input: str,
    wiki_context: str,
    skill_context: str,
    knowledge_context: str,
) -> tuple[str, str, str]:
    """Drop wiki/knowledge/skill context, lowest priority first, if the
    assembled prompt would still exceed the ctx budget after the per-call
    caps applied at the call site. Returns the (possibly trimmed) three
    context strings in their original order."""
    budget = int(LLM_CTX_SIZE * AGENT_CONTEXT_BUDGET_RATIO)
    fixed = persona + agentic_policy_context + memory_context + user_input
    fixed_tokens = _estimate_tokens(fixed)

    # Priority order to drop, weakest first: wiki > knowledge > skill.
    # Skill workflows are kept longest since they carry concrete tool
    # instructions the agent loop actually needs to act correctly; wiki
    # pages are the most "nice to have" background context.
    blocks = {"wiki": wiki_context, "knowledge": knowledge_context, "skill": skill_context}
    drop_order = ["wiki", "knowledge", "skill"]

    for name in drop_order:
        total_tokens = fixed_tokens + sum(_estimate_tokens(v) for v in blocks.values())
        if total_tokens <= budget:
            break
        log.warning(
            "[agentic] context budget exceeded (%s > %s est. tokens); dropping %s block",
            total_tokens, budget, name,
        )
        blocks[name] = f"<{name}_context>\nOmitted this turn — context budget exceeded.\n</{name}_context>"

    return blocks["wiki"], blocks["skill"], blocks["knowledge"]


def run_agentic_chat(owner, user_input: str, token_callback=None) -> str:
    """Run task mode using the owning AikoThink instance for model/memory/output."""
    tools = tool_schemas()

    if not owner.wait_for_memory(timeout=AGENT_MEMORY_DRAIN_TIMEOUT):
        log.debug("Agent memory queue still draining; continuing without blocking turn start.")
    memories = owner._memorize.search(user_input, limit=AGENT_MEMORY_RECALL_LIMIT)
    memory_block = owner._memorize.format_for_context(memories)
    memory_context = memory_block or "<memory_context>\nNo relevant memories found.\n</memory_context>"
    agentic_policy_context = _agentic_policy_context()
    # These previously had no size caps at all in the agentic path (unlike
    # think.py's normal chat path, which caps knowledge_context_for at
    # limit=3/max_chars=4500). An unrelated but coincidentally-matched skill
    # or doc could inject thousands of tokens into a task-mode turn with no
    # ceiling, which is what caused the 12288-ctx overflow on a routine
    # "let's make cookies" turn. Cap all three here; the relevance-scoring
    # fixes in knowledge.py/skills.py reduce *bad* matches, but these caps
    # bound the damage even from a still-imperfect match.
    # Reuse the same HarrierEmbedder instance already warm for memory search
    # and intent routing (think.py's _semantic_all_scores uses the same
    # object) — no extra model load, just one more embed_query() call per
    # context type. Falls back to keyword scoring automatically if this
    # embedder is missing or an embed call fails (see knowledge.py/skills.py).
    _embedder = getattr(getattr(owner._memorize, "_mem", None), "_embedder", None)
    wiki_context = wiki_context_for(user_input, limit=1, max_chars=1500, embedder=_embedder)
    skill_context = skill_context_for(user_input, limit=2, max_chars=3000, embedder=_embedder)
    knowledge_context = knowledge_context_for(user_input, limit=2, max_chars=2500, embedder=_embedder)
    wiki_context, skill_context, knowledge_context = _enforce_agentic_context_budget(
        owner._persona, agentic_policy_context, memory_context, user_input,
        wiki_context, skill_context, knowledge_context,
    )

    agent_system = (
        f"{owner._persona}\n\n"
        f"{agentic_policy_context}\n\n"
        f"{wiki_context}\n\n"
        "[TASK MODE OVERRIDE] The speech style limits in the persona do NOT apply "
        "in task mode. Do not summarize in 1-2 sentences. Call tools first, speak after. "
        "Output length is irrelevant until final_answer is reached.\n\n"
        f"{memory_context}\n\n"
        f"{skill_context}\n\n"
        f"{knowledge_context}\n\n"
        "[TASK MODE] You MUST use tools to complete tasks. Treat agentic work as "
        "a sequence of steps, not one category: plan/decide when useful, research "
        "with web_search for snippet-only discovery and deep_search for fetched-page evidence when current or external facts are needed, "
        "inspect repository files for coding or architecture work, schedule with "
        "schedule_job or schedule_reminder when requested, and write or save the "
        "result when the user asks for an artifact. Research tasks should normally "
        "end in a written summary/report, even if the user only asked you to look "
        "something up, unless they explicitly ask you not to write it down. Never "
        "describe or simulate tool results in text — always call the actual tool. "
        "If the user asks you to save, write, schedule, or search: call the tool "
        "first, then confirm with final_answer. Do not call final_answer until all "
        "needed tool calls are complete. Keep reasoning private. Never write tool names "
        "or JSON in your spoken answer — speak naturally after the work is done. "
        "Tool observations are structured JSON. If ok=false, do not pretend the "
        "action succeeded: retry with corrected arguments, choose another tool or "
        "query, or clearly disclose the limitation in the final answer. "
        "Use deep_search at most once per agentic workflow. After deep_search returns, read its evidence and continue with the next productive step (plan, summarize, save, or answer) instead of searching again. Use web_search only for lightweight snippet discovery; snippets alone are not enough for verified detailed notes. "
        "When writing notes after research: cross-check any hardware specs, "
        "commands, or version numbers against fetched page content only — "
        "never state technical facts from memory alone. If a fact cannot be "
        "confirmed from fetched content, omit it or flag it as unverified. "
        "Use <skill_context> and <knowledge_context> when they match the task. For repeatable workflows, "
        "prefer the predefined skill's workflow and local knowledge and operating cards over inventing a new process. "
        "If no matching skill exists, continue with generic tools. "
        "CRITICAL: When asked to save a file, call save_note BEFORE writing "
        "any content in chat. Do not describe what you will save — just save it. "
        "Never say 'I'll now open a file' or 'I'll generate' — call the tool immediately. "
    )
    messages = [
        {"role": "system", "content": agent_system},
        {"role": "user", "content": user_input},
    ]
    owner.last_prompt_debug = {
        "mode": "agentic",
        "system_prompt": owner._persona,
        "memory_prompt": memory_context,
        "web_prompt": "",
        "agentic_prompts": [
            {"label": "agentic_policy", "content": agentic_policy_context},
            {"label": "wiki_context", "content": wiki_context},
            {"label": "skill_context", "content": skill_context},
            {"label": "knowledge_context", "content": knowledge_context},
            {"label": "task_mode_system", "content": agent_system},
        ],
        "previous_chat_messages": [],
    }
    owner.last_usage = {
        "prompt_messages": list(messages),
        "completion_text": "",
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
    }

    final_text = ""
    last_content = ""
    seen_calls: set[tuple[str, str]] = set()
    state = TaskState(goal=user_input)
    final_repairs = 0

    for step in range(MAX_AGENT_ITER):
        if token_callback:
            token_callback("__THINKING__\n")

        try:
            resp = owner._client.chat.completions.create(
                model=owner._llm_model, messages=messages, tools=tools,
                tool_choice="auto", stream=False, max_tokens=AGENT_MAX_TOKENS,
                temperature=0.3,
            )
            usage = getattr(resp, "usage", None)
            msg = resp.choices[0].message
            last_content = msg.content or ""
            owner.last_usage = {
                "prompt_messages": list(messages),
                "completion_text": last_content,
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }
            messages.append(msg.model_dump(exclude_none=True))
            _compact_processed_research_context(messages)
        except Exception as e:
            log.error("Agent LLM call failed: %s", e)
            state.record(ToolResult(
                ok=False, tool="llm_call", args={},
                content=f"[llm_call_failed: {e}]",
                error_type="llm_call_failed",
                retryable=False,
            ))
            break

        if not msg.tool_calls:
            candidate = msg.content or ""
            if AGENT_VERIFY_FINAL:
                verdict = _verify_final_answer(owner, user_input, candidate, state)
                if not verdict.ok and final_repairs < AGENT_MAX_FINAL_REPAIRS:
                    final_repairs += 1
                    messages.append({
                        "role": "user",
                        "content": (
                            "Verifier rejected the candidate final answer. "
                            "Repair the task or answer before finalizing.\n"
                            f"Verifier score: {verdict.score}\n"
                            f"Feedback:\n{verdict.feedback}\n\n"
                            f"Task ledger:\n{state.summary()}"
                        ),
                    })
                    continue
            final_text = candidate
            break

        for call in msg.tool_calls:
            name = call.function.name
            try:
                args = json.loads(call.function.arguments)
            except json.JSONDecodeError as e:
                args = {}
                result = ToolResult(
                    ok=False, tool=name, args=args,
                    content=f"Invalid JSON arguments: {e}. Reissue this tool call with valid JSON.",
                    error_type="invalid_json",
                    retryable=True,
                )
                state.record(result)
                messages.append({
                    "role": "tool", "tool_call_id": call.id,
                    "name": name, "content": result.observation(),
                })
                continue

            log.info("[agent] step %s → %s(%s)", step, name, args)
            if name == "deep_search" and _has_successful_tool_call(state, "deep_search"):
                result = ToolResult(
                    ok=False, tool=name, args=args,
                    content=(
                        "deep_search was already used in this agentic workflow. "
                        "Do not search again; use the fetched evidence already "
                        "processed in the prior step to plan, summarize, save, "
                        "or answer."
                    ),
                    error_type="deep_search_limit_reached",
                    retryable=False,
                )
                state.record(result)
                messages.append({
                    "role": "tool", "tool_call_id": call.id,
                    "name": name, "content": result.observation(),
                })
                continue
            call_key = (name, json.dumps(args, sort_keys=True))
            if name != "final_answer" and call_key in seen_calls:
                result = ToolResult(
                    ok=False, tool=name, args=args,
                    content=(
                        f"Repeated tool call skipped for {name}. Choose a different "
                        "query/argument/tool, or finalize with a disclosed limitation."
                    ),
                    error_type="repeated_tool_call",
                    retryable=True,
                )
                state.record(result)
                messages.append({
                    "role": "tool", "tool_call_id": call.id,
                    "name": name, "content": result.observation(),
                })
                continue
            if name != "final_answer":
                seen_calls.add(call_key)
            if token_callback:
                token_callback(f"__TOOL__:{name}({args})\n")

            if name == "final_answer":
                candidate = args.get("answer", "")
                if AGENT_VERIFY_FINAL:
                    verdict = _verify_final_answer(owner, user_input, candidate, state)
                    if not verdict.ok and final_repairs < AGENT_MAX_FINAL_REPAIRS:
                        final_repairs += 1
                        messages.append({
                            "role": "tool", "tool_call_id": call.id,
                            "name": name,
                            "content": json.dumps({
                                "ok": False,
                                "error_type": "verification_failed",
                                "score": verdict.score,
                                "feedback": verdict.feedback,
                                "task_ledger": json.loads(state.summary()),
                                "instruction": "Repair the missing/unsupported parts, then call final_answer again.",
                            }, ensure_ascii=False, indent=2),
                        })
                        continue
                final_text = candidate
                messages.append({
                    "role": "tool", "tool_call_id": call.id,
                    "name": name, "content": "Answer submitted.",
                })
                break

            result = execute_tool_with_policy(name, args, state)
            messages.append({
                "role": "tool", "tool_call_id": call.id,
                "name": name, "content": result.observation(),
            })

        if final_text:
            break

    if not final_text:
        log.warning(
            "Agent loop ended without a final answer after %s iterations; tools=%s failures=%s",
            MAX_AGENT_ITER, len(state.steps), len(state.failures),
        )
        final_text = _build_incomplete_task_answer(state, last_content)

    owner._emit(final_text, token_callback=token_callback)

    with owner._history_lock:
        owner._history.append({"role": "user", "content": user_input})
        owner._history.append({"role": "assistant", "content": final_text})

    owner._store_async(user_input, final_text)
    return final_text