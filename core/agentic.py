"""
core/agentic.py

Aiko's task-mode loop: tool schemas, ReAct-style dispatch, and final response
handling. Pure tool implementations stay in core/tools.py; chat facade, TTS,
history, and memory queue ownership stay in core/think.py.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field

from core.log import get_logger
from core.skills import list_skillsets, load_skillset, search_skillsets_json, skill_context_for
from core.tools import (
    fetch_and_extract,
    deep_search,
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
)

log = get_logger(__name__)

MAX_AGENT_ITER = int(os.getenv("MAX_AGENT_ITER", 8))
AGENT_MAX_TOKENS = int(os.getenv("AGENT_MAX_TOKENS", os.getenv("LLM_MAX_TOKENS", 512)))
AGENT_MEMORY_DRAIN_TIMEOUT = float(os.getenv("MEMORY_AGENT_DRAIN_TIMEOUT", 0.25))
AGENT_MEMORY_RECALL_LIMIT = int(os.getenv("AGENT_MEMORY_RECALL_LIMIT", min(int(os.getenv("MEMORY_RECALL_LIMIT", 3)), 2)))
AGENT_NOTE_MAX_CHARS = int(os.getenv("AGENT_NOTE_MAX_CHARS", 1500))
AGENT_TOOL_RESULT_MAX_CHARS = int(os.getenv("AGENT_TOOL_RESULT_MAX_CHARS", 3000))
AGENT_VERIFY_FINAL = os.getenv("AGENT_VERIFY_FINAL", "1").lower() in {"1", "true", "yes", "on"}
AGENT_VERIFY_LLM = os.getenv("AGENT_VERIFY_LLM", "1").lower() in {"1", "true", "yes", "on"}
AGENT_MAX_FINAL_REPAIRS = int(os.getenv("AGENT_MAX_FINAL_REPAIRS", 2))
AGENT_TOOL_RETRY_BACKOFF = float(os.getenv("AGENT_TOOL_RETRY_BACKOFF", 0.4))

_ERROR_PREFIX_RE = re.compile(r"^\[(?P<label>[^\]:]+)(?::\s*(?P<detail>.*))?\]$", re.DOTALL)
_DISCLOSURE_RE = re.compile(
    r"\b(couldn'?t|cannot|can't|failed|unavailable|not available|limitation|"
    r"could not|wasn'?t able|unable|unverified|not verified|partial)\b",
    re.IGNORECASE,
)
_EXTERNAL_ACTION_RE = re.compile(r"\b(send|sent|email|post|posted|buy|bought|book|booked|order|ordered|delete|deleted)\b", re.IGNORECASE)
_LOCAL_ARTIFACT_RE = re.compile(r"\b(saved|created|scheduled|cancelled|path|id|draft|note|workspace)\b", re.IGNORECASE)


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
            "name": "web_search", "description": "Search web.",
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string", "description": "The search query."}},
                "required": ["query"]}}},
        {"type": "function", "function": {
            "name": "fetch_page", "description": "Fetch page text.",
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
            "name": "final_answer", "description": "Final answer.",
            "parameters": {"type": "object", "properties": {
                "answer": {"type": "string", "description": "The final answer text."}},
                "required": ["answer"]}}},
    ]


def _register_tools() -> None:
    handlers = {
        "web_search": lambda args: deep_search(args.get("query", "")),
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
    if name in {"web_search", "fetch_page"}:
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


def _verify_final_answer(owner, user_input: str, answer: str, state: TaskState) -> VerificationResult:
    """Check answer completeness and faithfulness before Aiko speaks it."""
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
        "You are Aiko's verifier. Check whether the candidate answer is accurate, complete, "
        "and supported by the task ledger. Return ONLY compact JSON with keys: "
        "pass (boolean), score (0-1), feedback (string). Do not add markdown.\n\n"
        f"User request:\n{user_input}\n\n"
        f"Task ledger:\n{state.summary()}\n\n"
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
        ok = bool(data.get("pass"))
        score = float(data.get("score", 1.0 if ok else 0.0))
        feedback = str(data.get("feedback") or ("Verifier passed." if ok else "Verifier failed."))
        return VerificationResult(ok=ok, feedback=feedback, score=score)
    except Exception as e:
        log.warning("Agent verifier failed; falling back to deterministic pass: %s", e)
        return VerificationResult(ok=True, feedback="Verifier unavailable; deterministic checks passed.", score=0.75)


def run_agentic_chat(owner, user_input: str, token_callback=None) -> str:
    """Run task mode using the owning AikoThink instance for model/memory/output."""
    tools = tool_schemas()

    if not owner.wait_for_memory(timeout=AGENT_MEMORY_DRAIN_TIMEOUT):
        log.debug("Agent memory queue still draining; continuing without blocking turn start.")
    memories = owner._memorize.search(user_input, limit=AGENT_MEMORY_RECALL_LIMIT)
    memory_block = owner._memorize.format_for_context(memories)
    memory_context = memory_block or "<memory_context>\nNo relevant memories found.\n</memory_context>"
    skill_context = skill_context_for(user_input)

    agent_system = (
        f"{owner._persona}\n\n"
        "[TASK MODE OVERRIDE] The speech style limits in the persona do NOT apply "
        "in task mode. Do not summarize in 1-2 sentences. Call tools first, speak after. "
        "Output length is irrelevant until final_answer is reached.\n\n"
        f"{memory_context}\n\n"
        f"{skill_context}\n\n"
        "[TASK MODE] You MUST use tools to complete tasks. Never describe or "
        "simulate tool results in text — always call the actual tool. If the user "
        "asks you to save, write, schedule, or search: call the tool first, then "
        "confirm with final_answer. Do not call final_answer until all requested "
        "tool calls are complete. Keep reasoning private. Never write tool names "
        "or JSON in your spoken answer — speak naturally after the work is done. "
        "Tool observations are structured JSON. If ok=false, do not pretend the "
        "action succeeded: retry with corrected arguments, choose another tool or "
        "query, or clearly disclose the limitation in the final answer. "
        "When writing notes after research: cross-check any hardware specs, "
        "commands, or version numbers against fetched page content only — "
        "never state technical facts from memory alone. If a fact cannot be "
        "confirmed from fetched content, omit it or flag it as unverified. "
        "Use <skill_context> when it matches the task. For repeatable workflows, "
        "prefer the predefined skill's workflow and tools over inventing a new process. "
        "If no matching skill exists, continue with generic tools."
        "CRITICAL: When asked to save a file, call save_note BEFORE writing "
        "any content in chat. Do not describe what you will save — just save it. "
        "Never say 'I'll now open a file' or 'I'll generate' — call the tool immediately. "
    )
    messages = [
        {"role": "system", "content": agent_system},
        {"role": "user", "content": user_input},
    ]

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
            msg = resp.choices[0].message
            last_content = msg.content or ""
            messages.append(msg.model_dump(exclude_none=True))
        except Exception as e:
            log.error("Agent LLM call failed: %s", e)
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
        if state.failures:
            failed = "; ".join(f"{f.tool}: {f.content[:160]}" for f in state.failures[-3:])
            final_text = (
                "I got a bit lost trying to complete that task. "
                f"Recent blocker(s): {failed}\n"
                f"Here is what I have so far:\n{last_content}"
            )
        else:
            final_text = "I got a bit lost trying to complete that task. Here is what I have so far:\n" + last_content

    owner._emit(final_text, token_callback=token_callback)

    with owner._history_lock:
        owner._history.append({"role": "user", "content": user_input})
        owner._history.append({"role": "assistant", "content": final_text})

    owner._record_experience(user_input, final_text)
    owner._store_async(user_input, final_text)
    return final_text
