"""
core/agentic.py

Aiko's task-mode loop: tool schemas, ReAct-style dispatch, and final response
handling. Pure tool implementations stay in core/tools.py; chat facade, TTS,
history, and memory queue ownership stay in core/think.py.
"""

from __future__ import annotations

import json
import os

from core.log import get_logger
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
)

log = get_logger(__name__)

MAX_AGENT_ITER = int(os.getenv("MAX_AGENT_ITER", 8))
AGENT_MAX_TOKENS = int(os.getenv("AGENT_MAX_TOKENS", os.getenv("LLM_MAX_TOKENS", 512)))
AGENT_MEMORY_DRAIN_TIMEOUT = float(os.getenv("MEMORY_AGENT_DRAIN_TIMEOUT", 0.25))
AGENT_MEMORY_RECALL_LIMIT = int(os.getenv("AGENT_MEMORY_RECALL_LIMIT", min(int(os.getenv("MEMORY_RECALL_LIMIT", 3)), 2)))
AGENT_NOTE_MAX_CHARS = int(os.getenv("AGENT_NOTE_MAX_CHARS", 1500))


def _tool(schema: dict):
    """Decorator used to keep schemas and dispatch handlers in one registry."""
    def decorator(func):
        _TOOLS[schema["function"]["name"]] = (schema, func)
        return func
    return decorator


_TOOLS: dict[str, tuple[dict, object]] = {}


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
            "name": "schedule_job", "description": "Schedule local job/alarm. HH:MM. Frequencies: once,daily,weekdays,weekly,biweekly,monthly,custom_weekdays.",
            "parameters": {"type": "object", "properties": {
                "title": {"type": "string"}, "task": {"type": "string"},
                "time_of_day": {"type": "string", "description": "24-hour local time, e.g. 06:00"},
                "frequency": {"type": "string", "enum": ["once", "daily", "weekdays", "weekly", "biweekly", "monthly", "custom_weekdays"]},
                "timezone": {"type": "string"},
                "days_of_week": {"type": "string", "description": "Optional weekdays, e.g. Monday Wednesday Friday"},
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
        "schedule_job": lambda args: schedule_job(args.get("title", "Scheduled job"), args.get("task", "Scheduled job"), args.get("time_of_day", "06:00"), args.get("frequency", "daily"), args.get("timezone"), args.get("days_of_week"), args.get("action", "agentic")),
        "list_schedule": lambda args: list_schedule(bool(args.get("include_disabled", False))),
        "cancel_schedule": lambda args: cancel_schedule(args.get("job_id", "")),
        "schedule_reminder": lambda args: schedule_reminder(args.get("title", "Reminder"), args.get("message", "Reminder"), args.get("time_of_day", "06:00"), args.get("repeat", "daily"), args.get("timezone")),
        "list_reminders": lambda args: list_reminders(bool(args.get("include_disabled", False))),
        "cancel_reminder": lambda args: cancel_reminder(args.get("reminder_id", "")),
    }
    for schema in _TOOL_SCHEMAS:
        name = schema["function"]["name"]
        _TOOLS[name] = (schema, handlers.get(name, lambda _args, n=name: f"[unknown tool: {n}]"))


_register_tools()


def dispatch_tool(name: str, args: dict) -> str:
    """Run one named tool with already-decoded JSON args."""
    entry = _TOOLS.get(name)
    if not entry:
        return f"[unknown tool: {name}]"
    if name == "save_note":
        args["content"] = args.get("content", "")[:AGENT_NOTE_MAX_CHARS]
        args["title"] = args.get("title", "aiko-note")
    return entry[1](args)


def run_agentic_chat(owner, user_input: str, token_callback=None) -> str:
    """Run task mode using the owning AikoThink instance for model/memory/output."""
    tools = tool_schemas()

    if not owner.wait_for_memory(timeout=AGENT_MEMORY_DRAIN_TIMEOUT):
        log.debug("Agent memory queue still draining; continuing without blocking turn start.")
    memories = owner._memorize.search(user_input, limit=AGENT_MEMORY_RECALL_LIMIT)
    memory_block = owner._memorize.format_for_context(memories)
    memory_context = memory_block or "<memory_context>\nNo relevant memories found.\n</memory_context>"

    agent_system = (
        f"{owner._persona}\n\n"
        f"{memory_context}\n\n"
        "[TASK MODE] You MUST use tools to complete tasks. Never describe or "
        "simulate tool results in text — always call the actual tool. If the user "
        "asks you to save, write, schedule, or search: call the tool first, then "
        "confirm with final_answer. Do not call final_answer until all requested "
        "tool calls are complete. Keep reasoning private. Never write tool names "
        "or JSON in your spoken answer — speak naturally after the work is done. "
        "When writing notes after research: cross-check any hardware specs, "
        "commands, or version numbers against fetched page content only — "
        "never state technical facts from memory alone. If a fact cannot be "
        "confirmed from fetched content, omit it or flag it as unverified."
    )
    messages = [
        {"role": "system", "content": agent_system},
        {"role": "user", "content": user_input},
    ]

    final_text = ""
    last_content = ""
    seen_calls: set[tuple[str, str]] = set()

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
            final_text = msg.content or ""
            break

        for call in msg.tool_calls:
            name = call.function.name
            try:
                args = json.loads(call.function.arguments)
            except json.JSONDecodeError:
                args = {}

            log.info("[agent] step %s → %s(%s)", step, name, args)
            call_key = (name, json.dumps(args, sort_keys=True))
            if call_key in seen_calls:
                result = f"[loop guard: repeated tool call skipped for {name}]"
                messages.append({
                    "role": "tool", "tool_call_id": call.id,
                    "name": name, "content": result,
                })
                continue
            seen_calls.add(call_key)
            if token_callback:
                token_callback(f"__TOOL__:{name}({args})\n")

            if name == "final_answer":
                final_text = args.get("answer", "")
                messages.append({
                    "role": "tool", "tool_call_id": call.id,
                    "name": name, "content": "Answer submitted.",
                })
                break

            result = dispatch_tool(name, args)
            messages.append({
                "role": "tool", "tool_call_id": call.id,
                "name": name, "content": result[:3000],
            })

        if final_text:
            break

    if not final_text:
        final_text = "I got a bit lost trying to complete that task. Here is what I have so far:\n" + last_content

    owner._emit(final_text, token_callback=token_callback)

    with owner._history_lock:
        owner._history.append({"role": "user", "content": user_input})
        owner._history.append({"role": "assistant", "content": final_text})

    owner._record_experience(user_input, final_text)
    owner._store_async(user_input, final_text)
    return final_text
