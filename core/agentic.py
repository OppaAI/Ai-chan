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


def tool_schemas() -> list[dict]:
    """Return OpenAI-compatible tool schemas for autonomous task mode."""
    return [
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
            "name": "save_note", "description": "Save note/draft.",
            "parameters": {"type": "object", "properties": {
                "title": {"type": "string"},
                "content": {"type": "string"},
                "folder": {"type": "string"}},
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


def dispatch_tool(name: str, args: dict) -> str:
    """Run one named tool with already-decoded JSON args."""
    if name == "web_search":
        return deep_search(args.get("query", ""))
    if name == "fetch_page":
        return fetch_and_extract(args.get("url", ""))
    if name == "make_plan":
        return make_plan(args.get("goal", ""), args.get("constraints", ""), int(args.get("max_steps", 8) or 8))
    if name == "create_checklist":
        return create_checklist(args.get("title", "Checklist"), args.get("items", ""))
    if name == "save_note":
        return save_note(args.get("title", "Aiko note"), args.get("content", ""), args.get("folder", "notes"))
    if name == "read_workspace_file":
        return read_workspace_file(args.get("relative_path", ""))
    if name == "summarize_task_state":
        return summarize_task_state(args.get("goal", ""), args.get("done", ""), args.get("next_action", ""), args.get("risks", ""))
    if name == "schedule_job":
        return schedule_job(args.get("title", "Scheduled job"), args.get("task", "Scheduled job"), args.get("time_of_day", "06:00"), args.get("frequency", "daily"), args.get("timezone"), args.get("days_of_week"), args.get("action", "agentic"))
    if name == "list_schedule":
        return list_schedule(bool(args.get("include_disabled", False)))
    if name == "cancel_schedule":
        return cancel_schedule(args.get("job_id", ""))
    if name == "schedule_reminder":
        return schedule_reminder(args.get("title", "Reminder"), args.get("message", "Reminder"), args.get("time_of_day", "06:00"), args.get("repeat", "daily"), args.get("timezone"))
    if name == "list_reminders":
        return list_reminders(bool(args.get("include_disabled", False)))
    if name == "cancel_reminder":
        return cancel_reminder(args.get("reminder_id", ""))
    return f"[unknown tool: {name}]"


def run_agentic_chat(owner, user_input: str, token_callback=None) -> str:
    """Run task mode using the owning AikoThink instance for model/memory/output."""
    tools = tool_schemas()

    owner.wait_for_memory()
    memories = owner._memorize.search(user_input, limit=int(os.getenv("MEMORY_RECALL_LIMIT", 3)))
    memory_block = owner._memorize.format_for_context(memories)
    memory_context = memory_block or "<memory_context>\nNo relevant memories found.\n</memory_context>"

    agent_system = (
        f"{owner._persona}\n\n"
        f"{memory_context}\n\n"
        "[TASK MODE] Plan briefly, use tools only when useful, and finish "
        "with final_answer. Keep private reasoning private. Never claim work "
        "outside available tools was completed. Use memory context silently "
        "when it helps choose tools, interpret the request, or personalize the final answer."
    )
    messages = [
        {"role": "system", "content": agent_system},
        {"role": "user", "content": user_input},
    ]

    final_text = ""
    last_content = ""

    for step in range(MAX_AGENT_ITER):
        if token_callback:
            token_callback("__THINKING__\n")

        try:
            resp = owner._client.chat.completions.create(
                model=owner._llm_model, messages=messages, tools=tools,
                tool_choice="auto", stream=False, max_tokens=1024,
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
