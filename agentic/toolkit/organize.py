"""
toolkit/organize.py

Schedule and reminder tools.

This module provides job scheduling and reminder functionality:

  - schedule_job()       — schedule a local recurring job while Aiko is running
  - list_schedule()      — list all scheduled jobs
  - cancel_schedule()    — cancel a scheduled job
  - schedule_reminder()  — schedule a one-time reminder
  - list_reminders()     — list all reminders
  - cancel_reminder()    — cancel a reminder

Uses system.schedule for persistent record management.
"""

from __future__ import annotations

from system.schedule import (
    cancel_reminder_record,
    cancel_schedule_record,
    list_reminder_records,
    list_schedule_records,
    schedule_job_record,
    schedule_reminder_record,
)
from agentic.toolkit.common import json_block
from agentic.registry import tool


@tool(
    name="schedule_job",
    description="Schedule local job/alarm. HH:MM. Frequencies: once,hourly,daily,weekdays,weekly,biweekly,monthly,custom_weekdays. Supports relative_days for today/tomorrow/day-after-tomorrow offsets.",
    props={"title": {"type": "string"}, "task": {"type": "string"}, "time_of_day": {"type": "string", "description": "24-hour local time, e.g. 06:00"}, "frequency": {"type": "string", "enum": ["once", "hourly", "daily", "weekdays", "weekly", "biweekly", "monthly", "custom_weekdays"]}, "timezone": {"type": "string"}, "days_of_week": {"type": "string", "description": "Optional weekdays, e.g. Monday Wednesday Friday"}, "relative_days": {"type": "string", "description": "Optional day offset/phrase for the first due date, e.g. 0/today, 1/tomorrow, 2/day after tomorrow"}, "action": {"type": "string", "enum": ["announce", "agentic", "tool"], "description": "announce, agentic task, or direct registered tool invocation"}, "tool_call": {"type": "object", "description": "Required for action=tool: {name: registered tool name, arguments: object}"}, "skill": {"type": "string", "description": "Optional custom SKILL.md-style instructions for an agentic job"}},
    required=["title", "task", "time_of_day"],
    domain="scheduling",
    react=True,
    graph=True,
)
def schedule_job(
    title: str,
    task: str,
    time_of_day: str,
    frequency: str = "daily",
    timezone: str | None = None,
    days_of_week: list[str] | str | None = None,
    action: str = "agentic",
    relative_days: int | str | None = None,
    tool_call: dict | None = None,
    skill: str | None = None,
    user_id: str | None = None,
) -> str:
    """Schedule a local recurring job while Aiko is running."""
    try:
        job = schedule_job_record(
            title,
            task,
            time_of_day,
            frequency,
            timezone,
            days_of_week,
            action,
            relative_days,
            tool_call=tool_call,
            skill=skill,
            user_id=user_id,
        )
        # Notify the running scheduler so it picks up the new job immediately
        from system.schedule import notify_scheduler_new_job
        notify_scheduler_new_job()
        return json_block("scheduled job created", job)
    except Exception as e:
        return f"[schedule failed: {e}]"


@tool(
    name="list_schedule",
    description="List schedule.",
    props={"include_disabled": {"type": "boolean"}},
    domain="scheduling",
    react=True,
    graph=True,
)
def list_schedule(include_disabled: bool = False, user_id: str | None = None) -> str:
    """List local scheduled jobs from Aiko's schedule file."""
    jobs = list_schedule_records(include_disabled=include_disabled, user_id=user_id)
    return json_block("schedule", {"count": len(jobs), "items": jobs})


@tool(
    name="cancel_schedule",
    description="Cancel schedule item.",
    props={"job_id": {"type": "string"}},
    required=["job_id"],
    domain="scheduling",
    react=True,
    graph=True,
)
def cancel_schedule(job_id: str, user_id: str | None = None) -> str:
    """Cancel/disable a local scheduled job by id."""
    if cancel_schedule_record(job_id, user_id=user_id):
        return json_block("scheduled job cancelled", {"id": job_id})
    return f"[scheduled job not found: {job_id}]"


@tool(
    name="schedule_reminder",
    description="Simple once/daily reminder.",
    props={"title": {"type": "string"}, "message": {"type": "string"}, "time_of_day": {"type": "string"}, "repeat": {"type": "string", "enum": ["once", "daily"]}, "timezone": {"type": "string"}},
    required=["title", "message", "time_of_day"],
    domain="scheduling",
    react=True,
    graph=True,
)
def schedule_reminder(
    title: str,
    message: str,
    time_of_day: str,
    repeat: str = "daily",
    timezone: str | None = None,
    user_id: str | None = None,
) -> str:
    """Schedule a local reminder/alarm while Aiko is running."""
    try:
        reminder = schedule_reminder_record(title, message, time_of_day, repeat, timezone, user_id=user_id)
        # Notify the running scheduler so it picks up the new reminder immediately
        from system.schedule import notify_scheduler_new_job
        notify_scheduler_new_job()
        return json_block("reminder scheduled", reminder)
    except Exception as e:
        return f"[reminder failed: {e}]"


@tool(
    name="list_reminders",
    description="List reminders.",
    props={"include_disabled": {"type": "boolean"}},
    domain="scheduling",
    react=True,
    graph=True,
)
def list_reminders(include_disabled: bool = False, user_id: str | None = None) -> str:
    """List reminders stored in Aiko's local reminder file."""
    reminders = list_reminder_records(include_disabled=include_disabled, user_id=user_id)
    return json_block("reminders", {"count": len(reminders), "items": reminders})


@tool(
    name="cancel_reminder",
    description="Cancel reminder by id.",
    props={"reminder_id": {"type": "string"}},
    required=["reminder_id"],
    domain="scheduling",
    react=True,
    graph=True,
)
def cancel_reminder(reminder_id: str, user_id: str | None = None) -> str:
    """Cancel/disable a local reminder by id."""
    if cancel_reminder_record(reminder_id, user_id=user_id):
        return json_block("reminder cancelled", {"id": reminder_id})
    return f"[reminder not found: {reminder_id}]"
