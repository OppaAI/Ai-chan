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
from agentic.registry import TOOLS, tool


@tool(TOOLS["schedule_job"])
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


@tool(TOOLS["list_schedule"])
def list_schedule(include_disabled: bool = False, user_id: str | None = None) -> str:
    """List local scheduled jobs from Aiko's schedule file."""
    jobs = list_schedule_records(include_disabled=include_disabled, user_id=user_id)
    return json_block("schedule", {"count": len(jobs), "items": jobs})


@tool(TOOLS["cancel_schedule"])
def cancel_schedule(job_id: str, user_id: str | None = None) -> str:
    """Cancel/disable a local scheduled job by id."""
    if cancel_schedule_record(job_id, user_id=user_id):
        return json_block("scheduled job cancelled", {"id": job_id})
    return f"[scheduled job not found: {job_id}]"


@tool(TOOLS["schedule_reminder"])
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


@tool(TOOLS["list_reminders"])
def list_reminders(include_disabled: bool = False, user_id: str | None = None) -> str:
    """List reminders stored in Aiko's local reminder file."""
    reminders = list_reminder_records(include_disabled=include_disabled, user_id=user_id)
    return json_block("reminders", {"count": len(reminders), "items": reminders})


@tool(TOOLS["cancel_reminder"])
def cancel_reminder(reminder_id: str, user_id: str | None = None) -> str:
    """Cancel/disable a local reminder by id."""
    if cancel_reminder_record(reminder_id, user_id=user_id):
        return json_block("reminder cancelled", {"id": reminder_id})
    return f"[reminder not found: {reminder_id}]"
