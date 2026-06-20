"""
core/schedule.py

Persistent scheduled jobs, reminders, and wake-up alarms for Aiko.

A scheduled job is a small local record with:
  - time_of_day: local wall-clock time in HH:MM format
  - frequency: once, daily, weekdays, weekly, biweekly, monthly, or custom_weekdays
  - days_of_week: optional weekday names for custom_weekdays/weekly jobs
  - task: what Aiko should do or say when the job fires
  - action: announce or agentic

The scheduler is deliberately local-first: jobs are stored in JSON under
AIKO_WORKSPACE_ROOT and a daemon thread polls for due entries.  It can announce
or initiate jobs only while Aiko is running on an awake machine.  It does not
install OS-level cron jobs, wake a sleeping computer, or run after Aiko exits.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from core.log import get_logger

log = get_logger(__name__)

WORKSPACE_ROOT = Path(os.getenv("WORKSPACE_ROOT", "workspace")).resolve()
SCHEDULE_PATH = Path(os.getenv("SCHEDULE_PATH", WORKSPACE_ROOT / "schedule.json")).resolve()
LEGACY_REMINDERS_PATH = Path(os.getenv("REMINDERS_PATH", WORKSPACE_ROOT / "reminders.json")).resolve()
DEFAULT_TIMEZONE = os.getenv("TIMEZONE", "UTC")
POLL_SECONDS = float(os.getenv("SCHEDULE_POLL_SECONDS", 15))

FREQUENCIES = {"once", "daily", "weekdays", "weekly", "biweekly", "monthly", "custom_weekdays"}
_WEEKDAYS = {
    "monday": 0, "mon": 0,
    "tuesday": 1, "tue": 1, "tues": 1,
    "wednesday": 2, "wed": 2,
    "thursday": 3, "thu": 3, "thur": 3, "thurs": 3,
    "friday": 4, "fri": 4,
    "saturday": 5, "sat": 5,
    "sunday": 6, "sun": 6,
}


def _timezone(name: str | None = None) -> ZoneInfo:
    """Return a ZoneInfo object, falling back to UTC when the name is invalid."""
    try:
        return ZoneInfo(name or DEFAULT_TIMEZONE)
    except ZoneInfoNotFoundError:
        log.warning("Unknown timezone %s; falling back to UTC", name or DEFAULT_TIMEZONE)
        return ZoneInfo("UTC")


def _now(tz_name: str | None = None) -> datetime:
    """Return the current timezone-aware datetime for schedule calculations."""
    return datetime.now(_timezone(tz_name))


def _parse_time_of_day(time_of_day: str) -> tuple[int, int]:
    """Parse HH:MM or H:MM into hour/minute integers."""
    hour_text, sep, minute_text = time_of_day.strip().partition(":")
    if not sep:
        raise ValueError("time_of_day must be HH:MM")
    hour = int(hour_text)
    minute = int(minute_text)
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError("time_of_day must be a valid 24-hour time")
    return hour, minute


def _normalize_weekdays(days_of_week: list[str] | str | None) -> list[int]:
    """Normalize weekday names/integers into sorted Python weekday numbers."""
    if days_of_week is None:
        return []
    if isinstance(days_of_week, str):
        parts = [p.strip().lower() for p in days_of_week.replace(",", " ").split()]
    else:
        parts = [str(p).strip().lower() for p in days_of_week]
    days: set[int] = set()
    for part in parts:
        if not part:
            continue
        if part.isdigit():
            value = int(part)
            if 0 <= value <= 6:
                days.add(value)
                continue
        if part not in _WEEKDAYS:
            raise ValueError(f"unknown weekday: {part}")
        days.add(_WEEKDAYS[part])
    return sorted(days)


def _candidate_at(now: datetime, time_of_day: str) -> datetime:
    """Return today's candidate datetime at a given wall-clock time."""
    hour, minute = _parse_time_of_day(time_of_day)
    return now.replace(hour=hour, minute=minute, second=0, microsecond=0)


def _next_for_weekdays(time_of_day: str, weekdays: list[int], tz_name: str | None = None) -> datetime:
    """Return the next datetime matching one of the requested weekdays."""
    if not weekdays:
        raise ValueError("days_of_week is required for this frequency")
    now = _now(tz_name)
    base = _candidate_at(now, time_of_day)
    for offset in range(0, 14):
        candidate = base + timedelta(days=offset)
        if candidate.weekday() in weekdays and candidate > now:
            return candidate
    raise ValueError("could not calculate next weekday occurrence")


def _next_monthly(time_of_day: str, tz_name: str | None = None, anchor_day: int | None = None) -> datetime:
    """Return the next monthly occurrence on the anchor day, clamped to month length."""
    import calendar

    now = _now(tz_name)
    anchor = anchor_day or now.day
    hour, minute = _parse_time_of_day(time_of_day)
    year, month = now.year, now.month
    for _ in range(14):
        last_day = calendar.monthrange(year, month)[1]
        day = min(anchor, last_day)
        candidate = now.replace(year=year, month=month, day=day, hour=hour, minute=minute, second=0, microsecond=0)
        if candidate > now:
            return candidate
        month += 1
        if month > 12:
            month = 1
            year += 1
    raise ValueError("could not calculate next monthly occurrence")


def calculate_next_due(
    time_of_day: str,
    frequency: str = "daily",
    timezone: str | None = None,
    days_of_week: list[str] | str | None = None,
    after: datetime | None = None,
    anchor_day: int | None = None,
) -> datetime:
    """Calculate the next due datetime for a scheduled job."""
    frequency = (frequency or "daily").lower().strip()
    if frequency not in FREQUENCIES:
        raise ValueError(f"frequency must be one of: {', '.join(sorted(FREQUENCIES))}")

    tz_name = timezone or DEFAULT_TIMEZONE
    now = after.astimezone(_timezone(tz_name)) if after else _now(tz_name)
    candidate = _candidate_at(now, time_of_day)

    if frequency in {"once", "daily"}:
        return candidate if candidate > now else candidate + timedelta(days=1)
    if frequency == "weekdays":
        return _next_for_weekdays(time_of_day, [0, 1, 2, 3, 4], tz_name)
    if frequency == "custom_weekdays":
        return _next_for_weekdays(time_of_day, _normalize_weekdays(days_of_week), tz_name)
    if frequency == "weekly":
        weekdays = _normalize_weekdays(days_of_week) or [now.weekday()]
        return _next_for_weekdays(time_of_day, weekdays, tz_name)
    if frequency == "biweekly":
        base = candidate if candidate > now else candidate + timedelta(days=14)
        return base
    if frequency == "monthly":
        return _next_monthly(time_of_day, tz_name, anchor_day=anchor_day)
    raise ValueError(f"unsupported frequency: {frequency}")


def _read_raw(path: Path) -> list[dict]:
    """Read schedule JSON from a path, returning [] when absent/invalid."""
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception as e:
        log.error("Failed reading schedule file %s: %s", path, e)
        return []


def _migrate_legacy_reminders(records: list[dict]) -> list[dict]:
    """Convert older reminder records into the scheduled-job shape."""
    migrated = []
    for record in records:
        migrated.append({
            "id": record.get("id", uuid.uuid4().hex[:12]),
            "title": record.get("title", "Reminder"),
            "task": record.get("message", record.get("title", "Reminder")),
            "time_of_day": record.get("time_of_day", "06:00"),
            "frequency": "daily" if record.get("repeat") == "daily" else "once",
            "days_of_week": [],
            "timezone": record.get("timezone", DEFAULT_TIMEZONE),
            "next_due": record.get("next_due"),
            "created_at": record.get("created_at", _now(record.get("timezone")).isoformat()),
            "last_ran_at": None,
            "enabled": record.get("enabled", True),
            "kind": "reminder",
            "action": "announce",
        })
    return migrated


def _read_all() -> list[dict]:
    """Read scheduled jobs, including one-time migration from reminders.json."""
    records = _read_raw(SCHEDULE_PATH)
    if records:
        return records
    legacy = _read_raw(LEGACY_REMINDERS_PATH)
    if legacy:
        records = _migrate_legacy_reminders(legacy)
        _write_all(records)
    return records


def _write_all(jobs: list[dict]) -> None:
    """Persist scheduled jobs atomically enough for a single local process."""
    SCHEDULE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = SCHEDULE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(jobs, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(SCHEDULE_PATH)


def schedule_job_record(
    title: str,
    task: str,
    time_of_day: str,
    frequency: str = "daily",
    timezone: str | None = None,
    days_of_week: list[str] | str | None = None,
    action: str = "agentic",
) -> dict:
    """Create and persist a scheduled job record, returning the stored dict."""
    action = (action or "agentic").lower().strip()
    if action not in {"announce", "agentic"}:
        raise ValueError("action must be 'announce' or 'agentic'")
    tz_name = timezone or DEFAULT_TIMEZONE
    normalized_days = _normalize_weekdays(days_of_week)
    due = calculate_next_due(time_of_day, frequency, tz_name, normalized_days)
    job = {
        "id": uuid.uuid4().hex[:12],
        "title": title.strip() or "Scheduled job",
        "task": task.strip() or title.strip() or "Scheduled job",
        "time_of_day": time_of_day,
        "frequency": (frequency or "daily").lower().strip(),
        "days_of_week": normalized_days,
        "timezone": tz_name,
        "next_due": due.isoformat(),
        "created_at": _now(tz_name).isoformat(),
        "last_ran_at": None,
        "enabled": True,
        "kind": "scheduled_job",
        "action": action,
    }
    jobs = _read_all()
    jobs.append(job)
    _write_all(jobs)
    return job


def list_schedule_records(include_disabled: bool = False) -> list[dict]:
    """Return persisted scheduled jobs, optionally including disabled records."""
    jobs = _read_all()
    if include_disabled:
        return jobs
    return [job for job in jobs if job.get("enabled", True)]


def cancel_schedule_record(job_id: str) -> bool:
    """Disable a scheduled job by id; returns True when a matching record changed."""
    changed = False
    jobs = _read_all()
    for job in jobs:
        if job.get("id") == job_id:
            job["enabled"] = False
            changed = True
    if changed:
        _write_all(jobs)
    return changed


# Backwards-compatible reminder names used by older tools/tests.
def schedule_reminder_record(title: str, message: str, time_of_day: str, repeat: str = "daily", timezone: str | None = None) -> dict:
    """Compatibility wrapper: schedule a reminder as a scheduled job."""
    frequency = "daily" if repeat == "daily" else "once"
    return schedule_job_record(title, message, time_of_day, frequency, timezone, action="announce")


def list_reminder_records(include_disabled: bool = False) -> list[dict]:
    """Compatibility wrapper: list scheduled jobs."""
    return list_schedule_records(include_disabled)


def cancel_reminder_record(reminder_id: str) -> bool:
    """Compatibility wrapper: cancel a scheduled job by id."""
    return cancel_schedule_record(reminder_id)


@dataclass
class DueJob:
    """A scheduled job event ready to announce or execute."""
    id: str
    title: str
    task: str
    action: str = "agentic"


DueReminder = DueJob


class ScheduleRunner:
    """Background poller that announces due scheduled jobs while Aiko is running."""

    def __init__(self, on_due: Callable[[DueJob], None] | None = None) -> None:
        self._on_due = on_due
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the daemon scheduler thread if it is not already running."""
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name="aiko-schedule", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Request scheduler shutdown."""
        self._stop.set()

    def _run(self) -> None:
        """Poll schedule storage and announce due records."""
        while not self._stop.is_set():
            try:
                self._fire_due()
            except Exception as e:
                log.error("Schedule runner failed: %s", e)
            self._stop.wait(POLL_SECONDS)

    def _fire_due(self) -> None:
        """Find due jobs, reschedule recurring ones, and disable one-shots."""
        jobs = _read_all()
        changed = False
        due_events: list[DueJob] = []

        for job in jobs:
            if not job.get("enabled", True):
                continue
            tz_name = job.get("timezone") or DEFAULT_TIMEZONE
            try:
                due = datetime.fromisoformat(job["next_due"])
                if due.tzinfo is None:
                    due = due.replace(tzinfo=_timezone(tz_name))
            except Exception:
                due = calculate_next_due(
                    job.get("time_of_day", "06:00"),
                    job.get("frequency", "daily"),
                    tz_name,
                    job.get("days_of_week"),
                )
                job["next_due"] = due.isoformat()
                changed = True

            if due <= _now(tz_name):
                due_events.append(DueJob(
                    id=job.get("id", ""),
                    title=job.get("title", "Scheduled job"),
                    task=job.get("task", "Scheduled job"),
                    action=job.get("action", "agentic"),
                ))
                job["last_ran_at"] = _now(tz_name).isoformat()
                if job.get("frequency") == "once":
                    job["enabled"] = False
                else:
                    job["next_due"] = calculate_next_due(
                        job.get("time_of_day", "06:00"),
                        job.get("frequency", "daily"),
                        tz_name,
                        job.get("days_of_week"),
                        after=_now(tz_name),
                    ).isoformat()
                changed = True

        if changed:
            _write_all(jobs)
        for event in due_events:
            if self._on_due:
                self._on_due(event)


ReminderScheduler = ScheduleRunner
