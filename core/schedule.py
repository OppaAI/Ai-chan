"""
core/schedule.py

Persistent scheduled jobs, reminders, and wake-up alarms for Aiko.

A scheduled job is a small local record with:
  - time_of_day: local wall-clock time in HH:MM format
  - frequency: once, hourly, daily, weekdays, weekly, biweekly, monthly, or custom_weekdays
  - days_of_week: optional weekday names for custom_weekdays/weekly jobs
  - relative_days: optional integer day offset for phrases like tomorrow or the day after tomorrow
  - task: what Aiko should do or say when the job fires
  - action: announce or agentic

The scheduler is deliberately local-first: jobs are stored in JSON under
WORKSPACE_ROOT and a single daemon thread sleeps until the next due event.
It can announce or initiate jobs only while Aiko is running on an awake machine.
It does not install OS-level cron jobs, wake a sleeping computer, or run after
Aiko exits.

Two hardcoded system jobs run outside schedule.json and cannot be modified
by the user:
  - daily_reflect_and_dream    fires every day at DAILY_JOB_HOUR:DAILY_JOB_MINUTE (default 00:00)
  - monthly_consolidate        fires on the 1st of each month at MONTHLY_JOB_HOUR:MONTHLY_JOB_MINUTE (default 00:05)

Other system-style behaviors (e.g. weekly_social) live entirely in
schedule.json as ordinary jobs, but instead of routing through on_due/chat,
they name a "handler" — a Python callable registered once at startup via
register_system_handler(). schedule.json can only ever select a handler
from that pre-registered allowlist; it can never name or execute an
arbitrary function. This lets timing/enable/disable be fully data-driven
(edit schedule.json, no code change, no restart needed if the caller
notifies the scheduler) while the actual behavior each handler runs is
still something a human explicitly wired up in code.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from core.log import get_logger
from core.user_context import user_workspace_root

log = get_logger(__name__)

WORKSPACE_ROOT = Path(os.getenv("WORKSPACE_ROOT") or user_workspace_root()).resolve()
SCHEDULE_PATH = Path(os.getenv("SCHEDULE_PATH") or WORKSPACE_ROOT / "schedule.json").resolve()
LEGACY_REMINDERS_PATH = Path(os.getenv("REMINDERS_PATH") or WORKSPACE_ROOT / "reminders.json").resolve()
DEFAULT_TIMEZONE = os.getenv("TIMEZONE", "UTC")

# System job timing — env overridable, not user-modifiable via schedule.json
DAILY_JOB_HOUR   = int(os.getenv("DAILY_JOB_HOUR",   "0"))
DAILY_JOB_MINUTE = int(os.getenv("DAILY_JOB_MINUTE", "0"))
MONTHLY_JOB_HOUR   = int(os.getenv("MONTHLY_JOB_HOUR",   "0"))
MONTHLY_JOB_MINUTE = int(os.getenv("MONTHLY_JOB_MINUTE", "5"))

FREQUENCIES = {"once", "hourly", "daily", "weekdays", "weekly", "biweekly", "monthly", "custom_weekdays"}
RELATIVE_DAY_ALIASES = {
    "today": 0,
    "tonight": 0,
    "tomorrow": 1,
    "tmr": 1,
    "tmrw": 1,
    "the day after tomorrow": 2,
    "day after tomorrow": 2,
    "overmorrow": 2,
}

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


def _normalize_relative_days(relative_days: int | str | None = None) -> int | None:
    """Normalize a relative day offset or phrase into an integer day count."""
    if relative_days is None or relative_days == "":
        return None
    if isinstance(relative_days, int):
        days = relative_days
    else:
        text = str(relative_days).strip().lower().replace("-", " ")
        if text in RELATIVE_DAY_ALIASES:
            days = RELATIVE_DAY_ALIASES[text]
        else:
            days = int(text)
    if not (0 <= days <= 366):
        raise ValueError("relative_days must be between 0 and 366")
    return days


def _candidate_at(now: datetime, time_of_day: str, relative_days: int | str | None = None) -> datetime:
    """Return the candidate datetime at a wall-clock time, optionally offset by days."""
    hour, minute = _parse_time_of_day(time_of_day)
    days = _normalize_relative_days(relative_days) or 0
    base = now + timedelta(days=days)
    return base.replace(hour=hour, minute=minute, second=0, microsecond=0)


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

def _reflection_post_exists(date: datetime) -> bool:
    """Check if a reflection post already exists on GitHub for the given date."""
    import os, requests
    token = os.getenv("GITHUB_TOKEN", "")
    repo  = os.getenv("GITHUB_REPO", "")
    branch = os.getenv("GITHUB_BRANCH", "main")
    hugo_path = os.getenv("HUGO_CONTENT_PATH", "content/posts")
    if not token or not repo:
        return False
    slug = date.strftime("%Y-%m-%d") + "-day-reflection"
    url  = f"https://api.github.com/repos/{repo}/contents/{hugo_path}/{slug}.md"
    resp = requests.get(url, headers={
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }, params={"ref": branch}, timeout=10)
    return resp.status_code == 200

def calculate_next_due(
    time_of_day: str,
    frequency: str = "daily",
    timezone: str | None = None,
    days_of_week: list[str] | str | None = None,
    after: datetime | None = None,
    anchor_day: int | None = None,
    relative_days: int | str | None = None,
) -> datetime:
    """Calculate the next due datetime for a scheduled job."""
    frequency = (frequency or "daily").lower().strip()
    if frequency not in FREQUENCIES:
        raise ValueError(f"frequency must be one of: {', '.join(sorted(FREQUENCIES))}")

    tz_name = timezone or DEFAULT_TIMEZONE
    now = after.astimezone(_timezone(tz_name)) if after else _now(tz_name)
    relative_offset = _normalize_relative_days(relative_days)
    candidate = _candidate_at(now, time_of_day, relative_offset)

    if frequency in {"once", "daily"}:
        return candidate if candidate > now else candidate + timedelta(days=1)
    if frequency == "hourly":
        _, minute = _parse_time_of_day(time_of_day)
        hourly_candidate = now.replace(minute=minute, second=0, microsecond=0)
        if relative_offset:
            hourly_candidate = candidate
        return hourly_candidate if hourly_candidate > now else hourly_candidate + timedelta(hours=1)
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
    relative_days: int | str | None = None,
) -> dict:
    """Create and persist a scheduled job record, returning the stored dict."""
    action = (action or "agentic").lower().strip()
    if action not in {"announce", "agentic"}:
        raise ValueError("action must be 'announce' or 'agentic'")
    tz_name = timezone or DEFAULT_TIMEZONE
    normalized_days = _normalize_weekdays(days_of_week)
    normalized_relative_days = _normalize_relative_days(relative_days)
    due = calculate_next_due(
        time_of_day,
        frequency,
        tz_name,
        normalized_days,
        relative_days=normalized_relative_days,
    )
    job = {
        "id": uuid.uuid4().hex[:12],
        "title": title.strip() or "Scheduled job",
        "task": task.strip() or title.strip() or "Scheduled job",
        "time_of_day": time_of_day,
        "frequency": (frequency or "daily").lower().strip(),
        "days_of_week": normalized_days,
        "relative_days": normalized_relative_days,
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


# ── scheduler instance registry ───────────────────────────────────────────────

_scheduler_instance: ScheduleRunner | None = None


def register_scheduler(scheduler: ScheduleRunner) -> None:
    """Register the active scheduler instance so tools can notify it of new jobs."""
    global _scheduler_instance
    _scheduler_instance = scheduler


def notify_scheduler_new_job() -> None:
    """Notify the scheduler that a new job was added, so it wakes early to pick it up."""
    if _scheduler_instance is not None:
        _scheduler_instance.notify_new_job()


# ── system handler registry ───────────────────────────────────────────────────
# Allows schedule.json jobs to trigger a real Python function on fire, without
# giving the JSON file the ability to name or execute arbitrary code. Only
# names registered here via register_system_handler() at startup can ever be
# invoked — a job in schedule.json can select a handler, never define one.

_SYSTEM_HANDLERS: dict[str, Callable[[Any], Any]] = {}


def register_system_handler(name: str, fn: Callable[[Any], Any]) -> None:
    """Register a callable that a schedule.json job can reference by name.

    `fn` is called as fn(memorize) when a job with matching "handler" fires.
    Call this once at startup for each system-style behavior you want
    schedule.json to be able to schedule (e.g. weekly_social).
    """
    _SYSTEM_HANDLERS[name] = fn


@dataclass
class DueJob:
    """A scheduled job event ready to announce or execute."""
    id: str
    title: str
    task: str
    action: str = "agentic"


DueReminder = DueJob

# ── system job timing ─────────────────────────────────────────────────────────

def _next_daily_reflect_and_dream() -> datetime:
    """Next wall-clock occurrence of the daily reflect+dream window."""
    tz  = _timezone()
    now = datetime.now(tz)
    candidate = now.replace(
        hour=DAILY_JOB_HOUR,
        minute=DAILY_JOB_MINUTE,
        second=0,
        microsecond=0,
    )
    if candidate <= now:
        candidate += timedelta(days=1)
    return candidate


def _next_monthly_consolidate() -> datetime:
    """Next 1st-of-month occurrence of the monthly consolidation window."""
    tz  = _timezone()
    now = datetime.now(tz)
    # advance to next month's 1st
    if now.month == 12:
        first = now.replace(year=now.year + 1, month=1, day=1,
                            hour=MONTHLY_JOB_HOUR, minute=MONTHLY_JOB_MINUTE,
                            second=0, microsecond=0)
    else:
        first = now.replace(month=now.month + 1, day=1,
                            hour=MONTHLY_JOB_HOUR, minute=MONTHLY_JOB_MINUTE,
                            second=0, microsecond=0)
    return first


# ── scheduler ─────────────────────────────────────────────────────────────────

class ScheduleRunner:
    """
    Single daemon thread that sleeps until the next due event.

    Two hardcoded system jobs are managed internally and never written to
    schedule.json:
      - daily_reflect_and_dream   every day at DAILY_JOB_HOUR:DAILY_JOB_MINUTE
      - monthly_consolidate       every 1st of month at MONTHLY_JOB_HOUR:MONTHLY_JOB_MINUTE

    User reminders and scheduled jobs are read from schedule.json. Jobs with
    a "handler" field call into a registered Python function directly
    (see register_system_handler) instead of going through on_due/chat.

    The thread sleeps until the soonest of all targets, waking early only
    when notify_new_job() is called (e.g. after a new job is registered at
    runtime).
    """

    def __init__(
        self,
        on_due: Callable[[DueJob], None] | None = None,
        memorize=None,
        generate_and_post_fn: Callable | None = None,
        consolidate_fn: Callable | None = None,
    ) -> None:
        self._on_due               = on_due
        self._memorize             = memorize
        self._generate_and_post_fn = generate_and_post_fn
        self._consolidate_fn       = consolidate_fn
        self._wakeup               = threading.Event()
        self._stop                 = threading.Event()
        self._thread: threading.Thread | None = None

        # calculated once at startup, updated after each fire
        self._next_daily   = _next_daily_reflect_and_dream()
        self._next_monthly = _next_monthly_consolidate()

        # catch-up flag — checked on start()
        self._catchup_needed = self._check_catchup()


    def _check_catchup(self) -> bool:
        """
        Returns True if yesterday's reflection was missed.
        Missed = next_daily is more than 20 hours away AND no post exists for yesterday.
        """
        tz  = _timezone()
        now = datetime.now(tz)
        hours_until_next = (self._next_daily - now).total_seconds() / 3600
        if hours_until_next < 20:
            return False  # job fired recently enough, no catchup needed
        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0,
        )
        exists = _reflection_post_exists(yesterday)
        if not exists:
            log.info("Catch-up needed: no reflection post found for %s.", yesterday.date())
            return True
        return False

    def notify_new_job(self) -> None:
        """Interrupt the sleep early so a newly added job is picked up immediately."""
        self._wakeup.set()

    def start(self) -> None:
        """Start the daemon scheduler thread if it is not already running."""
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name="aiko-schedule", daemon=True)
        self._thread.start()
        log.info(
            "Scheduler started — daily_reflect_and_dream at %02d:%02d, "
            "monthly_consolidate on 1st at %02d:%02d.",
            DAILY_JOB_HOUR, DAILY_JOB_MINUTE,
            MONTHLY_JOB_HOUR, MONTHLY_JOB_MINUTE,
        )
        if self._catchup_needed and self._memorize and self._generate_and_post_fn:
            log.info("Scheduler: running missed daily reflect+dream on startup.")
            catchup_thread = threading.Thread(
                target=self._run_daily_reflect_and_dream,
                name="aiko-schedule-catchup",
                daemon=False,
            )
            catchup_thread.start()

    def stop(self) -> None:
        """Request scheduler shutdown."""
        self._stop.set()
        self._wakeup.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            now = datetime.now(_timezone())

            # ── fire overdue system jobs ──────────────────────────────────────
            system_due = sorted(
                [(t, name) for t, name in [
                    (self._next_daily, "daily"),
                    (self._next_monthly, "monthly"),
                ] if t <= now],
                key=lambda x: x[0],
            )
            for _target, name in system_due:
                if name == "daily":
                    self._run_daily_reflect_and_dream()
                    self._next_daily = _next_daily_reflect_and_dream()
                else:
                    self._run_monthly_consolidate()
                    self._next_monthly = _next_monthly_consolidate()

            # ── fire overdue user jobs ────────────────────────────────────────
            self._fire_due_user_jobs()

            # ── sleep until soonest next target ──────────────────────────────
            user_jobs = [
                datetime.fromisoformat(j["next_due"])
                for j in _read_all()
                if j.get("enabled", True)
            ]
            candidates = [self._next_daily, self._next_monthly, *user_jobs]
            next_target = min(candidates)

            delta = (next_target - datetime.now(_timezone())).total_seconds()
            if delta > 0:
                log.debug("Scheduler sleeping %.0fs until %s", delta, next_target.isoformat())
                self._wakeup.wait(timeout=delta)
                self._wakeup.clear()

    # ── system job runners ────────────────────────────────────────────────────

    def _run_daily_reflect_and_dream(self) -> None:
        """
        Hardcoded nightly job. Not in schedule.json. Not user-modifiable.

        Order:
          1. reflect  — LLM summary + image + GitHub push (reads memories before dream prunes)
          2. dream    — sqlite-vec consolidation, boost, merge, prune (no LLM)
        """
        if not self._memorize or not self._generate_and_post_fn:
            log.warning("daily_reflect_and_dream: memorize or generate_and_post_fn not set — skipping.")
            return

        try:
            log.info("daily_reflect_and_dream: starting.")

            tz = _timezone()
            now_local = datetime.now(tz)
            yesterday_local = (now_local - timedelta(days=1)).replace(
                hour=0, minute=0, second=0, microsecond=0,
            )
            yesterday_end_local = yesterday_local + timedelta(days=1)

            yesterday_query_start = yesterday_local.astimezone(timezone.utc)
            yesterday_query_end   = yesterday_end_local.astimezone(timezone.utc)

            yesterday = yesterday_local

            from core.reflect import REFLECT_MAX_MEMS
            memories = self._memorize.get_between(yesterday_query_start, yesterday_query_end)
            log.info("daily_reflect_and_dream: %d memories fetched.", len(memories))

            log.info("daily_reflect_and_dream: running reflect...")
            self._generate_and_post_fn(
                memories[:REFLECT_MAX_MEMS],
                date=yesterday,
                memorize=self._memorize,
            )
            log.info("daily_reflect_and_dream: reflect done.")

            log.info("daily_reflect_and_dream: running dream...")
            result = self._memorize.dream()
            log.info("daily_reflect_and_dream: dream done — %s", result)

        except Exception as e:
            log.error("daily_reflect_and_dream failed: %s", e)

    def _run_monthly_consolidate(self) -> None:
        """
        Hardcoded monthly job. Not in schedule.json. Not user-modifiable.
        Delegates entirely to consolidate.maybe_run_consolidation().
        """
        if not self._memorize or not self._consolidate_fn:
            log.warning("monthly_consolidate: memorize or consolidate_fn not set — skipping.")
            return

        try:
            log.info("monthly_consolidate: starting.")
            result = self._consolidate_fn(self._memorize, now=datetime.now(_timezone()))
            log.info("monthly_consolidate: done — %s", result)
        except Exception as e:
            log.error("monthly_consolidate failed: %s", e)

    # ── user job runner ───────────────────────────────────────────────────────

    def _fire_due_user_jobs(self) -> None:
        """Find due user jobs, reschedule recurring ones, disable one-shots.

        Jobs whose "handler" (or legacy "kind": "system_weekly_social") names
        a registered system handler call directly into that Python function
        instead of going through on_due/chat.
        """
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
                handler_name = job.get("handler") or (
                    "weekly_social" if job.get("kind") == "system_weekly_social" else None
                )
                if handler_name and handler_name in _SYSTEM_HANDLERS:
                    try:
                        _SYSTEM_HANDLERS[handler_name](self._memorize)
                    except Exception as e:
                        log.error("system handler %r failed: %s", handler_name, e)
                elif handler_name:
                    log.warning("job references unregistered handler %r — skipping fire.", handler_name)
                else:
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

        # fire sequentially — preserves order and avoids concurrent job side effects
        for event in due_events:
            if self._on_due:
                self._on_due(event)


ReminderScheduler = ScheduleRunner