"""
system/schedule.py

Persistent scheduled jobs, reminders, and wake-up alarms for Aiko.

A scheduled job is a small local record with:
  - time_of_day: local wall-clock time in HH:MM format
  - frequency: once, interval, hourly, daily, weekdays, weekly, biweekly, monthly, or custom_weekdays
  - days_of_week: optional weekday names for custom_weekdays/weekly jobs
  - relative_days: optional integer day offset for phrases like tomorrow or the day after tomorrow
  - task: what Aiko should do or say when the job fires
  - action: announce or agentic
  - handler: optional name of a pre-registered system handler (see
    register_system_handler) to call directly instead of going through
    on_due/chat — used for window-style jobs like the deep_studying
    start/stop pair (see ensure_deep_study_window_jobs), the periodic
    workspace/knowledge folder scan, and the periodic photo/video social
    inbox scans (see ensure_workspace_knowledge_job,
    ensure_photo_social_job, ensure_video_social_job).

The scheduler is deliberately local-first: jobs are stored in JSON under
WORKSPACE_ROOT and a single daemon thread sleeps until the next due event.
It can announce or initiate jobs only while Aiko is running on an awake machine.
It does not install OS-level cron jobs, wake a sleeping computer, or run after
Aiko exits.

Two hardcoded system jobs run outside schedule.json and cannot be modified
by the user:
  - daily_reflect_and_dream    fires every day at DAILY_JOB_HOUR:DAILY_JOB_MINUTE (default 00:00)
  - monthly_consolidate        fires on the 1st of each month at MONTHLY_JOB_HOUR:MONTHLY_JOB_MINUTE (default 00:05)

Both hardcoded jobs have startup catch-up logic: if the scheduler process
was offline/asleep across a scheduled firing, the missed run(s) are
detected and backfilled once on the next start() call, before the normal
sleep loop begins.

  - daily_reflect_and_dream: catch-up is detected per-date via
    _reflection_post_exists() (a live GitHub API check against the Hugo
    post path), scanned back up to CATCHUP_MAX_LOOKBACK_DAYS days. Every
    missing date found is backfilled sequentially via
    _run_catchup_backfill(), oldest first. The dream() consolidation pass
    only runs on the regular (non-catch-up) nightly call — see the
    for_date gate in _run_daily_reflect_and_dream — so a multi-day
    backfill doesn't trigger redundant consolidation passes.

  - monthly_consolidate: catch-up is detected via a small local state file
    (tasks/monthly_consolidate_state.json under the workspace root)
    recording the last "YYYY-MM" the job actually completed. If the
    current month doesn't match on startup and we're not still waiting
    for this month's scheduled window, one catch-up run fires.

Other system-style behaviors (e.g. weekly_social, photo_social, video_social,
deep_study_start/stop, workspace_knowledge_scan) live entirely in
schedule.json as ordinary jobs, but instead of routing through on_due/chat,
they name a "handler" — a Python callable registered once at startup via
register_system_handler(). schedule.json can only ever select a handler
from that pre-registered allowlist; it can never name or execute an
arbitrary function. This lets timing/enable/disable be fully data-driven
(edit schedule.json, no code change, no restart needed if the caller
notifies the scheduler) while the actual behavior each handler runs is
still something a human explicitly wired up in code.

Every registered handler is called as fn(memorize) — see
register_system_handler — even if the underlying function doesn't need
memorize (e.g. the photo/video social scans); those handlers just take and
ignore the argument, same convention as everything else the scheduler
fires.

Timezone resolution no longer lives here — every "now"/timezone lookup in
this file goes through system.bioclock, the app-wide single source of truth
(config/bioclock.yaml). A job may still carry its own "timezone" field
(e.g. a reminder scoped to a different zone than the app default); that
value is simply passed through to bioclock as an override rather than
resolved independently.
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

from system import bioclock
from system.log import get_logger
from system.userspace import current_user_id, user_state_path, user_workspace_root

log = get_logger(__name__)

def workspace_root() -> Path:
    """Resolve the active user workspace root lazily."""
    override = os.getenv("WORKSPACE_ROOT")
    return (Path(override).expanduser() if override else user_workspace_root()).resolve()


def user_state_root() -> Path:
    """Resolve the active user state root lazily."""
    override = os.getenv("USER_STATE_ROOT")
    return (Path(override).expanduser() if override else Path.home() / ".aiko").resolve()


def schedule_path() -> Path:
    """Resolve the active user schedule path lazily."""
    override = os.getenv("SCHEDULE_PATH")
    if override:
        return Path(override).expanduser().resolve()
    return (user_workspace_root() / "tasks" / "schedule.json").resolve()

# System job timing — env overridable, not user-modifiable via schedule.json
DAILY_JOB_HOUR   = int(os.getenv("DAILY_JOB_HOUR",   "0"))
DAILY_JOB_MINUTE = int(os.getenv("DAILY_JOB_MINUTE", "0"))
MONTHLY_JOB_HOUR   = int(os.getenv("MONTHLY_JOB_HOUR",   "0"))
MONTHLY_JOB_MINUTE = int(os.getenv("MONTHLY_JOB_MINUTE", "5"))

# How many days back to scan for missed daily_reflect_and_dream runs on
# scheduler startup. Bounded so a long outage doesn't trigger an unbounded
# GitHub API scan or an unbounded backfill run.
CATCHUP_MAX_LOOKBACK_DAYS = int(os.getenv("CATCHUP_MAX_LOOKBACK_DAYS", "7"))

# Filename for the small local state file tracking the last month
# monthly_consolidate actually completed. Lives under
# <workspace_root>/tasks/, alongside schedule.json.
MONTHLY_CATCHUP_STATE_PATH_NAME = "monthly_consolidate_state.json"

FREQUENCIES = {"once", "interval", "hourly", "daily", "weekdays", "weekly", "biweekly", "monthly", "custom_weekdays"}
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
    now = bioclock.local_now(tz_name)
    base = _candidate_at(now, time_of_day)
    for offset in range(0, 14):
        candidate = base + timedelta(days=offset)
        if candidate.weekday() in weekdays and candidate > now:
            return candidate
    raise ValueError("could not calculate next weekday occurrence")


def _next_monthly(time_of_day: str, tz_name: str | None = None, anchor_day: int | None = None) -> datetime:
    """Return the next monthly occurrence on the anchor day, clamped to month length."""
    import calendar

    now = bioclock.local_now(tz_name)
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


# ── monthly consolidate catch-up state ────────────────────────────────────────
# Small local marker (not schedule.json — schedule.json is user job storage)
# recording the last "YYYY-MM" monthly_consolidate actually completed. There
# is no external check available for this job (unlike daily reflect's GitHub
# post existence check), so the scheduler writes this itself on success.

def _monthly_state_path() -> Path:
    """Resolve the local monthly-consolidate catch-up state file path."""
    return (workspace_root() / "tasks" / MONTHLY_CATCHUP_STATE_PATH_NAME).resolve()


def _read_last_consolidated_month() -> str | None:
    """Return the last 'YYYY-MM' monthly_consolidate completed, or None."""
    path = _monthly_state_path()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("last_run_month")
    except Exception as e:
        log.error("Failed reading monthly consolidate state %s: %s", path, e)
        return None


def _write_last_consolidated_month(month_str: str) -> None:
    """Persist the 'YYYY-MM' that monthly_consolidate just completed for."""
    path = _monthly_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"last_run_month": month_str}), encoding="utf-8")
    tmp.replace(path)


def calculate_next_due(
    time_of_day: str,
    frequency: str = "daily",
    timezone: str | None = None,
    days_of_week: list[str] | str | None = None,
    after: datetime | None = None,
    anchor_day: int | None = None,
    relative_days: int | str | None = None,
    interval_seconds: int | str | None = None,
) -> datetime:
    """Calculate the next due datetime for a scheduled job."""
    frequency = (frequency or "daily").lower().strip()
    if frequency not in FREQUENCIES:
        raise ValueError(f"frequency must be one of: {', '.join(sorted(FREQUENCIES))}")

    tz_name = bioclock.timezone_name(timezone)
    now = after.astimezone(bioclock.get_timezone(tz_name)) if after else bioclock.local_now(tz_name)
    relative_offset = _normalize_relative_days(relative_days)
    candidate = _candidate_at(now, time_of_day, relative_offset)

    if frequency == "interval":
        seconds = int(interval_seconds or 60)
        if seconds < 60:
            raise ValueError("interval_seconds must be at least 60")
        return now + timedelta(seconds=seconds)

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


def _read_all() -> list[dict]:
    """Read scheduled jobs for the active user (cached)."""
    if _schedule_cache is not None:
        return _schedule_cache
    return _read_and_cache()

_schedule_cache: list[dict] | None = None
_schedule_dirty: bool = True

def _read_and_cache() -> list[dict]:
    global _schedule_cache, _schedule_dirty
    data = _read_raw(schedule_path())
    _schedule_cache = data
    _schedule_dirty = False
    return data

def _invalidate_cache() -> None:
    global _schedule_cache, _schedule_dirty
    _schedule_cache = None
    _schedule_dirty = True


def _write_all(jobs: list[dict]) -> None:
    """Persist scheduled jobs atomically enough for a single local process."""
    _invalidate_cache()
    path = schedule_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(jobs, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def schedule_job_record(
    title: str,
    task: str,
    time_of_day: str,
    frequency: str = "daily",
    timezone: str | None = None,
    days_of_week: list[str] | str | None = None,
    action: str = "agentic",
    relative_days: int | str | None = None,
    handler: str | None = None,
    interval_seconds: int | str | None = None,
) -> dict:
    """Create and persist a scheduled job record, returning the stored dict.

    `handler`, if given, must name a callable registered via
    register_system_handler() before this job ever fires. When set, the
    scheduler calls that handler directly instead of going through
    on_due/chat with `task` (see _fire_due_user_jobs) — `title`/`task` are
    still stored for readability/logging but are otherwise unused for
    handler-based jobs.
    """
    action = (action or "agentic").lower().strip()
    if action not in {"announce", "agentic"}:
        raise ValueError("action must be 'announce' or 'agentic'")
    tz_name = bioclock.timezone_name(timezone)
    normalized_days = _normalize_weekdays(days_of_week)
    normalized_relative_days = _normalize_relative_days(relative_days)
    due = calculate_next_due(
        time_of_day,
        frequency,
        tz_name,
        normalized_days,
        relative_days=normalized_relative_days,
        interval_seconds=interval_seconds,
    )
    job = {
        "id": uuid.uuid4().hex[:12],
        "title": title.strip() or "Scheduled job",
        "task": task.strip() or title.strip() or "Scheduled job",
        "time_of_day": time_of_day,
        "frequency": (frequency or "daily").lower().strip(),
        "days_of_week": normalized_days,
        "relative_days": normalized_relative_days,
        "interval_seconds": int(interval_seconds) if interval_seconds not in (None, "") else None,
        "timezone": tz_name,
        "next_due": due.isoformat(),
        "created_at": bioclock.local_now(tz_name).isoformat(),
        "last_ran_at": None,
        "enabled": True,
        "kind": "scheduled_job",
        "action": action,
        "handler": handler,
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


# ── deep-study window job seeding ─────────────────────────────────────────────
# These four jobs bound the wall-clock window in which the deep_studying
# handlers (registered in memory/learn.py — see register_deep_study_handlers)
# are allowed to run: weekdays 05:00-18:00, weekends 05:00-10:00. They are
# ordinary handler-based schedule.json jobs, not a new job type — the
# "window" behavior comes entirely from pairing a *_start job with a
# *_stop job on matching days, not from any scheduler-level concept of
# windows.

DEEP_STUDY_WINDOW_JOB_TITLES: dict[str, tuple[str, list[str], str]] = {
    "deep_study_weekday_start": ("05:00", ["mon", "tue", "wed", "thu", "fri"], "deep_study_start"),
    "deep_study_weekday_stop":  ("18:00", ["mon", "tue", "wed", "thu", "fri"], "deep_study_stop"),
    "deep_study_weekend_start": ("05:00", ["sat", "sun"], "deep_study_start"),
    "deep_study_weekend_stop":  ("10:00", ["sat", "sun"], "deep_study_stop"),
}


WORKSPACE_KNOWLEDGE_JOB_TITLE = "workspace_knowledge_scan"
WORKSPACE_KNOWLEDGE_SCAN_INTERVAL_SECONDS = int(os.getenv("WORKSPACE_KNOWLEDGE_SCAN_INTERVAL_SECONDS", "60"))


def ensure_workspace_knowledge_job(timezone: str | None = None) -> None:
    """Idempotently seed the scheduled KB folder scan job.

    The scheduler owns this periodic check so document-drop monitoring lives
    alongside other schedule.json-driven system behaviors instead of running
    a separate ticker thread.
    """
    existing_titles = {job.get("title") for job in _read_all()}
    if WORKSPACE_KNOWLEDGE_JOB_TITLE in existing_titles:
        return
    schedule_job_record(
        title=WORKSPACE_KNOWLEDGE_JOB_TITLE,
        task="Scan workspace/knowledge for new RAG documents",
        time_of_day="00:00",
        frequency="interval",
        timezone=timezone,
        action="agentic",
        handler="workspace_knowledge_scan",
        interval_seconds=max(60, WORKSPACE_KNOWLEDGE_SCAN_INTERVAL_SECONDS),
    )
    log.info("Seeded workspace knowledge scan job every %ss", max(60, WORKSPACE_KNOWLEDGE_SCAN_INTERVAL_SECONDS))


# ── social folder-monitoring job seeding ──────────────────────────────────────
# Lane A (weekly postcard) is a true weekly cadence job, not a folder scan —
# see ensure_weekly_social_job below, and agentic/toolkit/social.py's module
# docstring for why it stays out of the agent tool loop entirely.
#
# Lanes B/C (photo, video) are folder-drop workflows: there's no fixed
# cadence to "check the inbox", so — same pattern as
# ensure_workspace_knowledge_job above — they're seeded as interval jobs.
# Both run_scheduled_photo_social() and run_scheduled_video_social() take no
# arguments, but register_system_handler's calling convention always passes
# one positional arg (memorize), so the registered handler needs a one-line
# wrapper to absorb it — see register_social_handlers() below, which does
# the registration and seeding described in this comment automatically.

WEEKLY_SOCIAL_JOB_TITLE = "weekly_social_post"
# Runs once per week, the morning after a Sun-Sat window closes. The handler
# itself (run_scheduled_weekly_social) is idempotent per calendar week
# (generate_weekly_draft skips if a draft already exists), so a slightly
# early/late fire here is harmless.
WEEKLY_SOCIAL_TIME_OF_DAY = os.getenv("WEEKLY_SOCIAL_TIME_OF_DAY", "08:00")
WEEKLY_SOCIAL_RETRY_JOB_TITLE = "weekly_social_retry_check"
WEEKLY_SOCIAL_RETRY_INTERVAL_SECONDS = int(os.getenv("WEEKLY_SOCIAL_RETRY_INTERVAL_SECONDS", str(30 * 60)))

PHOTO_SOCIAL_JOB_TITLE = "photo_social_scan"
PHOTO_SOCIAL_SCAN_INTERVAL_SECONDS = int(os.getenv("PHOTO_SOCIAL_SCAN_INTERVAL_SECONDS", str(6 * 60 * 60)))  # 6h default

VIDEO_SOCIAL_JOB_TITLE = "video_social_scan"
VIDEO_SOCIAL_SCAN_INTERVAL_SECONDS = int(os.getenv("VIDEO_SOCIAL_SCAN_INTERVAL_SECONDS", str(6 * 60 * 60)))  # 6h default


def ensure_weekly_social_job(timezone: str | None = None) -> None:
    """Idempotently seed the weekly memory-postcard job (Lane A).

    Fires once a week; the handler itself decides which completed Sun-Sat
    window to draft from (see agentic.toolkit.social.last_completed_sunday_saturday),
    so the exact day/time here just needs to land safely after a week closes
    — it does not need to be precise.
    """
    existing_titles = {job.get("title") for job in _read_all()}
    if WEEKLY_SOCIAL_JOB_TITLE in existing_titles:
        return
    schedule_job_record(
        title=WEEKLY_SOCIAL_JOB_TITLE,
        task=WEEKLY_SOCIAL_JOB_TITLE,
        time_of_day=WEEKLY_SOCIAL_TIME_OF_DAY,
        frequency="weekly",
        timezone=timezone,
        days_of_week=["sun"],
        action="agentic",
        handler="weekly_social",
    )
    log.info("Seeded weekly social job (Sundays at %s)", WEEKLY_SOCIAL_TIME_OF_DAY)


def ensure_weekly_social_retry_job(timezone: str | None = None) -> None:
    """Idempotently seed the Sunday-bounded retry check for Lane A.

    Fires every WEEKLY_SOCIAL_RETRY_INTERVAL_SECONDS regardless of day; the
    handler itself (retry_weekly_social_if_needed) is what limits action to
    Sundays, so there's nothing day-specific to seed here.
    """
    existing_titles = {job.get("title") for job in _read_all()}
    if WEEKLY_SOCIAL_RETRY_JOB_TITLE in existing_titles:
        return
    schedule_job_record(
        title=WEEKLY_SOCIAL_RETRY_JOB_TITLE,
        task="Retry the weekly postcard if it hasn't posted yet (Sundays only)",
        time_of_day="00:00",
        frequency="interval",
        timezone=timezone,
        action="agentic",
        handler="weekly_social_retry",
        interval_seconds=max(60, WEEKLY_SOCIAL_RETRY_INTERVAL_SECONDS),
    )
    log.info("Seeded weekly social retry-check job every %ss", max(60, WEEKLY_SOCIAL_RETRY_INTERVAL_SECONDS))


def ensure_photo_social_job(timezone: str | None = None) -> None:
    """Idempotently seed the photo-inbox scan job (Lane B)."""
    existing_titles = {job.get("title") for job in _read_all()}
    if PHOTO_SOCIAL_JOB_TITLE in existing_titles:
        return
    schedule_job_record(
        title=PHOTO_SOCIAL_JOB_TITLE,
        task="Scan photo inbox for postable content",
        time_of_day="00:00",
        frequency="interval",
        timezone=timezone,
        action="agentic",
        handler="photo_social",
        interval_seconds=max(60, PHOTO_SOCIAL_SCAN_INTERVAL_SECONDS),
    )
    log.info("Seeded photo social scan job every %ss", max(60, PHOTO_SOCIAL_SCAN_INTERVAL_SECONDS))


def ensure_video_social_job(timezone: str | None = None) -> None:
    """Idempotently seed the video-inbox scan job (Lane C)."""
    existing_titles = {job.get("title") for job in _read_all()}
    if VIDEO_SOCIAL_JOB_TITLE in existing_titles:
        return
    schedule_job_record(
        title=VIDEO_SOCIAL_JOB_TITLE,
        task="Scan video inbox for a described, not-yet-drafted video",
        time_of_day="00:00",
        frequency="interval",
        timezone=timezone,
        action="agentic",
        handler="video_social",
        interval_seconds=max(60, VIDEO_SOCIAL_SCAN_INTERVAL_SECONDS),
    )
    log.info("Seeded video social scan job every %ss", max(60, VIDEO_SOCIAL_SCAN_INTERVAL_SECONDS))


def register_social_handlers(timezone: str | None = None) -> None:
    """Register the weekly/photo/video social handlers and seed their jobs.

    This is the concrete version of the pattern this module's module-level
    comment used to only describe in prose: it registers all three social
    handlers with register_system_handler() and then seeds all three jobs
    via ensure_weekly_social_job / ensure_photo_social_job /
    ensure_video_social_job. Safe to call on every app startup — handler
    registration is just a dict update, and each ensure_*_job() call is
    already idempotent by title.

    Call this once at startup, alongside wherever deep_study/workspace
    knowledge handlers are already registered (see
    memory.learn.register_deep_study_handlers and
    ensure_workspace_knowledge_job for the equivalent pattern). Imported
    lazily so schedule.py doesn't take a hard, always-on dependency on
    agentic.toolkit.social (and its heavier deps like the vision/LLM clients,
    requests, OpenAI client, etc.) at module import time.
    """
    from agentic.toolkit.social import (
        run_scheduled_weekly_social,
        run_scheduled_photo_social,
        run_scheduled_video_social,
        retry_weekly_social_if_needed,
    )

    register_system_handler("weekly_social", run_scheduled_weekly_social)
    register_system_handler("photo_social", lambda memorize: run_scheduled_photo_social())
    register_system_handler("video_social", lambda memorize: run_scheduled_video_social())
    register_system_handler("weekly_social_retry", retry_weekly_social_if_needed)

    ensure_weekly_social_job(timezone)
    ensure_photo_social_job(timezone)
    ensure_video_social_job(timezone)
    ensure_weekly_social_retry_job(timezone)

    log.info("Registered social handlers (weekly_social, weekly_social_retry, photo_social, video_social) and seeded their jobs.")


def ensure_deep_study_window_jobs(timezone: str | None = None) -> None:
    """Idempotently seed the four recurring jobs that bound Aiko's
    scheduled deep_studying window (weekdays 05:00-18:00, weekends
    05:00-10:00). Safe to call on every app startup — existing jobs (by
    title) are left alone rather than duplicated, so hand-edits to
    schedule.json (e.g. disabling one window) survive restarts.

    The handlers named here ("deep_study_start" / "deep_study_stop") must
    be registered via register_system_handler() before these jobs can
    actually fire anything — see memory.learn.register_deep_study_handlers,
    which does both the handler registration and calls this function.
    """
    existing_titles = {job.get("title") for job in _read_all()}
    for title, (time_of_day, days, handler) in DEEP_STUDY_WINDOW_JOB_TITLES.items():
        if title in existing_titles:
            continue
        schedule_job_record(
            title=title,
            task=title,
            time_of_day=time_of_day,
            frequency="custom_weekdays",
            timezone=timezone,
            days_of_week=days,
            action="agentic",
            handler=handler,
        )
        log.info("Seeded deep-study window job %r (%s, %s)", title, time_of_day, days)


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
    schedule.json to be able to schedule (e.g. weekly_social, photo_social,
    video_social, deep_study_start/deep_study_stop). If `fn` needs extra
    context (an LLM client/model, say) or doesn't need memorize at all
    (e.g. photo_social/video_social), bind or absorb it with
    functools.partial or a small lambda before registering — the scheduler
    always calls it with exactly one positional arg, memorize.
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
    now = bioclock.local_now()
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
    now = bioclock.local_now()
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

    Startup catch-up:
      - daily_reflect_and_dream: _missing_reflection_dates() scans back up
        to CATCHUP_MAX_LOOKBACK_DAYS days via _reflection_post_exists()
        (live GitHub check) and backfills every missing date found,
        oldest first, on a background thread started from start().
      - monthly_consolidate: _monthly_catchup_needed() compares the
        current "YYYY-MM" against a small local state file
        (tasks/monthly_consolidate_state.json) recording the last month
        that actually completed, and fires one catch-up run if they
        don't match and we're not still waiting on this month's window.
    """

    def __init__(
        self,
        on_due: Callable[[DueJob], None] | None = None,
        memorize=None,
        generate_and_post_fn: Callable | None = None,
        consolidate_fn: Callable | None = None,
        user_id: str | None = None,
    ) -> None:
        self._on_due               = on_due
        self._memorize             = memorize
        self._generate_and_post_fn = generate_and_post_fn
        self._consolidate_fn       = consolidate_fn
        self._user_id              = user_id or (memorize.get_user_id() if memorize and memorize.get_user_id() else None) or current_user_id()
        self._wakeup               = threading.Event()
        self._stop                 = threading.Event()
        self._thread: threading.Thread | None = None

        # calculated once at startup, updated after each fire
        self._next_daily   = _next_daily_reflect_and_dream()
        self._next_monthly = _next_monthly_consolidate()

        # catch-up state — checked on start()
        self._catchup_dates = self._missing_reflection_dates()
        self._monthly_catchup_needed_flag = self._monthly_catchup_needed()

    # ── daily catch-up ────────────────────────────────────────────────────────

    def _missing_reflection_dates(self) -> list[datetime]:
        """
        Scan back up to CATCHUP_MAX_LOOKBACK_DAYS from yesterday and return
        every date (oldest first) with no existing reflection post. Stops
        scanning further back once it hits a date that DOES have a post, on
        the assumption that a contiguous run existed before any outage —
        avoids a full-history GitHub API scan on every startup.
        """
        missing: list[datetime] = []
        for offset in range(1, CATCHUP_MAX_LOOKBACK_DAYS + 1):
            day = (bioclock.utc_now() - timedelta(days=offset)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            if _reflection_post_exists(day):
                break
            missing.append(day)

        if missing:
            log.info(
                "Catch-up needed: no reflection post found for %s.",
                ", ".join(d.strftime("%Y-%m-%d") for d in missing),
            )
        return list(reversed(missing))  # oldest first

    # ── monthly catch-up ──────────────────────────────────────────────────────

    def _monthly_catchup_needed(self) -> bool:
        """
        True if we're already partway into a new month and the last
        completed consolidation (per the local state file) doesn't match
        the current month — i.e. the 1st-of-month window was missed
        (machine off/asleep at MONTHLY_JOB_HOUR:MONTHLY_JOB_MINUTE on the
        1st). There's no external check available for this job (unlike
        daily reflect's GitHub post existence check), so this relies on
        the scheduler recording its own completions.
        """
        now = bioclock.local_now()
        hours_until_next = (self._next_monthly - now).total_seconds() / 3600
        if hours_until_next < 20:
            return False  # job due soon / just ran, nothing to catch up

        current_month = now.strftime("%Y-%m")
        last_run = _read_last_consolidated_month()
        if last_run == current_month:
            return False

        log.info(
            "Monthly catch-up needed: last consolidation recorded for %s, now in %s.",
            last_run, current_month,
        )
        return True

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
        if self._catchup_dates and self._memorize and self._generate_and_post_fn:
            log.info(
                "Scheduler: running %d missed daily reflect+dream job(s) on startup.",
                len(self._catchup_dates),
            )
            catchup_thread = threading.Thread(
                target=self._run_catchup_backfill,
                name="aiko-schedule-catchup",
                daemon=True,
            )
            catchup_thread.start()

        if self._monthly_catchup_needed_flag and self._memorize and self._consolidate_fn:
            log.info("Scheduler: running missed monthly_consolidate on startup.")
            monthly_catchup_thread = threading.Thread(
                target=self._run_monthly_consolidate,
                name="aiko-schedule-monthly-catchup",
                daemon=True,
            )
            monthly_catchup_thread.start()

    def stop(self) -> None:
        """Request scheduler shutdown."""
        self._stop.set()
        self._wakeup.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            now = bioclock.local_now()

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

            delta = (next_target - bioclock.local_now()).total_seconds()
            if delta > 0:
                log.debug("Scheduler sleeping %.0fs until %s", delta, next_target.isoformat())
                bioclock.wait_seconds(self._wakeup, delta)
                self._wakeup.clear()

    # ── system job runners ────────────────────────────────────────────────────

    def _run_catchup_backfill(self) -> None:
        """Sequentially backfill every date found missing by
        _missing_reflection_dates(), oldest first."""
        for date in self._catchup_dates:
            self._run_daily_reflect_and_dream(for_date=date)

    def _run_daily_reflect_and_dream(self, for_date: datetime | None = None) -> None:
        """
        Hardcoded nightly job. Not in schedule.json. Not user-modifiable.

        Order:
          1. reflect  — LLM summary + image + GitHub push (reads memories before dream prunes)
          2. dream    — sqlite-vec consolidation, boost, merge, prune (no LLM)

        for_date: when set (used by catch-up backfill), generates the
        reflection for this specific date instead of "yesterday" relative
        to now. dream() consolidation only runs on the regular (for_date is
        None) nightly call — running it once per backfilled date during a
        multi-day catch-up would just repeat the same boost/merge/prune
        pass redundantly against the same live memory store.
        """
        if not self._memorize or not self._generate_and_post_fn:
            log.warning("daily_reflect_and_dream: memorize or generate_and_post_fn not set — skipping.")
            return

        try:
            log.info("daily_reflect_and_dream: starting.")

            now_local = bioclock.local_now()
            target_local = for_date or (now_local - timedelta(days=1)).replace(
                hour=0, minute=0, second=0, microsecond=0,
            )
            target_end_local = target_local + timedelta(days=1)

            query_start = target_local.astimezone(timezone.utc)
            query_end   = target_end_local.astimezone(timezone.utc)

            from memory.reflect import REFLECT_MAX_MEMS
            memories = self._memorize.get_between(
                query_start, query_end, user_id=self._memorize.get_user_id()
            )            
            log.info(
                "daily_reflect_and_dream: %d memories fetched for %s.",
                len(memories), target_local.date(),
            )

            log.info("daily_reflect_and_dream: running reflect for %s...", target_local.date())
            reflect_result = self._generate_and_post_fn(
                memories[:REFLECT_MAX_MEMS],
                date=target_local,
                memorize=self._memorize,
                display_name=self._memorize.get_display_name() if self._memorize else None,
            )
            if isinstance(reflect_result, dict) and not reflect_result.get("success", False):
                log.error(
                    "daily_reflect_and_dream: reflect FAILED for %s — error=%s",
                    target_local.date(), reflect_result.get("error", "unknown"),
                )
            else:
                log.info(
                    "daily_reflect_and_dream: reflect done for %s — %s",
                    target_local.date(), reflect_result,
                )

            if for_date is None:
                log.info("daily_reflect_and_dream: running dream...")
                result = self._memorize.dream()
                log.info("daily_reflect_and_dream: dream done — %s", result)

        except Exception as e:
            log.error("daily_reflect_and_dream failed: %s", e)

    def _run_monthly_consolidate(self) -> None:
        """
        Hardcoded monthly job. Not in schedule.json. Not user-modifiable.
        Delegates entirely to consolidate.maybe_run_consolidation().

        On success, records the completed "YYYY-MM" to a local state file
        so _monthly_catchup_needed() can detect a missed window on a future
        startup.
        """
        if not self._memorize or not self._consolidate_fn:
            log.warning("monthly_consolidate: memorize or consolidate_fn not set — skipping.")
            return

        try:
            log.info("monthly_consolidate: starting.")
            now = bioclock.local_now()
            result = self._consolidate_fn(self._memorize, now=now, user_id=self._user_id)
            log.info("monthly_consolidate: done — %s", result)
            _write_last_consolidated_month(now.strftime("%Y-%m"))
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
            tz_name = bioclock.timezone_name(job.get("timezone"))
            try:
                due = datetime.fromisoformat(job["next_due"])
                if due.tzinfo is None:
                    due = due.replace(tzinfo=bioclock.get_timezone(tz_name))
            except Exception:
                due = calculate_next_due(
                    job.get("time_of_day", "06:00"),
                    job.get("frequency", "daily"),
                    tz_name,
                    job.get("days_of_week"),
                    interval_seconds=job.get("interval_seconds"),
                )
                job["next_due"] = due.isoformat()
                changed = True

            if due <= bioclock.local_now(tz_name):
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
                job["last_ran_at"] = bioclock.local_now(tz_name).isoformat()
                if job.get("frequency") == "once":
                    job["enabled"] = False
                else:
                    job["next_due"] = calculate_next_due(
                        job.get("time_of_day", "06:00"),
                        job.get("frequency", "daily"),
                        tz_name,
                        job.get("days_of_week"),
                        after=bioclock.local_now(tz_name),
                        interval_seconds=job.get("interval_seconds"),
                    ).isoformat()
                changed = True

        if changed:
            _write_all(jobs)

        # fire sequentially — preserves order and avoids concurrent job side effects
        for event in due_events:
            if self._on_due:
                try:
                    self._on_due(event)
                except Exception:
                    log.exception("Scheduled job handler failed for %s", event.get("title", event.get("id", "?")))


ReminderScheduler = ScheduleRunner