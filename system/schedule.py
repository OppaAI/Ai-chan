"""
system/schedule.py

Persistent scheduled jobs, reminders, and wake-up alarms for Aiko.

A scheduled job is a small local record with:
  - time_of_day: local wall-clock time in HH:MM format
  - frequency: once, interval, hourly, daily, weekdays, weekly, biweekly, monthly, or custom_weekdays
  - days_of_week: optional weekday names for custom_weekdays/weekly jobs
  - relative_days: optional integer day offset for phrases like tomorrow or the day after tomorrow
  - task: what Aiko should do or say when the job fires
  - action: announce, agentic, or tool
  - tool_call: optional registered agentic tool invocation, e.g.
    {"name": "draft_job_post_social", "arguments": {}}
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
    (memory/monthly_consolidate_state.json under the user memory directory)
    recording the last "YYYY-MM" the job actually completed. If the
    current month doesn't match on startup and we're not still waiting
    for this month's scheduled window, one catch-up run fires.

Other system-style behaviors (e.g. weekly_dev_repost, photo_social, video_social,
daily_job_post_social, deep_study_start/stop, workspace_knowledge_scan) live entirely in
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
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import fcntl

from system import bioclock
from system.log import get_logger
from system.turngate import AIKO_BUSY_LOCK
from system.userspace import (
    all_known_user_ids, current_user_id, reset_current_user_id,
    set_current_user_id, user_state_path, user_workspace_root,
)

log = get_logger(__name__)

_knowledge_folder_watcher = None  # KnowledgeFolderWatcher instance (lazy)


@contextmanager
def _scheduler_run_lock(user_id: str, name: str):
    """Serialize one scheduler stream across processes sharing user state."""
    path = user_state_path(f".locks/scheduler-{name}.lock", user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        yield False
        return
    try:
        yield True
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)

def workspace_root() -> Path:
    """Resolve the active user workspace root lazily."""
    override = os.getenv("WORKSPACE_ROOT")
    return (Path(override).expanduser() if override else user_workspace_root()).resolve()


def user_state_root() -> Path:
    """Resolve the active user state root lazily."""
    override = (
        os.getenv("USER_SPACE_ROOT")
        or os.getenv("USER_STATE_ROOT")
        or os.getenv("AIKO_USER_STATE_ROOT")
    )
    return (Path(override).expanduser() if override else Path.home() / ".aiko").resolve()


def _reflection_lock_path(target_date: datetime) -> Path:
    """Return the process-shared, date-specific reflection lock path.

    Daily reflections publish to one shared GitHub post per date, so their
    lock must not be placed below an individual owner's user-state folder.
    """
    return user_state_root() / ".locks" / f"reflect-{target_date.strftime('%Y-%m-%d')}.lock"


def all_user_ids() -> list[str]:
    """Enumerate every user_id with a state directory under user space.

    Thin wrapper over userspace.all_known_user_ids(), the single source of
    truth for discovering users below the configured state root.

    Used by ScheduleRunner._run() to check every user's due jobs each tick
    instead of only whichever single user_id the runner happens to be
    bound to. "guest" is excluded — guest never has a persistent job store
    (see AikoMemorize's pre-auth lazy-guest-DB note in memorize.py).
    """
    return all_known_user_ids()


def schedule_path(user_id: str | None = None) -> Path:
    """Resolve the active user canonical schedule path lazily."""
    override = os.getenv("SCHEDULE_PATH")
    if override:
        return Path(override).expanduser().resolve()
    return user_state_path("tasks/schedule.json", user_id=user_id).resolve()


def schedule_graphs_path(user_id: str | None = None) -> Path:
    """Resolve the schedule-graphs path (DAG-based scheduled workflows)."""
    return user_state_path("tasks/schedule_graphs.json", user_id=user_id).resolve()

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
# monthly_consolidate actually completed. It belongs with the user memory DB.
# It is intentionally separate from the editable task schedule directory.
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
    try:
        resp = requests.get(url, headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
        }, params={"ref": branch}, timeout=10)
    except Exception:
        return False
    if resp.status_code == 401:
        # Bad credentials — don't treat as missing, or every restart backfills 7 days and blocks chat
        from system.log import get_logger
        get_logger(__name__).warning("GitHub 401 for reflection check — treating %s as exists to avoid catch-up loop (rotate GITHUB_TOKEN)", date.strftime("%Y-%m-%d"))
        return True
    return resp.status_code == 200


# ── monthly consolidate catch-up state ────────────────────────────────────────
# Small local marker (not schedule.json — schedule.json is user job storage)
# recording the last "YYYY-MM" monthly_consolidate actually completed. There
# is no external check available for this job (unlike daily reflect's GitHub
# post existence check), so the scheduler writes this itself on success.

def _monthly_state_path(user_id: str | None = None) -> Path:
    """Resolve the local monthly-consolidate catch-up state file path."""
    return user_state_path(f"memory/{MONTHLY_CATCHUP_STATE_PATH_NAME}", user_id=user_id).resolve()


def _read_last_consolidated_month(user_id: str | None = None) -> str | None:
    """Return the last 'YYYY-MM' monthly_consolidate completed, or None."""
    path = _monthly_state_path(user_id=user_id)
    if not path.exists():
        # Preserve the marker written by pre-migration scheduler versions.
        legacy_path = (workspace_root() / "tasks" / MONTHLY_CATCHUP_STATE_PATH_NAME).resolve()
        if legacy_path.exists():
            try:
                legacy_data = json.loads(legacy_path.read_text(encoding="utf-8"))
                if isinstance(legacy_data, dict):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(legacy_data), encoding="utf-8")
                    log.info("Migrated monthly consolidation state from %s to %s", legacy_path, path)
            except Exception as e:
                log.warning("Failed migrating monthly consolidation state %s: %s", legacy_path, e)
        if not path.exists():
            return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("last_run_month")
    except Exception as e:
        log.error("Failed reading monthly consolidate state %s: %s", path, e)
        return None


def _write_last_consolidated_month(month_str: str, user_id: str | None = None) -> None:
    """Persist the 'YYYY-MM' that monthly_consolidate just completed for."""
    path = _monthly_state_path(user_id=user_id)
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


_schedule_cache: dict[str, list[dict]] = {}

def _cache_key(user_id: str | None = None) -> str:
    return user_id or current_user_id() or ""


def _read_all(user_id: str | None = None) -> list[dict]:
    """Read scheduled jobs for the active user (cached)."""
    key = _cache_key(user_id)
    if key in _schedule_cache:
        return _schedule_cache[key]
    return _read_and_cache(user_id=user_id)


def _read_and_cache(user_id: str | None = None) -> list[dict]:
    key = _cache_key(user_id)
    path = schedule_path(user_id=user_id)
    data = _read_raw(path)
    # One-time migration from older schedule locations.
    if not data and not path.exists() and not os.getenv("SCHEDULE_PATH"):
        legacy_paths = [
            (user_state_path("task/schedule.json", user_id=user_id).resolve(), "user task folder"),
            ((user_workspace_root(user_id) / "tasks" / "schedule.json").resolve(), "workspace task folder"),
        ]
        for legacy_path, legacy_label in legacy_paths:
            legacy_data = _read_raw(legacy_path)
            if legacy_data:
                _write_all(legacy_data, user_id=user_id)
                data = legacy_data
                log.info("Migrated schedule from %s to %s (%s)", legacy_path, path, legacy_label)
                break
    _schedule_cache[key] = data
    return data


def _invalidate_cache(user_id: str | None = None) -> None:
    if user_id is None:
        _schedule_cache.clear()
    else:
        _schedule_cache.pop(_cache_key(user_id), None)


def _write_all(jobs: list[dict], user_id: str | None = None) -> None:
    """Persist scheduled jobs atomically enough for a single local process."""
    _invalidate_cache(user_id)
    path = schedule_path(user_id=user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(jobs, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


# ── schedule-graphs I/O ────────────────────────────────────────────────────────
# Schedule graphs are DAG-based workflows with integrated triggers. Stored in a
# separate file from schedule.json so the two systems coexist during migration.
# Each entry has: id, trigger (time/frequency), nodes (graph DAG), next_due,
# last_ran_at, enabled.

_graphs_cache: dict[str, list[dict]] = {}


def _read_schedule_graphs(user_id: str | None = None) -> list[dict]:
    key = _cache_key(user_id)
    if key in _graphs_cache:
        return _graphs_cache[key]
    path = schedule_graphs_path(user_id=user_id)
    data = _read_raw(path)
    _graphs_cache[key] = data
    return data


def _invalidate_graphs_cache(user_id: str | None = None) -> None:
    if user_id is None:
        _graphs_cache.clear()
    else:
        _graphs_cache.pop(_cache_key(user_id), None)


def _write_schedule_graphs(graphs: list[dict], user_id: str | None = None) -> None:
    _invalidate_graphs_cache(user_id)
    path = schedule_graphs_path(user_id=user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(graphs, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _schedule_graph_next_due(graph_def: dict, after: datetime | None = None) -> datetime:
    trigger = graph_def.get("trigger", {})
    return calculate_next_due(
        time_of_day=trigger.get("time", "08:00"),
        frequency=trigger.get("frequency", "daily"),
        timezone=trigger.get("timezone"),
        days_of_week=trigger.get("days_of_week"),
        after=after,
        interval_seconds=trigger.get("interval_seconds"),
    )


def _default_schedule_graphs(user_id: str | None = None) -> list[dict]:
    """Seed entries that reference playbook IDs.

    Each entry has: id, trigger (timing), graph_id (playbook to execute),
    enabled, plus runtime state (next_due, last_ran_at).

    The daily_job_post_social trigger time, graph_id, and enabled state
    are read from the user's editable schedule.json (see
    _job_post_social_config) so timing, graph choice, and enablement
    can be changed without a code change.
    """
    cfg = _job_post_social_config(user_id)
    return [
        {
            "id": "daily_job_post_social",
            "trigger": {"time": cfg["time"], "frequency": "daily"},
            "graph_id": cfg["graph_id"],
            "enabled": cfg["enabled"],
            "next_due": "",
            "last_ran_at": None,
        },
        {
            "id": "hourly_aurora_forecast",
            "trigger": {"time": "00:05", "frequency": "hourly"},
            "graph_id": "aurora_forecast",
            "enabled": True,
            "next_due": "",
            "last_ran_at": None,
        },
        {
            "id": "owner_email_poll",
            "trigger": {"frequency": "interval", "interval_seconds": 600},
            "graph_id": "owner_email",
            "enabled": True,
            "next_due": "",
            "last_ran_at": None,
        },
        {
            "id": "nightly_codebase_refresh",
            "trigger": {"time": "22:00", "frequency": "daily"},
            "graph_id": "codebase_refresh",
            "enabled": True,
            "next_due": "",
            "last_ran_at": None,
        },
    ]


def ensure_schedule_graphs(user_id: str | None = None) -> None:
    """Initialize schedule graphs file with defaults if it exists, or create it if missing."""
    path = schedule_graphs_path(user_id=user_id)
    if path.exists():
        graphs = _read_schedule_graphs(user_id=user_id)
        # Sync the daily_job_post_social trigger time, graph_id, and
        # enabled state from schedule.json so edits to schedule.json
        # take effect on the next fire cycle without a restart.
        now = bioclock.local_now()
        changed = False
        cfg = _job_post_social_config(user_id)
        # Seed aurora graph if missing (existing installs)
        if not any(g.get("id") == "hourly_aurora_forecast" or g.get("graph_id") == "aurora_forecast" for g in graphs):
            graphs.append({
                "id": "hourly_aurora_forecast",
                "trigger": {"time": "00:05", "frequency": "hourly"},
                "graph_id": "aurora_forecast",
                "enabled": True,
                "next_due": _schedule_graph_next_due(
                    {"trigger": {"time": "00:05", "frequency": "hourly"}}, after=now
                ).isoformat(),
                "last_ran_at": None,
            })
            changed = True
        # Seed owner email poll if missing (existing installs)
        if not any(g.get("id") == "owner_email_poll" or g.get("graph_id") == "owner_email" for g in graphs):
            graphs.append({
                "id": "owner_email_poll",
                "trigger": {"frequency": "interval", "interval_seconds": 600},
                "graph_id": "owner_email",
                "enabled": True,
                "next_due": _schedule_graph_next_due(
                    {"trigger": {"frequency": "interval", "interval_seconds": 600}}, after=now
                ).isoformat(),
                "last_ran_at": None,
            })
            changed = True
        # Seed nightly codebase refresh if missing (existing installs)
        if not any(g.get("id") == "nightly_codebase_refresh" or g.get("graph_id") == "codebase_refresh" for g in graphs):
            graphs.append({
                "id": "nightly_codebase_refresh",
                "trigger": {"time": "22:00", "frequency": "daily"},
                "graph_id": "codebase_refresh",
                "enabled": True,
                "next_due": _schedule_graph_next_due(
                    {"trigger": {"time": "22:00", "frequency": "daily"}}, after=now
                ).isoformat(),
                "last_ran_at": None,
            })
            changed = True
        for g in graphs:
            if g.get("id") == JOB_POST_SOCIAL_JOB_TITLE or g.get("graph_id") == "gen_job_post":
                if g.get("trigger", {}).get("time") != cfg["time"]:
                    g["trigger"] = dict(g.get("trigger", {}), time=cfg["time"])
                    g["next_due"] = _schedule_graph_next_due(g, after=now).isoformat()
                    changed = True
                if g.get("graph_id") != cfg["graph_id"]:
                    g["graph_id"] = cfg["graph_id"]
                    changed = True
                if g.get("enabled") != cfg["enabled"]:
                    g["enabled"] = cfg["enabled"]
                    changed = True
        if changed:
            _write_schedule_graphs(graphs, user_id=user_id)
        return
    graphs = _default_schedule_graphs(user_id=user_id)
    now = bioclock.local_now()
    for g in graphs:
        g["next_due"] = _schedule_graph_next_due(g, after=now).isoformat()
    _write_schedule_graphs(graphs, user_id=user_id)
    log.info("Seeded schedule graphs: %s", [g["id"] for g in graphs])

    # Disable the migrated schedule.json entry so there's no double-fire.
    jobs = _read_all(user_id=user_id)
    changed = False
    for job in jobs:
        if job.get("title") == JOB_POST_SOCIAL_JOB_TITLE and job.get("enabled", True):
            job["enabled"] = False
            job["migrated_to"] = "schedule_graph"
            changed = True
            log.info("Migrated %r from schedule.json to schedule_graphs.json", job.get("title"))
    if changed:
        _write_all(jobs, user_id=user_id)

    # Seed a config marker in schedule.json so the user has
    # an editable time_of_day / graph_id / enabled knob.
    _ensure_job_post_config_marker(user_id=user_id)


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
    tool_call: dict[str, Any] | None = None,
    skill: str | None = None,
    user_id: str | None = None,
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
    if action not in {"announce", "agentic", "tool"}:
        raise ValueError("action must be 'announce', 'agentic', or 'tool'")
    if skill is not None and not isinstance(skill, str):
        raise ValueError("skill must be Markdown text")
    normalized_skill = skill.strip() if skill else None
    normalized_tool_call: dict[str, Any] | None = None
    if tool_call is not None:
        if not isinstance(tool_call, dict) or not isinstance(tool_call.get("name"), str) or not tool_call["name"].strip():
            raise ValueError("tool_call must contain a non-empty tool name")
        arguments = tool_call.get("arguments", tool_call.get("args", {}))
        if not isinstance(arguments, dict):
            raise ValueError("tool_call arguments must be an object")
        normalized_tool_call = {"name": tool_call["name"].strip(), "arguments": arguments}
    if action == "tool" and normalized_tool_call is None:
        raise ValueError("tool action requires tool_call")
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
        "tool_call": normalized_tool_call,
        "skill": normalized_skill,
    }
    jobs = _read_all(user_id=user_id)
    jobs.append(job)
    _write_all(jobs, user_id=user_id)
    return job


def list_schedule_records(include_disabled: bool = False, user_id: str | None = None) -> list[dict]:
    """Return persisted scheduled jobs, optionally including disabled records."""
    jobs = _read_all(user_id=user_id)
    if include_disabled:
        return jobs
    return [job for job in jobs if job.get("enabled", True)]


def cancel_schedule_record(job_id: str, user_id: str | None = None) -> bool:
    """Disable a scheduled job by id; returns True when a matching record changed."""
    changed = False
    jobs = _read_all(user_id=user_id)
    for job in jobs:
        if job.get("id") == job_id:
            job["enabled"] = False
            changed = True
    if changed:
        _write_all(jobs, user_id=user_id)
    return changed


# Backwards-compatible reminder names used by older tools/tests.
def schedule_reminder_record(title: str, message: str, time_of_day: str, repeat: str = "daily", timezone: str | None = None, user_id: str | None = None) -> dict:
    """Compatibility wrapper: schedule a reminder as a scheduled job."""
    frequency = "daily" if repeat == "daily" else "once"
    return schedule_job_record(title, message, time_of_day, frequency, timezone, action="announce", user_id=user_id)


def list_reminder_records(include_disabled: bool = False, user_id: str | None = None) -> list[dict]:
    """Compatibility wrapper: list scheduled jobs."""
    return list_schedule_records(include_disabled, user_id=user_id)


def cancel_reminder_record(reminder_id: str, user_id: str | None = None) -> bool:
    """Compatibility wrapper: cancel a scheduled job by id."""
    return cancel_schedule_record(reminder_id, user_id=user_id)


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


def ensure_workspace_knowledge_job(timezone: str | None = None, user_id: str | None = None) -> None:
    """Idempotently seed the scheduled KB folder scan job.

    The scheduler owns this periodic check so document-drop monitoring lives
    alongside other schedule.json-driven system behaviors instead of running
    a separate ticker thread.
    """
    existing_titles = {job.get("title") for job in _read_all(user_id=user_id)}
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
        user_id=user_id,
    )
    log.info("Seeded workspace knowledge scan job every %ss", max(60, WORKSPACE_KNOWLEDGE_SCAN_INTERVAL_SECONDS))


# ── social folder-monitoring job seeding ──────────────────────────────────────
# Lane A1 (weekly Patreon dev-post syndication) is a true weekly cadence job, not a folder scan —
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
# Runs once per week on Saturday evening. The handler
# itself (run_scheduled_weekly_social) is idempotent per Patreon post
# (generate_weekly_draft skips if a draft already exists), so a slightly
# early/late fire here is harmless.
WEEKLY_SOCIAL_TIME_OF_DAY = os.getenv("WEEKLY_SOCIAL_TIME_OF_DAY", "18:00")
WEEKLY_SOCIAL_RETRY_JOB_TITLE = "weekly_social_retry_check"
WEEKLY_SOCIAL_RETRY_INTERVAL_SECONDS = int(os.getenv("WEEKLY_SOCIAL_RETRY_INTERVAL_SECONDS", str(30 * 60)))

PHOTO_SOCIAL_JOB_TITLE = "photo_social_scan"
PHOTO_SOCIAL_SCAN_INTERVAL_SECONDS = int(os.getenv("PHOTO_SOCIAL_SCAN_INTERVAL_SECONDS", str(6 * 60 * 60)))  # 6h default

VIDEO_SOCIAL_JOB_TITLE = "video_social_scan"
VIDEO_SOCIAL_SCAN_INTERVAL_SECONDS = int(os.getenv("VIDEO_SOCIAL_SCAN_INTERVAL_SECONDS", str(6 * 60 * 60)))  # 6h default

THREADS_REPLY_MONITOR_JOB_TITLE = "threads_reply_monitor"
THREADS_REPLY_MONITOR_INTERVAL_SECONDS = int(os.getenv("THREADS_REPLY_MONITOR_INTERVAL_SECONDS", "180"))

JOB_POST_SOCIAL_JOB_TITLE = "daily_job_post_social"
JOB_POST_SOCIAL_DEFAULT_TIME = "23:00"


def _job_post_social_config(user_id: str | None = None) -> dict:
    """Return the daily job-post config from the user's editable schedule.json.

    Reads the ``daily_job_post_social`` entry for ``time_of_day``,
    ``graph_id``, and ``enabled``. Falls back to sensible defaults
    when the entry is absent or malformed.
    """
    cfg = {
        "time": JOB_POST_SOCIAL_DEFAULT_TIME,
        "graph_id": "gen_job_post",
        "enabled": True,
    }
    try:
        for job in _read_all(user_id=user_id):
            if job.get("title") == JOB_POST_SOCIAL_JOB_TITLE:
                t = str(job.get("time_of_day") or "").strip()
                if t:
                    cfg["time"] = t
                cfg["graph_id"] = str(job.get("graph_id") or cfg["graph_id"])
                cfg["enabled"] = bool(job.get("enabled", True))
                break
    except Exception:
        pass
    return cfg


def _ensure_job_post_config_marker(user_id: str | None = None) -> None:
    """Seed a disabled config marker for daily_job_post_social in schedule.json
    so the user has an editable time_of_day knob. The marker is inert (no
    registered handler) and never fires — it only stores the configured time."""
    try:
        jobs = _read_all(user_id=user_id)
        existing = None
        for j in jobs:
            if j.get("title") == JOB_POST_SOCIAL_JOB_TITLE:
                existing = j
                break
        if existing is None:
            jobs.append({
                "id": JOB_POST_SOCIAL_JOB_TITLE,
                "title": JOB_POST_SOCIAL_JOB_TITLE,
                "task": JOB_POST_SOCIAL_JOB_TITLE,
                "time_of_day": JOB_POST_SOCIAL_DEFAULT_TIME,
                "frequency": "daily",
                "timezone": "America/Vancouver",
                "enabled": True,
                "graph_id": "gen_job_post",
                "kind": "schedule_config",
                "note": "Config for the daily job-post schedule graph. Set enabled=false to disable, or change graph_id to swap the playbook.",
            })
            _write_all(jobs, user_id=user_id)
        elif existing.get("kind") == "time_config":
            # Upgrade old config marker format.
            existing["enabled"] = True
            existing["graph_id"] = existing.get("graph_id", "gen_job_post")
            existing["kind"] = "schedule_config"
            existing["note"] = "Config for the daily job-post schedule graph. Set enabled=false to disable, or change graph_id to swap the playbook."
            _write_all(jobs, user_id=user_id)
    except Exception:
        log.exception("Failed to seed job-post config marker in schedule.json")


# PATCHED: disable_legacy_job_post_tool_jobs
def disable_legacy_job_post_tool_jobs(user_id: str | None = None) -> None:
    """Disable schedule.json tool jobs that call run_job_post_playbook."""
    data = _read_all(user_id=user_id)
    if not data:
        return
    changed = False
    for job in data:
        if not isinstance(job, dict):
            continue
        tc = job.get("tool_call") or {}
        name = str(tc.get("name") or "").strip()
        if name != "run_job_post_playbook":
            continue
        if job.get("enabled", True):
            job["enabled"] = False
            changed = True
            log.info(
                "Disabled legacy schedule tool job %r (use schedule_graphs gen_job_post instead)",
                job.get("id") or job.get("title"),
            )
    if changed:
        _write_all(data, user_id=user_id)


def ensure_weekly_social_job(timezone: str | None = None, user_id: str | None = None) -> None:
    """Idempotently seed the weekly Patreon dev-post syndication job (Lane A1).

    Fires once a week on Saturday at the configured time. Existing installs
    that still have the old built-in Sunday 08:00 record are migrated, while
    user-customized weekly_social_post records are preserved.
    """
    jobs = _read_all(user_id=user_id)
    for job in jobs:
        if job.get("title") != WEEKLY_SOCIAL_JOB_TITLE:
            continue
        old_builtin = (
            job.get("frequency") == "weekly"
            and str(job.get("time_of_day") or "") == "08:00"
            and _normalize_weekdays(job.get("days_of_week")) == [6]  # Sunday
            and (job.get("handler") == "weekly_social" or job.get("kind") == "system_weekly_social")
            and (job.get("action") in {None, "agentic"})
        )
        if old_builtin:
            job["time_of_day"] = WEEKLY_SOCIAL_TIME_OF_DAY
            job["days_of_week"] = [5]  # Saturday
            job["timezone"] = timezone or job.get("timezone")
            job["action"] = "agentic"
            job["handler"] = "weekly_social"
            job.pop("kind", None)
            job["next_due"] = calculate_next_due(
                WEEKLY_SOCIAL_TIME_OF_DAY,
                "weekly",
                job["timezone"],
                job["days_of_week"],
            ).isoformat()
            _write_all(jobs, user_id=user_id)
            log.info(
                "Migrated weekly social job from old Sunday 08:00 builtin to Saturdays at %s (next_due=%s)",
                WEEKLY_SOCIAL_TIME_OF_DAY,
                job["next_due"],
            )
        return
    schedule_job_record(
        title=WEEKLY_SOCIAL_JOB_TITLE,
        task=WEEKLY_SOCIAL_JOB_TITLE,
        time_of_day=WEEKLY_SOCIAL_TIME_OF_DAY,
        frequency="weekly",
        timezone=timezone,
        days_of_week=["sat"],
        action="agentic",
        handler="weekly_social",
        user_id=user_id,
    )
    log.info("Seeded weekly social job (Saturdays at %s)", WEEKLY_SOCIAL_TIME_OF_DAY)


def ensure_weekly_social_retry_job(timezone: str | None = None, user_id: str | None = None) -> None:
    """Idempotently seed the Saturday-bounded retry check for Lane A.

    Fires every WEEKLY_SOCIAL_RETRY_INTERVAL_SECONDS regardless of day; the
    handler itself (retry_weekly_social_if_needed) is what limits action to
    Saturdays, so there's nothing day-specific to seed here.
    """
    existing_titles = {job.get("title") for job in _read_all(user_id=user_id)}
    if WEEKLY_SOCIAL_RETRY_JOB_TITLE in existing_titles:
        return
    schedule_job_record(
        title=WEEKLY_SOCIAL_RETRY_JOB_TITLE,
        task="Retry the weekly dev-post syndication if it has not posted yet (Saturdays only)",
        time_of_day="00:00",
        frequency="interval",
        timezone=timezone,
        action="agentic",
        handler="weekly_social_retry",
        interval_seconds=max(60, WEEKLY_SOCIAL_RETRY_INTERVAL_SECONDS),
        user_id=user_id,
    )
    log.info("Seeded weekly social retry-check job every %ss", max(60, WEEKLY_SOCIAL_RETRY_INTERVAL_SECONDS))


def ensure_photo_social_job(timezone: str | None = None, user_id: str | None = None) -> None:
    """Idempotently seed the photo-inbox scan job (Lane B)."""
    existing_titles = {job.get("title") for job in _read_all(user_id=user_id)}
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
        user_id=user_id,
    )
    log.info("Seeded photo social scan job every %ss", max(60, PHOTO_SOCIAL_SCAN_INTERVAL_SECONDS))


def ensure_video_social_job(timezone: str | None = None, user_id: str | None = None) -> None:
    """Idempotently seed the video-inbox scan job (Lane C)."""
    existing_titles = {job.get("title") for job in _read_all(user_id=user_id)}
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
        user_id=user_id,
    )
    log.info("Seeded video social scan job every %ss", max(60, VIDEO_SOCIAL_SCAN_INTERVAL_SECONDS))


def disable_threads_reply_monitor_job(user_id: str | None = None) -> None:
    """Retire the legacy scheduler threads_reply_monitor job.

    The always-on reply daemon (interface.mcp_server.social.monitor_daemon)
    is the single Threads poller. Running the scheduler job alongside it made
    two pollers race past the has-processed check and answer the same comment
    twice, so any persisted monitor job is disabled on every social bootstrap.
    """
    data = _read_all(user_id=user_id)
    if not data:
        return
    changed = False
    for job in data:
        if not isinstance(job, dict):
            continue
        if job.get("title") != THREADS_REPLY_MONITOR_JOB_TITLE and job.get("handler") != THREADS_REPLY_MONITOR_JOB_TITLE:
            continue
        if job.get("enabled", True):
            job["enabled"] = False
            changed = True
            log.info(
                "Disabled scheduler Threads reply monitor job %r (reply daemon is the single poller)",
                job.get("id") or job.get("title"),
            )
    if changed:
        _write_all(data, user_id=user_id)


def register_social_handlers(
    timezone: str | None = None,
    user_id: str | None = None,
    seed_jobs: bool = True,
) -> None:
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

    if not seed_jobs:
        # Pre-auth registration — job seeding writes into the user's
        # schedule.json and must wait for a real identity (see
        # register_system_handlers_only / bootstrap_non_system_jobs).
        log.info("Registered social handlers (pre-auth, seeding deferred).")
        return

    ensure_weekly_social_job(timezone, user_id=user_id)
    ensure_photo_social_job(timezone, user_id=user_id)
    ensure_video_social_job(timezone, user_id=user_id)
    ensure_weekly_social_retry_job(timezone, user_id=user_id)
    disable_threads_reply_monitor_job(user_id=user_id)

    # Schedule-graphs (DAG-based scheduled workflows) — migrated out of
    # schedule.json one lane at a time.  Currently seeds Lane D only.
    ensure_schedule_graphs(user_id=user_id)

    disable_legacy_job_post_tool_jobs(user_id=user_id)
    log.info("Registered social handlers and seeded social jobs; Lane D uses schedule_graphs.json.")


def _workspace_scan_handler(_memorize) -> None:
    """System-handler callable: ingest the workspace knowledge folder.

    Module-level so both pre-auth handler registration
    (register_system_handlers_only) and post-login bootstrap can bind the
    same callable; identity is read from _memorize at fire time.
    """
    from cognition.knowledge import ingest_workspace_knowledge_folder

    ingest_workspace_knowledge_folder(
        embedder=_memorize.embedder(),
        user_id=_memorize.get_user_id(),
    )


def _check_email_handler(_memorize) -> None:
    """System-handler callable: check ProtonMail for new job-posting emails."""
    try:
        from agentic.registry import registry
        spec = registry.get("read_protonmail")
        if spec is None or spec.handler is None:
            log.warning("Email check: read_protonmail MCP tool is not registered")
            return
        result = spec.handler(max_results=20, list_only=True)
        if not isinstance(result, dict) or not result.get("ok"):
            return
        messages = result.get("messages") or []
        if not messages:
            return

        # Filter for job-related emails
        job_keywords = ["linkedin", "glassdoor", "indeed", "job alert", "job notification",
                        "new job", "recommended job", "job match", "career", "hiring",
                        "software engineer", "developer", "programmer", "devops", "data scientist"]
        job_alerts = []
        for msg in messages:
            subject = str(msg.get("subject") or "").strip()
            snippet = str(msg.get("snippet") or "").strip()
            content = f"{subject} {str(msg.get('from') or '')} {snippet}".casefold()
            if any(kw in content for kw in job_keywords):
                job_alerts.append(f"📧 {msg.get('from')}: {subject[:80]}")

        if job_alerts:
            # Use the memorize's think reference to speak
            think = getattr(_memorize, '_think', None) or getattr(_memorize, '_think_ref', None)
            if think:
                speak = think._get_speak()
                if speak:
                    speak.speak("New job alerts in email: " + "; ".join(job_alerts))
                else:
                    log.info("Email job alerts found: %s", job_alerts)
            else:
                log.info("Email job alerts found: %s", job_alerts)
    except Exception as e:
        log.warning("Email check handler failed: %s", e)


def register_system_handlers_only(
    *,
    think: Any | None = None,
    memorize: Any | None = None,
    timezone: str | None = None,
) -> None:
    """Register every system-handler callable WITHOUT touching user files.

    Guest-safe subset of bootstrap_non_system_jobs(): populates the
    process-global _SYSTEM_HANDLERS registry so scheduled jobs that fire
    before a real login find their handlers instead of being skipped. Job
    seeding, watchers, and anything writing under USER_SPACE_ROOT stay in
    bootstrap_non_system_jobs(), which runs once a real user exists.
    """
    if think is not None:
        try:
            from cognition.memory import learn

            learn.register_deep_study_handlers(
                client=getattr(think, "_client", None),
                model=getattr(think, "_llm_model", None),
                timezone=timezone,
                seed_jobs=False,
            )
        except Exception:
            log.exception("Failed to register deep-study schedule handlers.")

    try:
        register_system_handler("workspace_knowledge_scan", _workspace_scan_handler)
        register_system_handler("check_email", _check_email_handler)
    except Exception:
        log.exception("Failed to register knowledge/email schedule handlers.")

    try:
        register_social_handlers(timezone=timezone, seed_jobs=False)
    except Exception:
        log.exception("Failed to register social schedule handlers.")


def bootstrap_non_system_jobs(
    *,
    think: Any | None = None,
    memorize: Any | None = None,
    timezone: str | None = None,
) -> None:
    """Register and seed every startup schedule behavior except the hardcoded system jobs.

    This keeps wakeup.py focused on booting subsystems while the scheduler
    module owns the runtime job wiring:
      - deep-study handlers and their window jobs
      - workspace knowledge scan job
      - social jobs, including the daily job-post tool call
    """
    user_id = memorize.get_user_id() if memorize and hasattr(memorize, 'get_user_id') else None

    # Handler registration (process-global dict writes — guest-safe).
    # Re-registering here on the post-login path is an idempotent overwrite.
    register_system_handlers_only(think=think, memorize=memorize, timezone=timezone)

    # Deep-study window job seeding (handlers were registered above without
    # seeding; see learn.register_deep_study_handlers(seed_jobs=False)).
    try:
        ensure_deep_study_window_jobs(
            timezone=timezone or os.getenv("DEEP_STUDY_WINDOW_TIMEZONE", ""),
            user_id=user_id,
        )
    except Exception:
        log.exception("Failed to seed deep-study window schedule jobs.")

    if memorize is not None:
        try:
            # Seed email checking job (every 30 minutes during day hours).
            # Guard by title like the other seeders — the old seeder appended a
            # fresh record on every boot, producing N duplicate "Check email for
            # job alerts" jobs that all fired at once and read the whole mailbox.
            try:
                from system.schedule import _read_all, schedule_job_record
                existing_titles = {job.get("title") for job in _read_all(user_id=user_id)}
                if "Check email for job alerts" not in existing_titles:
                    schedule_job_record(
                        title="Check email for job alerts",
                        task="Check ProtonMail inbox for new job alert emails and notify me",
                        time_of_day="08:00",
                        frequency="interval",
                        interval_seconds=1800,  # every 30 minutes
                        timezone=timezone,
                        handler="check_email",
                        user_id=user_id,
                    )
            except Exception:
                pass

            # Event-driven ingest: inotify watcher replaces the periodic poll.
            from cognition.knowledge.watcher import KnowledgeFolderWatcher
            from cognition.knowledge.schema import KNOWLEDGE_WORKSPACE_DIR

            global _knowledge_folder_watcher
            if _knowledge_folder_watcher is None:
                _knowledge_folder_watcher = KnowledgeFolderWatcher(
                    knowledge_dir=KNOWLEDGE_WORKSPACE_DIR,
                    on_files=lambda files: _workspace_scan_handler(memorize),
                )
            if not _knowledge_folder_watcher.start():
                # inotify unavailable — keep the interval job as a safety net.
                ensure_workspace_knowledge_job(timezone, user_id=user_id)
            else:
                log.info("Knowledge folder watcher active — interval scan disabled.")
        except Exception:
            log.exception("Failed to bootstrap workspace knowledge schedule job.")

    try:
        register_social_handlers(timezone=timezone, user_id=user_id, seed_jobs=True)
    except Exception:
        log.exception("Failed to bootstrap social schedule jobs.")


def ensure_deep_study_window_jobs(timezone: str | None = None, user_id: str | None = None) -> None:
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
    existing_titles = {job.get("title") for job in _read_all(user_id=user_id)}
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
            user_id=user_id,
        )
        log.info("Seeded deep-study window job %r (%s, %s)", title, time_of_day, days)


# ── scheduler instance registry ───────────────────────────────────────────────

_scheduler_instance: ScheduleRunner | None = None


def register_scheduler(scheduler: ScheduleRunner) -> None:
    """Register the active scheduler instance so tools can notify it of new jobs."""
    global _scheduler_instance
    _scheduler_instance = scheduler


def get_scheduler() -> "ScheduleRunner | None":
    """Return the registered scheduler instance, if one has been started."""
    return _scheduler_instance


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
    tool_call: dict[str, Any] | None = None
    skill: str | None = None


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
        (memory/monthly_consolidate_state.json) recording the last month
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
        llm_client=None,
        llm_model: str | None = None,
    ) -> None:
        self._on_due               = on_due
        self._memorize             = memorize
        self._generate_and_post_fn = generate_and_post_fn
        self._consolidate_fn       = consolidate_fn
        self._llm_client           = llm_client
        self._llm_model            = llm_model
        self._user_id               = user_id or (memorize.get_user_id() if memorize and memorize.get_user_id() else None) or current_user_id()
        # _owner_user_id is the identity the hardcoded system jobs (daily
        # reflect+dream, monthly consolidate — both single-stream, not
        # per-connected-user) always run as. It used to be frozen at
        # construction, which meant that if the scheduler was built during
        # wakeup.py's pre-auth boot (the normal path — see start_scheduler()),
        # _user_id resolved to "guest" and system jobs were permanently
        # pinned to guest for the life of the process, even after a real
        # user logged in. It is now promoted exactly once, away from guest,
        # either explicitly via set_owner() (called from webui's
        # _on_user_active() post-login hook — the preferred path) or lazily
        # via _resolve_owner() (checked at the top of every _run() tick —
        # the fallback for CLI/headless boots that never fire a login hook).
        # _user_id, by contrast, remains a transient pointer that _run()
        # flips per due job while iterating all_user_ids() — see _run().
        self._owner_user_id        = self._user_id
        self._owner_promoted       = threading.Event()
        if self._owner_user_id and self._owner_user_id != "guest":
            self._owner_promoted.set()
        self._wakeup               = threading.Event()
        self._stop                 = threading.Event()
        self._thread: threading.Thread | None = None
        self._catchup_lock          = threading.Lock()

        # calculated once at startup, updated after each fire
        self._next_daily   = _next_daily_reflect_and_dream()
        self._next_monthly = _next_monthly_consolidate()

        # catch-up state — checked on start(). NOTE: if _owner_user_id is
        # still "guest" at this point (pre-auth boot), the monthly check
        # below reads guest's (nonexistent/irrelevant) state file. That's
        # fine — _promote_owner() recomputes both flags and re-fires
        # catch-up as soon as a real owner is known, so nothing is lost,
        # just deferred. See _promote_owner().
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
            day = (bioclock.local_now() - timedelta(days=offset)).replace(
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
        last_run = _read_last_consolidated_month(user_id=self._owner_user_id)
        if last_run == current_month:
            return False

        log.info(
            "Monthly catch-up needed: last consolidation recorded for %s, now in %s.",
            last_run, current_month,
        )
        return True

    # ── system-job owner resolution ─────────────────────────────────────────

    def set_owner(self, user_id: str) -> None:
        """Explicitly assign the identity the hardcoded system jobs (daily
        reflect+dream, monthly consolidate) run as.

        Intended to be called once, from webui's _on_user_active()
        post-login hook, the first time a real user authenticates after a
        pre-auth ("guest") boot — see start_scheduler()'s guest-safe boot
        path. This is the preferred promotion path; _resolve_owner() below
        is only the lazy fallback for CLI/headless boots that never fire a
        login hook.

        No-op if the owner has already been promoted away from guest — the
        system-job stream is single-owner for the life of the process, it
        does not migrate to whoever happens to log in most recently.
        """
        if not user_id or user_id == "guest":
            return
        self._promote_owner(user_id)

    def _resolve_owner(self) -> str:
        """Lazy fallback for set_owner(): called at the top of every _run()
        tick. If nobody has explicitly promoted an owner yet, re-derive one
        from all_user_ids() so a CLI/headless boot (no webui login hook)
        still eventually finds a real owner instead of staying pinned to
        "guest" — and therefore never firing system jobs at all — for the
        life of the process."""
        if not self._owner_promoted.is_set():
            real = [u for u in all_user_ids() if u != "guest"]
            if real:
                self._promote_owner(real[0])
        return self._owner_user_id

    def _promote_owner(self, user_id: str) -> None:
        """Common path for set_owner()/_resolve_owner(): assign the
        system-job owner identity exactly once.

        If the owner was still "guest" when __init__ computed the catch-up
        flags (see the NOTE above self._catchup_dates), that computation
        was checked against the wrong identity — recompute both flags now
        against the real owner and kick off any missed-run backfill
        immediately, mirroring what start() itself does at boot, so a
        scheduler that boots pre-login doesn't just silently lose its
        first catch-up window.
        """
        if self._owner_promoted.is_set():
            return
        was_guest = self._owner_user_id in (None, "guest")
        self._owner_user_id = user_id
        self._owner_promoted.set()
        log.info("[scheduler] system-job owner resolved to %s", user_id)
        if not was_guest:
            return

        self._catchup_dates = self._missing_reflection_dates()
        self._monthly_catchup_needed_flag = self._monthly_catchup_needed()

        if self._owner_promoted.is_set() and self._catchup_dates and self._memorize and self._generate_and_post_fn:
            log.info(
                "Scheduler: running %d missed daily reflect+dream job(s) after owner resolution.",
                len(self._catchup_dates),
            )
            threading.Thread(
                target=self._run_catchup_backfill,
                name="aiko-schedule-catchup-late",
                daemon=True,
            ).start()

        if self._monthly_catchup_needed_flag and self._memorize and self._consolidate_fn:
            log.info("Scheduler: running missed monthly_consolidate after owner resolution.")
            threading.Thread(
                target=self._run_monthly_consolidate,
                name="aiko-schedule-monthly-catchup-late",
                daemon=True,
            ).start()

        self.notify_new_job()

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
            # Always run as self._owner_user_id — daily reflect/dream and
            # monthly consolidate are a single process-level stream (one
            # GitHub reflection post), not per-connected-user. Resolve/
            # promote the owner first (lazy fallback for boots that never
            # call set_owner() explicitly — see _resolve_owner()), and skip
            # firing entirely while the owner is still unresolved ("guest"):
            # there's nobody real to run these as yet, and set_owner()/
            # _resolve_owner() will re-check and backfill as soon as there
            # is. Rebind self._user_id explicitly before running since it
            # may have been left pointing at whichever user's job fired
            # last in the per-user loop below.
            self._resolve_owner()
            system_due = sorted(
                [(t, name) for t, name in [
                    (self._next_daily, "daily"),
                    (self._next_monthly, "monthly"),
                ] if t <= now],
                key=lambda x: x[0],
            ) if self._owner_promoted.is_set() else []
            for _target, name in system_due:
                with AIKO_BUSY_LOCK:
                    self.set_user(self._owner_user_id)
                    # _run() executes on its own daemon thread, which starts
                    # with a fresh contextvars context — current_user_id()
                    # (read throughout think.py) won't see the right user
                    # unless we set it here, matching self.set_user()/
                    # memorize.switch_user() below.
                    uid_token = set_current_user_id(self._owner_user_id)
                    try:
                        if self._memorize is not None:
                            self._memorize.switch_user(self._owner_user_id)
                        if name == "daily":
                            self._run_daily_reflect_and_dream()
                            self._next_daily = _next_daily_reflect_and_dream()
                        else:
                            self._run_monthly_consolidate()
                            self._next_monthly = _next_monthly_consolidate()
                    finally:
                        reset_current_user_id(uid_token)

            # ── fire overdue jobs/graphs for EVERY user ───────────────────────
            # Not just self._user_id: a scheduler bound to whoever last
            # connected would silently stop firing every other user's
            # reminders and scheduled graphs. Checking due-ness (a cheap
            # JSON read) is unlocked; only the actual run — which needs the
            # memory/think singletons bound to that user's identity — takes
            # the shared gate.
            for uid in all_user_ids():
                due_user_jobs = self._due_user_jobs(uid)
                due_graphs    = self._due_schedule_graphs(uid)
                if not due_user_jobs and not due_graphs:
                    continue
                with AIKO_BUSY_LOCK:
                    self.set_user(uid)
                    uid_token = set_current_user_id(uid)
                    try:
                        if self._memorize is not None:
                            self._memorize.switch_user(uid)
                        if due_user_jobs:
                            self._fire_due_user_jobs(uid)
                        if due_graphs:
                            self._fire_due_schedule_graphs(uid)
                    finally:
                        reset_current_user_id(uid_token)

            # ── sleep until soonest next target across all users ──────────────
            candidates = [self._next_daily, self._next_monthly]
            for uid in all_user_ids():
                candidates.extend(
                    datetime.fromisoformat(j["next_due"])
                    for j in _read_all(user_id=uid)
                    if j.get("enabled", True) and j.get("next_due")
                )
                candidates.extend(
                    datetime.fromisoformat(g["next_due"])
                    for g in _read_schedule_graphs(user_id=uid)
                    if g.get("enabled", True) and g.get("next_due")
                )
            next_target = min(candidates)

            delta = (next_target - bioclock.local_now()).total_seconds()
            if delta > 0:
                log.debug("Scheduler sleeping %.0fs until %s", delta, next_target.isoformat())
                bioclock.wait_seconds(self._wakeup, delta)
            else:
                # next_target is already due but nothing above consumed it —
                # a malformed/unfixable next_due, or a stuck candidate whose
                # due-ness the fire loops above didn't resolve, would
                # otherwise spin this loop with no sleep at all. Backstop
                # wait so it always yields instead of busy-looping.
                log.debug(
                    "Scheduler: next_target %s already due with nothing to fire — backstop wait.",
                    next_target.isoformat(),
                )
                bioclock.wait_seconds(self._wakeup, 5)
            self._wakeup.clear()

    # ── system job runners ────────────────────────────────────────────────────

    def _run_catchup_backfill(self) -> None:
        """Sequentially backfill every date found missing by
        _missing_reflection_dates(), oldest first.

        Can potentially run concurrently on two threads — the startup
        catch-up thread from start(), and a late one from
        _promote_owner() if the owner resolves mid-backfill. Guard against
        a duplicate post for the same date two ways: pop dates off the
        shared self._catchup_dates list one at a time (so a date claimed
        by one thread isn't also picked up by the other), and re-check
        _reflection_post_exists() immediately before firing (the
        underlying operation is idempotent by design, so this is a
        belt-and-suspenders check against any window between the pop and
        the fire).
        """
        if not self._catchup_lock.acquire(blocking=False):
            log.info("Scheduler: daily catch-up already running — skipping duplicate worker.")
            return
        try:
            while self._catchup_dates:
                try:
                    date = self._catchup_dates.pop(0)
                except IndexError:
                    break
                with AIKO_BUSY_LOCK:
                    uid = self._resolve_owner()
                    if not uid or uid == "guest":
                        return
                    token = set_current_user_id(uid)
                    try:
                        self.set_user(uid)
                        if self._memorize is not None:
                            self._memorize.switch_user(uid)
                        self._run_daily_reflect_and_dream(for_date=date)
                    finally:
                        reset_current_user_id(token)
        finally:
            self._catchup_lock.release()

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

            # Headless boot binds a guest/tempfile store; get_between() only
            # filters rows inside whichever DB is open, so reflecting while
            # guest-bound reads an empty DB and reports 0 memories. Rebind
            # to the resolved owner store first.
            if self._memorize.get_user_id() in (None, "", "guest"):
                from system.userspace import resolve_owner_user_id
                owner = resolve_owner_user_id()
                if owner:
                    try:
                        self._memorize.switch_user(owner)
                        log.info("[scheduler] memorize rebound to owner store %s", owner)
                    except Exception as e:
                        log.error("[scheduler] memorize rebind to %s failed: %s", owner, e)
                else:
                    log.warning(
                        "[scheduler] could not resolve owner identity for reflection — "
                        "reflecting against guest store."
                    )

            # Cross-process guard: two app instances (or a manual rerun) may
            # share state storage over network mounts. Only one reflection
            # per date wins; losers skip instead of double-posting.
            lock_path = user_state_path(
                f".locks/reflect-{target_local.strftime('%Y-%m-%d')}.lock",
                self._owner_user_id,
            )
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                os.close(lock_fd)
                log.info(
                    "daily_reflect_and_dream: another instance is reflecting for %s — skipping.",
                    target_local.strftime("%Y-%m-%d"),
                )
                return
            try:
                # Re-check after taking the lock. A catch-up worker and the
                # midnight runner can both observe a missing post before one
                # of them acquires the per-date lock.
                already_reflected = _reflection_post_exists(target_local)
                if already_reflected:
                    log.info("daily_reflect_and_dream: reflection already exists for %s — skipping post.", target_local.date())
                else:
                    from cognition.consolidate.reflect import REFLECT_MAX_MEMS, filter_reflect_snippets
                    memories = self._memorize.get_between(
                        query_start, query_end, user_id=self._memorize.get_user_id()
                    )
                    memories = filter_reflect_snippets(memories, target_local)
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
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                os.close(lock_fd)

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
            result = self._consolidate_fn(self._memorize, now=now, user_id=self._owner_user_id)
            log.info("monthly_consolidate: done — %s", result)
            _write_last_consolidated_month(now.strftime("%Y-%m"), user_id=self._owner_user_id)
        except Exception as e:
            log.error("monthly_consolidate failed: %s", e)

    # ── user job runner ───────────────────────────────────────────────────────

    def set_user(self, user_id: str) -> None:
        """Point self._user_id at another user's job store.

        Formerly called externally (post-login, once per WebSocket
        connect) to pick which single user's jobs the scheduler watched.
        Now _run() calls this itself, per due job, while iterating
        all_user_ids() — every user's jobs are checked every tick (see
        _due_user_jobs / _due_schedule_graphs below), so this just flips
        the pointer immediately before firing that user's due jobs rather
        than sticking on whichever user connected most recently.
        Callers should hold AIKO_BUSY_LOCK when this changes which
        identity self._memorize/self._on_due will act as. No-op when the
        id already matches.
        """
        if not user_id or user_id == self._user_id:
            return
        log.debug("[scheduler] job store switched to user %s", user_id)
        self._user_id = user_id
        self.notify_new_job()

    def _due_user_jobs(self, user_id: str) -> bool:
        """Cheap unlocked check: does this user have any enabled job whose
        next_due has passed? Used by _run() to decide, per user per tick,
        whether it's worth taking AIKO_BUSY_LOCK and actually switching
        context at all."""
        for job in _read_all(user_id=user_id):
            if not job.get("enabled", True) or not job.get("next_due"):
                continue
            tz_name = bioclock.timezone_name(job.get("timezone"))
            try:
                due = datetime.fromisoformat(job["next_due"])
                if due.tzinfo is None:
                    due = due.replace(tzinfo=bioclock.get_timezone(tz_name))
            except Exception:
                return True  # malformed next_due — let _fire_due_user_jobs fix it up
            if due <= bioclock.local_now(tz_name):
                return True
        return False

    def _due_schedule_graphs(self, user_id: str) -> bool:
        """Cheap unlocked check mirroring _due_user_jobs, for schedule graphs."""
        for g in _read_schedule_graphs(user_id=user_id):
            if not g.get("enabled", True) or not g.get("next_due"):
                continue
            tz_name = bioclock.timezone_name(g.get("trigger", {}).get("timezone"))
            try:
                due = datetime.fromisoformat(g["next_due"])
                if due.tzinfo is None:
                    due = due.replace(tzinfo=bioclock.get_timezone(tz_name))
            except Exception:
                return True
            if due <= bioclock.local_now(tz_name):
                return True
        return False

    def _fire_due_user_jobs(self, user_id: str | None = None) -> None:
        user_id = user_id or self._user_id
        with _scheduler_run_lock(user_id, "jobs") as acquired:
            if not acquired:
                return
            # A second process may have advanced next_due while this runner
            # waited for the lock, so discard its in-process JSON cache.
            _invalidate_cache(user_id)
            self._fire_due_user_jobs_locked(user_id)

    def _fire_due_user_jobs_locked(self, user_id: str) -> None:
        """Find due user jobs, reschedule recurring ones, disable one-shots.

        Jobs whose "handler" (or legacy "kind": "system_weekly_social") names
        a registered system handler call directly into that Python function
        instead of going through on_due/chat.

        user_id defaults to self._user_id for backward compatibility, but
        _run() now always passes it explicitly right after set_user(uid),
        so callers should not rely on the implicit fallback going forward.
        """
        jobs = _read_all(user_id=user_id)
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
                        tool_call=job.get("tool_call"),
                        skill=job.get("skill"),
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
            _write_all(jobs, user_id=user_id)

        # fire sequentially — preserves order and avoids concurrent job side effects
        for event in due_events:
            if self._on_due:
                try:
                    self._on_due(event)
                except Exception:
                    log.exception("Scheduled job handler failed for %s", event.title or event.id or "?")

    # ── schedule-graph runner ───────────────────────────────────────────────────

    def _fire_due_schedule_graphs(self, user_id: str | None = None) -> None:
        user_id = user_id or self._user_id
        with _scheduler_run_lock(user_id, "graphs") as acquired:
            if not acquired:
                return
            _invalidate_graphs_cache(user_id)
            self._fire_due_schedule_graphs_locked(user_id)

    def _fire_due_schedule_graphs_locked(self, user_id: str) -> None:
        graphs = _read_schedule_graphs(user_id=user_id)
        changed = False
        now = bioclock.local_now()

        # Keep the daily_job_post_social config in sync with
        # the user's editable schedule.json so timing, graph_id,
        # and enabled changes take effect on the next fire cycle.
        cfg = _job_post_social_config(user_id)
        for g in graphs:
            if g.get("id") == JOB_POST_SOCIAL_JOB_TITLE or g.get("graph_id") == "gen_job_post":
                if g.get("trigger", {}).get("time") != cfg["time"]:
                    g["trigger"] = dict(g.get("trigger", {}), time=cfg["time"])
                    g["next_due"] = _schedule_graph_next_due(g, after=now).isoformat()
                    changed = True
                if g.get("graph_id") != cfg["graph_id"]:
                    g["graph_id"] = cfg["graph_id"]
                    changed = True
                if g.get("enabled") != cfg["enabled"]:
                    g["enabled"] = cfg["enabled"]
                    changed = True

        for g in graphs:
            if not g.get("enabled", True):
                continue
            tz_name = bioclock.timezone_name(g.get("trigger", {}).get("timezone"))
            try:
                due = datetime.fromisoformat(g["next_due"])
                if due.tzinfo is None:
                    due = due.replace(tzinfo=bioclock.get_timezone(tz_name))
            except Exception:
                due = _schedule_graph_next_due(g)
                g["next_due"] = due.isoformat()
                changed = True

            if due <= now:
                self._run_schedule_graph(g)
                g["last_ran_at"] = now.isoformat()
                g["next_due"] = _schedule_graph_next_due(g, after=now).isoformat()
                changed = True

        if changed:
            _write_schedule_graphs(graphs, user_id=user_id)

    def _run_schedule_graph(self, graph_def: dict) -> None:
        from agentic.graph_engine import PlanGraph, PlanNode, execute_graph, get_playbook_by_id

        graph_id = graph_def.get("graph_id") or graph_def.get("id", "")
        playbook = get_playbook_by_id(graph_id) or {}

        # Resolve PlanGraph from the shared workflow registry (job_hunt, aurora, …)
        registered_graph = None
        try:
            from agentic.workflows.common.graphs import get_graph as _get_graph
            registered_graph = _get_graph(graph_id)
        except Exception as exc:
            log.debug("Schedule graph: failed to resolve graph %r: %s", graph_id, exc)

        if registered_graph is not None:
            # Use the registered graph with the scheduled goal
            graph = PlanGraph(
                id=registered_graph.id,
                name=registered_graph.name,
                goal=playbook.get("goal") or registered_graph.goal or f"Scheduled run: {graph_id}",
                nodes=registered_graph.nodes,
                source=registered_graph.source,
                reducers=registered_graph.reducers,
            )
        else:
            # Fallback: build from playbook nodes (legacy)
            nodes = []
            for raw in playbook.get("nodes", []):
                if isinstance(raw, dict) and raw.get("id") and raw.get("tool"):
                    nodes.append(PlanNode(
                        id=str(raw["id"]),
                        tool=str(raw["tool"]),
                        args=dict(raw.get("args", {})),
                        depends_on=tuple(str(d) for d in raw.get("depends_on", [])),
                        loop_to=str(raw["loop_to"]) if raw.get("loop_to") else None,
                        loop_condition=dict(raw["loop_condition"]) if raw.get("loop_condition") else None,
                        max_visits=int(raw["max_visits"]) if raw.get("max_visits") else 0,
                    ))
            if not nodes:
                log.warning("Playbook %r has no valid nodes — skipping", graph_id)
                return
            graph = PlanGraph(
                id=graph_id,
                name=playbook.get("name", graph_id),
                goal=playbook.get("goal", f"Scheduled run: {graph_id}"),
                nodes=tuple(nodes),
            )

        try:
            result = execute_graph(
                graph,
                llm_client=self._llm_client,
                llm_model=self._llm_model,
            )
            log.info(
                "Schedule graph %s completed — ok=%s, nodes=%d",
                graph.id, all(r.ok for r in result.results), len(result.results),
            )
        except Exception as e:
            log.error("Schedule graph %s failed: %s", graph_def.get("id", "?"), e)


def start_scheduler(
    *,
    on_due: Callable[[DueJob], None] | None = None,
    memorize=None,
    think=None,
    timezone: str | None = None,
    user_id: str | None = None,
) -> ScheduleRunner:
    """Construct, register, start, and seed the app scheduler in one place.

    This keeps wakeup.py free of scheduler wiring so it only boots the live
    subsystems and then delegates the rest here.

    Guest-safe: when boot runs pre-login (memorize still bound to the guest
    sentinel), user-scoped seeding is skipped entirely — playbook.json and
    schedule jobs live under USER_SPACE_ROOT/<uid>/ and must never be created
    for guest. The seeding is re-run at first authenticated login instead
    (see interface/webui/webui.py's _on_user_active()).

    The same applies to the hardcoded system-job owner (daily reflect+dream,
    monthly consolidate — see ScheduleRunner._promote_owner()): if this is
    called pre-login, effective_uid is "guest" and the scheduler's owner
    stays unresolved until either _on_user_active() calls
    scheduler.set_owner(uid) at first login, or the lazy fallback in
    ScheduleRunner._resolve_owner() finds a real user on its own. If
    effective_uid is already a real user here (e.g. this is called
    post-login, or in a single-user CLI/headless setup), the owner is
    promoted immediately below instead of waiting on either of those.
    """
    from agentic.graph_engine import ensure_playbooks
    from cognition.consolidate import generate_and_post, maybe_run_consolidation

    effective_uid = (
        user_id
        or (memorize.get_user_id() if memorize and hasattr(memorize, "get_user_id") else None)
        or current_user_id()
    )

    if effective_uid != "guest":
        ensure_playbooks(user_id=effective_uid)

    # Guest-safe handler registration BEFORE the runner starts ticking —
    # due jobs that fire pre-login must find their callables. Without this,
    # every boot logged a burst of "job references unregistered handler —
    # skipping fire" warnings and silently dropped those runs. Job DATA
    # seeding still waits for a real identity (see bootstrap_non_system_jobs).
    register_system_handlers_only(think=think, memorize=memorize, timezone=timezone)

    scheduler = ScheduleRunner(
        on_due=on_due,
        memorize=memorize,
        generate_and_post_fn=generate_and_post,
        consolidate_fn=maybe_run_consolidation,
        user_id=user_id,
        llm_client=getattr(think, "_client", None),
        llm_model=getattr(think, "_llm_model", None),
    )
    register_scheduler(scheduler)
    if effective_uid != "guest":
        scheduler.set_owner(effective_uid)
        bootstrap_non_system_jobs(think=think, memorize=memorize, timezone=timezone)
    scheduler.start()
    scheduler.notify_new_job()
    return scheduler


ReminderScheduler = ScheduleRunner
