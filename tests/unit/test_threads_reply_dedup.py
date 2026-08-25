"""Unit tests against double replies in the Threads monitor.

Two issues produced duplicate answers for a single triggered comment:
  1. The always-on reply daemon AND a seeded scheduler job polled
     concurrently, both passing the has-processed check before either
     published (seen in production: the same in_reply_to answered twice).
  2. Nothing serialized overlapping monitor runs inside one process.

Covers: run-lock skip behavior, lock release on failure, and schedule
seeding retiring the scheduler job while the daemon owns polling.
"""

import importlib
import json
import threading

import pytest

threads = importlib.import_module("interface.mcp_server.social.services.threads")
schedule = importlib.import_module("system.schedule")


def test_concurrent_monitor_run_skips_instead_of_double_polling(monkeypatch):
    inner_started = threading.Event()
    release_inner = threading.Event()

    def slow_locked(memorize=None):
        inner_started.set()
        release_inner.wait(timeout=5)
        return {"ok": True, "provider": "threads", "matched": 1, "answered": 1, "errors": []}

    monkeypatch.setattr(threads, "_monitor_threads_replies_locked", slow_locked)

    first_result = {}
    worker = threading.Thread(
        target=lambda: first_result.update(threads.monitor_threads_replies())
    )
    worker.start()
    assert inner_started.wait(timeout=5)

    second = threads.monitor_threads_replies()
    release_inner.set()
    worker.join(timeout=5)

    assert first_result.get("answered") == 1
    assert second.get("skipped") == "monitor_run_in_progress"
    assert second.get("answered") == 0


def test_monitor_run_lock_released_after_exception(monkeypatch):
    def boom(memorize=None):
        raise RuntimeError("publish failed")

    monkeypatch.setattr(threads, "_monitor_threads_replies_locked", boom)
    with pytest.raises(RuntimeError):
        threads.monitor_threads_replies()
    # Lock must be free again — the next call reaches the inner function.
    monkeypatch.setattr(
        threads, "_monitor_threads_replies_locked", lambda memorize=None: {"ok": True}
    )
    assert threads.monitor_threads_replies()["ok"] is True


@pytest.fixture
def schedule_env(tmp_path, monkeypatch):
    path = tmp_path / "schedule.json"
    monkeypatch.setenv("SCHEDULE_PATH", str(path))
    schedule._invalidate_cache()
    yield path
    schedule._invalidate_cache()


def test_scheduler_poller_job_is_retired(schedule_env):
    """The reply daemon is the single poller — persisted scheduler jobs must
    be disabled on every social bootstrap, never re-seeded."""
    schedule_env.write_text(json.dumps([
        {
            "id": "j1",
            "title": schedule.THREADS_REPLY_MONITOR_JOB_TITLE,
            "handler": schedule.THREADS_REPLY_MONITOR_JOB_TITLE,
            "enabled": True,
        }
    ]))
    schedule._invalidate_cache()

    schedule.disable_threads_reply_monitor_job(user_id="u1")

    job = json.loads(schedule_env.read_text())[0]
    assert job["enabled"] is False


def test_no_scheduler_seeding_or_handler_for_threads_monitor():
    """The scheduler path for Threads polling is gone entirely."""
    assert not hasattr(schedule, "ensure_threads_reply_monitor_job")
    assert "threads_reply_monitor" not in schedule._SYSTEM_HANDLERS


def test_disable_is_a_noop_without_persisted_job(schedule_env):
    schedule.disable_threads_reply_monitor_job(user_id="u1")
    assert not schedule_env.exists()
