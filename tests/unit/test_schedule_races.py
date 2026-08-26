from __future__ import annotations

from datetime import datetime, timezone
from types import ModuleType
import sys


class _Memorize:
    def __init__(self) -> None:
        self.user_id = "github_alice"

    def get_user_id(self) -> str:
        return self.user_id

    def get_display_name(self) -> str:
        return "Alice"

    def get_between(self, *_args, **_kwargs):
        return []


def test_daily_reflection_rechecks_existence_after_lock(monkeypatch, tmp_path):
    """A second runner sees the post produced by the first after locking."""
    import system.schedule as schedule

    monkeypatch.setenv("USER_SPACE_ROOT", str(tmp_path))
    exists = {"value": False}
    posted = []
    monkeypatch.setattr(schedule, "_reflection_post_exists", lambda _date: exists["value"])
    reflect = ModuleType("cognition.consolidate.reflect")
    reflect.REFLECT_MAX_MEMS = 10
    reflect.filter_reflect_snippets = lambda memories, _date: memories
    monkeypatch.setitem(sys.modules, "cognition.consolidate.reflect", reflect)

    def post(*_args, **_kwargs):
        posted.append(True)
        exists["value"] = True
        return {"success": True}

    runner = schedule.ScheduleRunner(
        memorize=_Memorize(),
        generate_and_post_fn=post,
        user_id="github_alice",
    )
    target = datetime(2026, 8, 25, tzinfo=timezone.utc)
    runner._run_daily_reflect_and_dream(for_date=target)
    runner._run_daily_reflect_and_dream(for_date=target)

    assert posted == [True]


def test_scheduler_lock_is_released_after_critical_section(monkeypatch, tmp_path):
    import system.schedule as schedule

    monkeypatch.setenv("USER_SPACE_ROOT", str(tmp_path))
    with schedule._scheduler_run_lock("github_alice", "jobs") as acquired:
        assert acquired
        with schedule._scheduler_run_lock("github_alice", "jobs") as contender:
            assert not contender
    with schedule._scheduler_run_lock("github_alice", "jobs") as acquired_after_release:
        assert acquired_after_release
