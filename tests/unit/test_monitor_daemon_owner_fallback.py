"""Unit tests for the monitor daemon's headless owner-store memory fallback.

Without a WebUI login since boot, the shared AikoMemorize is still an
unopened guest and social replies ran with memorize=None — no recall, no
interaction-memory saving. The daemon must fall back to a dedicated handle
bound to the owner's on-disk store instead.
"""

import importlib
import sys
import types

import pytest

monitor_daemon = importlib.import_module("interface.mcp_server.social.monitor_daemon")


class FakeMemorize:
    def __init__(self, open_: bool = True):
        self._open = open_
        self.switched_to: list[str] = []

    def is_open(self):
        return self._open

    def switch_user(self, uid):
        self.switched_to.append(uid)


@pytest.fixture(autouse=True)
def _reset_handles():
    monitor_daemon._SHARED_MEMORIZE["ref"] = None
    monitor_daemon._FALLBACK_MEMORIZE["ref"] = None
    yield
    monitor_daemon._SHARED_MEMORIZE["ref"] = None
    monitor_daemon._FALLBACK_MEMORIZE["ref"] = None


def test_owner_user_id_env_override(monkeypatch):
    monkeypatch.setenv("AIKO_USER_ID", "github_123")
    assert monitor_daemon._owner_user_id() == "github_123"


def test_owner_user_id_unique_on_disk(tmp_path, monkeypatch):
    (tmp_path / "github_1" / "memory").mkdir(parents=True)
    (tmp_path / "github_1" / "memory" / "memory.db").touch()
    (tmp_path / "github_1" / "profile").mkdir()
    (tmp_path / "github_1" / "profile" / "USER.md").touch()
    monkeypatch.delenv("AIKO_USER_ID", raising=False)
    monkeypatch.setattr(
        "system.userspace._user_state_root_value", lambda: str(tmp_path)
    )
    assert monitor_daemon._owner_user_id() == "github_1"


def test_owner_user_id_ambiguous_returns_none(tmp_path, monkeypatch):
    for name in ("github_1", "github_2"):
        (tmp_path / name / "memory").mkdir(parents=True)
        (tmp_path / name / "memory" / "memory.db").touch()
        (tmp_path / name / "profile").mkdir()
        (tmp_path / name / "profile" / "USER.md").touch()
    monkeypatch.delenv("AIKO_USER_ID", raising=False)
    monkeypatch.setattr(
        "system.userspace._user_state_root_value", lambda: str(tmp_path)
    )
    assert monitor_daemon._owner_user_id() is None


def test_bound_memorize_prefers_open_shared_instance():
    shared = FakeMemorize(open_=True)
    monitor_daemon._SHARED_MEMORIZE["ref"] = shared
    fallback = FakeMemorize(open_=True)
    monitor_daemon._FALLBACK_MEMORIZE["ref"] = fallback
    assert monitor_daemon._bound_memorize() is shared


def test_bound_memorize_falls_back_when_shared_is_guest(monkeypatch):
    monitor_daemon._SHARED_MEMORIZE["ref"] = FakeMemorize(open_=False)
    fallback = FakeMemorize(open_=True)
    calls = []
    monkeypatch.setattr(
        monitor_daemon, "_fallback_owner_memorize", lambda: calls.append(1) or fallback
    )
    assert monitor_daemon._bound_memorize() is fallback
    assert len(calls) == 1


def test_fallback_owner_memorize_constructs_binds_and_caches(monkeypatch):
    stub_module = types.ModuleType("cognition.memory.memorize")
    constructed = []

    class StubAikoMemorize:
        def __init__(self, silent=False):
            constructed.append(("ctor", silent))

        def switch_user(self, uid):
            constructed.append(("switch", uid))

    stub_module.AikoMemorize = StubAikoMemorize
    monkeypatch.setitem(sys.modules, "cognition.memory.memorize", stub_module)
    monkeypatch.setattr(monitor_daemon, "_owner_user_id", lambda: "github_123")

    first = monitor_daemon._fallback_owner_memorize()
    second = monitor_daemon._fallback_owner_memorize()

    assert first is second
    assert constructed == [("ctor", True), ("switch", "github_123")]
    assert monitor_daemon._FALLBACK_MEMORIZE["ref"] is first


def test_fallback_owner_memorize_returns_none_without_owner(monkeypatch):
    monkeypatch.setattr(monitor_daemon, "_owner_user_id", lambda: None)
    assert monitor_daemon._fallback_owner_memorize() is None


def test_fallback_owner_memorize_swallows_bind_failure(monkeypatch):
    class ExplodingAikoMemorize:
        def __init__(self, silent=False):
            raise RuntimeError("no embedder")

    stub_module = types.ModuleType("cognition.memory.memorize")
    stub_module.AikoMemorize = ExplodingAikoMemorize
    monkeypatch.setitem(sys.modules, "cognition.memory.memorize", stub_module)
    monkeypatch.setattr(monitor_daemon, "_owner_user_id", lambda: "github_123")

    assert monitor_daemon._fallback_owner_memorize() is None
    assert monitor_daemon._FALLBACK_MEMORIZE["ref"] is None
