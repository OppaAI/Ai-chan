"""
tests/test_userspace.py
Tests for system/userspace.py — per-user path resolution and context vars.

Focus areas:
  - guest sentinel never creates a directory (no stray folders pre-login)
  - real user_id gets a locked-down (0o700) directory
  - path sanitization strips unsafe characters consistently
  - contextvar set/reset round-trips cleanly (important for multi-request
    servers where a stale contextvar leaking across requests would put one
    user's data under another user's path)
  - env var fallback chain and override precedence (WORKSPACE_ROOT,
    USER_PROFILE_PATH, USER_STATE_ROOT aliases)
  - current_display_name() fallback chain, since this is exactly what
    memory/memorize.py leans on to know who it's talking to
"""
from __future__ import annotations

import os
import stat

import pytest

from system.userspace import (
    current_display_name,
    current_user_id,
    normalize_user_id,
    reset_current_display_name,
    reset_current_user_id,
    set_current_display_name,
    set_current_user_id,
    user_profile_path,
    user_state_dir,
    user_state_path,
    user_workspace_root,
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Make sure no ambient env vars leak between tests."""
    for var in (
        "AIKO_USER_ID", "AIKO_DISPLAY_NAME", "USER_STATE_ROOT",
        "AIKO_USER_STATE_ROOT", "USER_SPACE_ROOT", "WORKSPACE_ROOT",
        "USER_PROFILE_PATH",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def state_root(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_STATE_ROOT", str(tmp_path))
    return tmp_path


# ─────────────────────────────────────────────────────────────────────────────
# current_user_id / contextvar behavior
# ─────────────────────────────────────────────────────────────────────────────

class TestCurrentUserId:
    def test_defaults_to_guest(self):
        assert current_user_id() == "guest"

    def test_env_var_override(self, monkeypatch):
        monkeypatch.setenv("AIKO_USER_ID", "env_user")
        assert current_user_id() == "env_user"

    def test_contextvar_takes_priority_over_env(self, monkeypatch):
        monkeypatch.setenv("AIKO_USER_ID", "env_user")
        token = set_current_user_id("ctx_user")
        try:
            assert current_user_id() == "ctx_user"
        finally:
            reset_current_user_id(token)
        # after reset, falls back to env again
        assert current_user_id() == "env_user"

    def test_reset_restores_prior_value_not_just_default(self, monkeypatch):
        """Nested set/reset should behave like a stack, not clobber to None --
        important if two layers of request handling both set the contextvar."""
        outer_token = set_current_user_id("outer")
        try:
            inner_token = set_current_user_id("inner")
            try:
                assert current_user_id() == "inner"
            finally:
                reset_current_user_id(inner_token)
            assert current_user_id() == "outer"
        finally:
            reset_current_user_id(outer_token)


class TestCurrentDisplayName:
    """This fallback chain is exactly what memorize.py's _extract_facts
    depends on to label facts with the right name in the LLM prompt."""

    def test_defaults_to_user_id_when_nothing_set(self, monkeypatch):
        monkeypatch.setenv("AIKO_USER_ID", "some_user")
        assert current_display_name() == "some_user"

    def test_env_display_name_overrides_user_id_fallback(self, monkeypatch):
        monkeypatch.setenv("AIKO_USER_ID", "some_user")
        monkeypatch.setenv("AIKO_DISPLAY_NAME", "Oppa")
        assert current_display_name() == "Oppa"

    def test_contextvar_takes_priority_over_env(self, monkeypatch):
        monkeypatch.setenv("AIKO_DISPLAY_NAME", "env_name")
        token = set_current_display_name("ctx_name")
        try:
            assert current_display_name() == "ctx_name"
        finally:
            reset_current_display_name(token)
        assert current_display_name() == "env_name"

    def test_reset_does_not_leak_into_next_call(self):
        """Regression guard for the exact bug class implied by 'Aiko doesn't
        know who she's talking to' -- a display name set for user A must not
        still be active once reset, so a subsequent unrelated call (e.g. a
        background write for a different user) doesn't silently inherit it."""
        token = set_current_display_name("UserA")
        reset_current_display_name(token)
        assert current_display_name() != "UserA"


class TestNormalizeUserId:
    def test_strips_unsafe_characters(self):
        result = normalize_user_id("github", "some user!@#/id")
        assert result == "github_some_user_id"

    def test_falls_back_to_local_and_guest_on_empty(self):
        assert normalize_user_id(None, None) == "local_guest"

    def test_provider_and_id_both_scoped(self):
        result = normalize_user_id("patreon", 12345)
        assert result.startswith("patreon_")
        assert "12345" in result


# ─────────────────────────────────────────────────────────────────────────────
# Filesystem isolation
# ─────────────────────────────────────────────────────────────────────────────

class TestUserStateDir:
    def test_guest_does_not_create_directory(self, state_root):
        path = user_state_dir("guest")
        assert not path.exists(), "guest sentinel must never touch disk before login"

    def test_real_user_creates_locked_directory(self, state_root):
        path = user_state_dir("real_user_123")
        assert path.exists()
        mode = stat.S_IMODE(os.stat(path).st_mode)
        assert mode == 0o700

    def test_sanitizes_unsafe_user_id_in_path(self, state_root):
        path = user_state_dir("weird/../user id!")
        assert ".." not in str(path)
        assert path.is_relative_to(state_root)

    def test_two_users_get_isolated_paths(self, state_root):
        a = user_state_dir("user_a")
        b = user_state_dir("user_b")
        assert a != b
        assert a.parent == b.parent == state_root

    def test_uses_current_user_id_when_none_passed(self, state_root, monkeypatch):
        monkeypatch.setenv("AIKO_USER_ID", "implicit_user")
        path = user_state_dir()
        assert path.name == "implicit_user"


class TestUserStatePath:
    def test_joins_filename_under_user_dir(self, state_root):
        path = user_state_path("memory/memory.db", user_id="real_user")
        assert path == user_state_dir("real_user") / "memory/memory.db"


class TestUserWorkspaceRoot:
    def test_defaults_under_user_state_dir(self, state_root):
        path = user_workspace_root("real_user")
        assert path == (user_state_dir("real_user") / "workspace").resolve()

    def test_workspace_root_env_overrides_and_is_shared(self, state_root, tmp_path, monkeypatch):
        shared = tmp_path / "shared_workspace"
        monkeypatch.setenv("WORKSPACE_ROOT", str(shared))
        path_a = user_workspace_root("user_a")
        path_b = user_workspace_root("user_b")
        # explicit override is NOT per-user -- both users resolve to the same path
        assert path_a == path_b == shared.resolve()


class TestUserProfilePath:
    def test_defaults_to_profile_user_md(self, state_root):
        path = user_profile_path("real_user")
        assert path == (user_state_dir("real_user") / "profile/USER.md").resolve()

    def test_env_override_takes_precedence(self, state_root, tmp_path, monkeypatch):
        custom = tmp_path / "custom_profile.md"
        monkeypatch.setenv("USER_PROFILE_PATH", str(custom))
        path = user_profile_path("real_user")
        assert path == custom.resolve()
