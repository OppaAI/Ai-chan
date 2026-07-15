from pathlib import Path

from system import userspace


def test_user_state_root_accepts_legacy_user_space_root(monkeypatch, tmp_path):
    monkeypatch.delenv("USER_STATE_ROOT", raising=False)
    monkeypatch.delenv("AIKO_USER_STATE_ROOT", raising=False)
    monkeypatch.setenv("USER_SPACE_ROOT", str(tmp_path))

    assert userspace.user_state_dir("github_123") == tmp_path / "github_123"


def test_user_profile_path_points_to_profile_user_md(monkeypatch, tmp_path):
    monkeypatch.setenv("USER_STATE_ROOT", str(tmp_path))
    monkeypatch.delenv("USER_PROFILE_PATH", raising=False)

    assert userspace.user_profile_path("github_123") == (
        tmp_path / "github_123" / "profile" / "user.md"
    ).resolve()
