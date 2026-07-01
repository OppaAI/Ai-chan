from __future__ import annotations

from datetime import datetime, timedelta, timezone
import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

mock_requests = SimpleNamespace(get=MagicMock(), post=MagicMock(), RequestException=RuntimeError)
sys.modules.setdefault("requests", mock_requests)
sys.modules.setdefault("openai", MagicMock())
sys.modules.setdefault("dotenv", MagicMock(load_dotenv=lambda: None))

mock_log_module = MagicMock()
mock_log_module.get_logger.return_value = MagicMock()
sys.modules.setdefault("core.log", mock_log_module)
mock_memorize_module = MagicMock()
mock_memorize_module.AikoMemorize = MagicMock
mock_memorize_module.USER_ID = "test-user"
sys.modules.setdefault("core.memorize", mock_memorize_module)
mock_reflect_module = MagicMock()
mock_reflect_module._generate_image.return_value = ""
mock_reflect_module._load_soul.return_value = ""
sys.modules.setdefault("core.reflect", mock_reflect_module)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import social


class DummyResponse:
    def __init__(self, status_code: int, payload: dict[str, object]):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self) -> dict[str, object]:
        return self._payload


def test_refresh_threads_token_updates_process_env_and_optional_env_file(monkeypatch, tmp_path):
    calls = []

    def fake_get(url, params, timeout):
        calls.append((url, params, timeout))
        return DummyResponse(200, {"access_token": "new-token", "expires_in": 5_184_000})

    env_path = tmp_path / ".env"
    env_path.write_text("export THREADS_ACCESS_TOKEN=old-token\nKEEP=value\n", encoding="utf-8")
    monkeypatch.setenv("THREADS_ACCESS_TOKEN", "old-token")
    monkeypatch.setenv("THREADS_API_BASE", "https://graph.threads.net/v1.0")
    monkeypatch.setattr(social.requests, "get", fake_get)

    result = social.refresh_threads_token(persist_env=True, env_path=env_path)

    assert result["ok"] is True
    assert result["expires_in"] == 5_184_000
    assert result["response"]["access_token"] == "[redacted]"
    assert calls == [
        (
            "https://graph.threads.net/v1.0/refresh_access_token",
            {"grant_type": "th_refresh_token", "access_token": "old-token"},
            120,
        )
    ]
    assert social.os.environ["THREADS_ACCESS_TOKEN"] == "new-token"
    written = env_path.read_text(encoding="utf-8")
    assert "export THREADS_ACCESS_TOKEN=new-token" in written
    assert "THREADS_ACCESS_TOKEN_EXPIRES_AT=" in written
    assert "KEEP=value" in written
    assert "new-token" not in str(result["response"])


def test_refresh_threads_token_failure_leaves_env_and_file_untouched(monkeypatch, tmp_path):
    def fake_get(url, params, timeout):
        return DummyResponse(400, {"error": "too early"})

    env_path = tmp_path / ".env"
    original_env = "THREADS_ACCESS_TOKEN=old-token\nKEEP=value\n"
    env_path.write_text(original_env, encoding="utf-8")
    monkeypatch.setenv("THREADS_ACCESS_TOKEN", "old-token")
    monkeypatch.setattr(social.requests, "get", fake_get)

    result = social.refresh_threads_token(persist_env=True, env_path=env_path)

    assert result["ok"] is False
    assert social.os.environ["THREADS_ACCESS_TOKEN"] == "old-token"
    assert env_path.read_text(encoding="utf-8") == original_env


def test_refresh_threads_token_if_due_skips_when_expiry_is_unknown(monkeypatch):
    monkeypatch.delenv("THREADS_ACCESS_TOKEN_EXPIRES_AT", raising=False)

    def fail_refresh(**kwargs):  # pragma: no cover - should never be called
        raise AssertionError("refresh should not be called")

    monkeypatch.setattr(social, "refresh_threads_token", fail_refresh)

    result = social.refresh_threads_token_if_due()

    assert result["ok"] is True
    assert result["skipped"] is True
    assert result["reason"] == "expiry_unknown"


def test_refresh_threads_token_if_due_skips_when_expiry_is_after_threshold(monkeypatch):
    future = datetime.now(timezone.utc) + timedelta(days=59)
    monkeypatch.setenv("THREADS_ACCESS_TOKEN_EXPIRES_AT", future.isoformat())
    monkeypatch.setattr(social, "THREADS_REFRESH_WINDOW_DAYS", 55)

    def fail_refresh(**kwargs):  # pragma: no cover - should never be called
        raise AssertionError("refresh should not be called")

    monkeypatch.setattr(social, "refresh_threads_token", fail_refresh)

    result = social.refresh_threads_token_if_due()

    assert result["ok"] is True
    assert result["skipped"] is True
    assert result["reason"] == "not_due"


def test_refresh_threads_token_if_due_refreshes_inside_threshold(monkeypatch):
    future = datetime.now(timezone.utc) + timedelta(days=50)
    monkeypatch.setenv("THREADS_ACCESS_TOKEN_EXPIRES_AT", future.isoformat())
    monkeypatch.setattr(social, "THREADS_REFRESH_WINDOW_DAYS", 55)

    def fake_refresh(*, persist_env=False):
        return {"ok": True, "persist_env": persist_env}

    monkeypatch.setattr(social, "refresh_threads_token", fake_refresh)

    result = social.refresh_threads_token_if_due(persist_env=True)

    assert result["ok"] is True
    assert result["persist_env"] is True
    assert result["seconds_remaining_before_refresh"] is not None


def test_refresh_threads_token_cli_returns_nonzero_on_failure(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["social.py", "--refresh-threads-token"])
    monkeypatch.setattr(social, "refresh_threads_token", lambda **kwargs: {"ok": False, "error": "bad token"})

    exit_code = social._cmd()

    assert exit_code == 1
    assert '"ok": false' in capsys.readouterr().out
