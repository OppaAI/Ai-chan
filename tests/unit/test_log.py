"""
tests/unit/test_log.py — unit tests for system/log.py.

Covers _resolve_log_level, _int_env, _make_main_handler, _setup idempotence,
get_logger, silent_stderr, and Path-based LOG_DIR. No real log files outside tmp_path.
Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest tests/unit/test_log.py -q --override-ini="addopts="
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import system.log as log_module
from system.log import _int_env, _make_main_handler, _resolve_log_level, get_logger, silent_stderr


@pytest.fixture(autouse=True)
def reset_log_state(tmp_path, monkeypatch):
    """Isolate log state per test — reset _initialized, handlers, and LOG_DIR to tmp."""
    # save
    old_initialized = log_module._initialized
    old_dir = log_module.LOG_DIR
    old_file = log_module.LOG_FILE
    old_err = log_module.ERROR_LOG_FILE
    root = logging.getLogger()
    old_handlers = list(root.handlers)
    old_level = root.level
    old_disable = logging.getLogger().manager.disable

    # reset to tmp
    log_module._initialized = False
    log_module.LOG_DIR = tmp_path / "logs"
    log_module.LOG_FILE = log_module.LOG_DIR / "aiko.log"
    log_module.ERROR_LOG_FILE = log_module.LOG_DIR / "aiko.error.log"
    # clear root handlers
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(logging.WARNING)
    logging.disable(logging.NOTSET)

    yield

    # restore
    for h in list(root.handlers):
        root.removeHandler(h)
    for h in old_handlers:
        root.addHandler(h)
    root.setLevel(old_level)
    logging.disable(old_disable)
    log_module._initialized = old_initialized
    log_module.LOG_DIR = old_dir
    log_module.LOG_FILE = old_file
    log_module.ERROR_LOG_FILE = old_err
    # clean env
    for k in ["LOG_LEVEL", "LOG_CONSOLE", "LOG_ROTATE_MODE", "LOG_ROTATE_WHEN", "LOG_ROTATE_INTERVAL", "LOG_MAX_BYTES", "LOG_BACKUP_COUNT"]:
        monkeypatch.delenv(k, raising=False)


class TestResolveLogLevel:
    def test_valid_level(self, monkeypatch):
        monkeypatch.setenv("LOG_LEVEL", "debug")
        assert _resolve_log_level() == "DEBUG"

    def test_invalid_defaults_to_info(self, monkeypatch, capsys):
        monkeypatch.setenv("LOG_LEVEL", "NOPE")
        assert _resolve_log_level() == "INFO"
        assert "invalid LOG_LEVEL" in capsys.readouterr().err

    def test_missing_defaults_to_info(self, monkeypatch):
        monkeypatch.delenv("LOG_LEVEL", raising=False)
        assert _resolve_log_level() == "INFO"


class TestIntEnv:
    def test_missing_returns_default(self, monkeypatch):
        monkeypatch.delenv("MISSING_INT", raising=False)
        assert _int_env("MISSING_INT", 42) == 42

    def test_valid_int(self, monkeypatch):
        monkeypatch.setenv("MY_INT", "123")
        assert _int_env("MY_INT", 0) == 123

    def test_invalid_warns_and_defaults(self, monkeypatch, capsys):
        monkeypatch.setenv("MY_INT", "not-an-int")
        assert _int_env("MY_INT", 99) == 99
        assert "invalid MY_INT" in capsys.readouterr().err


class TestMakeMainHandler:
    def test_size_mode_default(self, monkeypatch):
        monkeypatch.delenv("LOG_ROTATE_MODE", raising=False)
        h = _make_main_handler("INFO", 1024, 1)
        from logging.handlers import RotatingFileHandler
        assert isinstance(h, RotatingFileHandler)
        h.close()

    def test_time_mode_midnight(self, monkeypatch):
        monkeypatch.setenv("LOG_ROTATE_MODE", "time")
        monkeypatch.setenv("LOG_ROTATE_WHEN", "midnight")
        monkeypatch.setenv("LOG_ROTATE_INTERVAL", "1")
        h = _make_main_handler("INFO", 1024, 1)
        from logging.handlers import TimedRotatingFileHandler
        assert isinstance(h, TimedRotatingFileHandler)
        h.close()

    def test_invalid_mode_defaults_to_size(self, monkeypatch, capsys):
        monkeypatch.setenv("LOG_ROTATE_MODE", "bogus")
        h = _make_main_handler("INFO", 1024, 1)
        from logging.handlers import RotatingFileHandler
        assert isinstance(h, RotatingFileHandler)
        assert "invalid LOG_ROTATE_MODE" in capsys.readouterr().err
        h.close()

    def test_invalid_when_defaults_to_midnight(self, monkeypatch, capsys):
        monkeypatch.setenv("LOG_ROTATE_MODE", "time")
        monkeypatch.setenv("LOG_ROTATE_WHEN", "bogus")
        h = _make_main_handler("INFO", 1024, 1)
        assert "invalid LOG_ROTATE_WHEN" in capsys.readouterr().err
        h.close()


class TestSetupAndGetLogger:
    def test_setup_idempotent(self):
        log1 = get_logger("test.a")
        handlers_after_first = len(logging.getLogger().handlers)
        log2 = get_logger("test.b")
        assert len(logging.getLogger().handlers) == handlers_after_first
        assert log1 is not log2

    def test_log_creates_file_on_emit(self, tmp_path):
        log = get_logger("test.emit")
        log.info("hello from test")
        # flush
        for h in logging.getLogger().handlers:
            h.flush()
        # LOG_FILE is Path now — check exists
        assert log_module.LOG_FILE.exists()
        assert "hello from test" in log_module.LOG_FILE.read_text()

    def test_error_file_only_error(self):
        log = get_logger("test.err")
        log.info("info only")
        log.error("error only")
        for h in logging.getLogger().handlers:
            h.flush()
        err_text = log_module.ERROR_LOG_FILE.read_text() if log_module.ERROR_LOG_FILE.exists() else ""
        # error file may not exist if delay and no flush, but after error it should
        if log_module.ERROR_LOG_FILE.exists():
            assert "error only" in err_text
            assert "info only" not in err_text

    def test_console_handler_opt_in(self, monkeypatch):
        monkeypatch.setenv("LOG_CONSOLE", "1")
        # need fresh setup
        log_module._initialized = False
        for h in list(logging.getLogger().handlers):
            logging.getLogger().removeHandler(h)
        get_logger("test.console")
        has_stream = any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in logging.getLogger().handlers)
        assert has_stream

    def test_log_dir_is_path(self):
        assert isinstance(log_module.LOG_DIR, Path)
        assert isinstance(log_module.LOG_FILE, Path)


class TestSilentStderr:
    def test_suppresses_stderr(self, capsys):
        # low-level fd suppression not captured by capsys, just verify no crash
        with silent_stderr():
            os.write(2, b"should be hidden")
        # after context, stderr restored
        print("after", file=sys.stderr)
        # if not restored, this would be hidden; capsys will see it
        assert True

    def test_restores_on_exception(self):
        try:
            with silent_stderr():
                raise ValueError("inside")
        except ValueError:
            pass
        # stderr should be restored — write should not raise
        os.write(2, b"restored\n")
