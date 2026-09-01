"""
tests/unit/test_main.py — unit tests for main.py (thin entry point).

Covers parse_args validation, _setup_exit_logging gating, _run_trapped,
_handle_clear_mem/_handle_logout branches, and main() return-code contract.
All heavy deps mocked — no FastAPI/uvicorn/model loads.
Run: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest tests/unit/test_main.py -q --override-ini="addopts="
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

import main as main_module
from main import _handle_clear_mem, _handle_logout, _run_trapped, _setup_exit_logging, parse_args


def _argv(*args):
    return ["main.py", *args]


class TestParseArgs:
    def test_defaults(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", _argv())
        ns = parse_args()
        assert ns.text is False
        assert ns.no_asr is False
        assert ns.debug is False
        assert ns.trace is False
        assert ns.cli is False
        assert ns.clear_mem is False
        assert ns.logout is False
        assert ns.name == ""

    def test_text_and_no_asr_flags(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", _argv("--text", "--no-asr"))
        ns = parse_args()
        assert ns.text is True
        assert ns.no_asr is True

    def test_cli_and_debug(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", _argv("--cli", "--debug"))
        ns = parse_args()
        assert ns.cli is True
        assert ns.debug is True
        assert ns.trace is False

    def test_trace_alone(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", _argv("--trace"))
        ns = parse_args()
        assert ns.trace is True
        assert ns.debug is False

    def test_debug_and_trace_independent(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", _argv("--debug", "--trace"))
        ns = parse_args()
        assert ns.debug is True
        assert ns.trace is True

    def test_name_requires_cli(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", _argv("--name", "Bob"))
        with pytest.raises(SystemExit) as e:
            parse_args()
        assert e.value.code == 2  # argparse error

    def test_name_with_cli_ok(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", _argv("--cli", "--name", "Bob"))
        ns = parse_args()
        assert ns.name == "Bob"

    def test_mutually_exclusive_clear_mem_logout(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", _argv("--clear-mem", "--logout"))
        with pytest.raises(SystemExit) as e:
            parse_args()
        assert e.value.code == 2

    def test_version_exits(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", _argv("--version"))
        with pytest.raises(SystemExit) as e:
            parse_args()
        assert e.value.code == 0

    def test_clear_mem_alone_ok(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", _argv("--clear-mem"))
        ns = parse_args()
        assert ns.clear_mem is True
        assert ns.logout is False


class TestSetupExitLogging:
    def test_no_patch_when_env_not_set(self, monkeypatch):
        monkeypatch.delenv("AIKO_TRACE_EXIT", raising=False)
        orig = main_module._os._exit
        _setup_exit_logging(MagicMock())
        assert main_module._os._exit is orig

    def test_patches_when_enabled(self, monkeypatch):
        monkeypatch.setenv("AIKO_TRACE_EXIT", "1")
        orig = main_module._os._exit
        try:
            _setup_exit_logging(MagicMock())
            assert main_module._os._exit is not orig
        finally:
            main_module._os._exit = orig
            monkeypatch.delenv("AIKO_TRACE_EXIT", raising=False)


class TestRunTrapped:
    def test_success_no_log(self):
        log = MagicMock()
        _run_trapped(log, "ok", lambda: None)
        log.exception.assert_not_called()

    def test_exception_logged_and_reraised(self):
        log = MagicMock()

        def boom():
            raise ValueError("oops")

        with pytest.raises(ValueError, match="oops"):
            _run_trapped(log, "boom", boom)
        log.exception.assert_called_once()


class TestHandleClearMem:
    def test_abort_on_n(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "n")
        assert _handle_clear_mem(MagicMock()) == 0

    def test_abort_on_eof(self, monkeypatch):
        def raise_eof(_):
            raise EOFError
        monkeypatch.setattr("builtins.input", raise_eof)
        assert _handle_clear_mem(MagicMock()) == 0

    def test_confirm_y_calls_memorize(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "y")
        fake_mem = MagicMock()
        fake_cls = MagicMock(return_value=fake_mem)
        with patch.dict("sys.modules", {"cognition.memory.memorize": MagicMock(AikoMemorize=fake_cls)}):
            # need real import path: patch where main imports
            import main as m
            with patch.object(m, "AikoMemorize", fake_cls, create=True):
                # instead patch the import inside function via sys.modules trick
                pass
        # simpler: mock the module import directly
        import importlib
        mock_mod = MagicMock()
        mock_mod.AikoMemorize = fake_cls
        with patch.dict(sys.modules, {"cognition.memory.memorize": mock_mod}):
            assert _handle_clear_mem(MagicMock()) == 0
        fake_cls.assert_called_once()
        fake_mem.clear.assert_called_once()


class TestHandleLogout:
    def test_import_error_returns_1(self):
        with patch.dict(sys.modules, {"interface.cli.cli": None}):
            # force ImportError by making import fail
            import builtins
            orig_import = builtins.__import__

            def fake_import(name, *a, **kw):
                if name == "interface.cli.cli":
                    raise ImportError("missing")
                return orig_import(name, *a, **kw)

            with patch("builtins.__import__", side_effect=fake_import):
                assert _handle_logout(MagicMock()) == 1

    def test_success_returns_0(self):
        mock_cli = MagicMock()
        with patch.dict(sys.modules, {"interface.cli.cli": mock_cli}):
            assert _handle_logout(MagicMock()) == 0
        mock_cli.handle_logout.assert_called_once()


class TestMainReturnCode:
    def test_main_clear_mem_delegates(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", _argv("--clear-mem"))
        with patch("main._handle_clear_mem", return_value=0) as mock_clear, \
             patch("system.config.load_config"), \
             patch("system.log.get_logger", return_value=MagicMock()):
            assert main_module.main() == 0
            mock_clear.assert_called_once()

    def test_main_logout_delegates(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", _argv("--logout"))
        with patch("main._handle_logout", return_value=0) as mock_out, \
             patch("system.config.load_config"), \
             patch("system.log.get_logger", return_value=MagicMock()):
            assert main_module.main() == 0
            mock_out.assert_called_once()

    def test_main_dispatches_cli(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", _argv("--cli"))
        with patch("system.config.load_config"), \
             patch("system.log.get_logger", return_value=MagicMock()), \
             patch.dict(sys.modules, {"interface.cli.cli": MagicMock(run_cli=MagicMock())}):
            # need to ensure run_cli not actually imported from real file
            mock_run = MagicMock()
            with patch.dict(sys.modules, {"interface.cli.cli": MagicMock(run_cli=mock_run)}):
                main_module.main()
                mock_run.assert_called_once()

    def test_main_handles_keyboard_interrupt(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", _argv("--cli"))
        with patch("system.config.load_config"), \
             patch("system.log.get_logger", return_value=MagicMock()), \
             patch.dict(sys.modules, {"interface.cli.cli": MagicMock(run_cli=MagicMock(side_effect=KeyboardInterrupt))}):
            with pytest.raises(KeyboardInterrupt):
                main_module.main()
