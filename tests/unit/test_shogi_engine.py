from __future__ import annotations

import asyncio
import importlib.util
import sys
import time
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


games_shogi = _load_module("games_shogi_under_test", "interface/webui/shogi/games_shogi.py")
yaneuraou = _load_module("yaneuraou_under_test", "interface/webui/shogi/yaneuraou.py")


class _FakeProcess:
    def __init__(self, returncode=None):
        self.returncode = returncode
        self.killed = False

    def poll(self):
        return self.returncode

    def kill(self):
        self.killed = True


def test_normalized_movetime_uses_default_and_clamps(monkeypatch):
    monkeypatch.setenv("YANEURAOU_MOVETIME_MS", "not-an-integer")
    assert yaneuraou.normalized_movetime_ms() == 800
    assert yaneuraou.normalized_movetime_ms(-1) == 50
    assert yaneuraou.normalized_movetime_ms(50_000) == 30_000


def test_timeout_stops_and_drains_before_next_readiness_check(monkeypatch):
    proc = _FakeProcess()
    commands = []
    reads = iter([""] * 52 + ["bestmove 7g7f", "readyok"])
    monkeypatch.setattr(yaneuraou, "_proc", proc)
    monkeypatch.setattr(yaneuraou, "_ready", True)
    monkeypatch.setattr(yaneuraou, "_write", lambda _proc, command: commands.append(command))
    monkeypatch.setattr(yaneuraou, "_readline", lambda _proc, timeout: next(reads))

    assert yaneuraou.best_move_usi("position", movetime_ms=50) is None
    assert commands[-1] == "stop"
    assert yaneuraou._proc is proc
    assert yaneuraou._ready is False

    assert yaneuraou._ensure_engine() is proc
    assert commands[-1] == "isready"
    assert yaneuraou._ready is True


def test_failed_timeout_drain_shuts_engine_down(monkeypatch):
    proc = _FakeProcess(returncode=1)
    commands = []
    monkeypatch.setattr(yaneuraou, "_proc", proc)
    monkeypatch.setattr(yaneuraou, "_ready", True)
    monkeypatch.setattr(yaneuraou, "_write", lambda _proc, command: commands.append(command))
    monkeypatch.setattr(yaneuraou, "_readline", lambda _proc, timeout: "")

    assert yaneuraou._stop_and_drain(proc) is False
    assert commands == ["stop", "quit"]
    assert proc.killed is True
    assert yaneuraou._proc is None
    assert yaneuraou._ready is False


def test_state_response_uses_stored_engine(monkeypatch):
    class Board:
        def sfen(self):
            return "test-sfen"

    games_shogi._games["user"] = {
        "board": Board(),
        "mode": "vs_ai",
        "status": "playing",
        "engine": "random",
    }
    monkeypatch.setattr(games_shogi, "_turn_label", lambda board: "black")
    monkeypatch.setattr(games_shogi, "_status_for", lambda board: "playing")

    assert games_shogi._state_response("user").engine == "random"
    assert asyncio.run(games_shogi.game_state({"user_id": "user"})).engine == "random"
    assert asyncio.run(games_shogi.resign({"user_id": "user"})).engine == "random"


def test_make_move_runs_ai_search_in_worker_and_updates_engine(monkeypatch):
    class Move:
        def __init__(self, value):
            self.value = value

        def __eq__(self, other):
            return isinstance(other, Move) and self.value == other.value

        def usi(self):
            return self.value

    class MoveFactory:
        @staticmethod
        def from_usi(value):
            return Move(value)

    class Board:
        turn = 0

        def __init__(self):
            self.legal_moves = [Move("7g7f")]

        def push(self, move):
            self.legal_moves = [Move("3c3d")]

        def sfen(self):
            return "test-sfen"

    board = Board()
    games_shogi._games["user"] = {
        "board": board,
        "mode": "vs_ai",
        "status": "playing",
        "engine": "yaneuraou",
    }
    monkeypatch.setattr(games_shogi, "_import_shogi", lambda: type("Shogi", (), {"Move": MoveFactory}))
    monkeypatch.setattr(games_shogi, "_status_for", lambda board: "playing")
    monkeypatch.setattr(games_shogi, "_turn_label", lambda board: "black")
    calls = []

    async def fake_to_thread(function, *args):
        calls.append((function, *args))
        if function is games_shogi._banter_for:
            return "mocked banter"
        return Move("3c3d"), "random"

    monkeypatch.setattr(games_shogi.asyncio, "to_thread", fake_to_thread)
    response = asyncio.run(
        games_shogi.make_move(games_shogi.MoveRequest(move="7g7f"), {"user_id": "user"})
    )

    assert calls == [
        (games_shogi._ai_move, board, None, None),
        (games_shogi._banter_for, "3c3d", None, "playing"),
    ]
    assert response.engine == "random"
    assert response.ai_comment.endswith("mocked banter")
    assert games_shogi._games["user"]["engine"] == "random"


def test_normalized_depth_caps_and_defaults():
    assert yaneuraou.normalized_depth() is None
    assert yaneuraou.normalized_depth(None) is None
    assert yaneuraou.normalized_depth("nope") is None
    assert yaneuraou.normalized_depth(0) == 1
    assert yaneuraou.normalized_depth(3) == 3
    assert yaneuraou.normalized_depth(99) == 32


def test_difficulty_name_parsing(monkeypatch):
    monkeypatch.delenv("SHOGI_DIFFICULTY", raising=False)
    assert games_shogi.difficulty_name() == "medium"
    assert games_shogi.difficulty_name("HARD") == "hard"
    assert games_shogi.difficulty_name("  easy  ") == "easy"
    assert games_shogi.difficulty_name("grandmaster") == "medium"
    monkeypatch.setenv("SHOGI_DIFFICULTY", "easy")
    assert games_shogi.difficulty_name() == "easy"
    assert games_shogi.difficulty_name("hard") == "hard"


def test_ai_move_blunder_skips_engine(monkeypatch):
    class Move:
        def __init__(self, value):
            self.value = value

        def __eq__(self, other):
            return isinstance(other, Move) and self.value == other.value

        def usi(self):
            return self.value

    class Board:
        legal_moves = [Move("7g7f")]

        def sfen(self):
            return "test-sfen"

    def _boom():
        raise AssertionError("engine must not be consulted on a blunder")

    monkeypatch.setattr(games_shogi.random, "random", lambda: 0.0)
    monkeypatch.setattr(games_shogi, "_engine_bridge", _boom)
    move, engine = games_shogi._ai_move(Board(), difficulty="easy")
    assert move.usi() == "7g7f"
    assert engine == "random"


def test_ai_move_hard_asks_engine_with_full_strength(monkeypatch):
    class Move:
        def __init__(self, value):
            self.value = value

        def __eq__(self, other):
            return isinstance(other, Move) and self.value == other.value

        def usi(self):
            return self.value

    class MoveFactory:
        @staticmethod
        def from_usi(value):
            return Move(value)

    class Board:
        legal_moves = [Move("7g7f")]

        def sfen(self):
            return "test-sfen"

    seen = {}

    class FakeBridge:
        @staticmethod
        def best_move_usi(sfen, movetime_ms=None, depth="unset"):
            seen["movetime_ms"] = movetime_ms
            seen["depth"] = depth
            return "7g7f"

    monkeypatch.setattr(games_shogi, "_import_shogi", lambda: type("Shogi", (), {"Move": MoveFactory}))
    monkeypatch.setattr(games_shogi, "_engine_bridge", lambda: FakeBridge)
    move, engine = games_shogi._ai_move(Board(), difficulty="hard")
    assert move.usi() == "7g7f"
    assert engine == "yaneuraou"
    assert seen == {"movetime_ms": None, "depth": None}


def test_ai_move_medium_caps_depth(monkeypatch):
    class Move:
        def __init__(self, value):
            self.value = value

        def __eq__(self, other):
            return isinstance(other, Move) and self.value == other.value

        def usi(self):
            return self.value

    class MoveFactory:
        @staticmethod
        def from_usi(value):
            return Move(value)

    class Board:
        legal_moves = [Move("7g7f")]

        def sfen(self):
            return "test-sfen"

    seen = {}

    class FakeBridge:
        @staticmethod
        def best_move_usi(sfen, movetime_ms=None, depth="unset"):
            seen["movetime_ms"] = movetime_ms
            seen["depth"] = depth
            return "7g7f"

    monkeypatch.setattr(games_shogi, "_import_shogi", lambda: type("Shogi", (), {"Move": MoveFactory}))
    monkeypatch.setattr(games_shogi.random, "random", lambda: 0.99)  # no blunder
    monkeypatch.setattr(games_shogi, "_engine_bridge", lambda: FakeBridge)
    move, engine = games_shogi._ai_move(Board(), difficulty="medium")
    assert move.usi() == "7g7f"
    assert engine == "yaneuraou"
    assert seen == {"movetime_ms": 400, "depth": 6}


def test_banter_disabled_skips_llm_call(monkeypatch):
    class Move:
        def __init__(self, value):
            self.value = value

        def __eq__(self, other):
            return isinstance(other, Move) and self.value == other.value

        def usi(self):
            return self.value

    class MoveFactory:
        @staticmethod
        def from_usi(value):
            return Move(value)

    class Board:
        turn = 0

        def __init__(self):
            self.legal_moves = [Move("7g7f")]

        def push(self, move):
            self.legal_moves = [Move("3c3d")]

        def sfen(self):
            return "test-sfen"

    board = Board()
    games_shogi._games["user"] = {
        "board": board,
        "mode": "vs_ai",
        "status": "playing",
        "engine": "yaneuraou",
    }
    monkeypatch.setenv("SHOGI_BANTER", "0")
    monkeypatch.setattr(games_shogi, "_import_shogi", lambda: type("Shogi", (), {"Move": MoveFactory}))
    monkeypatch.setattr(games_shogi, "_status_for", lambda board: "playing")
    monkeypatch.setattr(games_shogi, "_turn_label", lambda board: "black")
    calls = []

    async def fake_to_thread(function, *args):
        calls.append((function, *args))
        return Move("3c3d"), "random"

    monkeypatch.setattr(games_shogi.asyncio, "to_thread", fake_to_thread)
    response = asyncio.run(
        games_shogi.make_move(games_shogi.MoveRequest(move="7g7f"), {"user_id": "user"})
    )

    assert calls == [(games_shogi._ai_move, board, None, None)]
    assert response.ai_comment == "Aiko plays 3c3d"


def test_banter_for_returns_none_without_llm():
    assert games_shogi._banter_for("7g7f", "medium", "playing") is None


def test_clock_charge_main_then_byoyomi(monkeypatch):
    monkeypatch.setenv("SHOGI_MAIN_TIME_S", "60")
    monkeypatch.setenv("SHOGI_BYOYOMI_S", "30")
    game = {"clock": games_shogi._new_clock()}
    assert game["clock"] is not None
    assert games_shogi._charge_clock(game, "black", 10_000) is True
    assert game["clock"]["remaining"]["black"] == pytest.approx(50_000)
    assert games_shogi._charge_clock(game, "black", 50_000) is True
    assert game["clock"]["remaining"]["black"] == 0
    assert games_shogi._charge_clock(game, "black", 31_000) is False


def test_clock_off_when_zeroed(monkeypatch):
    monkeypatch.setenv("SHOGI_MAIN_TIME_S", "0")
    monkeypatch.setenv("SHOGI_BYOYOMI_S", "0")
    assert games_shogi._new_clock() is None
    assert games_shogi._clock_view({"clock": None}) == (None, None, None)


def test_make_move_flags_on_timeout(monkeypatch):
    class Move:
        def __init__(self, value):
            self.value = value

        def __eq__(self, other):
            return isinstance(other, Move) and self.value == other.value

        def usi(self):
            return self.value

    class MoveFactory:
        @staticmethod
        def from_usi(value):
            return Move(value)

    class Board:
        turn = 0

        def __init__(self):
            self.legal_moves = [Move("7g7f")]
            self.pushed = []

        def push(self, move):
            self.pushed.append(move)

        def sfen(self):
            return "test-sfen"

    monkeypatch.setenv("SHOGI_MAIN_TIME_S", "600")
    monkeypatch.setenv("SHOGI_BYOYOMI_S", "30")
    board = Board()
    games_shogi._games["user"] = {
        "board": board,
        "mode": "vs_ai",
        "status": "playing",
        "engine": "yaneuraou",
        "clock": {
            "remaining": {"black": 1_000.0, "white": 600_000.0},
            "byoyomi_ms": 30_000.0,
            "stamp": time.monotonic() - 60.0,  # 60s think > 30s byoyomi
        },
    }
    monkeypatch.setattr(games_shogi, "_import_shogi", lambda: type("Shogi", (), {"Move": MoveFactory}))
    monkeypatch.setattr(games_shogi, "_turn_label", lambda board: "black")
    monkeypatch.setattr(games_shogi, "_status_for", lambda board: "playing")
    calls = []

    async def fake_to_thread(function, *args):
        calls.append((function, *args))
        raise AssertionError("no AI search after flag")

    monkeypatch.setattr(games_shogi.asyncio, "to_thread", fake_to_thread)
    response = asyncio.run(
        games_shogi.make_move(games_shogi.MoveRequest(move="7g7f"), {"user_id": "user"})
    )

    assert calls == []
    assert board.pushed == []
    assert response.status == "timeout"
    assert "Flag" in (response.ai_comment or "")


def test_ai_move_caps_movetime_by_clock(monkeypatch):
    class Move:
        def __init__(self, value):
            self.value = value

        def __eq__(self, other):
            return isinstance(other, Move) and self.value == other.value

        def usi(self):
            return self.value

    class MoveFactory:
        @staticmethod
        def from_usi(value):
            return Move(value)

    class Board:
        legal_moves = [Move("7g7f")]

        def sfen(self):
            return "test-sfen"

    seen = {}

    class FakeBridge:
        @staticmethod
        def best_move_usi(sfen, movetime_ms=None, depth="unset"):
            seen["movetime_ms"] = movetime_ms
            seen["depth"] = depth
            return "7g7f"

    monkeypatch.setattr(games_shogi, "_import_shogi", lambda: type("Shogi", (), {"Move": MoveFactory}))
    monkeypatch.setattr(games_shogi, "_engine_bridge", lambda: FakeBridge)
    move, engine = games_shogi._ai_move(Board(), difficulty="hard", movetime_cap_ms=120.0)
    assert move.usi() == "7g7f"
    assert engine == "yaneuraou"
    assert seen == {"movetime_ms": 120, "depth": None}
