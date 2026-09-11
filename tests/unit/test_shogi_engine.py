from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


games_shogi = _load_module("games_shogi_under_test", "interface/webui/lingo/games_shogi.py")
yaneuraou = _load_module("yaneuraou_under_test", "interface/webui/lingo/yaneuraou.py")


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

    async def fake_to_thread(function, arg):
        calls.append((function, arg))
        return Move("3c3d"), "random"

    monkeypatch.setattr(games_shogi.asyncio, "to_thread", fake_to_thread)
    response = asyncio.run(
        games_shogi.make_move(games_shogi.MoveRequest(move="7g7f"), {"user_id": "user"})
    )

    assert calls == [(games_shogi._ai_move, board)]
    assert response.engine == "random"
    assert games_shogi._games["user"]["engine"] == "random"
