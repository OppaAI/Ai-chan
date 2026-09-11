"""
Optional KataGo GTP bridge for Go.

Setup:
  export KATAGO_PATH=/path/to/katago
  export KATAGO_MODEL=/path/to/model.bin.gz   # required for analysis/gtp
  export KATAGO_CONFIG=/path/to/gtp_example.cfg  # optional
  export KATAGO_MOVETIME_MS=1000   # soft hint; we use genmove + time settings

If unavailable, callers fall back to random legal moves.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import threading
import time
from typing import List, Optional

log = logging.getLogger(__name__)

_LOCK = threading.Lock()
_proc: Optional[subprocess.Popen] = None
_ready = False
_board_size: Optional[int] = None


def engine_path() -> Optional[str]:
    raw = (os.getenv("KATAGO_PATH") or "").strip()
    if raw:
        return raw
    return shutil.which("katago")


def model_path() -> Optional[str]:
    return (os.getenv("KATAGO_MODEL") or "").strip() or None


def config_path() -> Optional[str]:
    return (os.getenv("KATAGO_CONFIG") or "").strip() or None


def available() -> bool:
    p = engine_path()
    m = model_path()
    if not p or not os.path.isfile(p) or not os.access(p, os.X_OK):
        return False
    # model optional for some builds, but usually required
    if m and not os.path.isfile(m):
        return False
    return True


def _readline(proc: subprocess.Popen, timeout: float = 30.0) -> str:
    import select

    if proc.stdout is None:
        return ""
    try:
        r, _, _ = select.select([proc.stdout], [], [], timeout)
        if not r:
            return ""
    except (ValueError, OSError):
        pass
    line = proc.stdout.readline()
    return line.decode("utf-8", errors="replace").strip() if line else ""


def _write(proc: subprocess.Popen, cmd: str) -> None:
    if proc.stdin is None:
        return
    proc.stdin.write((cmd.strip() + "\n").encode("utf-8"))
    proc.stdin.flush()


def _read_gtp_response(proc: subprocess.Popen, timeout: float = 60.0) -> str:
    """Read until a GTP response line (= or ?)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        line = _readline(proc, timeout=max(0.1, deadline - time.monotonic()))
        if not line:
            if proc.poll() is not None:
                break
            continue
        if line.startswith("=") or line.startswith("?"):
            return line
    return ""


def _shutdown() -> None:
    global _proc, _ready, _board_size
    _ready = False
    _board_size = None
    if _proc is not None:
        try:
            _write(_proc, "quit")
        except Exception:
            pass
        try:
            _proc.kill()
        except Exception:
            pass
        _proc = None


def _ensure_engine() -> Optional[subprocess.Popen]:
    global _proc, _ready
    if _proc is not None and _proc.poll() is None and _ready:
        return _proc
    if _proc is not None:
        _shutdown()

    path = engine_path()
    if not path or not os.path.isfile(path):
        return None

    cmd = [path, "gtp"]
    m = model_path()
    if m:
        cmd.extend(["-model", m])
    cfg = config_path()
    if cfg:
        cmd.extend(["-config", cfg])

    try:
        _proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        # handshake
        _write(_proc, "name")
        resp = _read_gtp_response(_proc, timeout=30.0)
        if not resp.startswith("="):
            log.warning("KataGo name failed: %s", resp)
            _shutdown()
            return None
        _write(_proc, "protocol_version")
        _read_gtp_response(_proc, timeout=10.0)
        _ready = True
        log.info("KataGo GTP ready: %s", path)
        return _proc
    except Exception:
        log.warning("Failed to start KataGo", exc_info=True)
        _shutdown()
        return None


def ensure_ready() -> bool:
    with _LOCK:
        return _ensure_engine() is not None


def best_move_gtp(size: int, moves: List[str], color: str = "B") -> Optional[str]:
    """
    Replay move list and genmove for `color` (B/W).
    moves: GTP strings including 'pass'.
    Returns GTP move e.g. 'D4' or 'pass', or None.
    """
    with _LOCK:
        proc = _ensure_engine()
        if proc is None:
            return None
        global _board_size
        try:
            if _board_size != size:
                _write(proc, f"boardsize {size}")
                resp = _read_gtp_response(proc, timeout=15.0)
                if not resp.startswith("="):
                    log.warning("KataGo boardsize failed: %s", resp)
                    return None
                _board_size = size
            _write(proc, "clear_board")
            _read_gtp_response(proc, timeout=10.0)

            to_play = "B"
            for mv in moves:
                m = mv.strip().upper()
                if m in ("RESIGN", "R"):
                    break
                gtp_mv = "pass" if m in ("PASS", "PA") else m
                _write(proc, f"play {to_play} {gtp_mv}")
                resp = _read_gtp_response(proc, timeout=15.0)
                if not resp.startswith("="):
                    log.warning("KataGo play failed %s %s: %s", to_play, gtp_mv, resp)
                    return None
                to_play = "W" if to_play == "B" else "B"

            color_ch = "B" if color.upper().startswith("B") else "W"
            _write(proc, f"genmove {color_ch}")
            # genmove can take a while
            try:
                timeout = float(os.getenv("KATAGO_GENMOVE_TIMEOUT_S", "30"))
            except ValueError:
                timeout = 30.0
            resp = _read_gtp_response(proc, timeout=timeout)
            if not resp.startswith("="):
                log.warning("KataGo genmove failed: %s", resp)
                return None
            move = resp[1:].strip().split()[0] if resp[1:].strip() else ""
            if not move:
                return None
            if move.upper() in ("RESIGN",):
                return "resign"
            return move.upper() if move.upper() != "PASS" else "pass"
        except Exception:
            log.warning("KataGo best_move_gtp failed", exc_info=True)
            _shutdown()
            return None
