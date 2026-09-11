"""
Optional YaneuraOu USI bridge.

Aiko asks the engine for the best move; she does not embed search herself.

Setup:
  1. Build/download YaneuraOu: https://github.com/yaneurao/YaneuraOu
  2. export YANEURAOU_PATH=/path/to/YaneuraOu  (or YaneuraOu-byoyomi)
  3. Optional: YANEURAOU_MOVETIME_MS=800  (default 800)

If the binary is missing or errors, callers should fall back to another AI.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import threading
from typing import Optional

log = logging.getLogger(__name__)

_DEFAULT_MOVETIME_MS = int(os.getenv("YANEURAOU_MOVETIME_MS", "800"))
_LOCK = threading.Lock()
_proc: Optional[subprocess.Popen] = None
_ready = False


def engine_path() -> Optional[str]:
    """Resolve YaneuraOu binary path, or None if not configured/found."""
    raw = (os.getenv("YANEURAOU_PATH") or "").strip()
    if raw:
        return raw if os.path.isfile(raw) and os.access(raw, os.X_OK) else raw
    # Common names on PATH
    for name in ("YaneuraOu", "yaneuraou", "YaneuraOu-byoyomi", "YaneuraOu_NNUE"):
        found = shutil.which(name)
        if found:
            return found
    return None


def available() -> bool:
    p = engine_path()
    return bool(p and os.path.isfile(p) and os.access(p, os.X_OK))


def _readline(proc: subprocess.Popen, timeout: float = 30.0) -> str:
    import select

    if proc.stdout is None:
        return ""
    # Blocking readline with crude timeout via select when possible
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


def _ensure_engine() -> Optional[subprocess.Popen]:
    global _proc, _ready
    if _proc is not None and _proc.poll() is None and _ready:
        return _proc

    path = engine_path()
    if not path or not os.path.isfile(path):
        log.debug("YaneuraOu not found (set YANEURAOU_PATH)")
        return None

    try:
        _proc = subprocess.Popen(
            [path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        _write(_proc, "usi")
        # Wait for usiok
        for _ in range(200):
            line = _readline(_proc, timeout=5.0)
            if line == "usiok":
                break
            if not line and _proc.poll() is not None:
                log.warning("YaneuraOu exited during usi handshake")
                _proc = None
                return None
        else:
            log.warning("YaneuraOu usiok timeout")
            _shutdown()
            return None

        # Light options — keep memory small for a companion process
        _write(_proc, "setoption name Threads value 1")
        _write(_proc, "setoption name Hash value 64")
        _write(_proc, "setoption name USI_Ponder value false")
        _write(_proc, "isready")
        for _ in range(200):
            line = _readline(_proc, timeout=10.0)
            if line == "readyok":
                _ready = True
                log.info("YaneuraOu ready at %s", path)
                return _proc
            if not line and _proc.poll() is not None:
                break
        log.warning("YaneuraOu readyok timeout")
        _shutdown()
        return None
    except Exception:
        log.warning("Failed to start YaneuraOu", exc_info=True)
        _shutdown()
        return None


def _shutdown() -> None:
    global _proc, _ready
    _ready = False
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


def best_move_usi(sfen: str, movetime_ms: Optional[int] = None) -> Optional[str]:
    """
    Ask YaneuraOu for the best USI move from an SFEN position.

    Returns e.g. "7g7f" or "B*5e", or None if unavailable/failed.
    """
    if not sfen or not sfen.strip():
        return None

    movetime = int(movetime_ms if movetime_ms is not None else _DEFAULT_MOVETIME_MS)
    movetime = max(50, min(movetime, 30_000))

    with _LOCK:
        proc = _ensure_engine()
        if proc is None:
            return None
        try:
            # python-shogi sfen() is already full SFEN; USI wants "position sfen ..."
            sfen_cmd = sfen.strip()
            if not sfen_cmd.startswith("sfen"):
                sfen_cmd = f"sfen {sfen_cmd}"
            _write(proc, "usinewgame")
            _write(proc, f"position {sfen_cmd}")
            _write(proc, f"go movetime {movetime}")

            # Read until bestmove (ignore info lines)
            deadline_reads = max(50, movetime // 20 + 50)
            for _ in range(deadline_reads):
                line = _readline(proc, timeout=max(2.0, movetime / 1000.0 + 2.0))
                if not line:
                    if proc.poll() is not None:
                        log.warning("YaneuraOu died during search")
                        _shutdown()
                        return None
                    continue
                if line.startswith("bestmove"):
                    parts = line.split()
                    if len(parts) < 2:
                        return None
                    move = parts[1]
                    if move in ("resign", "win", "none"):
                        return None
                    return move
            log.warning("YaneuraOu bestmove timeout")
            return None
        except Exception:
            log.warning("YaneuraOu best_move_usi failed", exc_info=True)
            _shutdown()
            return None
