"""
system/log.py
Central logger for Aiko-chan.

All modules import get_logger() and use it instead of print().
Output goes to logs/aiko.log only (file). Console output was deliberately
removed — see LOG_CONSOLE below if you want it back for a given run.
A second file, logs/aiko.error.log, mirrors ERROR+ records only, so
"what broke last night" is a short file instead of a grep through
megabytes of INFO chatter.

Log level is controllable via LOG_LEVEL in .env. Invalid values fall back
to INFO with a one-line warning printed to stderr (the logger doesn't
exist yet at that point).

Usage:
    from system.log import get_logger
    log = get_logger(__name__)
    log.info("Ready.")
    log.warning("Something looks off.")
    log.error("Something broke.")
"""
from __future__ import annotations

import logging
import os
import threading
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler

# ── config ────────────────────────────────────────────────────────────────────
LOG_DIR        = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
LOG_FILE       = os.path.join(LOG_DIR, "aiko.log")
ERROR_LOG_FILE = os.path.join(LOG_DIR, "aiko.error.log")

# Optional: set LOG_CONSOLE=1 to also echo to stdout for a given run
# (e.g. `LOG_CONSOLE=1 python main.py`) without touching the code.
LOG_CONSOLE = os.getenv("LOG_CONSOLE", "0") == "1"

_VALID_LEVELS = logging.getLevelNamesMapping()  # py3.11+


def _resolve_log_level() -> str:
    raw = os.getenv("LOG_LEVEL", "INFO").upper()
    if raw not in _VALID_LEVELS:
        print(f"[log] invalid LOG_LEVEL={raw!r}, defaulting to INFO")
        return "INFO"
    return raw


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"[log] invalid {name}={raw!r}, using default {default}")
        return default


LOG_LEVEL = _resolve_log_level()

# Rotate at 5MB, keep 3 backups → aiko.log, aiko.log.1, aiko.log.2
LOG_MAX_BYTES    = _int_env("LOG_MAX_BYTES",    5 * 1024 * 1024)
LOG_BACKUP_COUNT = _int_env("LOG_BACKUP_COUNT", 3)

_FORMAT      = "%(asctime)s.%(msecs)03d  [%(levelname)-8s]  %(name)s — %(message)s"
_DATE_FMT    = "%Y-%m-%d %H:%M:%S"
_initialized = False
_init_lock   = threading.Lock()

# ── setup ─────────────────────────────────────────────────────────────────────

def _setup() -> None:
    """Configure root logger once. Subsequent calls are no-ops.

    Thread-safe: guarded by _init_lock so concurrent get_logger() calls
    (e.g. from multiple WebUI connections at startup) can't both pass the
    _initialized check and register duplicate handlers.
    """
    global _initialized
    if _initialized:
        return

    with _init_lock:
        if _initialized:  # re-check inside the lock (another thread may have won the race)
            return

        os.makedirs(LOG_DIR, exist_ok=True)

        # Let this module's configured levels decide what gets emitted. A previous
        # process-wide disable() call can otherwise make the file logger look dead.
        logging.disable(logging.NOTSET)

        root = logging.getLogger()
        root.setLevel(LOG_LEVEL)

        fmt = logging.Formatter(_FORMAT, datefmt=_DATE_FMT)

        # Main file handler — rotating, never pollutes stdout.
        # delay=True: don't create/touch aiko.log until something is actually logged.
        fh = RotatingFileHandler(
            LOG_FILE,
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT,
            encoding="utf-8",
            delay=True,
        )
        fh.setLevel(LOG_LEVEL)
        fh.setFormatter(fmt)
        root.addHandler(fh)

        # Error-only tail file — small, fast to scan after a bad session.
        err_fh = RotatingFileHandler(
            ERROR_LOG_FILE,
            maxBytes=1 * 1024 * 1024,
            backupCount=2,
            encoding="utf-8",
            delay=True,
        )
        err_fh.setLevel(logging.ERROR)
        err_fh.setFormatter(fmt)
        root.addHandler(err_fh)

        # Console handler — opt-in via LOG_CONSOLE=1, off by default.
        if LOG_CONSOLE:
            ch = logging.StreamHandler()
            ch.setLevel(LOG_LEVEL)
            ch.setFormatter(fmt)
            root.addHandler(ch)

        _initialized = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger. Initialises root logger on first call."""
    _setup()
    return logging.getLogger(name)


@contextmanager
def silent_stderr():
    """Redirect fd 2 to /dev/null — silences C-library noise (ALSA, ONNX, PyAudio)."""
    devnull_fd      = os.open(os.devnull, os.O_WRONLY)
    real_stderr_fd  = os.dup(2)
    try:
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(real_stderr_fd, 2)
        os.close(real_stderr_fd)
        os.close(devnull_fd)
