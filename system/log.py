"""
system/log.py
Central logger for Aiko-chan.

All modules import get_logger() and use it instead of print().
Output goes to logs/aiko.log only (file) by default. Set LOG_CONSOLE=1
(e.g. via a --debug flag in main.py, or `LOG_CONSOLE=1 python main.py`)
to also echo to stdout for a given run.
A second file, logs/aiko.error.log, mirrors ERROR+ records only, so
"what broke last night" is a short file instead of a grep through
megabytes of INFO chatter.

Log level is controllable via LOG_LEVEL in .env, or set programmatically
(e.g. by --debug) before the first get_logger() call. Invalid values fall
back to INFO with a one-line warning printed to stderr (the logger
doesn't exist yet at that point).

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

_VALID_LEVELS = logging.getLevelNamesMapping()  # py3.11+


def _resolve_log_level() -> str:
    raw = os.getenv("LOG_LEVEL", "INFO").upper()
    if raw not in _VALID_LEVELS:
        print(f"[log] invalid LOG_LEVEL={raw!r}, defaulting to INFO", file=sys.stderr)
        return "INFO"
    return raw


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"[log] invalid {name}={raw!r}, using default {default}", file=sys.stderr)
        return default


# NOTE: these are intentionally NOT resolved at import time. main.py may
# set LOG_LEVEL / LOG_CONSOLE from a --debug flag after other modules have
# already `import system.log` transitively — resolving here, inside
# _setup(), means whatever's in os.environ at first get_logger() call wins,
# regardless of import order.

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

        # Resolved here (not at import time) so a --debug flag setting
        # LOG_LEVEL/LOG_CONSOLE in main.py just before the first log call
        # is always honored — see note above.
        log_level = _resolve_log_level()
        log_console = os.getenv("LOG_CONSOLE", "0") == "1"
        log_max_bytes = _int_env("LOG_MAX_BYTES", 5 * 1024 * 1024)
        log_backup_count = _int_env("LOG_BACKUP_COUNT", 3)

        # Let this module's configured levels decide what gets emitted. A previous
        # process-wide disable() call can otherwise make the file logger look dead.
        logging.disable(logging.NOTSET)

        root = logging.getLogger()
        root.setLevel(log_level)

        fmt = logging.Formatter(_FORMAT, datefmt=_DATE_FMT)

        # Main file handler — rotating, never pollutes stdout.
        # delay=True: don't create/touch aiko.log until something is actually logged.
        fh = RotatingFileHandler(
            LOG_FILE,
            maxBytes=log_max_bytes,
            backupCount=log_backup_count,
            encoding="utf-8",
            delay=True,
        )
        fh.setLevel(log_level)
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

        # Console handler — opt-in via LOG_CONSOLE=1 (e.g. set by --debug in
        # main.py), off by default.
        if log_console:
            ch = logging.StreamHandler()
            ch.setLevel(log_level)
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
