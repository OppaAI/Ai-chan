"""
system/log.py

Central logger for Aiko-chan.

Usage:
    from system.log import get_logger
    log = get_logger(__name__)
    log.info("Ready.")
    log.warning("Something looks off.")
    log.error("Something broke.")

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

── Rotation & pruning ──────────────────────────────────────────────────────
Controlled by LOG_ROTATE_MODE, either "size" (default) or "time":

  size mode (default):
    - LOG_MAX_BYTES     bytes before rotating aiko.log      (default 5 MiB)
    - LOG_BACKUP_COUNT  how many rotated backups to KEEP    (default 3)
  Rotation happens the moment a write would push aiko.log past
  LOG_MAX_BYTES. The current file is renamed aiko.log.1 (existing .1
  becomes .2, etc). Once more than LOG_BACKUP_COUNT backups exist, the
  oldest is deleted automatically on the next rotation — nothing else
  to run, no separate prune step needed.

  time mode:
    - LOG_ROTATE_WHEN      rotation cadence, e.g. "midnight", "D", "H"
                           (default "midnight" — once per day)
    - LOG_ROTATE_INTERVAL  how many of the above units between rotations
                           (default 1 -> every day)
    - LOG_BACKUP_COUNT     how many rotated backups to KEEP (default 3)
  Same pruning behavior as size mode, just triggered by wall-clock time
  instead of file size: aiko.log.YYYY-MM-DD accumulates one entry per
  day, and anything older than LOG_BACKUP_COUNT days is deleted the
  next time a rotation fires. Rotation is checked on the first log call
  after the interval has passed, not on a background timer — if the
  boundary is crossed while the process is not logging, rotation occurs
  on the next write (whether the process remained running or restarted).

  The error-only tail file (aiko.error.log) always rotates by size
  (1 MiB, 2 backups) regardless of LOG_ROTATE_MODE, since it's meant to
  stay small and fast to scan, not to track calendar history.

This module only resolves LOG_LEVEL / LOG_CONSOLE at first get_logger()
call (not at import time), so main.py's --debug can flip the environment
before the root logger is configured — see main.py's module docstring for
why import-time resolution would be too late.

Flow:

                                      get_logger(__name__)
                                              │
                                              ▼
                                         _setup()  (once, thread-safe)
                                              │
                 ┌────────────────┼────────────────┼─────────────────┐
                 ▼                ▼                ▼                 ▼
           RotatingFileHandler  RotatingFileHandler  StreamHandler  (LOG_CONSOLE=1)
            aiko.log (size/time)  aiko.error.log     stdout
                 │                │                  │
                 ▼                ▼                  ▼
            delay=True until first log; rotation per LOG_ROTATE_MODE
"""
from __future__ import annotations            # evaluates type annotations later

# Public libraries
import logging                                # for root logger configuration
import os                                     # for reading LOG_* environment variables
import sys                                    # for stderr fallback before logger exists
import threading                              # for thread-safe one-time setup
from contextlib import contextmanager         # for silent_stderr() helper
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler  # for file rotation
from pathlib import Path                      # for log directory resolution (py312 modern)

# ── config ────────────────────────────────────────────────────────────────────
LOG_DIR        = Path(__file__).resolve().parents[1] / "logs"  # logs/ at repo root
LOG_FILE       = LOG_DIR / "aiko.log"        # main rotating log file
ERROR_LOG_FILE = LOG_DIR / "aiko.error.log"  # error-only tail file

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


def _make_main_handler(log_level: str, log_max_bytes: int, log_backup_count: int) -> RotatingFileHandler | TimedRotatingFileHandler:
    """Build the main rotating file handler per LOG_ROTATE_MODE (size|time)."""
    mode = os.getenv("LOG_ROTATE_MODE", "size").strip().lower()

    if mode == "time":
        when = os.getenv("LOG_ROTATE_WHEN", "midnight")
        _valid_when = {"S", "M", "H", "D", "MIDNIGHT",
                       "W0", "W1", "W2", "W3", "W4", "W5", "W6"}
        if when.upper() not in _valid_when:
            print(f"[log] invalid LOG_ROTATE_WHEN={when!r}, defaulting to 'midnight'", file=sys.stderr)
            when = "midnight"
        interval = _int_env("LOG_ROTATE_INTERVAL", 1)
        handler: RotatingFileHandler | TimedRotatingFileHandler = TimedRotatingFileHandler(
            LOG_FILE,
            when=when,
            interval=interval,
            backupCount=log_backup_count,
            encoding="utf-8",
            delay=True,
        )
    else:
        if mode != "size":
            print(f"[log] invalid LOG_ROTATE_MODE={mode!r}, defaulting to 'size'", file=sys.stderr)
        handler = RotatingFileHandler(
            LOG_FILE,
            maxBytes=log_max_bytes,
            backupCount=log_backup_count,
            encoding="utf-8",
            delay=True,
        )

    handler.setLevel(log_level)
    return handler


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
        logging.disable(logging.NOTSET)  # resets prior disable() — intentional per module note

        root = logging.getLogger()
        root.setLevel(log_level)

        fmt = logging.Formatter(_FORMAT, datefmt=_DATE_FMT)

        # Main file handler — rotating (size or time, see module docstring),
        # never pollutes stdout. delay=True: don't create/touch aiko.log
        # until something is actually logged.
        fh = _make_main_handler(log_level, log_max_bytes, log_backup_count)
        fh.setFormatter(fmt)
        root.addHandler(fh)

        # Error-only tail file — small, fast to scan after a bad session.
        # Always size-based regardless of LOG_ROTATE_MODE (see docstring).
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
    """Redirect fd 2 to /dev/null — silences C-library noise (ALSA, ONNX, PyAudio).

    NOT thread-safe: globally mutes fd 2 for all threads while active.
    Use only outside request threads (e.g. during boot model loads), not
    per-turn inside WebUI handlers — otherwise concurrent callers stomp
    each other's real_stderr_fd. For per-call suppression prefer
    contextlib.redirect_stderr() which is thread-local.
    """
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        real_stderr_fd = os.dup(2)
    except OSError:
        os.close(devnull_fd)
        raise
    try:
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(real_stderr_fd, 2)
        os.close(real_stderr_fd)
        os.close(devnull_fd)
