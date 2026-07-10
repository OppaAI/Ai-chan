"""
core/log.py
Central logger for Aiko-chan.

All modules import get_logger() and use it instead of print().
Output goes to logs/aiko.log (file) and stdout (console) simultaneously,
with log level controllable via LOG_LEVEL in .env.

Usage:
    from core.log import get_logger
    log = get_logger(__name__)
    log.info("Ready.")
    log.warning("Something looks off.")
    log.error("Something broke.")
"""
import logging
import os
from logging.handlers import RotatingFileHandler

# ── config ────────────────────────────────────────────────────────────────────
LOG_DIR   = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
LOG_FILE  = os.path.join(LOG_DIR, "aiko.log")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Rotate at 5MB, keep 3 backups → aiko.log, aiko.log.1, aiko.log.2
LOG_MAX_BYTES    = int(os.getenv("LOG_MAX_BYTES",    5 * 1024 * 1024))
LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", 3))

from contextlib import contextmanager

_FORMAT     = "%(asctime)s  [%(levelname)-8s]  %(name)s — %(message)s"
_DATE_FMT   = "%Y-%m-%d %H:%M:%S"
_initialized = False

# ── setup ─────────────────────────────────────────────────────────────────────

def _setup() -> None:
    """Configure root logger once. Subsequent calls are no-ops."""
    global _initialized
    if _initialized:
        return

    os.makedirs(LOG_DIR, exist_ok=True)

    # Let this module's configured levels decide what gets emitted. A previous
    # process-wide disable() call can otherwise make the file logger look dead.
    logging.disable(logging.NOTSET)

    root = logging.getLogger()
    root.setLevel(LOG_LEVEL)

    fmt = logging.Formatter(_FORMAT, datefmt=_DATE_FMT)

    # File handler — rotating, never pollutes stdout
    fh = RotatingFileHandler(
        LOG_FILE,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    fh.setLevel(LOG_LEVEL)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    # Console handler removed per user request
    # ch = logging.StreamHandler()
    # ch.setLevel(logging.INFO)
    # ch.setFormatter(fmt)
    # root.addHandler(ch)

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