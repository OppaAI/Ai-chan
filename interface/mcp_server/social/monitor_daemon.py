"""
interface/mcp_server/social/monitor_daemon.py

Login-independent social reply monitor daemons (Threads + Bluesky).

Starts background daemon threads that poll monitor_*_replies() at a fixed
interval, completely independent of the WebUI login gate.

Usage (called once from run_webui() before run_session()):

    from interface.mcp_server.social.monitor_daemon import (
        start_threads_monitor_daemon,
        start_bluesky_monitor_daemon,
    )
    start_threads_monitor_daemon()
    start_bluesky_monitor_daemon()
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

from system.log import get_logger

log = get_logger(__name__)

_DAEMON_THREAD: threading.Thread | None = None
_STOP_EVENT = threading.Event()

_BSKY_DAEMON_THREAD: threading.Thread | None = None
_BSKY_STOP_EVENT = threading.Event()

_DEFAULT_INTERVAL = 180  # seconds — same default as scheduler job

# Late-bound memory handle. Daemons start before wakeup boots the memory
# subsystem (and before anyone logs in), so they can't receive AikoMemorize
# at construction time. run_session() injects it post-boot via
# set_shared_memorize(); each poll cycle reads whatever is current, so
# replies gain long-term recall + memory saving without a restart.
_SHARED_MEMORIZE: dict = {"ref": None}

# Dedicated fallback handle for headless periods: while nobody has logged
# into the WebUI since boot, the shared instance is still a lazy guest with
# no store open, and passing it (or None) made every Threads/Bluesky reply
# run without recall and without interaction-memory saving — even though the
# owner's memory.db sits right there on disk. The fallback binds its own
# AikoMemorize to that store once, then both daemons reuse it.
_FALLBACK_MEMORIZE: dict = {"ref": None}
_FALLBACK_LOCK = threading.Lock()


def set_shared_memorize(memorize) -> None:
    """Inject the live AikoMemorize instance into both monitor daemons."""
    _SHARED_MEMORIZE["ref"] = memorize


def _owner_user_id() -> str | None:
    """Resolve the owner's user_id without requiring a browser login.

    Order: explicit AIKO_USER_ID env override, else the unique non-guest
    user directory under USER_SPACE_ROOT that has both a memory store and a
    profile. Ambiguous multi-user layouts return None (recall stays off
    rather than guessing and reading/writing the wrong person's memories).
    """
    override = os.getenv("AIKO_USER_ID", "").strip()
    if override:
        return override
    try:
        from system.userspace import _user_state_root_value

        root = Path(_user_state_root_value()).expanduser()
        candidates = []
        for child in sorted(root.iterdir()):
            if not child.is_dir() or child.name == "guest":
                continue
            if (child / "memory" / "memory.db").is_file() and (
                child / "profile" / "USER.md"
            ).is_file():
                candidates.append(child.name)
    except Exception:
        return None
    return candidates[0] if len(candidates) == 1 else None


def _fallback_owner_memorize():
    """Return a monitor-owned AikoMemorize bound to the owner's store.

    Built lazily on first headless poll and cached; returns None when no
    unambiguous owner identity can be resolved on disk.
    """
    with _FALLBACK_LOCK:
        mem = _FALLBACK_MEMORIZE["ref"]
        if mem is not None:
            return mem
        uid = _owner_user_id()
        if not uid:
            return None
        try:
            from cognition.memory.memorize import AikoMemorize

            mem = AikoMemorize(silent=True)
            mem.switch_user(uid)
        except Exception:
            log.exception("[monitor_daemon] Owner-store fallback bind failed for %s", uid)
            return None
        _FALLBACK_MEMORIZE["ref"] = mem
        log.info("[monitor_daemon] Headless recall bound to owner store %s", uid)
        return mem


def _bound_memorize():
    """Return the best available AikoMemorize for social reply monitors.

    Preference order: the shared live instance once wakeup boot + login have
    bound a REAL per-user store (is_open()), else the dedicated owner-store
    fallback for headless operation. Before either exists, monitors run
    recall-free like before.
    """
    mem = _SHARED_MEMORIZE["ref"]
    if mem is not None and mem.is_open():
        return mem
    return _fallback_owner_memorize()


def _daemon_loop(interval: int) -> None:
    """Background loop: poll Threads, sleep, repeat."""
    log.info("[threads_daemon] Monitor started (interval=%ds)", interval)
    while not _STOP_EVENT.is_set():
        try:
            from interface.mcp_server.social.services.threads import monitor_threads_replies
            result = monitor_threads_replies(memorize=_bound_memorize())
            answered = result.get("answered", 0)
            matched = result.get("matched", 0)
            errors = result.get("errors", [])
            if answered:
                log.info("[threads_daemon] Answered %d / %d matched replies", answered, matched)
            if errors:
                log.warning("[threads_daemon] %d error(s): %s", len(errors), errors[:3])
        except Exception:
            log.exception("[threads_daemon] Unexpected error in monitor loop")
        _STOP_EVENT.wait(timeout=interval)
    log.info("[threads_daemon] Monitor stopped")


def start_threads_monitor_daemon(interval_seconds: int | None = None) -> threading.Thread | None:
    """Start the Threads reply monitor background daemon."""
    global _DAEMON_THREAD

    if _DAEMON_THREAD is not None and _DAEMON_THREAD.is_alive():
        log.debug("[threads_daemon] Already running, skipping duplicate start")
        return _DAEMON_THREAD

    if os.getenv("THREADS_MONITOR_DAEMON_ENABLED", "1").strip().lower() in {"0", "false", "no", "off"}:
        log.info("[threads_daemon] Disabled via THREADS_MONITOR_DAEMON_ENABLED=0, skipping")
        return None

    if not os.getenv("THREADS_ACCESS_TOKEN") or not os.getenv("THREADS_USER_ID"):
        log.info(
            "[threads_daemon] THREADS_ACCESS_TOKEN or THREADS_USER_ID not set — "
            "Threads monitor daemon will not start"
        )
        return None

    interval = interval_seconds or max(
        60,
        int(os.getenv("THREADS_REPLY_MONITOR_INTERVAL_SECONDS", str(_DEFAULT_INTERVAL))),
    )

    _STOP_EVENT.clear()
    thread = threading.Thread(
        target=_daemon_loop,
        args=(interval,),
        name="threads-reply-monitor-daemon",
        daemon=True,
    )
    thread.start()
    _DAEMON_THREAD = thread
    return thread


def stop_threads_monitor_daemon() -> None:
    """Signal the Threads daemon to stop at the next sleep boundary."""
    _STOP_EVENT.set()


def _bluesky_daemon_loop(interval: int) -> None:
    log.info("[bluesky_daemon] Monitor started (interval=%ds)", interval)
    while not _BSKY_STOP_EVENT.is_set():
        try:
            from interface.mcp_server.social.services.bluesky import monitor_bluesky_replies

            result = monitor_bluesky_replies(memorize=_bound_memorize())
            answered = result.get("answered", 0)
            matched = result.get("matched", 0)
            errors = result.get("errors", [])
            if answered:
                log.info("[bluesky_daemon] Answered %d / %d matched replies", answered, matched)
            if errors:
                log.warning("[bluesky_daemon] %d error(s): %s", len(errors), errors[:3])
        except Exception:
            log.exception("[bluesky_daemon] Unexpected error in monitor loop")
        _BSKY_STOP_EVENT.wait(timeout=interval)
    log.info("[bluesky_daemon] Monitor stopped")


def start_bluesky_monitor_daemon(interval_seconds: int | None = None) -> threading.Thread | None:
    """Start the Bluesky reply monitor background daemon."""
    global _BSKY_DAEMON_THREAD

    if _BSKY_DAEMON_THREAD is not None and _BSKY_DAEMON_THREAD.is_alive():
        log.debug("[bluesky_daemon] Already running, skipping duplicate start")
        return _BSKY_DAEMON_THREAD

    if os.getenv("BLUESKY_MONITOR_DAEMON_ENABLED", "1").strip().lower() in {"0", "false", "no", "off"}:
        log.info("[bluesky_daemon] Disabled via BLUESKY_MONITOR_DAEMON_ENABLED=0, skipping")
        return None

    if not os.getenv("BLUESKY_HANDLE") or not os.getenv("BLUESKY_APP_PASS"):
        log.info(
            "[bluesky_daemon] BLUESKY_HANDLE or BLUESKY_APP_PASS not set — "
            "Bluesky monitor daemon will not start"
        )
        return None

    try:
        raw = os.getenv("BLUESKY_REPLY_MONITOR_INTERVAL_SECONDS", str(_DEFAULT_INTERVAL))
        parsed = int(raw)
        interval = interval_seconds or max(60, parsed)
    except ValueError:
        log.warning(
            "[bluesky_daemon] Invalid BLUESKY_REPLY_MONITOR_INTERVAL_SECONDS=%r, using default %ds",
            os.getenv("BLUESKY_REPLY_MONITOR_INTERVAL_SECONDS"),
            _DEFAULT_INTERVAL,
        )
        interval = interval_seconds or _DEFAULT_INTERVAL

    _BSKY_STOP_EVENT.clear()
    thread = threading.Thread(
        target=_bluesky_daemon_loop,
        args=(interval,),
        name="bluesky-reply-monitor-daemon",
        daemon=True,
    )
    thread.start()
    _BSKY_DAEMON_THREAD = thread
    return thread


def stop_bluesky_monitor_daemon() -> None:
    """Signal the Bluesky daemon to stop at the next sleep boundary."""
    _BSKY_STOP_EVENT.set()
