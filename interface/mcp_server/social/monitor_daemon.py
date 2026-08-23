"""
interface/mcp_server/social/monitor_daemon.py

Login-independent Threads reply monitor daemon.

Starts a background daemon thread that polls monitor_threads_replies()
at a fixed interval, completely independent of the WebUI login gate.

This solves the boot-ordering problem where the Threads poller only
started after a user logged into the WebUI (because the scheduler is
seeded inside AikoWakeup().boot(), which is deferred until first login).

Usage (called once from run_webui() before run_session()):

    from interface.mcp_server.social.monitor_daemon import start_threads_monitor_daemon
    start_threads_monitor_daemon()

The daemon:
  - Only starts if THREADS_ACCESS_TOKEN and THREADS_USER_ID are set
  - Runs as a daemon thread — auto-killed when the process exits
  - Uses THREADS_REPLY_MONITOR_INTERVAL_SECONDS (default 180 s)
  - Passes memorize=None — the monitor handles this gracefully; memory
    features (pin/learn) won't work pre-login, but replies will
  - Is idempotent: calling it twice won't start two threads
  - Coexists safely with the scheduler's own threads_reply_monitor job
    (post-login); has_processed_threads_reply() in the DB prevents
    double-replies

Env vars read:
  THREADS_ACCESS_TOKEN              — required; skip if absent
  THREADS_USER_ID                   — required; skip if absent
  THREADS_REPLY_MONITOR_INTERVAL_SECONDS — default 180
  THREADS_MONITOR_DAEMON_ENABLED    — set to 0/false to disable
"""

from __future__ import annotations

import os
import threading
import time

from system.log import get_logger

log = get_logger(__name__)

_DAEMON_THREAD: threading.Thread | None = None
_STOP_EVENT = threading.Event()

_DEFAULT_INTERVAL = 180  # seconds — same default as scheduler job


def _daemon_loop(interval: int) -> None:
    """Background loop: poll Threads, sleep, repeat."""
    log.info("[threads_daemon] Monitor started (interval=%ds)", interval)
    while not _STOP_EVENT.is_set():
        try:
            from interface.mcp_server.social.services.threads import monitor_threads_replies
            result = monitor_threads_replies(memorize=None)
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
    """Start the Threads reply monitor background daemon.

    Safe to call multiple times — only one daemon thread will ever run.

    Args:
        interval_seconds: Poll interval in seconds. Defaults to
            THREADS_REPLY_MONITOR_INTERVAL_SECONDS env var (default 180).

    Returns:
        The daemon Thread if started, or None if skipped (missing env vars
        or explicitly disabled).
    """
    global _DAEMON_THREAD

    # Already running — don't start a second thread.
    if _DAEMON_THREAD is not None and _DAEMON_THREAD.is_alive():
        log.debug("[threads_daemon] Already running, skipping duplicate start")
        return _DAEMON_THREAD

    # Explicit opt-out.
    if os.getenv("THREADS_MONITOR_DAEMON_ENABLED", "1").strip().lower() in {"0", "false", "no", "off"}:
        log.info("[threads_daemon] Disabled via THREADS_MONITOR_DAEMON_ENABLED=0, skipping")
        return None

    # Required credentials check — fail fast and silently if not configured.
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
        daemon=True,  # auto-killed when the main process exits
    )
    thread.start()
    _DAEMON_THREAD = thread
    return thread


def stop_threads_monitor_daemon() -> None:
    """Signal the daemon to stop at the next sleep boundary.

    Normally not needed — daemon threads die with the process. Exposed
    for testing and graceful shutdown paths.
    """
    _STOP_EVENT.set()
