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
import time
from pathlib import Path

from system.log import get_logger

log = get_logger(__name__)

_DEFAULT_INTERVAL = 180  # seconds — same default as scheduler job

# ── unified poller ────────────────────────────────────────────────────────────
# One background thread polls BOTH platforms sequentially, each on its own
# interval. Sequential matters: a reply turn drives LLM inference (+TTS sink),
# and two platforms answering simultaneously would contend for those. RAM cost
# of the second thread was never the issue — the concurrent heavy work was.
_SOCIAL_THREAD: threading.Thread | None = None
_SOCIAL_STOP_EVENT = threading.Event()
_DESIRED: dict[str, dict] = {}   # label -> {"interval": s, "fn": callable}

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


def _poll_threads() -> None:
    from interface.mcp_server.social.services.threads import monitor_threads_replies
    result = monitor_threads_replies(memorize=_bound_memorize())
    answered = result.get("answered", 0)
    matched = result.get("matched", 0)
    errors = result.get("errors", [])
    if answered:
        log.info("[threads_daemon] Answered %d / %d matched replies", answered, matched)
    if errors:
        log.warning("[threads_daemon] %d error(s): %s", len(errors), errors[:3])


def _poll_bluesky() -> None:
    from interface.mcp_server.social.services.bluesky import monitor_bluesky_replies

    result = monitor_bluesky_replies(memorize=_bound_memorize())
    answered = result.get("answered", 0)
    matched = result.get("matched", 0)
    errors = result.get("errors", [])
    if answered:
        log.info("[bluesky_daemon] Answered %d / %d matched replies", answered, matched)
    if errors:
        log.warning("[bluesky_daemon] %d error(s): %s", len(errors), errors[:3])


def _social_loop() -> None:
    """Single-thread poller: each registered platform fires on its own due time.

    Platforms can be added while running (legacy start_* wrappers merge into
    the live thread via _DESIRED). A slow poll (e.g. Threads
    Graph API with a 120s timeout) delays the other platform's next cycle
    rather than overlapping heavy reply work — deliberate trade-off.
    """
    def _labels():
        return "/".join(_DESIRED.keys()) or "<none>"

    log.info("[social_daemon] Unified monitor started (platforms=%s)", _labels())
    next_due: dict[str, float] = {}
    while not _SOCIAL_STOP_EVENT.is_set():
        now = time.monotonic()
        for label, cfg in list(_DESIRED.items()):
            if label not in next_due:
                # Stagger new entrants 5s apart so platforms never first-fire together.
                next_due[label] = now + 5.0 * len(next_due)
                log.info("[%s_daemon] Monitor started (first poll in %.0fs)", label, max(0.0, next_due[label] - now))
            if now < next_due[label]:
                continue
            try:
                cfg["fn"]()
            except Exception:
                log.exception("[%s_daemon] Unexpected error in monitor loop", label)
            next_due[label] = time.monotonic() + cfg["interval"]
        pause = max(0.5, min(next_due.values()) - time.monotonic()) if next_due else 1.0

        # Chunked sleep: re-check _DESIRED at most 1s after a new platform
        # registers, without needing a separate wake-event mechanism.
        deadline = time.monotonic() + pause
        while not _SOCIAL_STOP_EVENT.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0 or _SOCIAL_STOP_EVENT.wait(timeout=min(1.0, max(0.0, remaining))):
                break
    log.info("[social_daemon] Monitor stopped (%s)", _labels())


def _register_platform(label: str, interval_seconds: int | None, fn) -> bool:
    """Add a platform to the desired set; True when it was newly added.

    Explicit interval_seconds is honored as-is (old daemons allowed sub-60s
    values); only the env-derived default carries the 60s floor.
    """
    iv = int(interval_seconds) if interval_seconds else _DEFAULT_INTERVAL
    job = {"interval": iv, "fn": fn}
    if label in _DESIRED:
        return False
    _DESIRED[label] = job
    return True


def start_social_monitor_daemon(interval_seconds: int | None = None, only: str | None = None) -> threading.Thread | None:
    """Start (or merge into) the unified social reply poller.

    Safe to call repeatedly and per-platform: legacy start_threads/_bluesky
    wrappers both land here; a second call registers its platform into the
    live thread instead of skipping. Env enable-checks and credential gates
    behave exactly like the old separate daemons.
    """
    def _interval(env_key: str, label: str) -> int:
        try:
            return max(30, int(os.getenv(env_key, str(_DEFAULT_INTERVAL))))
        except ValueError:
            log.warning(
                "[%s_daemon] Invalid %s=%r, using default %ds",
                label, env_key, os.getenv(env_key), _DEFAULT_INTERVAL,
            )
            return _DEFAULT_INTERVAL

    # Explicit interval_seconds (legacy wrappers/tests) wins over env default.
    # Honored as-is (old daemons allowed sub-60s values); only the env-derived
    # default carries the 60s floor.
    explicit = int(interval_seconds) if interval_seconds else None

    candidates: list[tuple[str, int, object]] = []  # (label, interval, poll_fn)

    if only in (None, "threads"):
        if os.getenv("THREADS_MONITOR_DAEMON_ENABLED", "1").strip().lower() in {"0", "false", "no", "off"}:
            log.info("[threads_daemon] Disabled via THREADS_MONITOR_DAEMON_ENABLED=0, skipping")
        elif not os.getenv("THREADS_ACCESS_TOKEN") or not os.getenv("THREADS_USER_ID"):
            log.info(
                "[threads_daemon] THREADS_ACCESS_TOKEN or THREADS_USER_ID not set — "
                "Threads monitor daemon will not start"
            )
        else:
            candidates.append(("threads", explicit or _interval("THREADS_REPLY_MONITOR_INTERVAL_SECONDS", "threads"), _poll_threads))

    if only in (None, "bluesky"):
        if os.getenv("BLUESKY_MONITOR_DAEMON_ENABLED", "1").strip().lower() in {"0", "false", "no", "off"}:
            log.info("[bluesky_daemon] Disabled via BLUESKY_MONITOR_DAEMON_ENABLED=0, skipping")
        elif not os.getenv("BLUESKY_HANDLE") or not os.getenv("BLUESKY_APP_PASS"):
            log.info(
                "[bluesky_daemon] BLUESKY_HANDLE or BLUESKY_APP_PASS not set — "
                "Bluesky monitor daemon will not start"
            )
        else:
            candidates.append(("bluesky", explicit or _interval("BLUESKY_REPLY_MONITOR_INTERVAL_SECONDS", "bluesky"), _poll_bluesky))

    if not candidates:
        return None

    newly = False
    for label, iv, fn in candidates:
        if _register_platform(label, iv, fn):
            newly = True

    global _SOCIAL_THREAD
    if _SOCIAL_THREAD is not None and _SOCIAL_THREAD.is_alive():
        # newly-registered platform is picked up within ≤1s by the chunked
        # sleep in _social_loop — nothing to signal.
        return _SOCIAL_THREAD

    _SOCIAL_STOP_EVENT.clear()
    thread = threading.Thread(
        target=_social_loop,
        name="social-reply-monitor-daemon",
        daemon=True,
    )
    thread.start()
    _SOCIAL_THREAD = thread
    return thread


def stop_social_monitor_daemon() -> None:
    """Signal the unified poller to stop and forget registered platforms."""
    _DESIRED.clear()
    stop_event_was_set = _SOCIAL_STOP_EVENT.is_set()
    _SOCIAL_STOP_EVENT.set()
    if not stop_event_was_set:
        pass  # loop exits on next wake


# ── legacy single-platform wrappers (kept so existing callers/tests keep working) ──

def start_threads_monitor_daemon(interval_seconds: int | None = None) -> threading.Thread | None:
    """Deprecated: prefer start_social_monitor_daemon()."""
    return start_social_monitor_daemon(interval_seconds=interval_seconds, only="threads")


def stop_threads_monitor_daemon() -> None:
    """Deprecated: prefer stop_social_monitor_daemon()."""
    stop_social_monitor_daemon()


def start_bluesky_monitor_daemon(interval_seconds: int | None = None) -> threading.Thread | None:
    """Deprecated: prefer start_social_monitor_daemon()."""
    return start_social_monitor_daemon(interval_seconds=interval_seconds, only="bluesky")


def stop_bluesky_monitor_daemon() -> None:
    """Deprecated: prefer stop_social_monitor_daemon()."""
    stop_social_monitor_daemon()
