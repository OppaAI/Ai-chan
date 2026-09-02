"""
interface/mcp_server/social/monitor_daemon.py

Login-independent social reply monitor daemons (Threads + Bluesky + Mastodon).

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

_SOCIAL_THREAD: threading.Thread | None = None
_SOCIAL_STOP_EVENT = threading.Event()
_DESIRED: dict[str, dict] = {}   # label -> {"interval": s, "fn": callable}

_SHARED_MEMORIZE: dict = {"ref": None}
_FALLBACK_MEMORIZE: dict = {"ref": None}
_FALLBACK_LOCK = threading.Lock()


def set_shared_memorize(memorize) -> None:
    """Inject the live AikoMemorize instance into both monitor daemons."""
    _SHARED_MEMORIZE["ref"] = memorize


def _owner_user_id() -> str | None:
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


def _resolve_owner_display_name(uid: str) -> str | None:
    try:
        import json as _json

        token_path = Path.home() / ".aiko" / "auth_token.json"
        if token_path.is_file():
            data = _json.loads(token_path.read_text(encoding="utf-8"))
            user = data.get("user") or {}
            try:
                from system.userspace import normalize_user_id

                expected = normalize_user_id("github", user.get("id")) if user.get("id") else None
                if expected == uid or uid is None:
                    login = (user.get("login") or "").strip()
                    if login:
                        return login
                login = (user.get("login") or "").strip()
                if login and uid == _owner_user_id():
                    return login
            except Exception:
                login = (user.get("login") or "").strip()
                if login:
                    return login
    except Exception:
        pass
    try:
        from interface.mcp_server.social.services.identity import owner_display_name

        name = owner_display_name().strip()
        if name and name.casefold() not in {"owner", "guest"}:
            return name
    except Exception:
        pass
    if uid in {"github_205369547", "OppaAI"}:
        return "OppaAI"
    return None


def _fallback_owner_memorize():
    with _FALLBACK_LOCK:
        mem = _FALLBACK_MEMORIZE["ref"]
        if mem is not None:
            try:
                if mem.get_display_name() in {"github_205369547", "oppa.ai.bot", "oppa.ai"}:
                    disp = _resolve_owner_display_name(mem.get_user_id())
                    if disp:
                        mem.set_display_name(disp)
            except Exception:
                pass
            return mem
        uid = _owner_user_id()
        if not uid:
            return None
        try:
            from cognition.memory.memorize import AikoMemorize

            mem = AikoMemorize(silent=True)
            mem.switch_user(uid)
            disp = _resolve_owner_display_name(uid)
            if disp:
                try:
                    mem.set_display_name(disp)
                except Exception:
                    pass
        except Exception:
            log.exception("[monitor_daemon] Owner-store fallback bind failed for %s", uid)
            return None
        _FALLBACK_MEMORIZE["ref"] = mem
        log.info("[monitor_daemon] Headless recall bound to owner store %s display=%s", uid, mem.get_display_name())
        return mem


def _bound_memorize():
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


def _poll_mastodon() -> None:
    from interface.mcp_server.social.services.mastodon import monitor_mastodon_replies

    result = monitor_mastodon_replies(memorize=_bound_memorize())
    answered = result.get("answered", 0)
    matched = result.get("matched", 0)
    errors = result.get("errors", [])
    if answered:
        log.info("[mastodon_daemon] Answered %d / %d matched replies", answered, matched)
    if errors:
        log.warning("[mastodon_daemon] %d error(s): %s", len(errors), errors[:3])


def _social_loop() -> None:
    def _labels():
        return "/".join(_DESIRED.keys()) or "<none>"

    log.info("[social_daemon] Unified monitor started (platforms=%s)", _labels())
    next_due: dict[str, float] = {}
    while not _SOCIAL_STOP_EVENT.is_set():
        now = time.monotonic()
        for label, cfg in list(_DESIRED.items()):
            if label not in next_due:
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

        deadline = time.monotonic() + pause
        while not _SOCIAL_STOP_EVENT.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0 or _SOCIAL_STOP_EVENT.wait(timeout=min(1.0, max(0.0, remaining))):
                break
    log.info("[social_daemon] Monitor stopped (%s)", _labels())


def _register_platform(label: str, interval_seconds: int | None, fn) -> bool:
    iv = int(interval_seconds) if interval_seconds else _DEFAULT_INTERVAL
    job = {"interval": iv, "fn": fn}
    if label in _DESIRED:
        return False
    _DESIRED[label] = job
    return True


def start_social_monitor_daemon(interval_seconds: int | None = None, only: str | None = None) -> threading.Thread | None:
    def _interval(env_key: str, label: str, default: int | None = None) -> int:
        floor = 30
        base = default if default is not None else _DEFAULT_INTERVAL
        try:
            return max(floor, int(os.getenv(env_key, str(base))))
        except ValueError:
            log.warning(
                "[%s_daemon] Invalid %s=%r, using default %ds",
                label, env_key, os.getenv(env_key), base,
            )
            return base

    explicit = int(interval_seconds) if interval_seconds else None

    candidates: list[tuple[str, int, object]] = []

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

    if only in (None, "mastodon"):
        if os.getenv("MASTODON_MONITOR_DAEMON_ENABLED", "1").strip().lower() in {"0", "false", "no", "off"}:
            log.info("[mastodon_daemon] Disabled via MASTODON_MONITOR_DAEMON_ENABLED=0, skipping")
        elif not os.getenv("MASTODON_ACCESS_TOKEN") or not os.getenv("MASTODON_INSTANCE"):
            log.info(
                "[mastodon_daemon] MASTODON_ACCESS_TOKEN or MASTODON_INSTANCE not set — "
                "Mastodon monitor daemon will not start"
            )
        else:
            candidates.append(("mastodon", explicit or _interval("MASTODON_REPLY_MONITOR_INTERVAL_SECONDS", "mastodon"), _poll_mastodon))

    if not candidates:
        return None

    newly = False
    for label, iv, fn in candidates:
        if _register_platform(label, iv, fn):
            newly = True

    global _SOCIAL_THREAD
    if _SOCIAL_THREAD is not None and _SOCIAL_THREAD.is_alive():
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
    _DESIRED.clear()
    stop_event_was_set = _SOCIAL_STOP_EVENT.is_set()
    _SOCIAL_STOP_EVENT.set()
    if not stop_event_was_set:
        pass


def start_threads_monitor_daemon(interval_seconds: int | None = None) -> threading.Thread | None:
    return start_social_monitor_daemon(interval_seconds=interval_seconds, only="threads")


def stop_threads_monitor_daemon() -> None:
    stop_social_monitor_daemon()


def start_bluesky_monitor_daemon(interval_seconds: int | None = None) -> threading.Thread | None:
    return start_social_monitor_daemon(interval_seconds=interval_seconds, only="bluesky")


def stop_bluesky_monitor_daemon() -> None:
    stop_social_monitor_daemon()


def start_mastodon_monitor_daemon(interval_seconds: int | None = None) -> threading.Thread | None:
    return start_social_monitor_daemon(interval_seconds=interval_seconds, only="mastodon")


def stop_mastodon_monitor_daemon() -> None:
    stop_social_monitor_daemon()
