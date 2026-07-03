"""
main.py

Aiko-chan CLI — entry point and session orchestrator.

Usage:
    python main.py               # full voice — ASR (SenseVoice + Silero VAD) + TTS (MioTTS)
    python main.py --text        # keyboard input + TTS/ASR loaded but toggled off
    python main.py --no-asr      # keyboard input + TTS on, ASR loaded but toggled off
    python main.py               # browser WebUI (default)
    python main.py --tui         # curses TUI
    python main.py --debug       # show memory debug info each turn
    python main.py --clear-mem   # wipe all stored memories and exit

Responsibilities:
    - Parse CLI arguments
    - Delegate subsystem boot to core/wakeup.py
    - Drive the UI init phase and transition to active chat
    - Run the main input → inference → render loop
    - Handle commands (/quit, /reset, /memory, /clear, /remember, /think,
                       /voice, /listen, /web, /help)
    - Fuzzy-match spoken voice commands to slash equivalents in ASR mode
    - Clean shutdown on Ctrl-C / Ctrl-D
"""
from core.config import load_config
load_config()

import argparse
import collections
from datetime import datetime, timedelta
import difflib
import os
import random
import re
import sys
import threading
import time
import warnings
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

warnings.filterwarnings("ignore")

import curses
import logging

from core.silence import silent_stderr
from core.log     import get_logger
from core.wakeup  import AikoWakeup
from core.toolkit.web import web_search

log = get_logger(__name__)

with silent_stderr():
    from core.memorize import AikoMemorize

from tui.tui import AikoTUI
from webui.webui import AikoWeb, HTTP_PORT, WEBUI_HTTPS

# ── env ───────────────────────────────────────────────────────────────────────

AI_NAME = os.getenv("AI_NAME", "Aiko")
USER_ID = os.getenv("USER_ID", "")
STREAM_DRAW_INTERVAL = float(os.getenv("STREAM_DRAW_INTERVAL", "0.05"))
LATENCY_LOG_ENABLED = os.getenv("LATENCY_LOG", "1").lower() in {"1", "true", "yes", "on"}
PROACTIVE_ENABLED = os.getenv("PROACTIVE_ENABLED", "1").lower() in {"1", "true", "yes", "on"}
PROACTIVE_FIRST_IDLE_MIN_SECONDS = float(os.getenv("PROACTIVE_FIRST_IDLE_MIN_SECONDS", os.getenv("PROACTIVE_IDLE_SECONDS", "300")))
PROACTIVE_FIRST_IDLE_MAX_SECONDS = float(os.getenv("PROACTIVE_FIRST_IDLE_MAX_SECONDS", "900"))
PROACTIVE_COOLDOWN_SECONDS = float(os.getenv("PROACTIVE_COOLDOWN_SECONDS", "1800"))
PROACTIVE_MAX_PER_HOUR = int(os.getenv("PROACTIVE_MAX_PER_HOUR", "2"))
PROACTIVE_REST_AFTER_SECONDS = float(os.getenv("PROACTIVE_REST_AFTER_SECONDS", "3600"))
PROACTIVE_USE_LLM = os.getenv("PROACTIVE_USE_LLM", "1").lower() in {"1", "true", "yes", "on"}
PROACTIVE_TIMEZONE = os.getenv("PROACTIVE_TIMEZONE", "").strip() or os.getenv("TIMEZONE", "").strip()
PROACTIVE_QUIET_WINDOWS = [w.strip() for w in os.getenv("PROACTIVE_QUIET_WINDOWS", "00:00-06:00").split(",") if w.strip()]
PROACTIVE_FOCUS_WINDOWS = [w.strip() for w in os.getenv("PROACTIVE_FOCUS_WINDOWS", "mon-fri 06:00-19:00,sat-sun 06:00-11:00").split(",") if w.strip()]
PROACTIVE_SPEAK = os.getenv("PROACTIVE_SPEAK", "1").lower() in {"1", "true", "yes", "on"}
PROACTIVE_MESSAGES = [
    msg.strip()
    for msg in os.getenv(
        "PROACTIVE_MESSAGES",
        "You've been quiet for a bit. Still with me?,"
        "Checking in. Do you want focus time or should I keep you company?,"
        "Still here. If you're deep in something I can stay quiet.,"
        "Tiny ping. Need anything or are we in quiet mode?",
    ).split(",")
    if msg.strip()
]
PROACTIVE_REST_MESSAGE = os.getenv(
    "PROACTIVE_REST_MESSAGE",
    "You've been away for a while, so I'll go quiet and rest. Ping me when you need me.",
).strip()
PROACTIVE_PROMPT_HINTS = [
    msg.strip()
    for msg in os.getenv(
        "PROACTIVE_PROMPT_HINTS",
        "{user} has not spoken to you for a while. What short gentle thing do you want to say now?,"
        "{user} has been quiet for a while. Offer company without being needy or disruptive.,"
        "{user} may be focused or away. Say one brief check-in and make it easy to ignore.",
    ).split(",")
    if msg.strip()
]
PROACTIVE_REST_PROMPT_HINT = os.getenv(
    "PROACTIVE_REST_PROMPT_HINT",
    "{user} has not spoken to you for about an hour. Say one short warm line that you are going quiet and resting until they return.",
).strip()


def _personalize_proactive_text(text: str) -> str:
    """Fill lightweight proactive placeholders from identity config/env."""
    user = USER_ID or "the user"
    return (
        text.replace("{user}", user)
        .replace("{USER_ID}", user)
        .replace("{ai}", AI_NAME)
        .replace("{AI_NAME}", AI_NAME)
    )


_DAY_ALIASES = {
    "mon": 0, "monday": 0,
    "tue": 1, "tues": 1, "tuesday": 1,
    "wed": 2, "wednesday": 2,
    "thu": 3, "thur": 3, "thurs": 3, "thursday": 3,
    "fri": 4, "friday": 4,
    "sat": 5, "saturday": 5,
    "sun": 6, "sunday": 6,
}


def _parse_day_spec(spec: str) -> set[int] | None:
    """Parse day specs like mon-fri, sat-sun, weekdays, weekend, or daily."""
    spec = spec.strip().lower()
    if not spec or spec in {"daily", "everyday", "all"}:
        return None
    if spec in {"weekday", "weekdays"}:
        return {0, 1, 2, 3, 4}
    if spec in {"weekend", "weekends"}:
        return {5, 6}
    days: set[int] = set()
    for part in spec.split("|"):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_name, end_name = [p.strip() for p in part.split("-", 1)]
            start = _DAY_ALIASES.get(start_name)
            end = _DAY_ALIASES.get(end_name)
            if start is None or end is None:
                continue
            day = start
            while True:
                days.add(day)
                if day == end:
                    break
                day = (day + 1) % 7
        else:
            day = _DAY_ALIASES.get(part)
            if day is not None:
                days.add(day)
    return days or None


def _parse_hhmm(value: str) -> int | None:
    try:
        hour_s, minute_s = value.strip().split(":", 1)
        hour = int(hour_s)
        minute = int(minute_s)
    except ValueError:
        return None
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return hour * 60 + minute


def _time_window_matches(window: str, now: datetime | None = None) -> bool:
    """Return True when now is inside a YAML window.

    Supported examples:
      - "00:00-06:00"
      - "mon-fri 06:00-19:00"
      - "weekend 06:00-11:00"
      - "fri-mon 22:00-06:00"
    """
    if now is None:
        try:
            now = datetime.now(ZoneInfo(PROACTIVE_TIMEZONE)) if PROACTIVE_TIMEZONE else datetime.now()
        except ZoneInfoNotFoundError:
            now = datetime.now()
    raw = window.strip().lower()
    if not raw:
        return False
    if " " in raw:
        day_spec, time_spec = raw.rsplit(" ", 1)
    else:
        day_spec, time_spec = "", raw
    if "-" not in time_spec:
        return False
    start_s, end_s = [part.strip() for part in time_spec.split("-", 1)]
    start = _parse_hhmm(start_s)
    end = _parse_hhmm(end_s)
    if start is None or end is None:
        return False

    days = _parse_day_spec(day_spec)
    if days is not None and now.weekday() not in days:
        return False

    minute = now.hour * 60 + now.minute
    if start <= end:
        return start <= minute < end
    return minute >= start or minute < end


def _in_proactive_silence_window() -> bool:
    """True during configured quiet/focus windows."""
    return any(
        _time_window_matches(window)
        for window in [*PROACTIVE_QUIET_WINDOWS, *PROACTIVE_FOCUS_WINDOWS]
    )


def _seconds_until_proactive_silence_ends() -> float:
    """Return seconds until configured quiet/focus windows no longer match."""
    try:
        now = datetime.now(ZoneInfo(PROACTIVE_TIMEZONE)) if PROACTIVE_TIMEZONE else datetime.now()
    except ZoneInfoNotFoundError:
        now = datetime.now()

    windows = [*PROACTIVE_QUIET_WINDOWS, *PROACTIVE_FOCUS_WINDOWS]
    if not any(_time_window_matches(window, now) for window in windows):
        return 0.0

    # Window definitions are minute-granular, so scan minute boundaries instead
    # of waking every few seconds while Aiko is configured to be quiet.
    cursor = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
    for _ in range(8 * 24 * 60):  # enough to cover weekly wraparound windows
        if not any(_time_window_matches(window, cursor) for window in windows):
            return max(1.0, (cursor - now).total_seconds())
        cursor += timedelta(minutes=1)
    return 3600.0


class ProactiveIdleRunner:
    """Lightweight monitor that lets Aiko send gentle idle check-ins.

    This is deliberately local: it does not call the LLM, does not inspect the
    screen/camera, and never runs while a turn is active. The first check-in is
    jittered so Aiko feels less timer-like; after a longer idle window she goes
    quiet until the user returns.
    """

    def __init__(self, tui, speak, speak_enabled_fn, active_turn: threading.Event, generate_fn=None) -> None:
        self._tui = tui
        self._speak = speak
        self._speak_enabled_fn = speak_enabled_fn
        self._active_turn = active_turn
        self._generate_fn = generate_fn
        self._stop = threading.Event()
        self._wakeup = threading.Event()
        self._lock = threading.Lock()
        self._last_activity = time.monotonic()
        self._last_prompt = 0.0
        self._prompt_times: collections.deque[float] = collections.deque()
        self._message_index = 0
        self._enabled = PROACTIVE_ENABLED
        self._next_checkin_after = self._random_first_idle_delay()
        self._resting = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if PROACTIVE_FIRST_IDLE_MAX_SECONDS <= 0:
            return
        self._thread = threading.Thread(target=self._loop, name="aiko-proactive", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wakeup.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def touch(self) -> None:
        with self._lock:
            self._last_activity = time.monotonic()
            self._next_checkin_after = self._random_first_idle_delay()
            self._resting = False
        self._wakeup.set()

    def set_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._enabled = enabled
            self._last_activity = time.monotonic()
            self._next_checkin_after = self._random_first_idle_delay()
            self._resting = False
        self._wakeup.set()

    def is_enabled(self) -> bool:
        with self._lock:
            return self._enabled

    def _next_message(self) -> str:
        messages = PROACTIVE_MESSAGES or ["You've been quiet for a bit. Still with me?"]
        msg = messages[self._message_index % len(messages)]
        self._message_index += 1
        return _personalize_proactive_text(msg)

    def _next_prompt_hint(self) -> str:
        hints = PROACTIVE_PROMPT_HINTS or [
            "{user} has not spoken to you for a while. What short gentle thing do you want to say now?"
        ]
        hint = hints[self._message_index % len(hints)]
        self._message_index += 1
        return _personalize_proactive_text(hint)

    @staticmethod
    def _random_first_idle_delay() -> float:
        low = max(0.0, min(PROACTIVE_FIRST_IDLE_MIN_SECONDS, PROACTIVE_FIRST_IDLE_MAX_SECONDS))
        high = max(low, PROACTIVE_FIRST_IDLE_MAX_SECONDS)
        return random.uniform(low, high)

    def _rate_limit_allows_prompt(self, now: float) -> bool:
        one_hour_ago = now - 3600
        while self._prompt_times and self._prompt_times[0] < one_hour_ago:
            self._prompt_times.popleft()
        return PROACTIVE_MAX_PER_HOUR > 0 and len(self._prompt_times) < PROACTIVE_MAX_PER_HOUR

    def _prompt_due(self, now: float) -> tuple[str, str] | None:
        with self._lock:
            enabled = self._enabled
            idle_for = now - self._last_activity
            cooldown_for = now - self._last_prompt if self._last_prompt else float("inf")
            next_checkin_after = self._next_checkin_after
            resting = self._resting

        if not enabled:
            return None
        if resting:
            return None
        if _in_proactive_silence_window():
            return None
        if self._active_turn.is_set():
            return None
        if self._speak is not None and self._speak.is_playing():
            return None
        if PROACTIVE_REST_AFTER_SECONDS > 0 and idle_for >= PROACTIVE_REST_AFTER_SECONDS:
            if PROACTIVE_USE_LLM and self._generate_fn is not None and PROACTIVE_REST_PROMPT_HINT:
                return ("rest_prompt", _personalize_proactive_text(PROACTIVE_REST_PROMPT_HINT))
            if PROACTIVE_REST_MESSAGE:
                return ("rest_text", _personalize_proactive_text(PROACTIVE_REST_MESSAGE))
            with self._lock:
                self._resting = True
            return None
        if idle_for < next_checkin_after:
            return None
        if cooldown_for < PROACTIVE_COOLDOWN_SECONDS:
            return None
        if not self._rate_limit_allows_prompt(now):
            return None
        if PROACTIVE_USE_LLM and self._generate_fn is not None:
            return ("checkin_prompt", self._next_prompt_hint())
        return ("checkin_text", self._next_message())

    def _mark_prompt(self, now: float, *, rest: bool) -> None:
        with self._lock:
            self._last_prompt = now
            if rest:
                self._resting = True
            else:
                idle_for = now - self._last_activity
                self._next_checkin_after = idle_for + PROACTIVE_COOLDOWN_SECONDS
                self._prompt_times.append(now)

    def _seconds_until_next_check(self, now: float) -> float:
        """Sleep until the next meaningful proactive deadline or wakeup event."""
        with self._lock:
            enabled = self._enabled
            resting = self._resting
            last_activity = self._last_activity
            last_prompt = self._last_prompt
            next_checkin_after = self._next_checkin_after

        if not enabled or resting:
            return 3600.0

        quiet_remaining = _seconds_until_proactive_silence_ends()
        if quiet_remaining > 0:
            return quiet_remaining

        if self._active_turn.is_set():
            return 30.0
        if self._speak is not None and self._speak.is_playing():
            return 10.0

        candidates = [last_activity + next_checkin_after]
        if last_prompt:
            candidates.append(last_prompt + PROACTIVE_COOLDOWN_SECONDS)
        if PROACTIVE_REST_AFTER_SECONDS > 0:
            candidates.append(last_activity + PROACTIVE_REST_AFTER_SECONDS)
        if PROACTIVE_MAX_PER_HOUR > 0 and len(self._prompt_times) >= PROACTIVE_MAX_PER_HOUR:
            candidates.append(self._prompt_times[0] + 3600)

        future = [target - now for target in candidates if target > now]
        return max(1.0, min(future) if future else 1.0)

    def _loop(self) -> None:
        while not self._stop.is_set():
            now = time.monotonic()
            due = self._prompt_due(now)
            if due:
                kind, payload = due
                rest = kind.startswith("rest")
                self._mark_prompt(now, rest=rest)
                log.info("[proactive] idle %s: %s", "rest" if rest else "check-in", payload)
                try:
                    if kind.endswith("_prompt") and self._generate_fn is not None:
                        text = (self._generate_fn(payload) or "").strip()
                        if not text:
                            text = PROACTIVE_REST_MESSAGE if rest else self._next_message()
                    else:
                        text = payload
                    self._tui.add_message("aiko", text)
                    if (
                        not kind.endswith("_prompt")
                        and PROACTIVE_SPEAK
                        and self._speak is not None
                        and self._speak_enabled_fn()
                    ):
                        self._speak.speak(text)
                    self._tui._draw()
                except Exception as e:
                    log.warning("Proactive idle check-in failed: %s", e)

            sleep_for = self._seconds_until_next_check(time.monotonic())
            self._wakeup.wait(timeout=sleep_for)
            self._wakeup.clear()


# ── voice command map ─────────────────────────────────────────────────────────
#
# Maps spoken phrases to their slash-command equivalents.
# ASR can prepend filler words ("uh", "um", "okay", "hey aiko") — the
# matcher strips these before comparison. Fuzzy matching handles minor
# transcription drift (e.g. "reset context" → /reset).
#
# Keep phrases short and phonetically distinct so ASR catches them reliably.

_VOICE_COMMANDS: dict[str, str] = {
    # session control
    "stop":              "/quit",
    "hey stop":          "/quit",
    "quit":              "/quit",
    "exit":              "/quit",
    "goodbye":           "/quit",
    # context
    "reset":             "/reset",
    "forget that":       "/reset",
    "clear context":     "/reset",
    "start over":        "/reset",
    # memory
    "remember this":     "/remember",
    "pin this":          "/remember",
    "save this":         "/remember",
    "show memory":       "/memory",
    "show memories":     "/memory",
    "what do you remember": "/memory",
    "clear memory":      "/clear",
    "wipe memory":       "/clear",
    "delete memories":   "/clear",
    # voice toggles
    "mute":              "/voice",
    "unmute":            "/voice",
    "toggle voice":      "/voice",
    "stop talking":      "/voice",
    "toggle listen":     "/listen",
    "stop listening":    "/listen",
    # meta
    "help":              "/help",
    "what can you do":   "/help",
}

# Filler words ASR may prepend that we should strip before matching
_FILLER_RE = re.compile(
    r"^\s*(uh+|um+|ah+|okay|hey\s+aiko|aiko)[,.]?\s*",
    flags=re.IGNORECASE,
)


def _match_voice_command(text: str) -> str | None:
    """
    Fuzzy-match transcribed text against known voice commands.

    Strips leading filler words before comparing, then tries exact match
    first and falls back to difflib fuzzy matching with a 0.75 cutoff.
    Returns the slash command string if confident, None otherwise.

    Args:
        text: Raw ASR transcript for the current turn.

    Returns:
        Slash command string (e.g. "/reset") or None if no match.
    """
    clean = _FILLER_RE.sub("", text.strip()).lower().rstrip(".,!?")
    if not clean:
        return None
    # exact match
    if clean in _VOICE_COMMANDS:
        return _VOICE_COMMANDS[clean]
    # fuzzy — conservative cutoff to avoid false positives mid-conversation
    matches = difflib.get_close_matches(
        clean, _VOICE_COMMANDS.keys(), n=1, cutoff=0.75,
    )
    if matches:
        return _VOICE_COMMANDS[matches[0]]
    return None


# ═════════════════════════════════════════════════════════════════════════════
# ARGUMENT PARSING
# ═════════════════════════════════════════════════════════════════════════════

def parse_args():
    """Parse and return the CLI argument namespace for Aiko-chan's launch options."""
    p = argparse.ArgumentParser(description="Aiko-chan CLI")
    p.add_argument("--text",      action="store_true",
                   help="keyboard input + TTS/ASR initially off; both subsystems still load for /voice and /listen toggles")
    p.add_argument("--no-asr",    action="store_true",
                   help="keyboard input but keep TTS on; ASR still loads for /listen")
    p.add_argument("--debug",     action="store_true",
                   help="show memory hits each turn")
    p.add_argument("--tui",       action="store_true",
                   help="use the curses TUI (default)")
    p.add_argument("--webui",     action="store_true",
                   help="use the browser WebUI instead of the default curses TUI")
    p.add_argument("--clear-mem", action="store_true",
                   help="wipe all stored memories and exit")
    return p.parse_args()


# ═════════════════════════════════════════════════════════════════════════════
# SESSION ORCHESTRATOR
# ═════════════════════════════════════════════════════════════════════════════
        
def _run_tui(stdscr, args):
    """
    Orchestrate the full session lifecycle from boot to shutdown inside the
    curses wrapper.

    Stages:
        1. Spawn the TUI and begin the init spin loop.
        2. Delegate all subsystem boot to AikoWakeup, passing TUI callbacks.
        3. Transition the TUI to the active chat phase.
        4. Enter the main input → inference → render loop.
        5. On exit, stop the barge-in monitor and wait for background memory
           writes to complete.
    """
    tui = AikoTUI(stdscr, no_voice=args.text, debug=args.debug)
    _run_session(tui, args)


def _run_webui(args):
    """Launch Aiko with the browser WebUI."""
    tui = AikoWeb(no_voice=args.text, debug=args.debug)
    import socket
    host_ip = socket.gethostbyname(socket.gethostname())
    scheme = "https" if WEBUI_HTTPS else "http"
    print(f"\n  🌸 Aiko-chan is ready → {scheme}://{host_ip}:{HTTP_PORT}/\n")
    _run_session(tui, args)


def _run_session(tui, args):
    """Run Aiko using any UI object that implements the AikoTUI-compatible API."""
    last_stream_draw = 0.0

    def token_cb(token):
        nonlocal last_stream_draw
        if token:
            _latency_mark("first_token_at")
            if not token.startswith("__"):
                _latency_mark("first_assistant_token_at")
        if token.startswith("__THINKING__") or token.startswith("__TOOL__:"):
            if current_latency is not None:
                current_latency["mode"] = "agentic"
        if token.startswith("__SEARCHING__:"):
            query = token.split(":", 1)[1]
            tui.add_message('sys', f'Searching: {query}...')
            tui._draw()
        else:
            tui.stream_token(token)
            now = time.monotonic()
            if now - last_stream_draw >= STREAM_DRAW_INTERVAL:
                tui._draw(buf=[])
                last_stream_draw = now

    # ── init spin ─────────────────────────────────────────────────────────────

    spin_stop = threading.Event()
    spin_t    = threading.Thread(target=tui.spin_loop, args=(spin_stop,), daemon=True)
    spin_t.start()

    # ── boot all subsystems via wakeup ────────────────────────────────────────

    # Load voice subsystems even when initially toggled off so /voice and
    # /listen can turn them on without a second boot-time model load.
    result = AikoWakeup(text_mode=False).boot(
        on_loading = tui.step_loading,
        on_done    = tui.step_done,
        on_skip    = tui.step_skip,
    )

    think    = result.think
    memorize = result.memorize
    speak    = result.speak
    listen   = result.listen

    if speak and hasattr(tui, "broadcast_audio_bytes"):
        speak.set_audio_sink(tui.broadcast_audio_bytes)
        if hasattr(tui, "set_viseme"):
            speak.set_viseme_sink(tui.set_viseme)
        if os.getenv("WEBUI_LOCAL_PLAYBACK", "1").lower() in {"0", "false", "no", "off"}:
            speak.local_playback = False

    # ── transition to chat ────────────────────────────────────────────────────

    spin_stop.set()
    spin_t.join()
    tui.status_finish()
    tui._draw()

    tts_enabled = not args.text
    asr_enabled = not args.text and not args.no_asr
    if hasattr(tui, "_stats"):
        tui._stats['tts_on'] = tts_enabled
        tui._stats['asr_on'] = asr_enabled

    current_latency: dict | None = None

    def _latency_mark(name: str) -> None:
        if current_latency is not None and name not in current_latency:
            current_latency[name] = time.monotonic()

    if speak is not None and hasattr(speak, "set_first_audio_callback"):
        speak.set_first_audio_callback(lambda: _latency_mark("first_audio_at"))

    def _latency_seconds(timing: dict, start: str, end: str) -> float | None:
        if timing.get(start) is None or timing.get(end) is None:
            return None
        return max(0.0, timing[end] - timing[start])

    def _fmt_latency(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.3f}s"

    def _latency_parts(timing: dict) -> dict:
        return {
            "voice_end_to_submit": _latency_seconds(timing, "voice_recording_stopped_at", "submitted_at"),
            "submit_to_first_token": _latency_seconds(timing, "submitted_at", "first_assistant_token_at"),
            "submit_to_assistant_done": _latency_seconds(timing, "submitted_at", "assistant_done_at"),
            "assistant_done_to_first_audio": _latency_seconds(timing, "assistant_done_at", "first_audio_at"),
            "voice_end_to_first_audio": _latency_seconds(timing, "voice_recording_stopped_at", "first_audio_at"),
            "submit_to_first_audio": _latency_seconds(timing, "submitted_at", "first_audio_at"),
            "submit_to_turn_done": _latency_seconds(timing, "submitted_at", "turn_done_at"),
        }

    def _update_latency_stats(timing: dict) -> None:
        if not hasattr(tui, "set_latency_stats"):
            return
        parts = _latency_parts(timing)
        tui.set_latency_stats({
            "voice_end_to_first_audio": _fmt_latency(parts["voice_end_to_first_audio"]),
        })

    def _log_latency(timing: dict) -> None:
        if not LATENCY_LOG_ENABLED:
            return
        mode = timing.get("mode", "text")
        parts = _latency_parts(timing)
        log.info(
            "[latency] mode=%s voice_end→submit=%s submit→first_token=%s "
            "submit→assistant_done=%s assistant_done→first_audio=%s "
            "voice_end→first_audio=%s submit→first_audio=%s submit→turn_done=%s",
            mode,
            _fmt_latency(parts["voice_end_to_submit"]),
            _fmt_latency(parts["submit_to_first_token"]),
            _fmt_latency(parts["submit_to_assistant_done"]),
            _fmt_latency(parts["assistant_done_to_first_audio"]),
            _fmt_latency(parts["voice_end_to_first_audio"]),
            _fmt_latency(parts["submit_to_first_audio"]),
            _fmt_latency(parts["submit_to_turn_done"]),
        )

    # ── shutdown helper ───────────────────────────────────────────────────────

    session_active = threading.Event()

    def _generate_proactive_checkin(prompt_hint: str) -> str:
        session_active.set()
        original_speak = getattr(think, "_speak", None)
        try:
            if not tts_enabled:
                think.set_speak(None)
            return think.proactive_checkin(prompt_hint)
        finally:
            if not tts_enabled:
                think.set_speak(original_speak)
            session_active.clear()
            if hasattr(think, "_last_chat_time"):
                think._last_chat_time = time.time()

    proactive = ProactiveIdleRunner(
        tui,
        speak,
        speak_enabled_fn=lambda: tts_enabled,
        active_turn=session_active,
        generate_fn=_generate_proactive_checkin,
    )
    proactive.start()

    def _shutdown():
        """Stop background daemons and flush memory writes before exit."""
        proactive.stop()
        if listen is not None:
            listen.stop_barge_in_monitor()
        think.wait_for_memory()

    # ── main loop ─────────────────────────────────────────────────────────────

    while True:
        try:
            voice_info = None
            if listen and asr_enabled:
                result = tui.get_voice_input(
                    listen,
                    speak   = speak if tts_enabled else None,
                    wait_fn = None,
                )
                user_input = result[0] if isinstance(result, tuple) else result
                voice_info = result[1] if isinstance(result, tuple) and len(result) > 1 and isinstance(result[1], dict) else None
            else:
                result = tui.get_input()
                user_input = result[0] if isinstance(result, tuple) else result
        except KeyboardInterrupt:
            tui.add_message('sys', "Fine... I'll be here when you come back.")
            tui._draw()
            _shutdown()
            time.sleep(0.8)
            return

        if not user_input:
            continue

        proactive.touch()

        # ── voice command check (ASR mode only) ───────────────────────────────
        #
        # Run before the /cmd block so spoken commands like "forget that"
        # are rewritten to "/reset" and fall through into the normal handler.
        # Only active in ASR mode — text-mode users type slash commands directly.

        if listen and asr_enabled and not user_input.startswith('/'):
            matched = _match_voice_command(user_input)
            if matched:
                user_input = matched

        # ── commands ──────────────────────────────────────────────────────────

        if user_input.startswith('/'):
            cmd = user_input.lower().strip()

            if cmd in ('/quit', '/exit'):
                tui.add_message('sys', 'Already leaving? ...Be safe out there.')
                tui._draw()
                _shutdown()
                time.sleep(0.8)
                return

            elif cmd == '/reset':
                think.reset_context()
                tui.add_message('sys', 'Short-term context cleared.')

            elif cmd == '/memory':
                all_mem = memorize.get_all()
                if not all_mem:
                    tui.add_message('sys', 'No memories stored yet.')
                else:
                    tui.add_message('sys', f'{len(all_mem)} memories stored:')
                    for i, m in enumerate(all_mem, 1):
                        tui.add_message('sys',
                            f'  {i:02d}. {m.get("memory") or m.get("text") or m}')

            elif cmd == '/clear':
                memorize.clear()
                tui.add_message('sys', 'All persistent memories cleared.')

            elif cmd == '/remember':
                # pin the last user + assistant exchange permanently
                turn = think.last_turn()
                if not turn:
                    tui.add_message('sys', 'Nothing to remember yet — send a message first.')
                else:
                    think.wait_for_memory()
                    user_text, ai_text = turn
                    msgs = [
                        {"role": "user",      "content": user_text},
                        {"role": "assistant", "content": ai_text},
                    ]
                    ok = memorize.pin(msgs)
                    if ok:
                        tui.add_message('sys', "Got it — I'll remember that forever.")
                    else:
                        tui.add_message('sys', 'Failed to pin memory — check logs.')

            elif cmd.startswith('/think'):
                query = user_input[6:].strip()
                if not query:
                    tui.add_message('sys', 'Usage: /think <question>')
                    tui._draw()
                    continue

                think.set_reasoning(True)
                tui.add_message('you', f'[think] {query}')
                tui.turn_start()
                tui._draw()

                raw_chunks     = []
                in_think_block = False
                think_closed   = False

                def _think_token_cb(token):
                    nonlocal in_think_block, think_closed
                    raw_chunks.append(token)
                    assembled = "".join(raw_chunks)

                    if not think_closed:
                        if "<think>" in assembled and not in_think_block:
                            in_think_block = True
                        if in_think_block:
                            if "</think>" in assembled:
                                in_think_block = False
                                think_closed   = True
                            return

                    tui.stream_token(token)
                    tui._draw(buf=[])

                think.chat(query, token_callback=_think_token_cb)

                assembled_full   = "".join(raw_chunks)
                scratchpad_match = re.search(r"<think>(.*?)</think>", assembled_full, re.DOTALL)
                if scratchpad_match:
                    inner = scratchpad_match.group(1).strip()
                    if inner:
                        tui.add_message('sys',
                            f'[scratchpad] {inner[:300]}{"…" if len(inner) > 300 else ""}')

                tui.stream_commit()
                tui._draw()
                continue

            elif cmd == '/voice':
                if speak is None:
                    tui.add_message('sys', 'TTS unavailable — voice subsystem did not load.')
                else:
                    tts_enabled = not tts_enabled
                    think.set_speak(speak if tts_enabled else None)
                    tui._stats['tts_on'] = tts_enabled
                    tui.add_message('sys',
                        f'Voice output (TTS): {"ON  🔊" if tts_enabled else "OFF 🔇"}')

            elif cmd == '/listen':
                if listen is None:
                    tui.add_message('sys', 'ASR unavailable — voice subsystem did not load.')
                else:
                    asr_enabled = not asr_enabled
                    tui._stats['asr_on'] = asr_enabled
                    tui.add_message('sys',
                        f'Voice input  (ASR): {"ON  🎤" if asr_enabled else "OFF ⌨ "}')

            elif cmd == '/proactive':
                proactive.set_enabled(not proactive.is_enabled())
                tui.add_message('sys',
                    f'Proactive idle check-ins: {"ON  🌸" if proactive.is_enabled() else "OFF 💤"}')

            elif cmd == '/help':
                for line in [
                    '/quit /exit              — end session',
                    '/reset                   — clear short-term context',
                    '/clear                   — wipe long-term memories',
                    '/remember                — pin last turn forever (decay-proof)',
                    '/memory                  — show stored memories',
                    '/think <question>        — reason step-by-step (single-shot, 3× token budget)',
                    '/web <query>             — web search',
                    '/voice                   — toggle TTS on/off',
                    '/listen                  — toggle ASR on/off',
                    '/proactive               — toggle idle check-ins on/off',
                    '/help                    — show this list',
                    '',
                    'Voice commands (say aloud in ASR mode):',
                    '  "forget that"          → /reset',
                    '  "remember this"        → /remember',
                    '  "show memory"          → /memory',
                    '  "clear memory"         → /clear',
                    '  "mute" / "unmute"      → /voice',
                    '  "stop" / "goodbye"     → /quit',
                    '  "help"                 → /help',
                ]:
                    tui.add_message('sys', line)

            elif cmd.startswith('/web '):
                query = user_input[5:].strip()
                if not query:
                    tui.add_message('sys', 'Usage: /web <query>')
                else:
                    tui.add_message('sys', f'Searching: "{query}"')
                    tui._draw()
                    try:
                        results = web_search(query)
                    except Exception as e:
                        tui.add_message('sys', f'Search failed: {e}')
                        tui._draw()
                        continue
                    tui.turn_start()
                    def _web_token_cb(token):
                        tui.stream_token(token)
                        tui._draw(buf=[])
                    think.chat(
                        f"Use these web search results to answer the question: {query}\n\n{results}",
                        token_callback=_web_token_cb,
                    )
                    tui.stream_commit()
                tui._draw()

            else:
                tui.add_message('sys', f'Unknown command: {user_input}')

            tui._draw()
            continue

        # ── normal turn ───────────────────────────────────────────────────────

        if args.debug:
            hits = memorize.search(user_input)
            if hits:
                tui.add_message('sys', f'{len(hits)} memories retrieved:')
                for m in hits:
                    tui.add_message('sys',
                        f'  → {m.get("memory") or m.get("text") or m}')

        current_latency = {
            "mode": "voice" if (listen and asr_enabled and voice_info) else "text",
            "submitted_at": time.monotonic(),
        }
        if voice_info:
            current_latency["listen_started_at"] = voice_info.get("listen_started_at")
            current_latency["voice_recording_stopped_at"] = voice_info.get("recording_stopped_at")

        tui.add_message('you', user_input)
        tui.turn_start()
        session_active.set()
        tui._draw()

        try:
            think.route(user_input, token_callback=token_cb)
            current_latency["assistant_done_at"] = time.monotonic()
            if speak and tts_enabled:
                speak.wait()
            tui.stream_commit()
            tui._draw()
            current_latency["turn_done_at"] = time.monotonic()
            _update_latency_stats(current_latency)
            _log_latency(current_latency)
            tui._draw()
        finally:
            session_active.clear()
            proactive.touch()
            current_latency = None


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def main():
    """Primary entry point for the Aiko-chan CLI."""
    args = parse_args()
    if args.clear_mem:
        log.info("Clearing all memories...")
        AikoMemorize().clear()
        sys.exit(0)
    if args.tui and args.webui:
        raise SystemExit("Choose only one UI: --tui or --webui")
    if args.webui:
        _run_webui(args)
    else:
        curses.wrapper(lambda scr: _run_tui(scr, args))


if __name__ == '__main__':
    main()
