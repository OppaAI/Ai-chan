"""
main.py

Aiko-chan CLI — entry point and session orchestrator.

Usage:
    python main.py               # browser WebUI (default) — full voice, ASR + TTS
    python main.py --text        # WebUI, keyboard input + TTS/ASR toggled off
    python main.py --no-asr      # WebUI, keyboard input but keep TTS on
    python main.py --cli         # plain no-curses CLI, for local testing only
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
    - Surface live agentic status (thinking / tool calls / search) as
      readable status lines instead of a generic "thinking..." spinner
    - Clean shutdown on Ctrl-C / Ctrl-D

Note: the old curses TUI (tui/tui.py) has been retired in favor of the
WebUI as the default front end, plus cli/simple_cli.py for quick, no-frills
local testing. Move tui/tui.py to archive/tui/ in your own checkout —
nothing here imports curses anymore.
"""
from core.config import load_config
load_config()

import argparse
import collections
from datetime import datetime, timedelta
import difflib
import json
import os
import random
import re
import sys
import threading
import time
import warnings
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

warnings.filterwarnings("ignore")

import logging

from core.silence import silent_stderr
from core.log     import get_logger
from core.wakeup  import AikoWakeup
from core.toolkit.web import web_search

log = get_logger(__name__)

with silent_stderr():
    from core.memorize import AikoMemorize

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


def _parse_env_list(name: str, default: list[str]) -> list[str]:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]
    return [item.strip() for item in raw.split(",") if item.strip()]


PROACTIVE_TIMEZONE = os.getenv("PROACTIVE_TIMEZONE", "").strip() or os.getenv("TIMEZONE", "").strip()
PROACTIVE_QUIET_WINDOWS = _parse_env_list("PROACTIVE_QUIET_WINDOWS", ["00:00-06:00"])
PROACTIVE_FOCUS_WINDOWS = _parse_env_list(
    "PROACTIVE_FOCUS_WINDOWS",
    ["mon-fri 06:00-19:00", "sat-sun 06:00-11:00"],
)
PROACTIVE_SPEAK = os.getenv("PROACTIVE_SPEAK", "1").lower() in {"1", "true", "yes", "on"}
PROACTIVE_MESSAGES = _parse_env_list("PROACTIVE_MESSAGES", [
    "You've been quiet for a bit. Still with me?",
    "Checking in. Do you want focus time or should I keep you company?",
    "Still here. If you're deep in something I can stay quiet.",
    "Tiny ping. Need anything or are we in quiet mode?",
])
PROACTIVE_REST_MESSAGE = os.getenv(
    "PROACTIVE_REST_MESSAGE",
    "You've been away for a while, so I'll go quiet and rest. Ping me when you need me.",
).strip()
PROACTIVE_PROMPT_HINTS = _parse_env_list("PROACTIVE_PROMPT_HINTS", [
    "{user} has not spoken to you for a while. What short gentle thing do you want to say now?",
    "{user} has been quiet for a while. Offer company without being needy or disruptive.",
    "{user} may be focused or away. Say one brief check-in and make it easy to ignore.",
])
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

    def __init__(self, ui, speak, speak_enabled_fn, active_turn: threading.Event, generate_fn=None) -> None:
        self._ui = ui
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
                    self._ui.add_message("aiko", text)
                    if (
                        not kind.endswith("_prompt")
                        and PROACTIVE_SPEAK
                        and self._speak is not None
                        and self._speak_enabled_fn()
                    ):
                        self._speak.speak(text)
                    self._ui._draw()
                except Exception:
                    log.exception("Proactive idle check-in failed")

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


# ── agentic status markers ─────────────────────────────────────────────────────
#
# core/think.py streams special "__MARKER__" or "__MARKER__:payload" tokens
# to surface what Aiko is actually doing mid-turn (searching, calling a tool,
# reasoning) instead of a generic "thinking" spinner. This maps each marker
# to an icon + label, and _handle_status_marker() renders it as a 'sys'
# message rather than letting the raw marker leak into the streamed reply.
#
# If core/think.py emits additional marker types, add them here rather than
# letting them fall through to ui.stream_token() as literal text.

_STATUS_MARKERS: dict[str, tuple[str, str]] = {
    "__THINKING__":  ("🤔", "Thinking"),
    "__TOOL__":      ("🔧", "Using tool"),
    "__SEARCHING__": ("🔍", "Searching"),
}


def _handle_status_marker(ui, token: str) -> bool:
    """
    Detect a "__MARKER__" or "__MARKER__:payload" token and render it as a
    live status line describing what Aiko is actually doing, instead of
    streaming the raw marker into the chat text.

    core/agentic.py emits these with a trailing newline (e.g. "__THINKING__\n"
    and "__TOOL__:name(args)\n"), so strip trailing newlines before matching.

    Returns True if the token was a recognized status marker (caller should
    not also pass it to ui.stream_token), False otherwise.
    """
    stripped_token = token.rstrip("\r\n")
    for marker, (icon, label) in _STATUS_MARKERS.items():
        if stripped_token == marker:
            ui.add_message('sys', f'{icon} {label}...')
            ui._draw()
            return True
        prefix = marker + ":"
        if stripped_token.startswith(prefix):
            payload = stripped_token[len(prefix):].strip()
            text = f'{icon} {label}: {payload}' if payload else f'{icon} {label}...'
            ui.add_message('sys', text)
            ui._draw()
            return True
    return False


# ═════════════════════════════════════════════════════════════════════════════
# SIMPLE CLI (testing only — no curses, no layout, just plain stdout)
# ═════════════════════════════════════════════════════════════════════════════
#
# Implements the same duck-typed "ui" interface _run_session() expects
# (add_message, turn_start, stream_token, stream_commit, get_input,
# get_voice_input, _draw, _stats, boot callbacks) but renders everything as
# plain scrolling stdout lines. Intended for quick manual testing — the
# WebUI is the default front end for real use.

_CLI_ROLE_PREFIX = {
    "you":  "You",
    "aiko": "Aiko",
    "sys":  "·",
}


class AikoSimpleCLI:
    def __init__(self, no_voice: bool = False, debug: bool = False) -> None:
        self.no_voice = no_voice
        self.debug = debug
        self._stats: dict = {"tts_on": not no_voice, "asr_on": not no_voice}
        self._streaming = False
        self._stream_buf: list[str] = []
        self._latency_stats: dict = {}

    # ── boot / status ────────────────────────────────────────────────────
    def spin_loop(self, stop_event: threading.Event) -> None:
        frames = "|/-\\"
        i = 0
        while not stop_event.is_set():
            print(f"\r  {frames[i % len(frames)]} booting Aiko...", end="", flush=True)
            i += 1
            time.sleep(0.15)
        print("\r" + " " * 40 + "\r", end="", flush=True)

    def step_loading(self, name: str) -> None:
        print(f"  ⏳ loading {name}...")

    def step_done(self, name: str) -> None:
        print(f"  ✅ {name} ready")

    def step_skip(self, name: str) -> None:
        print(f"  ⏭  {name} skipped")

    def status_finish(self) -> None:
        print("\n🌸 Aiko-chan is ready. Type a message, or /help for commands.\n")

    # ── rendering ────────────────────────────────────────────────────────
    def _draw(self, buf: list | None = None) -> None:
        # Plain scrolling CLI — nothing to redraw, output is already live.
        pass

    def add_message(self, role: str, text: str) -> None:
        prefix = _CLI_ROLE_PREFIX.get(role, role)
        print(f"{prefix}: {text}")

    def turn_start(self) -> None:
        self._streaming = True
        self._stream_buf = []
        print("Aiko: ", end="", flush=True)

    def stream_token(self, token: str) -> None:
        if not token:
            return
        self._stream_buf.append(token)
        print(token, end="", flush=True)

    def stream_commit(self) -> None:
        if self._streaming:
            print()  # newline after streamed reply
        self._streaming = False
        self._stream_buf = []

    def set_latency_stats(self, stats: dict) -> None:
        self._latency_stats = stats
        v_to_a = stats.get("voice_end_to_first_audio")
        if v_to_a and v_to_a != "n/a":
            print(f"  ⏱  V→A (voice end → Aiko's first audio): {v_to_a}")
        if self.debug:
            print(f"  [latency] {stats}")

    # ── input ────────────────────────────────────────────────────────────
    def get_input(self):
        try:
            text = input("You: ").strip()
        except EOFError:
            return "/quit"
        return text

    def get_voice_input(self, listen, speak=None, wait_fn=None):
        """
        Best-effort voice input for CLI testing. The CLI is text-first; if a
        `listen` backend is loaded, try a few common blocking method names.
        Rename to match your real ASR backend's single-utterance API if it
        differs — this is a testing convenience, not the primary voice path
        (that's the WebUI's AudioWorklet pipeline).

        Returns (text, timing_dict) on a successful voice capture so
        _run_session() can compute V->A latency. `recording_stopped_at` here
        is approximate — it's stamped when the blocking listen call returns,
        which includes ASR transcription time, not just end-of-speech. The
        WebUI's real pipeline stamps this earlier (right at end-of-speech),
        so CLI latency numbers will read a bit higher than the WebUI's.
        """
        for method_name in ("listen_once", "transcribe_once", "listen"):
            method = getattr(listen, method_name, None)
            if callable(method):
                print("🎤 listening... (speak now)")
                listen_started_at = time.monotonic()
                try:
                    result = method()
                except Exception as e:
                    print(f"  [voice input failed: {e}]")
                    return self.get_input()
                recording_stopped_at = time.monotonic()
                text = (result[0] if isinstance(result, tuple) else result) or ""
                text = text.strip()
                if text:
                    print(f"You (voice): {text}")
                return text, {
                    "listen_started_at": listen_started_at,
                    "recording_stopped_at": recording_stopped_at,
                }
        # No compatible voice API found on this backend — fall back to typed input.
        return self.get_input()


# ═════════════════════════════════════════════════════════════════════════════
# ARGUMENT PARSING
# ═════════════════════════════════════════════════════════════════════════════

def parse_args():
    """Parse and return the CLI argument namespace for Aiko-chan's launch options."""
    p = argparse.ArgumentParser(description="Aiko-chan")
    p.add_argument("--text",      action="store_true",
                   help="keyboard input + TTS/ASR initially off; both subsystems still load for /voice and /listen toggles")
    p.add_argument("--no-asr",    action="store_true",
                   help="keyboard input but keep TTS on; ASR still loads for /listen")
    p.add_argument("--debug",     action="store_true",
                   help="show memory hits each turn")
    p.add_argument("--cli",       action="store_true",
                   help="use the plain no-curses CLI instead of the WebUI — for local testing only")
    p.add_argument("--clear-mem", action="store_true",
                   help="wipe all stored memories and exit")
    return p.parse_args()


# ═════════════════════════════════════════════════════════════════════════════
# SESSION ORCHESTRATOR
# ═════════════════════════════════════════════════════════════════════════════

def _run_cli(args):
    """Launch Aiko with the plain no-curses CLI (testing only)."""
    ui = AikoSimpleCLI(no_voice=args.text, debug=args.debug)
    _run_session(ui, args)


def _run_webui(args):
    """Launch Aiko with the browser WebUI (default)."""
    ui = AikoWeb(no_voice=args.text, debug=args.debug)
    import socket
    host_ip = socket.gethostbyname(socket.gethostname())
    scheme = "https" if WEBUI_HTTPS else "http"
    print(f"\n  🌸 Aiko-chan is ready → {scheme}://{host_ip}:{HTTP_PORT}/\n")
    _run_session(ui, args)


def _run_session(ui, args):
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
        if _handle_status_marker(ui, token):
            return
        ui.stream_token(token)
        now = time.monotonic()
        if now - last_stream_draw >= STREAM_DRAW_INTERVAL:
            ui._draw(buf=[])
            last_stream_draw = now

    # ── init spin ─────────────────────────────────────────────────────────────

    spin_stop = threading.Event()
    spin_t    = threading.Thread(target=ui.spin_loop, args=(spin_stop,), daemon=True)
    spin_t.start()

    # ── boot all subsystems via wakeup ────────────────────────────────────────

    # Load voice subsystems even when initially toggled off so /voice and
    # /listen can turn them on without a second boot-time model load.
    result = AikoWakeup(text_mode=False).boot(
        on_loading = ui.step_loading,
        on_done    = ui.step_done,
        on_skip    = ui.step_skip,
    )

    think    = result.think
    memorize = result.memorize
    speak    = result.speak
    listen   = result.listen

    if speak and hasattr(ui, "broadcast_audio_bytes"):
        speak.set_audio_sink(ui.broadcast_audio_bytes)
        if hasattr(ui, "set_viseme"):
            speak.set_viseme_sink(ui.set_viseme)
        if os.getenv("WEBUI_LOCAL_PLAYBACK", "1").lower() in {"0", "false", "no", "off"}:
            speak.local_playback = False

    # ── transition to chat ────────────────────────────────────────────────────

    spin_stop.set()
    spin_t.join()
    ui.status_finish()
    ui._draw()

    tts_enabled = not args.text
    asr_enabled = not args.text and not args.no_asr
    if hasattr(ui, "_stats"):
        ui._stats['tts_on'] = tts_enabled
        ui._stats['asr_on'] = asr_enabled

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
        if not hasattr(ui, "set_latency_stats"):
            return
        parts = _latency_parts(timing)
        ui.set_latency_stats({
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
            if not tts_enabled or not PROACTIVE_SPEAK:
                think.set_speak(None)
            return think.proactive_checkin(prompt_hint)
        finally:
            if not tts_enabled or not PROACTIVE_SPEAK:
                think.set_speak(original_speak)
            session_active.clear()
            if hasattr(think, "_last_chat_time"):
                think._last_chat_time = time.time()

    proactive = ProactiveIdleRunner(
        ui,
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
                result = ui.get_voice_input(
                    listen,
                    speak   = speak if tts_enabled else None,
                    wait_fn = None,
                )
                user_input = result[0] if isinstance(result, tuple) else result
                voice_info = result[1] if isinstance(result, tuple) and len(result) > 1 and isinstance(result[1], dict) else None
            else:
                result = ui.get_input()
                user_input = result[0] if isinstance(result, tuple) else result
        except KeyboardInterrupt:
            ui.add_message('sys', "Fine... I'll be here when you come back.")
            ui._draw()
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
                ui.add_message('sys', 'Already leaving? ...Be safe out there.')
                ui._draw()
                _shutdown()
                time.sleep(0.8)
                return

            elif cmd == '/reset':
                think.reset_context()
                ui.add_message('sys', 'Short-term context cleared.')

            elif cmd == '/memory':
                all_mem = memorize.get_all()
                if not all_mem:
                    ui.add_message('sys', 'No memories stored yet.')
                else:
                    ui.add_message('sys', f'{len(all_mem)} memories stored:')
                    for i, m in enumerate(all_mem, 1):
                        ui.add_message('sys',
                            f'  {i:02d}. {m.get("memory") or m.get("text") or m}')

            elif cmd == '/clear':
                memorize.clear()
                ui.add_message('sys', 'All persistent memories cleared.')

            elif cmd == '/remember':
                # pin the last user + assistant exchange permanently
                turn = think.last_turn()
                if not turn:
                    ui.add_message('sys', 'Nothing to remember yet — send a message first.')
                else:
                    think.wait_for_memory()
                    user_text, ai_text = turn
                    msgs = [
                        {"role": "user",      "content": user_text},
                        {"role": "assistant", "content": ai_text},
                    ]
                    ok = memorize.pin(msgs)
                    if ok:
                        ui.add_message('sys', "Got it — I'll remember that forever.")
                    else:
                        ui.add_message('sys', 'Failed to pin memory — check logs.')

            elif cmd.startswith('/think'):
                query = user_input[6:].strip()
                if not query:
                    ui.add_message('sys', 'Usage: /think <question>')
                    ui._draw()
                    continue

                think.set_reasoning(True)
                ui.add_message('you', f'[think] {query}')
                ui.turn_start()
                ui._draw()

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

                    ui.stream_token(token)
                    ui._draw(buf=[])

                think.chat(query, token_callback=_think_token_cb)

                assembled_full   = "".join(raw_chunks)
                scratchpad_match = re.search(r"<think>(.*?)</think>", assembled_full, re.DOTALL)
                if scratchpad_match:
                    inner = scratchpad_match.group(1).strip()
                    if inner:
                        ui.add_message('sys',
                            f'[scratchpad] {inner[:300]}{"…" if len(inner) > 300 else ""}')

                ui.stream_commit()
                ui._draw()
                continue

            elif cmd == '/voice':
                if speak is None:
                    ui.add_message('sys', 'TTS unavailable — voice subsystem did not load.')
                else:
                    tts_enabled = not tts_enabled
                    think.set_speak(speak if tts_enabled else None)
                    ui._stats['tts_on'] = tts_enabled
                    ui.add_message('sys',
                        f'Voice output (TTS): {"ON  🔊" if tts_enabled else "OFF 🔇"}')

            elif cmd == '/listen':
                if listen is None:
                    ui.add_message('sys', 'ASR unavailable — voice subsystem did not load.')
                else:
                    asr_enabled = not asr_enabled
                    ui._stats['asr_on'] = asr_enabled
                    ui.add_message('sys',
                        f'Voice input  (ASR): {"ON  🎤" if asr_enabled else "OFF ⌨ "}')

            elif cmd == '/proactive':
                proactive.set_enabled(not proactive.is_enabled())
                ui.add_message('sys',
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
                    ui.add_message('sys', line)

            elif cmd.startswith('/web '):
                query = user_input[5:].strip()
                if not query:
                    ui.add_message('sys', 'Usage: /web <query>')
                else:
                    ui.add_message('sys', f'Searching: "{query}"')
                    ui._draw()
                    try:
                        results = web_search(query)
                    except Exception as e:
                        ui.add_message('sys', f'Search failed: {e}')
                        ui._draw()
                        continue
                    ui.turn_start()
                    def _web_token_cb(token):
                        ui.stream_token(token)
                        ui._draw(buf=[])
                    think.chat(
                        f"Use these web search results to answer the question: {query}\n\n{results}",
                        token_callback=_web_token_cb,
                    )
                    ui.stream_commit()
                ui._draw()

            else:
                ui.add_message('sys', f'Unknown command: {user_input}')

            ui._draw()
            continue

        # ── normal turn ───────────────────────────────────────────────────────

        if args.debug:
            hits = memorize.search(user_input)
            if hits:
                ui.add_message('sys', f'{len(hits)} memories retrieved:')
                for m in hits:
                    ui.add_message('sys',
                        f'  → {m.get("memory") or m.get("text") or m}')

        current_latency = {
            "mode": "voice" if (listen and asr_enabled and voice_info) else "text",
            "submitted_at": time.monotonic(),
        }
        if voice_info:
            current_latency["listen_started_at"] = voice_info.get("listen_started_at")
            current_latency["voice_recording_stopped_at"] = voice_info.get("recording_stopped_at")

        ui.add_message('you', user_input)
        ui.turn_start()
        session_active.set()
        ui._draw()

        try:
            think.route(user_input, token_callback=token_cb)
            current_latency["assistant_done_at"] = time.monotonic()
            if speak and tts_enabled:
                speak.wait()
            ui.stream_commit()
            ui._draw()
            current_latency["turn_done_at"] = time.monotonic()
            _update_latency_stats(current_latency)
            _log_latency(current_latency)
            ui._draw()
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
    if args.cli:
        _run_cli(args)
    else:
        _run_webui(args)


if __name__ == '__main__':
    main()