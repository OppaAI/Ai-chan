"""
system/orchestrate.py

Aiko-chan's shared session orchestrator.

Both front ends (interface/cli/cli.py's run_cli and interface/webui/webui.py's
run_webui) construct a UI object implementing the AikoTUI-compatible duck-typed
API (add_message / stream_token / stream_commit / turn_start / step_loading /
step_done / step_skip / status_finish / get_input / get_voice_input /
spin_loop / _draw) and hand it to run_session(ui, args) here. Everything from
subsystem boot through the main input -> inference -> render loop is identical
regardless of transport.

Responsibilities:
    - Block (WebUI only) until the first authenticated browser session logs
      in, then boot all subsystems via system.wakeup.AikoWakeup
    - Drive the UI init phase and transition to active chat
    - Run the main input -> inference -> render loop
    - Handle commands (/quit, /reset, /memory, /clear, /remember, /think,
                       /voice, /listen, /web, /proactive, /karaoke, /help)
    - Fuzzy-match spoken voice commands to slash equivalents in ASR mode
    - Reveal the streamed reply text in sync with TTS playback (karaoke
      typewriter) instead of dumping it the instant tokens arrive
    - Track a detailed per-turn latency/debug breakdown (tokens, tok/s,
      ASR/intent/search/LLM/TTS timing) when --debug is set
    - Run the proactive idle check-in daemon
    - Clean shutdown on Ctrl-C / Ctrl-D (raised by the UI's get_input)

Boot ordering note (login-gated wakeup):
    For the WebUI path, AikoWakeup().boot() is deferred until the first
    authenticated browser session connects (see AikoWeb.wait_for_first_login()
    in interface/webui/webui.py). This guarantees system.userspace.current_user_id()
    already resolves to a real, logged-in user_id by the time AikoMemorize,
    schedule.json seeding, and the ScheduleRunner are constructed inside
    boot() — no subsystem ever touches USER_STATE_ROOT/guest/ on disk. The
    CLI path already resolves a real USER_ID via GitHub OAuth in
    interface/cli/cli.py's run_cli() before run_session() is ever called, so
    it needs no change here — run_session() only checks hasattr(ui,
    "wait_for_first_login"), which AikoSimpleCLI simply doesn't define.

Status-marker handling (fix, see interface/cli/cli.py docstring too):
    cognition/think.py streams special "__MARKER__" or "__MARKER__:payload"
    tokens (__THINKING__, __TOOL__:name(args), __SEARCHING__:query) so Aiko's
    live agentic status (thinking / tool calls / search) can be surfaced
    instead of a generic spinner. Previously these were intercepted and
    *rendered* centrally here (as ANSI-colored CLI lines), which meant (a)
    WebUI's own — already correct — tool/pose broadcast handling in
    AikoWeb.stream_token() never ran, so the browser never got live status
    updates, and (b) the /think and /web command paths call ui.stream_token()
    directly and bypassed the central interception entirely, so a marker
    fired during those commands leaked as literal text on CLI too. The fix:
    this module no longer renders markers at all. It only does latency
    bookkeeping on them, then forwards every token — marker or not — to
    ui.stream_token() unconditionally. Each UI's own stream_token() is now
    responsible for intercepting and rendering its own markers (AikoWeb
    already did this correctly; AikoSimpleCLI now does too).

Known follow-up (not fixed here, flagged): the --debug context-dump block
below still builds ANSI-colored strings (via _c()/_CTX_COLORS) and pushes
them through ui.add_message(). That's the same class of issue as the old
status-marker bug — WebUI will render raw escape codes in chat bubbles
instead of colors. Left alone for this pass since it's a larger change
(ui.add_message would need a structured/plain-text contract instead of
pre-rendered ANSI); worth a dedicated pass later.
"""
from __future__ import annotations

import collections
from datetime import datetime, timedelta
import difflib
import json
import os
import queue
import random
import re
import sys
import threading
import time

from system.bioclock import local_now
from system.log import get_logger
from system.wakeup import AikoWakeup
from agentic.toolkit.research import web_search

log = get_logger(__name__)

# ── env ───────────────────────────────────────────────────────────────────────

AI_NAME = os.getenv("AI_NAME", "Aiko")
STREAM_DRAW_INTERVAL = float(os.getenv("STREAM_DRAW_INTERVAL", "0.05"))
LATENCY_LOG_ENABLED = os.getenv("LATENCY_LOG", "1").lower() in {"1", "true", "yes", "on"}
PROACTIVE_ENABLED = os.getenv("PROACTIVE_ENABLED", "1").lower() in {"1", "true", "yes", "on"}
PROACTIVE_FIRST_IDLE_MIN_SECONDS = float(os.getenv("PROACTIVE_FIRST_IDLE_MIN_SECONDS", os.getenv("PROACTIVE_IDLE_SECONDS", "300")))
PROACTIVE_FIRST_IDLE_MAX_SECONDS = float(os.getenv("PROACTIVE_FIRST_IDLE_MAX_SECONDS", "900"))
PROACTIVE_COOLDOWN_SECONDS = float(os.getenv("PROACTIVE_COOLDOWN_SECONDS", "1800"))
PROACTIVE_MAX_PER_HOUR = int(os.getenv("PROACTIVE_MAX_PER_HOUR", "2"))
PROACTIVE_REST_AFTER_SECONDS = float(os.getenv("PROACTIVE_REST_AFTER_SECONDS", "3600"))
PROACTIVE_USE_LLM = os.getenv("PROACTIVE_USE_LLM", "1").lower() in {"1", "true", "yes", "on"}

# Karaoke typewriter — reveal streamed reply text paced to TTS playback
# instead of the instant the LLM emits tokens. See TypewriterSync below.
KARAOKE_SYNC = os.getenv("KARAOKE_SYNC", "1").lower() in {"1", "true", "yes", "on"}
KARAOKE_WPS = float(os.getenv("KARAOKE_WPS", "2.6"))  # fallback reveal pace (words/sec) when no per-chunk audio timing is available

# Debug token accounting — hits the local LLM server's /tokenize endpoint
# (llama-server compatible) for real counts; falls back to a crude
# whitespace-split estimate if that endpoint isn't reachable.
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8080/v1").rstrip("/")
if LLM_BASE_URL.endswith("/v1"):
    _TOKENIZE_BASE_URL = LLM_BASE_URL[: -len("/v1")]
else:
    _TOKENIZE_BASE_URL = LLM_BASE_URL

# ── debug ANSI colors ────────────────────────────────────────────────────────
# Kept as plain ANSI escapes. Used by the --debug context-dump block below,
# and imported by interface/cli/cli.py for its own status-marker rendering.
# Auto-disabled when stdout isn't a real terminal (e.g. piped/redirected)
# so log files and pipes don't fill up with escape codes.
_COLOR_ENABLED = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    if not _COLOR_ENABLED:
        return text
    return f"{code}{text}\033[0m"


_GREY      = "\033[38;5;250m"   # system prompt
_MED_RED   = "\033[38;5;203m"   # web prompt
_MED_CYAN  = "\033[38;5;80m"    # agentic prompts
_LAVENDER  = "\033[38;5;183m"   # memory entries
_DIM       = "\033[2m"

# ── context-type ANSI colours for compact debug ──────────────────────────
_CTX_COLORS = {
    "input":    "\033[38;5;47m",    # green
    "text":     "\033[38;5;47m",    # green
    "voice":    "\033[38;5;50m",    # teal
    "sys":      "\033[38;5;250m",   # grey
    "mem":      "\033[38;5;183m",   # lavender
    "kb":       "\033[38;5;215m",   # orange
    "knowledge":"\033[38;5;215m",   # orange (alias for kb)
    "wiki":     "\033[38;5;159m",   # light cyan
    "exp":      "\033[38;5;218m",   # pink
    "experience":"\033[38;5;218m",  # pink (alias for exp)
    "agentic":  "\033[38;5;80m",    # med cyan
    "web":      "\033[38;5;203m",   # med red
    "tools":    "\033[38;5;226m",   # yellow
    "skill":    "\033[38;5;226m",   # yellow
    "query":    "\033[38;5;154m",   # lime
    "wip":      "\033[38;5;209m",   # orange-red
    "cond":     "\033[38;5;141m",   # purple
    "summary":  "\033[38;5;117m",   # light blue
    "answer":   "\033[38;5;51m",    # bright cyan
    "output":   "\033[38;5;87m",    # light cyan
    "chat":     "\033[38;5;222m",   # tan
    "tts":      "\033[38;5;198m",   # hot pink
    "thinking": "\033[38;5;226m",   # yellow
    "plan":     "\033[38;5;141m",   # purple
    "task":     "\033[38;5;80m",    # med cyan (alias for agentic)
}


def _ctx_color(label: str) -> str:
    """Return the ANSI colour for a context label, falling back through
    partial matches (e.g. 'knowledge_context' -> 'kb')."""
    if label in _CTX_COLORS:
        return _CTX_COLORS[label]
    for key, code in _CTX_COLORS.items():
        if key in label:
            return code
    return _GREY


def _ctx_preview(content: str, max_chars: int = 500) -> str:
    """Truncate content to ~max_chars, replacing newlines with ↵."""
    if not content:
        return ""
    preview = content[:max_chars].replace("\n", "↵")
    if len(content) > max_chars:
        preview += "…"
    return preview


def _ctx_line(label: str, tok: int, latency_ms: float, content: str,
              max_chars: int = 500, max_lines: int = 5) -> str:
    """Multi-line content with subsequent lines indented under the header.

    [label][+XXms][YYY tok] first line of content
                             second line indented
                             third line indented …
    """
    color = _ctx_color(label)
    raw_lines = content.split("\n")
    out: list[str] = []
    char_count = 0
    for i, line in enumerate(raw_lines):
        if i >= max_lines:
            out[-1] += "…"
            break
        remaining = max_chars - char_count
        if remaining <= 0:
            break
        trunc = line[:remaining]
        if len(line) > remaining:
            trunc = trunc[:max(0, remaining - 1)] + "…"
        out.append(trunc)
        char_count += len(trunc) + 1

    header = f"[{label}][{latency_ms:+.0f}ms][{tok} tok]"
    indent = " " * (len(header) + 1)
    body = ("\n" + indent).join(out)
    return _c(color, f"{header} {body}")


def _gantt_lines(items: list[tuple[str, float, int, str]],
                 max_bar: int = 48,
                 total_time_ms: float | None = None) -> list[str]:
    """Dual-bar chart: left = time latency, right = token proportion + %.
    items: (label, latency_ms, tokens, colour_code)
    """
    if not items:
        return []
    total_tok = max(sum(t for _, _, t, _ in items), 1)
    max_tok   = max(t for _, _, t, _ in items)
    max_lat   = max((lat for _, lat, _, _ in items if lat > 0), default=1)
    hw = max_bar // 2  # each bar gets half the width
    out = ["── context gantt ───────────────────────────────────────"]
    for label, lat_ms, tok, color in items:
        pct = tok / total_tok * 100
        # latency bar (only shown for items with actual latency)
        lat_len = int(hw * lat_ms / max_lat) if lat_ms > 0 else 0
        lat_bar = "█" * lat_len
        # token bar (at least 1 char so every item is visible)
        tok_len = max(1, int(hw * tok / max_tok))
        tok_bar = "█" * tok_len
        out.append(_c(color,
                      f" {label:<8} {lat_bar:<{hw}} {lat_ms:+.0f}ms  "
                      f"{tok:>5}tok  {tok_bar:<{hw}} {pct:5.1f}%"))
    total_t = sum(t for _, _, t, _ in items)
    total_time_str = f" {total_time_ms:+.0f}ms" if total_time_ms is not None else ""
    out.append(_c(_DIM,
                  f" {'total':<8} {'─' * hw}{total_time_str}  "
                  f"{total_t:>5}tok  {'─' * hw} {100:5.1f}%"))
    return out


def _log_ctx(logger, label: str, tok: int, latency_ms: float,
             content: str, max_log_chars: int = 2000) -> None:
    """Log a single context entry at INFO level."""
    preview = (content[:max_log_chars].replace("\n", "\\n")
               if content else "")
    if len(content or "") > max_log_chars:
        preview += "..."
    logger.info("[ctx] [%s][%+.0fms][%d tok] %s",
                label, latency_ms, tok, preview)


def _count_tokens(text: str) -> int:
    """
    Best-effort token count via the local LLM server's /tokenize endpoint
    (llama-server compatible: POST {content: str} -> {tokens: [...]})
    Falls back to a crude whitespace-split estimate on any failure so
    debug output degrades gracefully instead of raising.
    """
    if not text:
        return 0
    try:
        import urllib.request
        req = urllib.request.Request(
            f"{_TOKENIZE_BASE_URL}/tokenize",
            data=json.dumps({"content": text}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            tokens = data.get("tokens")
            if isinstance(tokens, list):
                return len(tokens)
    except Exception:
        pass
    # crude fallback — roughly ~0.75 tokens/word for English text
    return max(1, int(len(text.split()) * 1.3))


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
    user = os.environ.get("AIKO_DISPLAY_NAME") or os.environ.get("AIKO_USER_ID") or "the user"
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
        now = local_now(PROACTIVE_TIMEZONE)
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
    now = local_now(PROACTIVE_TIMEZONE)

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

    def __init__(self, ui, speak, speak_enabled_fn, active_turn: threading.Event, generate_fn=None, on_rest_change=None) -> None:
        self._ui = ui
        self._speak = speak
        self._speak_enabled_fn = speak_enabled_fn
        self._active_turn = active_turn
        self._generate_fn = generate_fn
        self._on_rest_change = on_rest_change
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
        if self._on_rest_change:
            self._on_rest_change(False)
        self._wakeup.set()

    def set_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._enabled = enabled
            self._last_activity = time.monotonic()
            self._next_checkin_after = self._random_first_idle_delay()
            self._resting = False
        if self._on_rest_change:
            self._on_rest_change(False)
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
                if self._on_rest_change:
                    self._on_rest_change(True)
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


# ── karaoke typewriter sync ─────────────────────────────────────────────────
#
# Decouples "when the LLM emits a token" from "when the UI reveals it", so
# the streamed reply reads more like a trailing caption for the TTS audio
# instead of a spoiler that finishes long before the voice does.
#
# Two modes, auto-detected purely from what `speak` happens to expose
# (duck-typed — no changes to speak.py required to run, but speak.py can
# opt this into precise per-sentence sync later just by adding the method):
#
#   - precise: speak.set_chunk_playback_callback(fn) exists on the speak
#     object. fn(text, duration_seconds) is expected to fire the instant
#     that sentence-chunk's audio starts playing, so each word is paced by
#     duration_seconds / word_count — genuinely in sync with the voice.
#   - fallback: only speak.set_first_audio_callback exists (this is the
#     hook already wired into run_session below). The whole buffered
#     reply is released at once when audio starts, paced by KARAOKE_WPS,
#     and — importantly — any further sentences fed in AFTER first audio
#     has already fired are released to the reveal queue immediately
#     (previously they silently accumulated in the buffer forever and
#     were dropped from the UI; see feed_sentence()/on_first_audio() below).

_SENTENCE_END_RE = re.compile(r'(?<=[.!?…])\s+')


class TypewriterSync:
    """Reveal streamed reply text to the UI paced to TTS playback, not to
    LLM token arrival. See module-level comment above for the two modes.

    Markers (__THINKING__ / __TOOL__:.. / __SEARCHING__:..) never reach this
    class — orchestrate.token_cb intercepts and forwards them straight to
    ui.stream_token() before they'd ever be fed into a sentence buffer here."""

    def __init__(self, ui, speak) -> None:
        self._ui = ui
        self._speak = speak
        self._q: "queue.Queue[tuple[str, float] | None]" = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._fallback_buf: list[str] = []
        self._first_audio_fired = False
        self.precise = speak is not None and hasattr(speak, "set_chunk_playback_callback")
        if self.precise:
            speak.set_chunk_playback_callback(self._on_chunk_start)

    def start(self) -> None:
        """Begin a fresh reveal for a new turn — call right before think.route()."""
        self._stop.clear()
        self._fallback_buf = []
        self._first_audio_fired = False
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        self._thread = threading.Thread(target=self._drain, name="aiko-typewriter", daemon=True)
        self._thread.start()

    def stop(self, flush: bool = True) -> None:
        """Stop the reveal thread. If flush, dump any still-queued words to
        the UI immediately (used on interrupt/quit so nothing is silently
        swallowed); otherwise assumes the queue is already empty (normal
        end-of-turn, after speak.wait() has returned)."""
        if flush:
            if self._fallback_buf:
                leftover = " ".join(self._fallback_buf)
                self._fallback_buf = []
                for w in leftover.split():
                    self._ui.stream_token(w + " ")
            while not self._q.empty():
                try:
                    item = self._q.get_nowait()
                    if item is None:
                        continue
                    word, _ = item
                except queue.Empty:
                    break
                self._ui.stream_token(word + " ")
        self._stop.set()
        self._q.put(None)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    # ── feed path: token_cb calls this as complete sentences accumulate ────
    def feed_sentence(self, sentence: str) -> None:
        """In fallback mode: buffer the sentence until first audio starts,
        then release it (and everything fed afterward) straight to the
        reveal queue paced by KARAOKE_WPS. In precise mode this is a
        no-op — speak's own chunk callback drives the reveal instead."""
        if self.precise:
            return
        sentence = sentence.strip()
        if not sentence:
            return
        if self._first_audio_fired:
            for w in sentence.split():
                self._q.put((w, 1.0 / KARAOKE_WPS))
        else:
            self._fallback_buf.append(sentence)

    def on_first_audio(self) -> None:
        """Fallback-mode trigger: release the whole buffered reply, paced by
        KARAOKE_WPS, the instant TTS playback actually starts."""
        if self.precise:
            return
        self._first_audio_fired = True
        text = " ".join(self._fallback_buf)
        self._fallback_buf = []
        for w in text.split():
            self._q.put((w, 1.0 / KARAOKE_WPS))

    def _on_chunk_start(self, text: str, duration_s: float) -> None:
        """Precise-mode trigger: called by speak.py the instant a given
        sentence-chunk's audio begins playing."""
        words = text.split()
        if not words:
            return
        per_word = duration_s / len(words) if duration_s and duration_s > 0 else 1.0 / KARAOKE_WPS
        for w in words:
            self._q.put((w, per_word))

    def _drain(self) -> None:
        while not self._stop.is_set():
            item = self._q.get()
            if item is None:
                return
            word, delay = item
            self._ui.stream_token(word + " ")
            self._ui._draw(buf=[])
            time.sleep(delay)


# ── voice command map ─────────────────────────────────────────────────────────

_VOICE_COMMANDS: dict[str, str] = {
    "stop":              "/quit",
    "hey stop":          "/quit",
    "quit":              "/quit",
    "exit":              "/quit",
    "goodbye":           "/quit",
    "reset":             "/reset",
    "forget that":       "/reset",
    "clear context":     "/reset",
    "start over":        "/reset",
    "remember this":     "/remember",
    "pin this":          "/remember",
    "save this":         "/remember",
    "show memory":       "/memory",
    "show memories":     "/memory",
    "what do you remember": "/memory",
    "clear memory":      "/clear",
    "wipe memory":       "/clear",
    "delete memories":   "/clear",
    "mute":              "/voice",
    "unmute":            "/voice",
    "toggle voice":      "/voice",
    "stop talking":      "/voice",
    "toggle listen":     "/listen",
    "stop listening":    "/listen",
    "help":              "/help",
    "what can you do":   "/help",
}

_FILLER_RE = re.compile(
    r"^\s*(uh+|um+|ah+|okay|hey\s+aiko|aiko)[,.]?\s*",
    flags=re.IGNORECASE,
)


def _match_voice_command(text: str) -> str | None:
    """
    Fuzzy-match transcribed text against known voice commands.
    Strips leading filler words before comparing, then tries exact match
    first and falls back to difflib fuzzy matching with a 0.75 cutoff.
    """
    clean = _FILLER_RE.sub("", text.strip()).lower().rstrip(".,!?")
    if not clean:
        return None
    if clean in _VOICE_COMMANDS:
        return _VOICE_COMMANDS[clean]
    matches = difflib.get_close_matches(
        clean, _VOICE_COMMANDS.keys(), n=1, cutoff=0.75,
    )
    if matches:
        return _VOICE_COMMANDS[matches[0]]
    return None


# ── status-marker recognition ──────────────────────────────────────────────
#
# cognition/think.py streams special "__MARKER__" or "__MARKER__:payload"
# tokens to surface what Aiko is actually doing mid-turn (searching, calling
# a tool, reasoning). token_cb below uses this purely to (a) do
# latency-stage bookkeeping and (b) make sure marker tokens are forwarded
# straight to ui.stream_token() instead of ever entering the karaoke
# sentence buffer. It does NOT render them — that's each UI's own job now
# (AikoWeb.stream_token / AikoSimpleCLI.stream_token both intercept these
# same prefixes and render appropriately for their transport).
#
# NOTE: a bare "__SEARCHING__" (no colon/payload) is tracked separately below
# for latency bookkeeping only — neither UI's stream_token currently
# intercepts that exact bare form (only "__SEARCHING__:<query>" with a
# payload). If cognition/think.py actually emits a bare "__SEARCHING__" as
# its own token, it will currently leak as literal text on both front ends.
# This is a pre-existing gap (not introduced by this refactor) — worth
# checking against think.py's actual marker contract.

def _is_status_marker(stripped_token: str) -> bool:
    return (
        stripped_token == "__THINKING__"
        or stripped_token.startswith("__TOOL__:")
        or stripped_token.startswith("__SEARCHING__:")
    )


# ═════════════════════════════════════════════════════════════════════════════
# SESSION ORCHESTRATOR
# ═════════════════════════════════════════════════════════════════════════════

def run_session(ui, args) -> None:
    """Run Aiko using any UI object that implements the AikoTUI-compatible API."""
    last_stream_draw = 0.0

    # Karaoke typewriter state — created after `speak` boots below.
    typewriter: "TypewriterSync | None" = None
    _sentence_buf: list[str] = []

    def token_cb(token):
        nonlocal last_stream_draw, _sentence_buf
        now = time.monotonic()
        stripped = token.rstrip("\r\n") if token else ""

        if token:
            _latency_mark("first_token_at")

        # ── detailed stage timing (debug breakdown) ─────────────────────
        if current_latency is not None:
            if stripped == "__THINKING__":
                current_latency.setdefault("intent_start_at", now)
                current_latency["mode"] = "agentic"
            elif stripped.startswith("__TOOL__:"):
                current_latency["mode"] = "agentic"
                payload = stripped[len("__TOOL__:"):]
                current_latency["agentic_tokens"] = (
                    current_latency.get("agentic_tokens", 0) + _count_tokens(payload)
                )
            elif stripped == "__SEARCHING__":
                current_latency.setdefault("search_start_at", now)
            if token and not token.startswith("__"):
                if "intent_start_at" in current_latency and "intent_done_at" not in current_latency:
                    current_latency["intent_done_at"] = now
                if "search_start_at" in current_latency and "search_done_at" not in current_latency:
                    current_latency["search_done_at"] = now
                _latency_mark("first_assistant_token_at")

        # ── status/control markers ───────────────────────────────────────
        # Never buffered for karaoke/TTS reveal — these aren't spoken
        # content. Forward straight to the UI, which intercepts and
        # renders its own markers (see _is_status_marker docstring above).
        if _is_status_marker(stripped):
            ui.stream_token(token)
            return

        # ── karaoke reveal path ──────────────────────────────────────────
        if KARAOKE_SYNC and tts_enabled and typewriter is not None:
            _sentence_buf.append(token)
            joined = "".join(_sentence_buf)
            parts = _SENTENCE_END_RE.split(joined)
            if len(parts) > 1:
                for complete in parts[:-1]:
                    if complete.strip():
                        typewriter.feed_sentence(complete)
                _sentence_buf = [parts[-1]]
            return  # typewriter thread owns the reveal, not the raw stream

        ui.stream_token(token)
        if now - last_stream_draw >= STREAM_DRAW_INTERVAL:
            ui._draw(buf=[])
            last_stream_draw = now

    # ── login gate (WebUI only) ─────────────────────────────────────────────
    if hasattr(ui, "wait_for_first_login"):
        uid = ui.wait_for_first_login()
        if uid:
            from system.userspace import set_current_user_id, set_current_display_name
            set_current_user_id(uid)
            os.environ["AIKO_USER_ID"] = uid
            display_name = getattr(ui, "_authenticated_display_name", None) or uid
            set_current_display_name(display_name)
            os.environ["AIKO_DISPLAY_NAME"] = display_name
            log.info("First login received (user_id=%s, display=%s) — starting subsystem boot.", uid, display_name)            
        else:
            log.warning("wait_for_first_login() returned no uid — proceeding with default identity.")

    # ── init spin ─────────────────────────────────────────────────────────────

    spin_stop = threading.Event()
    spin_t    = threading.Thread(target=ui.spin_loop, args=(spin_stop,), daemon=True)
    spin_t.start()

    # ── boot all subsystems via wakeup ────────────────────────────────────────

    result = AikoWakeup(text_mode=False).boot(
        on_loading = ui.step_loading,
        on_done    = ui.step_done,
        on_skip    = ui.step_skip,
    )

    think    = result.think
    memorize = result.memorize
    speak    = result.speak
    listen   = result.listen

    if hasattr(ui, "set_voice_backends"):
        ui.set_voice_backends(speak, listen)
        
    if memorize is not None and hasattr(ui, "set_memorize"):
        ui.set_memorize(memorize)

    if memorize is None:
        ui.add_message('sys', '⚠️ Memory backend failed to load — check logs. Running without persistent memory this session.')

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

    typewriter = TypewriterSync(ui, speak) if speak is not None else None

    def _on_first_audio():
        _latency_mark("first_audio_at")
        if typewriter is not None:
            typewriter.on_first_audio()

    if speak is not None and hasattr(speak, "set_first_audio_callback"):
        speak.set_first_audio_callback(_on_first_audio)

    def _latency_seconds(timing: dict, start: str, end: str) -> float | None:
        if timing.get(start) is None or timing.get(end) is None:
            return None
        return max(0.0, timing[end] - timing[start])

    def _fmt_latency(value) -> str:
        if value is None:
            return "n/a"
        if isinstance(value, int):
            return str(value)
        return f"{value:.3f}s"

    def _latency_parts(timing: dict) -> dict:
        gen_time = _latency_seconds(timing, "first_assistant_token_at", "assistant_done_at")
        out_tok = timing.get("output_tokens")
        tok_per_sec = (out_tok / gen_time) if (out_tok and gen_time and gen_time > 0) else None
        asr_time = _latency_seconds(timing, "voice_recording_stopped_at", "asr_done_at")

        parts = {
            "voice_end_to_submit":           _latency_seconds(timing, "voice_recording_stopped_at", "submitted_at"),
            "asr_inference":                 asr_time,
            "agentic_intent":                _latency_seconds(timing, "intent_start_at", "intent_done_at"),
            "web_search":                    _latency_seconds(timing, "search_start_at", "search_done_at"),
            "submit_to_first_token":         _latency_seconds(timing, "submitted_at", "first_assistant_token_at"),
            "llm_inference":                 gen_time,
            "tts_inference":                 timing.get("tts_synth_total"),
            "submit_to_assistant_done":      _latency_seconds(timing, "submitted_at", "assistant_done_at"),
            "assistant_done_to_first_audio": _latency_seconds(timing, "assistant_done_at", "first_audio_at"),
            "voice_end_to_first_audio":      _latency_seconds(timing, "voice_recording_stopped_at", "first_audio_at"),
            "submit_to_first_audio":         _latency_seconds(timing, "submitted_at", "first_audio_at"),
            "submit_to_turn_done":           _latency_seconds(timing, "submitted_at", "turn_done_at"),
            "input_tokens":                  timing.get("input_tokens"),
            "output_tokens":                 out_tok,
            "total_tokens":                  timing.get("total_tokens"),
            "tokens_per_second":             round(tok_per_sec, 1) if tok_per_sec is not None else None,
        }

        if "system_prompt_tokens" in timing:
            mem_tok_list = timing.get("memory_entry_tokens", [])
            agentic_tok = timing.get("agentic_tokens", 0)
            out_tok_val = out_tok or 0
            web_tok = timing.get("web_prompt_tokens", 0)
            agentic_prompt_tok = timing.get("agentic_prompt_tokens", agentic_tok)
            memory_prompt_tok = timing.get("memory_prompt_tokens", sum(mem_tok_list))
            previous_chat_tok = timing.get("previous_chat_tokens", 0)
            user_turn_tok = timing.get("user_turn_tokens", 0)
            total = timing.get("total_tokens") or (
                timing["system_prompt_tokens"]
                + web_tok
                + agentic_prompt_tok
                + memory_prompt_tok
                + previous_chat_tok
                + user_turn_tok
                + out_tok_val
            )
            parts["token_breakdown"] = {
                "system_prompt_tokens": timing["system_prompt_tokens"],
                "web_prompt_tokens":    web_tok,
                "agentic_prompt_tokens": agentic_prompt_tok,
                "memory_prompt_tokens": memory_prompt_tok,
                "memory_entry_tokens":  mem_tok_list,
                "memory_entry_count":   timing.get("memory_entry_count", len(mem_tok_list)),
                "previous_chat_tokens": timing.get("previous_chat_tokens", 0),
                "previous_chat_message_count": timing.get("previous_chat_message_count", 0),
                "user_turn_tokens":     user_turn_tok,
                "output_tokens":        out_tok_val,
                "total_tokens":         total,
            }
            parts["total_tokens"] = total

        return parts

    def _update_latency_stats(timing: dict) -> None:
        if not hasattr(ui, "set_latency_stats"):
            return
        parts = _latency_parts(timing)
        if args.debug:
            display = {
                k: (_fmt_latency(v) if k not in (
                        "input_tokens", "output_tokens", "total_tokens", "tokens_per_second", "token_breakdown"
                    ) else v)
                for k, v in parts.items()
            }
            ui.set_latency_stats(display)
        else:
            ui.set_latency_stats({
                "voice_end_to_first_audio": _fmt_latency(parts["voice_end_to_first_audio"]),
            })

    def _log_latency(timing: dict) -> None:
        if not LATENCY_LOG_ENABLED:
            return
        mode = timing.get("mode", "text")
        p = _latency_parts(timing)
        log.info(
            "[latency] mode=%s in_tok=%s out_tok=%s tok/s=%s | "
            "asr=%s intent=%s search=%s llm=%s tts=%s | "
            "voice→submit=%s submit→1st_tok=%s submit→done=%s "
            "done→1st_audio=%s voice→1st_audio=%s submit→turn_done=%s",
            mode,
            p["input_tokens"] if p["input_tokens"] is not None else "n/a",
            p["output_tokens"] if p["output_tokens"] is not None else "n/a",
            p["tokens_per_second"] if p["tokens_per_second"] is not None else "n/a",
            _fmt_latency(p["asr_inference"]),
            _fmt_latency(p["agentic_intent"]),
            _fmt_latency(p["web_search"]),
            _fmt_latency(p["llm_inference"]),
            _fmt_latency(p["tts_inference"]),
            _fmt_latency(p["voice_end_to_submit"]),
            _fmt_latency(p["submit_to_first_token"]),
            _fmt_latency(p["submit_to_assistant_done"]),
            _fmt_latency(p["assistant_done_to_first_audio"]),
            _fmt_latency(p["voice_end_to_first_audio"]),
            _fmt_latency(p["submit_to_turn_done"]),
        )

    # ── shutdown helper ───────────────────────────────────────────────────────

    session_active = threading.Event()

    def _generate_proactive_checkin(prompt_hint: str) -> str:
        session_active.set()
        if listen is not None and hasattr(listen, "extend_activation"):
            listen.extend_activation()
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
        on_rest_change=think.set_proactive_resting,
    )
    proactive.start()

    def _shutdown():
        """Stop background daemons and flush memory writes before exit."""
        proactive.stop()
        if typewriter is not None:
            typewriter.stop(flush=True)
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
                if memorize is None:
                    ui.add_message('sys', 'Memory backend unavailable.')
                else:
                    all_mem = memorize.get_all()
                    if not all_mem:
                        ui.add_message('sys', 'No memories stored yet.')
                    else:
                        ui.add_message('sys', f'{len(all_mem)} memories stored:')
                        for i, m in enumerate(all_mem, 1):
                            ui.add_message('sys',
                                f'  {i:02d}. {m.get("memory") or m.get("text") or m}')

            elif cmd == '/clear':
                if memorize is None:
                    ui.add_message('sys', 'Memory backend unavailable.')
                else:
                    memorize.clear()
                    ui.add_message('sys', 'All persistent memories cleared.')

            elif cmd == '/remember':
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

            elif cmd == '/karaoke':
                globals()['KARAOKE_SYNC'] = not KARAOKE_SYNC
                ui.add_message('sys',
                    f'Karaoke text sync: {"ON  🎤📝" if KARAOKE_SYNC else "OFF (instant stream)"}')

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
                    '/karaoke                 — toggle text-sync-to-voice reveal on/off',
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

        current_latency = {
            "mode": "voice" if (listen and asr_enabled and voice_info) else "text",
            "submitted_at": time.monotonic(),
        }
        if voice_info:
            current_latency["listen_started_at"] = voice_info.get("listen_started_at")
            current_latency["voice_recording_stopped_at"] = voice_info.get("recording_stopped_at")
            if voice_info.get("asr_done_at") is not None:
                current_latency["asr_done_at"] = voice_info["asr_done_at"]

        if args.debug:
            system_prompt = getattr(think, "_persona", None) or ""
            current_latency["system_prompt_tokens"] = _count_tokens(system_prompt)

        ui.add_message('you', user_input)
        ui.turn_start()
        session_active.set()
        ui._draw()

        if typewriter is not None:
            typewriter.start()
            _sentence_buf = []

        if speak is not None and hasattr(speak, "reset_synth_timer"):
            speak.reset_synth_timer()

        try:
            think.route(user_input, token_callback=token_cb)
            current_latency["assistant_done_at"] = time.monotonic()

            usage = getattr(think, "last_usage", None) or {}
            if usage.get("prompt_tokens") is not None:
                current_latency["input_tokens"] = usage["prompt_tokens"]
            if usage.get("completion_tokens") is not None:
                current_latency["output_tokens"] = usage["completion_tokens"]
            if usage.get("total_tokens") is not None:
                current_latency["total_tokens"] = usage["total_tokens"]

            if args.debug:
                prompt_debug = getattr(think, "last_prompt_debug", None) or {}
                system_prompt = prompt_debug.get("system_prompt") or system_prompt or ""
                web_prompt = prompt_debug.get("web_prompt") or ""
                memory_prompt = prompt_debug.get("memory_prompt") or ""
                agentic_prompts = prompt_debug.get("agentic_prompts") or []
                previous_chat_messages = prompt_debug.get("previous_chat_messages") or []
                knowledge_prompt = prompt_debug.get("knowledge_prompt") or ""

                previous_chat_texts = [
                    f"{m.get('role', 'unknown')}: {m.get('content', '')}"
                    for m in previous_chat_messages
                    if isinstance(m, dict)
                ]
                prompt_messages = usage.get("prompt_messages") or []
                completion_text = usage.get("completion_text") or ""

                ctx_entries: list[tuple[str, float, int, str]] = []

                sys_tok = _count_tokens(system_prompt)
                ctx_entries.append(("sys", 0.0, sys_tok, system_prompt))

                web_tok = _count_tokens(web_prompt)
                if web_prompt:
                    web_lat = _latency_seconds(current_latency, "search_start_at", "search_done_at")
                    ctx_entries.append(("web", (web_lat or 0) * 1000, web_tok, web_prompt))

                mem_tok = _count_tokens(memory_prompt)
                if memory_prompt:
                    ctx_entries.append(("mem", 0.0, mem_tok, memory_prompt))

                kb_tok = _count_tokens(knowledge_prompt)
                if knowledge_prompt:
                    ctx_entries.append(("kb", 0.0, kb_tok, knowledge_prompt))

                agentic_tok = 0
                for item in agentic_prompts:
                    lbl = item.get("label", "agentic") if isinstance(item, dict) else "agentic"
                    content = item.get("content", "") if isinstance(item, dict) else str(item)
                    if content:
                        tok = _count_tokens(content)
                        agentic_tok += tok
                        ctx_entries.append((lbl, 0.0, tok, content))

                chat_tok = sum(_count_tokens(t) for t in previous_chat_texts)
                if chat_tok > 0:
                    chat_content = "\n".join(previous_chat_texts)
                    ctx_entries.append(("chat", 0.0, chat_tok, chat_content))

                user_tok = _count_tokens(user_input)
                ctx_entries.append(("input", 0.0, user_tok, user_input))

                out_tok = current_latency.get("output_tokens") or (len(completion_text.split()) if completion_text else 0) or _count_tokens(completion_text)
                llm_lat = _latency_seconds(current_latency, "first_assistant_token_at", "assistant_done_at")
                if completion_text:
                    ctx_entries.append(("output", (llm_lat or 0) * 1000, out_tok, completion_text))

                def _ctx_group(l: str) -> str:
                    for prefix, grp in (("sys","ctx"),("mem","ctx"),("kb","ctx"),
                                        ("wiki","ctx"),("knowledge","ctx"),
                                        ("exp","ctx"),("experience","ctx"),
                                        ("agentic","inst"),("tool","inst"),
                                        ("skill","inst"),("task","inst"),
                                        ("web","web"),
                                        ("chat","hist")):
                        if l.startswith(prefix) or prefix in l:
                            return grp
                    return "turn"
                _GROUP_LABEL = {"ctx":"context", "inst":"instructions",
                                "web":"web results", "hist":"history",
                                "turn":"current turn"}
                _last_group = ""
                for label, lat_ms, tok, content in ctx_entries:
                    if not content:
                        continue
                    grp = _ctx_group(label)
                    if grp != _last_group:
                        _last_group = grp
                        ui.add_message('sys',
                                       _c(_DIM, f"── {_GROUP_LABEL.get(grp, grp)} ──"))
                    ui.add_message('sys', _ctx_line(label, tok, lat_ms, content))

                current_latency["system_prompt_tokens"] = sys_tok
                current_latency["web_prompt_tokens"] = web_tok
                current_latency["memory_prompt_tokens"] = mem_tok
                current_latency["agentic_prompt_tokens"] = agentic_tok
                current_latency["previous_chat_tokens"] = chat_tok
                current_latency["previous_chat_message_count"] = len(previous_chat_texts)
                current_latency["user_turn_tokens"] = user_tok

                if current_latency.get("input_tokens") is None and prompt_messages:
                    current_latency["input_tokens"] = sum(
                        _count_tokens(str(m.get("content", "")))
                        for m in prompt_messages
                        if isinstance(m, dict)
                    )
                if current_latency.get("output_tokens") is None and completion_text:
                    current_latency["output_tokens"] = _count_tokens(completion_text)
                if current_latency.get("total_tokens") is None:
                    in_tok_val = current_latency.get("input_tokens") or 0
                    out_tok_val = current_latency.get("output_tokens") or 0
                    current_latency["total_tokens"] = in_tok_val + out_tok_val

                gantt_items: list[tuple[str, float, int, str]] = []
                for label, lat_ms, tok, content in ctx_entries:
                    if tok > 0:
                        gantt_items.append((label, lat_ms, tok, _ctx_color(label)))
                turn_total_ms = _latency_seconds(current_latency, "submitted_at", "turn_done_at")
                turn_total_ms = (turn_total_ms * 1000) if turn_total_ms is not None else None
                for g_line in _gantt_lines(gantt_items, total_time_ms=turn_total_ms):
                    ui.add_message('sys', g_line)
                t_to_a = _latency_seconds(current_latency, "submitted_at", "first_audio_at")
                if t_to_a is not None:
                    ui.add_message('sys', _c(_DIM, f"  T→A (submit → first audio): {_fmt_latency(t_to_a)}"))

                _log_ctx(log, "turn", 0, 0, f"mode={current_latency.get('mode','?')} "
                         f"user={user_input[:200]}")
                for label, lat_ms, tok, content in ctx_entries:
                    _log_ctx(log, label, tok, lat_ms, content)
                log.info("[ctx] gantt: %s",
                         " | ".join(f"{l}={lat:.0f}ms/{t}tok"
                                    for l, lat, t, _ in gantt_items))

            if typewriter is not None and _sentence_buf:
                typewriter.feed_sentence("".join(_sentence_buf))
                _sentence_buf = []

            if speak and tts_enabled:
                speak.wait()

            if speak is not None and hasattr(speak, "pop_synth_time"):
                current_latency["tts_synth_total"] = speak.pop_synth_time()

            if typewriter is not None:
                typewriter.stop(flush=True)

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
