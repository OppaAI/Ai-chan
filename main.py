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
from dotenv import load_dotenv
load_dotenv()

import argparse
import difflib
import os
import re
import sys
import threading
import time
import warnings

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
from webui.aiko_web import AikoWeb

# ── env ───────────────────────────────────────────────────────────────────────

AI_NAME = os.getenv("AI_NAME", "Aiko")
USER_ID = os.getenv("USER_ID", "")
STREAM_DRAW_INTERVAL = float(os.getenv("AIKO_STREAM_DRAW_INTERVAL", "0.05"))
LATENCY_LOG_ENABLED = os.getenv("AIKO_LATENCY_LOG", "1").lower() in {"1", "true", "yes", "on"}


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
    print(f"\n  🌸 Aiko-chan is ready → http://{host_ip}:{8787}/\n")
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
        if os.getenv("AIKO_WEBUI_LOCAL_PLAYBACK", "1").lower() in {"0", "false", "no", "off"}:
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

    def _shutdown():
        """Stop background daemons and flush memory writes before exit."""
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
        tui._draw()

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
