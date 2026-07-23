"""
interface/cli/cli.py

Aiko-chan's plain no-curses CLI front end (testing only — no layout, just
plain scrolling stdout). The WebUI (interface/webui/webui.py) is the default
front end for real use; run_cli() here is invoked from main.py via --cli.

Implements the same duck-typed "ui" interface system.orchestrate.run_session()
expects (add_message, turn_start, stream_token, stream_commit, get_input,
get_voice_input, _draw, _stats, boot callbacks) but renders everything as
plain scrolling stdout lines.

Status-marker rendering (fix):
    cognition/think.py streams "__THINKING__" / "__TOOL__:name(args)" /
    "__SEARCHING__:query" control tokens mid-turn. These used to be
    intercepted and rendered by a module-level helper in the old main.py
    that only the shared main-loop token callback called — meaning the
    /think and /web commands' own token callbacks (which call
    ui.stream_token() directly) bypassed it and would print raw marker
    text inline. Each UI is now responsible for intercepting its own
    markers inside stream_token() itself, so every call site — main loop,
    /think, /web — gets correct rendering automatically. See
    system/orchestrate.py's module docstring for the WebUI side of this
    same fix.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

from system.log import get_logger
from system.orchestrate import _c, _CTX_COLORS, _ctx_preview
from system.wakeup import AikoWakeup

log = get_logger(__name__)

# CLI auth (GitHub device flow) — lazy-imported so it doesn't pull in
# httpx for users who never use --cli
_CliAuth = None
def _get_cli_auth():
    global _CliAuth
    if _CliAuth is None:
        from interface.cli.auth import CliAuth
        _CliAuth = CliAuth()
    return _CliAuth


def handle_logout() -> None:
    """Clear stored CLI (GitHub OAuth) auth token. Called from main.py's --logout."""
    _get_cli_auth().logout()


_CLI_ROLE_PREFIX = {
    "you":  "You",
    "aiko": "Aiko",
    "sys":  "·",
}


class AikoSimpleCLI:
    def __init__(self, no_voice: bool = False, debug: bool = False) -> None:
        self.no_voice = no_voice
        self.debug = debug
        self._chat_started = False
        self._stats: dict = {"tts_on": not no_voice, "asr_on": not no_voice}
        self._streaming = False
        self._stream_buf: list[str] = []
        self._latency_stats: dict = {}
        self._boot_total = len(AikoWakeup.ALL_BOOT_LABELS)
        self._boot_done = 0
        self._boot_current = ""
        self._boot_lock = threading.Lock()
        # per-instance agentic step counter for __THINKING__ rendering
        # (previously a module global in main.py — instance state is
        # cleaner and doesn't leak across AikoSimpleCLI instances)
        self._agent_step = 0

    # ── boot / status ────────────────────────────────────────────────────
    def spin_loop(self, stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            stop_event.wait(0.25)

    def step_loading(self, key: str) -> None:
        if self._chat_started:
            return
        label = AikoWakeup.ALL_BOOT_LABELS.get(key, key)
        with self._boot_lock:
            self._boot_current = label
            self._render_progress()

    def step_done(self, key: str) -> None:
        if self._chat_started:
            return
        with self._boot_lock:
            self._boot_done += 1
            self._render_progress()

    def step_skip(self, key: str) -> None:
        if self._chat_started:
            return
        with self._boot_lock:
            self._boot_done += 1
            self._render_progress()

    def _render_progress(self) -> None:
        pct = int(100 * self._boot_done / (self._boot_total or 1))
        print(f"\r  [{pct:3d}%] {self._boot_current:<50}", end="", flush=True)

    def status_finish(self) -> None:
        self._chat_started = True
        print("\r" + " " * 70 + "\r", end="", flush=True)
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
        """
        Forward a streaming token to stdout.

        Intercepts agentic status/control markers (__THINKING__,
        __TOOL__:name(args), __SEARCHING__:query) and renders them as
        colored 'sys' lines instead of letting the raw marker text leak
        into the streamed reply — mirrors AikoWeb.stream_token()'s own
        marker handling for the WebUI front end. See module docstring.
        """
        if not token:
            return
        stripped = token.rstrip("\r\n")

        if stripped == "__THINKING__":
            self._agent_step += 1
            self.add_message('sys', _c(_CTX_COLORS["thinking"], f"[thinking] step {self._agent_step}"))
            return

        if stripped.startswith("__SEARCHING__:"):
            query = stripped[len("__SEARCHING__:"):].strip()
            self.add_message('sys', _c(_CTX_COLORS["web"], f"[web] {_ctx_preview(query, 300)}"))
            return

        if stripped.startswith("__TOOL__:"):
            payload = stripped[len("__TOOL__:"):].strip()
            if "(" in payload and payload.endswith(")"):
                name = payload[:payload.index("(")]
                args_raw = payload[payload.index("(") + 1:-1]
                display = name
                if args_raw and args_raw.strip("{} "):
                    display = f"{name}({_ctx_preview(args_raw, 200)})"
            else:
                display = _ctx_preview(payload, 300)
            self.add_message('sys', _c(_CTX_COLORS["tools"], f"[tools] {display}"))
            return

        self._stream_buf.append(token)
        print(token, end="", flush=True)

    def stream_commit(self) -> None:
        if self._streaming:
            print()  # newline after streamed reply
        self._streaming = False
        self._stream_buf = []

    def set_latency_stats(self, stats: dict) -> None:
        """
        Per-turn debug / latency summary.
        Detailed context entries are already shown inline via add_message();
        this just adds a one-line latency summary and the V→A voice metric.
        """
        self._latency_stats = stats
        v_to_a = stats.get("voice_end_to_first_audio")
        if v_to_a and v_to_a != "n/a":
            print(f"  ⏱  V→A (voice end → Aiko's first audio): {v_to_a}")
        if not self.debug:
            return
        tok_s = stats.get("tokens_per_second", "n/a")
        in_t = stats.get("input_tokens", "n/a")
        out_t = stats.get("output_tokens", "n/a")
        total = stats.get("total_tokens", "n/a")
        gen = stats.get("llm_inference", "n/a")
        asr = stats.get("asr_inference", "n/a")
        intent = stats.get("agentic_intent", "n/a")
        search = stats.get("web_search", "n/a")
        print(f"  ⚡ {tok_s} tok/s  |  in/out/total: {in_t}/{out_t}/{total}  "
              f"|  asr={asr} intent={intent} search={search} llm={gen}")

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
        run_session() can compute V->A latency. `recording_stopped_at` here
        is approximate — it's stamped when the blocking listen call returns,
        which includes ASR transcription time, not just end-of-speech. The
        WebUI's real pipeline stamps this earlier (right at end-of-speech),
        so CLI latency numbers will read a bit higher than the WebUI's.

        `asr_done_at` is also stamped here (right after the blocking call
        returns) so main.py can report an ASR-only inference time, even
        though for this CLI backend it will read the same as
        recording_stopped_at above — the WebUI path can eventually pass a
        tighter one from the real ASR component if it starts separating
        end-of-speech from transcription-complete.
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
                asr_done_at = recording_stopped_at
                text = (result[0] if isinstance(result, tuple) else result) or ""
                text = text.strip()
                if text:
                    print(f"You (voice): {text}")
                return text, {
                    "listen_started_at": listen_started_at,
                    "recording_stopped_at": recording_stopped_at,
                    "asr_done_at": asr_done_at,
                }
        # No compatible voice API found on this backend — fall back to typed input.
        return self.get_input()


# ═════════════════════════════════════════════════════════════════════════════
# CLI-only auth / identity helpers
# ═════════════════════════════════════════════════════════════════════════════

_CLI_NAME_FILE: Path | None = None


def _cli_name_path() -> Path | None:
    global _CLI_NAME_FILE
    if _CLI_NAME_FILE is not None:
        return _CLI_NAME_FILE
    try:
        from system.userspace import user_state_dir
        _CLI_NAME_FILE = user_state_dir() / "cli_name.txt"
    except Exception:
        _CLI_NAME_FILE = Path.home() / ".aiko_cli_name.txt"
    return _CLI_NAME_FILE


def _cli_display_name(args) -> str:
    """Resolve the CLI display name from --name, stored name, or prompt."""
    if args.name:
        return args.name.strip()
    name_path = _cli_name_path()
    if name_path and name_path.exists():
        stored = name_path.read_text(encoding="utf-8").strip()
        if stored:
            return stored
    try:
        raw = input("  What's your name? ").strip()
        if raw:
            if name_path:
                name_path.parent.mkdir(parents=True, exist_ok=True)
                name_path.write_text(raw, encoding="utf-8")
            return raw
    except (EOFError, KeyboardInterrupt):
        pass
    return "guest"


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT (called from main.py)
# ═════════════════════════════════════════════════════════════════════════════

def run_cli(args) -> None:
    """Launch Aiko with the plain no-curses CLI (testing only).
    Enforces GitHub OAuth if GITHUB_CLIENT_ID is set in .env."""
    from system.orchestrate import run_session

    cli_auth = _get_cli_auth()
    if cli_auth.is_configured():
        if not cli_auth.is_authenticated():
            print("  GitHub OAuth is required for CLI access.")
            if not cli_auth.login():
                print("  Authentication failed — exiting.")
                sys.exit(1)
        gh_user = cli_auth.get_user_id()
        os.environ["AIKO_USER_ID"] = gh_user
        os.environ["AIKO_DISPLAY_NAME"] = cli_auth.get_display_name()
        from system.userspace import set_current_display_name
        set_current_display_name(cli_auth.get_display_name())
        log.info("CLI session user_id=%s display=%s", gh_user, cli_auth.get_display_name())
    else:
        display_name = _cli_display_name(args)
        os.environ["AIKO_DISPLAY_NAME"] = display_name
        from system.userspace import set_current_display_name
        set_current_display_name(display_name)
        log.info("CLI session display=%s", display_name)

    ui = AikoSimpleCLI(no_voice=args.text, debug=args.debug)
    run_session(ui, args)