"""
webui/webui.py
Aiko-chan's browser-based UI backend — drop-in replacement for AikoTUI.

Responsibilities:
    - Serve aiko.html + assets from webui/static/ over HTTP(S) (localhost:HTTP_PORT)
    - Host a bidirectional WebSocket endpoint multiplexed on the same
      HTTP(S) port via FastAPI/uvicorn (no separate WS listener/port)
    - Expose the same public API as AikoTUI so main.py needs minimal changes:
        add_message / stream_token / stream_commit / turn_start
        step_loading / step_done / step_skip / step_error / status_finish
        get_input / get_voice_input / spin_loop
    - Block get_input() until the browser sends {"type":"user_input","text":"..."}
    - Auto-open the browser on first connection
    - Expose run_webui(args): the WebUI launcher, called from main.py

Environment variables (all optional):
    HTTP_PORT   — HTTP(S)/WS port for serving the UI (default 8787)
    NO_BROWSER  — set to "1" to suppress auto-open
    WEBUI_HTTPS — set to "1" to serve HTTPS/WSS for remote browser microphones
    WEBUI_BROWSER_VAD_GATE — set to "0" to stream raw WebUI PCM for VAD diagnostics
    SSL_CERT    — optional TLS certificate path
    SSL_KEY     — optional TLS private key path

Boot ordering note (login-gated wakeup):
    AikoWakeup().boot() — which constructs AikoMemorize, seeds schedule.json
    jobs, and starts the global ScheduleRunner — must never run before a real
    user is authenticated, since every one of those subsystems resolves paths
    via system.userspace.current_user_id(). This class exposes
    wait_for_first_login() so system.orchestrate.run_session() can block until
    the first authenticated WebSocket session connects (see _ws_handler below)
    before calling AikoWakeup().boot(). The HTTP server (login page, OAuth
    flow, static assets) is already serving at this point — only the heavy
    model/subsystem boot is deferred.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import ssl
import subprocess
import threading
import time
import webbrowser
from pathlib import Path

from system.config import load_config
from system.userspace import reset_current_user_id, set_current_user_id, set_current_display_name
load_config()

from system import bioclock

log = logging.getLogger(__name__)

# ── config ────────────────────────────────────────────────────────────────────

HTTP_PORT  = int(os.getenv("HTTP_PORT", "8787"))
STATIC_DIR = Path(__file__).parent / "static"
NO_BROWSER = os.getenv("NO_BROWSER", "0") == "1"
WEBUI_HTTPS = os.getenv("WEBUI_HTTPS", "0").lower() in {"1", "true", "yes", "on"}
SSL_CERT = os.getenv("SSL_CERT", "")
SSL_KEY = os.getenv("SSL_KEY", "")
WEBUI_BROWSER_VAD_GATE = os.getenv("WEBUI_BROWSER_VAD_GATE", "1").lower() in {"1", "true", "yes", "on"}


def _load_stored_display_name(uid: str) -> str:
    """Return a stored display name from the user's state dir, if any."""
    try:
        from system.userspace import user_state_dir
        name_file = user_state_dir(uid) / "cli_name.txt"
        if name_file.exists():
            stored = name_file.read_text(encoding="utf-8").strip()
            if stored:
                return stored
    except Exception:
        pass
    return ""


def _make_ssl_context(hostname: str, host_ip: str) -> ssl.SSLContext | None:
    """Return a server TLS context when WEBUI_HTTPS is enabled."""
    if not WEBUI_HTTPS:
        return None

    cert_path = Path(SSL_CERT) if SSL_CERT else Path(__file__).parent / ".cert" / "webui.crt"
    key_path = Path(SSL_KEY) if SSL_KEY else Path(__file__).parent / ".cert" / "webui.key"

    if not cert_path.exists() or not key_path.exists():
        cert_path.parent.mkdir(parents=True, exist_ok=True)
        key_path.parent.mkdir(parents=True, exist_ok=True)
        alt_names = ["DNS:localhost", f"DNS:{hostname}", "IP:127.0.0.1"]
        if host_ip and host_ip != "127.0.0.1":
            alt_names.append(f"IP:{host_ip}")
        try:
            subprocess.run(
                [
                    "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
                    "-keyout", str(key_path),
                    "-out", str(cert_path),
                    "-days", "3650",
                    "-subj", f"/CN={hostname}",
                    "-addext", f"subjectAltName={','.join(alt_names)}",
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            log.info("[aiko-web] generated self-signed TLS cert at %s", cert_path)
        except Exception as exc:
            raise RuntimeError(
                "WEBUI_HTTPS=1 requires openssl or SSL_CERT/SSL_KEY pointing at an existing certificate."
            ) from exc

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(certfile=cert_path, keyfile=key_path)
    ctx.set_alpn_protocols(["http/1.1"])
    return ctx

# ── message types (server → browser) ─────────────────────────────────────────
#
# chat      {"type":"chat",    "sender":"you"|"aiko"|"sys", "text":"..."}
# token     {"type":"token",   "text":"..."}
# commit    {"type":"commit"}
# step      {"type":"step",    "key":"...", "state":"loading"|"done"|"skip"|"error", "detail":"..."}
# phase     {"type":"phase",   "value":"init"|"chat"}
# vitals    {"type":"vitals",  "tokens":0, "tok_s":0.0, "ram":"...", "uptime":"...", "asr":true, "tts":true}
# voice     {"type":"voice",   "status":"waiting"|"listening"|"transcribing"|"idle"}
# mic       {"type":"mic",     "action":"start"|"stop", "bytes_per_chunk":2048}
# tool      {"type":"tool",    "status":"..."|null}
# expression{"type":"expression","name":"...","intensity":0.8}
# viseme    {"type":"viseme",  "viseme":"A","weight":0.6}
# pose      {"type":"pose",    "name":"thinking", "active":true}
#
# browser → server:
# user_input{"type":"user_input","text":"..."}
# vad       {"type":"vad",     "event":"start"|"end"}   ← browser VAD sentinels


class AikoWeb:
    """
    Browser-based drop-in for AikoTUI.

    All drawing methods become JSON WebSocket broadcasts.
    get_input() blocks on a threading.Queue until the browser submits a message.

    Browser VAD gates WebUI microphone audio by default so silence/private
    background audio is not streamed to the server. For diagnostics, set
    WEBUI_BROWSER_VAD_GATE=0 to stream raw PCM and let server-side VAD segment it.

    NOTE: interface.webui.auth.aiko_web_instance is set to `self` at the end
    of __init__ as a module-global singleton so auth.py's request handlers
    can reach the live UI instance. This means only one AikoWeb can exist per
    process — fine for this app's single-session model, but worth knowing if
    that ever changes.
    """

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    def __init__(self, no_voice: bool = False, debug: bool = False):
        self._no_voice = no_voice
        self._debug    = debug
        self._ts       = time.time()
        self._lock     = threading.Lock()

        # current authenticated user — set by _ws_handler, consumed by get_input()
        self._current_user_id: str = "guest"
        self._current_display_name: str = "Guest"

        # login gate — set the first time a real authenticated WS session
        # connects, so main.py can defer AikoWakeup().boot() until a real
        # (non-guest) user_id is known. See wait_for_first_login() below.
        self._login_event = threading.Event()
        self._authenticated_uid: str | None = None
        self._authenticated_display_name: str | None = None

        # input queue — browser posts here, get_input() reads here
        self._input_q: queue.Queue[str] = queue.Queue()

        # binary mic-audio frames from the browser, consumed by get_voice_input()
        # an empty-bytes sentinel (b"") signals end-of-utterance from browser VAD
        # maxsize=10000 (~20 seconds at 512 frames/s) prevents unbounded growth
        # since the mic stays open continuously for instant barge-in
        self._audio_q: queue.Queue[bytes] = queue.Queue(maxsize=10000)
        self._mic_active = threading.Event()
        self._mic_started: bool = False
        self._did_barge_in: bool = False

        # connected browser websocket clients
        self._clients: set = set()
        self._clients_lock = threading.Lock()

        # memory backend (injected after boot by run_session)
        self._memorize = None

        # streaming state
        self._streaming   = ""
        self._tool_status = None

        # session stats
        self._stats: dict = {
            "tokens":     0,
            "turn_tok":   0,
            "turn_start": None,
            "tok_s":      0.0,
            "asr_on":     not no_voice,
            "tts_on":     not no_voice,
        }

        # asyncio event loop running in a background thread
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_ready = threading.Event()
        self._ssl_context: ssl.SSLContext | None = None

        # Register instance for WebSocket routing
        import interface.webui.auth
        interface.webui.auth.aiko_web_instance = self

        self._start_servers()

    def set_voice_backends(self, speak, listen) -> None:
        """Inject speak/listen (called from run_session after boot) so
        _ws_handler can act on a browser-reported barge_in message."""
        self._speak = speak
        self._listen = listen
    
    def set_memorize(self, memorize) -> None:
        """Inject the memory backend (called from run_session after boot)."""
        self._memorize = memorize

    def wait_for_first_login(self, timeout: float | None = None) -> str | None:
        """Block the calling thread until the first authenticated WebSocket
        session connects, then return that session's real user_id.

        Used by system.orchestrate.run_session() to defer AikoWakeup().boot()
        (and therefore AikoMemorize construction, schedule.json seeding, and
        the ScheduleRunner) until a genuine logged-in user_id is known — never
        the "guest" default. The HTTP server (login page, OAuth flow, static
        assets) is already reachable while this blocks; only the heavy
        subsystem boot is gated.

        Returns None if `timeout` is given and no login arrives in time.
        """
        self._login_event.wait(timeout)
        return self._authenticated_uid

    # ------------------------------------------------------------------
    # server lifecycle
    # ------------------------------------------------------------------

    def _start_servers(self) -> None:
        """Spin up the HTTP / WebSocket server using FastAPI & uvicorn."""
        import socket
        hostname = socket.gethostname()
        host_ip = socket.gethostbyname(hostname)
        self._ssl_context = _make_ssl_context(hostname, host_ip)
        scheme = "https" if self._ssl_context else "http"

        # Mount static files to auth app dynamically
        from interface.webui.auth import app as auth_app
        from fastapi.staticfiles import StaticFiles

        has_static = False
        for route in auth_app.routes:
            if hasattr(route, "name") and route.name == "static":
                has_static = True
                break
        if not has_static:
            auth_app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

        http_t = threading.Thread(target=self._run_http, daemon=True, name="aiko-http")
        http_t.start()

        self._loop_ready.wait(timeout=5)

        if not NO_BROWSER:
            threading.Timer(0.6, lambda: webbrowser.open(f"{scheme}://{host_ip}:{HTTP_PORT}/")).start()

    def _run_http(self) -> None:
        """Serve the FastAPI app over HTTP or HTTPS via uvicorn."""
        import uvicorn
        from interface.webui.auth import app as auth_app

        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        cert_path = Path(SSL_CERT) if SSL_CERT else Path(__file__).parent / ".cert" / "webui.crt"
        key_path = Path(SSL_KEY) if SSL_KEY else Path(__file__).parent / ".cert" / "webui.key"

        config = uvicorn.Config(
            auth_app,
            host="0.0.0.0",
            port=HTTP_PORT,
            ssl_keyfile=str(key_path) if self._ssl_context else None,
            ssl_certfile=str(cert_path) if self._ssl_context else None,
            log_level="warning",
            loop="asyncio"
        )
        server = uvicorn.Server(config)
        self._loop_ready.set()
        self._loop.run_until_complete(server.serve())

    async def _ws_handler(self, ws) -> None:
        """Handle one browser WebSocket connection via FastAPI WebSocket."""
        # 1. Enforce session authentication (Auth is mandatory)
        from interface.webui.auth import sessions, signer, SESSION_MAX_AGE_SECONDS
        from itsdangerous import BadSignature, SignatureExpired
        from datetime import datetime, timedelta

        cookie_value = ws.cookies.get("session_id")
        if not cookie_value:
            log.warning("[aiko-web] unauthenticated WebSocket connection attempt")
            await ws.close(code=1008)
            return

        try:
            session_id = signer.loads(cookie_value, max_age=SESSION_MAX_AGE_SECONDS)
        except (BadSignature, SignatureExpired):
            log.warning("[aiko-web] WebSocket connection with invalid/expired session cookie")
            await ws.close(code=1008)
            return

        if session_id not in sessions:
            log.warning("[aiko-web] unauthenticated WebSocket connection attempt")
            await ws.close(code=1008)
            return

        session = sessions[session_id]
        if bioclock.local_now() - session["created_at"] > timedelta(days=30):
            log.warning("[aiko-web] expired WebSocket session")
            await ws.close(code=1008)
            return

        uid = str(session["user_id"])

        # Login gate — record the first real authenticated uid so
        # run_session()'s wait_for_first_login() can unblock and start
        # AikoWakeup().boot() with a genuine user_id already in place. Safe
        # to check/set on every connection; only the first one matters
        # (Event.set() is idempotent, and later re-logins/reconnects don't
        # rewind boot).
        self._current_user_id = uid

        # Display name: stored name file > session username (GitHub login) > uid
        stored_name = _load_stored_display_name(uid)
        session_name = (session.get("username") or "")
        self._current_display_name = stored_name or session_name or uid

        if not self._login_event.is_set():
            self._authenticated_uid = uid
            self._authenticated_display_name = self._current_display_name
            self._login_event.set()
        user_context_token = set_current_user_id(uid)
        set_current_display_name(self._current_display_name)
        os.environ["AIKO_USER_ID"] = uid
        os.environ["AIKO_DISPLAY_NAME"] = self._current_display_name
        if self._memorize:
            self._memorize.switch_user(uid)
            self._memorize.set_display_name(self._current_display_name)
            self._memorize.set_display_name(self._current_display_name)
        await ws.accept()

        with self._clients_lock:
            self._clients.add(ws)
        log.info("[aiko-web] browser connected  (total=%d)", len(self._clients))
        try:
            while True:
                message = await ws.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                raw_bytes = message.get("bytes")
                if raw_bytes is not None:
                    # browser mic PCM frame — only buffer during an active voice
                    # turn so stale/late frames don't pollute the next recording.
                    # Mic stays open continuously (for instant barge-in), so
                    # drop frames if queue is full to avoid blocking the handler.
                    if self._mic_active.is_set():
                        try:
                            self._audio_q.put_nowait(raw_bytes)
                        except queue.Full:
                            pass
                    continue

                raw_text = message.get("text")
                if raw_text is not None:
                    try:
                        msg = json.loads(raw_text)
                    except json.JSONDecodeError:
                        continue

                    mtype = msg.get("type")

                    if mtype == "user_input":
                        text = (msg.get("text") or "").strip()
                        if text:
                            uid = str(session["user_id"])
                            self._current_user_id = uid
                            stored_name = _load_stored_display_name(uid)
                            session_name = (session.get("username") or "")
                            self._current_display_name = stored_name or session_name or uid
                            set_current_user_id(uid)
                            set_current_display_name(self._current_display_name)
                            os.environ["AIKO_USER_ID"] = uid
                            os.environ["AIKO_DISPLAY_NAME"] = self._current_display_name
                            if self._memorize:
                                self._memorize.switch_user(uid)
                            self._input_q.put(text)

                    elif mtype == "vad":
                        # browser energy VAD sentinels — update voice status display
                        # and inject end-of-utterance sentinel into the audio queue
                        event = msg.get("event")
                        if event == "start":
                            # speech onset — update UI; listen.py will see audio frames arriving
                            self._broadcast({"type": "voice", "status": "listening"})
                        elif event == "end":
                            # speech ended — push empty-bytes sentinel so _chunk_source
                            # returns None cleanly, ending the recording loop in listen.py
                            self._broadcast({"type": "voice", "status": "transcribing"})
                            if WEBUI_BROWSER_VAD_GATE and self._mic_active.is_set():
                                self._audio_q.put(b"")  # end-of-utterance sentinel
                                
                    elif mtype == "barge_in":
                        # Browser's client-side energy VAD detected speech
                        # while TTS was playing. Browser already stopped its
                        # own playback locally (zero round-trip) — this stops
                        # backend generation/synthesis so it doesn't keep
                        # producing audio that would just refill the (already
                        # cleared) client queue, and unblocks listen.py's
                        # wait_or_barge_in() so the next turn can start.
                        #
                        # trigger_barge_in() MUST fire before speak.stop()
                        # so wait_or_barge_in() sees the event while
                        # is_playing() is still True, returns True, and
                        # listen.py clears the event — otherwise the stale
                        # event prematurely cuts off the next turn's TTS.
                        self._did_barge_in = True
                        if self._listen is not None:
                            self._listen.trigger_barge_in()
                        if self._speak is not None:
                            self._speak.stop()
        
        except Exception as e:
            log.exception("[aiko-web] error in WebSocket loop")
        finally:
            reset_current_user_id(user_context_token)
            with self._clients_lock:
                self._clients.discard(ws)
            log.info("[aiko-web] browser disconnected (total=%d)", len(self._clients))
            # reset mic state so a reconnecting browser gets mic:start again
            if not self._clients:
                self._mic_started = False
                self._did_barge_in = False

    # ------------------------------------------------------------------
    # broadcast helpers
    # ------------------------------------------------------------------

    def _broadcast(self, payload: dict) -> None:
        """Fire-and-forget: schedule a JSON broadcast on the asyncio loop."""
        if self._loop is None:
            return
        asyncio.run_coroutine_threadsafe(self._async_broadcast(payload), self._loop)

    async def _async_broadcast(self, payload: dict) -> None:
        raw = json.dumps(payload, ensure_ascii=False)
        with self._clients_lock:
            targets = list(self._clients)
        if not targets:
            return
        await asyncio.gather(
            *(self._safe_send(ws, raw) for ws in targets),
            return_exceptions=True,
        )

    def broadcast_audio_bytes(self, wav_bytes: bytes) -> None:
        """
        Fire-and-forget: send raw WAV bytes as a binary WS frame to every
        connected browser. Called by speak.py's audio sink so TTS audio plays
        in the remote browser instead of (or alongside) the Jetson's local speaker.
        """
        if self._loop is None:
            return
        with self._clients_lock:
            if not self._clients:
                return
        asyncio.run_coroutine_threadsafe(self._async_broadcast_bytes(wav_bytes), self._loop)

    async def _async_broadcast_bytes(self, raw: bytes) -> None:
        with self._clients_lock:
            targets = list(self._clients)
        if not targets:
            return
        await asyncio.gather(
            *(self._safe_send(ws, raw) for ws in targets),
            return_exceptions=True,
        )

    def has_remote_listener(self) -> bool:
        """True if at least one browser is connected — speak.py uses this to
        decide whether local Jetson playback should still also happen."""
        with self._clients_lock:
            return bool(self._clients)

    @staticmethod
    async def _safe_send(ws, raw) -> None:
        try:
            if isinstance(raw, bytes):
                await ws.send_bytes(raw)
            else:
                await ws.send_text(raw)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # ── TUI-compatible draw API ──────────────────────────────────────────
    # ------------------------------------------------------------------

    def _draw(self, buf=None) -> None:
        """No-op — browser redraws itself from pushed events."""
        pass

    def _draw_clock_only(self) -> None:
        """Push a vitals update (browser shows the clock)."""
        self._push_vitals()

    # ------------------------------------------------------------------
    # init / boot step API
    # ------------------------------------------------------------------

    def step_loading(self, key: str, detail: str = "") -> None:
        self._broadcast({"type": "step", "key": key, "state": "loading", "detail": detail})

    def step_done(self, key: str, detail: str = "") -> None:
        self._broadcast({"type": "step", "key": key, "state": "done",    "detail": detail})

    def step_skip(self, key: str, detail: str = "") -> None:
        self._broadcast({"type": "step", "key": key, "state": "skip",    "detail": detail})

    def step_error(self, key: str, detail: str = "") -> None:
        self._broadcast({"type": "step", "key": key, "state": "error",   "detail": detail})

    def status_finish(self) -> None:
        """Transition browser from init phase to chat phase."""
        self._broadcast({"type": "phase", "value": "chat"})

    # ------------------------------------------------------------------
    # chat API
    # ------------------------------------------------------------------

    def add_message(self, sender: str, text: str) -> None:
        """Commit a completed message to the browser conversation log."""
        self._broadcast({"type": "chat", "sender": sender, "text": text})

    def stream_token(self, token: str) -> None:
        """
        Forward a streaming token to the browser.
        Intercepts agentic control sentinels so they route to the
        tool-status display instead of chat text.
        """
        if token.startswith("__THINKING__"):
            self._broadcast({"type": "tool", "status": "thinking…"})
            self._broadcast({"type": "pose", "name": "thinking", "active": True})
            return
        if token.startswith("__TOOL__:"):
            name = token[len("__TOOL__:"):].split("(", 1)[0].strip()
            self._broadcast({"type": "tool", "status": f"using {name}"})
            return
        if token.startswith("__SEARCHING__:"):
            query = token[len("__SEARCHING__:"):].strip()
            self._broadcast({"type": "tool", "status": f"searching: {query}"})
            return

        with self._lock:
            self._streaming += token
            count = len(token)
            self._stats["tokens"]   += count
            self._stats["turn_tok"] += count
            if self._stats["turn_start"] is None:
                self._stats["turn_start"] = time.time()

        self._broadcast({"type": "pose", "name": "thinking", "active": False})
        self._broadcast({"type": "token", "text": token})

    def stream_commit(self) -> None:
        """Finalise the active streaming turn."""
        with self._lock:
            if self._stats["turn_start"] is not None:
                elapsed = time.time() - self._stats["turn_start"]
                self._stats["tok_s"] = (
                    self._stats["turn_tok"] / elapsed if elapsed > 0 else 0.0
                )
            self._stats["turn_tok"]   = 0
            self._stats["turn_start"] = None
            self._streaming           = ""

        self._broadcast({"type": "pose", "name": "thinking", "active": False})
        self._broadcast({"type": "commit"})
        self._push_vitals()

    def turn_start(self) -> None:
        """Signal the beginning of a new cognitive turn."""
        with self._lock:
            self._stats["turn_start"] = time.time()
            self._stats["turn_tok"]   = 0
        self._broadcast({"type": "pose", "name": "thinking", "active": True})

    # ------------------------------------------------------------------
    # vitals
    # ------------------------------------------------------------------

    def _push_vitals(self) -> None:
        """Broadcast a vitals snapshot to all connected browsers."""
        try:
            from system.health import _ram_used_str, _db_size_str, _fmt_uptime
            ram    = _ram_used_str()
            uptime = _fmt_uptime(time.time() - self._ts)
        except Exception:
            ram    = "—"
            uptime = "—"

        with self._lock:
            s = dict(self._stats)

        self._broadcast({
            "type":   "vitals",
            "tokens": s["tokens"],
            "tok_s":  round(s["tok_s"], 1),
            "ram":    ram,
            "uptime": uptime,
            "asr":    s["asr_on"],
            "tts":    s["tts_on"],
        })

    def update_stats(self, key: str, value) -> None:
        """Allow main.py to poke individual stat fields (e.g. asr_on, tts_on)."""
        with self._lock:
            self._stats[key] = value
        self._push_vitals()

    # ------------------------------------------------------------------
    # expression / viseme passthrough
    # ------------------------------------------------------------------

    def set_expression(self, name: str, intensity: float = 1.0) -> None:
        self._broadcast({"type": "expression", "name": name, "intensity": intensity})

    def set_viseme(self, viseme: str, weight: float = 1.0) -> None:
        self._broadcast({"type": "viseme", "viseme": viseme, "weight": weight})

    # ------------------------------------------------------------------
    # input
    # ------------------------------------------------------------------

    def get_input(self) -> str:
        """
        Block until the browser submits a user message.

        Pushes a vitals tick every 10 seconds while idle (reduced from 1s)
        so the browser clock stays live without unnecessary network traffic
        when the phone screen is on but no conversation is happening.
        """
        self._broadcast({"type": "voice", "status": "idle"})
        idle_ticks = 0
        while True:
            try:
                text = self._input_q.get(timeout=1.0)
                set_current_user_id(self._current_user_id)
                set_current_display_name(self._current_display_name)
                return text
            except queue.Empty:
                idle_ticks += 1
                if idle_ticks % 10 == 0:    # vitals every ~10 s when idle
                    self._push_vitals()

    def get_voice_input(self, listen, speak=None, wait_fn=None):
        """
        Capture a voice utterance via the browser's microphone and return the
        same (text, info) shape as AikoTUI.get_voice_input().

        The browser mic stays open continuously once started, so the browser
        VAD can detect speech during TTS playback and send instant barge-in
        messages (zero round-trip). Between turns, the audio queue is only
        drained if no barge-in occurred — otherwise the speech frames that
        triggered the barge-in are preserved for recording.

        By default, the browser VAD gates 16 kHz float32 PCM before it enters
        _audio_q, so only browser-detected speech is sent over the network. Set
        WEBUI_BROWSER_VAD_GATE=0 for a diagnostic raw-stream mode that bypasses
        browser gating and lets listen.py run server-side VAD.
        """
        result_holder = [None]
        done_event    = threading.Event()

        # If barge-in fired while TTS was playing, the speaker's voice frames
        # are already in the queue — don't discard them.
        if not self._did_barge_in:
            while True:
                try:
                    self._audio_q.get_nowait()
                except queue.Empty:
                    break
        self._did_barge_in = False

        BYTES_PER_CHUNK = 512 * 4   # 512 float32 samples = 2048 bytes
        FRAME_TIMEOUT_S = 5.0       # browser disconnected / stopped delivering PCM

        def _chunk_source(n: int):
            """
            Pulled by listen.py's _record() loop instead of parec.
            Returns None on:
              - empty-bytes sentinel  → browser VAD declared end-of-utterance
              - timeout               → browser disconnected / went silent
            listen.py treats None as clean end-of-recording.
            """
            try:
                raw = self._audio_q.get(timeout=FRAME_TIMEOUT_S)
            except queue.Empty:
                return None         # browser stalled or disconnected

            if raw == b"":
                return None         # browser VAD end-of-utterance sentinel

            if len(raw) != n:
                # tolerate boundary-size mismatches by padding/truncating
                raw = (raw + b"\x00" * n)[:n]
            return raw

        def _status_cb(token: str) -> None:
            # listen.py status tokens → voice status display
            # Note: listening/transcribing are now also set by vad sentinels
            # in _ws_handler, so this is a secondary/fallback update path
            mapping = {
                "__WAITING__":      "waiting",
                "__LISTENING__":    "listening",
                "__TRANSCRIBING__": "transcribing",
                "__IDLE__":         "idle",
            }
            status = mapping.get(token, "idle")
            self._broadcast({"type": "voice", "status": status})

        def _run() -> None:
            set_current_user_id(self._current_user_id)
            set_current_display_name(self._current_display_name)
            os.environ["AIKO_USER_ID"] = self._current_user_id
            os.environ["AIKO_DISPLAY_NAME"] = self._current_display_name
            result_holder[0] = listen.listen(
                status_callback=_status_cb,
                speak=speak,
                wait_fn=wait_fn,
                chunk_source=_chunk_source,
                vad_presegmented=WEBUI_BROWSER_VAD_GATE,
            )
            done_event.set()

        # Keep the browser mic open continuously once started, so the browser
        # VAD can detect speech during TTS playback for instant barge-in.
        if not self._mic_started:
            self._mic_active.set()
            self._broadcast({
                "type": "mic",
                "action": "start",
                "bytes_per_chunk": BYTES_PER_CHUNK,
                "browser_vad_gate": WEBUI_BROWSER_VAD_GATE,
            })
            self._mic_started = True

        threading.Thread(target=_run, daemon=True).start()

        text_input = None
        try:
            while not done_event.wait(timeout=0.1):
                self._push_vitals()
                try:
                    text_input = self._input_q.get_nowait()
                    self._audio_q.put(b"")  # Signal end-of-utterance to stop recording
                    done_event.wait()       # Wait for the listen thread to exit
                    break
                except queue.Empty:
                    pass
        finally:
            self._broadcast({"type": "voice", "status": "idle"})

        # Check one last time if a text input arrived as we finished
        if text_input is None:
            try:
                text_input = self._input_q.get_nowait()
            except queue.Empty:
                pass

        if text_input is not None:
            return (text_input, {})

        raw = result_holder[0]
        if isinstance(raw, tuple):
            return raw
        return (raw or "", {})

    # ------------------------------------------------------------------
    # spin loop
    # ------------------------------------------------------------------

    def spin_loop(self, stop_event: threading.Event) -> None:
        """Drive periodic vitals pushes during the init phase."""
        while not stop_event.is_set():
            self._push_vitals()
            stop_event.wait(0.25)
        self._push_vitals()


# HTTP static file handler removed. Serving is done via FastAPI StaticFiles.


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT (called from main.py)
# ═════════════════════════════════════════════════════════════════════════════

def run_webui(args) -> None:
    """Launch Aiko with the browser WebUI (default front end).

    Constructs the AikoWeb transport, prints the URL, and hands off to
    system.orchestrate.run_session() for the shared boot/turn-loop logic
    (identical code path as the CLI front end from this point on).
    """
    from system.orchestrate import run_session

    import socket
    ui = AikoWeb(no_voice=args.text, debug=args.debug)
    host_ip = socket.gethostbyname(socket.gethostname())
    scheme = "https" if WEBUI_HTTPS else "http"
    print(f"\n  🌸 Aiko-chan is ready → {scheme}://{host_ip}:{HTTP_PORT}/\n")
    print(f"  Waiting for login before waking up subsystems...\n")
    run_session(ui, args)
