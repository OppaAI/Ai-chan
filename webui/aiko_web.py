"""
webui/aiko_web.py
Aiko-chan's browser-based UI backend — drop-in replacement for AikoTUI.

Responsibilities:
    - Serve aiko.html + assets from webui/static/ over HTTP (localhost:PORT)
    - Host a bidirectional WebSocket server (localhost:WS_PORT)
    - Expose the same public API as AikoTUI so main.py needs minimal changes:
        add_message / stream_token / stream_commit / turn_start
        step_loading / step_done / step_skip / step_error / status_finish
        get_input / get_voice_input / spin_loop
    - Block get_input() until the browser sends {"type":"user_input","text":"..."}
    - Auto-open the browser on first connection

Environment variables (all optional):
    AIKO_HTTP_PORT   — HTTP port for serving the UI (default 8787)
    AIKO_WS_PORT     — WebSocket port                (default 8765)
    AIKO_NO_BROWSER  — set to "1" to suppress auto-open
"""

import asyncio
import http.server
import json
import logging
import os
import queue
import threading
import time
import webbrowser
from pathlib import Path

import websockets
from websockets.server import serve as ws_serve

log = logging.getLogger(__name__)

# ── config ────────────────────────────────────────────────────────────────────

HTTP_PORT  = int(os.getenv("AIKO_HTTP_PORT", "8787"))
WS_PORT    = int(os.getenv("AIKO_WS_PORT",   "8765"))
STATIC_DIR = Path(__file__).parent / "static"
NO_BROWSER = os.getenv("AIKO_NO_BROWSER", "0") == "1"

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
#
# browser → server:
# user_input{"type":"user_input","text":"..."}
# vad       {"type":"vad",     "event":"start"|"end"}   ← browser VAD sentinels


class AikoWeb:
    """
    Browser-based drop-in for AikoTUI.

    All drawing methods become JSON WebSocket broadcasts.
    get_input() blocks on a threading.Queue until the browser submits a message.

    VAD is handled entirely in the browser (vad.js / Silero ONNX WASM).
    The browser sends {"type":"vad","event":"start"} when it detects speech onset
    and {"type":"vad","event":"end"} when silence is detected after speech ends.
    "end" is translated to an empty-bytes sentinel in _audio_q so _chunk_source
    can signal listen.py to skip its own VAD pass and treat the frame stream as
    a pre-segmented utterance.
    """

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    def __init__(self, no_voice: bool = False, debug: bool = False):
        self._no_voice = no_voice
        self._debug    = debug
        self._ts       = time.time()
        self._lock     = threading.Lock()

        # input queue — browser posts here, get_input() reads here
        self._input_q: queue.Queue[str] = queue.Queue()

        # binary mic-audio frames from the browser, consumed by get_voice_input()
        # an empty-bytes sentinel (b"") signals end-of-utterance from browser VAD
        self._audio_q: queue.Queue[bytes] = queue.Queue()
        self._mic_active = threading.Event()

        # connected browser websocket clients
        self._clients: set = set()
        self._clients_lock = threading.Lock()

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

        self._start_servers()

    # ------------------------------------------------------------------
    # server lifecycle
    # ------------------------------------------------------------------

    def _start_servers(self) -> None:
        """Spin up the HTTP file server and WebSocket server in daemon threads."""
        import socket
        host_ip = socket.gethostbyname(socket.gethostname())
        http_t = threading.Thread(target=self._run_http, daemon=True, name="aiko-http")
        http_t.start()

        ws_t = threading.Thread(target=self._run_ws_loop, daemon=True, name="aiko-ws")
        ws_t.start()

        self._loop_ready.wait(timeout=5)

        if not NO_BROWSER:
            threading.Timer(0.6, lambda: webbrowser.open(f"http://{host_ip}:{HTTP_PORT}/")).start()

    def _run_http(self) -> None:
        """Serve webui/static/ over plain HTTP."""
        handler = _make_static_handler(STATIC_DIR)
        with http.server.HTTPServer(("0.0.0.0", HTTP_PORT), handler) as srv:
            srv.serve_forever()

    def _run_ws_loop(self) -> None:
        """Run the asyncio event loop (and WebSocket server) in this thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._ws_main())

    async def _ws_main(self) -> None:
        """Async entry point: start the WS server then signal ready."""
        async with ws_serve(self._ws_handler, "0.0.0.0", WS_PORT):
            self._loop_ready.set()
            await asyncio.Future()          # run forever

    async def _ws_handler(self, ws) -> None:
        """Handle one browser WebSocket connection."""
        with self._clients_lock:
            self._clients.add(ws)
        log.info("[aiko-web] browser connected  (total=%d)", len(self._clients))
        try:
            async for raw in ws:
                if isinstance(raw, bytes):
                    # browser mic PCM frame — only buffer during an active voice turn
                    # so stale/late frames from a previous turn don't pollute the next
                    if self._mic_active.is_set():
                        self._audio_q.put(raw)
                    continue
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                mtype = msg.get("type")

                if mtype == "user_input":
                    text = (msg.get("text") or "").strip()
                    if text:
                        self._input_q.put(text)

                elif mtype == "vad":
                    # browser Silero VAD sentinels — update voice status display
                    # and inject end-of-utterance sentinel into the audio queue
                    event = msg.get("event")
                    if event == "start":
                        # speech onset — update UI; listen.py will see audio frames arriving
                        self._broadcast({"type": "voice", "status": "listening"})
                    elif event == "end":
                        # speech ended — push empty-bytes sentinel so _chunk_source
                        # returns None cleanly, ending the recording loop in listen.py
                        self._broadcast({"type": "voice", "status": "transcribing"})
                        if self._mic_active.is_set():
                            self._audio_q.put(b"")  # end-of-utterance sentinel

        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            with self._clients_lock:
                self._clients.discard(ws)
            log.info("[aiko-web] browser disconnected (total=%d)", len(self._clients))

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
            await ws.send(raw)
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

        self._broadcast({"type": "commit"})
        self._push_vitals()

    def turn_start(self) -> None:
        """Signal the beginning of a new cognitive turn."""
        with self._lock:
            self._stats["turn_start"] = time.time()
            self._stats["turn_tok"]   = 0

    # ------------------------------------------------------------------
    # vitals
    # ------------------------------------------------------------------

    def _push_vitals(self) -> None:
        """Broadcast a vitals snapshot to all connected browsers."""
        try:
            from core.health import _ram_used_str, _db_size_str, _fmt_uptime
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
                return self._input_q.get(timeout=1.0)
            except queue.Empty:
                idle_ticks += 1
                if idle_ticks % 10 == 0:    # vitals every ~10 s when idle
                    self._push_vitals()

    def get_voice_input(self, listen, speak=None, wait_fn=None):
        """
        Capture a voice utterance via the browser's microphone and return the
        same (text, info) shape as AikoTUI.get_voice_input().

        The browser's Silero VAD (vad.js) handles speech/silence detection:
          - Only speech frames are sent as binary WS frames → _audio_q
          - vad:start  → UI status update (handled in _ws_handler)
          - vad:end    → empty-bytes sentinel pushed into _audio_q

        _chunk_source feeds these frames into listen.py. When it receives the
        empty-bytes sentinel it returns None, signalling end-of-utterance to
        listen.py so it can skip its own VAD pass and go straight to ASR.
        """
        result_holder = [None]
        done_event    = threading.Event()

        # drain stale frames from any previous turn
        while True:
            try:
                self._audio_q.get_nowait()
            except queue.Empty:
                break

        BYTES_PER_CHUNK = 512 * 4   # 512 float32 samples = 2048 bytes
        FRAME_TIMEOUT_S = 5.0       # longer timeout — browser VAD may take a moment
                                    # to detect speech onset; 2 s was too aggressive

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
            result_holder[0] = listen.listen(
                status_callback=_status_cb,
                speak=speak,
                wait_fn=wait_fn,
                chunk_source=_chunk_source,
            )
            done_event.set()

        self._mic_active.set()
        self._broadcast({"type": "mic", "action": "start", "bytes_per_chunk": BYTES_PER_CHUNK})
        threading.Thread(target=_run, daemon=True).start()

        try:
            while not done_event.wait(timeout=1.0):
                self._push_vitals()
        finally:
            self._mic_active.clear()
            self._broadcast({"type": "mic", "action": "stop"})

        self._broadcast({"type": "voice", "status": "idle"})
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


# ── HTTP static file handler ──────────────────────────────────────────────────

def _make_static_handler(root: Path):
    """
    Return an HTTPRequestHandler class that serves files from `root`.
    index.html is served at /. All other paths resolve relative to root.
    """
    class _Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(root), **kwargs)

        def log_message(self, fmt, *args):  # suppress access log spam
            pass

        def translate_path(self, path):
            return super().translate_path(path)

    return _Handler
