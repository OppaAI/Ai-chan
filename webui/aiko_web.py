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

HTTP_PORT   = int(os.getenv("AIKO_HTTP_PORT", "8787"))
WS_PORT     = int(os.getenv("AIKO_WS_PORT",   "8765"))
STATIC_DIR  = Path(__file__).parent / "static"
NO_BROWSER  = os.getenv("AIKO_NO_BROWSER", "0") == "1"

# ── message types (server → browser) ─────────────────────────────────────────
#
# chat      {"type":"chat",    "sender":"you"|"aiko"|"sys", "text":"..."}
# token     {"type":"token",   "text":"..."}                 # streaming chunk
# commit    {"type":"commit"}                                 # end of stream
# step      {"type":"step",    "key":"...", "state":"loading"|"done"|"skip"|"error", "detail":"..."}
# phase     {"type":"phase",   "value":"init"|"chat"}
# vitals    {"type":"vitals",  "tokens":0,  "tok_s":0.0, "ram":"...", "uptime":"...","asr":true,"tts":true}
# voice     {"type":"voice",   "status":"waiting"|"listening"|"transcribing"|"idle"}
# tool      {"type":"tool",    "status":"..."|null}          # agentic tool status
# expression{"type":"expression","name":"...","intensity":0.8}
# viseme    {"type":"viseme",  "viseme":"A","weight":0.6}
#
# browser → server:
# user_input{"type":"user_input","text":"..."}


class AikoWeb:
    """
    Browser-based drop-in for AikoTUI.

    All drawing methods become JSON WebSocket broadcasts.
    get_input() blocks on a threading.Queue until the browser submits a message.
    """

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    def __init__(self, no_voice: bool = False, debug: bool = False):
        self._no_voice  = no_voice
        self._debug     = debug
        self._ts        = time.time()
        self._lock      = threading.Lock()

        # input queue — browser posts here, get_input() reads here
        self._input_q: queue.Queue[str] = queue.Queue()

        # binary mic-audio frames from the browser, consumed by get_voice_input()
        self._audio_q: queue.Queue[bytes] = queue.Queue()
        self._mic_active = threading.Event()

        # connected browser websocket clients
        self._clients: set = set()
        self._clients_lock = threading.Lock()

        # streaming state
        self._streaming   = ""
        self._tool_status = None

        # session stats (mirrors AikoTUI._stats)
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
        # HTTP — serves webui/static/
        import socket
        host_ip = socket.gethostbyname(socket.gethostname())
        http_t = threading.Thread(target=self._run_http, daemon=True, name="aiko-http")
        http_t.start()

        # WebSocket — asyncio loop in its own thread
        ws_t = threading.Thread(target=self._run_ws_loop, daemon=True, name="aiko-ws")
        ws_t.start()

        # wait for the asyncio loop to be ready before returning
        self._loop_ready.wait(timeout=5)

        if not NO_BROWSER:
            # slight delay so the HTTP server is accepting before the browser hits it
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
                    # browser mic audio frame — only buffer it while a voice
                    # turn is actually waiting on input, so stray/late frames
                    # from a previous turn don't pollute the next recording.
                    if self._mic_active.is_set():
                        self._audio_q.put(raw)
                    continue
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if msg.get("type") == "user_input":
                    text = (msg.get("text") or "").strip()
                    if text:
                        self._input_q.put(text)
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

    @staticmethod
    async def _safe_send(ws, raw: str) -> None:
        try:
            await ws.send(raw)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # ── TUI-compatible draw API ──────────────────────────────────────────
    # All of these are no-ops or JSON broadcasts; none block.
    # ------------------------------------------------------------------

    def _draw(self, buf=None) -> None:
        """No-op — browser redraws itself from pushed events."""
        pass

    def _draw_clock_only(self) -> None:
        """Push a vitals update (browser shows the clock)."""
        self._push_vitals()

    # ------------------------------------------------------------------
    # init / boot step API  (mirrors AikoTUI step_* methods)
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
    # chat API  (mirrors AikoTUI add_message / stream_* / turn_start)
    # ------------------------------------------------------------------

    def add_message(self, sender: str, text: str) -> None:
        """Commit a completed message to the browser conversation log."""
        self._broadcast({"type": "chat", "sender": sender, "text": text})

    def stream_token(self, token: str) -> None:
        """
        Forward a streaming token to the browser.

        Intercepts agentic control sentinels the same way AikoTUI does so
        they route to the tool-status display instead of chat text.
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

        # normal visible token
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
    # expression / viseme passthrough  (also called by speak pipeline)
    # ------------------------------------------------------------------

    def set_expression(self, name: str, intensity: float = 1.0) -> None:
        self._broadcast({"type": "expression", "name": name, "intensity": intensity})

    def set_viseme(self, viseme: str, weight: float = 1.0) -> None:
        self._broadcast({"type": "viseme", "viseme": viseme, "weight": weight})

    # ------------------------------------------------------------------
    # input  (mirrors AikoTUI.get_input / get_voice_input)
    # ------------------------------------------------------------------

    def get_input(self) -> str:
        """
        Block until the browser submits a user message.

        Pushes a vitals tick every second while waiting so the browser clock
        stays live.
        """
        self._broadcast({"type": "voice", "status": "idle"})
        while True:
            try:
                return self._input_q.get(timeout=1.0)
            except queue.Empty:
                self._push_vitals()

    def get_voice_input(self, listen, speak=None, wait_fn=None) -> str:
        """
        Capture a voice utterance via the browser's microphone and return the
        transcript. Tells the browser to start streaming raw PCM mic frames
        over the WebSocket (binary frames), feeds them into the same VAD/ASR
        pipeline used locally (via chunk_source), and stops the browser mic
        once the utterance is captured or the turn times out.
        """
        result_holder = [None]
        done_event    = threading.Event()

        # drain any stale frames left over from a previous turn
        while True:
            try:
                self._audio_q.get_nowait()
            except queue.Empty:
                break

        BYTES_PER_CHUNK = 512 * 4  # must match listen.py's _CHUNK_SAMPLES_VAD * 4
        FRAME_TIMEOUT_S = 2.0      # if the browser goes quiet this long, end the turn

        def _chunk_source(n: int):
            """Pulled by listen.py's _record() loop instead of parec."""
            try:
                raw = self._audio_q.get(timeout=FRAME_TIMEOUT_S)
            except queue.Empty:
                return None  # browser stalled/disconnected — end the recording
            if len(raw) != n:
                # tolerate odd-sized frames by padding/truncating rather than
                # dropping the whole utterance over a boundary mismatch
                raw = (raw + b"\x00" * n)[:n]
            return raw

        def _status_cb(token: str) -> None:
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
            raw = raw[0]
        return raw or ""

    # ------------------------------------------------------------------
    # spin loop  (called during init phase — just keeps vitals ticking)
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

    index.html (aiko.html renamed) is served at /.
    All other paths resolve relative to root.
    """
    class _Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(root), **kwargs)

        def log_message(self, fmt, *args):  # suppress access log spam
            pass

        def translate_path(self, path):
            # map / → index.html inside our static root
            result = super().translate_path(path)
            return result

    return _Handler