"""
webui/webui.py — Aiko-chan's browser-based UI backend.

Serves the WebUI over HTTP(S) + WebSocket so any device on the LAN (PC,
phone, tablet) can connect — drop-in replacement for AikoTUI when running
headless (e.g. on Jetson).

Sections:
    Config          — env-driven ports, HTTPS, VAD-gate toggles (module level)
    TLS setup       — self-signed cert generation for LAN HTTPS/mic access
    AikoWeb         — server lifecycle, per-connection auth, chat/voice I/O
        .connection   — WebSocket auth handshake, per-user context binding
        .broadcast    — fan-out of chat tokens, tool status, vitals, poses
        .boot-report  — step_loading/step_done/... boot progress events
        .voice I/O    — mic audio queue, barge-in, VAD status relay
    run_webui()     — entry point called from main.py after boot

Notes:
    - barge_in WebSocket messages are ignored when BARGE_IN_ENABLED is off.
    - mic start payload includes echo_guard_ms for browser barge echo guard.
    - No local browser auto-launch: this runs headless (e.g. Jetson) and is
      always accessed remotely, so webbrowser/xdg-open is intentionally not used.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
import queue
import ssl
import subprocess
import threading
import time
from typing import ClassVar, TYPE_CHECKING
if TYPE_CHECKING:
    from system.wakeup import BootResult
    
from system.config import load_config
from system.userspace import reset_current_display_name, reset_current_user_id, set_current_user_id, set_current_display_name
load_config()

from system import bioclock

log = logging.getLogger(__name__)

HTTP_PORT  = int(os.getenv("HTTP_PORT", "8787"))
STATIC_DIR = Path(__file__).parent / "static"
NO_BROWSER = os.getenv("NO_BROWSER", "0") == "1"
WEBUI_HTTPS = os.getenv("WEBUI_HTTPS", "0").lower() in {"1", "true", "yes", "on"}
SSL_CERT = os.getenv("SSL_CERT", "")
SSL_KEY = os.getenv("SSL_KEY", "")
WEBUI_BROWSER_VAD_GATE = os.getenv("WEBUI_BROWSER_VAD_GATE", "1").lower() in {"1", "true", "yes", "on"}


def _barge_in_enabled() -> bool:
    # sensory.listen_native was removed — barge-in is native to sensory.listen
    # now (see that module's docstring). The bare env-parsing fallback below
    # only remains as a last resort if sensory.listen itself is unavailable
    # for some reason (e.g. import-order edge case at boot).
    try:
        from sensory.listen import barge_in_enabled
        return barge_in_enabled()
    except Exception:
        return os.getenv("BARGE_IN_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}


def _echo_guard_ms() -> int:
    try:
        return max(0, int(os.getenv("BARGE_IN_ECHO_GUARD_MS", "450")))
    except ValueError:
        return 450


def _load_stored_display_name(uid: str) -> str:
    try:
        from system.userspace import user_state_dir
        name_file = user_state_dir(uid) / "cli_name.txt"
        if name_file.exists():
            stored = name_file.read_text(encoding="utf-8").strip()
            if stored:
                return stored
    except Exception:
        log.warning("webui: failed to read cli_name.txt")
    return ""


def _make_ssl_context(hostname: str, host_ip: str) -> ssl.SSLContext | None:
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


class AikoWeb:
    def __init__(self, no_voice: bool = False, debug: bool = False, boot_result: BootResult | None = None):
        self._no_voice     = no_voice
        self._debug        = debug
        self._boot_result  = boot_result
        self._ts           = time.time()
        self._lock         = threading.Lock()

        self._login_event = threading.Event()
        self._authenticated_uid: str | None = None
        self._authenticated_display_name: str | None = None
        self._user_space_ready = False   # flipped after first authenticated connect seeds user space

        self._input_q: queue.Queue[tuple[str, str, str]] = queue.Queue()

        # 4096 frames × 512 samples ≈ 8MB ceiling ≈ 13s of 16kHz mono f32 —
        # ample headroom for VAD-gated speech bursts; raw-mic mode drops
        # frames when full instead of ballooning RAM (was 10000 / ~20MB).
        self._audio_q: queue.Queue[bytes | tuple[bytes, str, str]] = queue.Queue(maxsize=4096)
        self._mic_active = threading.Event()
        self._did_barge_in: bool = False

        self._clients: set = set()
        self._clients_lock = threading.Lock()

        self._memorize = None
        self._think = None
        self._speak = None
        self._listen = None

        self._streaming   = ""
        self._tool_status = None

        self._stats: dict = {
            "tokens":     0,
            "turn_tok":   0,
            "turn_start": None,
            "tok_s":      0.0,
            "asr_on":     not no_voice,
            "tts_on":     not no_voice,
        }

        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_ready = threading.Event()
        self._ssl_context: ssl.SSLContext | None = None

        import interface.webui.auth
        interface.webui.auth.aiko_web_instance = self

        self._start_servers()
        
    def set_boot_result(self, result: BootResult):
        """Called by main.py AFTER wakeup.boot() finishes (now before login)."""
        self._boot_result = result
        if result:
            self.status_finish()   # broadcasts the "phase: chat" event
            self._broadcast({"type": "status_finish", "ready": True, "message": "Aiko fully initialized — log in now"})
            
    def set_voice_backends(self, speak, listen) -> None:
        self._speak = speak
        self._listen = listen

    def set_memorize(self, memorize) -> None:
        self._memorize = memorize

    def set_think(self, think) -> None:
        self._think = think

    async def _bind_user_context(self, uid: str, session: dict) -> tuple:
        """Resolve display name and bind uid/display-name to this WebSocket task.

        WebUI can serve multiple browsers concurrently, so per-connection
        identity must stay local to the handler task. Do not mirror it into
        process-wide environment variables or a shared AikoMemorize instance;
        downstream turn code receives the uid/display name explicitly via the
        input queue and contextvars.
        """
        stored_name = _load_stored_display_name(uid)
        session_name = session.get("username") or ""
        display_name = stored_name or session_name or uid
        user_token = set_current_user_id(uid)
        display_token = set_current_display_name(display_name)
        return user_token, display_token, display_name
            
    def _on_user_active(self, uid: str) -> None:
        """Post-login hook: rebind user-scoped subsystems and run seeding that
        boot skipped because it ran pre-auth as guest.

        Called on every authenticated WS connect: the scheduler job store is
        rebound per-connect (each browser may be a different user), while
        one-time user-space seeding (playbooks + schedule jobs) runs only once.
        """
        try:
            from system.schedule import get_scheduler
            sched = get_scheduler()
            if sched is not None:
                sched.set_user(uid)
        except Exception:
            log.exception("[aiko-web] scheduler user switch failed for %s", uid)

        if self._user_space_ready or uid == "guest":
            return
        self._user_space_ready = True
        log.info("[aiko-web] first authenticated user (%s) — running deferred user-space seeding", uid)
        # Boot skipped memory cleanup for guest — run it now that the real
        # per-user store is open (parity with the old login-gated boot).
        try:
            if self._memorize is not None:
                self._memorize.cleanup()
        except Exception:
            log.exception("[aiko-web] post-login memory cleanup failed")
        try:
            from agentic.graph_engine import ensure_playbooks
            ensure_playbooks(user_id=uid)
        except Exception:
            log.exception("[aiko-web] playbook seeding failed")
        try:
            from system.schedule import bootstrap_non_system_jobs
            bootstrap_non_system_jobs(think=self._think, memorize=self._memorize)
        except Exception:
            log.exception("[aiko-web] schedule job bootstrap failed")

    def _start_servers(self) -> None:
        import socket
        hostname = socket.gethostname()
        host_ip = socket.gethostbyname(hostname)
        self._ssl_context = _make_ssl_context(hostname, host_ip)
    
        from interface.webui.auth import app as auth_app
        from starlette.staticfiles import StaticFiles
    
        has_static = any(getattr(route, "name", None) == "static" for route in auth_app.routes)
        if not has_static:
            auth_app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
    
        http_t = threading.Thread(target=self._run_http, daemon=True, name="aiko-http")
        http_t.start()
    
        self._loop_ready.wait(timeout=5)

    def _run_http(self) -> None:
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
        from interface.webui.auth import sessions, signer, SESSION_MAX_AGE_SECONDS
        from itsdangerous import BadSignature, SignatureExpired
        from datetime import timedelta

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
        user_context_token, display_context_token, display_name = await self._bind_user_context(uid, session)
        
        if not self._login_event.is_set():
            self._authenticated_uid = uid
            self._authenticated_display_name = display_name
            self._login_event.set()
        
        await ws.accept()

        # Post-login rebinding/seeding runs in the background so the socket
        # receive loop starts immediately.
        asyncio.ensure_future(asyncio.to_thread(self._on_user_active, uid))

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
                    if self._mic_active.is_set():
                        try:
                            self._audio_q.put_nowait((raw_bytes, uid, display_name))
                        except queue.Full:
                            log.debug("webui: audio queue full, dropping frame")
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
                            self._input_q.put((text, uid, display_name))

                    elif mtype == "vad":
                        event = msg.get("event")
                        if event == "start":
                            self._broadcast({"type": "voice", "status": "listening"})
                        elif event == "end":
                            self._broadcast({"type": "voice", "status": "transcribing"})
                            if WEBUI_BROWSER_VAD_GATE and self._mic_active.is_set():
                                self._audio_q.put((b"", uid, display_name))
                                
                    elif mtype == "barge_in":
                        # S0: master switch — ignore browser barge when disabled
                        if not _barge_in_enabled():
                            log.debug("[aiko-web] barge_in ignored (BARGE_IN_ENABLED=0)")
                            continue
                        self._did_barge_in = True
                        if self._listen is not None:
                            self._listen.trigger_barge_in()
                        if self._speak is not None:
                            # stop() busy-waits for the stream thread to end;
                            # run it off the event loop so a long TTS playback
                            # doesn't freeze every other browser/request.
                            await asyncio.to_thread(self._speak.stop)
        
        except Exception:
            log.exception("[aiko-web] error in WebSocket loop")
        finally:
            reset_current_display_name(display_context_token)
            reset_current_user_id(user_context_token)
            with self._clients_lock:
                self._clients.discard(ws)
            log.info("[aiko-web] browser disconnected (total=%d)", len(self._clients))
            if not self._clients:
                self._did_barge_in = False

    def _broadcast(self, payload: dict) -> None:
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
            log.warning("webui: failed to send ws message")

    def _draw(self, buf=None) -> None:
        # No-op: preserves AikoTUI's draw interface so orchestrate.run_session()
        # can call ui._draw() polymorphically regardless of TUI vs WebUI backend.
        pass
    
    def _draw_clock_only(self) -> None:
        # Same rationale as _draw(); WebUI pushes vitals instead of redrawing.
        self._push_vitals()

    _BOOT_LABELS: ClassVar[dict[str, str] | None] = None

    @classmethod
    def _ensure_boot_labels(cls) -> dict[str, str]:
        if cls._BOOT_LABELS is None:
            from system.wakeup import AikoWakeup
            cls._BOOT_LABELS = AikoWakeup.ALL_BOOT_LABELS
        return cls._BOOT_LABELS

    def step_loading(self, key: str, detail: str = "") -> None:
        labels = self._ensure_boot_labels()
        self._broadcast({"type": "step", "key": key, "state": "loading", "label": labels.get(key, key), "detail": detail})

    def step_done(self, key: str, detail: str = "") -> None:
        labels = self._ensure_boot_labels()
        self._broadcast({"type": "step", "key": key, "state": "done",    "label": labels.get(key, key), "detail": detail})

    def step_skip(self, key: str, detail: str = "") -> None:
        labels = self._ensure_boot_labels()
        self._broadcast({"type": "step", "key": key, "state": "skip",    "label": labels.get(key, key), "detail": detail})

    def step_error(self, key: str, detail: str = "") -> None:
        labels = self._ensure_boot_labels()
        self._broadcast({"type": "step", "key": key, "state": "error",   "label": labels.get(key, key), "detail": detail})

    def status_finish(self) -> None:
        self._broadcast({"type": "phase", "value": "chat"})

    def add_message(self, sender: str, text: str) -> None:
        self._broadcast({"type": "chat", "sender": sender, "text": text})

    def stream_token(self, token: str) -> None:
        if token.startswith("__THINKING__"):
            self._broadcast({"type": "tool", "status": "thinking…"})
            self._broadcast({"type": "pose", "name": "thinking", "active": True})
            return
        if token.startswith("__TOOL__:"):
            name = token[len("__TOOL__:"):].split("(", 1)[0].strip()
            self._broadcast({"type": "tool", "status": f"using {name}"})
            return
        if token.startswith("__STATUS__:"):
            status = token[len("__STATUS__:"):].strip().split("\n", 1)[0].strip()
            labels = {
                "searching": "searching…",
                "retry": "line’s quiet — trying again…",
                "offline": "no live results — using what I know…",
                "ok": "got sources…",
            }
            if status.startswith("tool:"):
                label = f"using {status[5:].strip()}"
            else:
                label = labels.get(status, status)
            self._broadcast({"type": "tool", "status": label})
            return
        if token.startswith("__SEARCHING__:"):
            query = token[len("__SEARCHING__:"):].strip()
            self._broadcast({"type": "tool", "status": f"searching: {query}"})
            return
        if token.startswith("__RETRYING__"):
            self._broadcast({"type": "tool", "status": "line’s quiet — trying again…"})
            return
        if token.startswith("__SOURCES__:"):
            raw = token[len("__SOURCES__:"):].strip()
            try:
                import json as _json
                items = _json.loads(raw)
            except Exception:
                items = []
            self._broadcast({"type": "sources", "items": items})
            return
        if token.startswith("__FILES__:"):
            raw = token[len("__FILES__:"):].strip()
            try:
                import json as _json
                items = _json.loads(raw)
            except Exception:
                items = []
            self._broadcast({"type": "files", "items": items})
            return
        if token.startswith("__META__:"):
            # emotion|action — optional UI hook; never show in bubble
            meta = token[len("__META__:"):].strip()
            parts = meta.split("|", 1)
            emotion = (parts[0] or "neutral").strip()
            action = (parts[1] if len(parts) > 1 else "none").strip()
            self._broadcast({"type": "meta", "emotion": emotion, "action": action})
            return
        # Never paint raw JSON tool observations or bracketed system notices
        stripped = token.strip()
        if stripped.startswith("[No web results"):
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
        with self._lock:
            self._stats["turn_start"] = time.time()
            self._stats["turn_tok"]   = 0
        self._broadcast({"type": "pose", "name": "thinking", "active": True})

    def _push_vitals(self) -> None:
        try:
            from system.health import _ram_used_str, _fmt_uptime
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
        with self._lock:
            self._stats[key] = value
        self._push_vitals()

    def set_expression(self, name: str, intensity: float = 1.0) -> None:
        self._broadcast({"type": "expression", "name": name, "intensity": intensity})

    def set_viseme(self, viseme: str, weight: float = 1.0) -> None:
        self._broadcast({"type": "viseme", "viseme": viseme, "weight": weight})

    def get_input(self) -> str:
        self._broadcast({"type": "voice", "status": "idle"})
        idle_ticks = 0
        while True:
            try:
                item = self._input_q.get(timeout=1.0)
                if isinstance(item, tuple):
                    text, uid, display_name = item
                else:
                    text, uid, display_name = item, "guest", "Guest"
                set_current_user_id(uid)
                set_current_display_name(display_name)
                return text
            except queue.Empty:
                idle_ticks += 1
                if idle_ticks % 10 == 0:
                    self._push_vitals()

    def get_voice_input(self, listen, speak=None, wait_fn=None):
        result_holder = [None]
        done_event    = threading.Event()
        # Track the accepted user identity from the first frame
        accepted_user = {"uid": None, "display_name": None}

        if not self._did_barge_in:
            while True:
                try:
                    self._audio_q.get_nowait()
                except queue.Empty:
                    break
        self._did_barge_in = False

        BYTES_PER_CHUNK = 512 * 4
        FRAME_TIMEOUT_S = 5.0

        def _chunk_source(n: int):
            try:
                item = self._audio_q.get(timeout=FRAME_TIMEOUT_S)
            except queue.Empty:
                return None

            if isinstance(item, tuple):
                raw, uid, display_name = item
                # Reject frames from different users once we've accepted one
                if accepted_user["uid"] is None:
                    accepted_user["uid"] = uid
                    accepted_user["display_name"] = display_name
                elif accepted_user["uid"] != uid:
                    # Drop frame from different user
                    return None
                set_current_user_id(uid)
                set_current_display_name(display_name)
            else:
                raw = item

            if raw == b"":
                return None

            if len(raw) != n:
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
            # Call listen() with WebUI chunk source. See sensory.listen.AikoListen.listen()
            # docstring for full parameter contract. Key points:
            #   - vad_presegmented parameter was removed (no longer supported)
            #   - Silero VAD scores ALL chunks regardless of source (local mic or WebUI)
            #   - WEBUI_BROWSER_VAD_GATE only controls pre-gating in browser (static/vad.js),
            #     not server-side Silero scoring (which is authoritative)
            result_holder[0] = listen.listen(
                status_callback=_status_cb,
                speak=speak,
                wait_fn=wait_fn,
                chunk_source=_chunk_source,
            )
            done_event.set()

        self._mic_active.set()

        # Lazy voice boot: overlap server-side ASR/VAD model load with the
        # browser's mic-permission/arming UX instead of blocking on it.
        if listen is not None and hasattr(listen, "ensure_ready"):
            threading.Thread(target=listen.ensure_ready, daemon=True).start()

        self._broadcast({
            "type": "mic",
            "action": "start",
            "bytes_per_chunk": BYTES_PER_CHUNK,
            "browser_vad_gate": WEBUI_BROWSER_VAD_GATE,
            "barge_in_enabled": _barge_in_enabled(),
            "echo_guard_ms": _echo_guard_ms(),
        })

        threading.Thread(target=_run, daemon=True).start()

        text_input = None
        try:
            while not done_event.wait(timeout=0.1):
                self._push_vitals()
                try:
                    text_input = self._input_q.get_nowait()
                    self._audio_q.put(b"")
                    done_event.wait()
                    break
                except queue.Empty:
                    log.debug("webui: voice input queue empty, retrying")
        finally:
            self._mic_active.clear()
            self._broadcast({"type": "voice", "status": "idle"})

        if text_input is None:
            try:
                text_input = self._input_q.get_nowait()
            except queue.Empty:
                log.debug("webui: final voice input queue empty")

        if text_input is not None:
            if isinstance(text_input, tuple):
                text, uid, display_name = text_input
                set_current_user_id(uid)
                set_current_display_name(display_name)
                return (text, {})
            return (text_input, {})

        # Restore the accepted user identity before returning ASR result
        if accepted_user["uid"] is not None:
            set_current_user_id(accepted_user["uid"])
            set_current_display_name(accepted_user["display_name"])

        raw = result_holder[0]
        if isinstance(raw, tuple):
            return raw
        return (raw or "", {})

    def spin_loop(self, stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            self._push_vitals()
            stop_event.wait(0.25)
        self._push_vitals()


def run_webui(args, boot_result: BootResult | None = None) -> None:
    from system.orchestrate import run_session
    import socket

    ui = AikoWeb(no_voice=args.text, debug=args.debug, boot_result=boot_result)
    host_ip = socket.gethostbyname(socket.gethostname())
    scheme = "https" if WEBUI_HTTPS else "http"
    print(f"\n  🌸 Aiko-chan is ready → {scheme}://{host_ip}:{HTTP_PORT}/\n")
    print("  Booting subsystems now — browsers can log in while Aiko wakes up.\n")

    run_session(ui, args)
