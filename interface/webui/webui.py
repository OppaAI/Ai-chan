"""
webui/webui.py
Aiko-chan's browser-based UI backend — drop-in replacement for AikoTUI.

Barge-in WebSocket messages are ignored when BARGE_IN_ENABLED is off;
mic start payload includes barge_in_enabled and echo_guard_ms for the browser.
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
    try:
        from sensory.listen_native import barge_in_enabled
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
                    "-subj", f"/CN:{hostname}",
                    "-addext", f"subjectAltName={','.join(alt_names)}",
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            log.info("[aiko-web] generated self-signed TLS cert at %s", cert_path)
        except Exception as exp:
            raise RuntimeError(
                "WEBUI_HTTPS=1 requires openssl or SSL_CERT/SSL_KEY pointing at an existing certificate."
            ) from exp

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(certfile=cert_path, keyfile=key_path)
    ctx.set_alpn_protocols(["http/1.1"])
    return ctx
