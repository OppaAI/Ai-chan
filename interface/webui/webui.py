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
