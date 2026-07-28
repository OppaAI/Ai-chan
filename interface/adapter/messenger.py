from __future__ import annotations

import os
import threading
import time

import requests

from system.log import get_logger
from interface.adapter.base import AdapterBase

log = get_logger(__name__)


class MessengerAdapter(AdapterBase):
    """Facebook Messenger bot adapter via the Graph API (polling).

    Requires:
        FB_PAGE_ID          — your Facebook Page ID
        FB_PAGE_ACCESS_TOKEN — Page-scoped access token from Meta App
        FB_API_VERSION       — optional, defaults to v21.0
    """

    POLL_INTERVAL = 5.0
    API_BASE = "https://graph.facebook.com"

    def __init__(self, config: dict[str, str] | None = None) -> None:
        super().__init__(config)
        self._page_id: str = ""
        self._token: str = ""
        self._api_ver: str = ""
        self._seen_messages: set[str] = set()
        self._poll_thread: threading.Thread | None = None
        self._stop_evt = threading.Event()

    def _read_config(self) -> dict[str, str]:
        return {
            "page_id": self._get_env("FB_PAGE_ID"),
            "token": self._get_env("FB_PAGE_ACCESS_TOKEN"),
            "api_ver": self._get_env("FB_API_VERSION", "v21.0"),
        }

    def _api_url(self, path: str) -> str:
        return f"{self.API_BASE}/{self._api_ver}/{path.lstrip('/')}"

    def start(self) -> None:
        cfg = self._read_config()
        missing = [k for k in ("page_id", "token") if not cfg.get(k)]
        if missing:
            log.error("[messenger] FB_PAGE_ID and FB_PAGE_ACCESS_TOKEN must be set in .env")
            return
        self._page_id = cfg["page_id"]
        self._token = cfg["token"]
        self._api_ver = cfg["api_ver"]

        # Quick auth check
        try:
            resp = requests.get(
                self._api_url(f"{self._page_id}"),
                params={"access_token": self._token, "fields": "name"},
                timeout=15,
            )
            resp.raise_for_status()
            page_name = resp.json().get("name", "?")
            log.info("[messenger] Connected as page '%s' (id=%s)", page_name, self._page_id)
        except Exception as exc:
            log.error("[messenger] Auth check failed: %s", exc)
            return

        self._stop_evt.clear()
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()
        self._running = True
        log.info("[messenger] Adapter started (polling every %.0fs)", self.POLL_INTERVAL)

    def stop(self) -> None:
        self._running = False
        self._stop_evt.set()

    def send_message(self, conversation_id: str, text: str) -> None:
        psid = conversation_id
        try:
            resp = requests.post(
                self._api_url("me/messages"),
                params={"access_token": self._token},
                json={
                    "recipient": {"id": psid},
                    "message": {"text": text},
                },
                timeout=15,
            )
            resp.raise_for_status()
        except Exception as exc:
            log.error("[messenger] Failed to send to %s: %s", psid, exc)

    def _poll_loop(self) -> None:
        while not self._stop_evt.is_set():
            try:
                self._check_conversations()
            except Exception as exc:
                log.error("[messenger] Poll error: %s", exc)
            self._stop_evt.wait(timeout=self.POLL_INTERVAL)

    def _check_conversations(self) -> None:
        resp = requests.get(
            self._api_url(f"{self._page_id}/conversations"),
            params={
                "access_token": self._token,
                "fields": "messages.limit(10){message,from,created_time,id}",
                "limit": 10,
            },
            timeout=15,
        )
        if resp.status_code != 200:
            return
        data = resp.json()
        convos = data.get("data", [])
        for convo in convos:
            messages = convo.get("messages", {}).get("data", [])
            for msg in messages:
                msg_id = msg.get("id", "")
                if not msg_id or msg_id in self._seen_messages:
                    continue
                self._seen_messages.add(msg_id)
                from_ = msg.get("from", {})
                sender_id = from_.get("id", "")
                if sender_id == self._page_id:
                    continue
                text = (msg.get("message") or "").strip()
                if not text:
                    continue
                display = from_.get("name", sender_id)
                log.info("[messenger] msg from %s: %.60s", display, text)
                session = self.handle_message(sender_id, sender_id, display, text)
                session.wait()

        if len(self._seen_messages) > 10000:
            self._seen_messages = set(list(self._seen_messages)[-5000:])
