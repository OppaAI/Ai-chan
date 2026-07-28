from __future__ import annotations

import os
import threading
import time
from urllib.parse import urlencode

import requests

from system.log import get_logger
from interface.adapter.base import ConnectorBase

log = get_logger(__name__)


class YouTubeConnector(ConnectorBase):
    """YouTube live chat & comments bot connector.

    Polls the active live broadcast's chat for new messages and replies
    in the live chat, or monitors video comments.

    Requires (same as existing YouTube upload flow):
        YOUTUBE_CLIENT_ID
        YOUTUBE_CLIENT_SECRET
        YOUTUBE_REFRESH_TOKEN
    """

    POLL_INTERVAL = 15.0
    TOKEN_URL = "https://oauth2.googleapis.com/token"

    def __init__(self, config: dict[str, str] | None = None) -> None:
        super().__init__(config)
        self._access_token: str = ""
        self._client_id: str = ""
        self._client_secret: str = ""
        self._refresh_token: str = ""
        self._channel_id: str = ""
        self._live_chat_id: str | None = None
        self._next_page_token: str | None = None
        self._poll_thread: threading.Thread | None = None
        self._stop_evt = threading.Event()
        self._processed_ids: set[str] = set()

    def _read_config(self) -> dict[str, str]:
        return {
            "client_id": self._get_env("YOUTUBE_CLIENT_ID"),
            "client_secret": self._get_env("YOUTUBE_CLIENT_SECRET"),
            "refresh_token": self._get_env("YOUTUBE_REFRESH_TOKEN"),
            "channel_id": self._get_env("YOUTUBE_CHANNEL_ID"),
        }

    def _refresh_access_token(self) -> bool:
        try:
            resp = requests.post(
                self.TOKEN_URL,
                data={
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                    "refresh_token": self._refresh_token,
                    "grant_type": "refresh_token",
                },
                timeout=30,
            )
            resp.raise_for_status()
            self._access_token = resp.json()["access_token"]
            return True
        except Exception as exc:
            log.error("[youtube] Token refresh failed: %s", exc)
            return False

    def _api_get(self, path: str, params: dict | None = None) -> dict | None:
        if not self._access_token:
            if not self._refresh_access_token():
                return None
        url = f"https://www.googleapis.com/youtube/v3/{path.lstrip('/')}"
        headers = {"Authorization": f"Bearer {self._access_token}"}
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=30)
            if resp.status_code == 401:
                if self._refresh_access_token():
                    headers["Authorization"] = f"Bearer {self._access_token}"
                    resp = requests.get(url, headers=headers, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            log.error("[youtube] API GET %s failed: %s", path, exc)
            return None

    def _api_post(self, path: str, body: dict) -> bool:
        if not self._access_token:
            if not self._refresh_access_token():
                return False
        url = f"https://www.googleapis.com/youtube/v3/{path.lstrip('/')}"
        headers = {"Authorization": f"Bearer {self._access_token}", "Content-Type": "application/json"}
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=30)
            if resp.status_code == 401:
                if self._refresh_access_token():
                    headers["Authorization"] = f"Bearer {self._access_token}"
                    resp = requests.post(url, headers=headers, json=body, timeout=30)
            resp.raise_for_status()
            return True
        except Exception as exc:
            log.error("[youtube] API POST %s failed: %s", path, exc)
            return False

    def start(self) -> None:
        cfg = self._read_config()
        missing = [k for k in ("client_id", "client_secret", "refresh_token") if not cfg.get(k)]
        if missing:
            log.error("[youtube] Missing env vars: %s", ", ".join(missing))
            return
        self._client_id = cfg["client_id"]
        self._client_secret = cfg["client_secret"]
        self._refresh_token = cfg["refresh_token"]
        self._channel_id = cfg.get("channel_id", "")

        if not self._refresh_access_token():
            return

        log.info("[youtube] Authenticated (channel_id=%s)", self._channel_id or "auto-detect")

        self._stop_evt.clear()
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()
        self._running = True
        log.info("[youtube] Connector started (polling every %.0fs)", self.POLL_INTERVAL)

    def stop(self) -> None:
        self._running = False
        self._stop_evt.set()

    def send_message(self, conversation_id: str, text: str) -> None:
        if not self._access_token:
            return
        kind, cid = conversation_id.split(":", 1) if ":" in conversation_id else ("chat", conversation_id)
        if kind == "chat":
            self._api_post(
                "liveChat/messages",
                {
                    "snippet": {
                        "liveChatId": cid,
                        "type": "textMessageEvent",
                        "textMessageDetails": {"messageText": text},
                    }
                },
            )
        elif kind == "comment":
            self._api_post(
                "comments",
                {"snippet": {"parentId": cid, "textOriginal": text}},
            )

    def _poll_loop(self) -> None:
        # First, find a live broadcast
        while not self._stop_evt.is_set():
            self._find_live_chat()
            if self._live_chat_id:
                break
            self._stop_evt.wait(timeout=60.0)

        while not self._stop_evt.is_set():
            if self._live_chat_id:
                try:
                    self._poll_live_chat()
                except Exception as exc:
                    log.error("[youtube] Live chat poll error: %s", exc)
            self._stop_evt.wait(timeout=self.POLL_INTERVAL)

    def _find_live_chat(self) -> None:
        params = {"part": "snippet", "broadcastType": "active", "mine": True, "maxResults": 1}
        data = self._api_get("liveBroadcasts", params)
        if not data or "items" not in data or not data["items"]:
            log.info("[youtube] No active live broadcast found")
            self._live_chat_id = None
            return
        item = data["items"][0]
        self._live_chat_id = item["snippet"].get("liveChatId")
        if self._live_chat_id:
            log.info("[youtube] Found live chat: %s", self._live_chat_id)

    def _poll_live_chat(self) -> None:
        if not self._live_chat_id:
            return
        params: dict = {
            "part": "snippet,authorDetails",
            "liveChatId": self._live_chat_id,
            "maxResults": 200,
        }
        if self._next_page_token:
            params["pageToken"] = self._next_page_token
        data = self._api_get("liveChat/messages", params)
        if not data:
            return
        self._next_page_token = data.get("nextPageToken")
        self._live_chat_id = data.get("nextPageToken") or data.get("pollingIntervalMillis") or self._live_chat_id
        if data.get("pollingIntervalMillis"):
            self.POLL_INTERVAL = max(5.0, data["pollingIntervalMillis"] / 1000.0)

        items = data.get("items", [])
        for msg in items:
            msg_id = msg.get("id", "")
            if not msg_id or msg_id in self._processed_ids:
                continue
            self._processed_ids.add(msg_id)
            snippet = msg.get("snippet", {})
            author = msg.get("authorDetails", {})
            author_channel_id = author.get("channelId", "")
            uid = author_channel_id or author.get("displayName", "unknown")
            display = author.get("displayName", uid)
            text = snippet.get("displayMessage", "").strip()
            if not text or not msg_id:
                continue
            if author.get("isChatOwner") or author.get("isChatModerator"):
                pass  # still respond to owner/moderator
            if author_channel_id and author_channel_id == self._channel_id:
                continue
            log.info("[youtube] live msg from %s: %.60s", display, text)
            cid = f"chat:{self._live_chat_id}"
            session = self.handle_message(cid, uid, display, text)
            session.wait()

            # Trim processed set to avoid unbounded growth
            if len(self._processed_ids) > 10000:
                self._processed_ids = set(list(self._processed_ids)[-5000:])
