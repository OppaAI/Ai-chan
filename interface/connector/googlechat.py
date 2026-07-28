from __future__ import annotations

import json
import os
import threading
import time

import requests

from system.log import get_logger
from interface.connector.base import ConnectorBase

log = get_logger(__name__)

try:
    from google.oauth2 import service_account
    import google.auth
except ImportError:
    service_account = None
    google = None

SCOPES = ["https://www.googleapis.com/auth/chat.bot"]


class GoogleChatConnector(ConnectorBase):
    """Google Chat bot connector using service account authentication (polling).

    Requires a Google Cloud service account with the Google Chat API enabled.
    The bot must be published to your Google Workspace domain.

    Requires:
        GOOGLE_CHAT_SERVICE_ACCOUNT — path to the service account JSON key file
        GOOGLE_CHAT_BOT_NAME        — the bot's name as shown in space membership
    """

    POLL_INTERVAL = 10.0
    API_BASE = "https://chat.googleapis.com/v1"

    def __init__(self, config: dict[str, str] | None = None) -> None:
        super().__init__(config)
        self._token: str = ""
        self._token_expiry: float = 0.0
        self._bot_name: str = ""
        self._sa_path: str = ""
        self._sa_info: dict | None = None
        self._seen_messages: set[str] = set()
        self._poll_thread: threading.Thread | None = None
        self._stop_evt = threading.Event()

    def _read_config(self) -> dict[str, str]:
        return {
            "sa_path": self._get_env("GOOGLE_CHAT_SERVICE_ACCOUNT"),
            "bot_name": self._get_env("GOOGLE_CHAT_BOT_NAME", "Aiko-chan"),
        }

    def _ensure_token(self) -> bool:
        if time.time() < self._token_expiry - 60 and self._token:
            return True
        if service_account is None:
            log.error("[googlechat] google-auth not installed. pip install google-auth")
            return False
        try:
            if self._sa_info:
                creds = service_account.Credentials.from_service_account_info(
                    self._sa_info, scopes=SCOPES
                )
            else:
                creds = service_account.Credentials.from_service_account_file(
                    self._sa_path, scopes=SCOPES
                )
            from google.auth.transport.requests import Request
            creds.refresh(Request())
            self._token = creds.token
            self._token_expiry = creds.expiry.timestamp() if creds.expiry else time.time() + 3600
            return True
        except Exception as exc:
            log.error("[googlechat] Token refresh failed: %s", exc)
            return False

    def _api_get(self, path: str, params: dict | None = None) -> dict | None:
        if not self._ensure_token():
            return None
        url = f"{self.API_BASE}/{path.lstrip('/')}"
        try:
            resp = requests.get(
                url,
                headers={"Authorization": f"Bearer {self._token}"},
                params=params,
                timeout=15,
            )
            if resp.status_code == 401:
                self._token = ""
                if not self._ensure_token():
                    return None
                resp = requests.get(
                    url,
                    headers={"Authorization": f"Bearer {self._token}"},
                    params=params,
                    timeout=15,
                )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            log.error("[googlechat] GET %s failed: %s", path, exc)
            return None

    def _api_post(self, path: str, body: dict) -> dict | None:
        if not self._ensure_token():
            return None
        url = f"{self.API_BASE}/{path.lstrip('/')}"
        try:
            resp = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {self._token}",
                    "Content-Type": "application/json",
                },
                json=body,
                timeout=15,
            )
            if resp.status_code == 401:
                self._token = ""
                if not self._ensure_token():
                    return None
                resp = requests.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {self._token}",
                        "Content-Type": "application/json",
                    },
                    json=body,
                    timeout=15,
                )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            log.error("[googlechat] POST %s failed: %s", path, exc)
            return None

    def start(self) -> None:
        cfg = self._read_config()
        if not cfg.get("sa_path"):
            log.error("[googlechat] GOOGLE_CHAT_SERVICE_ACCOUNT must be set in .env")
            return

        self._sa_path = cfg["sa_path"]
        self._bot_name = cfg["bot_name"]

        # Load service account JSON
        try:
            with open(self._sa_path) as f:
                self._sa_info = json.load(f)
        except Exception as exc:
            log.error("[googlechat] Failed to load service account: %s", exc)
            return

        if not self._ensure_token():
            return

        log.info("[googlechat] Authenticated as service account")

        self._stop_evt.clear()
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()
        self._running = True
        log.info("[googlechat] Connector started (polling every %.0fs)", self.POLL_INTERVAL)

    def stop(self) -> None:
        self._running = False
        self._stop_evt.set()

    def send_message(self, conversation_id: str, text: str) -> None:
        space_name = conversation_id
        self._api_post(f"{space_name}/messages", {"text": text})

    def _poll_loop(self) -> None:
        while not self._stop_evt.is_set():
            try:
                self._check_spaces()
            except Exception as exc:
                log.error("[googlechat] Poll error: %s", exc)
            self._stop_evt.wait(timeout=self.POLL_INTERVAL)

    def _check_spaces(self) -> None:
        spaces_data = self._api_get("spaces", {"pageSize": 50})
        if not spaces_data:
            return
        spaces = spaces_data.get("spaces", [])
        for space in spaces:
            space_name = space.get("name", "")
            if not space_name:
                continue
            self._check_space_messages(space_name)

    def _check_space_messages(self, space_name: str) -> None:
        params: dict = {"pageSize": 20, "orderBy": "createTime desc"}
        data = self._api_get(f"{space_name}/messages", params)
        if not data:
            return
        messages = data.get("messages", [])
        for msg in messages:
            msg_name = msg.get("name", "")
            if not msg_name or msg_name in self._seen_messages:
                continue
            self._seen_messages.add(msg_name)

            sender = msg.get("sender", {})
            sender_resource = sender.get("name", "")
            if not sender_resource or sender_resource.startswith("users/app"):
                continue

            text = ""
            text_obj = msg.get("text", "")
            if text_obj:
                text = text_obj.strip()
            if not text:
                continue

            uid = sender_resource
            display = sender.get("displayName", uid)

            log.info("[googlechat] msg from %s in %s: %.60s", display, space_name, text)
            session = self.handle_message(space_name, uid, display, text)
            session.wait()

        if len(self._seen_messages) > 10000:
            self._seen_messages = set(list(self._seen_messages)[-5000:])
