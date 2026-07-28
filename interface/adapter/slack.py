from __future__ import annotations

import os
import threading

from system.log import get_logger
from interface.adapter.base import AdapterBase

log = get_logger(__name__)

try:
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError
    from slack_sdk.socket_mode import SocketModeClient
    from slack_sdk.socket_mode.request import SocketModeRequest
except ImportError:
    WebClient = None


class SlackAdapter(AdapterBase):
    """Slack bot adapter using slack-sdk with Socket Mode (no public HTTP endpoint needed).

    Requires:
        SLACK_BOT_TOKEN       — xoxb-* token
        SLACK_APP_TOKEN       — xapp-* token (from Slack App > Socket Mode)
    """

    def __init__(self, config: dict[str, str] | None = None) -> None:
        super().__init__(config)
        self._web_client: WebClient | None = None
        self._socket_client: SocketModeClient | None = None
        self._thread: threading.Thread | None = None
        self._bot_user_id: str | None = None

    def _read_config(self) -> dict[str, str]:
        return {
            "bot_token": self._get_env("SLACK_BOT_TOKEN"),
            "app_token": self._get_env("SLACK_APP_TOKEN"),
        }

    def start(self) -> None:
        if WebClient is None:
            log.error("[slack] slack-sdk not installed. pip install slack-sdk")
            return
        cfg = self._read_config()
        if not cfg.get("bot_token") or not cfg.get("app_token"):
            log.error("[slack] SLACK_BOT_TOKEN and SLACK_APP_TOKEN must be set in .env")
            return

        self._web_client = WebClient(token=cfg["bot_token"])
        auth_test = self._web_client.auth_test()
        self._bot_user_id = auth_test.get("user_id", "")
        log.info("[slack] Connected as %s", auth_test.get("user", "?"))

        self._socket_client = SocketModeClient(
            app_token=cfg["app_token"],
            web_client=self._web_client,
        )

        @self._socket_client.on("events_api")
        def handle_event(client: SocketModeClient, req: SocketModeRequest) -> None:
            if req.type != "events_api":
                return
            payload = req.payload
            if not payload or "event" not in payload:
                return
            event = payload["event"]
            if event.get("type") != "message":
                return
            if event.get("subtype") in ("bot_message", "message_changed", "message_deleted"):
                return
            text = (event.get("text") or "").strip()
            if not text:
                return
            user = event.get("user", "")
            if not user or user == self._bot_user_id:
                return
            channel = event.get("channel", "")
            thread_ts = event.get("thread_ts") or event.get("ts", "")
            cid = channel
            uid = user
            display = f"<@{user}>"
            log.info("[slack] msg from %s in %s: %.60s", uid, channel, text)
            session = self.handle_message(cid, uid, display, text)
            session.wait()

        self._thread = threading.Thread(
            target=self._socket_client.connect, daemon=True
        )
        self._thread.start()
        self._running = True
        log.info("[slack] Adapter started")

    def stop(self) -> None:
        self._running = False
        if self._socket_client:
            try:
                self._socket_client.disconnect()
            except Exception:
                pass

    def send_message(self, conversation_id: str, text: str) -> None:
        if not self._web_client:
            return
        try:
            self._web_client.chat_postMessage(
                channel=conversation_id,
                text=text,
            )
        except SlackApiError as exc:
            log.error("[slack] Failed to send to %s: %s", conversation_id, exc)
