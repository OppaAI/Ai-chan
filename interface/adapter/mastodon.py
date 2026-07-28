from __future__ import annotations

import os
import threading
import time

from system.log import get_logger
from interface.adapter.base import ConnectorBase

log = get_logger(__name__)

try:
    from mastodon import Mastodon as MastodonClient, StreamListener
except ImportError:
    MastodonClient = None
    StreamListener = object


class _MastodonListener(StreamListener):
    def __init__(self, connector: MastodonConnector) -> None:
        super().__init__()
        self._connector = connector

    def on_notification(self, notification) -> None:
        if notification.type != "mention":
            return
        status = notification.status
        if not status or not status.content:
            return
        acct = notification.account
        if not acct:
            return
        uid = str(acct.id)
        display = acct.display_name or acct.username or uid
        text = status.content
        # Strip HTML tags from content
        import re
        text = re.sub(r"<[^>]+>", "", text).strip()
        if not text:
            return
        cid = str(status.id)
        log.info("[mastodon] mention from @%s: %.60s", acct.username, text)
        session = self._connector.handle_message(cid, uid, display, text)
        session.wait()

    def handle_heartbeat(self) -> None:
        pass


class MastodonConnector(ConnectorBase):
    """Mastodon bot connector using Mastodon.py (streaming API).

    Listens for mentions via the streaming API and replies in-thread.

    Requires:
        MASTODON_INSTANCE  — e.g. https://mastodon.social
        MASTODON_ACCESS_TOKEN — access token from Preferences > Development
    """

    def __init__(self, config: dict[str, str] | None = None) -> None:
        super().__init__(config)
        self._client: MastodonClient | None = None
        self._listener: _MastodonListener | None = None
        self._stream_thread: threading.Thread | None = None
        self._stop_evt = threading.Event()

    def _read_config(self) -> dict[str, str]:
        return {
            "instance": self._get_env("MASTODON_INSTANCE", "https://mastodon.social"),
            "access_token": self._get_env("MASTODON_ACCESS_TOKEN"),
        }

    def start(self) -> None:
        if MastodonClient is None:
            log.error("[mastodon] Mastodon.py not installed. pip install Mastodon.py")
            return
        cfg = self._read_config()
        if not cfg.get("access_token"):
            log.error("[mastodon] MASTODON_ACCESS_TOKEN must be set in .env")
            return

        self._client = MastodonClient(
            api_base_url=cfg["instance"],
            access_token=cfg["access_token"],
        )
        try:
            acct = self._client.account_verify_credentials()
            log.info("[mastodon] Connected as @%s", acct.username)
        except Exception as exc:
            log.error("[mastodon] Auth failed: %s", exc)
            return

        self._listener = _MastodonListener(self)
        self._stop_evt.clear()
        self._stream_thread = threading.Thread(target=self._run_stream, daemon=True)
        self._stream_thread.start()
        self._running = True
        log.info("[mastodon] Connector started (streaming)")

    def stop(self) -> None:
        self._running = False
        self._stop_evt.set()
        if self._client:
            try:
                self._client.stream_end()
            except Exception:
                pass

    def send_message(self, conversation_id: str, text: str) -> None:
        if not self._client:
            return
        try:
            status_id = int(conversation_id)
            self._client.status_post(text, in_reply_to_id=status_id)
        except Exception as exc:
            log.error("[mastodon] Failed to reply to %s: %s", conversation_id, exc)

    def _run_stream(self) -> None:
        if not self._client or not self._listener:
            return
        while not self._stop_evt.is_set():
            try:
                self._client.stream_user(self._listener, run_async=False)
            except Exception as exc:
                if self._stop_evt.is_set():
                    break
                log.error("[mastodon] Stream error, reconnecting in 10s: %s", exc)
                self._stop_evt.wait(timeout=10.0)
