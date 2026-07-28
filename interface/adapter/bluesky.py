from __future__ import annotations

import os
import threading
import time

from system.log import get_logger
from interface.adapter.base import AdapterBase

log = get_logger(__name__)

try:
    from atproto import Client, models
    from atproto.exceptions import AtprotoError
except ImportError:
    Client = None
    models = None


class BlueskyAdapter(AdapterBase):
    """Bluesky bot adapter using the AT Protocol (atproto library).

    Polls notifications for mentions/replies and responds in-thread.

    Requires:
        BLUESKY_HANDLE   — e.g. myhandle.bsky.social
        BLUESKY_APP_PASS — App password from Settings > App Passwords
    """

    POLL_INTERVAL = 30.0

    def __init__(self, config: dict[str, str] | None = None) -> None:
        super().__init__(config)
        self._client: Client | None = None
        self._did: str = ""
        self._poll_thread: threading.Thread | None = None
        self._stop_evt = threading.Event()
        self._reply_refs: dict[str, models.AppBskyFeedPost.ReplyRef] = {}
        self._ref_lock = threading.Lock()

    def _read_config(self) -> dict[str, str]:
        return {
            "handle": self._get_env("BLUESKY_HANDLE"),
            "app_pass": self._get_env("BLUESKY_APP_PASS"),
        }

    def start(self) -> None:
        if Client is None:
            log.error("[bluesky] atproto not installed. pip install atproto")
            return
        cfg = self._read_config()
        if not cfg.get("handle") or not cfg.get("app_pass"):
            log.error("[bluesky] BLUESKY_HANDLE and BLUESKY_APP_PASS must be set in .env")
            return

        self._client = Client()
        try:
            profile = self._client.login(cfg["handle"], cfg["app_pass"])
            self._did = profile.did
            log.info("[bluesky] Logged in as %s (%s)", profile.handle, self._did)
        except Exception as exc:
            log.error("[bluesky] Login failed: %s", exc)
            return

        self._stop_evt.clear()
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()
        self._running = True
        log.info("[bluesky] Adapter started (polling every %.0fs)", self.POLL_INTERVAL)

    def stop(self) -> None:
        self._running = False
        self._stop_evt.set()

    def send_message(self, conversation_id: str, text: str) -> None:
        if not self._client:
            return
        with self._ref_lock:
            reply_ref = self._reply_refs.pop(conversation_id, None)
        try:
            if reply_ref is not None:
                self._client.send_post(text=text, reply_to=reply_ref)
            else:
                self._client.send_post(text=text)
        except Exception as exc:
            log.error("[bluesky] Failed to reply to %s: %s", conversation_id, exc)

    def _poll_loop(self) -> None:
        while not self._stop_evt.is_set():
            try:
                self._check_notifications()
            except Exception as exc:
                log.error("[bluesky] Poll error: %s", exc)
            self._stop_evt.wait(timeout=self.POLL_INTERVAL)

    def _check_notifications(self) -> None:
        if not self._client:
            return
        try:
            cursor = None
            while True:
                params: dict = {"limit": 50}
                if cursor:
                    params["cursor"] = cursor
                notifs = self._client.app.bsky.notification.list(params)
                if not notifs or not notifs.notifications:
                    break
                for notif in reversed(notifs.notifications):
                    cursor = notif.cursor
                    if notif.reason != "mention":
                        continue
                    if notif.author.did == self._did:
                        continue
                    if notif.is_read:
                        continue
                    uid = notif.author.did
                    display = notif.author.display_name or notif.author.handle or uid
                    text = ""
                    if notif.record and hasattr(notif.record, "text"):
                        text = (notif.record.text or "").strip()
                    post_uri = notif.uri
                    if not text or not post_uri:
                        continue
                    log.info("[bluesky] mention from @%s: %.60s", notif.author.handle, text)
                    # Store reply reference so send_message can construct an in-thread reply
                    if notif.cid and notif.uri:
                        ref = models.AppBskyFeedPost.ReplyRef(
                            parent=models.create_strong_ref(notif.uri, notif.cid),
                            root=models.create_strong_ref(notif.uri, notif.cid),
                        )
                        with self._ref_lock:
                            self._reply_refs[post_uri] = ref
                    session = self.handle_message(post_uri, uid, display, text)
                    session.wait()
            self._client.app.bsky.notification.update_seen(
                {"seen_at": time.time()}
            )
        except AttributeError:
            pass
        except Exception as exc:
            log.error("[bluesky] Notification check error: %s", exc)
