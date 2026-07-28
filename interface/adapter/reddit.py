from __future__ import annotations

import os
import re
import threading
import time

from system.log import get_logger
from interface.adapter.base import AdapterBase

log = get_logger(__name__)

try:
    import praw
except ImportError:
    praw = None


class RedditAdapter(AdapterBase):
    """Reddit bot adapter using PRAW.

    Streams inbox mentions and replies to them.

    Requires:
        REDDIT_CLIENT_ID     — from reddit.com/prefs/apps (script app)
        REDDIT_CLIENT_SECRET — from the same app page
        REDDIT_USERNAME      — bot account username
        REDDIT_PASSWORD      — bot account password
        REDDIT_USER_AGENT    — e.g. "Aiko-chan bot v0.1 by u/yourname"
    """

    POLL_INTERVAL = 15.0

    def __init__(self, config: dict[str, str] | None = None) -> None:
        super().__init__(config)
        self._reddit: praw.Reddit | None = None
        self._username: str = ""
        self._poll_thread: threading.Thread | None = None
        self._stop_evt = threading.Event()
        self._processed_ids: set[str] = set()

    def _read_config(self) -> dict[str, str]:
        return {
            "client_id": self._get_env("REDDIT_CLIENT_ID"),
            "client_secret": self._get_env("REDDIT_CLIENT_SECRET"),
            "username": self._get_env("REDDIT_USERNAME"),
            "password": self._get_env("REDDIT_PASSWORD"),
            "user_agent": self._get_env("REDDIT_USER_AGENT", "Aiko-chan adapter v0.1"),
        }

    def start(self) -> None:
        if praw is None:
            log.error("[reddit] praw not installed. pip install praw")
            return
        cfg = self._read_config()
        missing = [k for k, v in cfg.items() if not v]
        if missing:
            log.error("[reddit] Missing env vars: %s", ", ".join(missing))
            return

        self._reddit = praw.Reddit(
            client_id=cfg["client_id"],
            client_secret=cfg["client_secret"],
            username=cfg["username"],
            password=cfg["password"],
            user_agent=cfg["user_agent"],
        )
        self._username = cfg["username"]
        try:
            me = self._reddit.user.me()
            log.info("[reddit] Connected as u/%s", me.name)
        except Exception as exc:
            log.error("[reddit] Auth failed: %s", exc)
            return

        self._stop_evt.clear()
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()
        self._running = True
        log.info("[reddit] Adapter started (polling every %.0fs)", self.POLL_INTERVAL)

    def stop(self) -> None:
        self._running = False
        self._stop_evt.set()

    def send_message(self, conversation_id: str, text: str) -> None:
        if not self._reddit:
            return
        try:
            # conversation_id = fullname of the item (t1_xxx for comment, t3_xxx for submission)
            item = self._reddit.comment(conversation_id)
            item.reply(text)
        except Exception as exc:
            log.error("[reddit] Failed to reply to %s: %s", conversation_id, exc)

    def _poll_loop(self) -> None:
        while not self._stop_evt.is_set():
            try:
                self._check_inbox()
            except Exception as exc:
                log.error("[reddit] Poll error: %s", exc)
            self._stop_evt.wait(timeout=self.POLL_INTERVAL)

    def _check_inbox(self) -> None:
        if not self._reddit:
            return
        for item in self._reddit.inbox.unread(limit=25):
            item_id = item.fullname
            if item_id in self._processed_ids:
                continue
            self._processed_ids.add(item_id)

            text = ""
            uid = ""
            display = ""
            cid = ""

            if isinstance(item, praw.models.Comment):
                text = item.body or ""
                author = item.author
                if author and author.name == self._username:
                    continue
                uid = author.name if author else "unknown"
                display = uid
                cid = item.fullname
                context_text = ""
                if item.parent():
                    try:
                        parent = item.parent()
                        context_text = parent.body if hasattr(parent, "body") else ""
                    except Exception:
                        pass
                log.info("[reddit] mention from u/%s in %s: %.60s", uid, item.subreddit, text)
            elif isinstance(item, praw.models.Message):
                text = item.body or ""
                author = item.author
                if author and author.name == self._username:
                    continue
                uid = author.name if author else "unknown"
                display = uid
                cid = item.fullname
                log.info("[reddit] PM from u/%s: %.60s", uid, text)
            else:
                continue

            if not text or not cid:
                continue

            session = self.handle_message(cid, uid, display, text)
            session.wait()

            try:
                item.mark_read()
            except Exception:
                pass

            if len(self._processed_ids) > 10000:
                self._processed_ids = set(list(self._processed_ids)[-5000:])
