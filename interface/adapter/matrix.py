from __future__ import annotations

import asyncio
import os
import threading
from pathlib import Path

from system.log import get_logger
from interface.adapter.base import AdapterBase

log = get_logger(__name__)

try:
    from nio import (
        AsyncClient,
        MatrixRoom,
        RoomMessageText,
        LoginResponse,
        SyncResponse,
    )
except ImportError:
    AsyncClient = None


class MatrixAdapter(AdapterBase):
    """Matrix bot adapter using matrix-nio.

    Requires:
        MATRIX_HOMESERVER   — e.g. https://matrix.org
        MATRIX_USER         — @user:homeserver.tld
        MATRIX_PASSWORD     — account password
        MATRIX_DEVICE_ID    — optional, persistent device name
    """

    STORE_DIR = "matrix_store"

    def __init__(self, config: dict[str, str] | None = None) -> None:
        super().__init__(config)
        self._client: AsyncClient | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._stop_evt = threading.Event()
        self._room_map: dict[str, str] = {}
        self._user_id_map: dict[str, str] = self._parse_user_map(
            self._get_env("MATRIX_USER_ID_MAP")
        )

    def _read_config(self) -> dict[str, str]:
        return {
            "homeserver": self._get_env("MATRIX_HOMESERVER", "https://matrix.org"),
            "user": self._get_env("MATRIX_USER"),
            "password": self._get_env("MATRIX_PASSWORD"),
            "device_id": self._get_env("MATRIX_DEVICE_ID", "aiko-adapter"),
        }

    @staticmethod
    def _parse_user_map(raw: str) -> dict[str, str]:
        mapping: dict[str, str] = {}
        if not raw:
            return mapping
        for token in raw.replace(",", " ").split():
            if "=" in token:
                key, val = token.split("=", 1)
            elif ":" in token:
                key, val = token.split(":", 1)
            else:
                continue
            key, val = key.strip(), val.strip()
            if key and val:
                mapping[key] = val
        return mapping

    def _canonical_user_id(self, raw: str) -> str:
        if raw not in self._user_id_map:
            log.debug("[matrix] user %s -> %s (default)", raw, raw)
            return raw
        mapped = self._user_id_map[raw]
        log.info("[matrix] user %s -> %s (mapped)", raw, mapped)
        return mapped

    def start(self) -> None:
        if AsyncClient is None:
            log.error("[matrix] matrix-nio not installed. pip install matrix-nio")
            return
        cfg = self._read_config()
        if not cfg.get("user") or not cfg.get("password"):
            log.error("[matrix] MATRIX_USER and MATRIX_PASSWORD must be set in .env")
            return

        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run_async, daemon=True)
        self._thread.start()
        self._running = True
        log.info("[matrix] Adapter started")

    def _run_async(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._async_main())

    async def _async_main(self) -> None:
        cfg = self._read_config()
        store_path = Path(self.STORE_DIR)
        store_path.mkdir(parents=True, exist_ok=True)

        self._client = AsyncClient(
            cfg["homeserver"],
            cfg["user"],
            store_path=str(store_path),
            device_id=cfg["device_id"],
        )

        resp = await self._client.login(cfg["password"])
        if not isinstance(resp, LoginResponse):
            log.error("[matrix] Login failed: %s", resp)
            return
        log.info("[matrix] Logged in as %s (device: %s)", resp.user_id, resp.device_id)

        async def message_cb(room: MatrixRoom, event: RoomMessageText) -> None:
            if event.sender == self._client.user_id:
                return
            cid = room.room_id
            uid = self._canonical_user_id(event.sender)
            display = event.sender
            text = event.body.strip()
            if not text:
                return
            self._room_map[uid] = cid
            log.info("[matrix] msg from %s in %s: %.60s", uid, cid, text)
            # Do NOT block on session.wait(): this callback runs on matrix-nio's
            # asyncio sync loop; blocking it during inference starves the sync
            # loop. The session runs inference on its own thread and delivers
            # the reply via _on_platform_response -> send_message, which
            # schedules back onto this loop with run_coroutine_threadsafe.
            self.handle_message(cid, uid, display, text)

        self._client.add_event_callback(message_cb, RoomMessageText)

        while not self._stop_evt.is_set():
            try:
                sync_resp = await self._client.sync(timeout=30000)
            except Exception as exc:
                log.error("[matrix] Sync error: %s", exc)
                await asyncio.sleep(5)
                continue

    def stop(self) -> None:
        self._running = False
        self._stop_evt.set()
        if self._client and self._loop:
            try:
                asyncio.run_coroutine_threadsafe(
                    self._client.close(), self._loop
                )
            except Exception:
                log.warning("matrix: async client close failed")

    def send_message(self, conversation_id: str, text: str) -> None:
        if not self._client or not self._loop:
            return
        try:
            asyncio.run_coroutine_threadsafe(
                self._client.room_send(
                    room_id=conversation_id,
                    message_type="m.room.message",
                    content={"msgtype": "m.text", "body": text},
                ),
                self._loop,
            )
        except Exception as exc:
            log.error("[matrix] Failed to send to %s: %s", conversation_id, exc)
