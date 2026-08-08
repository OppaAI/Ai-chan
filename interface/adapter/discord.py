from __future__ import annotations

import threading

from system.log import get_logger
from interface.adapter.base import AdapterBase

log = get_logger(__name__)

try:
    import discord
    from discord import Intents
except ImportError:
    discord = None


class DiscordAdapter(AdapterBase):
    """Discord bot adapter using discord.py."""

    def __init__(self, config: dict[str, str] | None = None) -> None:
        super().__init__(config)
        self._token: str = ""
        self._client: discord.Client | None = None
        self._thread: threading.Thread | None = None
        self._channel_map: dict[int, str] = {}
        self._user_id_map: dict[str, str] = self._parse_user_map(
            self._get_env("DISCORD_USER_ID_MAP")
        )

    def _read_config(self) -> dict[str, str]:
        return {
            "token": self._get_env("DISCORD_BOT_TOKEN"),
            "prefix": self._get_env("DISCORD_COMMAND_PREFIX", "!"),
        }

    def _canonical_user_id(self, raw: str) -> str:
        if raw not in self._user_id_map:
            log.debug("[discord] user %s -> %s (default)", raw, raw)
            return raw
        mapped = self._user_id_map[raw]
        log.info("[discord] user %s -> %s (mapped)", raw, mapped)
        return mapped

    def start(self) -> None:
        if discord is None:
            log.error("[discord] discord.py not installed. pip install discord-py")
            return
        cfg = self._read_config()
        self._token = cfg.get("token", "")
        if not self._token:
            log.error("[discord] DISCORD_BOT_TOKEN not set in .env")
            return

        intents = Intents.default()
        intents.message_content = True
        self._client = discord.Client(intents=intents)

        @self._client.event
        async def on_ready() -> None:
            log.info("[discord] Bot logged in as %s", self._client.user)

        @self._client.event
        async def on_message(message: discord.Message) -> None:
            if message.author.bot:
                return
            if self._client and self._client.user and message.author.id == self._client.user.id:
                return
            cid = str(message.channel.id)
            uid = self._canonical_user_id(str(message.author.id))
            display = message.author.display_name
            text = message.content.strip()
            if not text:
                return
            log.info("[discord] msg from %s in %s: %.60s", display, cid, text)
            # Do NOT block on session.wait() here: this coroutine runs on
            # Discord's event loop, and blocking it during inference starves
            # the gateway heartbeat (>20s) -> aiohttp resets the connection
            # and Discord reconnects. The session runs the reply on its own
            # thread and delivers it via _on_platform_response -> send_message,
            # which schedules back onto this loop with run_coroutine_threadsafe.
            self.handle_message(cid, uid, display, text)

        self._thread = threading.Thread(
            target=self._client.run, args=(self._token,), daemon=True
        )
        self._thread.start()
        self._running = True
        log.info("[discord] Adapter started")

    def stop(self) -> None:
        self._running = False
        if self._client:
            try:
                import asyncio
                asyncio.run_coroutine_threadsafe(
                    self._client.close(), self._client.loop
                )
            except Exception:
                log.warning("discord: async client close failed")

    def send_message(self, conversation_id: str, text: str) -> None:
        if not self._client or not self._client.is_ready():
            log.warning("[discord] Client not ready, can't send to %s", conversation_id)
            return
        channel_id = int(conversation_id)
        channel = self._client.get_channel(channel_id)
        if channel is None:
            log.warning("[discord] Channel %s not found in cache, trying fetch", conversation_id)
            try:
                import asyncio
                asyncio.run_coroutine_threadsafe(
                    self._fetch_and_send(channel_id, text), 
                    self._client.loop
                )
            except Exception as exc:
                log.error("[discord] Failed to fetch and send to %s: %s", conversation_id, exc)
            return
        try:
            import asyncio
            asyncio.run_coroutine_threadsafe(channel.send(text), self._client.loop)
        except Exception as exc:
            log.error("[discord] Failed to send to %s: %s", conversation_id, exc)

    async def _fetch_and_send(self, channel_id: int, text: str) -> None:
        """Fetch channel by ID and send message — used for DMs not in cache."""
        try:
            channel = await self._client.fetch_channel(channel_id)
            await channel.send(text)
            log.info("[discord] Sent to fetched channel %s", channel_id)
        except Exception as exc:
            log.error("[discord] Fetch and send failed for %s: %s", channel_id, exc)