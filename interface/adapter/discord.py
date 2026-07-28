from __future__ import annotations

import threading

from system.log import get_logger
from interface.adapter.base import ConnectorBase

log = get_logger(__name__)

try:
    import discord
    from discord import Intents
except ImportError:
    discord = None


class DiscordConnector(ConnectorBase):
    """Discord bot connector using discord.py."""

    def __init__(self, config: dict[str, str] | None = None) -> None:
        super().__init__(config)
        self._token: str = ""
        self._client: discord.Client | None = None
        self._thread: threading.Thread | None = None
        self._channel_map: dict[int, str] = {}

    def _read_config(self) -> dict[str, str]:
        return {
            "token": self._get_env("DISCORD_BOT_TOKEN"),
            "prefix": self._get_env("DISCORD_COMMAND_PREFIX", "!"),
        }

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
            uid = str(message.author.id)
            display = message.author.display_name
            text = message.content.strip()
            if not text:
                return
            log.info("[discord] msg from %s in %s: %.60s", display, cid, text)
            session = self.handle_message(cid, uid, display, text)
            session.wait()

        self._thread = threading.Thread(
            target=self._client.run, args=(self._token,), daemon=True
        )
        self._thread.start()
        self._running = True
        log.info("[discord] Connector started")

    def stop(self) -> None:
        self._running = False
        if self._client:
            try:
                import asyncio
                asyncio.run_coroutine_threadsafe(
                    self._client.close(), self._client.loop
                )
            except Exception:
                pass

    def send_message(self, conversation_id: str, text: str) -> None:
        if not self._client or not self._client.is_ready():
            log.warning("[discord] Client not ready, can't send to %s", conversation_id)
            return
        channel_id = int(conversation_id)
        channel = self._client.get_channel(channel_id)
        if channel is None:
            log.warning("[discord] Channel %s not found in cache", conversation_id)
            return
        try:
            import asyncio
            asyncio.run_coroutine_threadsafe(channel.send(text), self._client.loop)
        except Exception as exc:
            log.error("[discord] Failed to send to %s: %s", conversation_id, exc)
