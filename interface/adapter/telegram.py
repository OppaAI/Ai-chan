from __future__ import annotations

import threading

from system.log import get_logger
from interface.adapter.base import AdapterBase

log = get_logger(__name__)

try:
    from telegram import Update
    from telegram.ext import Application, CommandHandler, MessageHandler, filters
except ImportError:
    Application = None


class TelegramAdapter(AdapterBase):
    """Telegram bot adapter using python-telegram-bot."""

    def __init__(self, config: dict[str, str] | None = None) -> None:
        super().__init__(config)
        self._token: str = ""
        self._app: Application | None = None
        self._thread: threading.Thread | None = None

    def _read_config(self) -> dict[str, str]:
        return {
            "token": self._get_env("TELEGRAM_BOT_TOKEN"),
        }

    def start(self) -> None:
        if Application is None:
            log.error("[telegram] python-telegram-bot not installed. pip install python-telegram-bot")
            return
        cfg = self._read_config()
        self._token = cfg.get("token", "")
        if not self._token:
            log.error("[telegram] TELEGRAM_BOT_TOKEN not set in .env")
            return

        self._app = Application.builder().token(self._token).build()

        async def handle_text(update: Update, _context) -> None:
            if not update.message or not update.message.text:
                return
            user = update.message.from_user
            if user is None:
                return
            cid = str(update.effective_chat.id) if update.effective_chat else str(user.id)
            uid = str(user.id)
            display = user.full_name or user.username or uid
            text = update.message.text.strip()
            if not text:
                return
            if text.startswith("/start") or text.startswith("/help"):
                await update.message.reply_text(
                    "Hi! I'm Aiko-chan. Just send me a message and we can chat."
                )
                return
            log.info("[telegram] msg from %s: %.60s", display, text)
            session = self.handle_message(cid, uid, display, text)
            session.wait()

        if self._app:
            self._app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
            self._app.add_handler(CommandHandler("start", handle_text))

            self._thread = threading.Thread(
                target=self._app.run_polling, kwargs={"drop_pending_updates": True},
                daemon=True,
            )
            self._thread.start()
            self._running = True
            log.info("[telegram] Adapter started")

    def stop(self) -> None:
        self._running = False
        if self._app:
            try:
                import asyncio
                asyncio.run_coroutine_threadsafe(
                    self._app.stop(), self._app.loop
                )
            except Exception:
                log.warning("telegram: async app stop failed")

    def send_message(self, conversation_id: str, text: str) -> None:
        if not self._app:
            return
        import asyncio
        try:
            chat_id = int(conversation_id)
            asyncio.run_coroutine_threadsafe(
                self._app.bot.send_message(chat_id=chat_id, text=text),
                self._app.loop,
            )
        except Exception as exc:
            log.error("[telegram] Failed to send to %s: %s", conversation_id, exc)
