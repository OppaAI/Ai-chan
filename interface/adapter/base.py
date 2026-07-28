from __future__ import annotations

import os
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from system.log import get_logger
from system.userspace import (
    current_user_id,
    reset_current_display_name,
    reset_current_user_id,
    set_current_display_name,
    set_current_user_id,
)

log = get_logger(__name__)


class ConversationSession:
    """A single conversation turn handled through Aiko's cognition core.

    Each session runs on its own thread, sets the user context via
    contextvars, calls think.route(), and delivers the response back
    through a callback.
    """

    def __init__(
        self,
        conversation_id: str,
        platform_user_id: str,
        display_name: str,
        think: Any,
        memorize: Any,
        on_response: Callable[[str, str], None],
        on_error: Callable[[str, Exception], None] | None = None,
    ) -> None:
        self.conversation_id = conversation_id
        self.platform_user_id = platform_user_id
        self.display_name = display_name
        self._think = think
        self._memorize = memorize
        self._on_response = on_response
        self._on_error = on_error
        self._last_input: str = ""
        self._buffer: list[str] = []
        self._lock = threading.Lock()
        self._done = threading.Event()
        self._result: str | None = None
        self._error: Exception | None = None

    def _token_callback(self, token: str) -> None:
        stripped = token.rstrip("\r\n") if token else ""
        if stripped == "__THINKING__":
            return
        if stripped.startswith("__TOOL__:") or stripped.startswith("__SEARCHING__:"):
            return
        with self._lock:
            self._buffer.append(token)

    def _collect_response(self) -> str:
        with self._lock:
            text = "".join(self._buffer).strip()
            self._buffer.clear()
        return text

    def _run(self) -> None:
        user_token = set_current_user_id(self.platform_user_id)
        name_token = set_current_display_name(self.display_name)
        try:
            response = self._think.route(self._last_input, token_callback=self._token_callback)
            self._result = response.strip()
            self._on_response(self.conversation_id, self._result)
        except Exception as exc:
            log.exception("[adapter] session failed for %s", self.conversation_id)
            self._error = exc
            if self._on_error:
                self._on_error(self.conversation_id, exc)
        finally:
            reset_current_user_id(user_token)
            reset_current_display_name(name_token)
            self._done.set()

    def start(self, user_input: str) -> None:
        self._last_input = user_input
        self._buffer = []
        self._done.clear()
        self._result = None
        self._error = None
        t = threading.Thread(target=self._run, name=f"conn-{self.conversation_id}", daemon=True)
        t.start()

    def wait(self, timeout: float = 120.0) -> str | None:
        self._done.wait(timeout=timeout)
        return self._result

    @property
    def response(self) -> str | None:
        return self._result

    @property
    def error(self) -> Exception | None:
        return self._error


class AdapterBase(ABC):
    """Abstract base for messaging/social-media platform adapters.

    Subclasses implement start/stop for the platform listener and
    send_message for delivering responses. Incoming messages are routed
    through Aiko's cognition core via ConversationSession.
    """

    def __init__(self, config: dict[str, str] | None = None) -> None:
        self.config = config or {}
        self._think: Any = None
        self._memorize: Any = None
        self._running = False
        self._boot_lock = threading.Lock()

    @property
    def name(self) -> str:
        return self.__class__.__name__.replace("Adapter", "").lower()

    def boot(self, think: Any, memorize: Any) -> None:
        """Inject live subsystem references from AikoWakeup."""
        self._think = think
        self._memorize = memorize

    @abstractmethod
    def start(self) -> None:
        """Start the platform listener (bot daemon)."""

    @abstractmethod
    def stop(self) -> None:
        """Stop the platform listener gracefully."""

    @abstractmethod
    def send_message(self, conversation_id: str, text: str) -> None:
        """Send a text message to the given conversation/chat."""

    def handle_message(self, conversation_id: str, platform_user_id: str, display_name: str, text: str) -> ConversationSession:
        """Route an incoming message through Aiko's cognition core.

        Returns a ConversationSession that will populate .response
        when complete. The caller should call .wait() or poll .response.
        """
        session = ConversationSession(
            conversation_id=conversation_id,
            platform_user_id=platform_user_id,
            display_name=display_name,
            think=self._think,
            memorize=self._memorize,
            on_response=self._on_platform_response,
            on_error=self._on_platform_error,
        )
        session.start(text)
        return session

    def _on_platform_response(self, conversation_id: str, text: str) -> None:
        """Default response handler — sends the reply back to the platform."""
        if text:
            self.send_message(conversation_id, text)

    def _on_platform_error(self, conversation_id: str, exc: Exception) -> None:
        log.error("[adapter/%s] error for %s: %s", self.name, conversation_id, exc)
        self.send_message(conversation_id, f"Sorry, I encountered an error: {exc}")

    @staticmethod
    def _get_env(name: str, default: str = "") -> str:
        return os.getenv(name, default)

    def _read_config(self) -> dict[str, str]:
        return {}
