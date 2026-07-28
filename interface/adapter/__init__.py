from interface.adapter.base import AdapterBase, ConversationSession
from interface.adapter.discord import DiscordAdapter
from interface.adapter.telegram import TelegramAdapter
from interface.adapter.slack import SlackAdapter
from interface.adapter.matrix import MatrixAdapter
from interface.adapter.bluesky import BlueskyAdapter
from interface.adapter.mastodon import MastodonAdapter
from interface.adapter.reddit import RedditAdapter
from interface.adapter.youtube import YouTubeAdapter
from interface.adapter.messenger import MessengerAdapter
from interface.adapter.googlechat import GoogleChatAdapter

ADAPTER_REGISTRY: dict[str, type[AdapterBase]] = {
    "discord": DiscordAdapter,
    "telegram": TelegramAdapter,
    "slack": SlackAdapter,
    "matrix": MatrixAdapter,
    "bluesky": BlueskyAdapter,
    "mastodon": MastodonAdapter,
    "reddit": RedditAdapter,
    "youtube": YouTubeAdapter,
    "messenger": MessengerAdapter,
    "googlechat": GoogleChatAdapter,
}


def _bootstrap_adapter(adapter_name: str) -> tuple:
    """Boot Aiko subsystems needed by adapters.

    Returns (think, memorize) — both may be None on failure.
    """
    import threading
    import time
    import os

    from system.wakeup import AikoWakeup
    from system.userspace import set_current_user_id, set_current_display_name

    # Set a default identity for adapter sessions
    uid = os.getenv("AIKO_USER_ID", f"adapter_{adapter_name}")
    set_current_user_id(uid)
    set_current_display_name(uid)

    # Boot all subsystems
    result = AikoWakeup().boot(
        on_loading=lambda key: print(f"    [{key}] loading..."),
        on_done=lambda key: print(f"    [{key}] done"),
        on_skip=lambda key: print(f"    [{key}] skipped"),
    )

    if result.think is None:
        print(f"  [adapter] CRITICAL: AikoThink failed to boot.")
        return None, None

    return result.think, result.memorize


def run_adapter(name: str, args) -> None:
    """Boot Aiko subsystems and launch the named adapter."""
    from system.config import load_config

    load_config()
    name = name.lower()
    if name not in ADAPTER_REGISTRY:
        available = ", ".join(ADAPTER_REGISTRY)
        print(f"Unknown adapter '{name}'. Available: {available}")
        return

    print(f"  [adapter] Booting Aiko subsystems for {name}...")
    think, memorize = _bootstrap_adapter(name)

    if think is None:
        print("  [adapter] CRITICAL: AikoThink failed to boot. Cannot start adapter.")
        return

    cls = ADAPTER_REGISTRY[name]
    adapter = cls()
    adapter.boot(think, memorize)

    print(f"  [adapter] Starting {name}...")
    adapter.start()

    print(f"\n  {name} adapter is running. Press Ctrl+C to stop.\n")
    try:
        import signal
        signal.pause()
    except KeyboardInterrupt:
        pass
    finally:
        adapter.stop()
        print(f"  [adapter] {name} stopped.")


__all__ = [
    "AdapterBase",
    "ConversationSession",
    "DiscordAdapter",
    "TelegramAdapter",
    "TwitterAdapter",
    "SlackAdapter",
    "MatrixAdapter",
    "ADAPTER_REGISTRY",
    "run_adapter",
]
