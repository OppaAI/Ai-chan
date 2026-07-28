from interface.adapter.base import ConnectorBase, ConversationSession
from interface.adapter.discord import DiscordConnector
from interface.adapter.telegram import TelegramConnector
from interface.adapter.slack import SlackConnector
from interface.adapter.matrix import MatrixConnector
from interface.adapter.bluesky import BlueskyConnector
from interface.adapter.mastodon import MastodonConnector
from interface.adapter.reddit import RedditConnector
from interface.adapter.youtube import YouTubeConnector
from interface.adapter.messenger import MessengerConnector
from interface.adapter.googlechat import GoogleChatConnector

CONNECTOR_REGISTRY: dict[str, type[ConnectorBase]] = {
    "discord": DiscordConnector,
    "telegram": TelegramConnector,
    "slack": SlackConnector,
    "matrix": MatrixConnector,
    "bluesky": BlueskyConnector,
    "mastodon": MastodonConnector,
    "reddit": RedditConnector,
    "youtube": YouTubeConnector,
    "messenger": MessengerConnector,
    "googlechat": GoogleChatConnector,
}


def _bootstrap_connector(connector_name: str) -> tuple:
    """Boot Aiko subsystems needed by connectors.

    Returns (think, memorize) — both may be None on failure.
    """
    import threading
    import time
    import os

    from system.wakeup import AikoWakeup
    from system.userspace import set_current_user_id, set_current_display_name

    # Set a default identity for connector sessions
    uid = os.getenv("AIKO_USER_ID", f"connector_{connector_name}")
    set_current_user_id(uid)
    set_current_display_name(uid)

    # Boot all subsystems
    result = AikoWakeup().boot(
        on_loading=lambda key: print(f"    [{key}] loading..."),
        on_done=lambda key: print(f"    [{key}] done"),
        on_skip=lambda key: print(f"    [{key}] skipped"),
    )

    if result.think is None:
        print(f"  [connector] CRITICAL: AikoThink failed to boot.")
        return None, None

    return result.think, result.memorize


def run_connector(name: str, args) -> None:
    """Boot Aiko subsystems and launch the named connector."""
    from system.config import load_config

    load_config()
    name = name.lower()
    if name not in CONNECTOR_REGISTRY:
        available = ", ".join(CONNECTOR_REGISTRY)
        print(f"Unknown connector '{name}'. Available: {available}")
        return

    print(f"  [connector] Booting Aiko subsystems for {name}...")
    think, memorize = _bootstrap_connector(name)

    if think is None:
        print("  [connector] CRITICAL: AikoThink failed to boot. Cannot start connector.")
        return

    cls = CONNECTOR_REGISTRY[name]
    connector = cls()
    connector.boot(think, memorize)

    print(f"  [connector] Starting {name}...")
    connector.start()

    print(f"\n  {name} connector is running. Press Ctrl+C to stop.\n")
    try:
        import signal
        signal.pause()
    except KeyboardInterrupt:
        pass
    finally:
        connector.stop()
        print(f"  [connector] {name} stopped.")


__all__ = [
    "ConnectorBase",
    "ConversationSession",
    "DiscordConnector",
    "TelegramConnector",
    "TwitterConnector",
    "SlackConnector",
    "MatrixConnector",
    "CONNECTOR_REGISTRY",
    "run_connector",
]
