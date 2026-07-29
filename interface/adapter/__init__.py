from interface.adapter.base import AdapterBase, ConversationSession
from interface.adapter.discord import DiscordAdapter
from interface.adapter.telegram import TelegramAdapter
from interface.adapter.slack import SlackAdapter
from interface.adapter.matrix import MatrixAdapter
from system.log import get_logger

log = get_logger(__name__)

ADAPTER_REGISTRY: dict[str, type[AdapterBase]] = {
    "discord": DiscordAdapter,
    "telegram": TelegramAdapter,
    "slack": SlackAdapter,
    "matrix": MatrixAdapter,
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



def start_background_adapters(think, memorize, names: list[str] | None = None) -> list[AdapterBase]:
    """Start configured two-way messenger adapters beside WebUI/CLI sessions.

    These adapters are intentionally headless: inbound messages are routed
    through the same memory and intent pipeline as local chat turns, and
    replies are sent back to the originating messaging service without
    echoing the exchange into the active WebUI or CLI transcript.
    """
    import os

    raw = names if names is not None else [
        part.strip().lower()
        for part in os.getenv("AIKO_MESSENGER_ADAPTERS", "").split(",")
        if part.strip()
    ]
    adapters: list[AdapterBase] = []
    for name in raw:
        cls = ADAPTER_REGISTRY.get(name)
        if cls is None:
            log.warning("[adapter] ignoring unsupported two-way adapter: %s", name)
            continue
        try:
            adapter = cls()
            adapter.boot(think, memorize)
            adapter.start()
            adapters.append(adapter)
            log.info("[adapter] started %s in background", name)
        except Exception:
            log.exception("[adapter] failed to start %s in background", name)
    return adapters


__all__ = [
    "AdapterBase",
    "ConversationSession",
    "DiscordAdapter",
    "TelegramAdapter",
    "SlackAdapter",
    "MatrixAdapter",
    "ADAPTER_REGISTRY",
    "start_background_adapters",
]
