from __future__ import annotations

from typing import Any

from agentic.mcp_client import get_mcp_client, init_mcp_client
from agentic.registry import registry, register_tool_schema
from system.log import get_logger

log = get_logger(__name__)


def bootstrap_mcp(server_url: str = "") -> bool:
    """Connect to the MCP server and register all discovered tools as Aiko bridges.

    Returns True if MCP connected and at least one tool was registered.
    """
    client = init_mcp_client(server_url=server_url)
    if client is None:
        return False

    bridge_defs = client.get_bridge_tool_defs()
    if not bridge_defs:
        log.warning("[mcp] No tools discovered — no bridge tools registered")
        return False

    count = 0
    for name, description, props, required, bridge_fn in bridge_defs:
        register_tool_schema(
            name=name,
            description=description,
            props=props,
            required=required,
            domain="social",
            always_on=False,
            react=True,
            graph=True,
        )
        registry._tools[name].handler = bridge_fn
        count += 1

    log.info("[mcp] Registered %d bridge tools from MCP server", count)

    try:
        from agentic.mcp_client.social_bridge import patch_social_registries
        patch_social_registries()
    except Exception as e:
        log.warning("[mcp] Failed to patch social.py registries: %s", e)

    return True
