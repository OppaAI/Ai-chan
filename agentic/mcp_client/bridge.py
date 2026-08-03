from __future__ import annotations

from typing import Any

from agentic.mcp_client import get_mcp_client, init_mcp_client
from agentic.registry import registry, register_tool_schema
from system.log import get_logger

log = get_logger(__name__)

# Raw MCP posting tools that duplicate the approved-workflow agent wrappers
# (post_video_social / post_photo_social / post_to_social). Exposing these
# directly lets the LLM call them with raw args (e.g. a draft *directory* as
# video_path) and bypass the media_path/approval resolution those wrappers
# handle. They remain reachable internally via the *_social registry adapters
# (see social_bridge.patch_social_registries), just not as free LLM tools.
_HIDDEN_MCP_POST_TOOLS = frozenset({
    "post_youtube", "post_pixelset",
})


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
        if name in _HIDDEN_MCP_POST_TOOLS:
            log.info("[mcp] Hiding raw posting tool from LLM (use *_social wrapper): %s", name)
            continue
        is_protonmail = "protonmail" in name
        register_tool_schema(
            name=name,
            description=description,
            props=props,
            required=required,
            domain="social",
            always_on=is_protonmail,  # email tools always available, not just social capability
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
