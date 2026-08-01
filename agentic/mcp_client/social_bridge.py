from __future__ import annotations

from pathlib import Path
from typing import Any

from agentic.mcp_client import get_mcp_client
from system.log import get_logger

log = get_logger(__name__)


def _call_mcp(tool: str, **kwargs: Any) -> dict[str, Any]:
    """Call an MCP tool with the given keyword arguments.
    
    The MCP client's call_tool_sync expects (tool_name, args_dict), not
    unpacked keyword arguments — so we pass kwargs as a single dict argument.
    """
    client = get_mcp_client()
    if client is None:
        return {"ok": False, "error": "MCP client not connected"}
    try:
        return client.call_tool_sync(tool, kwargs)
    except Exception as e:
        return {"ok": False, "error": str(e), "tool": tool}


def _adapter_post_pixelset(selections: list[dict[str, Any]]) -> dict[str, Any]:
    if not selections:
        return {"ok": False, "provider": "pixelset", "error": "no selections"}
    sel = selections[0]
    return _call_mcp("post_social", services="pixelset", text=sel.get("caption", ""), image_path=sel.get("media_path", ""))


def _adapter_post_youtube(sel: dict[str, Any]) -> dict[str, Any]:
    return _call_mcp(
        "post_youtube",
        video_path=sel.get("media_path", ""),
        title=sel.get("title", ""),
        description=sel.get("description", ""),
    )


MCP_ADAPTERS = {
    "pixelset": _adapter_post_pixelset,
    "youtube": _adapter_post_youtube,
}


def patch_social_registries() -> None:
    """Patch remaining draft registries that intentionally dispatch through MCP."""
    import agentic.toolkit.social as social

    social._MEDIA_PROVIDERS_REGISTRY["pixelset"] = _adapter_post_pixelset
    social._VIDEO_PROVIDERS_REGISTRY["youtube"] = _adapter_post_youtube
    log.info("[mcp] Patched media/video social registries to route through MCP")