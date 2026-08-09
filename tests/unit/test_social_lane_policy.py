from __future__ import annotations

import importlib
import json


def test_adapter_registry_only_two_way_messengers():
    adapters = importlib.import_module("interface.adapter")
    assert set(adapters.ADAPTER_REGISTRY) == {"discord", "telegram", "slack", "matrix"}


def test_social_media_registry_has_pixelset_only():
    social = importlib.import_module("agentic.toolkit.social")
    assert set(social._MEDIA_PROVIDERS_REGISTRY) == {"pixelset"}
    assert social.PHOTO_SOCIAL_PROVIDERS == ("pixelset",)


def test_post_social_dispatches_selected_services():
    multipost = importlib.import_module("interface.mcp_server.social.tools.multipost")

    calls = []

    class Tool:
        def __init__(self, name):
            self.fn = lambda **kwargs: calls.append((name, kwargs)) or {"ok": True, "provider": name}

    class Manager:
        _tools = {
            "post_x": Tool("x"),
            "post_bluesky": Tool("bluesky"),
            "post_pixelset": Tool("pixelset"),
        }

    class MCP:
        _tool_manager = Manager()
        def tool(self, **_kwargs):
            def deco(fn):
                self.post_social = fn
                return fn
            return deco

    mcp = MCP()
    multipost.load_tools(mcp)
    result = mcp.post_social(services="x,bluesky,pixelset", text="hello", image_path="img.png")
    assert result["ok"] is True
    assert [name for name, _ in calls] == ["x", "bluesky", "pixelset"]
