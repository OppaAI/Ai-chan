import importlib
from types import SimpleNamespace


def test_messenger_registry_excludes_one_way_platforms():
    adapters = importlib.import_module("agentic.adapters")
    assert set(adapters.ADAPTER_REGISTRY) == {"discord", "telegram", "slack", "matrix"}


def test_social_media_registry_has_pixelfed_only():
    social = importlib.import_module("agentic.toolkit.social")
    assert set(social._MEDIA_PROVIDERS_REGISTRY) == {"pixelfed"}
    assert social.PHOTO_SOCIAL_PROVIDERS == ("pixelfed",)


def test_post_social_routes_known_services_only():
    multipost = importlib.import_module("interface.mcp_server.social.services.multipost")

    calls = []

    class Tool:
        def __init__(self, name):
            self.name = name

        def fn(self, **kwargs):
            calls.append((self.name, kwargs))
            return {"ok": True, "provider": self.name}

    class FakeMCP:
        def __init__(self):
            self._tool_manager = SimpleNamespace(_tools={
                "post_x": Tool("x"),
                "post_bluesky": Tool("bluesky"),
                "post_pixelfed": Tool("pixelfed"),
            })

        def tool(self, *args, **kwargs):
            def deco(fn):
                self.post_social = fn
                return fn
            return deco

    mcp = FakeMCP()
    multipost.load_tools(mcp)
    result = mcp.post_social(services="x,bluesky,pixelfed", text="hello", image_path="img.png")
    assert result["ok"] is True
    assert [name for name, _ in calls] == ["x", "bluesky", "pixelfed"]
