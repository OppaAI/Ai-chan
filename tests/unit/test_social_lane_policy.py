import importlib
from types import SimpleNamespace


def test_messenger_registry_excludes_one_way_platforms():
    adapters = importlib.import_module("agentic.adapters")
    assert set(adapters.ADAPTER_REGISTRY) == {"discord", "telegram", "slack", "matrix"}


def test_social_media_registry_has_pixelfed_only():
    """Lane B defaults to Pixelfed; MCP handlers are patched at runtime."""
    social = importlib.import_module("agentic.toolkit.social")
    assert social.PHOTO_SOCIAL_PROVIDERS == ("pixelfed",)
    # Registries start empty; agentic.mcp_client.social_bridge patches them
    # so uploads go through post_social / post_youtube rather than direct APIs.
    assert social._MEDIA_PROVIDERS_REGISTRY == {}
    assert social._VIDEO_PROVIDERS_REGISTRY == {}
    bridge = importlib.import_module("agentic.mcp_client.social_bridge")
    bridge.patch_social_registries()
    assert set(social._MEDIA_PROVIDERS_REGISTRY) == {"pixelfed"}
    assert set(social._VIDEO_PROVIDERS_REGISTRY) == {"youtube"}


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
    pixelfed_kwargs = dict(calls)["pixelfed"]
    assert pixelfed_kwargs == {"image_path": "img.png", "caption": "hello"}


def test_lane_d_surfaces_nested_threads_failure(tmp_path, monkeypatch):
    social = importlib.import_module("agentic.toolkit.social")
    monkeypatch.setenv("JOB_POST_SOCIAL_ROOT", str(tmp_path))
    draft_dir = tmp_path / "2026-08-20" / "tech" / "draft"
    draft_dir.mkdir(parents=True)
    (draft_dir / "draft_post.txt").write_text("A job post", encoding="utf-8")
    (draft_dir / "draft.json").write_text(
        '{"human_approved": true, "posted": false}', encoding="utf-8"
    )
    monkeypatch.setattr(
        social,
        "_call_social_mcp",
        lambda *args, **kwargs: {
            "ok": False,
            "results": [{
                "ok": False,
                "provider": "threads",
                "stage": "create",
                "status_code": 403,
                "response": '{"error":{"message":"token expired"}}',
            }],
        },
    )

    result = social.post_job_post_draft(draft_dir)

    assert result["posted"] is False
    assert "threads: create: HTTP 403" in result["error"]
    assert "token expired" in result["error"]


def test_social_db_token_cache_can_be_used_from_mcp_worker_thread(tmp_path):
    import threading
    from interface.mcp_server.social.state import MCPDatabase

    db = MCPDatabase(str(tmp_path / "social.db"))
    db.migrate()
    errors = []

    def worker():
        try:
            db.set_cached_token("threads", "token", 3600)
            assert db.get_cached_token("threads") == "token"
        except Exception as exc:  # pragma: no cover - assertion detail
            errors.append(exc)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    db.close()
    assert errors == []
