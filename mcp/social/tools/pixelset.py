from __future__ import annotations

from pathlib import Path

from social.tools.base import env


def load_tools(mcp):
    @mcp.tool(
        name="post_pixelset",
        description="Post a photo + caption to Pixelset/Pixelfed-compatible Mastodon API",
    )
    def post_pixelset(image_path: str, caption: str = "") -> dict:
        instance = env("PIXELSET_INSTANCE", env("PIXELFED_INSTANCE", "")).rstrip("/")
        access_token = env("PIXELSET_ACCESS_TOKEN", env("PIXELFED_ACCESS_TOKEN", ""))
        if not instance or not access_token:
            return {"ok": False, "provider": "pixelset", "error": "PIXELSET_INSTANCE and PIXELSET_ACCESS_TOKEN not set"}
        p = Path(image_path)
        if not p.exists():
            return {"ok": False, "provider": "pixelset", "error": f"image not found: {image_path}"}
        try:
            from mastodon import Mastodon
        except ImportError:
            return {"ok": False, "provider": "pixelset", "error": "Mastodon.py not installed — pip install Mastodon.py"}
        try:
            api = Mastodon(access_token=access_token, api_base_url=instance)
            media = api.media_post(p, mime_type=None)
            post = api.status_post(caption, media_ids=[media.id])
            return {"ok": True, "provider": "pixelset", "id": post.id, "url": getattr(post, "url", "")}
        except Exception as e:
            return {"ok": False, "provider": "pixelset", "error": str(e)}
