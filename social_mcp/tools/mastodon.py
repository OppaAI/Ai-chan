from __future__ import annotations

from pathlib import Path

from social_mcp.tools.base import env


def load_tools(mcp):
    @mcp.tool(
        name="post_mastodon",
        description="Post text + optional image to Mastodon",
    )
    def post_mastodon(text: str, image_path: str | None = None) -> dict:
        instance = env("MASTODON_INSTANCE", "https://mastodon.social").rstrip("/")
        access_token = env("MASTODON_ACCESS_TOKEN")

        if not access_token:
            return {"ok": False, "provider": "mastodon", "error": "MASTODON_ACCESS_TOKEN not set"}

        try:
            from mastodon import Mastodon
        except ImportError:
            return {"ok": False, "provider": "mastodon", "error": "Mastodon.py not installed — pip install Mastodon.py"}

        try:
            api = Mastodon(access_token=access_token, api_base_url=instance)
            media_ids = []
            if image_path:
                p = Path(image_path)
                if not p.exists():
                    return {"ok": False, "provider": "mastodon", "error": f"image not found: {image_path}"}
                media = api.media_post(p, mime_type=None)
                media_ids.append(media.id)
            post = api.status_post(text, media_ids=media_ids or None)
            return {"ok": True, "provider": "mastodon", "id": post.id, "url": post.url}
        except Exception as e:
            return {"ok": False, "provider": "mastodon", "error": str(e)}
