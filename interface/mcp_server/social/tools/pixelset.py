
from pathlib import Path

from social.tools.base import env


def _default_media_roots() -> list[Path]:
    try:
        from agentic.toolkit.social import photo_social_root
        return [photo_social_root()]
    except Exception:
        return []


def _approved_media_roots() -> list[Path]:
    raw = env("PIXELSET_MEDIA_ROOTS", "")
    roots = [Path(part).expanduser().resolve() for part in raw.split(",") if part.strip()]
    return roots or [root.resolve() for root in _default_media_roots()]


def _resolve_approved_media_path(image_path: str) -> Path | None:
    try:
        candidate = Path(image_path).expanduser().resolve(strict=True)
    except OSError:
        return None
    for root in _approved_media_roots():
        try:
            candidate.relative_to(root)
            return candidate
        except ValueError:
            continue
    return None


def load_tools(mcp):
    @mcp.tool(
        name="post_pixelset",
        description="Post an approved photo + caption to Pixelset via its Mastodon-compatible API",
    )
    def post_pixelset(image_path: str, caption: str = "") -> dict:
        instance = env("PIXELSET_INSTANCE", "").rstrip("/")
        access_token = env("PIXELSET_ACCESS_TOKEN", "")
        if not instance or not access_token:
            return {"ok": False, "provider": "pixelset", "error": "PIXELSET_INSTANCE and PIXELSET_ACCESS_TOKEN not set"}
        p = _resolve_approved_media_path(image_path)
        if p is None:
            roots = ", ".join(str(root) for root in _approved_media_roots()) or "<none>"
            return {"ok": False, "provider": "pixelset", "error": f"image not found or outside approved media roots: {image_path}", "approved_roots": roots}
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
