from pathlib import Path

from social.services import env


def _default_media_roots() -> list[Path]:
    try:
        from agentic.toolkit.social import photo_social_root
        return [photo_social_root()]
    except Exception:
        return []


def _approved_media_roots() -> list[Path]:
    raw = env("PIXELFED_MEDIA_ROOTS", "") or env("PIXELSET_MEDIA_ROOTS", "")
    roots = [Path(part.strip()).expanduser().resolve() for part in raw.split(",") if part.strip()]
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


def _pixelfed_instance() -> str:
    return (
        env("PIXELFED_INSTANCE", "")
        or env("PIXELFED_INSTANCE_URL", "")
        or env("PIXELSET_INSTANCE", "")
    ).rstrip("/")


def _pixelfed_token() -> str:
    return (
        env("PIXELFED_ACCESS_TOKEN", "")
        or env("PIXELFED_PAT", "")
        or env("PIXELSET_ACCESS_TOKEN", "")
    )


def load_tools(mcp):
    @mcp.tool(
        name="post_pixelfed",
        description="Post an approved photo + caption to Pixelfed via its Mastodon-compatible API",
    )
    def post_pixelfed(image_path: str, caption: str = "") -> dict:
        instance = _pixelfed_instance()
        access_token = _pixelfed_token()
        if not instance or not access_token:
            return {
                "ok": False,
                "provider": "pixelfed",
                "error": "PIXELFED_INSTANCE (or PIXELFED_INSTANCE_URL) and PIXELFED_ACCESS_TOKEN (or PIXELFED_PAT) not set",
            }
        p = _resolve_approved_media_path(image_path)
        if p is None:
            roots = ", ".join(str(root) for root in _approved_media_roots()) or "<none>"
            return {
                "ok": False,
                "provider": "pixelfed",
                "error": f"image not found or outside approved media roots: {image_path}",
                "approved_roots": roots,
            }
        try:
            from mastodon import Mastodon
        except ImportError:
            return {"ok": False, "provider": "pixelfed", "error": "Mastodon.py not installed — pip install Mastodon.py"}
        try:
            api = Mastodon(access_token=access_token, api_base_url=instance, ratelimit_method="throw", request_timeout=60)
            media = api.media_post(p, mime_type=None)
            post = api.status_post(caption, media_ids=[media.id])
            return {"ok": True, "provider": "pixelfed", "id": post.id, "url": getattr(post, "url", "")}
        except Exception as e:
            return {"ok": False, "provider": "pixelfed", "error": str(e)}
