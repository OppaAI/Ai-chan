from __future__ import annotations

from pathlib import Path
from typing import Any

from agentic.mcp_client import get_mcp_client
from system.log import get_logger

log = get_logger(__name__)


def _call_mcp(tool: str, **kwargs: Any) -> dict[str, Any]:
    client = get_mcp_client()
    if client is None:
        return {"ok": False, "error": "MCP client not connected"}
    import anyio
    try:
        return anyio.run(client.call_tool, tool, kwargs)
    except Exception as e:
        return {"ok": False, "error": str(e), "tool": tool}


def _adapter_post_x(text: str, image_path: Path | None) -> dict[str, Any]:
    return _call_mcp("post_x", text=text, image_path=str(image_path) if image_path else None)


def _adapter_post_threads(text: str, image_path: Path | None) -> dict[str, Any]:
    return _call_mcp("post_threads", text=text, image_path=str(image_path) if image_path else None)


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


def _adapter_post_threads_text(text: str, _image: Any = None) -> dict[str, Any]:
    return _call_mcp("post_threads", text=text, image_path=None)


MCP_ADAPTERS = {
    "x": _adapter_post_x,
    "threads": _adapter_post_threads,
    "pixelset": _adapter_post_pixelset,
    "youtube": _adapter_post_youtube,
}


def patch_social_registries() -> None:
    """Replace Aiko's social posting registries to route through MCP."""
    import agentic.toolkit.social as social

    social._WEEKLY_PROVIDERS_REGISTRY["x"] = _adapter_post_x
    social._WEEKLY_PROVIDERS_REGISTRY["threads"] = _adapter_post_threads
    social._MEDIA_PROVIDERS_REGISTRY["pixelset"] = _adapter_post_pixelset
    social._VIDEO_PROVIDERS_REGISTRY["youtube"] = _adapter_post_youtube

    original_post_job = social.post_job_post_draft

    def _patched_post_job_post_draft(draft_dir: str | Path) -> dict[str, Any]:
        try:
            path = Path(draft_dir).resolve()
            social._require_approved(path)
            draft_post_path = path / "draft_post.txt"
            if not draft_post_path.exists():
                raise social.SocialApprovalError(f"missing draft_post.txt at {path}")
            text = draft_post_path.read_text(encoding="utf-8").strip()
            if not text:
                raise social.SocialApprovalError(f"empty draft_post.txt at {path}")
        except (social.SocialApprovalError, OSError) as e:
            return {"posted": False, "error": str(e)}

        result = _adapter_post_threads_text(text)
        post_meta = {"posted": bool(result.get("ok")), "posted_at": "now", "results": [result]}
        (path / "posted.json").write_text(str(post_meta), encoding="utf-8")
        return post_meta

    social.post_job_post_draft = _patched_post_job_post_draft

    log.info("[mcp] Patched social.py provider registries to route through MCP")
