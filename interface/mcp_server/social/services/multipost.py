import os
from typing import Optional, Union


def _split_services(services: Union[str, list[str], tuple[str, ...]]) -> list[str]:
    """
    Parse service list from comma/semicolon-separated string or iterable.
    
    Docstring: Split input by delimiters, normalize to lowercase,
    apply service name aliases (twitter → x).
    
    Inline: Handle str/list/tuple input, strip whitespace, deduplicate.
    """
    if isinstance(services, str):
        raw = services.replace(";", ",").split(",")
    else:
        raw = list(services)
    
    aliases = {"twitter": "x"}
    return [
        aliases.get(str(item).strip().lower(), str(item).strip().lower())
        for item in raw
        if str(item).strip()
    ]


def env(name: str, default: str = "") -> str:
    """Get environment variable with default."""
    return os.getenv(name, default)


def int_env(name: str, default: int) -> int:
    """Get integer environment variable with fallback."""
    try:
        return int(os.getenv(name, str(default)))
    except (ValueError, TypeError):
        return default


def bool_env(name: str, default: bool = False) -> bool:
    """Get boolean environment variable (1/true/yes/on = True)."""
    return os.getenv(name, "").lower() in {"1", "true", "yes", "on"}


def err(provider: str, message: str) -> dict:
    """Return error dict."""
    return {"ok": False, "provider": provider, "error": message}


def load_tools(mcp):
    """Register multipost tool with MCP."""
    import inspect

    async def _invoke_platform(tool_obj, **kwargs) -> dict:
        """Call a registered platform tool; await if guards made it async.

        wrap_tool() always returns an async wrapper, so tool_obj.fn(...) is a
        coroutine. Calling it without await produced:
          RuntimeWarning: coroutine 'post_threads' was never awaited
          error='coroutine' object has no attribute 'get'
        """
        out = tool_obj.fn(**kwargs)
        if inspect.isawaitable(out):
            out = await out
        if not isinstance(out, dict):
            return {"ok": False, "error": f"tool returned non-dict: {type(out).__name__}"}
        return out

    @mcp.tool(
        name="post_social",
        description="Post one payload to selected social services: x, threads, bluesky, mastodon, youtube, pixelfed, discord.",
    )
    async def post_social(
        services: str,
        text: str = "",
        image_path: Optional[str] = None,
        title: str = "",
        video_path: Optional[str] = None,
        description: str = "",
        channel: str = "",
        topic_tag: Optional[str] = None,
    ) -> dict:
        """
        Post to multiple social platforms in one call.
        
        Docstring: Accept unified payload, route to individual
        platform tools with platform-specific argument mapping.
        Tools are pre-wrapped with rate limiting and logging.
        
        Inline: Get tool references from MCP registry, call each
        with appropriate arguments, collect results.
        """
        results = []
        tool_registry = mcp._tool_manager._tools

        for service in _split_services(services):
            tool_obj = tool_registry.get(f"post_{service}")
            
            if not tool_obj:
                result = {"ok": False, "provider": service, "error": "unsupported service"}
            else:
                try:
                    if service == "x":
                        result = await _invoke_platform(tool_obj, text=text, image_path=image_path)
                    elif service == "threads":
                        result = await _invoke_platform(
                            tool_obj, text=text, image_path=image_path, topic_tag=topic_tag
                        )
                    elif service == "bluesky":
                        result = await _invoke_platform(tool_obj, text=text, image_path=image_path)
                    elif service == "mastodon":
                        result = await _invoke_platform(tool_obj, text=text, image_path=image_path)
                    elif service == "youtube":
                        result = await _invoke_platform(
                            tool_obj,
                            video_path=video_path or image_path or "",
                            title=title or text[:90],
                            description=description or text,
                        )
                    elif service == "pixelfed":
                        result = await _invoke_platform(
                            tool_obj, image_path=image_path or "", caption=text
                        )
                    elif service == "discord":
                        result = await _invoke_platform(
                            tool_obj, text=text, image_path=image_path, channel_id=channel
                        )
                    else:
                        result = {"ok": False, "provider": service, "error": "unsupported service"}
                except Exception as e:
                    result = {"ok": False, "provider": service, "error": str(e)}
            
            results.append(result)

        return {
            "ok": all(isinstance(r, dict) and r.get("ok") for r in results) if results else False,
            "results": results,
        }
