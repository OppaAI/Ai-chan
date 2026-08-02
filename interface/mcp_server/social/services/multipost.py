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
    
    @mcp.tool(
        name="post_social",
        description="Post one payload to selected social services: x, threads, bluesky, mastodon, youtube, reddit, pixelset, discord, medium.",
    )
    def post_social(
        services: str,
        text: str = "",
        image_path: Optional[str] = None,
        title: str = "",
        subreddit: str = "OppaAI",
        video_path: Optional[str] = None,
        description: str = "",
        channel: str = "",
        topic_tag: Optional[str] = None,
        medium_tags: Optional[list[str]] = None,
        medium_publish_status: str = "public",
        medium_canonical_url: str = "",
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
            # Get pre-registered tool from MCP (already wrapped with middleware)
            tool_obj = tool_registry.get(f"post_{service}")
            
            if not tool_obj:
                result = {"ok": False, "provider": service, "error": "unsupported service"}
            else:
                # Call the tool directly (middleware already applied during registration)
                try:
                    if service == "x":
                        result = tool_obj.fn(text=text, image_path=image_path)
                    elif service == "threads":
                        result = tool_obj.fn(text=text, image_path=image_path, topic_tag=topic_tag)
                    elif service == "bluesky":
                        result = tool_obj.fn(text=text, image_path=image_path)
                    elif service == "mastodon":
                        result = tool_obj.fn(text=text, image_path=image_path)
                    elif service == "reddit":
                        result = tool_obj.fn(
                            title=title or text[:280] or "Aiko dev update",
                            text=text,
                            image_path=image_path,
                            subreddit=subreddit,
                        )
                    elif service == "youtube":
                        result = tool_obj.fn(
                            video_path=video_path or image_path or "",
                            title=title or text[:90],
                            description=description or text,
                        )
                    elif service == "pixelset":
                        result = tool_obj.fn(image_path=image_path or "", caption=text)
                    elif service == "discord":
                        result = tool_obj.fn(text=text, image_path=image_path, channel_id=channel)
                    elif service == "medium":
                        result = tool_obj.fn(
                            title=title or text[:100],
                            content=text,
                            tags=medium_tags,
                            publish_status=medium_publish_status,
                            canonical_url=medium_canonical_url,
                        )
                    else:
                        result = {"ok": False, "provider": service, "error": "unsupported service"}
                except Exception as e:
                    result = {"ok": False, "provider": service, "error": str(e)}
            
            results.append(result)

        return {"ok": all(r.get("ok") for r in results) if results else False, "results": results}