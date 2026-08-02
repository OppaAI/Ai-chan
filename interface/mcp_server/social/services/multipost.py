

def _split_services(services: str | list[str] | tuple[str, ...]) -> list[str]:
    if isinstance(services, str):
        raw = services.replace(";", ",").split(",")
    else:
        raw = list(services)
    aliases = {"twitter": "x"}
    return [aliases.get(str(item).strip().lower(), str(item).strip().lower()) for item in raw if str(item).strip()]


def load_tools(mcp):
    @mcp.tool(
        name="post_social",
        description="Post one payload to selected one-way social services: x, threads, bluesky, mastodon, youtube, reddit, pixelset, discord, medium.",
    )
    def post_social(
        services: str,
        text: str = "",
        image_path: str | None = None,
        title: str = "",
        subreddit: str = "OppaAI",
        video_path: str | None = None,
        description: str = "",
        channel: str = "",
        topic_tag: str | None = None,
        medium_tags: list[str] | None = None,
        medium_publish_status: str = "public",
        medium_canonical_url: str = "",
    ) -> dict:
        from social.middleware import wrap_tool

        results = []
        tools = mcp._tool_manager._tools
        for service in _split_services(services):
            raw_fn = tools.get(service)
            if not raw_fn:
                result = {"ok": False, "provider": service, "error": "unsupported service"}
            else:
                # Apply middleware (rate limit, idempotency, logging) manually
                # since we call .fn directly, bypassing the MCP protocol.
                wrapped = wrap_tool(f"post_{service}", raw_fn.fn)
                if service == "x":
                    result = wrapped(text=text, image_path=image_path)
                elif service == "threads":
                    result = wrapped(text=text, image_path=image_path, topic_tag=topic_tag)
                elif service == "bluesky":
                    result = wrapped(text=text, image_path=image_path)
                elif service == "mastodon":
                    result = wrapped(text=text, image_path=image_path)
                elif service == "reddit":
                    result = wrapped(title=title or text[:280] or "Aiko dev update", text=text, image_path=image_path, subreddit=subreddit)
                elif service == "youtube":
                    result = wrapped(video_path=video_path or image_path or "", title=title or text[:90], description=description or text)
                elif service == "pixelset":
                    result = wrapped(image_path=image_path or "", caption=text)
                elif service == "discord":
                    result = wrapped(text=text, image_path=image_path, channel_id=channel)
                elif service == "medium":
                    result = wrapped(title=title or text[:100], content=text, tags=medium_tags, publish_status=medium_publish_status, canonical_url=medium_canonical_url)
                else:
                    result = {"ok": False, "provider": service, "error": "unsupported service"}
            results.append(result)
        return {"ok": all(r.get("ok") for r in results) if results else False, "results": results}
