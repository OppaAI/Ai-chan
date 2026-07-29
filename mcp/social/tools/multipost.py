from __future__ import annotations


def _split_services(services: str | list[str] | tuple[str, ...]) -> list[str]:
    if isinstance(services, str):
        raw = services.replace(";", ",").split(",")
    else:
        raw = list(services)
    aliases = {"twitter": "x", "pixelfed": "pixelset"}
    return [aliases.get(str(item).strip().lower(), str(item).strip().lower()) for item in raw if str(item).strip()]


def load_tools(mcp):
    @mcp.tool(
        name="post_social",
        description="Post one payload to selected one-way social services: x, threads, bluesky, mastodon, youtube, reddit, pixelset, discord.",
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
    ) -> dict:
        results = []
        tools = mcp._tool_manager._tools
        for service in _split_services(services):
            if service == "x":
                result = tools["post_x"].fn(text=text, image_path=image_path)
            elif service == "threads":
                result = tools["post_threads"].fn(text=text, image_path=image_path)
            elif service == "bluesky":
                result = tools["post_bluesky"].fn(text=text, image_path=image_path)
            elif service == "mastodon":
                result = tools["post_mastodon"].fn(text=text, image_path=image_path)
            elif service == "reddit":
                result = tools["post_reddit"].fn(title=title or text[:280] or "Aiko dev update", text=text, image_path=image_path, subreddit=subreddit)
            elif service == "youtube":
                result = tools["post_youtube"].fn(video_path=video_path or image_path or "", title=title or text[:90], description=description or text)
            elif service == "pixelset":
                result = tools["post_pixelset"].fn(image_path=image_path or "", caption=text)
            elif service == "discord":
                result = tools["post_discord"].fn(text=text, image_path=image_path, channel_id=channel)
            else:
                result = {"ok": False, "provider": service, "error": "unsupported service"}
            results.append(result)
        return {"ok": all(r.get("ok") for r in results) if results else False, "results": results}
