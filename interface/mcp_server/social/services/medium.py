import requests

from social.services import env, err


def load_tools(mcp):
    @mcp.tool(
        name="post_medium",
        description="Post article to Medium",
    )
    def post_medium(
        title: str,
        content: str,
        content_format: str = "markdown",
        tags: list[str] | None = None,
        publish_status: str = "public",
        canonical_url: str = "",
    ) -> dict:
        token = env("MEDIUM_INTEGRATION_TOKEN")
        if not token:
            return err("medium", "MEDIUM_INTEGRATION_TOKEN not set")

        user_id = env("MEDIUM_USER_ID")
        if not user_id:
            return err("medium", "MEDIUM_USER_ID not set")

        url = f"https://api.medium.com/v1/users/{user_id}/posts"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        payload = {
            "title": title,
            "contentFormat": content_format,
            "content": content,
            "publishStatus": publish_status,
        }
        if tags:
            payload["tags"] = tags[:5]
        if canonical_url:
            payload["canonicalUrl"] = canonical_url

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            if 200 <= resp.status_code < 300:
                data = resp.json().get("data", {})
                return {
                    "ok": True,
                    "provider": "medium",
                    "post_id": data.get("id"),
                    "url": data.get("url"),
                }
            return err("medium", f"HTTP {resp.status_code}: {resp.text[:500]}")
        except Exception as e:
            return err("medium", str(e))