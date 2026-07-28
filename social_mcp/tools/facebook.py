from __future__ import annotations

from pathlib import Path

import requests

from social_mcp.tools.base import env


def load_tools(mcp):
    @mcp.tool(
        name="post_facebook",
        description="Post text + optional image to a Facebook Page feed",
    )
    def post_facebook(text: str, image_path: str | None = None) -> dict:
        page_id = env("FB_PAGE_ID")
        page_token = env("FB_PAGE_ACCESS_TOKEN")
        api_version = env("FB_API_VERSION", "v21.0")

        if not page_id or not page_token:
            return {"ok": False, "provider": "facebook", "error": "FB_PAGE_ID or FB_PAGE_ACCESS_TOKEN not set"}

        base = f"https://graph.facebook.com/{api_version}/{page_id}"

        try:
            if image_path:
                p = Path(image_path)
                if not p.exists():
                    return {"ok": False, "provider": "facebook", "error": f"image not found: {image_path}"}
                with open(p, "rb") as f:
                    resp = requests.post(
                        f"{base}/photos",
                        data={"caption": text, "access_token": page_token},
                        files={"source": f},
                        timeout=60,
                    )
            else:
                resp = requests.post(
                    f"{base}/feed",
                    data={"message": text, "access_token": page_token},
                    timeout=30,
                )

            ok = 200 <= resp.status_code < 300
            return {"ok": ok, "provider": "facebook", "status_code": resp.status_code, "response": resp.text[:500]}
        except Exception as e:
            return {"ok": False, "provider": "facebook", "error": str(e)}
