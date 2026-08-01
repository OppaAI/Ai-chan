from __future__ import annotations

import mimetypes
from pathlib import Path

import requests

from social.tools.base import env, int_env


def load_tools(mcp):
    @mcp.tool(
        name="post_x",
        description="Post text + optional image to X/Twitter via AIsa relay",
    )
    def post_x(text: str, image_path: str | None = None) -> dict:
        api_key = env("AISA_API_KEY")
        base_url = env("TWITTER_RELAY_BASE_URL", "https://api.aisa.one/apis/v1/twitter").rstrip("/")
        timeout = int_env("TWITTER_RELAY_TIMEOUT", 30)

        if not api_key:
            return {"ok": False, "provider": "x", "error": "AISA_API_KEY not set"}

        headers = {"Authorization": f"Bearer {api_key}"}
        payload = {"aisa_api_key": api_key, "content": text}
        files = None

        if image_path:
            p = Path(image_path)
            if p.exists():
                mime = mimetypes.guess_type(str(p))[0] or "image/png"
                with open(p, "rb") as f:
                    files = {"media_files": (p.name, f.read(), mime)}

        try:
            if files:
                resp = requests.post(
                    f"{base_url}/post_twitter", headers=headers, data=payload, files=files, timeout=timeout
                )
            else:
                resp = requests.post(
                    f"{base_url}/post_twitter",
                    headers={**headers, "Content-Type": "application/json"},
                    json=payload,
                    timeout=timeout,
                )
            ok = 200 <= resp.status_code < 300
            return {"ok": ok, "provider": "x", "status_code": resp.status_code, "response": resp.text[:2000]}
        except Exception as e:
            return {"ok": False, "provider": "x", "error": str(e)}
