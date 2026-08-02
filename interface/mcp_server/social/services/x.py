
import mimetypes
from pathlib import Path
from typing import Optional

from social.services import env, int_env, get_session, err


def load_tools(mcp):
    @mcp.tool(
        name="post_x",
        description="Post text + optional image to X/Twitter via AIsa relay",
    )
    def post_x(text: str, image_path: Optional[str] = None) -> dict:
        api_key = env("AISA_API_KEY")
        base_url = env("TWITTER_RELAY_BASE_URL", "https://api.aisa.one/apis/v1/twitter").rstrip("/")
        timeout = int_env("TWITTER_RELAY_TIMEOUT", 30)

        if not api_key:
            return err("x", "AISA_API_KEY not set")

        headers = {"Authorization": f"Bearer {api_key}"}
        payload = {"aisa_api_key": api_key, "content": text}
        files = None

        if image_path:
            p = Path(image_path)
            if p.exists():
                mime = mimetypes.guess_type(str(p))[0] or "image/png"
                with open(p, "rb") as f:
                    files = {"media_files": (p.name, f.read(), mime)}

        session = get_session()
        try:
            if files:
                resp = session.post(
                    f"{base_url}/post_twitter", headers=headers, data=payload, files=files, timeout=timeout
                )
            else:
                resp = session.post(
                    f"{base_url}/post_twitter",
                    headers={**headers, "Content-Type": "application/json"},
                    json=payload,
                    timeout=timeout,
                )
            ok = 200 <= resp.status_code < 300
            return {"ok": ok, "provider": "x", "status_code": resp.status_code, "response": resp.text[:2000]}
        except Exception as e:
            return err("x", str(e))
