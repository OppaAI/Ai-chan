import mimetypes
import base64
from pathlib import Path
from typing import Optional
from social.services import env, int_env, get_session, err


def load_tools(mcp):
    @mcp.tool(
        name="post_x",
        description="Post text + optional image to X/Twitter via twitterapi.io",
    )
    def post_x(text: str, image_path: Optional[str] = None) -> dict:
        api_key = env("TWITTERAPI_KEY")
        base_url = env("TWITTERAPI_BASE_URL", "https://api.twitterapi.io").rstrip("/")
        timeout = int_env("TWITTER_RELAY_TIMEOUT", 30)
        
        if not api_key:
            return err("x", "TWITTERAPI_KEY not set")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload = {"text": text}
        
        # Handle image attachment if provided
        if image_path:
            p = Path(image_path)
            if p.exists():
                try:
                    with open(p, "rb") as f:
                        image_data = base64.b64encode(f.read()).decode("utf-8")
                    mime = mimetypes.guess_type(str(p))[0] or "image/png"
                    payload["media"] = {
                        "data": image_data,
                        "media_type": mime,
                    }
                except Exception as e:
                    return err("x", f"Failed to read image: {str(e)}")

        session = get_session()
        try:
            # twitterapi.io endpoint for posting tweets
            resp = session.post(
                f"{base_url}/tweets/create",
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            ok = 200 <= resp.status_code < 300
            return {
                "ok": ok,
                "provider": "x",
                "status_code": resp.status_code,
                "response": resp.text[:2000],
            }
        except Exception as e:
            return err("x", str(e))
