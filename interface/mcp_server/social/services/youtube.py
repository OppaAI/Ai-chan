
import mimetypes
from pathlib import Path

from social.services import env, int_env, bool_env, get_session, err, refresh_oauth_token
from social.state import get_db


def _get_youtube_token() -> str | dict:
    """Get YouTube access token, using cache if available."""
    db = get_db()
    cached = db.get_cached_token("youtube")
    if cached:
        return cached

    client_id = env("YOUTUBE_CLIENT_ID")
    client_secret = env("YOUTUBE_CLIENT_SECRET")
    refresh_token = env("YOUTUBE_REFRESH_TOKEN")

    token = refresh_oauth_token(
        "youtube",
        "https://oauth2.googleapis.com/token",
        client_id,
        client_secret,
        refresh_token,
    )
    if isinstance(token, dict) and not token.get("ok", True):
        return token

    # Cache token (expires_in is typically 3600s)
    db.set_cached_token("youtube", token, 3600)
    return token


def load_tools(mcp):
    @mcp.tool(
        name="post_youtube",
        description="Upload video with title + optional description to YouTube",
    )
    def post_youtube(video_path: str, title: str, description: str = "") -> dict:
        token = _get_youtube_token()
        if isinstance(token, dict) and not token.get("ok", True):
            return token
        access_token = token

        timeout = int_env("YOUTUBE_TIMEOUT", 120)

        p = Path(video_path)
        if not p.exists():
            return err("youtube", f"video not found: {video_path}")

        category_id = env("YOUTUBE_CATEGORY_ID", "22")
        privacy_status = env("YOUTUBE_PRIVACY_STATUS", "public").strip().lower()
        made_for_kids = bool_env("YOUTUBE_MADE_FOR_KIDS", False)

        metadata = {
            "snippet": {
                "title": title[:100],
                "description": description[:5000],
                "categoryId": category_id,
            },
            "status": {
                "privacyStatus": privacy_status,
                "selfDeclaredMadeForKids": made_for_kids,
            },
        }
        mime = mimetypes.guess_type(str(p))[0] or "video/mp4"

        session = get_session()
        try:
            init = session.post(
                "https://www.googleapis.com/upload/youtube/v3/videos",
                params={"uploadType": "resumable", "part": "snippet,status"},
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json; charset=UTF-8",
                    "X-Upload-Content-Type": mime,
                },
                json=metadata,
                timeout=timeout,
            )
            if not (200 <= init.status_code < 300):
                return {"ok": False, "provider": "youtube", "stage": "init", "status_code": init.status_code, "response": init.text[:2000]}
            upload_url = init.headers.get("Location")
            if not upload_url:
                return err("youtube", "missing upload URL")

            with open(p, "rb") as f:
                upload = session.put(upload_url, headers={"Content-Type": mime}, data=f, timeout=timeout)
            ok = 200 <= upload.status_code < 300
            upload_payload = upload.json() if upload.text else {}
            return {
                "ok": ok, "provider": "youtube", "status_code": upload.status_code,
                "video_id": upload_payload.get("id") if isinstance(upload_payload, dict) else None,
                "response": upload_payload,
            }
        except Exception as e:
            return err("youtube", str(e))
