from __future__ import annotations

import mimetypes
from pathlib import Path

import requests

from social_mcp.tools.base import env, int_env, bool_env


def load_tools(mcp):
    @mcp.tool(
        name="post_youtube",
        description="Upload video with title + optional description to YouTube",
    )
    def post_youtube(video_path: str, title: str, description: str = "") -> dict:
        client_id = env("YOUTUBE_CLIENT_ID")
        client_secret = env("YOUTUBE_CLIENT_SECRET")
        refresh_token = env("YOUTUBE_REFRESH_TOKEN")
        timeout = int_env("YOUTUBE_TIMEOUT", 120)

        if not client_id or not client_secret or not refresh_token:
            return {"ok": False, "provider": "youtube", "error": "YOUTUBE_CLIENT_ID, YOUTUBE_CLIENT_SECRET, or YOUTUBE_REFRESH_TOKEN not set"}

        try:
            resp = requests.post(
                "https://oauth2.googleapis.com/token",
                data={"client_id": client_id, "client_secret": client_secret, "refresh_token": refresh_token, "grant_type": "refresh_token"},
                timeout=30,
            )
            payload = resp.json()
            access_token = payload.get("access_token") if 200 <= resp.status_code < 300 else None
            if not access_token:
                return {"ok": False, "provider": "youtube", "stage": "token_refresh", "status_code": resp.status_code, "response": payload}
        except Exception as e:
            return {"ok": False, "provider": "youtube", "stage": "token_refresh", "error": str(e)}

        p = Path(video_path)
        if not p.exists():
            return {"ok": False, "provider": "youtube", "error": f"video not found: {video_path}"}

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

        try:
            init = requests.post(
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
                return {"ok": False, "provider": "youtube", "stage": "init", "error": "missing upload URL"}

            with open(p, "rb") as f:
                upload = requests.put(upload_url, headers={"Content-Type": mime}, data=f, timeout=timeout)
            ok = 200 <= upload.status_code < 300
            upload_payload = upload.json() if upload.text else {}
            return {
                "ok": ok, "provider": "youtube", "status_code": upload.status_code,
                "video_id": upload_payload.get("id") if isinstance(upload_payload, dict) else None,
                "response": upload_payload,
            }
        except Exception as e:
            return {"ok": False, "provider": "youtube", "error": str(e)}
