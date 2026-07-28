from __future__ import annotations

import mimetypes
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from social_mcp.tools.base import env, int_env


def _upload_to_imgbb(image_path: str) -> dict:
    api_key = env("IMGBB_API_KEY")
    timeout = int_env("IMGBB_UPLOAD_TIMEOUT", 30)
    p = Path(image_path)
    if not api_key:
        return {"ok": False, "provider": "imgbb", "error": "IMGBB_API_KEY not set"}
    if not p.exists():
        return {"ok": False, "provider": "imgbb", "error": f"image not found: {image_path}"}
    mime, _ = mimetypes.guess_type(str(p))
    try:
        with open(p, "rb") as f:
            resp = requests.post(
                "https://api.imgbb.com/1/upload",
                data={"key": api_key, "name": p.stem},
                files={"image": (p.name, f, mime or "image/jpeg")},
                timeout=timeout,
            )
        payload = resp.json() if resp.text else {}
        image_url = ""
        if isinstance(payload, dict):
            data = payload.get("data")
            if isinstance(data, dict):
                image_url = str(data.get("url") or data.get("display_url") or "").strip()
        ok = 200 <= resp.status_code < 300 and bool(image_url)
        result = {"ok": ok, "provider": "imgbb", "status_code": resp.status_code}
        if image_url:
            result["url"] = image_url
        if not ok:
            result["response"] = payload
        return result
    except Exception as e:
        return {"ok": False, "provider": "imgbb", "error": str(e)}


def _refresh_token_if_due() -> bool:
    raw = env("IG_ACCESS_TOKEN_EXPIRES_AT")
    if not raw:
        return True
    try:
        expiry = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return True
    if expiry.tzinfo is None:
        expiry = expiry.replace(tzinfo=timezone.utc)
    remaining = (expiry - datetime.now(timezone.utc)).total_seconds()
    threshold = int_env("IG_REFRESH_WINDOW_DAYS", 55) * 86400
    if remaining > threshold:
        return True
    token = env("IG_ACCESS_TOKEN")
    base = env("IG_API_BASE", "https://graph.instagram.com").rstrip("/")
    if not token:
        return False
    try:
        resp = requests.get(
            f"{base}/refresh_access_token",
            params={"grant_type": "ig_refresh_token", "access_token": token},
            timeout=120,
        )
        if not (200 <= resp.status_code < 300):
            return False
        payload = resp.json()
        new_token = payload.get("access_token")
        if not new_token:
            return False
        os.environ["IG_ACCESS_TOKEN"] = new_token
        expires_in = int(payload.get("expires_in") or 0)
        if expires_in > 0:
            os.environ["IG_ACCESS_TOKEN_EXPIRES_AT"] = (
                datetime.now(timezone.utc) + timedelta(seconds=expires_in)
            ).isoformat()
        return True
    except Exception:
        return False


def load_tools(mcp):
    @mcp.tool(
        name="post_instagram",
        description="Post image + optional caption to Instagram",
    )
    def post_instagram(image_path: str, caption: str = "") -> dict:
        _refresh_token_if_due()
        token = env("IG_ACCESS_TOKEN")
        ig_user_id = env("IG_BUSINESS_ACCOUNT_ID")
        base = env("IG_API_BASE", "https://graph.instagram.com").rstrip("/")

        if not token or not ig_user_id:
            return {"ok": False, "provider": "instagram", "error": "IG_ACCESS_TOKEN or IG_BUSINESS_ACCOUNT_ID not set"}

        upload = _upload_to_imgbb(image_path)
        if not upload.get("ok"):
            return {"ok": False, "provider": "instagram", "stage": "image_upload", "upload": upload}
        image_url = upload["url"]

        timeout = int_env("IG_TIMEOUT", 60)
        try:
            create = requests.post(
                f"{base}/{ig_user_id}/media",
                data={"image_url": image_url, "caption": caption[:2200], "access_token": token},
                timeout=timeout,
            )
            if not (200 <= create.status_code < 300):
                return {"ok": False, "provider": "instagram", "stage": "create", "status_code": create.status_code, "response": create.text[:2000]}
            creation_id = create.json().get("id")
            if not creation_id:
                return {"ok": False, "provider": "instagram", "stage": "create", "error": "missing creation id"}
            time.sleep(int_env("IG_PUBLISH_DELAY_SECONDS", 5))
            publish = requests.post(
                f"{base}/{ig_user_id}/media_publish",
                data={"creation_id": creation_id, "access_token": token},
                timeout=timeout,
            )
            ok = 200 <= publish.status_code < 300
            return {"ok": ok, "provider": "instagram", "status_code": publish.status_code, "creation_id": creation_id, "response": publish.text[:2000], "image_upload": upload}
        except Exception as e:
            return {"ok": False, "provider": "instagram", "error": str(e)}
