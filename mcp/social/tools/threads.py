from __future__ import annotations

import mimetypes
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from social.tools.base import env, int_env


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
    raw = env("THREADS_ACCESS_TOKEN_EXPIRES_AT")
    if not raw:
        return True
    try:
        expiry = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return True
    if expiry.tzinfo is None:
        expiry = expiry.replace(tzinfo=timezone.utc)
    remaining = (expiry - datetime.now(timezone.utc)).total_seconds()
    threshold = int_env("THREADS_REFRESH_WINDOW_DAYS", 55) * 86400
    if remaining > threshold:
        return True
    token = env("THREADS_ACCESS_TOKEN")
    base = env("THREADS_API_BASE", "https://graph.threads.net/v1.0").rstrip("/")
    if not token:
        return False
    try:
        resp = requests.get(
            f"{base}/refresh_access_token",
            params={"grant_type": "th_refresh_token", "access_token": token},
            timeout=120,
        )
        if not (200 <= resp.status_code < 300):
            return False
        payload = resp.json()
        new_token = payload.get("access_token")
        if not new_token:
            return False
        os.environ["THREADS_ACCESS_TOKEN"] = new_token
        expires_in = int(payload.get("expires_in") or 0)
        if expires_in > 0:
            os.environ["THREADS_ACCESS_TOKEN_EXPIRES_AT"] = (
                datetime.now(timezone.utc) + timedelta(seconds=expires_in)
            ).isoformat()
        return True
    except Exception:
        return False


def load_tools(mcp):
    @mcp.tool(
        name="post_threads",
        description="Post text + optional image to Meta Threads",
    )
    def post_threads(text: str, image_path: str | None = None) -> dict:
        _refresh_token_if_due()
        token = env("THREADS_ACCESS_TOKEN")
        user_id = env("THREADS_USER_ID")
        base = env("THREADS_API_BASE", "https://graph.threads.net/v1.0").rstrip("/")

        if not token or not user_id:
            return {"ok": False, "provider": "threads", "error": "THREADS_ACCESS_TOKEN or THREADS_USER_ID not set"}

        image_url = None
        upload_result = None
        if image_path:
            upload_result = _upload_to_imgbb(image_path)
            if not upload_result.get("ok"):
                return {"ok": False, "provider": "threads", "stage": "image_upload", "upload": upload_result}
            image_url = upload_result["url"]

        create_url = f"{base}/{user_id}/threads"
        publish_url = f"{base}/{user_id}/threads_publish"
        params = {"access_token": token, "text": text}
        if image_url:
            params.update({"media_type": "IMAGE", "image_url": image_url})
        else:
            params["media_type"] = "TEXT"

        try:
            create = requests.post(create_url, data=params, timeout=120)
            if not (200 <= create.status_code < 300):
                return {"ok": False, "provider": "threads", "stage": "create", "status_code": create.status_code, "response": create.text[:2000]}
            creation_id = create.json().get("id")
            if not creation_id:
                return {"ok": False, "provider": "threads", "stage": "create", "error": "missing creation id"}
            time.sleep(int_env("THREADS_PUBLISH_DELAY_SECONDS", 5))
            publish = requests.post(publish_url, data={"access_token": token, "creation_id": creation_id}, timeout=120)
            ok = 200 <= publish.status_code < 300
            result = {"ok": ok, "provider": "threads", "status_code": publish.status_code, "creation_id": creation_id, "response": publish.text[:2000]}
            if upload_result:
                result["image_upload"] = upload_result
            return result
        except Exception as e:
            return {"ok": False, "provider": "threads", "error": str(e)}
