
import mimetypes
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from social.services import env, int_env, get_session, err
from social.state import get_db


def _upload_to_imgbb(image_path: str) -> dict:
    api_key = env("IMGBB_API_KEY")
    timeout = int_env("IMGBB_UPLOAD_TIMEOUT", 30)
    p = Path(image_path)
    if not api_key:
        return err("imgbb", "IMGBB_API_KEY not set")
    if not p.exists():
        return err("imgbb", f"image not found: {image_path}")
    mime, _ = mimetypes.guess_type(str(p))
    try:
        session = get_session()
        with open(p, "rb") as f:
            resp = session.post(
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
        return err("imgbb", str(e))


def _get_threads_token() -> str | dict:
    """Get Threads access token, using cache if available."""
    db = get_db()
    cached = db.get_cached_token("threads")
    if cached:
        return cached

    token = env("THREADS_ACCESS_TOKEN")
    base = env("THREADS_API_BASE", "https://graph.threads.net/v1.0").rstrip("/")
    if not token:
        return err("threads", "THREADS_ACCESS_TOKEN not set")

    # Meta's long-lived token lasts ~60 days; refresh proactively within
    # THREADS_REFRESH_BEFORE_EXPIRY_DAYS (default 6) of expiry.
    raw = env("THREADS_ACCESS_TOKEN_EXPIRES_AT")
    if raw:
        try:
            expiry = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if expiry.tzinfo is None:
                expiry = expiry.replace(tzinfo=timezone.utc)
            remaining = (expiry - datetime.now(timezone.utc)).total_seconds()
            refresh_before = int_env("THREADS_REFRESH_BEFORE_EXPIRY_DAYS", 6) * 86400
            if remaining > refresh_before:
                return token  # plenty of time left, use cached
        except ValueError:
            pass

    # Refresh via Meta's th_refresh_token grant
    session = get_session()
    try:
        resp = session.get(
            f"{base}/refresh_access_token",
            params={"grant_type": "th_refresh_token", "access_token": token},
            timeout=120,
        )
        if not (200 <= resp.status_code < 300):
            return err("threads", f"Token refresh failed: {resp.status_code}")
        payload = resp.json()
        new_token = payload.get("access_token")
        if not new_token:
            return err("threads", "No access token in refresh response")
        os.environ["THREADS_ACCESS_TOKEN"] = new_token
        expires_in = int(payload.get("expires_in") or 0)
        if expires_in > 0:
            os.environ["THREADS_ACCESS_TOKEN_EXPIRES_AT"] = (
                datetime.now(timezone.utc) + timedelta(seconds=expires_in)
            ).isoformat()
        db.set_cached_token("threads", new_token, expires_in or 3600)
        return new_token
    except Exception as e:
        return err("threads", f"Token refresh error: {e}")


def load_tools(mcp):
    @mcp.tool(
        name="post_threads",
        description="Post text + optional image + optional single topic tag to Meta Threads",
    )
    def post_threads(text: str, image_path: str | None = None, topic_tag: str | None = None) -> dict:
        token = _get_threads_token()
        if isinstance(token, dict) and not token.get("ok", True):
            return token

        user_id = env("THREADS_USER_ID")
        base = env("THREADS_API_BASE", "https://graph.threads.net/v1.0").rstrip("/")

        if not token or not user_id:
            return err("threads", "THREADS_ACCESS_TOKEN or THREADS_USER_ID not set")

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
        if topic_tag:
            params["topic_tag"] = topic_tag[:50]

        session = get_session()
        try:
            create = session.post(create_url, data=params, timeout=120)
            if not (200 <= create.status_code < 300):
                return {"ok": False, "provider": "threads", "stage": "create", "status_code": create.status_code, "response": create.text[:2000]}
            creation_id = create.json().get("id")
            if not creation_id:
                return {"ok": False, "provider": "threads", "stage": "create", "error": "missing creation id"}
            time.sleep(int_env("THREADS_PUBLISH_DELAY_SECONDS", 5))
            publish = session.post(publish_url, data={"access_token": token, "creation_id": creation_id}, timeout=120)
            ok = 200 <= publish.status_code < 300
            result = {"ok": ok, "provider": "threads", "status_code": publish.status_code, "creation_id": creation_id, "response": publish.text[:2000]}
            if upload_result:
                result["image_upload"] = upload_result
            return result
        except Exception as e:
            return err("threads", str(e))
