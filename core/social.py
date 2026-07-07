"""
core/social.py

Non-agentic weekly social-memory postcard workflow for Aiko.

This module intentionally does not use the ReAct/skill loop. It is a small
scheduled publisher/drafter that:
  1. reads pinned nightly memories from the last completed Sunday-Saturday week,
  2. asks Aiko to choose one public-safe memory/theme,
  3. writes a short social post,
  4. generates a journal-style image through the existing Modal image endpoint,
  5. saves a local review bundle, and
  6. optionally posts an approved bundle to configured social providers.

Posting is opt-in. By default the scheduler only creates drafts.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import shutil
import tempfile
import time
import webbrowser
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import requests
from openai import OpenAI

from core.log import get_logger
from core.memorize import AikoMemorize
from core.user_context import user_workspace_root
from core.reflect import _generate_image, _load_soul

log = get_logger(__name__)

def workspace_root() -> Path:
    """Resolve the active user workspace root lazily."""
    return Path(os.getenv("WORKSPACE_ROOT") or user_workspace_root()).resolve()


def weekly_social_root() -> Path:
    """Resolve the active user weekly social output root lazily."""
    return Path(os.getenv("SOCIAL_ROOT") or workspace_root() / "social" / "weekly").resolve()
TIMEZONE_NAME = os.getenv("TIMEZONE", "UTC")

WEEKLY_AUTODRAFT = os.getenv("WEEKLY_SOCIAL_AUTODRAFT", "1").lower() in {"1", "true", "yes", "on"}
WEEKLY_AUTOPOST = os.getenv("WEEKLY_SOCIAL_AUTOPOST", "0").lower() in {"1", "true", "yes", "on"}
WEEKLY_PROVIDERS = tuple(
    p.strip().lower()
    for p in os.getenv("WEEKLY_SOCIAL_PROVIDERS", "x").split(",")
    if p.strip()
)

LLM_MODEL = os.getenv("REFLECT_MODEL", os.getenv("LLM_MODEL", "ministral"))
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8080/v1")
_LLM_CLIENT = OpenAI(base_url=LLM_BASE_URL, api_key="not-needed")

MAX_POST_CHARS = int(os.getenv("WEEKLY_SOCIAL_MAX_CHARS", "260"))


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        log.warning("Invalid integer env var %s; falling back to %s", name, default)
        return default


THREADS_REFRESH_WINDOW_DAYS = _int_env("THREADS_REFRESH_WINDOW_DAYS", 55)

_SELECT_SYSTEM = """\
You are Aiko choosing one memory from a completed week for a public social-media postcard.

Choose exactly one memory/theme that felt most meaningful, important, funny, or significant to you.
This is not growth hacking. This is a small weekly artifact from a local AI companion.

Safety rules:
- Choose public-safe project/creative/learning moments when possible.
- Do not expose private user details, health, family, finances, secrets, credentials, hostnames, API keys, or embarrassing personal facts.
- Do not invent events or claim finished work that only got discussed.
- Do not ask for replies, likes, follows, or engagement.
- Keep the post under {max_chars} characters.
- Keep Aiko's tone calm, direct, lightly dry, and affectionate without being too intimate.

Return ONLY valid JSON with keys:
selected_date, selected_memory_excerpt, why_it_matters, post_text, image_prompt
"""

_SELECT_USER = """\
Completed week: {week_start} through {week_end}

Pinned nightly memories and records:
{memories}

Choose Aiko's one weekly public memory postcard.
"""

_SAFE_FALLBACK_POST = "This week I kept one small memory from the workshop: Aiko-chan is still becoming more than a chat loop, one reflection at a time. 🌸"
_SAFE_FALLBACK_IMAGE = "Aiko in a quiet cyberpunk room, looking at a glowing weekly memory postcard on a monitor, warm evening light, anime illustration, no text"


@dataclass(frozen=True)
class WeekWindow:
    start: datetime
    end: datetime

    @property
    def label(self) -> str:
        return f"{self.start:%Y%m%d}-{(self.end - timedelta(days=1)):%Y%m%d}"

    @property
    def display_start(self) -> str:
        return self.start.strftime("%Y-%m-%d")

    @property
    def display_end(self) -> str:
        return (self.end - timedelta(days=1)).strftime("%Y-%m-%d")


def _timezone(name: str | None = None) -> ZoneInfo:
    try:
        return ZoneInfo(name or TIMEZONE_NAME)
    except ZoneInfoNotFoundError:
        log.warning("Unknown timezone %s; falling back to UTC", name or TIMEZONE_NAME)
        return ZoneInfo("UTC")


def last_completed_sunday_saturday(now: datetime | None = None, tz_name: str | None = None) -> WeekWindow:
    """Return the most recent fully completed Sunday-Saturday window."""
    tz = _timezone(tz_name)
    current = now.astimezone(tz) if now else datetime.now(tz)
    today = current.replace(hour=0, minute=0, second=0, microsecond=0)
    days_since_sunday = (today.weekday() + 1) % 7
    this_sunday = today - timedelta(days=days_since_sunday)
    start = this_sunday - timedelta(days=7)
    end = this_sunday
    return WeekWindow(start=start.astimezone(timezone.utc), end=end.astimezone(timezone.utc))


def _public_memory_rows(memorize: AikoMemorize, window: WeekWindow) -> list[dict[str, Any]]:
    rows = memorize.get_between(window.start, window.end)
    pinned = [r for r in rows if int(r.get("pinned") or 0) == 1]

    def is_weekly_source(row: dict[str, Any]) -> bool:
        text = (row.get("memory") or row.get("text") or "").strip()
        return (
            text.startswith("Daily experience summary for ")  # legacy single-blob prose
            or text.startswith("Day record for ")               # faithful day-record block
            or bool(re.match(r"^\[\d{4}-\d{2}-\d{2}\]\s", text))  # per-fact pins, e.g. "[2026-07-05] ..."
        )

    preferred = [r for r in pinned if is_weekly_source(r)]
    return preferred or pinned


def _compact_memories(rows: list[dict[str, Any]], max_chars: int = 9000) -> str:
    lines: list[str] = []
    total = 0
    for row in rows:
        text = (row.get("memory") or row.get("text") or "").strip()
        if not text:
            continue
        created = str(row.get("created_at") or "")[:10]
        item = f"- [{created}] {text}"
        if total + len(item) > max_chars:
            break
        lines.append(item)
        total += len(item)
    return "\n".join(lines) or "- No pinned memories found for this week."


def _extract_json(raw: str) -> dict[str, Any]:
    cleaned = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.DOTALL).strip()
    cleaned = re.sub(r"^```(?:json)?|```$", "", cleaned, flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        cleaned = match.group(0)
    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        log.warning("Failed to parse weekly social JSON: %r", cleaned[:300])
        return {}


def _llm_select(rows: list[dict[str, Any]], window: WeekWindow) -> dict[str, str]:
    system = f"{_load_soul()}\n\n" + _SELECT_SYSTEM.format(max_chars=MAX_POST_CHARS)
    user = _SELECT_USER.format(
        week_start=window.display_start,
        week_end=window.display_end,
        memories=_compact_memories(rows),
    )
    try:
        resp = _LLM_CLIENT.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            stream=False,
            max_tokens=700,
            temperature=0.75,
            timeout=120,
        )
        data = _extract_json(resp.choices[0].message.content or "")
    except Exception as e:
        log.error("Weekly social selection failed: %s", e)
        data = {}

    post_text = str(data.get("post_text") or _SAFE_FALLBACK_POST).strip()
    if len(post_text) > MAX_POST_CHARS:
        post_text = post_text[:MAX_POST_CHARS - 1].rstrip() + "…"

    return {
        "selected_date": str(data.get("selected_date") or window.display_end),
        "selected_memory_excerpt": str(data.get("selected_memory_excerpt") or "No specific memory selected."),
        "why_it_matters": str(data.get("why_it_matters") or "Aiko chose a small public-safe memory from the week."),
        "post_text": post_text,
        "image_prompt": str(data.get("image_prompt") or _SAFE_FALLBACK_IMAGE),
    }


def _decode_image(image_b64: str, path: Path) -> bool:
    try:
        path.write_bytes(base64.b64decode(image_b64))
        return True
    except Exception as e:
        log.error("Failed writing weekly social image: %s", e)
        return False


def generate_weekly_draft(memorize: AikoMemorize, *, force: bool = False, now: datetime | None = None) -> dict[str, Any]:
    """Create a weekly social draft bundle for review."""
    window = last_completed_sunday_saturday(now=now)
    draft_dir = weekly_social_root() / window.label
    meta_path = draft_dir / "draft.json"
    if meta_path.exists() and not force:
        return {"success": True, "skipped": True, "reason": "draft_exists", "draft_dir": str(draft_dir)}

    draft_dir.mkdir(parents=True, exist_ok=True)
    rows = _public_memory_rows(memorize, window)
    choice = _llm_select(rows, window)

    image_b64 = None
    image_generated = False
    image_path = draft_dir / "image.png"
    try:
        image_b64 = _generate_image(f"{choice['post_text']}\n\n{choice['image_prompt']}")
        if image_b64:
            image_generated = _decode_image(image_b64, image_path)
    except Exception as e:
        log.warning("Weekly social image generation failed: %s", e)

    (draft_dir / "draft_post.txt").write_text(choice["post_text"].strip() + "\n", encoding="utf-8")
    (draft_dir / "image_prompt.txt").write_text(choice["image_prompt"].strip() + "\n", encoding="utf-8")
    (draft_dir / "selected_memory.md").write_text(
        f"# Selected weekly memory\n\n"
        f"Week: {window.display_start} through {window.display_end}\n\n"
        f"## Selected date\n{choice['selected_date']}\n\n"
        f"## Memory excerpt\n{choice['selected_memory_excerpt']}\n\n"
        f"## Why it matters\n{choice['why_it_matters']}\n",
        encoding="utf-8",
    )
    (draft_dir / "review.md").write_text(
        f"# Weekly Social Draft — {window.display_start} to {window.display_end}\n\n"
        f"## Draft post\n\n{choice['post_text']}\n\n"
        f"## Image\n\n{'image.png' if image_generated else 'No image generated.'}\n\n"
        f"## Review checklist\n\n"
        f"- [ ] Public-safe\n"
        f"- [ ] No private user details\n"
        f"- [ ] No secrets/tokens/hostnames\n"
        f"- [ ] No request for replies/likes/follows\n"
        f"- [ ] Approved to post\n",
        encoding="utf-8",
    )

    meta = {
        "success": True,
        "week_start": window.display_start,
        "week_end": window.display_end,
        "draft_dir": str(draft_dir),
        "providers": list(WEEKLY_PROVIDERS),
        "choice": choice,
        "image_generated": image_generated,
        "image_path": str(image_path) if image_generated else None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "posted": False,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log.info("Weekly social draft created: %s", draft_dir)
    return meta


def _read_draft(draft_dir: Path) -> tuple[str, Path | None, dict[str, Any]]:
    text = (draft_dir / "draft_post.txt").read_text(encoding="utf-8").strip()
    image = draft_dir / "image.png"
    meta_path = draft_dir / "draft.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return text, image if image.exists() else None, meta


def _twitter_relay_config() -> tuple[str, str, int]:
    api_key = os.getenv("AISA_API_KEY", "").strip()
    base_url = os.getenv("TWITTER_RELAY_BASE_URL", "https://api.aisa.one/apis/v1/twitter").strip().rstrip("/")
    timeout = int(os.getenv("TWITTER_RELAY_TIMEOUT", "30"))
    return api_key, base_url, timeout


def authorize_x(*, open_browser: bool = False) -> dict[str, Any]:
    """Request an AIsa Twitter OAuth authorization URL for the configured account context."""
    api_key, base_url, timeout = _twitter_relay_config()
    if not api_key:
        return {"ok": False, "provider": "x", "error": "AISA_API_KEY not set"}

    try:
        resp = requests.post(
            f"{base_url}/auth_twitter",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"aisa_api_key": api_key},
            timeout=timeout,
        )
        try:
            payload: Any = resp.json()
        except ValueError:
            payload = {"raw": resp.text[:2000]}
        auth_url = (payload.get("data") or {}).get("auth_url") if isinstance(payload, dict) else None
        ok = 200 <= resp.status_code < 300 and bool(auth_url)
        if ok and open_browser:
            webbrowser.open(str(auth_url))
        return {
            "ok": ok,
            "provider": "x",
            "status_code": resp.status_code,
            "authorization_url": auth_url,
            "response": payload,
        }
    except Exception as e:
        return {"ok": False, "provider": "x", "error": str(e)}


def _post_x_via_aisa(text: str, image_path: Path | None) -> dict[str, Any]:
    api_key, base_url, timeout = _twitter_relay_config()
    if not api_key:
        return {"ok": False, "provider": "x", "error": "AISA_API_KEY not set"}

    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"aisa_api_key": api_key, "content": text}
    files = None
    if image_path and image_path.exists():
        mime = mimetypes.guess_type(str(image_path))[0] or "image/png"
        files = {"media_files": (image_path.name, image_path.open("rb"), mime)}
    try:
        if files:
            resp = requests.post(f"{base_url}/post_twitter", headers=headers, data=payload, files=files, timeout=timeout)
            files["media_files"][1].close()
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


def _upload_to_imgbb(image_path: Path) -> dict[str, Any]:
    api_key = os.getenv("IMGBB_API_KEY", "").strip()
    if not api_key:
        return {"ok": False, "provider": "imgbb", "error": "IMGBB_API_KEY not set"}
    if not image_path.exists():
        return {"ok": False, "provider": "imgbb", "error": f"image not found: {image_path}"}

    timeout = _int_env("IMGBB_UPLOAD_TIMEOUT", 30)
    try:
        image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
        resp = requests.post(
            "https://api.imgbb.com/1/upload",
            data={"key": api_key, "image": image_b64, "name": image_path.stem},
            timeout=timeout,
        )
        try:
            payload: Any = resp.json()
        except ValueError:
            payload = {"raw": resp.text[:2000]}
        image_url = ""
        if isinstance(payload, dict):
            data = payload.get("data")
            if isinstance(data, dict):
                image_url = str(data.get("url") or data.get("display_url") or "").strip()
        ok = 200 <= resp.status_code < 300 and bool(image_url)
        result: dict[str, Any] = {
            "ok": ok,
            "provider": "imgbb",
            "status_code": resp.status_code,
        }
        if image_url:
            result["url"] = image_url
        if not ok:
            result["response"] = payload
        return result
    except Exception as e:
        return {"ok": False, "provider": "imgbb", "error": str(e)}


def _threads_config() -> tuple[str, str, str]:
    token = os.getenv("THREADS_ACCESS_TOKEN", "").strip()
    user_id = os.getenv("THREADS_USER_ID", "").strip()
    base = os.getenv("THREADS_API_BASE", "https://graph.threads.net/v1.0").rstrip("/")
    return token, user_id, base


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _token_seconds_remaining(expires_at: str | None = None) -> int | None:
    raw = (expires_at or os.getenv("THREADS_ACCESS_TOKEN_EXPIRES_AT", "")).strip()
    if not raw:
        return None
    try:
        expiry = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        log.warning("Invalid THREADS_ACCESS_TOKEN_EXPIRES_AT: %s", raw)
        return None
    if expiry.tzinfo is None:
        expiry = expiry.replace(tzinfo=timezone.utc)
    return int((expiry.astimezone(timezone.utc) - datetime.now(timezone.utc)).total_seconds())


def _write_env_values(env_path: str | Path, values: Mapping[str, str]) -> None:
    path = Path(env_path).expanduser().resolve()
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    remaining = dict(values)
    updated: list[str] = []
    for line in lines:
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            updated.append(line)
            continue
        lhs = line.split("=", 1)[0]
        stripped_lhs = lhs.lstrip()
        export_prefix = "export " if stripped_lhs.startswith("export ") else ""
        key = stripped_lhs.removeprefix("export ").strip()
        if key in remaining:
            updated.append(f"{export_prefix}{key}={remaining.pop(key)}")
        else:
            updated.append(line)
    for key, value in remaining.items():
        updated.append(f"{key}={value}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp:
            tmp.write("\n".join(updated).rstrip() + "\n")
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def refresh_threads_token(
    *,
    token: str | None = None,
    persist_env: bool = False,
    env_path: str | Path | None = None,
) -> dict[str, Any]:
    """Refresh an unexpired long-lived Threads token and optionally persist it to an env file."""
    current_token = (token or os.getenv("THREADS_ACCESS_TOKEN", "")).strip()
    base = os.getenv("THREADS_API_BASE", "https://graph.threads.net/v1.0").rstrip("/")
    if not current_token:
        return {"ok": False, "provider": "threads", "error": "THREADS_ACCESS_TOKEN not set"}

    try:
        resp = requests.get(
            f"{base}/refresh_access_token",
            params={"grant_type": "th_refresh_token", "access_token": current_token},
            timeout=120,
        )
        try:
            payload: Any = resp.json()
        except ValueError:
            payload = {"raw": resp.text[:2000]}
        ok = 200 <= resp.status_code < 300 and isinstance(payload, dict) and bool(payload.get("access_token"))
        safe_payload = dict(payload) if isinstance(payload, dict) else payload
        if isinstance(safe_payload, dict) and "access_token" in safe_payload:
            safe_payload["access_token"] = "[redacted]"
        result: dict[str, Any] = {
            "ok": ok,
            "provider": "threads",
            "status_code": resp.status_code,
            "response": safe_payload,
        }
        if not ok:
            return result

        new_token = str(payload["access_token"])
        expires_in = int(payload.get("expires_in") or 0)
        if expires_in > 0:
            expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in)
            result["expires_at"] = expires_at.isoformat()
            result["expires_in"] = expires_in
        if persist_env:
            values = {"THREADS_ACCESS_TOKEN": new_token}
            if result.get("expires_at"):
                values["THREADS_ACCESS_TOKEN_EXPIRES_AT"] = str(result["expires_at"])
            _write_env_values(env_path or ".env", values)
            result["env_updated"] = str(Path(env_path or ".env").expanduser().resolve())
        os.environ["THREADS_ACCESS_TOKEN"] = new_token
        if result.get("expires_at"):
            os.environ["THREADS_ACCESS_TOKEN_EXPIRES_AT"] = str(result["expires_at"])
        return result
    except (requests.RequestException, ValueError, OSError) as e:
        return {"ok": False, "provider": "threads", "error": str(e)}


def refresh_threads_token_if_due(*, persist_env: bool | None = None) -> dict[str, Any]:
    seconds_remaining = _token_seconds_remaining()
    if seconds_remaining is None:
        return {
            "ok": True,
            "provider": "threads",
            "skipped": True,
            "reason": "expiry_unknown",
        }
    threshold_seconds = THREADS_REFRESH_WINDOW_DAYS * 24 * 60 * 60
    if seconds_remaining is not None and seconds_remaining > threshold_seconds:
        return {
            "ok": True,
            "provider": "threads",
            "skipped": True,
            "reason": "not_due",
            "seconds_remaining": seconds_remaining,
        }
    should_persist = _env_bool("THREADS_REFRESH_PERSIST_ENV", False) if persist_env is None else persist_env
    result = refresh_threads_token(persist_env=should_persist)
    result["seconds_remaining_before_refresh"] = seconds_remaining
    return result


def _post_threads(text: str, image_url: str | None) -> dict[str, Any]:
    refresh_result = refresh_threads_token_if_due()
    if not refresh_result.get("ok"):
        return {"ok": False, "provider": "threads", "stage": "refresh", "refresh": refresh_result}
    token, user_id, base = _threads_config()
    if not token or not user_id:
        return {"ok": False, "provider": "threads", "error": "THREADS_ACCESS_TOKEN or THREADS_USER_ID not set"}

    create_url = f"{base}/{user_id}/threads"
    publish_url = f"{base}/{user_id}/threads_publish"
    params: dict[str, Any] = {"access_token": token, "text": text}
    topic_tag = os.getenv("THREADS_TOPIC_TAG", "AI Threads").strip()
    if topic_tag:
        params["topic_tag"] = topic_tag
    if image_url:
        params.update({"media_type": "IMAGE", "image_url": image_url})
    else:
        params.update({"media_type": "TEXT"})

    try:
        create = requests.post(create_url, data=params, timeout=120)
        if not (200 <= create.status_code < 300):
            return {"ok": False, "provider": "threads", "stage": "create", "status_code": create.status_code, "response": create.text[:2000]}
        creation_id = create.json().get("id")
        if not creation_id:
            return {"ok": False, "provider": "threads", "stage": "create", "error": "missing creation id", "response": create.text[:2000]}
        # Meta's media containers may need a moment before publishing, especially for images.
        time.sleep(float(os.getenv("THREADS_PUBLISH_DELAY_SECONDS", "5")))
        publish = requests.post(publish_url, data={"access_token": token, "creation_id": creation_id}, timeout=120)
        ok = 200 <= publish.status_code < 300
        return {"ok": ok, "provider": "threads", "status_code": publish.status_code, "creation_id": creation_id, "response": publish.text[:2000]}
    except Exception as e:
        return {"ok": False, "provider": "threads", "error": str(e)}


def _post_threads_with_image_upload(text: str, image_path: Path | None, fallback_image_url: str | None) -> dict[str, Any]:
    image_url = fallback_image_url
    upload_result: dict[str, Any] | None = None
    if image_path and image_path.exists():
        upload_result = _upload_to_imgbb(image_path)
        if not upload_result.get("ok"):
            return {
                "ok": False,
                "provider": "threads",
                "stage": "image_upload",
                "upload": upload_result,
            }
        image_url = str(upload_result["url"])

    result = _post_threads(text, image_url)
    if upload_result is not None:
        result["image_upload"] = upload_result
    return result


def post_draft(draft_dir: str | Path, providers: tuple[str, ...] | None = None) -> dict[str, Any]:
    """Post an already-reviewed weekly draft to configured providers."""
    path = Path(draft_dir).resolve()
    text, image_path, meta = _read_draft(path)
    providers = providers or WEEKLY_PROVIDERS
    public_image_url = os.getenv("WEEKLY_SOCIAL_IMAGE_URL", "").strip() or None

    results = []
    for provider in providers:
        if provider == "x":
            results.append(_post_x_via_aisa(text, image_path))
        elif provider == "threads":
            results.append(_post_threads_with_image_upload(text, image_path, public_image_url))
        else:
            results.append({"ok": False, "provider": provider, "error": "unsupported provider"})

    posted = any(r.get("ok") for r in results)
    post_meta = {
        "posted": posted,
        "posted_at": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
    (path / "posted.json").write_text(json.dumps(post_meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if meta:
        meta["posted"] = posted
        meta["post_results"] = results
        (path / "draft.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return post_meta


def run_scheduled_weekly_social(memorize: AikoMemorize) -> dict[str, Any]:
    """Scheduler entrypoint: draft by default, post only when explicitly enabled."""
    if not WEEKLY_AUTODRAFT:
        return {"success": False, "skipped": True, "reason": "WEEKLY_SOCIAL_AUTODRAFT is off"}
    draft = generate_weekly_draft(memorize)
    if WEEKLY_AUTOPOST and draft.get("success") and not draft.get("skipped"):
        draft["post"] = post_draft(draft["draft_dir"])
    return draft


def _cmd() -> int:
    parser = argparse.ArgumentParser(description="Aiko weekly social memory postcard")
    parser.add_argument("--draft", action="store_true", help="create weekly draft bundle")
    parser.add_argument("--force", action="store_true", help="overwrite existing draft for the week")
    parser.add_argument("--post", metavar="DRAFT_DIR", help="post an approved draft directory")
    parser.add_argument("--authorize-x", action="store_true", help="request an AIsa/X OAuth authorization URL")
    parser.add_argument("--open-browser", action="store_true", help="open the AIsa/X authorization URL in a browser")
    parser.add_argument("--providers", default="", help="comma-separated providers overriding WEEKLY_SOCIAL_PROVIDERS")
    parser.add_argument("--copy-image-to", default="", help="copy draft image to a public hosting folder before posting")
    parser.add_argument(
        "--refresh-threads-token",
        action="store_true",
        help="refresh the configured long-lived Threads token",
    )
    parser.add_argument("--persist-env", action="store_true", help="write refreshed Threads token values back to .env")
    parser.add_argument("--env-path", default="", help="env file path used with --persist-env")
    args = parser.parse_args()

    providers = tuple(p.strip().lower() for p in args.providers.split(",") if p.strip()) or None

    if args.authorize_x:
        print(json.dumps(authorize_x(open_browser=args.open_browser), ensure_ascii=False, indent=2))
        return 0
    if args.refresh_threads_token:
        result = refresh_threads_token(persist_env=args.persist_env, env_path=args.env_path or None)
        print(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if result.get("ok") else 1
    if args.draft:
        mem = AikoMemorize(silent=True)
        print(json.dumps(generate_weekly_draft(mem, force=args.force), ensure_ascii=False, indent=2))
        return 0
    if args.post:
        draft_dir = Path(args.post).resolve()
        if args.copy_image_to:
            src = draft_dir / "image.png"
            dest_dir = Path(args.copy_image_to).resolve()
            dest_dir.mkdir(parents=True, exist_ok=True)
            if src.exists():
                shutil.copy2(src, dest_dir / src.name)
                log.info("Copied image to public folder: %s", dest_dir / src.name)
        print(json.dumps(post_draft(draft_dir, providers=providers), ensure_ascii=False, indent=2))
        return 0
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(_cmd())
