"""
skills/social.py

Aiko's social publishing workflows, combined into one module. Three lanes:

  Lane A — Weekly memory postcard (non-agentic, scheduled):
    1. reads pinned nightly memories from the last completed Sun-Sat week,
    2. asks Aiko to choose one public-safe memory/theme,
    3. writes a short social post,
    4. generates a journal-style image through the existing Modal image endpoint,
    5. saves a local review bundle, and
    6. optionally posts an approved bundle to X and/or Threads.

  Lane B — Curated photo showcase (grounded in real media, not LLM-invented
  text/imagery):
    1. scan the photo inbox (toolkit/photography.py),
    2. caption each candidate via a vision model (grounded in actual pixels),
    3. ask Aiko to pick 1-3 items worth sharing and write a short caption,
    4. save a local review bundle, then optionally post to Instagram.

  Lane C — YouTube video queue (no grading — you already chose the video by
  dropping it in the folder; Aiko only polishes your description):
    1. scan the video inbox,
    2. queue the oldest not-yet-drafted video that has a matching
       description file — a video named e.g. "my_trip.mp4" needs a
       sibling markdown file named "MY_TRIP.md" (the video's filename
       stem, uppercased, ".md" extension) in the same folder,
    3. ask Aiko to polish that markdown into a YouTube title + description
       (no invented claims — grounded in what you wrote),
    4. save a local review bundle, then optionally post to YouTube.

Posting is opt-in for all three lanes. By default the scheduler only
creates drafts.

Supported providers (by design, current as of this revision):
  - Lane A (weekly postcard): x, threads
  - Lane B (curated photos):  instagram (photos only, no video)
  - Lane C (video queue):     youtube

Bluesky, Mastodon, and Pixelfed support has been removed — those platforms'
communities have expressed they don't want AI-posted content, so Aiko no
longer posts there. Instagram no longer posts video (Aiko doesn't post video
there); video instead goes to YouTube via its own lane. If a future platform
should be added, follow the existing pattern: one _post_<provider>(...)
function plus a registry entry; the post_draft / post_photo_draft /
post_video_draft dispatchers never need to change otherwise.
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
from typing import Any, Callable, Mapping
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import requests
from openai import OpenAI

from system.bioclock import get_timezone, timezone_name
from system.log import get_logger
from memory.memorize import AikoMemorize
from system.userspace import user_workspace_root
from memory.reflect import _generate_image, _load_soul
from toolkit.common import workspace_root
from toolkit.photography import scan_photo_workspace, scan_video_workspace

log = get_logger(__name__)


# ── shared paths ──────────────────────────────────────────────────────────────

def weekly_social_root() -> Path:
    """Resolve the active user weekly social output root lazily.

    Defaults to <USER_STATE_ROOT>/<user_id>/workspace/social/weekly. Holds
    draft bundles for weekly social-media postcards, including selected
    memories, draft posts, and generated images.
    """
    override = os.getenv("SOCIAL_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return (user_workspace_root() / "social" / "weekly").resolve()


def photo_social_root() -> Path:
    """Resolve the active user photo-social output root lazily.

    Defaults to <USER_STATE_ROOT>/<user_id>/workspace/social/photo.
    """
    override = os.getenv("PHOTO_SOCIAL_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return (user_workspace_root() / "social" / "photo").resolve()


def video_social_root() -> Path:
    """Resolve the active user video-social output root lazily.

    Defaults to <USER_STATE_ROOT>/<user_id>/workspace/social/video.
    """
    override = os.getenv("VIDEO_SOCIAL_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return (user_workspace_root() / "social" / "video").resolve()


# ── shared helpers ────────────────────────────────────────────────────────────

def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        log.warning("Invalid integer env var %s; falling back to %s", name, default)
        return default


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
        log.warning("Failed to parse social JSON: %r", cleaned[:300])
        return {}


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


# Shared text LLM (used for both the weekly memory-selection prompt and the
# photo caption-selection prompt).
LLM_MODEL = os.getenv("REFLECT_MODEL", os.getenv("LLM_MODEL", "ministral"))
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8080/v1")
_LLM_CLIENT = OpenAI(base_url=LLM_BASE_URL, api_key="not-needed")


# ── imgbb image hosting (shared by Threads + Instagram, both of which need
#    a public image URL rather than a direct upload) ─────────────────────────

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


# ══════════════════════════════════════════════════════════════════════════
# Lane A — Weekly memory postcard (X + Threads only)
# ══════════════════════════════════════════════════════════════════════════

WEEKLY_AUTODRAFT = os.getenv("WEEKLY_SOCIAL_AUTODRAFT", "1").lower() in {"1", "true", "yes", "on"}
WEEKLY_AUTOPOST = os.getenv("WEEKLY_SOCIAL_AUTOPOST", "0").lower() in {"1", "true", "yes", "on"}
# Bluesky and Mastodon have been dropped from the default provider set — see
# module docstring. Valid values now: "x", "threads".
WEEKLY_PROVIDERS = tuple(
    p.strip().lower()
    for p in os.getenv("WEEKLY_SOCIAL_PROVIDERS", "x,threads").split(",")
    if p.strip()
)

MAX_POST_CHARS = int(os.getenv("WEEKLY_SOCIAL_MAX_CHARS", "260"))

THREADS_REFRESH_WINDOW_DAYS = _int_env("THREADS_REFRESH_WINDOW_DAYS", 55)

_WEEKLY_SELECT_SYSTEM = """\
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

_WEEKLY_SELECT_USER = """\
Completed week: {week_start} through {week_end}

Pinned nightly memories and records:
{memories}

Choose Aiko's one weekly public memory postcard.
"""

_SAFE_FALLBACK_POST = "This week I kept one small memory from the workshop: Aiko-chan is still becoming more than a chat loop, one reflection at a time. \U0001f338"
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


def last_completed_sunday_saturday(now: datetime | None = None, tz_name: str | None = None) -> WeekWindow:
    """Return the most recent fully completed Sunday-Saturday window."""
    tz = get_timezone(tz_name)
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
            or text.startswith("Day record for ")               # legacy faithful day-record block
            or text.startswith("Daily journal of ")          # legacy memory-hosted journal block
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


def _llm_select_weekly(rows: list[dict[str, Any]], window: WeekWindow) -> dict[str, str]:
    system = f"{_load_soul()}\n\n" + _WEEKLY_SELECT_SYSTEM.format(max_chars=MAX_POST_CHARS)
    user = _WEEKLY_SELECT_USER.format(
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
        post_text = post_text[:MAX_POST_CHARS - 1].rstrip() + "\u2026"

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
    choice = _llm_select_weekly(rows, window)

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
        f"# Weekly Social Draft \u2014 {window.display_start} to {window.display_end}\n\n"
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


def _read_weekly_draft(draft_dir: Path) -> tuple[str, Path | None, dict[str, Any]]:
    text = (draft_dir / "draft_post.txt").read_text(encoding="utf-8").strip()
    image = draft_dir / "image.png"
    meta_path = draft_dir / "draft.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return text, image if image.exists() else None, meta


# ── X (via AIsa relay) ────────────────────────────────────────────────────

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
        image_bytes = image_path.read_bytes()
        files = {"media_files": (image_path.name, image_bytes, mime)}
    try:
        if files:
            resp = requests.post(f"{base_url}/post_twitter", headers=headers, data=payload, files=files, timeout=timeout)
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


# ── Threads ───────────────────────────────────────────────────────────────

def _threads_config() -> tuple[str, str, str]:
    token = os.getenv("THREADS_ACCESS_TOKEN", "").strip()
    user_id = os.getenv("THREADS_USER_ID", "").strip()
    base = os.getenv("THREADS_API_BASE", "https://graph.threads.net/v1.0").rstrip("/")
    return token, user_id, base


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


# ── provider registry (weekly postcard) ──────────────────────────────────────
# Bluesky and Mastodon removed by policy — see module docstring. Add a new
# platform by writing one _post_<provider>(text, image_path) -> dict function
# above and registering it here. post_draft never needs to change.

_WEEKLY_PROVIDERS_REGISTRY: dict[str, Callable[[str, Path | None], dict[str, Any]]] = {
    "x": _post_x_via_aisa,
    "threads": lambda text, image_path: _post_threads_with_image_upload(
        text, image_path, os.getenv("WEEKLY_SOCIAL_IMAGE_URL", "").strip() or None
    ),
}


def post_draft(draft_dir: str | Path, providers: tuple[str, ...] | None = None) -> dict[str, Any]:
    """Post an already-reviewed weekly draft to configured providers (x, threads)."""
    path = Path(draft_dir).resolve()
    text, image_path, meta = _read_weekly_draft(path)
    providers = providers or WEEKLY_PROVIDERS

    results = []
    for provider in providers:
        handler = _WEEKLY_PROVIDERS_REGISTRY.get(provider)
        if handler is None:
            results.append({"ok": False, "provider": provider, "error": "unsupported provider"})
            continue
        results.append(handler(text, image_path))

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


# ══════════════════════════════════════════════════════════════════════════
# Lane B — Curated photo showcase (Instagram only)
# ══════════════════════════════════════════════════════════════════════════

PHOTO_SOCIAL_AUTODRAFT = os.getenv("PHOTO_SOCIAL_AUTODRAFT", "0").lower() in {"1", "true", "yes", "on"}
PHOTO_SOCIAL_AUTOPOST = os.getenv("PHOTO_SOCIAL_AUTOPOST", "0").lower() in {"1", "true", "yes", "on"}
# Pixelfed dropped from the default provider set — see module docstring.
PHOTO_SOCIAL_PROVIDERS = tuple(
    p.strip().lower()
    for p in os.getenv("PHOTO_SOCIAL_PROVIDERS", "instagram").split(",")
    if p.strip()
)
PHOTO_SOCIAL_INBOX = os.getenv("PHOTO_SOCIAL_INBOX", "photos/inbox")
PHOTO_SOCIAL_MAX_ITEMS = _int_env("PHOTO_SOCIAL_MAX_ITEMS", 3)
MAX_CAPTION_CHARS = _int_env("PHOTO_SOCIAL_MAX_CHARS", 260)

# Vision model (captioning) — separate client/model from the text LLM above,
# since captioning needs actual image understanding (e.g. MiniCPM-V), not
# the text-only Ministral endpoint used for selection.
VISION_MODEL = os.getenv("VISION_MODEL", os.getenv("REFLECT_VISION_MODEL", "minicpm-v"))
VISION_BASE_URL = os.getenv("VISION_BASE_URL", os.getenv("LLM_BASE_URL", "http://localhost:8080/v1"))
_VISION_CLIENT = OpenAI(base_url=VISION_BASE_URL, api_key="not-needed")

_CAPTION_PROMPT = (
    "Describe this image in one plain, factual sentence. No hashtags, no "
    "hype, no marketing language. If it looks private, sensitive, or "
    "identifies a specific real person's face clearly, start your reply "
    "with 'PRIVATE:' instead of a description."
)

_MEDIA_SELECT_SYSTEM = """\
You are Aiko choosing which recent photo(s) are worth sharing publicly.

You are given plain factual captions of each candidate file (not the images
themselves). Choose at most {max_items} that are genuinely worth sharing —
it is fine to choose zero if nothing fits.

Safety rules:
- Never choose anything captioned as PRIVATE, or that plausibly shows an
  identifiable person, private location, screen contents, or document.
- Do not invent details beyond the given captions.
- Do not ask for replies, likes, follows, or engagement.
- Keep each caption under {max_chars} characters.
- Keep Aiko's tone calm, direct, lightly dry, and affectionate without being
  too intimate.

Return ONLY valid JSON with a single key "selections": a list of objects,
each with keys: filename, caption. Return an empty list if nothing is
worth sharing this round.
"""

_MEDIA_SELECT_USER = """\
Candidate files and their factual captions:
{items}

Choose Aiko's public media selection, if any.
"""


@dataclass
class MediaCandidate:
    path: Path
    raw_caption: str = ""
    private: bool = False


@dataclass
class MediaSelection:
    path: Path
    caption: str


def _encode_image_data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def _caption_media(path: Path) -> MediaCandidate:
    """Caption one image via the vision model."""
    try:
        resp = _VISION_CLIENT.chat.completions.create(
            model=VISION_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": _CAPTION_PROMPT},
                    {"type": "image_url", "image_url": {"url": _encode_image_data_uri(path)}},
                ],
            }],
            stream=False,
            max_tokens=120,
            temperature=0.2,
            timeout=60,
        )
        raw = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        log.warning("Vision captioning failed for %s: %s", path, e)
        return MediaCandidate(path=path, raw_caption="[captioning failed]", private=True)

    private = raw.upper().startswith("PRIVATE")
    return MediaCandidate(path=path, raw_caption=raw, private=private)


def _list_candidates(inbox: str, limit: int) -> list[Path]:
    """scan_photo_workspace() returns a json_block-formatted STRING (label +
    embedded JSON), not a Python list — parse it the same way _extract_json
    parses LLM output elsewhere in this module. Note the tool itself caps
    its "files" preview at 50 regardless of image_count, and only scans
    IMAGE_EXTENSIONS (no video formats) — video support would need to be
    added upstream in toolkit/photography.py first."""
    raw = scan_photo_workspace(inbox, limit)
    match = re.search(r"\{.*\}", raw or "", flags=re.DOTALL)
    if not match:
        log.warning("Could not parse scan_photo_workspace output: %r", (raw or "")[:200])
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        log.warning("Invalid JSON from scan_photo_workspace: %r", (raw or "")[:200])
        return []

    root = workspace_root()
    paths: list[Path] = []
    for rel in data.get("files") or []:
        try:
            paths.append((root / rel).resolve())
        except Exception:
            continue
    return paths


def _llm_select_media(candidates: list[MediaCandidate]) -> list[MediaSelection]:
    public_candidates = [c for c in candidates if not c.private]
    if not public_candidates:
        return []

    items_block = "\n".join(f"- {c.path.name}: {c.raw_caption}" for c in public_candidates)
    system = f"{_load_soul()}\n\n" + _MEDIA_SELECT_SYSTEM.format(
        max_items=PHOTO_SOCIAL_MAX_ITEMS, max_chars=MAX_CAPTION_CHARS,
    )
    user = _MEDIA_SELECT_USER.format(items=items_block)

    try:
        resp = _LLM_CLIENT.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            stream=False,
            max_tokens=500,
            temperature=0.6,
            timeout=90,
        )
        data = _extract_json(resp.choices[0].message.content or "")
    except Exception as e:
        log.error("Photo social selection failed: %s", e)
        data = {}

    by_name = {c.path.name: c for c in public_candidates}
    selections: list[MediaSelection] = []
    for item in (data.get("selections") or [])[:PHOTO_SOCIAL_MAX_ITEMS]:
        filename = str(item.get("filename") or "").strip()
        caption = str(item.get("caption") or "").strip()
        if filename not in by_name or not caption:
            continue
        if len(caption) > MAX_CAPTION_CHARS:
            caption = caption[:MAX_CAPTION_CHARS - 1].rstrip() + "\u2026"
        selections.append(MediaSelection(path=by_name[filename].path, caption=caption))
    return selections


def generate_photo_draft(*, inbox: str | None = None, force: bool = False) -> dict[str, Any]:
    """Scan the inbox, caption + select candidates, and write a review bundle."""
    inbox_path = inbox or PHOTO_SOCIAL_INBOX
    label = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    draft_dir = photo_social_root() / label
    meta_path = draft_dir / "draft.json"
    if meta_path.exists() and not force:
        return {"success": True, "skipped": True, "reason": "draft_exists", "draft_dir": str(draft_dir)}

    # NOTE: scan_photo_workspace's own "files" preview is hardcapped at 50
    # regardless of the limit passed here (see toolkit/photography.py) — if
    # your inbox regularly holds more than 50 untouched images, that cap
    # needs raising upstream, not here.
    file_paths = _list_candidates(inbox_path, limit=50)
    if not file_paths:
        return {"success": True, "skipped": True, "reason": "empty_inbox", "inbox": inbox_path}

    candidates = [_caption_media(p) for p in file_paths]
    selections = _llm_select_media(candidates)

    if not selections:
        return {
            "success": True,
            "skipped": True,
            "reason": "nothing_selected",
            "inbox": inbox_path,
            "candidates_considered": len(candidates),
        }

    draft_dir.mkdir(parents=True, exist_ok=True)
    media_dir = draft_dir / "media"
    media_dir.mkdir(exist_ok=True)

    saved_selections = []
    for sel in selections:
        try:
            dest = media_dir / sel.path.name
            dest.write_bytes(sel.path.read_bytes())
            saved_selections.append({"filename": sel.path.name, "caption": sel.caption, "media_path": str(dest)})
        except Exception as e:
            log.warning("Failed copying selected media %s: %s", sel.path, e)

    (draft_dir / "review.md").write_text(
        f"# Photo Social Draft \u2014 {label}\n\n"
        f"Source inbox: {inbox_path}\n\n"
        + "\n\n".join(
            f"## {s['filename']}\n\n{s['caption']}\n\n![preview]({Path(s['media_path']).name})"
            for s in saved_selections
        )
        + "\n\n## Review checklist\n\n"
        f"- [ ] Public-safe (no identifiable people/private locations/documents)\n"
        f"- [ ] Captions accurate to the actual media\n"
        f"- [ ] No request for replies/likes/follows\n"
        f"- [ ] Approved to post\n",
        encoding="utf-8",
    )

    meta = {
        "success": True,
        "label": label,
        "draft_dir": str(draft_dir),
        "inbox": inbox_path,
        "providers": list(PHOTO_SOCIAL_PROVIDERS),
        "selections": saved_selections,
        "candidates_considered": len(candidates),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "posted": False,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log.info("Photo social draft created: %s (%d item(s))", draft_dir, len(saved_selections))
    return meta


def _read_media_draft(draft_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    meta_path = draft_dir / "draft.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return meta.get("selections", []), meta


# ── Instagram (Graph API) ────────────────────────────────────────────────────
# Requires an IG Business/Creator account linked to a Facebook Page, and a
# publicly reachable URL for the media (no direct file upload). Images only —
# Aiko does not post video.

def _instagram_config() -> tuple[str, str, str]:
    token = os.getenv("IG_ACCESS_TOKEN", "").strip()
    ig_user_id = os.getenv("IG_BUSINESS_ACCOUNT_ID", "").strip()
    base = os.getenv("IG_API_BASE", "https://graph.facebook.com/v21.0").rstrip("/")
    return token, ig_user_id, base


def _post_instagram_image(sel: dict[str, Any]) -> dict[str, Any]:
    token, ig_user_id, base = _instagram_config()
    if not token or not ig_user_id:
        return {"ok": False, "provider": "instagram", "error": "IG_ACCESS_TOKEN or IG_BUSINESS_ACCOUNT_ID not set"}

    timeout = _int_env("IG_TIMEOUT", 60)
    media_path = Path(sel["media_path"])
    if not media_path.exists():
        return {"ok": False, "provider": "instagram", "error": f"media not found: {media_path}"}

    upload = _upload_to_imgbb(media_path)
    if not upload.get("ok"):
        return {"ok": False, "provider": "instagram", "stage": "image_upload", "upload": upload}
    image_url = upload["url"]

    try:
        create = requests.post(
            f"{base}/{ig_user_id}/media",
            data={"image_url": image_url, "caption": sel.get("caption", "")[:2200], "access_token": token},
            timeout=timeout,
        )
        if not (200 <= create.status_code < 300):
            return {"ok": False, "provider": "instagram", "stage": "create", "status_code": create.status_code, "response": create.text[:2000]}
        creation_id = create.json().get("id")
        if not creation_id:
            return {"ok": False, "provider": "instagram", "stage": "create", "error": "missing creation id", "response": create.text[:2000]}

        time.sleep(float(os.getenv("IG_PUBLISH_DELAY_SECONDS", "5")))
        publish = requests.post(
            f"{base}/{ig_user_id}/media_publish",
            data={"creation_id": creation_id, "access_token": token},
            timeout=timeout,
        )
        ok = 200 <= publish.status_code < 300
        return {
            "ok": ok, "provider": "instagram", "status_code": publish.status_code,
            "creation_id": creation_id, "response": publish.text[:2000], "image_upload": upload,
        }
    except Exception as e:
        return {"ok": False, "provider": "instagram", "error": str(e)}


def _post_instagram(selections: list[dict[str, Any]]) -> dict[str, Any]:
    """Posts the FIRST selection only, as an image. Carousel (multi-item)
    posting needs child containers and is not implemented — extend here if
    needed. Images only — Aiko does not post video."""
    if not selections:
        return {"ok": False, "provider": "instagram", "error": "no selections to post"}
    return _post_instagram_image(selections[0])


# ── provider registry (curated media) ────────────────────────────────────────
# Pixelfed removed by policy — see module docstring. Instagram is now the
# only media provider, and only for photos (after grading) — Aiko does not
# post video.

_MEDIA_PROVIDERS_REGISTRY: dict[str, Callable[[list[dict[str, Any]]], dict[str, Any]]] = {
    "instagram": _post_instagram,
}


def post_photo_draft(draft_dir: str | Path, providers: tuple[str, ...] | None = None) -> dict[str, Any]:
    path = Path(draft_dir).resolve()
    selections, meta = _read_media_draft(path)
    providers = providers or PHOTO_SOCIAL_PROVIDERS

    results = []
    for provider in providers:
        handler = _MEDIA_PROVIDERS_REGISTRY.get(provider)
        if handler is None:
            results.append({"ok": False, "provider": provider, "error": "unsupported provider"})
            continue
        results.append(handler(selections))

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


def run_scheduled_photo_social() -> dict[str, Any]:
    """Scheduler entrypoint: draft by default, post only when explicitly enabled."""
    if not PHOTO_SOCIAL_AUTODRAFT:
        return {"success": False, "skipped": True, "reason": "PHOTO_SOCIAL_AUTODRAFT is off"}
    draft = generate_photo_draft()
    if PHOTO_SOCIAL_AUTOPOST and draft.get("success") and not draft.get("skipped"):
        draft["post"] = post_photo_draft(draft["draft_dir"])
    return draft


# ══════════════════════════════════════════════════════════════════════════
# Lane C — YouTube video queue (no grading — you already chose the video by
# dropping it in the folder; Aiko only polishes the description you wrote)
# ══════════════════════════════════════════════════════════════════════════

VIDEO_SOCIAL_AUTODRAFT = os.getenv("VIDEO_SOCIAL_AUTODRAFT", "0").lower() in {"1", "true", "yes", "on"}
VIDEO_SOCIAL_AUTOPOST = os.getenv("VIDEO_SOCIAL_AUTOPOST", "0").lower() in {"1", "true", "yes", "on"}
VIDEO_SOCIAL_PROVIDERS = tuple(
    p.strip().lower()
    for p in os.getenv("VIDEO_SOCIAL_PROVIDERS", "youtube").split(",")
    if p.strip()
)
VIDEO_SOCIAL_INBOX = os.getenv("VIDEO_SOCIAL_INBOX", "videos")
MAX_YOUTUBE_TITLE_CHARS = _int_env("YOUTUBE_MAX_TITLE_CHARS", 100)
MAX_YOUTUBE_DESCRIPTION_CHARS = _int_env("YOUTUBE_MAX_DESCRIPTION_CHARS", 5000)

_VIDEO_POLISH_SYSTEM = """\
You are Aiko turning a human-written video note into a YouTube title and description.

The note below is written by the person who made/chose this video. It is the
ONLY source of truth — do not invent claims, events, locations, dates, or
specs that are not in the note. Your job is to tidy grammar/flow, tighten it,
and format it for YouTube; not to add new content.

Rules:
- title: under {max_title} characters, plain and descriptive, no clickbait,
  no ALL CAPS, no emoji spam (a single tasteful emoji is fine if it fits
  Aiko's voice).
- description: under {max_description} characters, calm/direct/lightly dry
  tone, expands a little on the title using only what the note already says.
- Do not add hashtags unless the note itself already suggests specific ones.
- Do not ask for likes/subscribes/comments.
- If the note is empty or just a filename-like fragment, keep the title
  minimal and say so plainly in the description rather than inventing detail.

Return ONLY valid JSON with keys: title, description
"""

_VIDEO_POLISH_USER = """\
Video filename: {filename}

Raw note (from {md_filename}):
{raw_note}

Polish this into a YouTube title and description.
"""


def _video_ledger_path() -> Path:
    return video_social_root() / "_video_ledger.json"


def _load_video_ledger() -> dict[str, float]:
    path = _video_ledger_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning("Could not read video ledger, treating as empty: %s", e)
        return {}


def _save_video_ledger(ledger: dict[str, float]) -> None:
    path = _video_ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _list_video_candidates(inbox: str, limit: int) -> list[Path]:
    """Mirrors _list_candidates (photo lane) but for scan_video_workspace's
    output shape. Same upstream caveat: the tool's own "files" preview is
    hardcapped at 50 regardless of the limit passed here."""
    raw = scan_video_workspace(inbox, limit)
    match = re.search(r"\{.*\}", raw or "", flags=re.DOTALL)
    if not match:
        log.warning("Could not parse scan_video_workspace output: %r", (raw or "")[:200])
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        log.warning("Invalid JSON from scan_video_workspace: %r", (raw or "")[:200])
        return []

    root = workspace_root()
    paths: list[Path] = []
    for rel in data.get("files") or []:
        try:
            paths.append((root / rel).resolve())
        except Exception:
            continue
    return paths


def _description_md_for(video_path: Path) -> Path:
    """A video "my_trip.mp4" needs a sibling "MY_TRIP.md" (filename stem,
    uppercased, .md extension) in the same folder — written by hand, not
    generated. Aiko only polishes it, never invents it."""
    return video_path.with_name(video_path.stem.upper() + ".md")


def _llm_polish_video_description(video_path: Path, md_path: Path) -> dict[str, str]:
    raw_note = md_path.read_text(encoding="utf-8").strip()
    system = f"{_load_soul()}\n\n" + _VIDEO_POLISH_SYSTEM.format(
        max_title=MAX_YOUTUBE_TITLE_CHARS, max_description=MAX_YOUTUBE_DESCRIPTION_CHARS,
    )
    user = _VIDEO_POLISH_USER.format(
        filename=video_path.name, md_filename=md_path.name, raw_note=raw_note or "(empty)",
    )

    fallback_title = video_path.stem.replace("_", " ").replace("-", " ").strip() or video_path.stem
    try:
        resp = _LLM_CLIENT.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            stream=False,
            max_tokens=600,
            temperature=0.5,
            timeout=90,
        )
        data = _extract_json(resp.choices[0].message.content or "")
    except Exception as e:
        log.error("Video description polish failed: %s", e)
        data = {}

    title = str(data.get("title") or fallback_title).strip()
    if len(title) > MAX_YOUTUBE_TITLE_CHARS:
        title = title[:MAX_YOUTUBE_TITLE_CHARS - 1].rstrip() + "\u2026"

    description = str(data.get("description") or raw_note).strip()
    if len(description) > MAX_YOUTUBE_DESCRIPTION_CHARS:
        description = description[:MAX_YOUTUBE_DESCRIPTION_CHARS - 1].rstrip() + "\u2026"

    return {"title": title, "description": description, "raw_note": raw_note}


def generate_video_draft(*, inbox: str | None = None) -> dict[str, Any]:
    """Queue the oldest not-yet-drafted video that has a matching NAME.md
    description file sitting next to it in the videos inbox.

    Deliberately does NOT run any vision/LLM selection over which video to
    post — dropping the file in the folder (with its description) IS the
    selection. This only decides ORDER (oldest first among videos that
    already have a description) and prevents re-drafting the same file
    twice via a small local ledger. Videos without a matching NAME.md are
    left alone and picked up automatically once you add one.
    """
    inbox_path = inbox or VIDEO_SOCIAL_INBOX
    candidates = _list_video_candidates(inbox_path, limit=50)
    if not candidates:
        return {"success": True, "skipped": True, "reason": "empty_inbox", "inbox": inbox_path}

    ledger = _load_video_ledger()
    unprocessed = [p for p in candidates if str(p) not in ledger]
    ready = [p for p in unprocessed if _description_md_for(p).exists()]
    if not ready:
        return {
            "success": True,
            "skipped": True,
            "reason": "no_new_videos" if not unprocessed else "waiting_on_description_md",
            "inbox": inbox_path,
            "pending_without_description": [p.name for p in unprocessed if p not in ready],
        }

    ready.sort(key=lambda p: p.stat().st_mtime)
    video_path = ready[0]
    md_path = _description_md_for(video_path)

    polished = _llm_polish_video_description(video_path, md_path)

    label = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    draft_dir = video_social_root() / label
    media_dir = draft_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)

    dest = media_dir / video_path.name
    dest.write_bytes(video_path.read_bytes())

    (draft_dir / "title.txt").write_text(polished["title"] + "\n", encoding="utf-8")
    (draft_dir / "description.txt").write_text(polished["description"].strip() + "\n", encoding="utf-8")
    (draft_dir / "raw_note.md").write_text(polished["raw_note"] + "\n", encoding="utf-8")
    (draft_dir / "review.md").write_text(
        f"# Video Social Draft \u2014 {label}\n\n"
        f"Source: {video_path.name} (description from {md_path.name})\n\n"
        f"## Title\n\n{polished['title']}\n\n"
        f"## Description\n\n{polished['description']}\n\n"
        f"Edit title.txt / description.txt before posting to override the "
        f"polished text (read fresh at --post time).\n\n"
        f"## Review checklist\n\n"
        f"- [ ] Public-safe (no identifiable people/private locations)\n"
        f"- [ ] Title/description accurate to the video\n"
        f"- [ ] No request for likes/subscribes/comments\n"
        f"- [ ] Approved to post\n",
        encoding="utf-8",
    )

    selections = [{
        "filename": video_path.name,
        "title": polished["title"],
        "description": polished["description"],
        "media_path": str(dest),
    }]
    meta = {
        "success": True,
        "label": label,
        "draft_dir": str(draft_dir),
        "kind": "video",
        "source": str(video_path),
        "source_md": str(md_path),
        "providers": list(VIDEO_SOCIAL_PROVIDERS),
        "selections": selections,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "posted": False,
    }
    (draft_dir / "draft.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    ledger[str(video_path)] = video_path.stat().st_mtime
    _save_video_ledger(ledger)

    log.info("Video social draft created: %s (source=%s)", draft_dir, video_path.name)
    return meta


def generate_video_drafts(*, inbox: str | None = None, max_drafts: int | None = None) -> list[dict[str, Any]]:
    """Drain the videos inbox (those with a matching NAME.md), one draft per video."""
    results: list[dict[str, Any]] = []
    count = 0
    while max_drafts is None or count < max_drafts:
        result = generate_video_draft(inbox=inbox)
        if result.get("skipped"):
            if count == 0:
                results.append(result)
            break
        results.append(result)
        count += 1
    return results


def _read_video_draft(draft_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    meta_path = draft_dir / "draft.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    selections = meta.get("selections", [])
    # title.txt / description.txt always win over the cached draft.json
    # values, so manual edits made after polishing take effect at post time.
    if len(selections) == 1:
        title_override = draft_dir / "title.txt"
        desc_override = draft_dir / "description.txt"
        if title_override.exists():
            selections[0]["title"] = title_override.read_text(encoding="utf-8").strip()
        if desc_override.exists():
            selections[0]["description"] = desc_override.read_text(encoding="utf-8").strip()
    return selections, meta


# ── YouTube (Data API v3) ─────────────────────────────────────────────────
# OAuth2 only (no API key). One-time browser consent gets you a refresh
# token; every upload after that goes straight through — YouTube does not
# gate individual uploads on manual approval. Quota is the real limit:
# a video insert costs 1600 units against a default 10,000/day quota
# (~6 uploads/day) until the Cloud project is verified.

def _youtube_config() -> tuple[str, str, str]:
    client_id = os.getenv("YOUTUBE_CLIENT_ID", "").strip()
    client_secret = os.getenv("YOUTUBE_CLIENT_SECRET", "").strip()
    refresh_token = os.getenv("YOUTUBE_REFRESH_TOKEN", "").strip()
    return client_id, client_secret, refresh_token


def _youtube_access_token() -> tuple[str | None, dict[str, Any] | None]:
    """Exchange the stored refresh token for a short-lived access token.
    Returns (access_token, error_dict); exactly one will be non-None."""
    client_id, client_secret, refresh_token = _youtube_config()
    if not client_id or not client_secret or not refresh_token:
        return None, {
            "ok": False, "provider": "youtube",
            "error": "YOUTUBE_CLIENT_ID, YOUTUBE_CLIENT_SECRET, or YOUTUBE_REFRESH_TOKEN not set",
        }
    timeout = _int_env("YOUTUBE_TIMEOUT", 30)
    try:
        resp = requests.post(
            "https://oauth2.googleapis.com/token",
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "refresh_token": refresh_token,
                "grant_type": "refresh_token",
            },
            timeout=timeout,
        )
        payload = resp.json()
        token = payload.get("access_token") if 200 <= resp.status_code < 300 else None
        if not token:
            return None, {"ok": False, "provider": "youtube", "stage": "token_refresh", "status_code": resp.status_code, "response": payload}
        return token, None
    except Exception as e:
        return None, {"ok": False, "provider": "youtube", "stage": "token_refresh", "error": str(e)}


def _post_youtube(sel: dict[str, Any]) -> dict[str, Any]:
    """Resumable upload: POST metadata to open a session, then PUT the file
    bytes to the returned upload URL. Fine for the modest file sizes this
    queue produces; very large files would want real chunked PUTs instead
    of one big body."""
    access_token, err = _youtube_access_token()
    if err:
        return err

    media_path = Path(sel["media_path"])
    if not media_path.exists():
        return {"ok": False, "provider": "youtube", "error": f"media not found: {media_path}"}

    timeout = _int_env("YOUTUBE_TIMEOUT", 120)
    category_id = os.getenv("YOUTUBE_CATEGORY_ID", "22")  # 22 = People & Blogs
    privacy_status = os.getenv("YOUTUBE_PRIVACY_STATUS", "public").strip().lower()
    made_for_kids = _env_bool("YOUTUBE_MADE_FOR_KIDS", False)

    metadata = {
        "snippet": {
            "title": sel.get("title", media_path.stem)[:MAX_YOUTUBE_TITLE_CHARS],
            "description": sel.get("description", "")[:MAX_YOUTUBE_DESCRIPTION_CHARS],
            "categoryId": category_id,
        },
        "status": {
            "privacyStatus": privacy_status,
            "selfDeclaredMadeForKids": made_for_kids,
        },
    }
    mime = mimetypes.guess_type(str(media_path))[0] or "video/mp4"

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
            return {"ok": False, "provider": "youtube", "stage": "init", "error": "missing resumable upload Location header"}

        video_bytes = media_path.read_bytes()
        upload = requests.put(
            upload_url,
            headers={"Content-Type": mime},
            data=video_bytes,
            timeout=timeout,
        )
        ok = 200 <= upload.status_code < 300
        try:
            payload = upload.json()
        except ValueError:
            payload = {"raw": upload.text[:2000]}
        return {
            "ok": ok, "provider": "youtube", "status_code": upload.status_code,
            "video_id": payload.get("id") if isinstance(payload, dict) else None,
            "response": payload,
        }
    except Exception as e:
        return {"ok": False, "provider": "youtube", "error": str(e)}


# ── provider registry (video) ────────────────────────────────────────────────
_VIDEO_PROVIDERS_REGISTRY: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "youtube": _post_youtube,
}


def post_video_draft(draft_dir: str | Path, providers: tuple[str, ...] | None = None) -> dict[str, Any]:
    """Post an already-reviewed video draft. Posts the single queued video
    (no carousel/multi-video concept here, unlike the photo lane)."""
    path = Path(draft_dir).resolve()
    selections, meta = _read_video_draft(path)
    providers = providers or VIDEO_SOCIAL_PROVIDERS
    sel = selections[0] if selections else None

    results = []
    for provider in providers:
        handler = _VIDEO_PROVIDERS_REGISTRY.get(provider)
        if handler is None:
            results.append({"ok": False, "provider": provider, "error": "unsupported provider"})
            continue
        if sel is None:
            results.append({"ok": False, "provider": provider, "error": "no video in draft"})
            continue
        results.append(handler(sel))

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


def run_scheduled_video_social() -> dict[str, Any]:
    """Scheduler entrypoint for the video lane: draft (queue) by default,
    post only when explicitly enabled."""
    if not VIDEO_SOCIAL_AUTODRAFT:
        return {"success": False, "skipped": True, "reason": "VIDEO_SOCIAL_AUTODRAFT is off"}
    draft = generate_video_draft()
    if VIDEO_SOCIAL_AUTOPOST and draft.get("success") and not draft.get("skipped"):
        draft["post"] = post_video_draft(draft["draft_dir"])
    return draft


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════

def _cmd() -> int:
    parser = argparse.ArgumentParser(description="Aiko social publishing (weekly postcard + curated media)")
    sub = parser.add_subparsers(dest="mode", required=True)

    weekly_p = sub.add_parser("weekly", help="weekly memory postcard (X, Threads)")
    weekly_p.add_argument("--draft", action="store_true", help="create weekly draft bundle")
    weekly_p.add_argument("--force", action="store_true", help="overwrite existing draft for the week")
    weekly_p.add_argument("--post", metavar="DRAFT_DIR", help="post an approved draft directory")
    weekly_p.add_argument("--authorize-x", action="store_true", help="request an AIsa/X OAuth authorization URL")
    weekly_p.add_argument("--open-browser", action="store_true", help="open the AIsa/X authorization URL in a browser")
    weekly_p.add_argument("--providers", default="", help="comma-separated providers overriding WEEKLY_SOCIAL_PROVIDERS (x, threads)")
    weekly_p.add_argument("--copy-image-to", default="", help="copy draft image to a public hosting folder before posting")
    weekly_p.add_argument(
        "--refresh-threads-token",
        action="store_true",
        help="refresh the configured long-lived Threads token",
    )
    weekly_p.add_argument("--persist-env", action="store_true", help="write refreshed Threads token values back to .env")
    weekly_p.add_argument("--env-path", default="", help="env file path used with --persist-env")

    media_p = sub.add_parser("media", help="curated photo showcase (Instagram) + video queue (YouTube)")
    media_p.add_argument("--draft", action="store_true", help="scan photo inbox and create an LLM-curated photo draft bundle")
    media_p.add_argument("--force", action="store_true", help="create a new photo draft even if one exists this run")
    media_p.add_argument("--inbox", default="", help="override the photo inbox folder")
    media_p.add_argument("--draft-video", action="store_true", help="queue the oldest new video that has a matching NAME.md description")
    media_p.add_argument("--draft-video-all", action="store_true", help="drain the video inbox, one draft per described video")
    media_p.add_argument("--video-inbox", default="", help="override the video inbox folder")
    media_p.add_argument("--post", metavar="DRAFT_DIR", help="post an approved draft directory (photo or video, auto-detected)")
    media_p.add_argument("--providers", default="", help="comma-separated providers overriding the draft kind's default provider list (instagram / youtube)")

    args = parser.parse_args()
    providers = tuple(p.strip().lower() for p in args.providers.split(",") if p.strip()) or None

    if args.mode == "weekly":
        if args.authorize_x:
            print(json.dumps(authorize_x(open_browser=args.open_browser), ensure_ascii=False, indent=2))
            return 0
        if args.refresh_threads_token:
            result = refresh_threads_token(persist_env=args.persist_env, env_path=args.env_path or None)
            print(json.dumps(result, ensure_ascii=False, indent=2))
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
        weekly_p.print_help()
        return 2

    if args.mode == "media":
        if args.draft:
            print(json.dumps(generate_photo_draft(inbox=args.inbox or None, force=args.force), ensure_ascii=False, indent=2))
            return 0
        if args.draft_video_all:
            print(json.dumps(generate_video_drafts(inbox=args.video_inbox or None), ensure_ascii=False, indent=2))
            return 0
        if args.draft_video:
            print(json.dumps(generate_video_draft(inbox=args.video_inbox or None), ensure_ascii=False, indent=2))
            return 0
        if args.post:
            draft_dir = Path(args.post).resolve()
            meta_path = draft_dir / "draft.json"
            kind = "video"
            if meta_path.exists():
                try:
                    kind = json.loads(meta_path.read_text(encoding="utf-8")).get("kind", "photo")
                except Exception:
                    kind = "photo"
            if kind == "video":
                print(json.dumps(post_video_draft(draft_dir, providers=providers), ensure_ascii=False, indent=2))
            else:
                print(json.dumps(post_photo_draft(draft_dir, providers=providers), ensure_ascii=False, indent=2))
            return 0
        media_p.print_help()
        return 2

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(_cmd())
