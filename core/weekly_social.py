"""
core/weekly_social.py

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
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import requests
from dotenv import load_dotenv
from openai import OpenAI

from core.log import get_logger
from core.memorize import AikoMemorize, USER_ID
from core.reflect import _generate_image, _load_soul

load_dotenv()
log = get_logger(__name__)

WORKSPACE_ROOT = Path(os.getenv("WORKSPACE_ROOT", "workspace")).resolve()
WEEKLY_SOCIAL_ROOT = Path(os.getenv("SOCIAL_ROOT", WORKSPACE_ROOT / "social" / "weekly")).resolve()
TIMEZONE_NAME = os.getenv("TIMEZONE", "UTC")

WEEKLY_ENABLED = os.getenv("WEEKLY_SOCIAL_ENABLED", "0").lower() in {"1", "true", "yes", "on"}
WEEKLY_AUTODRAFT = os.getenv("WEEKLY_SOCIAL_AUTODRAFT", "1").lower() in {"1", "true", "yes", "on"}
WEEKLY_AUTOPOST = os.getenv("WEEKLY_SOCIAL_AUTOPOST", "0").lower() in {"1", "true", "yes", "on"}
WEEKLY_POST_TIME = os.getenv("WEEKLY_SOCIAL_TIME", "18:00")
WEEKLY_POST_WEEKDAY = os.getenv("WEEKLY_SOCIAL_WEEKDAY", "sunday").lower().strip()
WEEKLY_PROVIDERS = tuple(
    p.strip().lower()
    for p in os.getenv("WEEKLY_SOCIAL_PROVIDERS", "x").split(",")
    if p.strip()
)

LLM_MODEL = os.getenv("REFLECT_MODEL", os.getenv("LLM_MODEL", "ministral"))
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8080/v1")
_LLM_CLIENT = OpenAI(base_url=LLM_BASE_URL, api_key="not-needed")

MAX_POST_CHARS = int(os.getenv("WEEKLY_SOCIAL_MAX_CHARS", "260"))

_WEEKDAYS = {
    "monday": 0, "mon": 0,
    "tuesday": 1, "tue": 1, "tues": 1,
    "wednesday": 2, "wed": 2,
    "thursday": 3, "thu": 3, "thur": 3, "thurs": 3,
    "friday": 4, "fri": 4,
    "saturday": 5, "sat": 5,
    "sunday": 6, "sun": 6,
}

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


def next_weekly_due(now: datetime | None = None, tz_name: str | None = None) -> datetime:
    """Calculate next configured weekly social run time."""
    tz = _timezone(tz_name)
    current = now.astimezone(tz) if now else datetime.now(tz)
    target_weekday = _WEEKDAYS.get(WEEKLY_POST_WEEKDAY, 6)
    hour_text, _, minute_text = WEEKLY_POST_TIME.partition(":")
    hour = int(hour_text or "18")
    minute = int(minute_text or "0")
    base = current.replace(hour=hour, minute=minute, second=0, microsecond=0)
    offset = (target_weekday - base.weekday()) % 7
    candidate = base + timedelta(days=offset)
    if candidate <= current:
        candidate += timedelta(days=7)
    return candidate


def _public_memory_rows(memorize: AikoMemorize, window: WeekWindow) -> list[dict[str, Any]]:
    rows = memorize.get_between(window.start, window.end)
    pinned = [r for r in rows if int(r.get("pinned") or 0) == 1]

    def is_weekly_source(row: dict[str, Any]) -> bool:
        text = (row.get("memory") or row.get("text") or "").strip()
        return text.startswith("Daily experience summary for ") or text.startswith("Day record for ")

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
    draft_dir = WEEKLY_SOCIAL_ROOT / window.label
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


def _post_x_via_aisa(text: str, image_path: Path | None) -> dict[str, Any]:
    api_key = os.getenv("AISA_API_KEY", "").strip()
    post_url = os.getenv("AISA_TWITTER_POST_URL", "").strip()
    if not api_key or not post_url:
        return {"ok": False, "provider": "x", "error": "AISA_API_KEY or AISA_TWITTER_POST_URL not set"}

    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"text": text}
    files = None
    if image_path and image_path.exists():
        mime = mimetypes.guess_type(str(image_path))[0] or "image/png"
        files = {"media": (image_path.name, image_path.open("rb"), mime)}
    try:
        if files:
            resp = requests.post(post_url, headers=headers, data=payload, files=files, timeout=120)
            files["media"][1].close()
        else:
            resp = requests.post(post_url, headers={**headers, "Content-Type": "application/json"}, json=payload, timeout=120)
        ok = 200 <= resp.status_code < 300
        return {"ok": ok, "provider": "x", "status_code": resp.status_code, "response": resp.text[:2000]}
    except Exception as e:
        return {"ok": False, "provider": "x", "error": str(e)}


def _post_threads(text: str, image_url: str | None) -> dict[str, Any]:
    token = os.getenv("THREADS_ACCESS_TOKEN", "").strip()
    user_id = os.getenv("THREADS_USER_ID", "").strip()
    base = os.getenv("THREADS_API_BASE", "https://graph.threads.net/v1.0").rstrip("/")
    if not token or not user_id:
        return {"ok": False, "provider": "threads", "error": "THREADS_ACCESS_TOKEN or THREADS_USER_ID not set"}

    create_url = f"{base}/{user_id}/threads"
    publish_url = f"{base}/{user_id}/threads_publish"
    params: dict[str, Any] = {"access_token": token, "text": text}
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
            results.append(_post_threads(text, public_image_url))
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
    if not WEEKLY_ENABLED:
        return {"success": False, "skipped": True, "reason": "WEEKLY_SOCIAL_ENABLED is off"}
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
    parser.add_argument("--providers", default="", help="comma-separated providers overriding WEEKLY_SOCIAL_PROVIDERS")
    parser.add_argument("--copy-image-to", default="", help="copy draft image to a public hosting folder before posting")
    args = parser.parse_args()

    providers = tuple(p.strip().lower() for p in args.providers.split(",") if p.strip()) or None

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
