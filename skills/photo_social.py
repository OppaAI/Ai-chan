"""
skills/photo_social.py

Curated media-showcase workflow for Aiko (Pipeline B — distinct from the
text+generated-image weekly postcard in skills/social.py).

Two lanes, both grounded in real media, not LLM-invented text/imagery:

  Photo lane (LLM-curated):
    1. scan the photo inbox (reuses toolkit/photography.py),
    2. caption each candidate via a vision model (grounded in actual pixels),
    3. ask Aiko to pick 1-3 items worth sharing and write a short caption,
    4. save a local review bundle, then optionally post.

  Video lane (no grading — queue, don't curate):
    1. scan the video inbox,
    2. queue the oldest not-yet-drafted video one at a time,
    3. give you an editable caption.txt to fill in by hand,
    4. save a local review bundle, then optionally post.
  No vision/LLM judgment is applied to video — every file dropped in the
  folder is fair game; this just prevents re-posting the same file twice.

Posting is opt-in for both lanes. By default this only creates drafts.

Providers:
  - pixelfed: Mastodon-compatible REST API (multipart media upload +
    status create). Works for images and video without changes.
  - instagram: Graph API container-create -> publish flow. Requires an
    Instagram Business/Creator account linked to a Facebook Page, a Meta
    app, and a PUBLICLY reachable URL for the media (no direct file
    upload). Images reuse the imgbb hosting helper from social.py. Video
    has no equivalent free host built in — you must serve the draft's
    media/ folder over HTTPS yourself and set IG_VIDEO_PUBLIC_BASE_URL.
    Video (Reels) additionally requires polling the container's
    status_code until FINISHED before publish, since processing isn't
    instant like images.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import requests
from openai import OpenAI

from system.log import get_logger
from system.userspace import user_workspace_root
from memory.reflect import _load_soul
from toolkit.common import workspace_root
from toolkit.photography import scan_photo_workspace, scan_video_workspace, VIDEO_EXTENSIONS
from skills.social import _upload_to_imgbb, _int_env  # reuse existing image-hosting helper

log = get_logger(__name__)


def photo_social_root() -> Path:
    """Resolve the active user photo-social output root lazily.

    Defaults to <USER_STATE_ROOT>/<user_id>/workspace/social/photo.
    """
    override = os.getenv("PHOTO_SOCIAL_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return (user_workspace_root() / "social" / "photo").resolve()


PHOTO_SOCIAL_AUTODRAFT = os.getenv("PHOTO_SOCIAL_AUTODRAFT", "0").lower() in {"1", "true", "yes", "on"}
PHOTO_SOCIAL_AUTOPOST = os.getenv("PHOTO_SOCIAL_AUTOPOST", "0").lower() in {"1", "true", "yes", "on"}
PHOTO_SOCIAL_PROVIDERS = tuple(
    p.strip().lower()
    for p in os.getenv("PHOTO_SOCIAL_PROVIDERS", "pixelfed").split(",")
    if p.strip()
)
PHOTO_SOCIAL_INBOX = os.getenv("PHOTO_SOCIAL_INBOX", "photos/inbox")
PHOTO_SOCIAL_MAX_ITEMS = _int_env("PHOTO_SOCIAL_MAX_ITEMS", 3)
MAX_CAPTION_CHARS = _int_env("PHOTO_SOCIAL_MAX_CHARS", 260)

VIDEO_SOCIAL_INBOX = os.getenv("VIDEO_SOCIAL_INBOX", "videos")
VIDEO_SOCIAL_PROVIDERS = tuple(
    p.strip().lower()
    for p in os.getenv("VIDEO_SOCIAL_PROVIDERS", "pixelfed,instagram").split(",")
    if p.strip()
)

# Vision model (captioning) — separate client/model from the text LLM,
# since captioning needs actual image understanding (e.g. MiniCPM-V), not
# the text-only Ministral endpoint used elsewhere.
VISION_MODEL = os.getenv("VISION_MODEL", os.getenv("REFLECT_VISION_MODEL", "minicpm-v"))
VISION_BASE_URL = os.getenv("VISION_BASE_URL", os.getenv("LLM_BASE_URL", "http://localhost:8080/v1"))
_VISION_CLIENT = OpenAI(base_url=VISION_BASE_URL, api_key="not-needed")

LLM_MODEL = os.getenv("REFLECT_MODEL", os.getenv("LLM_MODEL", "ministral"))
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8080/v1")
_LLM_CLIENT = OpenAI(base_url=LLM_BASE_URL, api_key="not-needed")

_CAPTION_PROMPT = (
    "Describe this image in one plain, factual sentence. No hashtags, no "
    "hype, no marketing language. If it looks private, sensitive, or "
    "identifies a specific real person's face clearly, start your reply "
    "with 'PRIVATE:' instead of a description."
)

_SELECT_SYSTEM = """\
You are Aiko choosing which recent photo(s)/video(s) are worth sharing publicly.

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

_SELECT_USER = """\
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
    import base64
    mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def _caption_media(path: Path) -> MediaCandidate:
    """Caption one image via the vision model. Only reachable for the photo
    lane — scan_photo_workspace only ever returns IMAGE_EXTENSIONS, so
    video never lands here. The video lane (below) is deliberately
    caption-free by design; see generate_video_draft."""
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
    parses LLM output elsewhere in this codebase. Note the tool itself caps
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
        log.warning("Failed to parse photo_social JSON: %r", cleaned[:300])
        return {}


def _llm_select(candidates: list[MediaCandidate]) -> list[MediaSelection]:
    public_candidates = [c for c in candidates if not c.private]
    if not public_candidates:
        return []

    items_block = "\n".join(f"- {c.path.name}: {c.raw_caption}" for c in public_candidates)
    system = f"{_load_soul()}\n\n" + _SELECT_SYSTEM.format(
        max_items=PHOTO_SOCIAL_MAX_ITEMS, max_chars=MAX_CAPTION_CHARS,
    )
    user = _SELECT_USER.format(items=items_block)

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
    selections = _llm_select(candidates)

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


def _read_draft(draft_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    meta_path = draft_dir / "draft.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    selections = meta.get("selections", [])
    # Video drafts ship an empty caption.txt for hand-editing before --post
    # (no LLM caption exists for them). If present, it always wins over
    # whatever draft.json still has cached, so edits are picked up live.
    caption_override = draft_dir / "caption.txt"
    if len(selections) == 1 and caption_override.exists():
        selections[0]["caption"] = caption_override.read_text(encoding="utf-8").strip()
    return selections, meta


# ── video lane: no grading, just a de-duplicated posting queue ──────────────

def _video_ledger_path() -> Path:
    return photo_social_root() / "_video_ledger.json"


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
    """Mirrors _list_candidates but for scan_video_workspace's output shape.
    Same upstream caveat: the tool's own "files" preview is hardcapped at 50
    regardless of the limit passed here."""
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


def generate_video_draft(*, inbox: str | None = None) -> dict[str, Any]:
    """Queue the oldest not-yet-drafted video from the videos inbox.

    Deliberately does NOT run any vision/LLM selection or captioning —
    "no grading" means every file dropped in the folder is a candidate.
    This only decides ORDER (oldest first) and prevents re-drafting the
    same file twice via a small local ledger. Fill in draft_dir/caption.txt
    by hand before running --post.
    """
    inbox_path = inbox or VIDEO_SOCIAL_INBOX
    candidates = _list_video_candidates(inbox_path, limit=50)
    if not candidates:
        return {"success": True, "skipped": True, "reason": "empty_inbox", "inbox": inbox_path}

    ledger = _load_video_ledger()
    unprocessed = [p for p in candidates if str(p) not in ledger]
    if not unprocessed:
        return {"success": True, "skipped": True, "reason": "no_new_videos", "inbox": inbox_path}

    unprocessed.sort(key=lambda p: p.stat().st_mtime)
    video_path = unprocessed[0]

    label = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    draft_dir = photo_social_root() / f"video-{label}"
    media_dir = draft_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)

    dest = media_dir / video_path.name
    dest.write_bytes(video_path.read_bytes())

    (draft_dir / "caption.txt").write_text("", encoding="utf-8")
    (draft_dir / "review.md").write_text(
        f"# Video Social Draft \u2014 {label}\n\n"
        f"Source: {video_path.name}\n\n"
        f"## Caption\n\nEdit caption.txt with the post caption before posting "
        f"(this file is read fresh at --post time, so edits take effect "
        f"without regenerating the draft).\n\n"
        f"## Review checklist\n\n"
        f"- [ ] Public-safe (no identifiable people/private locations)\n"
        f"- [ ] caption.txt filled in\n"
        f"- [ ] No request for replies/likes/follows\n"
        f"- [ ] Approved to post\n",
        encoding="utf-8",
    )

    selections = [{"filename": video_path.name, "caption": "", "media_path": str(dest)}]
    meta = {
        "success": True,
        "label": label,
        "draft_dir": str(draft_dir),
        "kind": "video",
        "source": str(video_path),
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
    """Drain the videos inbox, creating one draft per not-yet-drafted video."""
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


# ── Pixelfed (Mastodon-compatible API) ───────────────────────────────────────

def _pixelfed_config() -> tuple[str, str]:
    token = os.getenv("PIXELFED_ACCESS_TOKEN", "").strip()
    base = os.getenv("PIXELFED_API_BASE", "").strip().rstrip("/")
    return token, base


def _post_pixelfed(selections: list[dict[str, Any]]) -> dict[str, Any]:
    token, base = _pixelfed_config()
    if not token or not base:
        return {"ok": False, "provider": "pixelfed", "error": "PIXELFED_ACCESS_TOKEN or PIXELFED_API_BASE not set"}
    if not selections:
        return {"ok": False, "provider": "pixelfed", "error": "no selections to post"}

    timeout = _int_env("PIXELFED_TIMEOUT", 30)
    headers = {"Authorization": f"Bearer {token}"}
    media_ids = []
    for sel in selections:
        media_path = Path(sel["media_path"])
        if not media_path.exists():
            continue
        try:
            mime = mimetypes.guess_type(str(media_path))[0] or "image/jpeg"
            with open(media_path, "rb") as f:
                resp = requests.post(
                    f"{base}/api/v2/media",
                    headers=headers,
                    files={"file": (media_path.name, f, mime)},
                    data={"description": sel.get("caption", "")[:420]},
                    timeout=timeout,
                )
            if 200 <= resp.status_code < 300:
                media_ids.append(resp.json().get("id"))
            else:
                log.warning("Pixelfed media upload failed: %s %s", resp.status_code, resp.text[:300])
        except Exception as e:
            log.warning("Pixelfed media upload failed for %s: %s", media_path, e)

    if not media_ids:
        return {"ok": False, "provider": "pixelfed", "error": "all media uploads failed"}

    status_text = "\n\n".join(sel.get("caption", "") for sel in selections)[:500]
    form: list[tuple[str, str]] = [("status", status_text)]
    for mid in media_ids:
        if mid:
            form.append(("media_ids[]", mid))

    try:
        resp = requests.post(
            f"{base}/api/v1/statuses",
            headers={**headers, "Idempotency-Key": os.urandom(8).hex()},
            data=form,
            timeout=timeout,
        )
        ok = 200 <= resp.status_code < 300
        return {"ok": ok, "provider": "pixelfed", "status_code": resp.status_code, "response": resp.text[:2000]}
    except Exception as e:
        return {"ok": False, "provider": "pixelfed", "error": str(e)}


# ── Instagram (Graph API) ────────────────────────────────────────────────────
# Requires an IG Business/Creator account linked to a Facebook Page, and a
# publicly reachable URL for the media (no direct file upload).

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


def _instagram_video_public_url(media_path: Path) -> str | None:
    """No auto-hosting for video (imgbb is image-only). You must serve
    photo_social_root()'s draft media/ folders over HTTPS yourself (a
    static route, or e.g. your existing Tailscale Funnel setup) and point
    IG_VIDEO_PUBLIC_BASE_URL at wherever that ends up publicly reachable —
    Meta's servers fetch the video FROM this URL, there's no push upload."""
    base = os.getenv("IG_VIDEO_PUBLIC_BASE_URL", "").strip().rstrip("/")
    if not base:
        return None
    try:
        rel = media_path.relative_to(photo_social_root())
    except ValueError:
        rel = media_path.name
    return f"{base}/{rel}"


def _post_instagram_video(sel: dict[str, Any]) -> dict[str, Any]:
    """Publishes as a Reel (media_type=REELS) — this is the current Graph
    API path for video, not the older direct feed-video type. Reels-tab
    eligibility additionally requires 9:16, 5-90s duration, H.264/HEVC;
    outside that range it still publishes but only as a regular video post,
    not in the Reels tab. Not validated here — check your file before
    posting if that distinction matters to you."""
    token, ig_user_id, base = _instagram_config()
    if not token or not ig_user_id:
        return {"ok": False, "provider": "instagram", "error": "IG_ACCESS_TOKEN or IG_BUSINESS_ACCOUNT_ID not set"}

    media_path = Path(sel["media_path"])
    if not media_path.exists():
        return {"ok": False, "provider": "instagram", "error": f"media not found: {media_path}"}

    video_url = _instagram_video_public_url(media_path)
    if not video_url:
        return {
            "ok": False, "provider": "instagram",
            "error": (
                "IG_VIDEO_PUBLIC_BASE_URL not set. Instagram's Graph API fetches "
                "video from a public URL it reaches itself — there is no direct "
                "upload. Serve the photo_social draft media/ folder over HTTPS "
                "and set IG_VIDEO_PUBLIC_BASE_URL to that base."
            ),
        }

    timeout = _int_env("IG_TIMEOUT", 60)
    try:
        create = requests.post(
            f"{base}/{ig_user_id}/media",
            data={
                "media_type": "REELS",
                "video_url": video_url,
                "caption": sel.get("caption", "")[:2200],
                "access_token": token,
            },
            timeout=timeout,
        )
        if not (200 <= create.status_code < 300):
            return {"ok": False, "provider": "instagram", "stage": "create", "status_code": create.status_code, "response": create.text[:2000]}
        creation_id = create.json().get("id")
        if not creation_id:
            return {"ok": False, "provider": "instagram", "stage": "create", "error": "missing creation id", "response": create.text[:2000]}

        # Video processing isn't instant like images — poll status_code
        # instead of a fixed sleep. Meta recommends checking every ~5-10s;
        # most containers finish in 30s-2min but large files can take longer.
        poll_interval = float(os.getenv("IG_VIDEO_POLL_SECONDS", "10"))
        max_polls = _int_env("IG_VIDEO_MAX_POLLS", 30)  # ~5 min at default interval
        status_code = None
        for _ in range(max_polls):
            time.sleep(poll_interval)
            status_resp = requests.get(
                f"{base}/{creation_id}",
                params={"fields": "status_code", "access_token": token},
                timeout=timeout,
            )
            if status_resp.status_code >= 300:
                break
            status_code = status_resp.json().get("status_code")
            if status_code == "FINISHED":
                break
            if status_code == "ERROR":
                return {"ok": False, "provider": "instagram", "stage": "processing", "status_code": status_code, "creation_id": creation_id}

        if status_code != "FINISHED":
            return {
                "ok": False, "provider": "instagram", "stage": "processing_timeout",
                "last_status": status_code, "creation_id": creation_id,
                "note": "container never reached FINISHED within the poll budget; raise IG_VIDEO_MAX_POLLS or check the file",
            }

        publish = requests.post(
            f"{base}/{ig_user_id}/media_publish",
            data={"creation_id": creation_id, "access_token": token},
            timeout=timeout,
        )
        ok = 200 <= publish.status_code < 300
        return {
            "ok": ok, "provider": "instagram", "status_code": publish.status_code,
            "creation_id": creation_id, "response": publish.text[:2000],
        }
    except Exception as e:
        return {"ok": False, "provider": "instagram", "error": str(e)}


def _post_instagram(selections: list[dict[str, Any]]) -> dict[str, Any]:
    """Posts the FIRST selection only. Carousel (multi-item) posting needs
    child containers and is not implemented — extend here if needed.
    Dispatches to the image or video (Reels) flow based on file extension."""
    if not selections:
        return {"ok": False, "provider": "instagram", "error": "no selections to post"}
    sel = selections[0]
    media_path = Path(sel["media_path"])
    if media_path.suffix.lower() in VIDEO_EXTENSIONS:
        return _post_instagram_video(sel)
    return _post_instagram_image(sel)


_PROVIDERS: dict[str, Callable[[list[dict[str, Any]]], dict[str, Any]]] = {
    "pixelfed": _post_pixelfed,
    "instagram": _post_instagram,
}


def post_photo_draft(draft_dir: str | Path, providers: tuple[str, ...] | None = None) -> dict[str, Any]:
    path = Path(draft_dir).resolve()
    selections, meta = _read_draft(path)
    providers = providers or PHOTO_SOCIAL_PROVIDERS

    results = []
    for provider in providers:
        handler = _PROVIDERS.get(provider)
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


VIDEO_SOCIAL_AUTODRAFT = os.getenv("VIDEO_SOCIAL_AUTODRAFT", "0").lower() in {"1", "true", "yes", "on"}
VIDEO_SOCIAL_AUTOPOST = os.getenv("VIDEO_SOCIAL_AUTOPOST", "0").lower() in {"1", "true", "yes", "on"}


def run_scheduled_video_social() -> dict[str, Any]:
    """Scheduler entrypoint for the video lane: draft (queue) by default,
    post only when explicitly enabled. VIDEO_SOCIAL_AUTOPOST should stay
    off unless caption.txt is filled in some other way before this runs —
    there's no LLM caption to fall back on for video."""
    if not VIDEO_SOCIAL_AUTODRAFT:
        return {"success": False, "skipped": True, "reason": "VIDEO_SOCIAL_AUTODRAFT is off"}
    draft = generate_video_draft()
    if VIDEO_SOCIAL_AUTOPOST and draft.get("success") and not draft.get("skipped"):
        draft["post"] = post_photo_draft(draft["draft_dir"])
    return draft


def _cmd() -> int:
    parser = argparse.ArgumentParser(description="Aiko curated media showcase (Pixelfed/Instagram)")
    parser.add_argument("--draft", action="store_true", help="scan photo inbox and create an LLM-curated media draft bundle")
    parser.add_argument("--force", action="store_true", help="create a new draft even if one exists this run")
    parser.add_argument("--inbox", default="", help="override the photo inbox folder")
    parser.add_argument("--draft-video", action="store_true", help="queue the oldest new video from the video inbox (no grading)")
    parser.add_argument("--draft-video-all", action="store_true", help="drain the video inbox, one draft per new video")
    parser.add_argument("--video-inbox", default="", help="override the video inbox folder")
    parser.add_argument("--post", metavar="DRAFT_DIR", help="post an approved draft directory (photo or video)")
    parser.add_argument("--providers", default="", help="comma-separated providers overriding the draft kind's default provider list")
    args = parser.parse_args()

    providers = tuple(p.strip().lower() for p in args.providers.split(",") if p.strip()) or None

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
        print(json.dumps(post_photo_draft(Path(args.post).resolve(), providers=providers), ensure_ascii=False, indent=2))
        return 0
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(_cmd())
