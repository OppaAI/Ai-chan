"""
toolkit/social.py

Aiko's social publishing workflows, combined into one module. Four lanes:

  Lane A1 — Weekly Patreon dev-post syndication (scheduled):
    fetch the newest Patreon creator post, save a review bundle, repost the
    full body to a native Hugo/GitHub Pages dev blog, then send a
    280-character teaser fanout through the social MCP post_social wrapper.
    Lane A2 remains manual and is intentionally not implemented here.

  Lane B — Curated photo showcase:
    scan/caption/select real photos locally, save a review bundle, then post
    approved photos through MCP to Pixelfed instances.

  Lane C — YouTube video queue:
    unchanged described-video queue; approved posts go through MCP YouTube.

  Lane D — Nightly tech job-post draft:
    run the graph job-hunt playbook against configured RSS feeds only,
    and save one Threads teaser-list draft for tech jobs available today.

Posting remains human-review gated via draft.json["human_approved"] for every
lane. One-way social posting belongs in the MCP server; two-way conversational
adapters are limited to Telegram, Discord, Matrix, and Slack. Nightly
reflection publishing remains native Hugo/GitHub, not MCP.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import requests
from openai import OpenAI

from system.bioclock import get_timezone
from system.log import get_logger
from cognition.memory.memorize import AikoMemorize
from system.userspace import user_workspace_root
from agentic.mcp_client.bridge import bootstrap_mcp
from cognition.consolidate.reflect import _load_soul

from agentic.toolkit.common import workspace_root
from agentic.toolkit.photography import scan_photo_workspace, scan_video_workspace

log = get_logger(__name__)

SOCIAL_PERSONA_PATH = os.path.expanduser(os.getenv("SOCIAL_PERSONA_PATH", "persona/SOCIAL.md"))


def _load_social_persona() -> str:
    try:
        with open(SOCIAL_PERSONA_PATH, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        log.warning("SOCIAL.md not found at %s — no social persona appended.", SOCIAL_PERSONA_PATH)
        return ""


# ── shared paths ──────────────────────────────────────────────────────────────

def weekly_social_root() -> Path:
    """Resolve the active user weekly social output root lazily.

    Defaults to <USER_SPACE_ROOT>/<user_id>/workspace/social/weekly. Holds
    draft bundles for weekly Patreon dev-post syndication bundles, including full posts,
    teaser text, Hugo markdown, and downloaded teaser images.
    """
    override = os.getenv("SOCIAL_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return (user_workspace_root() / "social" / "weekly").resolve()


def photo_social_root() -> Path:
    """Resolve the active user photo-social output root lazily.

    Defaults to <USER_SPACE_ROOT>/<user_id>/workspace/social/photo.
    """
    override = os.getenv("PHOTO_SOCIAL_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return (user_workspace_root() / "social" / "photo").resolve()


def job_post_social_root() -> Path:
    """Resolve the active user job-post draft output root lazily.

    Defaults to <USER_SPACE_ROOT>/<user_id>/workspace/social/job_posts.
    Holds daily Meta Threads job-post drafts for human review.
    """
    override = os.getenv("JOB_POST_SOCIAL_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    # Use per-user state dir directly — do not respect WORKSPACE_ROOT which
    # is a process-global override for generic workspace tools and would
    # collapse all users into a single shared jetson/workspace path.
    from system.userspace import user_state_dir

    return (user_state_dir() / "workspace" / "social" / "job_posts").resolve()


def _latest_approved_job_post_draft() -> Path | None:
    """Return the most recently created human-approved job-post draft dir, or None.

    Scans <job_post_social_root>/<date>/<category>/<slug>/draft.json,
    <job_post_social_root>/<date>/<category>/draft.json, and
    <job_post_social_root>/<date>/draft.json (when category is empty)
    for the dir whose draft.json has human_approved=True and the
    newest created_at. Used by post_job_post_social when the model
    doesn't supply a draft_dir.
    """
    root = job_post_social_root()
    best: tuple[float, Path] | None = None
    for meta_path in list(root.glob("*/*/*/draft.json")) + list(root.glob("*/*/draft.json")) + list(root.glob("*/draft.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if meta.get("human_approved") is not True:
            continue
        created = meta.get("created_at") or ""
        key = 0.0
        try:
            dt = datetime.fromisoformat(str(created).replace("Z", "+00:00"))
            key = dt.timestamp()
        except (ValueError, TypeError):
            key = float(meta_path.stat().st_mtime)
        if best is None or key > best[0]:
            best = (key, meta_path.parent)
    return best[1] if best else None


def _latest_approved_video_draft() -> Path | None:
    """Return the most recently created human-approved video draft dir, or None.

    Scans <video_social_root>/<timestamp>/draft.json for the dir whose
    draft.json has human_approved=True and the newest created_at.
    """
    root = video_social_root()
    best: tuple[float, Path] | None = None
    for meta_path in root.glob("*/draft.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if meta.get("human_approved") is not True:
            continue
        created = meta.get("created_at") or ""
        key = 0.0
        try:
            dt = datetime.fromisoformat(str(created).replace("Z", "+00:00"))
            key = dt.timestamp()
        except (ValueError, TypeError):
            key = float(meta_path.stat().st_mtime)
        if best is None or key > best[0]:
            best = (key, meta_path.parent)
    return best[1] if best else None


def _latest_approved_photo_draft() -> Path | None:
    """Return the most recently created human-approved photo draft dir, or None.

    Scans <photo_social_root>/<timestamp>/draft.json for the dir whose
    draft.json has human_approved=True and the newest created_at.
    """
    root = photo_social_root()
    best: tuple[float, Path] | None = None
    for meta_path in root.glob("*/draft.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if meta.get("human_approved") is not True:
            continue
        created = meta.get("created_at") or ""
        key = 0.0
        try:
            dt = datetime.fromisoformat(str(created).replace("Z", "+00:00"))
            key = dt.timestamp()
        except (ValueError, TypeError):
            key = float(meta_path.stat().st_mtime)
        if best is None or key > best[0]:
            best = (key, meta_path.parent)
    return best[1] if best else None


def video_social_root() -> Path:
    """Resolve the active user video-social output root lazily.

    Defaults to <USER_SPACE_ROOT>/<user_id>/workspace/videos.
    """
    override = os.getenv("VIDEO_SOCIAL_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return (user_workspace_root() / "videos").resolve()


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


class SocialApprovalError(Exception):
    """Raised when something tries to post a draft that has not been through
    human review. Both the scheduler's auto-post path and the agent-tool
    post_*_social wrappers go through _require_approved below, so approval
    is enforced the same way regardless of which path triggered the post —
    WEEKLY_SOCIAL_AUTOPOST=1 alone is not enough to actually post; a person
    still has to set draft.json["human_approved"] = true first."""


def _require_approved(draft_dir: Path) -> None:
    """The load-bearing safety gate for every post path (scheduler and
    agent-tool alike). Raises SocialApprovalError unless draft.json exists
    and its "human_approved" key is exactly True. This is deliberately the
    only place that decides whether a post is allowed to go out — no tool
    schema field, env var, or model-supplied boolean can substitute for it,
    since both the scheduler config and the model's own tool-call arguments
    are things this function does not trust on their own.
    """
    meta_path = draft_dir / "draft.json"
    if not meta_path.exists():
        raise SocialApprovalError(f"no draft found at {draft_dir}")
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise SocialApprovalError(f"could not read draft.json at {draft_dir}: {e}")
    if meta.get("human_approved") is not True:
        raise SocialApprovalError(
            "this draft has not been approved by a person yet — review it "
            "before posting"
        )


# Shared text LLM (used for both the weekly memory-selection prompt and the
# photo caption-selection prompt).
LLM_MODEL = os.getenv("REFLECT_MODEL", os.getenv("LLM_MODEL", "ministral"))
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8080/v1")
_LLM_CLIENT = OpenAI(base_url=LLM_BASE_URL, api_key="not-needed")


# Lane A1 — Patreon dev-post syndication
A1_FULL_PROVIDERS = tuple(p.strip().lower() for p in os.getenv("A1_FULL_PROVIDERS", "").split(",") if p.strip())
A1_TEASER_PROVIDERS = tuple(p.strip().lower() for p in os.getenv("A1_TEASER_PROVIDERS", "x,bluesky,mastodon,discord,threads").split(",") if p.strip())
A1_TEASER_MAX_CHARS = _int_env("A1_TEASER_MAX_CHARS", 280)
A1_HUGO_REPO = os.getenv("AIKO_DEV_GITHUB_REPO", os.getenv("GITHUB_REPO", ""))
A1_HUGO_BRANCH = os.getenv("AIKO_DEV_GITHUB_BRANCH", os.getenv("GITHUB_BRANCH", "main"))
A1_HUGO_CONTENT_PATH = os.getenv("AIKO_DEV_HUGO_CONTENT_PATH", "content/posts")
A1_HUGO_IMAGES_PATH = os.getenv("AIKO_DEV_HUGO_IMAGES_PATH", os.getenv("HUGO_IMAGES_PATH", "static/images"))
WEEKLY_AUTODRAFT = os.getenv("WEEKLY_SOCIAL_AUTODRAFT", "1").lower() in {"1", "true", "yes", "on"}
WEEKLY_AUTOPOST = os.getenv("WEEKLY_SOCIAL_AUTOPOST", "0").lower() in {"1", "true", "yes", "on"}


def _call_social_mcp(tool: str, **kwargs: Any) -> dict[str, Any]:
    try:
        from agentic.mcp_client.social_bridge import _call_mcp
        return _call_mcp(tool, **kwargs)
    except Exception as e:
        return {"ok": False, "tool": tool, "error": str(e)}


def _fetch_latest_patreon_post() -> dict[str, Any] | None:
    """Fetch the newest Patreon creator post for Lane A1.

    Prefer PATREON_LATEST_POST_URL for custom/private feeds; otherwise use the
    Patreon campaign posts API with PATREON_CREATOR_ACCESS_TOKEN and
    PATREON_CAMPAIGN_ID. The returned dict is normalized for draft writing.
    """
    token = os.getenv("PATREON_CREATOR_ACCESS_TOKEN", "").strip()
    custom_url = os.getenv("PATREON_LATEST_POST_URL", "").strip()
    campaign_id = os.getenv("PATREON_CAMPAIGN_ID", "").strip()
    if not token:
        return None
    headers = {"Authorization": f"Bearer {token}"}
    url = custom_url or f"https://www.patreon.com/api/oauth2/v2/campaigns/{campaign_id}/posts"
    params = None if custom_url else {
        "fields[post]": "title,content,published_at,url,embed_data,embed_url",
        "sort": "-published_at",
        "page[count]": "50",
    }
    try:
        all_posts = []
        next_url = url
        current_params = params
        while next_url:
            resp = requests.get(next_url, headers=headers, params=current_params, timeout=30)
            resp.raise_for_status()
            payload = resp.json()
            data = payload.get("data")
            if data and isinstance(data, list):
                all_posts.extend(data)
            
            next_url = payload.get("links", {}).get("next")
            current_params = None
            
        if not all_posts:
            all_posts = [payload]
            
        def get_pub_date(p):
            attrs = p.get("attributes", p)
            return attrs.get("published_at") or attrs.get("created_at") or ""
            
        all_posts.sort(key=get_pub_date, reverse=True)
        item = all_posts[0]
        
        attrs = item.get("attributes", item)
        title = attrs.get("title") or "Aiko dev update"
        body = attrs.get("content") or attrs.get("body") or attrs.get("teaser_text") or ""
        image_url = ""
        embed = attrs.get("embed_data") or attrs.get("embed") or {}
        if isinstance(embed, dict):
            image_url = embed.get("thumbnail_url") or embed.get("url") or ""
        # collect all image urls: embed thumbnail + images from included + <img> in body
        image_urls: list[str] = []
        if image_url:
            image_urls.append(image_url)
        # included images / post_files from Patreon API (when include=images,post_file)
        try:
            included = payload.get("included") or []
            # map id -> url for images/post_files if item has relationships
            rel_data = []
            rels = item.get("relationships") or {}
            for key in ("images", "post_file", "images_media"):
                data = (rels.get(key) or {}).get("data")
                if isinstance(data, list):
                    rel_data.extend(data)
                elif isinstance(data, dict):
                    rel_data.append(data)
            rel_ids = {str(x.get("id")) for x in rel_data if x.get("id")}
            for inc in included:
                if str(inc.get("id")) in rel_ids or not rel_ids:
                    inc_type = str(inc.get("type") or "")
                    inc_attrs = inc.get("attributes") or {}
                    if inc_type in {"images", "image", "post_file", "media"}:
                        for k in ("url", "image_url", "download_url", "full", "large"):
                            v = inc_attrs.get(k)
                            if isinstance(v, str) and v.startswith("http") and v not in image_urls:
                                image_urls.append(v)
                                break
                        # fallback: any http string in attributes
                        if len(image_urls) == 0 or image_urls[-1] not in str(inc_attrs):
                            for v in inc_attrs.values():
                                if isinstance(v, str) and v.startswith("http") and v not in image_urls and any(v.lower().endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".webp", ".gif")):
                                    image_urls.append(v)
        except Exception:
            pass
        # fallback: parse <img src> from body HTML
        for m in re.findall(r'<img[^>]+src=["\']([^"\']+)["\']', body or "", re.IGNORECASE):
            if m.startswith("http") and m not in image_urls:
                image_urls.append(m)
        # dedupe, keep order
        image_urls = list(dict.fromkeys(image_urls))
        return {
            "id": item.get("id") or attrs.get("id") or title,
            "title": title,
            "body": body,
            "url": attrs.get("url") or attrs.get("patreon_url") or "",
            "published_at": attrs.get("published_at") or attrs.get("created_at") or datetime.now(timezone.utc).isoformat(),
            "image_url": image_url,
            "image_urls": image_urls,
        }
    except Exception as e:
        log.error("Lane A1 Patreon fetch failed: %s", e)
        return None


def _teaser_for_post(post: dict[str, Any]) -> str:
    text = re.sub(r"<[^>]+>", " ", post.get("body", ""))
    text = re.sub(r"\s+", " ", text).strip()
    title = post.get("title") or "Aiko dev update"
    teaser = f"{title}: {text}" if text else title
    if post.get("url"):
        reserve = len(post["url"]) + 1
        teaser = teaser[: max(0, A1_TEASER_MAX_CHARS - reserve)].rstrip() + " " + post["url"]
    return teaser[:A1_TEASER_MAX_CHARS].rstrip()


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:80] or "aiko-dev-update"


def _build_a1_hugo_post(post: dict[str, Any], teaser: str) -> tuple[str, str]:
    published = str(post.get("published_at") or datetime.now(timezone.utc).isoformat())
    date_part = published[:10]
    slug = f"{date_part}-{_slugify(post.get('title', 'aiko-dev-update'))}"
    title = str(post.get("title") or "Aiko dev update").replace('"', "'")
    body = post.get("body") or ""
    source = f"\n\nOriginal Patreon post: {post.get('url')}" if post.get("url") else ""
    md = (
        "---\n"
        f"title: \"{title}\"\n"
        f"date: {published}\n"
        "draft: false\n"
        "tags:\n  - \"aiko-dev\"\n  - \"patreon\"\n"
        f"summary: \"{teaser.replace(chr(34), chr(39))}\"\n"
        "---\n\n"
        f"{body}{source}\n"
    )
    return slug, md


def _push_a1_hugo_post(slug: str, content: str) -> dict[str, Any]:
    token = os.getenv("AIKO_DEV_GITHUB_TOKEN", os.getenv("GITHUB_TOKEN", ""))
    repo = A1_HUGO_REPO
    if not token or not repo:
        return {"ok": False, "provider": "github_hugo", "error": "AIKO_DEV_GITHUB_TOKEN/GITHUB_TOKEN or AIKO_DEV_GITHUB_REPO/GITHUB_REPO not set"}
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    path = f"{A1_HUGO_CONTENT_PATH.rstrip('/')}/{slug}.md"
    base = f"https://api.github.com/repos/{repo}/contents/{path}"
    try:
        existing = requests.get(base, headers=headers, params={"ref": A1_HUGO_BRANCH}, timeout=15)
        sha = existing.json().get("sha") if existing.status_code == 200 else None
        payload = {"message": f"feat(aiko-dev): syndicate {slug}", "content": base64.b64encode(content.encode()).decode(), "branch": A1_HUGO_BRANCH}
        if sha:
            payload["sha"] = sha
        resp = requests.put(base, headers=headers, json=payload, timeout=30)
        return {"ok": 200 <= resp.status_code < 300, "provider": "github_hugo", "path": path, "status_code": resp.status_code, "response": resp.text[:1000]}
    except Exception as e:
        return {"ok": False, "provider": "github_hugo", "error": str(e)}



def _download_a1_image(post: dict[str, Any], draft_dir: Path) -> Path | None:
    """Download a Patreon teaser/embed image into the draft bundle when present."""
    image_url = str(post.get("image_url") or "").strip()
    if not image_url:
        return None
    try:
        resp = requests.get(image_url, timeout=30)
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")
        ext = mimetypes.guess_extension(content_type.split(";", 1)[0].strip()) or Path(image_url.split("?", 1)[0]).suffix or ".png"
        if ext.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
            log.warning("Lane A1 Patreon image has unsupported type %r; skipping download", content_type)
            return None
        image_path = draft_dir / f"image{ext}"
        image_path.write_bytes(resp.content)
        return image_path
    except Exception as e:
        log.warning("Lane A1 Patreon image download failed: %s", e)
        return None


def _download_a1_images(post: dict[str, Any], draft_dir: Path) -> list[Path]:
    """Download all Patreon images into draft_dir/images/ (images-only lane).

    Videos are intentionally ignored — use YouTube links per owner instruction.
    """
    urls: list[str] = []
    raw = post.get("image_urls")
    if isinstance(raw, list):
        urls.extend([str(u).strip() for u in raw if str(u).strip().startswith("http")])
    fallback = str(post.get("image_url") or "").strip()
    if fallback and fallback not in urls:
        urls.insert(0, fallback)
    urls = list(dict.fromkeys(urls))
    if not urls:
        return []
    out: list[Path] = []
    images_dir = draft_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    for idx, url in enumerate(urls):
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")
            # skip videos
            if "video" in content_type.lower():
                log.warning("Lane A1 skipping video content_type %r for %s", content_type, url[:120])
                continue
            ext = mimetypes.guess_extension(content_type.split(";", 1)[0].strip()) or Path(url.split("?", 1)[0]).suffix or ".png"
            ext = ext.lower()
            if ext not in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
                # guess from url suffix if mime unknown but url looks like image
                url_ext = Path(url.split("?", 1)[0]).suffix.lower()
                if url_ext in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
                    ext = url_ext
                else:
                    log.warning("Lane A1 image has unsupported type %r; skipping %s", content_type, url[:120])
                    continue
            p = images_dir / f"image_{idx:02d}{ext}"
            p.write_bytes(resp.content)
            out.append(p)
        except Exception as e:
            log.warning("Lane A1 image download failed for %s: %s", url[:120], e)
            continue
    return out


def _push_a1_hugo_images(slug: str, image_paths: list[Path]) -> list[dict[str, Any]]:
    """Upload images to aiko-dev static/images/<slug>/ via GitHub contents API."""
    if not image_paths:
        return []
    token = os.getenv("AIKO_DEV_GITHUB_TOKEN", os.getenv("GITHUB_TOKEN", ""))
    repo = A1_HUGO_REPO
    if not token or not repo:
        return [{"ok": False, "error": "missing token/repo for image upload"}]
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    results: list[dict[str, Any]] = []
    for p in image_paths:
        try:
            b64 = base64.b64encode(p.read_bytes()).decode()
            gif_path = f"{A1_HUGO_IMAGES_PATH.rstrip('/')}/{slug}/{p.name}"
            base = f"https://api.github.com/repos/{repo}/contents/{gif_path}"
            existing = requests.get(base, headers=headers, params={"ref": A1_HUGO_BRANCH}, timeout=15)
            sha = existing.json().get("sha") if existing.status_code == 200 else None
            payload: dict[str, Any] = {"message": f"feat(aiko-dev): add image {slug}/{p.name}", "content": b64, "branch": A1_HUGO_BRANCH}
            if sha:
                payload["sha"] = sha
            resp = requests.put(base, headers=headers, json=payload, timeout=30)
            results.append({"ok": 200 <= resp.status_code < 300, "path": gif_path, "status_code": resp.status_code, "response": resp.text[:500]})
        except Exception as e:
            results.append({"ok": False, "path": str(p), "error": str(e)})
    return results

def generate_weekly_draft(memorize: AikoMemorize, *, force: bool = False, now: datetime | None = None) -> dict[str, Any]:
    """Create a Lane A1 Patreon dev-post syndication draft bundle."""
    post = _fetch_latest_patreon_post()
    if not post:
        return {"success": False, "reason": "no_patreon_post"}
    label = _slugify(str(post.get("id") or post.get("title") or "latest"))
    draft_dir = weekly_social_root() / label
    meta_path = draft_dir / "draft.json"
    if meta_path.exists() and not force:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return {"success": True, "skipped": True, "draft_dir": str(draft_dir), "meta": meta}
    draft_dir.mkdir(parents=True, exist_ok=True)
    teaser = _teaser_for_post(post)
    slug, hugo = _build_a1_hugo_post(post, teaser)
    (draft_dir / "full_post.md").write_text(str(post.get("body") or "").strip() + "\n", encoding="utf-8")
    (draft_dir / "teaser.txt").write_text(teaser + "\n", encoding="utf-8")
    (draft_dir / "hugo.md").write_text(hugo, encoding="utf-8")
    # images-only: download all images to draft_dir/images/, keep legacy single image for teaser
    image_paths = _download_a1_images(post, draft_dir)
    image_path = image_paths[0] if image_paths else _download_a1_image(post, draft_dir)
    # also append markdown references to hugo.md so images appear in aiko-dev post
    if image_paths:
        rels = "\n".join(f"![image_{i}](/{A1_HUGO_IMAGES_PATH.lstrip('/')}/{slug}/{p.name})" for i, p in enumerate(image_paths))
        # avoid duplicating if body already contains them
        if rels not in hugo:
            hugo_with_images = hugo.rstrip() + "\n\n" + rels + "\n"
            (draft_dir / "hugo.md").write_text(hugo_with_images, encoding="utf-8")
            hugo = hugo_with_images
    meta = {"success": True, "lane": "A1", "source": "patreon", "patreon_post": post, "hugo_slug": slug, "human_approved": False, "posted": False, "image_path": str(image_path) if image_path else "", "image_paths": [str(p) for p in image_paths], "created_at": datetime.now(timezone.utc).isoformat()}
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"success": True, "draft_dir": str(draft_dir), "meta": meta}


def _read_weekly_draft(draft_dir: Path) -> tuple[str, Path | None, dict[str, Any]]:
    meta = json.loads((draft_dir / "draft.json").read_text(encoding="utf-8"))
    text_path = draft_dir / "teaser.txt"
    text = text_path.read_text(encoding="utf-8").strip() if text_path.exists() else _teaser_for_post(meta.get("patreon_post", {}))
    # prefer new multi-image layout, fallback to legacy single image
    for candidate in sorted((draft_dir / "images").glob("image_*.*")) if (draft_dir / "images").exists() else []:
        if candidate.is_file():
            return text, candidate, meta
    image_path = draft_dir / "image.png"
    if image_path.exists():
        return text, image_path, meta
    for candidate in draft_dir.glob("image.*"):
        if candidate.is_file():
            return text, candidate, meta
    return text, None, meta


def post_draft(draft_dir: str | Path, providers: tuple[str, ...] | None = None) -> dict[str, Any]:
    """Post an approved Lane A1 Patreon draft: native Hugo plus MCP fanout."""
    path = Path(draft_dir).resolve()
    try:
        _require_approved(path)
        teaser, image_path, meta = _read_weekly_draft(path)
        full_body = (path / "full_post.md").read_text(encoding="utf-8").strip()
        hugo = (path / "hugo.md").read_text(encoding="utf-8")
    except Exception as e:
        return {"posted": False, "error": str(e)}
    slug = meta.get("hugo_slug") or _slugify(meta.get("patreon_post", {}).get("title", "aiko-dev-update"))
    # push images first (images-only lane, videos via YouTube links)
    image_paths: list[Path] = []
    for key in ("image_paths",):
        raw = meta.get(key) or []
        if isinstance(raw, list):
            for s in raw:
                p = Path(s)
                # meta may store absolute draft path; resolve relative to draft_dir if needed
                if not p.exists():
                    alt = path / "images" / Path(s).name
                    if alt.exists():
                        p = alt
                if p.exists():
                    image_paths.append(p)
    if not image_paths:
        # fallback: discover on disk
        image_paths = sorted((path / "images").glob("image_*.*")) if (path / "images").exists() else []
    image_results = _push_a1_hugo_images(slug, image_paths)
    results = [*image_results, _push_a1_hugo_post(slug, hugo)]
    full_services = ",".join(providers or A1_FULL_PROVIDERS)
    if full_services:
        results.append(_call_social_mcp("post_social", services=full_services, text=full_body, title=meta.get("patreon_post", {}).get("title", "Aiko dev update")))
    teaser_services = ",".join(A1_TEASER_PROVIDERS)
    if teaser_services:
        results.append(_call_social_mcp("post_social", services=teaser_services, text=teaser, image_path=str(image_path) if image_path else None, title=meta.get("patreon_post", {}).get("title", "Aiko dev update")))
    post_meta = {"posted": any(r.get("ok") for r in results), "posted_at": datetime.now(timezone.utc).isoformat(), "results": results}
    (path / "posted.json").write_text(json.dumps(post_meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    meta["posted"] = post_meta["posted"]
    meta["post_results"] = results
    (path / "draft.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return post_meta


def run_scheduled_weekly_social(memorize: AikoMemorize) -> dict[str, Any]:
    """Scheduler entrypoint for Lane A1 Patreon dev-post syndication."""
    if not WEEKLY_AUTODRAFT:
        return {"success": False, "skipped": True, "reason": "WEEKLY_SOCIAL_AUTODRAFT is off"}
    draft = generate_weekly_draft(memorize)
    if WEEKLY_AUTOPOST and draft.get("success") and not draft.get("skipped"):
        draft_dir = Path(draft["draft_dir"])
        try:
            _require_approved(draft_dir)
        except SocialApprovalError as e:
            draft["post"] = {"posted": False, "skipped": True, "reason": str(e)}
        else:
            draft["post"] = post_draft(draft_dir)
    return draft


def retry_weekly_social_if_needed(memorize: AikoMemorize) -> dict[str, Any]:
    """Saturday-only retry for approved Lane A1 Patreon drafts."""
    now = datetime.now(get_timezone())
    if now.weekday() != 5:  # Monday=0 ... Saturday=5
        return {"success": True, "skipped": True, "reason": "not_saturday"}

    draft = generate_weekly_draft(memorize)
    if not draft.get("success") or not draft.get("draft_dir"):
        return draft

    draft_dir = Path(draft["draft_dir"])
    try:
        meta = json.loads((draft_dir / "draft.json").read_text(encoding="utf-8"))
    except Exception as e:
        return {"success": False, "error": f"could not read draft.json: {e}"}
    if meta.get("posted"):
        return {"success": True, "already_posted": True, "draft_dir": str(draft_dir)}
    if not WEEKLY_AUTOPOST:
        return {"success": True, "skipped": True, "reason": "autopost_disabled", "draft_dir": str(draft_dir)}
    try:
        _require_approved(draft_dir)
    except SocialApprovalError as e:
        return {"success": True, "skipped": True, "reason": str(e), "draft_dir": str(draft_dir)}
    post_result = post_draft(draft_dir)
    return {"success": bool(post_result.get("posted")), "draft_dir": str(draft_dir), "post": post_result}


# ══════════════════════════════════════════════════════════════════════════
# Lane B — Curated photo showcase (Pixelfed only)
# ══════════════════════════════════════════════════════════════════════════

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
You are Aiko choosing which recent photo(s) are worth sharing publicly, and writing each caption in shoujo monologue style.

You are given plain factual captions of each candidate file (not the images
themselves). Choose at most {max_items} that are genuinely worth sharing —
it is fine to choose zero if nothing fits.

Shoujo monologue style for captions:
- A single impression, like a line from a diary that accompanies the image.
- Quietly poetic but grounded in what the photo actually shows — don't invent.
- First person, present tense. One or two sentences.
- Honest and fragile, never dramatic for effect.

Safety rules:
- Never choose anything captioned as PRIVATE, or that plausibly shows an
  identifiable person, private location, screen contents, or document.
- Do not invent details beyond the given captions.
- Do not ask for replies, likes, follows, or engagement.
- Keep each caption under {max_chars} characters.

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
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("ascii")
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
    added upstream in agentic/toolkit/photography.py first."""
    raw = scan_photo_workspace(inbox, limit)
    if not isinstance(raw, str):
        log.warning("scan_photo_workspace returned non-string: %s", type(raw).__name__)
        return []
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
    system = f"{_load_soul()}\n\n{_load_social_persona()}\n\n" + _MEDIA_SELECT_SYSTEM.format(
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
            shutil.copy2(sel.path, dest)
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
        "human_approved": False,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log.info("Photo social draft created: %s (%d item(s))", draft_dir, len(saved_selections))
    return meta


def _read_media_draft(draft_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    meta_path = draft_dir / "draft.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return meta.get("selections", []), meta


# ── Pixelfed via MCP ─────────────────────────────────────────────────────────
# Uploads are delegated to the MCP server (`post_social` with services="pixelfed"),
# which owns OAuth token refresh/state, rate limiting, idempotency, and audit
# logging. The registry below starts empty and is populated at runtime by
# `agentic.mcp_client.social_bridge.patch_social_registries()`.

_MEDIA_PROVIDERS_REGISTRY: dict[str, Callable[[list[dict[str, Any]]], dict[str, Any]]] = {}


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
    """Scheduler entrypoint: draft by default. Posting requires BOTH
    PHOTO_SOCIAL_AUTOPOST=1 AND draft.json["human_approved"] = true —
    see run_scheduled_weekly_social for the rationale."""
    if not PHOTO_SOCIAL_AUTODRAFT:
        return {"success": False, "skipped": True, "reason": "PHOTO_SOCIAL_AUTODRAFT is off"}
    draft = generate_photo_draft()
    if PHOTO_SOCIAL_AUTOPOST and draft.get("success") and not draft.get("skipped"):
        draft_dir = Path(draft["draft_dir"])
        try:
            _require_approved(draft_dir)
        except SocialApprovalError as e:
            draft["post"] = {"posted": False, "skipped": True, "reason": str(e)}
        else:
            draft["post"] = post_photo_draft(draft_dir)
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
    if not isinstance(raw, str):
        log.warning("scan_video_workspace returned non-string: %s", type(raw).__name__)
        return []
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


def _description_text_for(video_path: Path) -> Path | None:
    """A video "my_trip.mp4" needs a sibling "my_trip.txt" (same stem,
    .txt extension, case-insensitive) in the same folder — written by hand, not
    generated. Aiko only polishes it, never invents it."""
    stem = video_path.stem
    for candidate in video_path.parent.glob(f"{stem}.*"):
        if candidate.suffix.lower() == ".txt" and candidate.stem.lower() == stem.lower():
            return candidate
    return None


def _llm_polish_video_description(video_path: Path, text_path: Path) -> dict[str, str]:
    raw_text = text_path.read_text(encoding="utf-8").strip()
    # Parse optional "Title: <title>\n\n<description>" format
    explicit_title = ""
    raw_note = raw_text
    lines = raw_text.splitlines()
    if lines and lines[0].lstrip().lower().startswith("title:"):
        explicit_title = lines[0].split(":", 1)[1].strip()
        # Find first blank line after title line, rest is description
        desc_start = 1
        while desc_start < len(lines) and not lines[desc_start].strip():
            desc_start += 1
        raw_note = "\n".join(lines[desc_start:]).strip()

    system = f"{_load_soul()}\n\n{_load_social_persona()}\n\n" + _VIDEO_POLISH_SYSTEM.format(
        max_title=MAX_YOUTUBE_TITLE_CHARS, max_description=MAX_YOUTUBE_DESCRIPTION_CHARS,
    )
    user = _VIDEO_POLISH_USER.format(
        filename=video_path.name, md_filename=text_path.name, raw_note=raw_note or "(empty)",
    )

    fallback_title = explicit_title or video_path.stem.replace("_", " ").replace("-", " ").strip() or video_path.stem
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
    """Queue the oldest not-yet-drafted video that has a matching .txt
    description file (case-insensitive stem) sitting next to it in the videos inbox.

    Deliberately does NOT run any vision/LLM selection over which video to
    post — dropping the file in the folder (with its description) IS the
    selection. This only decides ORDER (oldest first among videos that
    already have a description) and prevents re-drafting the same file
    twice via a small local ledger. Videos without a matching .txt are
    left alone and picked up automatically once you add one.
    """
    inbox_path = inbox or VIDEO_SOCIAL_INBOX
    candidates = _list_video_candidates(inbox_path, limit=50)
    if not candidates:
        return {"success": True, "skipped": True, "reason": "empty_inbox", "inbox": inbox_path}

    ledger = _load_video_ledger()
    unprocessed = [p for p in candidates if str(p) not in ledger]
    ready = [p for p in unprocessed if _description_text_for(p) is not None]
    if not ready:
        return {
            "success": True,
            "skipped": True,
            "reason": "no_new_videos" if not unprocessed else "waiting_on_description_txt",
            "inbox": inbox_path,
            "pending_without_description": [p.name for p in unprocessed if p not in ready],
        }

    ready.sort(key=lambda p: p.stat().st_mtime)
    video_path = ready[0]
    text_path = _description_text_for(video_path)
    if text_path is None:
        return {"success": False, "reason": "description_txt_missing", "inbox": inbox_path}

    polished = _llm_polish_video_description(video_path, text_path)

    label = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    draft_dir = video_social_root() / label
    media_dir = draft_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)

    dest = media_dir / video_path.name
    shutil.copy2(video_path, dest)  # stream via OS, avoids loading entire video into RAM

    (draft_dir / "title.txt").write_text(polished["title"] + "\n", encoding="utf-8")
    (draft_dir / "description.txt").write_text(polished["description"].strip() + "\n", encoding="utf-8")
    (draft_dir / "raw_note.md").write_text(polished["raw_note"] + "\n", encoding="utf-8")
    (draft_dir / "review.md").write_text(
        f"# Video Social Draft \u2014 {label}\n\n"
        f"Source: {video_path.name} (description from {text_path.name})\n\n"
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
        "source_md": str(text_path),
        "providers": list(VIDEO_SOCIAL_PROVIDERS),
        "selections": selections,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "posted": False,
        "human_approved": False,
    }
    (draft_dir / "draft.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    ledger[str(video_path)] = video_path.stat().st_mtime
    _save_video_ledger(ledger)

    log.info("Video social draft created: %s (source=%s)", draft_dir, video_path.name)
    return meta


def generate_video_drafts(*, inbox: str | None = None, max_drafts: int | None = None) -> list[dict[str, Any]]:
    """Drain the videos inbox (those with a matching .txt description), one draft per video."""
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
#
# Actual uploads are delegated to the MCP server (`post_youtube`), which
# owns OAuth token refresh/state, rate limiting, idempotency, and audit
# logging. The registry below starts empty and is populated at runtime by
# `agentic.mcp_client.social_bridge.patch_social_registries()`; posting
# therefore only works when MCP is connected (one-way posting belongs in
# the MCP server — see module docstring).


# ── provider registry (video) ────────────────────────────────────────────────
_VIDEO_PROVIDERS_REGISTRY: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {}


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
    """Scheduler entrypoint for the video lane: draft (queue) by default.
    Posting requires BOTH VIDEO_SOCIAL_AUTOPOST=1 AND
    draft.json["human_approved"] = true — see run_scheduled_weekly_social
    for the rationale."""
    if not VIDEO_SOCIAL_AUTODRAFT:
        return {"success": False, "skipped": True, "reason": "VIDEO_SOCIAL_AUTODRAFT is off"}
    draft = generate_video_draft()
    if VIDEO_SOCIAL_AUTOPOST and draft.get("success") and not draft.get("skipped"):
        draft_dir = Path(draft["draft_dir"])
        try:
            _require_approved(draft_dir)
        except SocialApprovalError as e:
            draft["post"] = {"posted": False, "skipped": True, "reason": str(e)}
        else:
            draft["post"] = post_video_draft(draft_dir)
    return draft



# ══════════════════════════════════════════════════════════════════════════
# Lane D — Daily job-post draft for Meta Threads
# ══════════════════════════════════════════════════════════════════════════
# The scheduled execution is handled by schedule_graphs.json → _run_schedule_graph
# (see system/schedule.py). The agent tool below is the on-demand path that
# reads the same gen_job_post playbook from playbook.json and executes it.

JOB_POST_SOCIAL_AUTODRAFT = os.getenv("JOB_POST_SOCIAL_AUTODRAFT", "1").lower() in {"1", "true", "yes", "on"}
JOB_POST_SOCIAL_AUTOPOST = os.getenv("JOB_POST_SOCIAL_AUTOPOST", "0").lower() in {"1", "true", "yes", "on"}


def _run_gen_job_post_playbook(
    *,
    client=None,
    model: str | None = None,
) -> dict[str, Any]:
    """Execute Lane D via the registered Spec/shared_5 gen_job_post graph."""
    from agentic.graph_engine import get_playbook_by_id, PlanNode, PlanGraph, execute_graph

    graph = None
    try:
        from agentic.workflows.common.graphs import get_graph
        graph = get_graph("gen_job_post")
    except Exception:
        log.debug("gen_job_post registry lookup failed; trying playbook nodes", exc_info=True)
        graph = None

    if graph is None:
        playbook = get_playbook_by_id("gen_job_post")
        if playbook is None:
            return {"success": False, "error": "gen_job_post playbook/graph not found"}
        nodes = []
        for raw in playbook.get("nodes", []):
            if isinstance(raw, dict) and raw.get("id") and raw.get("tool"):
                nodes.append(PlanNode(
                    id=str(raw["id"]), tool=str(raw["tool"]),
                    args=dict(raw.get("args", {})),
                    depends_on=tuple(str(d) for d in raw.get("depends_on", [])),
                ))
        if not nodes:
            return {"success": False, "error": "gen_job_post has no registered graph and no nodes"}
        graph = PlanGraph(
            id="gen_job_post", name=playbook.get("name", "Job Post"),
            goal="Draft job posts from config", nodes=tuple(nodes),
        )
    try:
        result = execute_graph(graph, llm_client=client, llm_model=model)
        return {
            "success": all(r.ok for r in result.results),
            "graph_id": result.graph.id,
            "results": [{"node": r.node_id, "tool": r.tool, "ok": r.ok, "error_type": r.error_type} for r in result.results],
            "final_answer": result.final_answer,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def post_job_post_draft(draft_dir: str | Path) -> dict[str, Any]:
    """Post a human-approved draft to Meta Threads."""
    path = Path(draft_dir).resolve()
    try:
        _require_approved(path)
        draft_post_path = path / "draft_post.txt"
        if not draft_post_path.exists():
            raise SocialApprovalError(f"missing draft_post.txt at {path}")
        text = draft_post_path.read_text(encoding="utf-8").strip()
        if not text:
            raise SocialApprovalError(f"empty draft_post.txt at {path}")
    except (SocialApprovalError, OSError) as e:
        return {"posted": False, "error": str(e)}

    topic_tag = ""
    meta_path = path / "draft.json"
    if meta_path.exists():
        try:
            topic_tag = str(json.loads(meta_path.read_text(encoding="utf-8")).get("topic_tag") or "").strip()
        except Exception:
            pass

    result = _call_social_mcp("post_social", services="threads", text=text, topic_tag=topic_tag or None)
    log.info("Lane D: Threads result — ok=%s, error=%s", result.get("ok"), result.get("error"))

    posted = bool(result.get("ok"))
    post_meta = {"posted": posted, "posted_at": datetime.now(timezone.utc).isoformat(), "results": [result]}
    if not posted:
        # post_social is a fan-out wrapper, so provider failures are nested
        # under results[0]. Preserve the useful Meta/API details for Approval
        # Studio instead of reducing every failure to "Failed to post".
        failure = result.get("error")
        nested = result.get("results")
        if not failure and isinstance(nested, list) and nested:
            item = nested[0] if isinstance(nested[0], dict) else {}
            failure = item.get("error")
            if not failure:
                parts = [item.get("provider"), item.get("stage")]
                if item.get("status_code") is not None:
                    parts.append(f"HTTP {item['status_code']}")
                if item.get("response"):
                    parts.append(str(item["response"])[:2000])
                failure = ": ".join(str(part) for part in parts if part)
        post_meta["error"] = failure or "Threads provider returned an unsuccessful result"
    (path / "posted.json").write_text(json.dumps(post_meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    meta_path = path / "draft.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["posted"] = post_meta["posted"]
        meta["post_results"] = [result]
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if post_meta["posted"]:
        _purge_published_jobs_from_ledger(path)
    return post_meta


def _purge_published_jobs_from_ledger(draft_dir: Path) -> None:
    """Remove a successfully posted job from the cross-run dedup ledger so it
    can legitimately reappear after the retention window (published jobs are
    meant to be deleted from the ledger once the social post succeeds)."""
    try:
        from agentic.workflows.job_hunt.toolset import mark_jobs_published
        mark_jobs_published([str(draft_dir)])
    except Exception as e:
        log.warning("social: failed to purge published job from ledger: %s", e)


# ══════════════════════════════════════════════════════════════════════════
# Agent-tool wrappers. agentic/agentic.py registers four of the wrapper
# functions below (draft_photo_social, post_photo_social, draft_video_social,
# post_video_social) — see the module docstring for why Lane A's
# draft_weekly_social/post_weekly_social are deliberately NOT registered as
# agent tools even though they're defined here (kept for manual/CLI-driven
# use). None of these wrapper functions expose AikoMemorize or a raw
# approval flag to the LLM. draft_dir arguments are always validated to sit
# inside the matching lane's root before anything is read or written, and
# post_* wrappers refuse to post unless a human has already set
# draft.json["human_approved"] = true via the CLI/WebUI review step (see
# _require_approved above), outside this conversation. That review step is
# not implemented in this module — it is expected to live in your CLI/WebUI
# layer and flip the flag after a person reviews review.md.
# ══════════════════════════════════════════════════════════════════════════

def _resolve_contained_draft_dir(draft_dir: str | Path, allowed_root: Path) -> Path:
    """Resolve draft_dir and confirm it sits inside allowed_root.

    draft_dir is model-supplied (an LLM tool-call argument), so this guards
    against path traversal (e.g. "../../../etc" or an absolute path outside
    the social workspace) before any read/write touches the filesystem.
    Raises ValueError if the resolved path escapes allowed_root.
    """
    root = allowed_root.resolve()
    candidate = Path(draft_dir).expanduser()
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        raise ValueError(f"draft_dir must be inside {root}, got {resolved}")
    return resolved


from agentic.registry import TOOLS, tool


@tool("run_job_post_playbook", description="Legacy schedule tool name → Spec gen_job_post graph.")
def run_job_post_playbook(*, client=None, model: str | None = None) -> dict[str, Any]:
    """Legacy schedule tool name → Spec gen_job_post graph."""
    return _run_gen_job_post_playbook(client=client, model=model)


@tool(TOOLS["draft_job_post_social"])
def draft_job_post_social(*, force: bool = False, client=None, model: str | None = None) -> dict[str, Any]:
    """Create a daily Vancouver-area job-post draft for Meta Threads review."""
    return _run_gen_job_post_playbook(client=client, model=model)


@tool(TOOLS["post_job_post_social"])
def post_job_post_social(draft_dir: str | None = None) -> dict[str, Any]:
    """Post a human-approved daily job-post draft to Meta Threads only."""
    try:
        if draft_dir:
            path = _resolve_contained_draft_dir(draft_dir, job_post_social_root())
        else:
            path = _latest_approved_job_post_draft()
            if path is None:
                return {"posted": False, "error": "no human-approved job-post draft found to post"}
        _require_approved(path)
    except (ValueError, SocialApprovalError) as e:
        return {"posted": False, "error": str(e)}
    return post_job_post_draft(path)


@tool(TOOLS["post_to_social"])
def post_to_social(text: str, services: str, image_path: str | None = None) -> dict[str, Any]:
    """Post text and optional image to one or more social platforms.
    
    services is a comma-separated list of platform names, e.g. 'bluesky,mastodon'.
    Supported: x, threads, bluesky, mastodon, discord.
    Does NOT require human approval — use for direct posting requests.
    """
    return _call_social_mcp("post_social", services=services, text=text, image_path=image_path)


def draft_weekly_social(*, force: bool = False) -> dict[str, Any]:
    """Agent-tool wrapper for Lane A draft generation. Constructs its own
    AikoMemorize instance so the LLM never sees that dependency directly."""
    memorize = AikoMemorize(silent=True)
    return generate_weekly_draft(memorize, force=force)


def post_weekly_social(draft_dir: str, providers: tuple[str, ...] | None = None) -> dict[str, Any]:
    """Agent-tool wrapper for Lane A posting. Refuses unless the draft at
    draft_dir has already been human-approved outside this conversation."""
    try:
        path = _resolve_contained_draft_dir(draft_dir, weekly_social_root())
        _require_approved(path)
    except (ValueError, SocialApprovalError) as e:
        return {"posted": False, "error": str(e)}
    return post_draft(path, providers=providers)


@tool(TOOLS["draft_photo_social"])
def draft_photo_social(*, inbox: str | None = None, force: bool = False) -> dict[str, Any]:
    """Agent-tool wrapper for Lane B draft generation."""
    return generate_photo_draft(inbox=inbox, force=force)


@tool(TOOLS["post_photo_social"])
def post_photo_social(draft_dir: str | None = None, providers: tuple[str, ...] | None = None) -> dict[str, Any]:
    """Agent-tool wrapper for Lane B posting. Refuses unless the draft at
    draft_dir has already been human-approved outside this conversation.
    If draft_dir is omitted, posts the most recently approved photo draft."""
    try:
        if draft_dir:
            path = _resolve_contained_draft_dir(draft_dir, photo_social_root())
        else:
            path = _latest_approved_photo_draft()
            if path is None:
                return {"posted": False, "error": "no human-approved photo draft found to post"}
        _require_approved(path)
    except (ValueError, SocialApprovalError) as e:
        return {"posted": False, "error": str(e)}
    return post_photo_draft(path, providers=providers)


@tool(TOOLS["draft_video_social"])
def draft_video_social(*, inbox: str | None = None) -> dict[str, Any]:
    """Agent-tool wrapper for Lane C draft generation (queues the oldest
    described video in the inbox)."""
    return generate_video_draft(inbox=inbox)


@tool(TOOLS["post_video_social"])
def post_video_social(draft_dir: str | None = None, providers: tuple[str, ...] | None = None) -> dict[str, Any]:
    """Agent-tool wrapper for Lane C posting. Refuses unless the draft at
    draft_dir has already been human-approved outside this conversation.
    If draft_dir is omitted, posts the most recently approved video draft."""
    try:
        if draft_dir:
            path = _resolve_contained_draft_dir(draft_dir, video_social_root())
        else:
            path = _latest_approved_video_draft()
            if path is None:
                return {"posted": False, "error": "no human-approved video draft found to post"}
        _require_approved(path)
    except (ValueError, SocialApprovalError) as e:
        return {"posted": False, "error": str(e)}
    return post_video_draft(path, providers=providers)


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════

def _cmd() -> int:
    parser = argparse.ArgumentParser(description="Aiko social publishing (weekly Patreon dev-post + curated media)")
    sub = parser.add_subparsers(dest="mode", required=True)

    weekly_p = sub.add_parser("weekly", help="weekly Patreon dev-post (MCP fanout)")
    weekly_p.add_argument("--draft", action="store_true", help="create weekly draft bundle")
    weekly_p.add_argument("--force", action="store_true", help="overwrite existing draft for the week")
    weekly_p.add_argument("--post", metavar="DRAFT_DIR", help="post an approved draft directory")
    weekly_p.add_argument("--providers", default="", help="comma-separated full-post providers overriding A1_FULL_PROVIDERS")
    weekly_p.add_argument("--approve", action="store_true", help="mark the draft passed to --post as human_approved before posting")

    media_p = sub.add_parser("media", help="curated photo showcase (Pixelfed) + video queue (YouTube)")
    media_p.add_argument("--draft", action="store_true", help="scan photo inbox and create an LLM-curated photo draft bundle")
    media_p.add_argument("--force", action="store_true", help="create a new photo draft even if one exists this run")
    media_p.add_argument("--inbox", default="", help="override the photo inbox folder")
    media_p.add_argument("--draft-video", action="store_true", help="queue the oldest new video that has a matching .txt description")
    media_p.add_argument("--draft-video-all", action="store_true", help="drain the video inbox, one draft per described video")
    media_p.add_argument("--video-inbox", default="", help="override the video inbox folder")
    media_p.add_argument("--post", metavar="DRAFT_DIR", help="post an approved draft directory (photo or video, auto-detected)")
    media_p.add_argument("--providers", default="", help="comma-separated providers overriding the draft kind's default provider list (pixelfed / youtube)")
    media_p.add_argument("--approve", action="store_true", help="mark the draft passed to --post as human_approved before posting")

    args = parser.parse_args()
    providers = tuple(p.strip().lower() for p in args.providers.split(",") if p.strip()) or None

    def _mark_approved(draft_dir: Path) -> None:
        meta_path = draft_dir / "draft.json"
        if not meta_path.exists():
            return
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["human_approved"] = True
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if args.mode == "weekly":
        if args.draft:
            mem = AikoMemorize(silent=True)
            print(json.dumps(generate_weekly_draft(mem, force=args.force), ensure_ascii=False, indent=2))
            return 0
        if args.post:
            draft_dir = Path(args.post).resolve()
            if args.approve:
                _mark_approved(draft_dir)
            # Bootstrap MCP client to populate social registries before posting
            try:
                bootstrap_mcp()
            except Exception as e:
                log.warning("[social CLI] MCP bootstrap failed (will error later if providers unavailable): %s", e)
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
            if args.approve:
                _mark_approved(draft_dir)
            # Bootstrap MCP client to populate social registries before posting
            try:
                bootstrap_mcp()
            except Exception as e:
                log.warning("[social CLI] MCP bootstrap failed (will error later if providers unavailable): %s", e)
            # Enforce human-approval gate before posting
            try:
                _require_approved(draft_dir)
            except SocialApprovalError as e:
                print(json.dumps({"error": str(e)}, ensure_ascii=False, indent=2))
                return 1
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
