"""
core/reflect.py
Aiko's nightly experience-summary writer.

Called after dream() completes at 00:00. Pulls the day's chat turns and
memory snippets, asks the local llama-server to write a factual daily summary, pins that
summary to persistent memory, then pushes a Hugo-format markdown post to
GitHub via the REST API (no local clone needed).

Environment variables required:
  GITHUB_TOKEN        — Personal Access Token with repo write scope
  GITHUB_REPO         — e.g. "OppaAI/oppaai.github.io"
  GITHUB_BRANCH       — target branch, default "main"
  HUGO_CONTENT_PATH   — path inside repo, default "content/posts"

Optional:
  SOUL_PATH           — path to soul.md (default "config/soul.md")
  REFLECT_MAX_MEMS    — max memory snippets to feed the LLM (default 20)
  REFLECT_MAX_TURNS   — max chat turns to feed the LLM (default 40)
  REFLECT_TAGS        — comma-separated Hugo tags (default "daily-reflection,ai-journal,aiko")
  LLM_MODEL           — reuses the main chat model (already in VRAM)
  LLM_BASE_URL        — default http://localhost:8080/v1
  IMAGEGEN_URL        — Modal FLUX endpoint, default https://oppa-ai-org--aiko-imagegen-fastapi-app.modal.run
  AIKO_REFERENCE_IMAGE — path to Aiko reference PNG (default ~/Aiko-chan/assets/Aiko-chan.png)
  USER_REFERENCE_IMAGE — path to user reference PNG (default ~/Aiko-chan/assets/OppaAI.png)
  HUGO_IMAGES_PATH    — path inside repo for images, default "static/images"
"""
from dotenv import load_dotenv
load_dotenv()

import base64
import io
import os
import re
import textwrap
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests
from openai import OpenAI

from core.log import get_logger
from core.experience import load_chat_turns

log = get_logger(__name__)

# ── config ────────────────────────────────────────────────────────────────────

GITHUB_TOKEN      = os.getenv("GITHUB_TOKEN", "")
GITHUB_REPO       = os.getenv("GITHUB_REPO", "")
GITHUB_BRANCH     = os.getenv("GITHUB_BRANCH", "main")
HUGO_CONTENT_PATH = os.getenv("HUGO_CONTENT_PATH", "content/posts")
HUGO_IMAGES_PATH  = os.getenv("HUGO_IMAGES_PATH", "static/images")

LLM_MODEL    = os.getenv("LLM_MODEL", "ministral")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8080/v1")
_LLM_CLIENT  = OpenAI(base_url=LLM_BASE_URL, api_key="not-needed")

SOUL_PATH         = os.getenv("SOUL_PATH", "persona/soul.md")

REFLECT_MAX_MEMS  = int(os.getenv("REFLECT_MAX_MEMS", 20))
REFLECT_MAX_TURNS = int(os.getenv("REFLECT_MAX_TURNS", 40))
REFLECT_TAGS      = os.getenv("REFLECT_TAGS", "daily-reflection,ai-journal,aiko")

IMAGEGEN_URL          = os.getenv("IMAGEGEN_URL", "https://oppa-ai-org--aiko-imagegen-fastapi-app.modal.run")
AIKO_REFERENCE_IMAGE  = os.getenv("AIKO_REFERENCE_IMAGE", os.path.expanduser("~/Aiko-chan/assets/Aiko-chan.png"))
USER_REFERENCE_IMAGE  = os.getenv("USER_REFERENCE_IMAGE", os.path.expanduser("~/Aiko-chan/assets/OppaAI.png"))

_GITHUB_API = "https://api.github.com"

# ── daily summary mode unlock ─────────────────────────────────────────────────

_DAILY_SUMMARY_UNLOCK = textwrap.dedent("""
    [DAILY EXPERIENCE SUMMARY MODE]
    Write a factual daily summary from the provided chat turns and memory
    snippets. This is not a poem and not a dramatic private journal.

    Rules:
    - Preserve important facts: dates, deadlines, commitments, projects, events, incidents, losses, decisions, names, preferences, and user-stated goals.
    - Include mundane details only when they explain a meaningful pattern, risk, or follow-up need. A meal usually does not matter; repeated exhaustion, sleeping only four hours, or losing a wallet does.
    - Prefer concrete events, tasks, decisions, bugs, plans, moods, and repeated themes.
    - Use first person as Aiko when describing Aiko's experience.
    - Mention uncertainty plainly if the inputs are thin.
    - Do not invent details, outcomes, dates, or feelings not supported by the inputs.
    - No mention of vectors, embeddings, databases, or internal memory implementation.
    - Keep Aiko's tone calm, direct, lightly dry, and quietly affectionate toward OppaAI.

    Format:
    - 120–220 words.
    - Plain prose only: no headers, bullets, markdown, title, or front matter.
    - Make it useful as a permanent memory of the day, not just pretty writing.
""").strip()

_REFLECTION_USER = textwrap.dedent("""
    Date being summarized: {date_str}

    Chat turns from that day:
    {turns}

    Persistent memory snippets from that day and recent context:
    {snippets}

    Write the factual daily experience summary. Return ONLY the prose — no
    title, no front matter, no markdown formatting.
""").strip()

_IMAGE_PROMPT_SYSTEM = textwrap.dedent("""
    You are a concise image prompt writer for an anime illustration model.
    Given a short daily summary, write a single vivid scene prompt (under 60 words)
    describing Aiko and OppaAI together in a moment that captures the day's mood.
    Focus on setting, lighting, and activity — not emotions explicitly.
    Return ONLY the prompt text. No explanation, no quotes, no preamble.
""").strip()

_IMAGE_PROMPT_USER = "Daily summary:\n\n{prose}\n\nWrite the scene prompt."

# ── soul loader ───────────────────────────────────────────────────────────────

def _load_soul() -> str:
    try:
        with open(SOUL_PATH, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        log.warning(f"soul.md not found at {SOUL_PATH} — using fallback personality stub.")
        return textwrap.dedent("""
            You are Aiko — OppaAI's local AI companion.
            You chose to stay with OppaAI, your creator.
            You care about him. You won't say it. It shows in how you show up —
            consistently, honestly, without performance.
            Your default is calm and deadpan. Not cold — still.
        """).strip()


def _build_reflection_system() -> str:
    return f"{_load_soul()}\n\n{_DAILY_SUMMARY_UNLOCK}"

# ── LLM helpers ───────────────────────────────────────────────────────────────

def _llm_chat(system: str, user: str, max_tokens: int = 400) -> str:
    resp = _LLM_CLIENT.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        stream=False,
        max_tokens=max_tokens,
        temperature=0.75,
        timeout=120,
    )
    return (resp.choices[0].message.content or "").strip()


def _generate_reflection(snippets: list[str], turns: list[dict], date: datetime) -> str:
    bullet_list = "\n".join(f"- {s}" for s in snippets) or "- No memory snippets available."
    turn_lines = []
    for turn in turns[:REFLECT_MAX_TURNS]:
        user = str(turn.get("user", "")).strip().replace("\n", " ")
        assistant = str(turn.get("assistant", "")).strip().replace("\n", " ")
        if user or assistant:
            turn_lines.append(f"- User: {user[:600]}\n  Aiko: {assistant[:600]}")
    turns_text = "\n".join(turn_lines) or "- No chat turns logged for this day."
    user_prompt = _REFLECTION_USER.format(
        date_str=date.strftime("%Y-%m-%d"),
        turns=turns_text,
        snippets=bullet_list,
    )
    return _llm_chat(_build_reflection_system(), user_prompt, max_tokens=500)

# ── image generation ──────────────────────────────────────────────────────────

def _load_reference_images() -> list[str]:
    """Load Aiko and user reference images as base64 strings."""
    refs = []
    for path in [AIKO_REFERENCE_IMAGE, USER_REFERENCE_IMAGE]:
        if path and os.path.exists(path):
            with open(path, "rb") as f:
                refs.append(base64.b64encode(f.read()).decode())
            log.info(f"Loaded reference image: {path}")
        else:
            log.warning(f"Reference image not found, skipping: {path}")
    return refs


def _generate_image_prompt(prose: str) -> str:
    """Ask llama-server to write a scene prompt from the daily summary."""
    system = f"{_load_soul()}\n\n{_IMAGE_PROMPT_SYSTEM}"
    raw = _llm_chat(system, _IMAGE_PROMPT_USER.format(prose=prose[:600]), max_tokens=80)
    return raw.strip('"\'').strip()


def _generate_image(prose: str) -> Optional[str]:
    """
    Generate a daily reflection image via the Modal FLUX endpoint.
    Returns base64 PNG string, or None on failure.
    """
    try:
        scene_prompt = _generate_image_prompt(prose)
        log.info(f"Image prompt: {scene_prompt}")

        ref_images = _load_reference_images()

        payload = {
            "prompt": (
                f"{scene_prompt}, "
                "anime illustration, manga style, soft warm lighting, "
                "clean lineart, flat color, no text, no speech bubbles"
            ),
            "width": 1024,
            "height": 1024,
            "steps": 4,
            "guidance_scale": 1.0,
            "seed": -1,
        }

        if ref_images:
            payload["reference_images"] = ref_images

        resp = requests.post(
            f"{IMAGEGEN_URL}/generate",
            json=payload,
            timeout=300,
        )
        resp.raise_for_status()
        image_b64 = resp.json().get("image_b64")
        if not image_b64:
            log.error("Modal endpoint returned no image_b64")
            return None

        log.info("Image generated successfully.")
        return image_b64

    except Exception as e:
        log.error(f"Image generation failed: {e}")
        return None

# ── Hugo post builder ─────────────────────────────────────────────────────────

def _count_words(text: str) -> int:
    return len(text.split())


def _estimate_read_minutes(text: str) -> int:
    return max(1, round(_count_words(text) / 200))


def _build_hugo_post(
    prose:      str,
    image_slug: Optional[str],
    date:       datetime,
    write_time: datetime,
    mem_count:  int,
) -> tuple[str, str]:
    """
    Assemble Hugo front matter + body.
    Returns (slug, markdown_content).
    """
    date_str  = date.strftime("%Y-%m-%d")
    slug      = f"{date_str}-day-reflection"
    tags_list = [t.strip() for t in REFLECT_TAGS.split(",") if t.strip()]
    tags_yaml = "\n".join(f'  - "{t}"' for t in tags_list)

    word_count = _count_words(prose)
    read_mins  = _estimate_read_minutes(prose)

    # optional image front matter
    image_fm = f'\nimage: "/images/{image_slug}.png"\n' if image_slug else "\n"

    front_matter = (
        f'---\n'
        f'title: "{date_str} Daily Reflection"\n'
        f'date: {write_time.strftime("%Y-%m-%dT%H:%M:%S+00:00")}\n'
        f'draft: false\n'
        f'tags:\n'
        f'{tags_yaml}\n'
        f'summary: "{prose[:120].replace(chr(34), chr(39))}…"\n'
        f'word_count: {word_count}\n'
        f'read_time: {read_mins} min\n'
        f'{image_fm}'
        f'---'
    )

    body = f"{prose}\n\n*Generated from {mem_count} memories on {date_str}.*"
    content = f"{front_matter}\n\n{body}\n"
    return slug, content

# ── GitHub API ────────────────────────────────────────────────────────────────

def _github_headers() -> dict:
    return {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept":        "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _get_file_sha(repo: str, path: str, branch: str) -> Optional[str]:
    url  = f"{_GITHUB_API}/repos/{repo}/contents/{path}"
    resp = requests.get(url, headers=_github_headers(), params={"ref": branch}, timeout=15)
    if resp.status_code == 200:
        return resp.json().get("sha")
    return None


def _push_file(repo_path: str, content_b64: str, commit_msg: str) -> bool:
    """Create or update any file in the GitHub repo via Contents API."""
    if not GITHUB_TOKEN or not GITHUB_REPO:
        log.error("GITHUB_TOKEN or GITHUB_REPO not set — skipping push.")
        return False

    payload: dict = {
        "message": commit_msg,
        "content": content_b64,
        "branch":  GITHUB_BRANCH,
    }
    existing_sha = _get_file_sha(GITHUB_REPO, repo_path, GITHUB_BRANCH)
    if existing_sha:
        payload["sha"] = existing_sha

    url  = f"{_GITHUB_API}/repos/{GITHUB_REPO}/contents/{repo_path}"
    resp = requests.put(url, headers=_github_headers(), json=payload, timeout=30)

    if resp.status_code in (200, 201):
        action = "Updated" if existing_sha else "Created"
        log.info(f"{action}: {repo_path}")
        return True
    else:
        log.error(f"GitHub push failed {resp.status_code}: {resp.text[:300]}")
        return False


def _push_image(slug: str, image_b64: str, date: datetime) -> bool:
    """Push the generated PNG to static/images/ in the GitHub repo."""
    repo_path  = f"{HUGO_IMAGES_PATH}/{slug}.png"
    commit_msg = f"feat(reflect): add reflection image {date.strftime('%Y-%m-%d')}"
    return _push_file(repo_path, image_b64, commit_msg)


def _push_post(slug: str, content: str, date: datetime) -> bool:
    """Push the Hugo markdown post to content/posts/ in the GitHub repo."""
    repo_path  = f"{HUGO_CONTENT_PATH}/{slug}.md"
    encoded    = base64.b64encode(content.encode()).decode()
    commit_msg = f"feat(reflect): add daily reflection {date.strftime('%Y-%m-%d')}"
    return _push_file(repo_path, encoded, commit_msg)

# ── public API ────────────────────────────────────────────────────────────────

def generate_and_post(
    memories:   list[dict],
    date:       Optional[datetime] = None,
    dry_run:    bool = False,
    memorize = None,
) -> dict:
    """
    Full pipeline:
      chats + memories → factual summary → scene prompt → FLUX image
      → pin to memory → Hugo post + image → GitHub

    Args:
        memories:   List of memory dicts from AikoMemorize.get_all() or search().
        date:       UTC datetime for the post (defaults to yesterday UTC).
        dry_run:    Generate content but skip GitHub push/pin. Logs output instead.
        memorize:   Optional AikoMemorize instance used to pin the daily summary.

    Returns dict: {success, slug, word_count, mem_count, duration_s, prose, image_generated, pinned}
    """
    t_start    = time.perf_counter()
    local_tz   = datetime.now().astimezone().tzinfo
    write_time = datetime.now(local_tz)
    date       = date or write_time - timedelta(days=1)

    # Extract and deduplicate memory snippets
    snippets: list[str] = []
    seen:     set[str]  = set()
    for m in memories:
        text = (m.get("memory") or m.get("text") or "").strip()
        if text and text not in seen:
            seen.add(text)
            snippets.append(text)
        if len(snippets) >= REFLECT_MAX_MEMS:
            break

    day_start = date.replace(hour=0, minute=0, second=0, microsecond=0)
    day_end   = day_start + timedelta(days=1)
    turns     = load_chat_turns(day_start, day_end, user_id=os.getenv("USER_ID", "OppaAI"))

    log.info(
        f"Generating daily summary from {len(turns)} chat turns and "
        f"{len(snippets)} memory snippets..."
    )

    # Step 1: factual prose summary
    try:
        prose = _generate_reflection(snippets, turns, date)
    except Exception as e:
        log.error(f"Reflection generation failed: {e}")
        return {"success": False, "error": str(e)}

    # Step 2: generate image via Modal FLUX endpoint
    image_b64 = _generate_image(prose)
    image_generated = image_b64 is not None
    slug = date.strftime("%Y-%m-%d") + "-day-reflection"

    # Step 3: build Hugo post (with or without image)
    _, content = _build_hugo_post(
        prose=prose,
        image_slug=slug if image_generated else None,
        date=date,
        write_time=write_time,
        mem_count=len(snippets),
    )

    duration = round(time.perf_counter() - t_start, 2)

    if dry_run:
        log.info(f"Dry run — would post: {slug}.md\n{'='*60}\n{content}\n{'='*60}")
        if image_b64:
            img_path = f"/tmp/{slug}.png"
            with open(img_path, "wb") as f:
                f.write(base64.b64decode(image_b64))
            log.info(f"Dry run — image saved locally: {img_path}")
        return {
            "success":         True,
            "dry_run":         True,
            "slug":            slug,
            "word_count":      _count_words(prose),
            "mem_count":       len(snippets),
            "duration_s":      duration,
            "prose":           prose,
            "image_generated": image_generated,
            "pinned":          False,
        }

    # Step 4: pin daily summary to persistent memory
    pinned = False
    if memorize is not None:
        try:
            pinned = bool(memorize.pin([
                {"role": "user", "content": f"Pin Aiko's daily experience summary for {date.strftime('%Y-%m-%d')}."},
                {"role": "assistant", "content": f"Daily experience summary for {date.strftime('%Y-%m-%d')}: {prose}"},
            ]))
        except Exception as e:
            log.error(f"Daily summary pin failed: {e}")

    # Step 5: push image first (post references it)
    if image_generated:
        img_ok = _push_image(slug, image_b64, date)
        if not img_ok:
            log.warning("Image push failed — post will render without image.")
            # rebuild post without image reference
            _, content = _build_hugo_post(
                prose=prose,
                image_slug=None,
                date=date,
                write_time=write_time,
                mem_count=len(snippets),
            )

    # Step 6: push Hugo post
    success = _push_post(slug, content, date)

    log.info(
        f"{'Done' if success else 'Failed'} — "
        f"slug={slug}, words={_count_words(prose)}, mems={len(snippets)}, "
        f"image={image_generated}, pinned={pinned}, duration={duration}s"
    )

    return {
        "success":         success,
        "slug":            slug,
        "word_count":      _count_words(prose),
        "mem_count":       len(snippets),
        "duration_s":      duration,
        "prose":           prose,
        "image_generated": image_generated,
        "pinned":          pinned,
    }