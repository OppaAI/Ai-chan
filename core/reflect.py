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
  REFLECT_TAGS        — comma-separated Hugo tags (default "daily-reflection,ai-journal,aiko")
  LLM_MODEL           — reuses the main chat model (already in VRAM)
  LLM_BASE_URL        — default http://localhost:8080/v1
  IMAGEGEN_URL        — Modal FLUX endpoint, default https://oppa-ai-org--aiko-imagegen-fastapi-app.modal.run
  REFERENCE_IMAGE — path to Aiko reference PNG (default ~/Aiko-chan/assets/Aiko-chan.png)
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

REFLECT_MAX_MEMS  = int(os.getenv("REFLECT_MAX_MEMS", 50))
REFLECT_TAGS      = os.getenv("REFLECT_TAGS", "daily-reflection,ai-journal,aiko")

IMAGEGEN_URL          = os.getenv("IMAGEGEN_URL", "https://oppa-ai-org--aiko-imagegen-fastapi-app.modal.run")
REFERENCE_IMAGE  = os.getenv("REFERENCE_IMAGE", os.path.expanduser("~/Aiko-chan/assets/Aiko-chan.png"))
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

    Persistent memory snippets from that day and recent context:
    {snippets}

    Write the factual daily experience summary. Return ONLY the prose — no
    title, no front matter, no markdown formatting.
""").strip()

_IMAGE_PROMPT_SYSTEM = textwrap.dedent("""
    You are Aiko imagining a scene from the day.
    Write a vivid scene prompt (under 60 words) for anime illustration.
    
    Include specific outfits matching the day's mood (NOT from reference images).
    Focus on setting, lighting, activity, and what people are wearing.
    
    Return ONLY the prompt text.
""").strip()

_IMAGE_PROMPT_USER = "Daily summary:\n\n{prose}\n\nImagine the scene. What do you both wear?"

_FEELINGS_SYSTEM = textwrap.dedent("""
    You are Aiko reflecting privately on how you feel about OppaAI.
    This is honest and direct — not a performance, not hidden.
    You notice things. You care. Say what's true.
    
    Keep it short (2–4 sentences). First person. Direct and real.
""").strip()

_FEELINGS_USER = "Today's summary:\n\n{prose}\n\nHow do you feel about OppaAI, honestly?"

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

def _llm_chat(system: str, user: str, max_tokens: int = 400, temperature: float = 0.75) -> str:
    resp = _LLM_CLIENT.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        stream=False,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=120,
    )
    return (resp.choices[0].message.content or "").strip()


def _generate_reflection(snippets: list[str], date: datetime) -> str:
    bullet_list = "\n".join(f"- {s}" for s in snippets) or "- No memory snippets available."
    user_prompt = _REFLECTION_USER.format(
        date_str=date.strftime("%Y-%m-%d"),
        snippets=bullet_list,
    )
    return _llm_chat(_build_reflection_system(), user_prompt, max_tokens=500, temperature=0.85)


def _generate_feelings(prose: str) -> str:
    """
    Ask Aiko to reflect honestly on how she feels about OppaAI,
    based on the day's summary.
    """
    system = f"{_load_soul()}\n\n{_FEELINGS_SYSTEM}"
    user_prompt = _FEELINGS_USER.format(prose=prose[:600])
    return _llm_chat(system, user_prompt, max_tokens=1024, temperature=0.8)

# ── image generation ──────────────────────────────────────────────────────────

def _load_reference_images() -> list[str]:
    """Load Aiko and user reference images as base64 strings."""
    refs = []
    for path in [REFERENCE_IMAGE, USER_REFERENCE_IMAGE]:
        if path and os.path.exists(path):
            with open(path, "rb") as f:
                refs.append(base64.b64encode(f.read()).decode())
            log.info(f"Loaded reference image: {path}")
        else:
            log.warning(f"Reference image not found, skipping: {path}")
    return refs


def _generate_image_prompt(prose: str) -> str:
    """Ask Aiko to imagine a scene from the daily summary."""
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
                "anime illustration, manga style, clean lineart, flat color, "
                "no text, no speech bubbles"
            ),
            "negative_prompt": "identical clothing to reference, same outfit",
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
    feelings:   Optional[str],
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

    body = prose
    if feelings:
        body += f"\n\n*How I feel:*\n\n{feelings}"
    
    body += f"\n\n*Generated from {mem_count} memories on {date_str}.*"
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


def _push_post_and_image(
    slug: str,
    content: str,
    image_b64: Optional[str],
    date: datetime,
) -> bool:
    if not GITHUB_TOKEN or not GITHUB_REPO:
        log.error("GITHUB_TOKEN or GITHUB_REPO not set — skipping push.")
        return False

    headers = _github_headers()
    base = f"{_GITHUB_API}/repos/{GITHUB_REPO}"

    # 1. Get current HEAD SHA
    ref_resp = requests.get(f"{base}/git/ref/heads/{GITHUB_BRANCH}", headers=headers, timeout=15)
    ref_resp.raise_for_status()
    head_sha = ref_resp.json()["object"]["sha"]

    # 2. Get base tree SHA
    commit_resp = requests.get(f"{base}/git/commits/{head_sha}", headers=headers, timeout=15)
    commit_resp.raise_for_status()
    base_tree_sha = commit_resp.json()["tree"]["sha"]

    # 3. Build tree entries
    tree = []

    # Markdown post (text blob)
    tree.append({
        "path": f"{HUGO_CONTENT_PATH}/{slug}.md",
        "mode": "100644",
        "type": "blob",
        "content": content,  # raw string, GitHub encodes it
    })

    # Image (binary blob — must pre-create blob)
    if image_b64:
        blob_resp = requests.post(
            f"{base}/git/blobs",
            headers=headers,
            json={"content": image_b64, "encoding": "base64"},
            timeout=30,
        )
        blob_resp.raise_for_status()
        image_blob_sha = blob_resp.json()["sha"]
        tree.append({
            "path": f"{HUGO_IMAGES_PATH}/{slug}.png",
            "mode": "100644",
            "type": "blob",
            "sha": image_blob_sha,
        })

    # 4. Create tree
    tree_resp = requests.post(
        f"{base}/git/trees",
        headers=headers,
        json={"base_tree": base_tree_sha, "tree": tree},
        timeout=30,
    )
    tree_resp.raise_for_status()
    new_tree_sha = tree_resp.json()["sha"]

    # 5. Create commit
    commit_msg = f"feat(reflect): daily reflection {date.strftime('%Y-%m-%d')}"
    new_commit_resp = requests.post(
        f"{base}/git/commits",
        headers=headers,
        json={"message": commit_msg, "tree": new_tree_sha, "parents": [head_sha]},
        timeout=30,
    )
    new_commit_resp.raise_for_status()
    new_commit_sha = new_commit_resp.json()["sha"]

    # 6. Update branch ref
    update_resp = requests.patch(
        f"{base}/git/refs/heads/{GITHUB_BRANCH}",
        headers=headers,
        json={"sha": new_commit_sha},
        timeout=15,
    )
    if update_resp.status_code in (200, 201):
        log.info(f"Pushed single commit: {slug} + image → {GITHUB_BRANCH}")
        return True
    else:
        log.error(f"Ref update failed {update_resp.status_code}: {update_resp.text[:300]}")
        return False

# ── faithful daily record (non-LLM, permanent) ────────────────────────────────

def build_daily_record(snippets: list[str], date: datetime) -> str:
    """
    Build a faithful, non-LLM record of one day's deduplicated memory facts.
    No paraphrasing, no invention — verbatim facts in chronological order.
    Meant to be pinned forever as ground truth, separate from the stylized
    prose reflection.
    """
    date_str = date.strftime("%Y-%m-%d")
    if not snippets:
        return f"Day record for {date_str}: no memories recorded."
    header = f"Day record for {date_str}:"
    return header + "\n" + "\n".join(f"- {s}" for s in snippets)

# ── public API ────────────────────────────────────────────────────────────────

def generate_and_post(
    memories:   list[dict],
    date:       Optional[datetime] = None,
    dry_run:    bool = False,
    memorize = None,
) -> dict:
    """
    Full pipeline:
      chats + memories → factual summary → Aiko's feelings
      → scene prompt → FLUX image → pin to memory → Hugo post + image → GitHub

    Args:
        memories:   List of memory dicts from AikoMemorize.get_all() or search().
        date:       UTC datetime for the post (defaults to yesterday UTC).
        dry_run:    Generate content but skip GitHub push/pin. Logs output instead.
        memorize:   Optional AikoMemorize instance used to pin the daily summary.

    Returns dict: {success, slug, word_count, mem_count, duration_s, prose, feelings, image_generated, pinned, record_pinned}
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

    log.info(
        f"Generating daily summary from {len(snippets)} memory snippets..."
    )

    # Step 1: factual prose summary
    try:
        prose = _generate_reflection(snippets, date)
    except Exception as e:
        log.error(f"Reflection generation failed: {e}")
        return {"success": False, "error": str(e)}

    # Step 1b: Aiko's feelings about you
    feelings = None
    try:
        feelings = _generate_feelings(prose)
        log.info(f"Feelings generated: {feelings[:80]}...")
    except Exception as e:
        log.warning(f"Feelings generation failed: {e}")

    # Step 2: generate image via Modal FLUX endpoint
    image_b64 = _generate_image(prose)
    image_generated = image_b64 is not None
    slug = date.strftime("%Y-%m-%d") + "-day-reflection"

    # Step 3: build Hugo post (with or without image)
    _, content = _build_hugo_post(
        prose=prose,
        feelings=feelings,
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
            "feelings":        feelings,
            "image_generated": image_generated,
            "pinned":          False,
            "record_pinned":   False,
        }

    # Step 4: pin daily summary to persistent memory. Store a raw curated
    # summary with a stable prefix so monthly consolidation can replace older
    # daily summaries without relying on another extraction pass.
    pinned = False
    if memorize is not None:
        try:
            daily_text = f"Daily experience summary for {date.strftime('%Y-%m-%d')}: {prose}"
            pinned = bool(memorize.add_raw(daily_text, pinned=True))
        except Exception as e:
            log.error(f"Daily summary pin failed: {e}")

    # Step 4b: pin the faithful fact-list day record — ground truth, separate
    # from the stylized prose above. Never paraphrased, never invented.
    record_pinned = False
    if memorize is not None:
        try:
            day_record = build_daily_record(snippets, date)
            record_pinned = bool(memorize.add_raw(day_record, pinned=True))
        except Exception as e:
            log.error(f"Daily record pin failed: {e}")

    # Step 5: push image and Hugo post together
    success = _push_post_and_image(slug, content, image_b64 if image_generated else None, date)

    log.info(
        f"{'Done' if success else 'Failed'} — "
        f"slug={slug}, words={_count_words(prose)}, mems={len(snippets)}, "
        f"image={image_generated}, pinned={pinned}, record_pinned={record_pinned}, duration={duration}s"
    )

    return {
        "success":         success,
        "slug":            slug,
        "word_count":      _count_words(prose),
        "mem_count":       len(snippets),
        "duration_s":      duration,
        "prose":           prose,
        "feelings":        feelings,
        "image_generated": image_generated,
        "pinned":          pinned,
        "record_pinned":   record_pinned,
    }