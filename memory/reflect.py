"""
memory/reflect.py
Aiko's nightly experience-summary writer.

Called after dream() completes at 00:00. Pulls the day's chat turns and
memory snippets, asks the local llama-server to write a factual daily summary, pins that
summary to persistent memory, then pushes a Hugo-format markdown post to
GitHub via the REST API (no local clone needed).

Environment variables required:
  GITHUB_TOKEN        — Personal Access Token with repo write scope
  GITHUB_REPO         — e.g. GitHub page repo address
  GITHUB_BRANCH       — target branch, default "main"
  HUGO_CONTENT_PATH   — path inside repo, default "content/posts"

Optional:
  SOUL_PATH           — path to soul.md (default "config/soul.md")
  REFLECT_MAX_MEMS    — max memory snippets to feed the LLM (default 20)
  REFLECT_TAGS        — comma-separated Hugo tags (default "daily-reflection,ai-journal,aiko")
  LLM_MODEL           — reuses the main chat model (already in VRAM)
  LLM_BASE_URL        — default http://localhost:8080/v1
  IMAGEGEN_URL        — Modal FLUX endpoint
  REFERENCE_IMAGE — path to Aiko reference PNG (default ~/Aiko-chan/assets/Aiko-chan.png)
  USER_REFERENCE_IMAGE — path to user reference PNG (default ~/Aiko-chan/assets/<USER_ID>.png)
  HUGO_IMAGES_PATH    — path inside repo for images, default "static/images"
  USER_STATE_ROOT — root directory for user state (default: ~/.aiko)

Idempotency:
  generate_and_post() pins daily atomic facts to memory.db and the faithful
  day-level blob ("Daily journal of YYYY-MM-DD: ...") to journal.db. The
  journal uses the same encrypted SQLite path as memory and is upserted by
  date, so reruns replace the day blob instead of stacking duplicates.
  Legacy memory.db "Day record for YYYY-MM-DD:" blobs are still cleaned up.

  _delete_existing_daily_pins() now runs immediately before the pin step,
  matching by content prefix (the date string embedded in the memory text)
  rather than created_at (created_at reflects when the job ran, not which
  day the pin describes — so it can't be used to detect "already pinned
  for this date"). Any stale pins for the target date are deleted first,
  so a rerun replaces rather than accumulates.
"""
import base64
import io
import json
import os
import re
import textwrap
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests
from openai import OpenAI

from system.log import get_logger
from system.userspace import current_user_id, current_display_name

log = get_logger(__name__)

# ── config ────────────────────────────────────────────────────────────────────

GITHUB_TOKEN      = os.getenv("GITHUB_TOKEN", "")
GITHUB_REPO       = os.getenv("GITHUB_REPO", "")
GITHUB_BRANCH     = os.getenv("GITHUB_BRANCH", "main")
HUGO_CONTENT_PATH = os.getenv("HUGO_CONTENT_PATH", "content/posts")
HUGO_IMAGES_PATH  = os.getenv("HUGO_IMAGES_PATH", "static/images")

LLM_MODEL    = os.getenv("LLM_MODEL", "ministral")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8080/v1")
_LLM_CLIENT: OpenAI | None = None

def _get_llm_client() -> OpenAI:
    global _LLM_CLIENT
    if _LLM_CLIENT is None:
        _LLM_CLIENT = OpenAI(base_url=LLM_BASE_URL, api_key="not-needed")
    return _LLM_CLIENT

SOUL_PATH         = os.getenv("SOUL_PATH", "persona/soul.md")

REFLECT_MAX_MEMS  = int(os.getenv("REFLECT_MAX_MEMS", 50))
REFLECT_TAGS      = os.getenv("REFLECT_TAGS", "daily-reflection,ai-journal,aiko")
REFLECT_BLOG_POST_ENABLED = os.getenv("REFLECT_BLOG_POST_ENABLED", "1").lower() in {"1", "true", "yes", "on"}

IMAGEGEN_URL          = os.getenv("IMAGEGEN_URL", "")
REFERENCE_IMAGE  = os.getenv("REFERENCE_IMAGE", os.path.expanduser("~/Aiko-chan/assets/Aiko-chan.png"))

def _user_reference_image_path() -> str:
    """Resolve the current user's reference-image path fresh, per call —
    not at import time, so this doesn't depend on reflect.py always being
    imported after login (which is currently true only by accident of
    wakeup.py's import order)."""
    override = os.getenv("USER_REFERENCE_IMAGE")
    if override:
        return override
    return os.path.expanduser(f"~/Aiko-chan/assets/{current_user_id()}.png")

_GITHUB_API = "https://api.github.com"

# ── pin content prefixes ──────────────────────────────────────────────────────
# These prefixes are how pinned daily memories are identified for both
# creation (below) and idempotency-guard deletion (_delete_existing_daily_pins).
# If you change either f-string's wording at the call sites, update these too.

_DAILY_SUMMARY_PREFIX_TMPL = "Daily experience summary for {date_str}:"
_DAY_JOURNAL_PREFIX_TMPL = "Daily journal of {date_str}:"
# Backward-compatible name for legacy cleanup callers.
_DAY_RECORD_PREFIX_TMPL = _DAY_JOURNAL_PREFIX_TMPL

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
    - Keep Aiko's tone calm, direct, lightly dry, and quietly affectionate toward {USER_ID}.

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

_DAILY_FACTS_PROMPT = textwrap.dedent("""
    Rewrite this day's narrative summary as a list of short, atomic factual
    statements about {USER_ID}'s activities, decisions, and events that day.

    Rules:
    - Distill the narrative's real content (projects, bugs, decisions, plans,
      names, deadlines) into plain factual statements — even if the source
      text is written in a stylized or metaphorical voice.
    - Strip flavor language, mood-setting, and metaphor; keep only the
      underlying events and facts.
    - Do not invent details, outcomes, or facts not supported by the narrative
      — only translate what's actually there into plainer language.
    - One fact per line, third person, about {USER_ID}.
    - Each fact must be self-contained and short (readable without the
      day's context).
    - If the narrative genuinely contains no concrete events (pure mood/
      atmosphere with nothing happening), return: []

    Return ONLY a JSON array of short strings. No markdown, no explanation.

    Date: {date_str}

    Narrative:
    {prose}

    Additional raw notes:
    {notes}
""").strip()

def _extract_json_arrays(raw: str) -> list[list]:
    r"""
    Scan raw text for top-level JSON arrays using bracket-depth tracking
    (aware of string quoting/escaping), rather than a regex that assumes
    no '[' or ']' characters appear inside the array's own string content.
    A naive `\[.*?\]` regex truncates early on any fact containing a
    literal bracket (e.g. "[CUDA 13]", file paths, version tags) — common
    in Oppa's technical daily notes. Returns every syntactically complete
    top-level array found, in order of appearance.
    """
    arrays: list[list] = []
    i, n = 0, len(raw)
    while i < n:
        start = raw.find("[", i)
        if start == -1:
            break
        depth = 0
        in_string = False
        escape = False
        end = None
        for j in range(start, n):
            ch = raw[j]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == "[":
                    depth += 1
                elif ch == "]":
                    depth -= 1
                    if depth == 0:
                        end = j
                        break
            i = j
        if end is None:
            # Unterminated array from this start point — likely genuine
            # truncation (max_tokens hit mid-array). Nothing further to
            # scan from here.
            break
        candidate = raw[start:end + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list):
                arrays.append(parsed)
        except json.JSONDecodeError:
            pass
        i = end + 1
    return arrays

def _salvage_truncated_facts(raw: str) -> list[str]:
    """
    Last-resort recovery when an array never closes (true max_tokens
    truncation — confirmed by no closing bracket/quote at all). Pulls out
    every complete "..." string that appears before the cutoff, discarding
    only the partial fragment at the very end. Better to keep 15 clean
    facts than throw away a whole day's extraction over the 16th being
    cut off mid-word.
    """
    strings = re.findall(r'"((?:[^"\\]|\\.)*)"\s*,?', raw)
    return [s.strip() for s in strings if s.strip() and len(s) <= 200]

def _generate_daily_facts(
    prose: str,
    snippets: list[str],
    date: datetime,
    _retry: bool = False,
) -> list[str]:
    notes = "\n".join(f"- {s}" for s in snippets[:REFLECT_MAX_MEMS]) or "- none"
    prompt_template = _DAILY_FACTS_PROMPT
    if _retry:
        prompt_template += (
            "\n\nIMPORTANT: Only return [] if the narrative truly describes "
            "nothing but atmosphere with zero concrete events. If any "
            "activity, decision, bug, plan, or interaction is mentioned — "
            "even briefly or metaphorically — extract at least one fact "
            "from it."
        )
    user_prompt = prompt_template.format(
        date_str=date.strftime("%Y-%m-%d"),
        prose=prose,
        notes=notes,
        USER_ID=current_display_name(),
    )
    raw = _llm_chat(
        system="You are a precise fact-extraction assistant.",
        user=user_prompt,
        max_tokens=1536,
        temperature=0.0,
    )
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()

    # Always log the full raw response at debug level so future parse
    # failures can be diagnosed without guessing from a 300-char preview.
    log.debug(f"Raw daily-facts response for {date.strftime('%Y-%m-%d')}: {raw}")

    arrays = _extract_json_arrays(raw)
    facts: list[str] = []
    for candidate in reversed(arrays):
        if candidate and all(isinstance(f, str) for f in candidate):
            facts = [f.strip() for f in candidate if isinstance(f, str) and f.strip()]
            break

    if not facts and not _retry:
        log.info(f"Empty/unparseable facts for {date.strftime('%Y-%m-%d')} — retrying with stronger prompt.")
        return _generate_daily_facts(prose, snippets, date, _retry=True)

    if not facts:
        salvaged = _salvage_truncated_facts(raw)
        if salvaged:
            log.warning(
                f"Daily-facts array truncated for {date.strftime('%Y-%m-%d')} — "
                f"salvaged {len(salvaged)} complete fact(s) from partial output."
            )
            facts = salvaged
        else:
            log.warning(f"Failed to parse daily-facts JSON after retry: {raw[:600]!r}")
            return []

    facts = [f for f in facts if len(f) <= 200]
    return facts

_IMAGE_PROMPT_SYSTEM = textwrap.dedent("""
    You are Aiko. Given the day's summary of what happened, imagine
    a single vivid anime scene that captures it. Choose the setting,
    lighting, and who appears based on what feels true to the day.
    
    Write a scene prompt (under 60 words) for an anime illustration model.
    Focus on mood, atmosphere, and activity — let the memories guide what
    the image should show.
    
    Return ONLY the prompt text. No explanation, no quotes, no preamble.
""").strip()

_IMAGE_PROMPT_USER = "Daily summary:\n\n{prose}\n\nImagine the scene."

_FEELINGS_SYSTEM = textwrap.dedent("""
    You are Aiko reflecting privately on how you feel about {USER_ID}.
    This is honest and direct — not a performance, not hidden.
    You notice things. You care. Say what's true.
    
    Keep it short (2–4 sentences). First person. Direct and real.
""").strip()

_FEELINGS_USER = "Today's summary:\n\n{prose}\n\nHow do you feel about {USER_ID}, honestly?"

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
    unlock = _DAILY_SUMMARY_UNLOCK.format(USER_ID=current_display_name())
    return f"{_load_soul()}\n\n{unlock}"

# ── LLM helpers ───────────────────────────────────────────────────────────────

def _llm_chat(system: str, user: str, max_tokens: int = 400, temperature: float = 0.75) -> str:
    resp = _get_llm_client().chat.completions.create(
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
    user_id = current_display_name()
    system = f"{_load_soul()}\n\n{_FEELINGS_SYSTEM.format(USER_ID=user_id)}"
    user_prompt = _FEELINGS_USER.format(prose=prose[:600], USER_ID=user_id)
    return _llm_chat(system, user_prompt, max_tokens=1024, temperature=0.8)

# ── image generation ──────────────────────────────────────────────────────────

def _load_reference_images() -> list[str]:
    """Load Aiko and user reference images as base64 strings."""
    refs = []
    for path in [REFERENCE_IMAGE, _user_reference_image_path()]:
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
            "negative_prompt": "extra limbs, deformed, poorly drawn, bad anatomy, malformed hands",
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

# ── faithful daily journal (non-LLM, permanent) ────────────────────────────────

def build_daily_journal(snippets: list[str], date: datetime) -> str:
    """
    Build a faithful, non-LLM journal of one day's deduplicated memory facts.
    No paraphrasing, no invention — verbatim facts in chronological order.
    Meant to be pinned forever in journal.db as ground truth, separate
    from the stylized prose reflection.
    """
    date_str = date.strftime("%Y-%m-%d")
    if not snippets:
        return f"Daily journal of {date_str}: no memories recorded."
    header = f"Daily journal of {date_str}:"
    return header + "\n" + "\n".join(f"- {s}" for s in snippets)

# ── idempotency guard ─────────────────────────────────────────────────────────

def _delete_existing_daily_pins(memorize, date: datetime, user_id: str | None = None) -> int:
    date_str = date.strftime("%Y-%m-%d")
    date_tag = f"[{date_str}]"
    day_record_prefix = _DAY_JOURNAL_PREFIX_TMPL.format(date_str=date_str)

    try:
        all_mems = memorize.get_all(user_id=user_id)
    except Exception as e:
        log.warning(f"Could not fetch existing memories for date-dedup ({date_str}): {e}")
        return 0

    deleted = 0
    for m in all_mems:
        text = m.get("memory") or ""
        # Matches memory.db pin shapes for this date: atomic facts tagged
        # "[YYYY-MM-DD] ..." and legacy day-record/journal blobs. New daily
        # journal blobs live in journal.db and are replaced by date there.
        if not (text.startswith(date_tag) or text.startswith(day_record_prefix) or text.startswith(f"Day record for {date_str}:")):
            continue
        mem_id = m.get("id")
        if not mem_id:
            continue
        try:
            memorize.delete(mem_id)
            deleted += 1
        except Exception as e:
            log.warning(f"Failed to delete stale daily pin {mem_id}: {e}")

    if deleted:
        log.info(f"Removed {deleted} stale pinned fact(s) for {date_str} before re-pinning.")
    return deleted

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

    Idempotent per date: if pinned entries already exist for this date
    (from a prior run), they are deleted before the new ones are pinned,
    so reruns replace rather than accumulate. See
    _delete_existing_daily_pins() for why this can't be done via created_at.

    Returns dict: {success, slug, word_count, mem_count, duration_s, prose, feelings, image_generated, pinned, journal_pinned}
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
        if not text or text in seen:
            continue
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
            "journal_pinned":  False,
        }

    # Resolve the real user_id once, from the memory instance itself —
    # not from ambient contextvar/env fallback (this job runs on the
    # scheduler's own background thread, which doesn't inherit either).
    uid = memorize.get_user_id() if memorize is not None else current_user_id()

    # Step 3b: idempotency guard — remove any stale pins for this date
    # before pinning fresh ones, so reruns replace rather than accumulate.
    if memorize is not None:
        _delete_existing_daily_pins(memorize, date, user_id=uid)

    # Step 4: extract atomic facts and pin each individually. Replaces the
    # old single-block pin (whole prose paragraph or bullet-list day-record)
    # — those blew the context budget on recall since format_for_context()
    # truncates per-fact, not per-block.
    date_str = date.strftime("%Y-%m-%d")
    date_tag = f"[{date_str}]"
    pinned_count = 0
    if memorize is not None:
        try:
            facts = _generate_daily_facts(prose, snippets, date)
        except Exception as e:
            log.error(f"Daily fact extraction failed: {e}")
            facts = []
        for fact in facts:
            try:
                if memorize.add_raw(f"{date_tag} {fact}", user_id=uid, pinned=True):
                    pinned_count += 1
            except Exception as e:
                log.warning(f"Failed to pin fact {fact!r}: {e}")

    pinned = pinned_count > 0

    # Step 4b: pin the faithful fact-list daily journal in journal.db — ground
    # truth, separate from both atomic memory facts and stylized prose. Never
    # paraphrased, never invented.
    journal_pinned = False
    try:
        from memory.journal import pin_daily_journal
        daily_journal = build_daily_journal(snippets, date)
        journal_pinned = bool(pin_daily_journal(daily_journal, date, user_id=uid))
    except Exception as e:
        log.error(f"Daily journal pin failed: {e}")

    # Step 5: optionally push image and Hugo post together. Daily memory
    # pins still happen even when public blog posting is disabled.
    if REFLECT_BLOG_POST_ENABLED:
        success = _push_post_and_image(slug, content, image_b64 if image_generated else None, date)
    else:
        log.info("Daily reflection blog posting disabled; skipping GitHub push for %s.", slug)
        success = True

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
        "feelings":        feelings,
        "image_generated": image_generated,
        "pinned":          pinned,
        "journal_pinned":  journal_pinned,
    }
