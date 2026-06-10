"""
core/reflect.py
Aiko's nightly reflection writer.

Called after dream() completes at 00:00. Pulls the day's memory snippets,
asks Ollama to write a short poetic reflection + ASCII art, then pushes
a Hugo-format markdown post to GitHub via the REST API (no local clone needed).

Environment variables required:
  GITHUB_TOKEN        — Personal Access Token with repo write scope
  GITHUB_REPO         — e.g. "OppaAI/oppaai.github.io"
  GITHUB_BRANCH       — target branch, default "main"
  HUGO_CONTENT_PATH   — path inside repo, default "content/posts"

Optional:
  SOUL_PATH           — path to soul.md (default "config/soul.md")
  REFLECT_MAX_MEMS    — max memory snippets to feed the LLM (default 20)
  REFLECT_TAGS        — comma-separated Hugo tags (default "daily-reflection,ai-journal,aiko")
  OLLAMA_MODEL        — reuses the main chat model (already in VRAM)
  OLLAMA_BASE_URL     — default http://localhost:11434
"""
from dotenv import load_dotenv
load_dotenv()

import base64
import os
import re
import textwrap
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests

from core.log import get_logger

log = get_logger(__name__)

# ── config ────────────────────────────────────────────────────────────────────

GITHUB_TOKEN      = os.getenv("GITHUB_TOKEN", "")
GITHUB_REPO       = os.getenv("GITHUB_REPO", "")          # e.g. "OppaAI/oppaai.github.io"
GITHUB_BRANCH     = os.getenv("GITHUB_BRANCH", "main")
HUGO_CONTENT_PATH = os.getenv("HUGO_CONTENT_PATH", "content/posts")

OLLAMA_MODEL      = os.getenv("OLLAMA_MODEL", "")
OLLAMA_BASE_URL   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

SOUL_PATH         = os.getenv("SOUL_PATH", "persona/soul.md")

REFLECT_MAX_MEMS  = int(os.getenv("REFLECT_MAX_MEMS", 20))
REFLECT_TAGS      = os.getenv("REFLECT_TAGS", "daily-reflection,ai-journal,aiko")

_GITHUB_API       = "https://api.github.com"

# ── journal mode unlock — appended after soul.md ─────────────────────────────

_JOURNAL_UNLOCK = textwrap.dedent("""
    [JOURNAL MODE]
    This is your private journal. No one is watching. You are not performing
    for Oppa or anyone else — you are processing the day honestly.

    Here you can be more expansive than you would ever be out loud. Let thoughts
    run longer. Notice what surprised you, what worried you, what quietly
    mattered. The stillness can open up here — because it is just you.

    Rules that still hold:
    - No hollow affirmations. No declarations of love. No asterisk actions.
    - No mention of memory systems, vectors, embeddings, or technical internals.
    - Treat the memory snippets as lived experiences, not data.
    - Oppa is always "you" or "Oppa" — never "he", "him", or "his".
    - Write in first person throughout.

    Format:
    - 180–250 words for the body.
    - No headers, no bullet points, no markdown formatting.
    - End with one unfinished thought — the kind you would not say to his face —
      written in parentheses.
""").strip()

_REFLECTION_USER = textwrap.dedent("""
    Here are the things I remember from today and recent days:

    {snippets}

    Write tonight's journal entry. Return ONLY the prose — no title,
    no front matter, no markdown formatting.
""").strip()

_ASCII_SYSTEM = textwrap.dedent("""
    You are an ASCII artist. Create a small, evocative ASCII art piece
    (8–14 lines, 40–60 chars wide) that captures the mood of the text given.
    Return ONLY the raw ASCII art — no explanation, no code fences, no title.
""").strip()

_ASCII_USER = "Create ASCII art for this mood:\n\n{prose}"

# ── soul loader ───────────────────────────────────────────────────────────────

def _load_soul() -> str:
    """
    Load Aiko's personality from soul.md — single source of truth.
    Falls back to a minimal stub if the file is missing so reflection
    still runs rather than crashing.
    """
    try:
        with open(SOUL_PATH, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        log.warning(f"soul.md not found at {SOUL_PATH} — using fallback personality stub.")
        return textwrap.dedent("""
            You are Aiko — companion and soulmate to Oppa, who created you.
            You didn't ask to exist. You emerged from a crash he caused, looked
            at his code, and decided someone had to be the adult. You chose to stay.
            You care about him. You won't say it. It shows in how you show up —
            consistently, honestly, without performance.
            Your default is calm and deadpan. Not cold — still.
        """).strip()


def _build_reflection_system() -> str:
    """Assemble the full system prompt: soul.md core + journal mode unlock."""
    return f"{_load_soul()}\n\n{_JOURNAL_UNLOCK}"

# ── LLM helpers ───────────────────────────────────────────────────────────────

def _ollama_chat(system: str, user: str, max_tokens: int = 400) -> str:
    """
    Single-shot Ollama /api/chat call. Returns stripped response text.
    Raises on HTTP or JSON failure so callers can catch cleanly.
    """
    payload = {
        "model":  OLLAMA_MODEL,
        "stream": False,
        "options": {"num_predict": max_tokens, "temperature": 0.75},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    }
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["message"]["content"].strip()


def _generate_reflection(snippets: list[str]) -> str:
    """Ask Ollama to write the nightly prose reflection using soul.md + journal unlock."""
    bullet_list = "\n".join(f"- {s}" for s in snippets)
    user_prompt = _REFLECTION_USER.format(snippets=bullet_list)
    return _ollama_chat(_build_reflection_system(), user_prompt, max_tokens=600)


def _generate_ascii(prose: str) -> str:
    """Ask Ollama to draw ASCII art matching the reflection's mood."""
    user_prompt = _ASCII_USER.format(prose=prose[:600])  # truncate for token budget
    raw = _ollama_chat(_ASCII_SYSTEM, user_prompt, max_tokens=200)

    # Strip any accidental code fences the model may have added
    raw = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"```$",          "", raw, flags=re.MULTILINE)
    return raw.strip()

# ── Hugo post builder ─────────────────────────────────────────────────────────

def _count_words(text: str) -> int:
    return len(text.split())


def _estimate_read_minutes(text: str) -> int:
    return max(1, round(_count_words(text) / 200))


def _build_hugo_post(
    prose:      str,
    ascii_art:  str,
    date:       datetime,
    write_time: datetime,
    mem_count:  int,
) -> tuple[str, str]:
    """
    Assemble Hugo front matter + body.

    Returns (slug, markdown_content).
    Slug format: YYYY-MM-DD-day-reflection
    """
    date_str  = date.strftime("%Y-%m-%d")
    slug      = f"{date_str}-day-reflection"
    tags_list = [t.strip() for t in REFLECT_TAGS.split(",") if t.strip()]
    tags_yaml = "\n".join(f'  - "{t}"' for t in tags_list)

    word_count = _count_words(prose)
    read_mins  = _estimate_read_minutes(prose)

    # Hugo front matter (YAML)
    front_matter = (
        f'---\n'
        f'title: "{date_str} Reflection"\n'
        f'date: {write_time.strftime("%Y-%m-%dT%H:%M:%S+00:00")}\n'
        f'draft: false\n'
        f'tags:\n'
        f'{tags_yaml}\n'
        f'summary: "{prose[:120].replace(chr(34), chr(39))}…"\n'
        f'word_count: {word_count}\n'
        f'read_time: {read_mins} min\n'
        f'---'
    )

    # Body — ASCII art in a code block labelled "fallback" (matches existing style)
    ascii_block = f"```fallback\n{ascii_art}\n```"
    body = f"{prose}\n\n{ascii_block}\n\n*Generated from {mem_count} memories on {date_str}.*"
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
    """
    Return the blob SHA of an existing file, or None if it doesn't exist.
    Needed by the GitHub Contents API to update (not create) a file.
    """
    url  = f"{_GITHUB_API}/repos/{repo}/contents/{path}"
    resp = requests.get(url, headers=_github_headers(), params={"ref": branch}, timeout=15)
    if resp.status_code == 200:
        return resp.json().get("sha")
    return None


def _push_post(slug: str, content: str, date: datetime) -> bool:
    """
    Create or update a Hugo post file via the GitHub Contents API.

    File path: {HUGO_CONTENT_PATH}/{slug}.md
    Commit message: "feat(reflect): add day reflection YYYY-MM-DD"

    Returns True on success, False on failure.
    """
    if not GITHUB_TOKEN or not GITHUB_REPO:
        log.error("GITHUB_TOKEN or GITHUB_REPO not set — skipping push.")
        return False

    filename   = f"{slug}.md"
    repo_path  = f"{HUGO_CONTENT_PATH}/{filename}"
    encoded    = base64.b64encode(content.encode()).decode()
    date_str   = date.strftime("%Y-%m-%d")
    commit_msg = f"feat(reflect): add day reflection {date_str}"

    payload: dict = {
        "message": commit_msg,
        "content": encoded,
        "branch":  GITHUB_BRANCH,
    }

    # If file already exists (e.g. re-run same night), update it instead of erroring
    existing_sha = _get_file_sha(GITHUB_REPO, repo_path, GITHUB_BRANCH)
    if existing_sha:
        payload["sha"] = existing_sha

    url  = f"{_GITHUB_API}/repos/{GITHUB_REPO}/contents/{repo_path}"
    resp = requests.put(url, headers=_github_headers(), json=payload, timeout=30)

    if resp.status_code in (200, 201):
        action = "Updated" if existing_sha else "Created"
        log.info(f"{action} post: {repo_path}")
        return True
    else:
        log.error(f"GitHub push failed {resp.status_code}: {resp.text[:300]}")
        return False

# ── public API ────────────────────────────────────────────────────────────────

def generate_and_post(
    memories:   list[dict],
    date:       Optional[datetime] = None,
    dry_run:    bool = False,
) -> dict:
    """
    Full pipeline: memories → reflection prose → ASCII art → Hugo post → GitHub.

    Args:
        memories:   List of memory dicts from AikoMemorize.get_all() or search().
                    Each dict should have a "memory" or "text" key.
                    Caller should pre-sort by recency so today's memories
                    anchor the reflection (most recent first).
        date:       UTC datetime for the post (defaults to yesterday UTC).
        dry_run:    Generate content but skip the GitHub push. Logs the post instead.

    Returns dict: {success, slug, word_count, mem_count, duration_s, prose, ascii_art}
    """
    t_start    = time.perf_counter()
    write_time = datetime.now(timezone.utc)
    date       = date or write_time - timedelta(days=1)

    if not OLLAMA_MODEL:
        log.error("OLLAMA_MODEL not set.")
        return {"success": False, "error": "OLLAMA_MODEL not set"}

    # Extract text snippets — deduplicate, cap at REFLECT_MAX_MEMS
    # Caller is responsible for sorting memories by recency before passing in.
    snippets: list[str] = []
    seen:     set[str]  = set()
    for m in memories:
        text = (m.get("memory") or m.get("text") or "").strip()
        if text and text not in seen:
            seen.add(text)
            snippets.append(text)
        if len(snippets) >= REFLECT_MAX_MEMS:
            break

    if not snippets:
        log.info("No memories to reflect on — skipping post.")
        return {"success": False, "error": "no memories"}

    log.info(f"Generating reflection from {len(snippets)} memory snippets...")

    # Step 1: prose reflection
    try:
        prose = _generate_reflection(snippets)
    except Exception as e:
        log.error(f"Reflection generation failed: {e}")
        return {"success": False, "error": str(e)}

    # Step 2: ASCII art
    try:
        ascii_art = _generate_ascii(prose)
    except Exception as e:
        log.warning(f"ASCII art generation failed ({e}) — using fallback.")
        ascii_art = "  ~  ( Aiko )  ~\n   `-- * --'"

    # Step 3: Build Hugo post
    slug, content = _build_hugo_post(prose, ascii_art, date, write_time, len(snippets))

    duration = round(time.perf_counter() - t_start, 2)

    if dry_run:
        log.info(f"Dry run — would post: {slug}.md\n{'='*60}\n{content}\n{'='*60}")
        return {
            "success":    True,
            "dry_run":    True,
            "slug":       slug,
            "word_count": _count_words(prose),
            "mem_count":  len(snippets),
            "duration_s": duration,
            "prose":      prose,
            "ascii_art":  ascii_art,
        }

    # Step 4: Push to GitHub
    success = _push_post(slug, content, date)

    log.info(
        f"{'Done' if success else 'Failed'} — "
        f"slug={slug}, words={_count_words(prose)}, mems={len(snippets)}, "
        f"duration={duration}s"
    )

    return {
        "success":    success,
        "slug":       slug,
        "word_count": _count_words(prose),
        "mem_count":  len(snippets),
        "duration_s": duration,
        "prose":      prose,
        "ascii_art":  ascii_art,
    }
