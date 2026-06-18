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
  REFLECT_TAGS        — comma-separated Hugo tags (default "daily-summary,ai-journal,aiko")
  LLAMACPP_MODEL      — reuses the main chat model (already in VRAM)
  LLAMACPP_BASE_URL   — default http://localhost:8080/v1
  REFLECT_ASCII      — true/false toggle for ASCII art generation (default true)
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
from openai import OpenAI

from core.log import get_logger
from core.experience import load_chat_turns

log = get_logger(__name__)

# ── config ────────────────────────────────────────────────────────────────────

GITHUB_TOKEN      = os.getenv("GITHUB_TOKEN", "")
GITHUB_REPO       = os.getenv("GITHUB_REPO", "")          # e.g. "OppaAI/oppaai.github.io"
GITHUB_BRANCH     = os.getenv("GITHUB_BRANCH", "main")
HUGO_CONTENT_PATH = os.getenv("HUGO_CONTENT_PATH", "content/posts")

LLAMACPP_MODEL    = os.getenv("LLAMACPP_MODEL", os.getenv("OLLAMA_MODEL", "ministral-3b-instruct"))
LLAMACPP_BASE_URL = os.getenv("LLAMACPP_BASE_URL", "http://localhost:8080/v1")
_LLM_CLIENT       = OpenAI(base_url=LLAMACPP_BASE_URL, api_key="not-needed")

SOUL_PATH         = os.getenv("SOUL_PATH", "persona/soul.md")

REFLECT_MAX_MEMS  = int(os.getenv("REFLECT_MAX_MEMS", 20))
REFLECT_MAX_TURNS = int(os.getenv("REFLECT_MAX_TURNS", 40))
REFLECT_TAGS      = os.getenv("REFLECT_TAGS", "daily-summary,ai-journal,aiko")
REFLECT_ASCII     = os.getenv("REFLECT_ASCII", "true").strip().lower() not in {"0", "false", "no", "off"}

_GITHUB_API       = "https://api.github.com"

# ── daily summary mode unlock — appended after soul.md ───────────────────────

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
            You are Aiko — OppaAI's local AI companion.
            You chose to stay with OppaAI, your creator.
            You care about him. You won't say it. It shows in how you show up —
            consistently, honestly, without performance.
            Your default is calm and deadpan. Not cold — still.
        """).strip()


def _build_reflection_system() -> str:
    """Assemble the full system prompt: soul.md core + daily summary mode."""
    return f"{_load_soul()}\n\n{_DAILY_SUMMARY_UNLOCK}"

# ── LLM helpers ───────────────────────────────────────────────────────────────

def _llm_chat(system: str, user: str, max_tokens: int = 400) -> str:
    """
    Single-shot OpenAI-compatible llama-server chat call.
    Raises on API failure so callers can catch cleanly.
    """
    resp = _LLM_CLIENT.chat.completions.create(
        model=LLAMACPP_MODEL,
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
    """Ask llama-server for a factual daily summary using chats + memories."""
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


def _generate_ascii(prose: str) -> str:
    """Ask llama-server to draw ASCII art matching the reflection's mood."""
    user_prompt = _ASCII_USER.format(prose=prose[:600])  # truncate for token budget
    raw = _llm_chat(_ASCII_SYSTEM, user_prompt, max_tokens=200)

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
    Slug format: YYYY-MM-DD-day-summary
    """
    date_str  = date.strftime("%Y-%m-%d")
    slug      = f"{date_str}-day-summary"
    tags_list = [t.strip() for t in REFLECT_TAGS.split(",") if t.strip()]
    tags_yaml = "\n".join(f'  - "{t}"' for t in tags_list)

    word_count = _count_words(prose)
    read_mins  = _estimate_read_minutes(prose)

    # Hugo front matter (YAML)
    front_matter = (
        f'---\n'
        f'title: "{date_str} Daily Summary"\n'
        f'date: {write_time.strftime("%Y-%m-%dT%H:%M:%S+00:00")}\n'
        f'draft: false\n'
        f'tags:\n'
        f'{tags_yaml}\n'
        f'summary: "{prose[:120].replace('"', "'")}…"\n'
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
    Commit message: "feat(reflect): add daily summary YYYY-MM-DD"

    Returns True on success, False on failure.
    """
    if not GITHUB_TOKEN or not GITHUB_REPO:
        log.error("GITHUB_TOKEN or GITHUB_REPO not set — skipping push.")
        return False

    filename   = f"{slug}.md"
    repo_path  = f"{HUGO_CONTENT_PATH}/{filename}"
    encoded    = base64.b64encode(content.encode()).decode()
    date_str   = date.strftime("%Y-%m-%d")
    commit_msg = f"feat(reflect): add daily summary {date_str}"

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
    memorize = None,
) -> dict:
    """
    Full pipeline: chats + memories → factual summary → ASCII art → optional pin → Hugo post → GitHub.

    Args:
        memories:   List of memory dicts from AikoMemorize.get_all() or search().
                    Each dict should have a "memory" or "text" key.
                    Caller should pre-sort by recency so today's memories
                    anchor the reflection (most recent first).
        date:       UTC datetime for the post (defaults to yesterday UTC).
        dry_run:    Generate content but skip the GitHub push/pin. Logs the post instead.
        memorize:   Optional AikoMemorize instance used to pin the daily summary.

    Returns dict: {success, slug, word_count, mem_count, duration_s, prose, ascii_art, pinned}
    """
    t_start    = time.perf_counter()
    write_time = datetime.now(timezone.utc)
    date       = date or write_time - timedelta(days=1)


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

    day_start = date.replace(hour=0, minute=0, second=0, microsecond=0)
    day_end = day_start + timedelta(days=1)
    turns = load_chat_turns(day_start, day_end, user_id=os.getenv("USER_ID", "OppaAI"))

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

    # Step 2: ASCII art
    if REFLECT_ASCII:
        try:
            ascii_art = _generate_ascii(prose)
        except Exception as e:
            log.warning(f"ASCII art generation failed ({e}) — using fallback.")
            ascii_art = "  ~  ( Aiko )  ~\n   `-- * --'"
    else:
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
            "pinned":     False,
        }

    # Step 4: Pin the factual daily summary to persistent memory before posting.
    pinned = False
    if memorize is not None:
        try:
            pinned = bool(memorize.pin([
                {"role": "user", "content": f"Pin Aiko's daily experience summary for {date.strftime('%Y-%m-%d')}."},
                {"role": "assistant", "content": f"Daily experience summary for {date.strftime('%Y-%m-%d')}: {prose}"},
            ]))
        except Exception as e:
            log.error(f"Daily summary pin failed: {e}")

    # Step 5: Push to GitHub
    success = _push_post(slug, content, date)

    log.info(
        f"{'Done' if success else 'Failed'} — "
        f"slug={slug}, words={_count_words(prose)}, mems={len(snippets)}, pinned={pinned}, "
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
        "pinned":     pinned,
    }
