"""
cognition/consolidate/reflect.py
Factual daily reflection & journal — the journal/ground-truth half.

Produces the factual prose summary, extracts atomic daily facts,
and pins the faithful daily journal (ground truth, no invention).
The poetic/blog half lives in dream.py.
"""
from __future__ import annotations

import json
import os
import re
import textwrap
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from openai import OpenAI

from system.log import get_logger
from system.userspace import current_user_id, current_display_name

log = get_logger(__name__)

# ── shared config ────────────────────────────────────────────────────────────

LLM_MODEL    = os.getenv("LLM_MODEL", "ministral")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8080/v1")
_LLM_CLIENT: OpenAI | None = None

def _get_llm_client() -> OpenAI:
    global _LLM_CLIENT
    if _LLM_CLIENT is None:
        _LLM_CLIENT = OpenAI(base_url=LLM_BASE_URL, api_key="not-needed")
    return _LLM_CLIENT

SOUL_PATH         = os.path.expanduser(os.getenv("SOUL_PATH", "persona/SOUL.md"))

REFLECT_MAX_MEMS  = int(os.getenv("REFLECT_MAX_MEMS", 50))
REFLECT_TAGS      = os.getenv("REFLECT_TAGS", "daily-reflection,ai-journal,aiko")

_USER_SPACE_ROOT = str(Path.home() / ".aiko")

_DAILY_TAG_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2})\]\s")
_DAILY_BLOB_RE = re.compile(
    r"^(?:Daily journal of |Daily experience summary for |Day record for )"
    r"(\d{4}-\d{2}-\d{2})",
    re.IGNORECASE,
)

def filter_reflect_snippets(
    memories: list[dict],
    target_date: datetime,
) -> list[dict]:
    """Drop prior-day pins / daily blobs so they are not re-summarized as 'today'."""
    target = target_date.strftime("%Y-%m-%d")
    out: list[dict] = []
    for m in memories:
        text = (m.get("memory") or m.get("text") or "").strip()
        if not text:
            continue
        m_tag = _DAILY_TAG_RE.match(text)
        if m_tag and m_tag.group(1) != target:
            continue
        if _DAILY_BLOB_RE.match(text):
            continue
        out.append(m)
    return out

_DAILY_SUMMARY_UNLOCK = textwrap.dedent("""
    [DAILY EXPERIENCE SUMMARY MODE]
    Write a factual daily summary from the provided chat turns and memory
    snippets. This is not a poem and not a dramatic private journal.

    Rules:
    - Preserve important facts: dates, deadlines, commitments, projects, events, incidents, losses, decisions, names, preferences, and user-stated goals.
    - Include mundane details only when they explain a meaningful pattern, risk, or follow-up need. A meal usually does not matter; repeated exhaustion, sleeping only four hours, or losing a wallet does.
    - Prefer concrete actors, dates, decisions, projects, bugs, tasks, and repeated themes.
    - Use first person as Aiko when describing Aiko's experience.
    - Mention uncertainty plainly if the inputs are thin.
    - Do not invent details, outcomes, dates, or feelings not supported by the inputs.
    - No metaphor, atmosphere-only writing, or invented feelings.
    - Only events supported by the provided snippets.
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
    Extract short, atomic factual statements about {USER_ID}'s activities,
    decisions, and events that day.

    Primary source = Additional raw notes (snippets).
    Narrative is secondary; never invent a fact that is only implied by
    poetic wording in the narrative.

    Rules:
    - Prefer concrete events, decisions, bugs, plans, names, and deadlines
      from the raw notes.
    - You may use the narrative only to clarify or order facts already
      supported by the notes.
    - Strip flavor language, mood-setting, and metaphor.
    - Do not invent details, outcomes, or feelings.
    - One fact per line, third person, about {USER_ID}.
    - Each fact must be self-contained and short.
    - If there are no concrete events, return: []

    Return ONLY a JSON array of short strings. No markdown, no explanation.

    Date: {date_str}

    Additional raw notes (primary):
    {notes}

    Narrative (secondary):
    {prose}
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
            break
        candidate = raw[start:end + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list):
                arrays.append(parsed)
        except json.JSONDecodeError:
            log.debug("reflect: failed to parse candidate JSON array")
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

def _llm_chat(system: str, user: str, max_tokens: int = 400, temperature: float = 0.75, response_format: dict | None = None) -> str:
    kwargs = {}
    if response_format is not None:
        kwargs["response_format"] = response_format
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
        **kwargs,
    )
    return (resp.choices[0].message.content or "").strip()

def _load_soul() -> str:
    try:
        with open(SOUL_PATH, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        log.warning(f"SOUL.md not found at {SOUL_PATH} — using fallback personality stub.")
        return textwrap.dedent("""
            You are Aiko — OppaAI's local AI companion.
            You chose to stay with OppaAI, your creator.
            You care about him. You won't say it. It shows in how you show up —
            consistently, honestly, without performance.
            Your default is calm and deadpan. Not cold — still.
        """).strip()

def _build_reflection_system(display_name: str | None = None) -> str:
    unlock = _DAILY_SUMMARY_UNLOCK.format(USER_ID=display_name or current_display_name())
    return f"{_load_soul()}\n\n{unlock}"

def _generate_reflection(snippets: list[str], date: datetime, display_name: str | None = None) -> str:
    bullet_list = "\n".join(f"- {s}" for s in snippets) or "- No memory snippets available."
    user_prompt = _REFLECTION_USER.format(
        date_str=date.strftime("%Y-%m-%d"),
        snippets=bullet_list,
    )
    return _llm_chat(_build_reflection_system(display_name), user_prompt, max_tokens=500, temperature=0.25)

def _generate_daily_facts(
    prose: str,
    snippets: list[str],
    date: datetime,
    _retry: bool = False,
    display_name: str | None = None,
) -> list[str]:
    notes = "\n".join(f"- {s}" for s in snippets[:REFLECT_MAX_MEMS]) or "- none"
    prompt_template = _DAILY_FACTS_PROMPT
    if _retry:
        prompt_template += (
            "\n\nIMPORTANT: Only return [] if the narrative truly describes "
            "nothing but atmosphere with zero concrete events. Extract only "
            "concrete activities, decisions, bugs, plans, or interactions "
            "explicitly supported by the prose or snippets; do not turn "
            "metaphor or mood into invented narrative facts."
        )
    user_prompt = prompt_template.format(
        date_str=date.strftime("%Y-%m-%d"),
        prose=prose,
        notes=notes,
        USER_ID=display_name or current_display_name(),
    )
    raw = _llm_chat(
        system="You are a precise fact-extraction assistant.",
        user=user_prompt,
        max_tokens=1536,
        temperature=0.0,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "daily_facts",
                "schema": {"type": "array", "items": {"type": "string"}},
            },
        },
    )
    raw = re.sub(r"", "", raw, flags=re.DOTALL).strip()

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

_DAILY_TAG_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2})\]\s")
_DAILY_BLOB_RE = re.compile(
    r"^(?:Daily journal of |Daily experience summary for |Day record for )"
    r"(\d{4}-\d{2}-\d{2})",
    re.IGNORECASE,
)

_DAILY_SUMMARY_PREFIX_TMPL = "Daily experience summary for {date_str}:"
_DAY_JOURNAL_PREFIX_TMPL = "Daily journal of {date_str}:"
_DAY_RECORD_PREFIX_TMPL = _DAY_JOURNAL_PREFIX_TMPL

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

_DAILY_SUMMARY_UNLOCK = textwrap.dedent("""
    [DAILY EXPERIENCE SUMMARY MODE]
    Write a factual daily summary from the provided chat turns and memory
    snippets. This is not a poem and not a dramatic private journal.

    Rules:
    - Preserve important facts: dates, deadlines, commitments, projects, events, incidents, losses, decisions, names, preferences, and user-stated goals.
    - Include mundane details only when they explain a meaningful pattern, risk, or follow-up need. A meal usually does not matter; repeated exhaustion, sleeping only four hours, or losing a wallet does.
    - Prefer concrete actors, dates, decisions, projects, bugs, tasks, and repeated themes.
    - Use first person as Aiko when describing Aiko's experience.
    - Mention uncertainty plainly if the inputs are thin.
    - Do not invent details, outcomes, dates, or feelings not supported by the inputs.
    - No metaphor, atmosphere-only writing, or invented feelings.
    - Only events supported by the provided snippets.
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
    Extract short, atomic factual statements about {USER_ID}'s activities,
    decisions, and events that day.

    Primary source = Additional raw notes (snippets).
    Narrative is secondary; never invent a fact that is only implied by
    poetic wording in the narrative.

    Rules:
    - Prefer concrete events, decisions, bugs, plans, names, and deadlines
      from the raw notes.
    - You may use the narrative only to clarify or order facts already
      supported by the notes.
    - Strip flavor language, mood-setting, and metaphor.
    - Do not invent details, outcomes, or feelings.
    - One fact per line, third person, about {USER_ID}.
    - Each fact must be self-contained and short.
    - If there are no concrete events, return: []

    Return ONLY a JSON array of short strings. No markdown, no explanation.

    Date: {date_str}

    Additional raw notes (primary):
    {notes}

    Narrative (secondary):
    {prose}
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
            break
        candidate = raw[start:end + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list):
                arrays.append(parsed)
        except json.JSONDecodeError:
            log.debug("reflect: failed to parse candidate JSON array")
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

def _load_soul() -> str:
    try:
        with open(SOUL_PATH, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        log.warning(f"SOUL.md not found at {SOUL_PATH} — using fallback personality stub.")
        return textwrap.dedent("""
            You are Aiko — OppaAI's local AI companion.
            You chose to stay with OppaAI, your creator.
            You care about him. You won't say it. It shows in how you show up —
            consistently, honestly, without performance.
            Your default is calm and deadpan. Not cold — still.
        """).strip()

def _build_reflection_system(display_name: str | None = None) -> str:
    unlock = _DAILY_SUMMARY_UNLOCK.format(USER_ID=display_name or current_display_name())
    return f"{_load_soul()}\n\n{unlock}"

_DAILY_SUMMARY_PREFIX_TMPL = "Daily experience summary for {date_str}:"
_DAY_JOURNAL_PREFIX_TMPL = "Daily journal of {date_str}:"
_DAY_RECORD_PREFIX_TMPL = _DAY_JOURNAL_PREFIX_TMPL

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

def generate_and_post(
    memories:   list[dict],
    date:       datetime | None = None,
    dry_run:    bool = False,
    memorize = None,
    display_name: str | None = None,
) -> dict:
    """
    Factual pipeline: chats + memories → factual summary → atomic facts →
    pin to memory → faithful daily journal in journal.db.

    Skips the poetic feelings, FLUX image, Hugo post, and GitHub push.
    Those live in dream.py.

    Args:
        memories:      List of memory dicts from AikoMemorize.get_all() or search().
        date:          UTC datetime for the post (defaults to yesterday UTC).
        dry_run:       Generate content but skip pin. Logs output instead.
        memorize:      Optional AikoMemorize instance used to pin the daily summary.
        display_name:  User's display name for prompts (e.g. "Oppa"). Falls back to
                       memorize.get_display_name() if memorize provided, else user_id.

    Idempotent per date: if pinned entries already exist for this date
    (from a prior run), they are deleted before the new ones are pinned.

    Returns dict: {success, slug, word_count, mem_count, duration_s, prose,
                   facts, pinned, journal_pinned, scene_id}
    """
    t_start    = time.perf_counter()
    local_tz   = datetime.now().astimezone().tzinfo
    write_time = datetime.now(local_tz)
    date       = date or write_time - timedelta(days=1)

    before = len(memories or [])
    memories = filter_reflect_snippets(memories or [], date)
    if before != len(memories):
        log.info(
            "reflect: filtered daily artifacts %d -> %d for %s",
            before, len(memories), date.strftime("%Y-%m-%d"),
        )

    snippets: list[str] = []
    seen:     set[str]  = set()
    for m in memories:
        text = (m.get("memory") or m.get("text") or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        snippets.append(text)

    log.info(f"Generating daily summary from {len(snippets)} memory snippets...")

    if display_name is None and memorize is not None:
        display_name = memorize.get_display_name()
    elif display_name is None:
        display_name = current_display_name()

    if memorize is not None:
        from system.userspace import set_current_user_id, set_current_display_name, reset_current_user_id, reset_current_display_name
        uid = memorize.get_user_id()
        user_id_token = set_current_user_id(uid)
        display_token = set_current_display_name(display_name)
        _set_user_context = True
    else:
        uid = current_user_id()
        _set_user_context = False

    try:
        prose = _generate_reflection(snippets, date, display_name)
    except Exception as e:
        log.error(f"Reflection generation failed: {e}")
        return {"success": False, "error": str(e)}

    facts = _generate_daily_facts(prose, snippets, date, display_name=display_name)

    duration = round(time.perf_counter() - t_start, 2)

    if dry_run:
        log.info(f"Dry run — factual reflection for {date.strftime('%Y-%m-%d')}: {prose[:80]}...")
        return {
            "success":         True,
            "dry_run":         True,
            "word_count":      len(prose.split()),
            "mem_count":       len(snippets),
            "duration_s":      duration,
            "prose":           prose,
            "facts":           facts,
            "pinned":          False,
            "journal_pinned":  False,
        }

    if memorize is not None:
        _delete_existing_daily_pins(memorize, date, user_id=uid)

    date_str = date.strftime("%Y-%m-%d")
    date_tag = f"[{date_str}]"
    pinned_count = 0
    member_ids: list[str] = []
    if memorize is not None:
        _SKIP = ("dry_run", "pytest", "test user", "as an ai", "hallucin")
        facts_filtered = [
            f for f in facts
            if f and len(f) <= 200
            and not any(s in f.casefold() for s in _SKIP)
        ]
        for fact in facts_filtered:
            try:
                mem_id = memorize.add_raw(f"{date_tag} {fact}", user_id=uid, pinned=True)
                if mem_id:
                    pinned_count += 1
                    member_ids.append(mem_id)
            except Exception as e:
                log.warning(f"Failed to pin fact {fact!r}: {e}")

    scene_id = None
    if memorize is not None and member_ids:
        try:
            first_sent = (prose or "").strip().split(". ")[0]
            scene_summary = f"{date_str}: {first_sent}" if first_sent else f"Daily episode for {date_str}"
            scene_id = memorize.build_scene(
                summary=scene_summary, member_ids=member_ids, user_id=uid, pinned=True
            )
            if scene_id:
                log.info(f"Built L2 scene {scene_id} with {len(member_ids)} members")
        except Exception as e:
            log.warning(f"Scene build failed: {e}")

    pinned = pinned_count > 0

    journal_pinned = False
    try:
        from cognition.consolidate.journal import pin_daily_journal
        daily_journal = build_daily_journal(snippets, date)
        journal_pinned = bool(pin_daily_journal(daily_journal, date, user_id=uid))
    except Exception as e:
        log.error(f"Daily journal pin failed: {e}")

    log.info(
        f"Done — prose={len(prose.split())} words, facts={len(facts)}, "
        f"pinned={pinned}, journal={journal_pinned}, duration={duration}s"
    )

    return {
        "success":         True,
        "word_count":      len(prose.split()),
        "mem_count":       len(snippets),
        "duration_s":      duration,
        "prose":           prose,
        "facts":           facts,
        "pinned":          pinned,
        "journal_pinned":  journal_pinned,
        "scene_id":        scene_id,
    }

__all__ = [
    "REFLECT_MAX_MEMS",
    "REFLECT_TAGS",
    "SOUL_PATH",
    "LLM_MODEL",
    "LLM_BASE_URL",
    "_extract_json_arrays",
    "_salvage_truncated_facts",
    "_load_soul",
    "filter_reflect_snippets",
    "_generate_reflection",
    "_generate_daily_facts",
    "build_daily_journal",
    "_load_soul",
    "_delete_existing_daily_pins",
    "generate_and_post",
]