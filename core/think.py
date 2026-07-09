"""
core/think.py

Aiko's chat facade.
  - Routes between single-shot chat and the agentic task loop in core.agentic.
  - Streams llama.cpp response to console + TTS simultaneously.
  - Queues long-term memory writes (delegated to core.memorize's async write queue).
  - Owns scheduled-job callbacks and idle learner handoff (delegated to core.learn).
  - Owns the proactive idle check-in state machine (config/proactive.yaml),
    which is also the "is Aiko resting" signal core.learn's idle_learner_loop
    waits on before starting autonomous quick-study top-ups.
"""

import logging
import os
import json
import random
import warnings

warnings.filterwarnings("ignore")
logging.getLogger("phonemizer").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

from datetime import datetime
from datetime import time as dt_time
from openai import OpenAI
from pathlib import Path
import re
import threading
import time
import unicodedata

try:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
except ImportError:  # pragma: no cover - py<3.9 fallback, shouldn't happen here
    ZoneInfo = None
    ZoneInfoNotFoundError = Exception

from core.memorize import AikoMemorize
from core.speak    import AikoSpeak
from core.tools    import web_search_context
from core.agentic  import run_agentic_chat, resolve_search_query, llm_resolve_search_query
from core.wiki import wiki_knowledge_context_for
from core.knowledge import knowledge_context_for
from core.log      import get_logger
from core.social import run_scheduled_weekly_social
from core.schedule import DueJob, register_system_handler
from core.userspace import current_user_id, user_profile_path
from core import reason
from core import learn

log = get_logger(__name__)
register_system_handler("weekly_social", run_scheduled_weekly_social)

# ── boot labels ───────────────────────────────────────────────────────────────

BOOT_LABELS = {
    'think_start':  'Loading llama.cpp client + persona...',
    'think_warmup': 'Warming up language model...',
    'think_reminders': 'Starting reminder scheduler...',
}

# ── config ────────────────────────────────────────────────────────────────────

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8080/v1")
LLM_MODEL    = os.getenv("LLM_MODEL",    "ministral")
ROUTER_MODEL = os.getenv("ROUTER_MODEL", LLM_MODEL)
LLM_TIMEOUT  = float(os.getenv("LLM_TIMEOUT", 120))
CONTEXT_WINDOW_TURNS = int(os.getenv("CONTEXT_WINDOW_TURNS", 8))

_BASE_PREDICT    = int(os.getenv("LLM_MAX_TOKENS", os.getenv("BASE_PREDICT", 280)))
_AGENT_MAX_TOKENS = int(os.getenv("AGENT_MAX_TOKENS", _BASE_PREDICT * 4))
_REASONING_SCALE = int(os.getenv("REASONING_SCALE", 3))
# Route task-vs-chat turns semantically by default using the same embedding
# model as memory/RAG. Set ROUTE_MODE=llm to let the local LLM classify
# instead, or ROUTE_MODE=chat to disable autonomous routing.
_ROUTE_ENABLED = os.getenv("ROUTE_ENABLED", "1").lower() in {"1", "true", "yes", "on"}
_ROUTE_MODE = os.getenv("ROUTE_MODE", "semantic").strip().lower()

# Three separate instruct strings, one per embedding context
_ROUTE_INSTRUCT_BINARY = "Does this message ask someone to perform a task or action, or is it just conversation?"
_ROUTE_INSTRUCT_TOOL   = "This is an autonomous task request. Which work steps are likely needed?"
_ROUTE_INSTRUCT_SEARCH = "Does answering this require looking up current or external data?"

_SEMANTIC_ROUTE_MIN_GAP = float(os.getenv("ROUTE_MIN_GAP", "0.10"))
_SEMANTIC_LABEL_TOP_K = int(os.getenv("ROUTE_LABEL_TOP_K", "3"))

_PERSONA_PATH = Path(__file__).resolve().parent.parent / "persona" / "soul.md"
_LOCAL_KNOWLEDGE_RE = re.compile(
    r"\b("
    r"aiko|your architecture|your hardware|your features?|your functions?|"
    r"what can you do|how do you work|how are you built|"
    r"knowledge base|wiki|docs?|readme|roadmap|install|config|"
    r"soul\.md|user\.md|skills?\.md|schedule\.md|"
    r"repo|repository|codebase|local files|your files"
    r")\b",
    re.IGNORECASE,
)


def _load_persona() -> str:
    """Read the lightweight normal-chat persona.

    Task/tool policy lives in the agentic prompt so casual chat does not pay
    for skills/schedule tokens on every turn.
    """
    if not _PERSONA_PATH.exists():
        raise FileNotFoundError(f"soul.md not found at {_PERSONA_PATH}")
    persona = _PERSONA_PATH.read_text(encoding="utf-8").strip()

    context_blocks = []
    user_path = user_profile_path()
    if user_path.exists():
        context_blocks.append(user_path.read_text(encoding="utf-8").strip())
    user_block = "\n\n" + "\n\n".join(context_blocks) if context_blocks else ""

    user_id = current_user_id()
    return persona.replace("USER_ID_HERE", user_id) + user_block


def _should_use_local_knowledge(user_input: str) -> bool:
    """Return True for normal-chat questions about Aiko's local docs/files.

    This keeps casual chat fast while still letting non-task questions about
    Aiko's architecture, features, hardware/docs, wiki, and knowledge base use
    the local markdown corpus without entering the tool loop.
    """
    return bool(_LOCAL_KNOWLEDGE_RE.search(user_input))

# ── semantic intent examples ──────────────────────────────────────────────────

import subprocess

def _play_beep() -> None:
    """Play a short system notification sound before a scheduled job announcement."""
    def _run():
        try:
            subprocess.run(
                ["paplay", "/usr/share/sounds/freedesktop/stereo/bell.oga"],
                check=False, timeout=6,
            )
        except Exception as e:
            log.warning("Beep playback failed: %s", e)
    threading.Thread(target=_run, daemon=True).start()

# load route examples
_EXAMPLES_PATH = Path(__file__).resolve().parent.parent / "persona" / "router_prompts.json"

def _load_route_examples() -> tuple[dict, dict]:
    with open(_EXAMPLES_PATH, encoding="utf-8") as f:
        data = json.load(f)
    ternary = {k: tuple(v) for k, v in data["ternary"].items()}  # 3 labels
    tools   = {k: tuple(v) for k, v in data["tools"].items()}
    return ternary, tools

_ROUTE_TERNARY_EXAMPLES, _ROUTE_TOOL_EXAMPLES = _load_route_examples()

_AGENTIC_ROUTE_RE = re.compile(
    r"\b("
    r"research|look up|search|fetch|find out|check whether|check if|"
    r"fix|debug|implement|refactor|patch|edit|modify|update tests?|inspect|open .*\.(?:py|json|md)|"
    r"write|draft|compose|save|create|prepare|"
    r"plan|roadmap|checklist|break down|schedule|remind|reminder|alarm|timer|ping me|notify me|"
    r"continue|resume|pick up where we left off|keep going|compare .* recommend|decide .* and"
    r")\b",
    re.IGNORECASE,
)


def _extract_search_results_block(system_prompt: str) -> str:
    match = re.search(r"<search_results\b[^>]*>.*?</search_results>", system_prompt or "", re.DOTALL)
    return match.group(0) if match else ""


# ── proactive check-in config (config/proactive.yaml, via os.environ) ─────────
# core.config.load_config() has already populated these into the process
# environment by the time this module is imported (see core/wakeup.py /
# core/learn.py — whichever entrypoint runs first calls it). We just read
# them here the same way every other module's config block does.

def _bool_env(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _parse_str_list(raw: str) -> list[str]:
    """Parse a YAML-list-turned-env-var back into a list of strings.

    core.config may have serialized a YAML list either as a JSON array
    string (e.g. '["00:00-06:00"]') or, if the YAML loader flattened it
    some other way, as a comma/newline separated string. Handle both so
    this doesn't silently break depending on how config.py encodes lists.
    """
    if not raw or not raw.strip():
        return []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except (json.JSONDecodeError, TypeError):
        pass
    return [p.strip() for p in re.split(r"[,\n]", raw) if p.strip()]


PROACTIVE_ENABLED = _bool_env("PROACTIVE_ENABLED", "1")
PROACTIVE_FIRST_IDLE_MIN_SECONDS = float(os.getenv("PROACTIVE_FIRST_IDLE_MIN_SECONDS", 300))
PROACTIVE_FIRST_IDLE_MAX_SECONDS = float(os.getenv("PROACTIVE_FIRST_IDLE_MAX_SECONDS", 900))
PROACTIVE_COOLDOWN_SECONDS = float(os.getenv("PROACTIVE_COOLDOWN_SECONDS", 1800))
PROACTIVE_MAX_PER_HOUR = int(os.getenv("PROACTIVE_MAX_PER_HOUR", 2))
PROACTIVE_REST_AFTER_SECONDS = float(os.getenv("PROACTIVE_REST_AFTER_SECONDS", 3600))
PROACTIVE_USE_LLM = _bool_env("PROACTIVE_USE_LLM", "1")
PROACTIVE_TIMEZONE = os.getenv("PROACTIVE_TIMEZONE", "").strip() or os.getenv("TIMEZONE", "UTC")
PROACTIVE_SPEAK = _bool_env("PROACTIVE_SPEAK", "1")
PROACTIVE_REST_MESSAGE = os.getenv(
    "PROACTIVE_REST_MESSAGE",
    "You've been away for a while so I'll go quiet and rest. Ping me when you need me.",
)
PROACTIVE_REST_PROMPT_HINT = os.getenv(
    "PROACTIVE_REST_PROMPT_HINT",
    "{user} has not spoken to you for about an hour. Say one short warm line that "
    "you are going quiet and resting until they return.",
)
PROACTIVE_QUIET_WINDOWS = _parse_str_list(os.getenv("PROACTIVE_QUIET_WINDOWS", ""))
PROACTIVE_FOCUS_WINDOWS = _parse_str_list(os.getenv("PROACTIVE_FOCUS_WINDOWS", ""))
PROACTIVE_PROMPT_HINTS = _parse_str_list(os.getenv("PROACTIVE_PROMPT_HINTS", ""))
PROACTIVE_MESSAGES = _parse_str_list(os.getenv("PROACTIVE_MESSAGES", ""))
# How often the background loop wakes to re-check idle/window conditions.
# Not itself a proactive.yaml key — deliberately short relative to
# PROACTIVE_FIRST_IDLE_MIN_SECONDS so a check-in fires close to its target
# time rather than up to a whole cycle late.
PROACTIVE_CHECK_INTERVAL_SECONDS = float(os.getenv("PROACTIVE_CHECK_INTERVAL_SECONDS", 60))

_WEEKDAY_INDEX = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}
_TIME_RANGE_RE = re.compile(r"^(\d{1,2}):(\d{2})-(\d{1,2}):(\d{2})$")
_DAY_RANGE_RE = re.compile(r"^([a-z]{3})-([a-z]{3})$")


def _parse_time_range(range_str: str) -> tuple[dt_time, dt_time] | None:
    match = _TIME_RANGE_RE.match(range_str.strip())
    if not match:
        return None
    h1, m1, h2, m2 = (int(g) for g in match.groups())
    try:
        return dt_time(h1, m1), dt_time(h2, m2)
    except ValueError:
        return None


def _time_in_range(now_t: dt_time, start_t: dt_time, end_t: dt_time) -> bool:
    if start_t <= end_t:
        return start_t <= now_t <= end_t
    return now_t >= start_t or now_t <= end_t  # window wraps past midnight


def _window_matches(window_str: str, now_dt: datetime) -> bool:
    """Match one proactive.yaml window entry against the current time.

    Two shapes are supported:
      "00:00-06:00"             — daily, any weekday
      "mon-fri 06:00-19:00"     — restricted to a weekday range
    A single weekday name ("sat 06:00-11:00") is also accepted.
    """
    window_str = window_str.strip()
    if not window_str:
        return False
    parts = window_str.split()
    if len(parts) == 2:
        day_part, time_part = parts
    else:
        day_part, time_part = None, parts[0]

    time_range = _parse_time_range(time_part)
    if not time_range:
        log.warning("[proactive] unparsable window entry, ignoring: %r", window_str)
        return False
    if not _time_in_range(now_dt.time(), *time_range):
        return False

    if day_part:
        day_part = day_part.lower()
        range_match = _DAY_RANGE_RE.match(day_part)
        if range_match:
            d1, d2 = range_match.groups()
            if d1 in _WEEKDAY_INDEX and d2 in _WEEKDAY_INDEX:
                start_idx, end_idx = _WEEKDAY_INDEX[d1], _WEEKDAY_INDEX[d2]
                today_idx = now_dt.weekday()
                if start_idx <= end_idx:
                    return start_idx <= today_idx <= end_idx
                return today_idx >= start_idx or today_idx <= end_idx
        elif day_part in _WEEKDAY_INDEX:
            return now_dt.weekday() == _WEEKDAY_INDEX[day_part]
        else:
            log.warning("[proactive] unparsable day range in window entry: %r", window_str)
            return False

    return True


def _current_datetime_block() -> str:
    tz_name = os.getenv("TIMEZONE", "UTC")
    now = _proactive_now()
    return (
        "<current_datetime>\n"
        f"Now: {now.strftime('%A, %B %d, %Y, %I:%M %p')} ({tz_name})\n"
        "</current_datetime>"
    )


def _proactive_now() -> datetime:
    if ZoneInfo is None:
        return datetime.now()
    try:
        return datetime.now(ZoneInfo(PROACTIVE_TIMEZONE))
    except ZoneInfoNotFoundError:
        log.warning("[proactive] unknown timezone %r, falling back to naive local time", PROACTIVE_TIMEZONE)
        return datetime.now()


# ── think ─────────────────────────────────────────────────────────────────────

class AikoThink:
    def __init__(self, memorize: AikoMemorize, speak: AikoSpeak | None = None) -> None:
        self._client    = OpenAI(base_url=LLM_BASE_URL, api_key="not-needed")
        self._llm_model = LLM_MODEL
        self._router_model = ROUTER_MODEL
        self._memorize  = memorize
        self._speak     = speak
        self._persona   = _load_persona()
        self._history:  list[dict] = []
        self._history_lock = threading.Lock()
        self._pending_search_query: str | None = None
        # Cache of (labels, embedding_matrix) per (example-corpus-id, instruct)
        # pair — built via reason.embed_example_matrix, which always
        # re-embeds; caching the result here avoids paying that cost on
        # every routing call for a static example corpus.
        self._semantic_example_cache: dict = {}
        self._semantic_example_cache_lock = threading.RLock()
        self._active_turn = threading.Event()
        self._turn_lock = threading.RLock()
        self._reasoning = False
        self.last_usage: dict = {}
        self.last_prompt_debug: dict = {}

        self._last_chat_time = time.time()
        self._idle_learner_thread = threading.Thread(
            target=learn.idle_learner_loop, args=(self,), daemon=True
        )
        self._idle_learner_thread.start()

        # ── proactive idle check-in state machine ──────────────────────────
        # See _proactive_tick / is_proactive_resting. Drives config/proactive.yaml
        # (PROACTIVE_ENABLED etc., module-level above). is_proactive_resting()
        # is the exact signal core.learn.idle_learner_loop waits on before it
        # will start an autonomous quick_studying pass, so the two idle
        # systems never compete for the same idle time.
        self._proactive_lock = threading.Lock()
        self._proactive_resting = False
        self._proactive_last_checkin_time: float | None = None
        self._proactive_checkin_times: list[float] = []  # rolling 1hr window, for MAX_PER_HOUR
        self._proactive_next_first_delay = random.uniform(
            PROACTIVE_FIRST_IDLE_MIN_SECONDS, PROACTIVE_FIRST_IDLE_MAX_SECONDS
        )
        self._proactive_thread: threading.Thread | None = None
        if PROACTIVE_ENABLED:
            self._proactive_thread = threading.Thread(target=self._proactive_loop, daemon=True)
            self._proactive_thread.start()
        else:
            log.info("[proactive] disabled via PROACTIVE_ENABLED — no idle check-ins will run.")

        # NOTE: this class used to also construct its own ScheduleRunner
        # here (self._reminders = ScheduleRunner(...); self._reminders.start()).
        # That has been removed — core.wakeup.AikoWakeup.boot() already
        # constructs and starts the single app-wide ScheduleRunner (wired to
        # memorize/reflect/consolidate as well as this instance's
        # handle_scheduled_job), and register_scheduler() makes it
        # discoverable to tools. Having two independent ScheduleRunner
        # threads both reading schedule.json meant every due job — reminders,
        # weekly_social, and the deep-study window jobs — fired twice.

        self._warmup_thread = threading.Thread(target=self._warmup_llm, daemon=True)
        self._warmup_thread.start()
      
    def _warmup_llm(self) -> None:
        try:
            self._client.chat.completions.create(
                model=self._llm_model,
                messages=[{"role": "user", "content": "hi"}],
                stream=False, max_tokens=1,
            )
        except Exception as e:
            log.warning("LLM warmup failed: %s", e)

    def join_warmup(self) -> None:
        if self._warmup_thread and self._warmup_thread.is_alive():
            self._warmup_thread.join()

    # ── public api ────────────────────────────────────────────────────────────

    def route(self, user_input: str, token_callback=None) -> str:
        """Main entry point. Ternary routing."""
        with self._turn_lock:
            self._last_chat_time = time.time()
            self._active_turn.set()
            self._note_user_activity()
            try:
                intent = self._route_intent(user_input)
                log.info("[route] intent=%s", intent)
                
                if intent == "agentic":
                    return self.agentic_chat(user_input, token_callback=token_callback)
                elif intent == "webchat":
                    return self.webchat(user_input, token_callback=token_callback)
                else:  # localchat
                    return self.chat(user_input, token_callback=token_callback, _skip_search=True)
            finally:
                self._last_chat_time = time.time()
                self._active_turn.clear()

    def _note_user_activity(self) -> None:
        """Reset the proactive state machine on real user activity: clear
        the resting flag (so the idle learner stops treating this as rest
        time) and re-roll the next first-check-in delay so it's measured
        from this turn, not whenever the last one happened to be."""
        with self._proactive_lock:
            self._proactive_resting = False
            self._proactive_next_first_delay = random.uniform(
                PROACTIVE_FIRST_IDLE_MIN_SECONDS, PROACTIVE_FIRST_IDLE_MAX_SECONDS
            )

    def is_proactive_resting(self) -> bool:
        """True once Aiko has sent her PROACTIVE_REST_MESSAGE and gone
        quiet for this idle stretch. This is the exact signal
        core.learn.idle_learner_loop polls before starting an autonomous
        quick_studying pass — see that function's docstring for why."""
        return self._proactive_resting

    def _route_intent(self, user_input: str) -> str:
        """Ternary routing: single embedding, three-way decision with a
        high-confidence margin so a close call doesn't get committed to
        agentic (or webchat) just because it happened to be checked first.

        The close-vector label-scoring math itself (normalize + batched
        matmul + top-k mean per label) lives in core.reason; this method
        only owns the routing policy (thresholds, gap, LLM tie-break).
        """
        self._pending_search_query = None  # reset
    
        if not _ROUTE_ENABLED or _ROUTE_MODE in {"0", "off", "false", "disabled"}:
            return "localchat"
    
        instruct = "What kind of task or question is this?"
        embedder = self._memorize._mem._embedder
        query_vec = embedder.embed_query(user_input, instruct=instruct)
        labels, example_vecs = self._semantic_example_vectors(_ROUTE_TERNARY_EXAMPLES, instruct)
        scores = reason.label_scores_topk(query_vec, labels, example_vecs, top_k=_SEMANTIC_LABEL_TOP_K)
    
        agentic_score = scores.get("agentic", 0.0)
        webchat_score = scores.get("webchat", 0.0)
    
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best_label, best_score = ranked[0] if ranked else ("localchat", 0.0)
        gap = best_score - ranked[1][1] if len(ranked) > 1 else 1.0
    
        agentic_threshold = float(os.getenv("ROUTE_AGENTIC_THRESHOLD", "0.65"))
        webchat_threshold = float(os.getenv("ROUTE_WEBCHAT_THRESHOLD", "0.60"))
    
        log.debug(
            "[route] ternary scores: agentic=%.3f webchat=%.3f best=%s gap=%.3f for: %r",
            agentic_score, webchat_score, best_label, gap, user_input
        )
    
        if best_label == "agentic" and agentic_score >= agentic_threshold and gap >= _SEMANTIC_ROUTE_MIN_GAP:
            return "agentic"
        if best_label == "webchat" and webchat_score >= webchat_threshold and gap >= _SEMANTIC_ROUTE_MIN_GAP:
            return "webchat"
    
        # Above threshold but too close to call cleanly — don't silently
        # default into agentic. Ask the LLM to break the tie; anything
        # that isn't clearly agentic falls back to local chat.
        if (agentic_score >= agentic_threshold or webchat_score >= webchat_threshold) and gap < _SEMANTIC_ROUTE_MIN_GAP:
            llm_label = self._classify_agent_intent(user_input)
            return "agentic" if llm_label == "agentic" else "localchat"
    
        return "localchat"

    def _semantic_example_vectors(self, examples_by_label: dict, instruct: str) -> tuple[list[str], object]:
        """Return cached (labels, matrix) for a static semantic example
        corpus, built via reason.embed_example_matrix. embed_example_matrix
        itself never caches — it always re-embeds — so the caching lives
        here, keyed on corpus identity + instruct string, same as before
        the math moved to core.reason."""
        cache_key = (id(examples_by_label), instruct)
        with self._semantic_example_cache_lock:
            cached = self._semantic_example_cache.get(cache_key)
            if cached is not None:
                return cached

            embedder = self._memorize._mem._embedder
            labels, vectors = reason.embed_example_matrix(embedder, examples_by_label, instruct=instruct)
            cached = (labels, vectors)
            self._semantic_example_cache[cache_key] = cached
            return cached

    def _classify_agent_intent(self, user_input: str, skip_regex: bool = False) -> str:
        """Ask the local model for a compact binary route label when semantics are ambiguous."""
        if not skip_regex and _AGENTIC_ROUTE_RE.search(user_input):
            return "agentic"            
        try:
            resp = self._client.chat.completions.create(
                model=self._router_model,
                messages=[{"role": "user", "content": (
                    f"Message: {user_input!r}\n\n"
                    "Output only the route label. No explanation.\n"
                    "Labels: [agentic, chat]\n\n"
                    "Message: 'set a reminder for 9pm'\n"
                    "Label: agentic\n\n"
                    "Message: 'write an email to my landlord'\n"
                    "Label: agentic\n\n"
                    "Message: 'debug why asyncio.run() hangs'\n"
                    "Label: agentic\n\n"
                    "Message: 'make a plan to learn Japanese'\n"
                    "Label: agentic\n\n"
                    "Message: 'search for latest llama.cpp release and summarize it'\n"
                    "Label: agentic\n\n"
                    "Message: 'compare ollama vs llama.cpp and recommend one'\n"
                    "Label: agentic\n\n"
                    "Message: 'open soul.md and show the persona block'\n"
                    "Label: agentic\n\n"
                    "Message: 'continue working on the reflection script'\n"
                    "Label: agentic\n\n"
                    "Message: 'what do you think about minimalism'\n"
                    "Label: chat\n\n"
                    "Message: 'explain semaphores from memory'\n"
                    "Label: chat\n\n"
                    "Message: 'is it weird that I find debugging more satisfying than writing features'\n"
                    "Label: chat\n\n"
                    "Label:"
                )}],
                stream=False, max_tokens=6, temperature=0.0, top_p=1.0, timeout=LLM_TIMEOUT,
            )
            label = (resp.choices[0].message.content or "chat").strip().lower()
            label = re.sub(r"[^a-z_].*$", "", label)
            return label if label in {"agentic", "chat"} else "chat"
        except Exception as e:
            log.warning("Intent routing failed: %s", e)
            return "chat"


    def agentic_chat(self, user_input: str, token_callback=None) -> str:
        """Delegate task-mode execution to core.agentic."""
        with self._turn_lock:
            self._last_chat_time = time.time()
            self._active_turn.set()
            try:
                return run_agentic_chat(self, user_input, token_callback=token_callback)
            finally:
                self._last_chat_time = time.time()
                self._active_turn.clear()
              
    def webchat(self, user_input: str, token_callback=None) -> str:
        """Web-aware chat: web_search + optional webfetch fallback."""
        if self._speak and self._speak.is_playing():
            self._speak.stop()
        
        # Build base system (persona + memory)
        system = self._persona
        system += "\n\n" + _current_datetime_block()
        memories = self._memorize.search(user_input, limit=3)
        memory_block = self._memorize.format_for_context(memories)
        if memory_block:
            system = f"{system}\n\n{memory_block}"
        
        # Try web_search first (fast) — query condensation now lives in
        # core.agentic (resolve_search_query / llm_resolve_search_query),
        # shared with the agentic tool loop's own search-query needs.
        search_query = resolve_search_query(self, user_input)
        if token_callback:
            token_callback(f"__SEARCHING__:{search_query}\n")
        
        context = web_search_context(search_query, max_results=int(os.getenv("SEARXNG_MAX_RESULTS", 3)))
        
        # Fallback: retry with better query or webfetch
        if not context or context.startswith("[search failed") or "no results" in context.lower():
            log.info("[webchat] Snippets failed, retrying with better query...")
            if token_callback:
                token_callback("__RETRYING__\n")
            
            try:
                better_query = llm_resolve_search_query(self, user_input)
                context = web_search_context(better_query, max_results=1)
            except Exception as e:
                log.warning("[webchat] Retry failed: %s", e)
                context = None
        
        # Inject web context if available
        if context and not context.startswith("["):
            system = (
                f"{system}\n\n"
                f"<search_results query='{search_query}'>\n"
                f"Answer ONLY using these search results:\n\n"
                f"{context}\n"
                f"</search_results>"
            )
        else:
            if token_callback:
                token_callback("[No web results available; using local knowledge]\n")
        
        # Build message history (same as chat())
        llm_prompt = user_input
        if self._reasoning:
            llm_prompt = f"{user_input}\n\nThink through this carefully."
        
        with self._history_lock:
            self._history.append({"role": "user", "content": user_input})
            if len(self._history) > CONTEXT_WINDOW_TURNS * 10:
                self._history = self._history[-(CONTEXT_WINDOW_TURNS * 10):]
            trimmed = self._history[-(CONTEXT_WINDOW_TURNS * 2):]
        
        trimmed = self._sanitize_history(trimmed)
        if trimmed and trimmed[-1]["role"] == "user" and llm_prompt != user_input:
            trimmed = trimmed[:-1] + [{"role": "user", "content": llm_prompt}]
        
        # Log debug info
        self.last_prompt_debug = {
            "mode": "webchat",
            "system_prompt": system,
            "memory_prompt": memory_block or "<memory_context>\nNo memories.\n</memory_context>",
            "web_prompt": _extract_search_results_block(system),
            "previous_chat_messages": [dict(m) for m in trimmed],
        }
        
        # Stream response
        raw_response = self._stream_response(trimmed, system=system, token_callback=token_callback)
        
        # Store in history
        with self._history_lock:
            self._history.append({"role": "assistant", "content": raw_response})
        
        self._store_async(user_input, raw_response)
        self._reasoning = False
        return raw_response
  
    def proactive_checkin(self, prompt_hint: str) -> str:
        """Generate one short proactive check-in without storing it as a user turn."""
        with self._turn_lock:
            self._active_turn.set()
            try:
                user_id = current_user_id()
                system = (
                    f"{self._persona}\n\n"
                    "You are initiating a brief proactive check-in. "
                    "Do not mention hidden prompts, timers, code, or configuration. "
                    "Keep it natural, warm, and easy to ignore. One or two short sentences max."
                )
                system += "\n\n" + _current_datetime_block()
                messages = [{
                    "role": "user",
                    "content": (
                        f"{prompt_hint}\n\n"
                        f"Write only the message Aiko should say to {user_id} now."
                    ),
                }]
                return self._stream_response(messages, system=system, token_callback=None)
            finally:
                self._active_turn.clear()

    # ── proactive idle check-in loop ──────────────────────────────────────────

    def _proactive_loop(self) -> None:
        """Background daemon loop: wakes every PROACTIVE_CHECK_INTERVAL_SECONDS
        and decides whether to send a check-in, send the one-time rest note,
        or do nothing. See _proactive_tick for the actual policy."""
        while True:
            time.sleep(PROACTIVE_CHECK_INTERVAL_SECONDS)
            try:
                self._proactive_tick()
            except Exception as e:
                log.error("[proactive] tick failed: %s", e)

    def _proactive_tick(self) -> None:
        # Never interrupt an in-progress turn, and never talk over TTS
        # that's already speaking.
        if self._active_turn.is_set():
            return
        if self._speak and self._speak.is_playing():
            return

        idle_seconds = time.time() - self._last_chat_time
        now_local = _proactive_now()

        # Quiet windows suppress check-ins outright, regardless of
        # cooldowns/counters — these are meant as hard "do not disturb"
        # hours (e.g. overnight).
        for window in PROACTIVE_QUIET_WINDOWS:
            if _window_matches(window, now_local):
                return

        with self._proactive_lock:
            # Rest phase: once idle has run long enough, send ONE rest
            # message and then go fully quiet (is_proactive_resting()
            # becomes True) until the user speaks again (see
            # _note_user_activity, called from route()).
            if idle_seconds >= PROACTIVE_REST_AFTER_SECONDS:
                if not self._proactive_resting:
                    self._fire_rest_message()
                return

            if self._proactive_resting:
                # Shouldn't normally reach here (resting only flips true
                # once idle_seconds has already crossed REST_AFTER_SECONDS,
                # and it's reset the moment the user speaks), but guard
                # against firing a normal check-in mid-rest regardless.
                return

            if idle_seconds < self._proactive_next_first_delay:
                return

            if self._proactive_last_checkin_time is not None:
                if time.time() - self._proactive_last_checkin_time < PROACTIVE_COOLDOWN_SECONDS:
                    return

            one_hour_ago = time.time() - 3600
            self._proactive_checkin_times = [t for t in self._proactive_checkin_times if t >= one_hour_ago]
            if len(self._proactive_checkin_times) >= PROACTIVE_MAX_PER_HOUR:
                return

            # Focus windows are the OPPOSITE of quiet windows: if any are
            # configured, check-ins are only permitted while inside one of
            # them (e.g. "don't ping outside work/focus hours"). If none
            # are configured, check-ins are allowed any time outside quiet
            # windows.
            if PROACTIVE_FOCUS_WINDOWS:
                in_focus = any(_window_matches(w, now_local) for w in PROACTIVE_FOCUS_WINDOWS)
                if not in_focus:
                    return

            self._fire_checkin()

    def _fire_rest_message(self) -> None:
        """Send the one-time PROACTIVE_REST_MESSAGE and flip resting=True.
        Called with self._proactive_lock already held."""
        user_id = current_user_id()
        hint = PROACTIVE_REST_PROMPT_HINT.replace("{user}", user_id)
        message = None
        if PROACTIVE_USE_LLM:
            try:
                message = self.proactive_checkin(hint)
            except Exception as e:
                log.warning("[proactive] rest message generation failed: %s", e)
        if not message:
            message = PROACTIVE_REST_MESSAGE

        log.info("[proactive] going quiet to rest: %s", message)
        if PROACTIVE_SPEAK and self._speak:
            self._speak.speak(message)
        self._proactive_resting = True

    def _fire_checkin(self) -> None:
        """Send one ordinary idle check-in. Called with self._proactive_lock
        already held."""
        user_id = current_user_id()
        message = None
        try:
            if PROACTIVE_USE_LLM and PROACTIVE_PROMPT_HINTS:
                hint = random.choice(PROACTIVE_PROMPT_HINTS).replace("{user}", user_id)
                message = self.proactive_checkin(hint)
            elif PROACTIVE_MESSAGES:
                message = random.choice(PROACTIVE_MESSAGES)
        except Exception as e:
            log.warning("[proactive] check-in generation failed: %s", e)
            message = random.choice(PROACTIVE_MESSAGES) if PROACTIVE_MESSAGES else None

        if not message:
            return

        log.info("[proactive] check-in: %s", message)
        if PROACTIVE_SPEAK and self._speak:
            self._speak.speak(message)

        now_ts = time.time()
        self._proactive_last_checkin_time = now_ts
        self._proactive_checkin_times.append(now_ts)

    def chat(self, user_input: str, token_callback=None, _skip_search: bool = True, _history_label: str | None = None) -> str:
        """Standard chat: local knowledge only (persona + memory)."""
        if self._speak and self._speak.is_playing():
            self._speak.stop()
        
        # Memory + persona (no web)
        memories = self._memorize.search(user_input, limit=3)
        memory_block = self._memorize.format_for_context(memories)
        
        system = self._persona
        system += "\n\n" + _current_datetime_block()
        if memory_block:
            system = f"{system}\n\n{memory_block}"
        else:
            system += "\n\n<memory_context>\nNo relevant memories found.\n</memory_context>"
        
        # Local knowledge context (if user asks about Aiko)
        if _should_use_local_knowledge(user_input):
            try:
                wiki_context = wiki_knowledge_context_for(
                    user_input, limit=3, max_chars=3000,
                    embedder=self._memorize._mem._embedder,
                )
                learned_context = knowledge_context_for(
                    user_input, limit=3, max_chars=2000,
                    embedder=self._memorize._mem._embedder,
                )
                system = f"{system}\n\n{wiki_context}\n\n{learned_context}"
            except Exception as e:
                log.error("Local knowledge lookup failed: %s", e)
        
        # Build messages
        llm_prompt = user_input
        if self._reasoning:
            llm_prompt = f"{user_input}\n\nThink through this carefully."
        
        with self._history_lock:
            self._history.append({"role": "user", "content": user_input})
            if len(self._history) > CONTEXT_WINDOW_TURNS * 10:
                self._history = self._history[-(CONTEXT_WINDOW_TURNS * 10):]
            trimmed = self._history[-(CONTEXT_WINDOW_TURNS * 2):]
        
        trimmed = self._sanitize_history(trimmed)
        if trimmed and trimmed[-1]["role"] == "user" and llm_prompt != user_input:
            trimmed = trimmed[:-1] + [{"role": "user", "content": llm_prompt}]
        
        # Log debug
        self.last_prompt_debug = {
            "mode": "localchat",
            "system_prompt": system,
            "memory_prompt": memory_block or "<memory_context>\nNo memories.\n</memory_context>",
            "web_prompt": "",
            "previous_chat_messages": [dict(m) for m in trimmed],
        }
        
        # Stream response
        raw_response = self._stream_response(trimmed, system=system, token_callback=token_callback)
        
        # Store
        with self._history_lock:
            self._history.append({"role": "assistant", "content": raw_response})
        
        self._store_async(user_input, raw_response)
        self._reasoning = False
        return raw_response

    def web_search(self, query: str, token_callback=None) -> str:
        """Explicit /web command path."""
        context = web_search_context(query)
        if not context or "no results" in context or "failed" in context:
            msg = f"[no results for: {query}]"
            if token_callback: token_callback(msg)
            return msg
        return self.chat(context, token_callback=token_callback, _skip_search=True, _history_label=query)

    def reset_context(self) -> None:
        with self._history_lock:
            self._history.clear()

    def last_turn(self) -> tuple[str, str] | None:
        with self._history_lock:
            history_snapshot = list(self._history)
        users = [m["content"].strip() for m in history_snapshot if m.get("role") == "user" and (m.get("content") or "").strip()]
        assistants = [m["content"].strip() for m in history_snapshot if m.get("role") == "assistant" and (m.get("content") or "").strip()]
        if not users or not assistants: return None
        return users[-1], assistants[-1]

    def set_reasoning(self, enabled: bool) -> None: self._reasoning = enabled
    def set_speak(self, speak) -> None: self._speak = speak

    def wait_for_memory(self, timeout: float | None = None) -> bool:
        """Block until AikoMemorize's async write queue drains, or timeout
        elapses. The queue itself now lives in core.memorize; this is a
        thin passthrough kept for call sites (core.agentic) that only know
        about the AikoThink instance."""
        return self._memorize.wait_for_writes(timeout=timeout)

    def handle_scheduled_job(self, job: DueJob) -> None:
        """Announce or execute a due scheduled job without blocking the scheduler."""
        text = f"{job.title}. {job.task}"
        log.info("[schedule] due %s action=%s: %s", job.id, job.action, text)
        if job.action == "announce":
            _play_beep()
            if self._speak:
                self._speak.speak(text)
            else:
                log.info(f"Aiko scheduled job: {text}")
            return
        threading.Thread(target=self._run_scheduled_agentic_job, args=(job,), daemon=True).start()

    def _run_scheduled_agentic_job(self, job: DueJob) -> None:
        """Run a scheduled autonomous task through Aiko's agent loop."""
        prompt = (
            "Scheduled job due. Use only local available tools. If external action "
            "is unavailable, draft/save the best local artifact and state next step.\n\n"
            f"Title: {job.title}\nTask: {job.task}"
        )
        try:
            self.agentic_chat(prompt)
        except Exception as e:
            log.error("Scheduled agentic job failed: %s", e)

    # ── internal ──────────────────────────────────────────────────────────────

    def _emit(self, text: str, token_callback=None) -> None:
        if not text: return

        # Always drive the TUI callback directly, regardless of TTS
        if token_callback:
            words = text.split(" ")
            for i, word in enumerate(words):
                token_callback(word if i == 0 else " " + word)
                time.sleep(float(os.getenv("EMIT_DELAY", 0.012)))

        # TTS runs independently
        if self._speak:
            self._speak.feed(text)
            self._speak.play_async()

    def _stream_response(self, messages: list[dict], system: str = "", token_callback=None) -> str:
        full_response = []
        max_tokens = _BASE_PREDICT * _REASONING_SCALE if self._reasoning else _BASE_PREDICT
        all_messages = [{"role": "system", "content": system}] + messages if system else messages
        self.last_usage = {
            "prompt_messages": all_messages,
            "completion_text": "",
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        }

        karaoke_text = bool(
            self._speak and token_callback and getattr(self._speak, "karaoke_text", False)
            and not self._reasoning
        )
        if self._speak:
            self._speak.start_speech_stream(token_callback if karaoke_text else None)

        sentence_buffer = ""
        stream_success = False
        try:
            stream = self._client.chat.completions.create(
                model=self._llm_model, messages=all_messages, stream=True,
                max_tokens=max_tokens,
                temperature=float(os.getenv("TEMPERATURE", 0.72)),
                top_p=float(os.getenv("TOP_P", 0.90)),
                stop=["<|im_end|>", "</s>", "[INST]"],
                timeout=LLM_TIMEOUT,
                extra_body={
                    "repeat_penalty": float(os.getenv("REPEAT_PENALTY", 1.15)),
                    "repeat_last_n":  int(os.getenv("REPEAT_LAST_N", 64)),
                    "top_k":          int(os.getenv("TOP_K", 40)),
                },
            )
            for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                token = (delta.content or "") if delta else ""
                
                if token_callback and token and not karaoke_text:
                    token_callback(token)
                
                full_response.append(token)
                
                if self._speak and token:
                    sentence_buffer += token
                    sentences, sentence_buffer = split_stream_sentences(sentence_buffer)
                    for sentence in sentences:
                        self._speak.feed_speech_stream(sentence)

            text = "".join(full_response).strip()
            if text:
                self.last_usage["completion_text"] = text
                stream_success = True
                if self._speak and sentence_buffer.strip():
                    self._speak.feed_speech_stream(sentence_buffer)
        except Exception as e:
            log.error(f"LLM stream failed: {e}")
        finally:
            if self._speak:
                self._speak.stop_speech_stream()

        if stream_success:
            return text

        fallback_text = self._fallback_completion(
            all_messages,
            max_tokens,
            "LLM stream failed or completed without content",
        )
        self._emit(fallback_text, token_callback=token_callback)
        return fallback_text

    def _fallback_completion(self, messages: list[dict], max_tokens: int, reason: str) -> str:
        """Try one non-streaming completion before surfacing the LLM error in chat."""
        try:
            resp = self._client.chat.completions.create(
                model=self._llm_model,
                messages=messages,
                stream=False,
                max_tokens=max_tokens,
                temperature=float(os.getenv("TEMPERATURE", 0.72)),
                top_p=float(os.getenv("TOP_P", 0.90)),
                stop=["<|im_end|>", "</s>", "[INST]"],
                timeout=LLM_TIMEOUT,
            )
            text = (resp.choices[0].message.content or "").strip()
            if text:
                usage = getattr(resp, "usage", None)
                self.last_usage.update({
                    "completion_text": text,
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                })
                log.warning("%s; recovered with non-streaming completion", reason)
                return text
            reason = f"{reason}; non-streaming completion was also empty"
        except Exception as e:
            reason = f"{reason}; non-streaming fallback failed: {e}"

        log.error(reason)
        return f"[LLM error] {reason}"

    def _is_data_intent(self, user_input: str) -> bool:
        # routing already set _pending_search_query in _route_intent
        return self._pending_search_query is not None

    def _sanitize_history(self, messages: list[dict]) -> list[dict]:
        if not messages: return []
        sanitized = [messages[0]]
        for msg in messages[1:]:
            if msg["role"] == sanitized[-1]["role"]: sanitized[-1] = msg
            else: sanitized.append(msg)
        while sanitized and sanitized[0]["role"] != "user": sanitized.pop(0)
        return sanitized

    def _store_async(self, user_input: str, response_text: str) -> None:
        """Queue a fire-and-forget memory write. The actual queue/worker
        thread now lives on AikoMemorize (core.memorize); this just wires
        up this instance's idle-tracking callables (is_active_turn /
        idle_since) so the write waits for a genuinely idle window before
        using the shared LLM for fact extraction. Kept as a method (rather
        than inlining self._memorize.queue_write(...) at every call site)
        because core.agentic's run_agentic_chat also calls
        owner._store_async(...) directly at the end of the agent loop.
        """
        self._memorize.queue_write(
            user_input,
            response_text,
            is_active_turn=self._active_turn.is_set,
            idle_since=lambda: self._last_chat_time,
        )


_STREAM_SENTENCE_END = set(".?!。？！")
_STREAM_CLOSERS = set("\"')]}」』”’")


def _is_stream_noise(char: str) -> bool:
    codepoint = ord(char)
    if char in {"\u200d", "\ufe0e", "\ufe0f", "\u20e3"}:
        return True
    if 0x1F000 <= codepoint <= 0x1FFFF:
        return True
    if 0x2600 <= codepoint <= 0x27BF:
        return True
    if 0x2300 <= codepoint <= 0x23FF:
        return True
    if 0x2B00 <= codepoint <= 0x2BFF:
        return True
    return unicodedata.category(char)[0] == "S"


def split_stream_sentences(buffer: str) -> tuple[list[str], str]:
    """
    Parse the streaming buffer, extract completed sentences, and return
    a list of completed sentences and the remaining partial sentence text.
    """
    sentences = []
    start = 0
    i = 0
    while i < len(buffer):
        char = buffer[i]
        if char in "\n\r":
            end = i + 1
        elif char in _STREAM_SENTENCE_END:
            end = i + 1
            while end < len(buffer) and buffer[end] in _STREAM_CLOSERS:
                end += 1
            if end == len(buffer):
                break
            if not (buffer[end].isspace() or _is_stream_noise(buffer[end])):
                i += 1
                continue
            while end < len(buffer) and (buffer[end].isspace() or _is_stream_noise(buffer[end])):
                end += 1
        else:
            i += 1
            continue

        sentence = buffer[start:end].strip()
        if sentence:
            sentences.append(sentence)
        start = end
        i = end

    remaining = buffer[start:]
    if not sentences and len(remaining) > 150:
        split_pts = [m.start() for m in re.finditer(r'[\s,、;:]', remaining)]
        if split_pts:
            split_pt = max([p for p in split_pts if p <= 150] or [split_pts[-1]])
            sentence = remaining[:split_pt + 1].strip()
            tail = remaining[split_pt + 1:]
            return ([sentence] if sentence else []), tail
    return sentences, remaining
