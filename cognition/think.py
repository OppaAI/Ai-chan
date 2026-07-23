"""
cognition/think.py

Aiko's chat facade.
  - Routes between single-shot chat and the agentic task loop in agentic.agentic.
  - Streams llama.cpp response to console + TTS simultaneously.
  - Queues long-term memory writes (delegated to memory.memorize's async write queue).
  - Owns scheduled-job callbacks and idle learner handoff (delegated to memory.learn).
  - Owns the proactive idle check-in state machine (config/proactive.yaml),
    which is also the "is Aiko resting" signal memory.learn's idle_learner_loop
    waits on before starting autonomous quick-study top-ups.

Memory + knowledge-base fetch:
  route() kicks off _fetch_memory_and_knowledge() on cognition's
  shared CONTEXT_POOL BEFORE intent is resolved, since every path
  (localchat/webchat/agentic) needs memory + KB regardless of which one
  intent routing picks. The resulting future is handed to whichever
  handler ends up running, so the fetch overlaps intent classification
  itself instead of waiting for it to finish first. Wiki/policy/skill/
  experience context is agentic-only and fetched separately, inside
  agentic.agentic.run_agentic_chat, only once intent has actually resolved
  to "agentic".
"""

from __future__ import annotations

import logging
import os
import json
import warnings
import hashlib

import numpy as np

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

from memory.memorize import AikoMemorize
from sensory.speak    import AikoSpeak
from agentic.tools    import web_search_context
from agentic.agentic  import run_agentic_chat
from agentic.wiki import wiki_knowledge_context_for
from memory.knowledge import knowledge_context_for
from cognition import CONTEXT_POOL
from system.log      import get_logger
from system.schedule import DueJob, register_system_handler
from system.userspace import current_user_id, current_display_name, user_profile_path, user_state_dir
from system import bioclock
from agentic.toolkit.social import run_scheduled_weekly_social
from cognition import reason
from memory import learn

log = get_logger(__name__)
register_system_handler("weekly_social", run_scheduled_weekly_social)

# ── boot labels ───────────────────────────────────────────────────────────────

BOOT_LABELS = {
    'think_start':    'Loading llama.cpp client + persona...',
    'think_warmup':   'Warming up language model...',
    'think_mem_wait': 'Waiting on memory system...',
    'think_prewarm':  'Warming semantic caches...',
}

# ── config ────────────────────────────────────────────────────────────────────

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8080/v1")
LLM_MODEL    = os.getenv("LLM_MODEL",    "ministral")
ROUTER_MODEL = os.getenv("ROUTER_MODEL", LLM_MODEL)
LLM_TIMEOUT  = float(os.getenv("LLM_TIMEOUT", 120))
# Stop sequences sent on every LLM call. Model-specific — only the real
# EOS matters; the third legacy token ([INST], raw instruct formatting)
# is never emitted in chat-completions mode and was dead weight. Default
# keeps the two common EOS tokens and drops [INST]; override per model
# via LLM_STOP_SEQUENCES (comma-separated) if a different EOS is needed.
LLM_STOP_SEQUENCES = [s.strip() for s in os.getenv("LLM_STOP_SEQUENCES", "</s>,<|im_end|>").split(",") if s.strip()]
CONTEXT_WINDOW_TURNS = int(os.getenv("CONTEXT_WINDOW_TURNS", 8))

# Shared default recall/knowledge depth across all three chat paths
# (localchat/webchat/agentic) — see _fetch_memory_and_knowledge below.
MEMORY_RECALL_LIMIT = int(os.getenv("MEMORY_RECALL_LIMIT", 3))
KNOWLEDGE_RECALL_LIMIT = int(os.getenv("KNOWLEDGE_RECALL_LIMIT", 3))
# Minimum recall score (see _MemoryBackend._rank_and_score's final_score in
# memory/memorize.py for the formula) a memory must clear to be included in
# context. Same numeric scale as memorize.py's MEMORY_RECALL_SCORE_THRESHOLD
# (~0.015) — that constant only decides quick-vs-wide search, this one
# actually filters weak individual results out of what gets returned.
# 0 = off (default) — no memory is ever dropped for being weak.
MEMORY_MIN_SCORE = float(os.getenv("MEMORY_MIN_SCORE", "0.0"))

_BASE_PREDICT    = int(os.getenv("LLM_MAX_TOKENS", os.getenv("BASE_PREDICT", 280)))
_REASONING_SCALE = int(os.getenv("REASONING_SCALE", 3))
_ROUTE_ENABLED = os.getenv("ROUTE_ENABLED", "1").lower() in {"1", "true", "yes", "on"}

# ROUTE_MODE selects the classification METHOD only (see yaml comment for
# the four options). It does not decide whether agentic is reachable —
# that's AGENTIC_MODE_ON, applied uniformly below regardless of method.
_ROUTE_VALID_MODES = {"semantic", "semantic_only", "llm", "llm_only"}
_ROUTE_MODE = os.getenv("ROUTE_MODE", "semantic").strip().lower()
if _ROUTE_MODE not in _ROUTE_VALID_MODES:
    log.warning("[route] invalid ROUTE_MODE=%r, defaulting to 'semantic'", _ROUTE_MODE)
    _ROUTE_MODE = "semantic"

# Whether "agentic" is a reachable routing outcome at all. Off = agentic
# is excluded from scoring AND from any LLM tie-break/classify, in every
# ROUTE_MODE, so requests degrade to webchat/localchat instead.
_AGENTIC_MODE_ON = os.getenv("AGENTIC_MODE_ON", "1").lower() in {"1", "true", "yes", "on"}

# Three separate instruct strings, one per embedding context
_ROUTE_INSTRUCT_TERNARY = "What kind of task or question is this?"  # used by route() for ternary intent routing

_SEMANTIC_ROUTE_MIN_GAP = float(os.getenv("ROUTE_MIN_GAP", "0.10"))
_SEMANTIC_LABEL_TOP_K = int(os.getenv("ROUTE_LABEL_TOP_K", "3"))
_ROUTE_VECTOR_CACHE_ENABLED = os.getenv("ROUTE_VECTOR_CACHE_ENABLED", "1").lower() in {"1", "true", "yes", "on"}
_ROUTE_VECTOR_CACHE_DIR = os.getenv("ROUTE_VECTOR_CACHE_DIR", "route_vectors")

_PERSONA_PATH = Path(__file__).resolve().parent.parent / "persona" / "SOUL.md"
_LOCAL_KNOWLEDGE_RE = re.compile(
    r"\b("
    r"aiko|your architecture|your hardware|your features?|your functions?|"
    r"what can you do|how do you work|how are you built|"
    r"knowledge base|wiki|docs?|readme|roadmap|install|config|"
    r"SOUL\.md|USER\.md|SKILLS?\.md|SCHEDULE\.md|"
    r"repo|repository|codebase|local files|your files"
    r")\b",
    re.IGNORECASE,
)

# ── conditional persona overrides ────────────────────────────────────────────
# SOUL.md is the always-loaded steady-state
# persona; the two override files below are only appended on turns that
# actually need them (mirrors _LOCAL_KNOWLEDGE_RE / _should_use_local_knowledge
# just below — same "gate the tokens, don't pay for them every turn" pattern).
_PERSONA_DIR = _PERSONA_PATH.parent
_PERSONA_CORE_PATH = _PERSONA_DIR / "SOUL.md"
_PERSONA_JP_PATH = _PERSONA_DIR / "JAPANESE_CHAT.md"
_PERSONA_CODE_PATH = _PERSONA_DIR / "CODING_CHAT.md"

_JAPANESE_TRIGGER_RE = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff]")
_CODE_TRIGGER_RE = re.compile(
    r"\b(debug|traceback|stack trace|error:|exception|refactor|"
    r"write (a|the) (function|script|class)|fix (this|my) code|"
    r"walk me through|\.py\b|\.js\b)\b",
    re.IGNORECASE,
)


def _load_static_persona() -> str:
    """Read the always-loaded persona core (SOUL.md — no per-user data,
    no conditional overrides).

    Task/tool policy lives in the agentic prompt so casual chat does not pay
    for agentic/schedule tokens on every turn. Japanese/coding overrides live
    in separate files and are appended per-turn by _conditional_persona_blocks
    only when triggered — see _current_system_prompt.
    """
    if not _PERSONA_CORE_PATH.exists():
        raise FileNotFoundError(f"SOUL.md not found at {_PERSONA_CORE_PATH}")
    return _PERSONA_CORE_PATH.read_text(encoding="utf-8").strip()


_persona_jp_cache: str | None = None
_persona_code_cache: str | None = None


def _conditional_persona_blocks(user_input: str) -> str:
    """Trigger-loaded persona overrides. Only paid for on turns that need them."""
    global _persona_jp_cache, _persona_code_cache
    blocks = []
    if _JAPANESE_TRIGGER_RE.search(user_input):
        if _persona_jp_cache is None:
            _persona_jp_cache = _PERSONA_JP_PATH.read_text(encoding="utf-8").strip()
        blocks.append(_persona_jp_cache)
    if _CODE_TRIGGER_RE.search(user_input):
        if _persona_code_cache is None:
            _persona_code_cache = _PERSONA_CODE_PATH.read_text(encoding="utf-8").strip()
        blocks.append(_persona_code_cache)
    return ("\n\n" + "\n\n".join(blocks)) if blocks else ""


_user_context_cache: dict[str, tuple[float, str]] = {}  # user_id -> (mtime, block)


def _load_user_context() -> tuple[str, str]:
    """Read the current turn's display name + profile block fresh, every call.

    Must be called from the turn/request context where current_user_id()
    already resolves to the real logged-in user — cached by (user_id, mtime)
    so repeated turns for the same user don't re-read an unchanged file.

    Returns (display_name, user_block) where user_block is either "" or a
    "\n\n"-prefixed profile chunk ready to append to the static persona.
    """
    display_name = current_display_name()
    user_path = user_profile_path()
    uid = current_user_id()
    if uid in _user_context_cache:
        cached_mtime, cached_block = _user_context_cache[uid]
        try:
            current_mtime = user_path.stat().st_mtime
        except OSError:
            current_mtime = 0.0
        if current_mtime == cached_mtime:
            return display_name, cached_block
    context_blocks = []
    if user_path.exists():
        raw = user_path.read_text(encoding="utf-8").strip()
        if raw:
            context_blocks.append(
                "<user_profile>\n"
                "Who you are speaking with — authoritative for identity. "
                "Never claim ignorance of this.\n\n"
                f"{raw}\n"
                "</user_profile>"
            )
    user_block = "\n\n" + "\n\n".join(context_blocks) if context_blocks else ""
    try:
        _user_context_cache[uid] = (user_path.stat().st_mtime, user_block)
    except OSError:
        pass
    return display_name, user_block


_DEBUG_PROMPT_DUMP_PATH = os.getenv("AIKO_DEBUG_PROMPT_DUMP", "/tmp/aiko_last_prompt.txt")

def _dump_full_prompt(debug: dict) -> None:
    if os.getenv("AIKO_DEBUG_FULL_PROMPT", "1").lower() not in {"1", "true", "yes", "on"}:
        return
    try:
        with open(_DEBUG_PROMPT_DUMP_PATH, "w", encoding="utf-8") as f:
            f.write(f"=== mode={debug.get('mode')} @ {datetime.now().isoformat()} ===\n\n")
            f.write("----- SYSTEM PROMPT (full, untruncated) -----\n")
            f.write(debug.get("system_prompt", "") + "\n\n")
            f.write("----- MEMORY -----\n")
            f.write(debug.get("memory_prompt", "") + "\n\n")
            f.write("----- KNOWLEDGE -----\n")
            f.write(debug.get("knowledge_prompt", "") + "\n\n")
            f.write("----- WEB -----\n")
            f.write(debug.get("web_prompt", "") + "\n\n")
            f.write("----- PREVIOUS CHAT MESSAGES -----\n")
            for m in debug.get("previous_chat_messages", []):
                f.write(f"[{m.get('role')}] {m.get('content')}\n")
    except Exception:
        log.exception("Failed to dump full prompt debug")
        

def _should_use_local_knowledge(user_input: str) -> bool:
    """Return True for normal-chat questions about Aiko's local docs/files.

    This is separate from the general memory+KB fetch every path already
    gets (see _fetch_memory_and_knowledge) — it's an additional, narrower
    lookup specifically for wiki-authored architecture/feature docs, gated
    so casual chat doesn't pay for it on every turn.
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

# load route examples (ternary intent only - tools/capability moved to agentic/router)
_EXAMPLES_PATH = Path(__file__).resolve().parent.parent / "agentic" / "router" / "intent_prompts.json"

def _load_route_examples() -> dict:
    with open(_EXAMPLES_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return {k: tuple(v) for k, v in data["ternary"].items()}

_ROUTE_TERNARY_EXAMPLES = _load_route_examples()

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
# system.config.load_config() has already populated these into the process
# environment by the time this module is imported (see system/wakeup.py /
# memory/learn.py — whichever entrypoint runs first calls it). We just read
# them here the same way every other module's config block does.
#
# Timezone is no longer configured here — every "now"/timezone lookup in
# this module goes through system.bioclock, the app-wide single source of
# truth (config/bioclock.yaml).




# ── think ─────────────────────────────────────────────────────────────────────

class AikoThink:
    def __init__(self) -> None:
        self._client    = OpenAI(base_url=LLM_BASE_URL, api_key="not-needed")
        self._llm_model = LLM_MODEL
        self._router_model = ROUTER_MODEL
        self._memorize  = None    # injected later via set_memorize() — see system/wakeup.py
        self._speak     = None    # injected later via set_speak()    — see system/wakeup.py
        # Guards self._speak against the toggle-vs-background-thread race.
        # set_speak() is called from the main thread (main.py's /voice
        # toggle). Readers snapshot self._speak under the lock so a toggle
        # landing mid-read can't produce a stale ref or a None mismatch.      
        self._speak_lock = threading.Lock()
        self._memorize_lock = threading.Lock()

        self._persona   = _load_static_persona()
        self._history:  list[dict] = []
        self._history_lock = threading.Lock()
        # Cache of (labels, embedding_matrix) per (example-corpus-id, instruct)
        # pair — built via reason.embed_example_matrix, which always
        # re-embeds; caching the result here avoids paying that cost on
        # every routing call for a static example corpus.
        self._semantic_example_cache: dict = {}
        self._semantic_example_cache_lock = threading.RLock()
        self._active_user_ids: set[str] = set()
        self._active_users_lock = threading.Lock()
        self._reasoning = False
        self.last_usage: dict = {}
        self.last_prompt_debug: dict = {}
        self._last_chat_time = time.time()
      
        self._idle_learner_thread: threading.Thread | None = None
        self._warmup_thread: threading.Thread | None = None
      
        # ── rest-signal state for learn.idle_learner_loop ───────────────────
        # The proactive idle check-in state machine lives in main.py's
        # ProactiveIdleRunner. That runner sets this flag via
        # set_proactive_resting() so learn.idle_learner_loop can see when
        # Aiko is "resting" and pause autonomous study. The flag is cleared
        # by _note_user_activity() on every normal turn.
        self._proactive_lock = threading.Lock()
        self._proactive_resting = False
      
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

    def start_warmup(self) -> None:
        """Kick off the LLM warmup call. Call once, right after construction."""
        if self._warmup_thread is not None:
            return
        self._warmup_thread = threading.Thread(target=self._warmup_llm, daemon=True)
        self._warmup_thread.start()
  
    def start_idle_learner(self) -> None:
        """Start the background idle-learning loop. Call only after
        set_memorize() has been called — the loop reads self._memorize
        on its first iteration."""
        if self._idle_learner_thread is not None:
            return
        if self._get_memorize() is None:
            log.warning("[think] Memory unavailable — idle learner not started.")
            return
        self._idle_learner_thread = threading.Thread(
            target=learn.idle_learner_loop, args=(self,), daemon=True
        )
        self._idle_learner_thread.start()
  
    def _current_system_prompt(self, user_input: str = "") -> str:
        """Assemble this turn's system prompt: static persona core + fresh
        per-user context + any conditional overrides this input triggers.

        Call only from within a turn where current_user_id()/current_display_name()
        already resolve to the real caller — never at construction time.
        """
        display_name, user_block = _load_user_context()
        base = self._persona.replace("USER_ID_HERE", display_name) + user_block
        return base + _conditional_persona_blocks(user_input)
  
    # ── public api ────────────────────────────────────────────────────────────

    def route(self, user_input: str, token_callback=None) -> str:
        """Main entry point. Ternary routing.

        Memory + knowledge-base fetch is kicked off here, BEFORE intent is
        known — every path (localchat/webchat/agentic) needs both, so
        there's no reason to wait for _route_intent() to finish before
        starting them. The future is threaded through to whichever handler
        ends up running.

        Per-user-active tracking: multiple users' turns can run concurrently
        (e.g. agentic loop for one user, quick chat for another). Shared
        state (_history, _speak) has its own per-resource lock.
        """
        user_id = current_user_id()
        with self._active_users_lock:
            self._active_user_ids.add(user_id)
        self._note_user_activity()
        try:
            embedder = self._get_memorize()._mem._embedder
            query_vec = embedder.embed_query(user_input)
            mem_kb_future = CONTEXT_POOL.submit(
                self._fetch_memory_and_knowledge, user_input, query_vec
            )

            intent = self._route_intent(user_input)
            log.info("[route] intent=%s", intent)

            if intent == "agentic":
                return self.agentic_chat(user_input, token_callback=token_callback, mem_kb_future=mem_kb_future, query_vec=query_vec)
            elif intent == "webchat":
                return self.webchat(user_input, token_callback=token_callback, mem_kb_future=mem_kb_future)
            else:  # localchat
                return self.chat(user_input, token_callback=token_callback, _skip_search=True, mem_kb_future=mem_kb_future)
        finally:
            with self._active_users_lock:
                self._active_user_ids.discard(user_id)
                if not self._active_user_ids:
                    self._last_chat_time = time.time()

    def _fetch_memory_and_knowledge(
        self, user_input: str, query_vector: np.ndarray | None = None,
        mem_limit: int = MEMORY_RECALL_LIMIT, know_limit: int = KNOWLEDGE_RECALL_LIMIT,
    ) -> tuple[list[dict], str]:
        """Fetch long-term memory + learned-knowledge (KB) concurrently.

        Both are independent reads against separate stores (memory.db /
        knowledge.db) with no dependency on user intent, so route() fires
        this off before intent routing even runs and hands the resulting
        future to whichever path (agentic/webchat/localchat) gets picked.
        Callers that run standalone (e.g. a scheduled agentic job with no
        prior route() call) can call this directly instead.

        query_vector — pre-computed _QUERY_INSTRUCT embedding of user_input,
        avoids a redundant HTTP call inside _MemoryBackend.search().

        Returns (memories, knowledge_block).
        """
        embedder = self._get_memorize()._mem._embedder
        mem_future = CONTEXT_POOL.submit(self._get_memorize().search, user_input, limit=mem_limit, query_vector=query_vector)
        know_future = CONTEXT_POOL.submit(
            knowledge_context_for, user_input, limit=know_limit, max_chars=2000, embedder=embedder
        )
        try:
            memories = mem_future.result()
        except Exception as e:
            log.error("Memory search failed: %s", e)
            know_future.cancel()
            memories = []

        if MEMORY_MIN_SCORE > 0:
            before = len(memories)
            memories = [m for m in memories if m.get("_recall_score", 0.0) >= MEMORY_MIN_SCORE]
            if len(memories) < before:
                log.debug(
                    "[memory] filtered %d/%d below MEMORY_MIN_SCORE=%.4f",
                    before - len(memories), before, MEMORY_MIN_SCORE,
                )

        try:
            knowledge_block = know_future.result()
        except Exception as e:
            log.error("Knowledge lookup failed: %s", e)
            knowledge_block = "<knowledge_context>\nLookup failed.\n</knowledge_context>"
        return memories, knowledge_block

    def _resolve_mem_kb(self, user_input: str, mem_kb_future) -> tuple[list[dict], str]:
        """Resolve a pending memory+KB future, or fetch directly if this
        handler was called standalone (no future supplied by route())."""
        if mem_kb_future is not None:
            try:
                return mem_kb_future.result()
            except Exception as e:
                log.error("Memory/KB fetch failed: %s", e)
                return [], "<knowledge_context>\nLookup failed.\n</knowledge_context>"
        return self._fetch_memory_and_knowledge(user_input, query_vector=None)

    def _note_user_activity(self) -> None:
        """Clear the rest flag on real user activity so learn.idle_learner_loop
        sees Aiko is no longer resting and can resume autonomous study."""
        with self._proactive_lock:
            self._proactive_resting = False

    def is_proactive_resting(self) -> bool:
        """True when Aiko is resting and should not start autonomous study.
        Set by set_proactive_resting() (called from main.py's
        ProactiveIdleRunner) and cleared by _note_user_activity() on
        every normal turn. Polled by learn.idle_learner_loop."""
        return self._proactive_resting

    def set_proactive_resting(self, resting: bool) -> None:
        """Set/clear the rest flag for learn.idle_learner_loop.
        Called from main.py's ProactiveIdleRunner."""
        with self._proactive_lock:
            self._proactive_resting = resting

    def _route_intent(self, user_input: str) -> str:
        """Ternary routing: single embedding, three-way decision with a
        high-confidence margin so a close call doesn't get committed to
        agentic (or webchat) just because it happened to be checked first.

        The close-vector label-scoring math itself (normalize + batched
        matmul + top-k mean per label) lives in cognition.reason; this method
        only owns the routing policy (thresholds, gap, LLM tie-break).
        ROUTE_MODE picks the classification method; AGENTIC_MODE_ON gates
        whether "agentic" can ever be the result, independent of method.
        """
    
        if not _ROUTE_ENABLED:
            return "localchat"
    
        instruct = _ROUTE_INSTRUCT_TERNARY
        embedder = self._get_memorize()._mem._embedder
        query_vec = embedder.embed_query(user_input, instruct=instruct)
        labels, example_vecs = self._semantic_example_vectors(_ROUTE_TERNARY_EXAMPLES, instruct)
        scores = reason.label_scores_topk(query_vec, labels, example_vecs, top_k=_SEMANTIC_LABEL_TOP_K)

        if not _AGENTIC_MODE_ON:
            # Remove agentic from consideration before ranking, so the
            # existing gap/threshold logic naturally degenerates into a
            # webchat-vs-localchat decision — no separate code path needed.
            scores.pop("agentic", None)
      
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
    
        # Above threshold but too close to call cleanly.
        ambiguous = (
            (agentic_score >= agentic_threshold or webchat_score >= webchat_threshold)
            and gap < _SEMANTIC_ROUTE_MIN_GAP
        )
        if ambiguous:
            if _ROUTE_MODE == "semantic_only":
                # Deterministic mode: no LLM call on ambiguity.
                log.debug("[route] semantic_only: ambiguous gap, defaulting localchat")
                return "localchat"
            if _ROUTE_MODE == "llm":
                return self._classify_ternary_intent_llm(user_input, allow_agentic=_AGENTIC_MODE_ON)
            # "semantic" mode's original binary tie-break. If agentic is
            # off, there's nothing left for this binary check to decide
            # (it only ever distinguishes agentic vs chat), so skip the
            # LLM call entirely rather than spend it on a moot question.
            if not _AGENTIC_MODE_ON:
                return "localchat"
            llm_label = self._classify_agent_intent(user_input)
            return "agentic" if llm_label == "agentic" else "localchat"
    
        return "localchat"

    def _semantic_example_vectors(self, examples_by_label: dict, instruct: str) -> tuple[list[str], object]:
        """Return cached route-example vectors.

        Hot turns use the in-memory cache. Cold boots can reuse a per-user
        on-disk NumPy archive keyed by the route examples, instruct string, and
        embedding backend metadata when ROUTE_VECTOR_CACHE_ENABLED=1. If the
        cache is missing/stale/unreadable, Aiko recomputes and overwrites it.
        """
        cache_key = (id(examples_by_label), instruct)
        with self._semantic_example_cache_lock:
            cached = self._semantic_example_cache.get(cache_key)
            if cached is not None:
                return cached

            embedder = self._get_memorize()._mem._embedder
            disk_path = self._route_vector_cache_path(examples_by_label, instruct, embedder)
            if disk_path is not None and disk_path.exists():
                try:
                    with disk_path.open("rb") as f:
                        data = np.load(f, allow_pickle=False)
                        cached = (list(data["labels"].astype(str)), data["vectors"])
                    self._semantic_example_cache[cache_key] = cached
                    return cached
                except Exception as exc:
                    log.debug("[route] ignoring stale vector cache %s: %s", disk_path, exc)

            labels, vectors = reason.embed_example_matrix(embedder, examples_by_label, instruct=instruct)
            cached = (labels, vectors)
            self._semantic_example_cache[cache_key] = cached
            if disk_path is not None:
                try:
                    disk_path.parent.mkdir(parents=True, exist_ok=True)
                    with disk_path.open("wb") as f:
                        np.savez(f, labels=np.asarray(cached[0], dtype=str), vectors=cached[1])
                except Exception as exc:
                    log.debug("[route] could not write vector cache %s: %s", disk_path, exc)
            return cached

    def _route_vector_cache_path(self, examples_by_label: dict, instruct: str, embedder) -> Path | None:
        if not _ROUTE_VECTOR_CACHE_ENABLED:
            return None
        try:
            payload = {
                "examples": examples_by_label,
                "instruct": instruct,
                "embedder": {
                    "class": type(embedder).__name__,
                    "model": getattr(embedder, "model", None) or getattr(embedder, "model_name", None) or getattr(embedder, "name", None),
                    "dims": os.getenv("EMBED_DIMS", ""),
                },
            }
            digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:24]
            raw_dir = Path(_ROUTE_VECTOR_CACHE_DIR)
            base = raw_dir if raw_dir.is_absolute() else user_state_dir(current_user_id()) / raw_dir
            return base / f"{digest}.npz"
        except Exception:
            return None

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
                    "Message: 'open SOUL.md and show the persona block'\n"
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

    def _classify_ternary_intent_llm(self, user_input: str, allow_agentic: bool = True) -> str:
        """LLM classify for ROUTE_MODE=llm (ambiguity tie-break) and
        ROUTE_MODE=llm_only (sole routing decision, every turn).

        allow_agentic=False collapses this to a binary webchat/chat
        classification — agentic is never offered as a label, so
        AGENTIC_MODE_ON=0 holds regardless of which ROUTE_MODE is active.
        """
        if allow_agentic and _AGENTIC_ROUTE_RE.search(user_input):
            return "agentic"

        if allow_agentic:
            labels_line = "Labels: [agentic, webchat, chat]"
            guidance = (
                "agentic = the message asks for an action/task (write, save, "
                "schedule, debug, plan, remind, research-and-report).\n"
                "webchat = the message needs current/external information "
                "(news, prices, scores, recent releases, real-time facts) but "
                "is not itself a task.\n"
                "chat = casual conversation, opinions, or something answerable "
                "from general/persona knowledge alone.\n\n"
                "Message: 'set a reminder for 9pm'\n"
                "Label: agentic\n\n"
                "Message: 'debug why asyncio.run() hangs'\n"
                "Label: agentic\n\n"
                "Message: 'what's the weather in Vancouver right now'\n"
                "Label: webchat\n\n"
                "Message: 'who won the game last night'\n"
                "Label: webchat\n\n"
                "Message: 'what do you think about minimalism'\n"
                "Label: chat\n\n"
                "Message: 'explain semaphores from memory'\n"
                "Label: chat\n\n"
            )
            valid = {"agentic", "webchat", "chat"}
        else:
            labels_line = "Labels: [webchat, chat]"
            guidance = (
                "webchat = the message needs current/external information "
                "(news, prices, scores, recent releases, real-time facts).\n"
                "chat = casual conversation, opinions, or something answerable "
                "from general/persona knowledge alone.\n\n"
                "Message: 'what's the weather in Vancouver right now'\n"
                "Label: webchat\n\n"
                "Message: 'who won the game last night'\n"
                "Label: webchat\n\n"
                "Message: 'what do you think about minimalism'\n"
                "Label: chat\n\n"
                "Message: 'explain semaphores from memory'\n"
                "Label: chat\n\n"
            )
            valid = {"webchat", "chat"}

        try:
            resp = self._client.chat.completions.create(
                model=self._router_model,
                messages=[{"role": "user", "content": (
                    f"Message: {user_input!r}\n\n"
                    "Output only the route label. No explanation.\n"
                    f"{labels_line}\n"
                    f"{guidance}"
                    "Label:"
                )}],
                stream=False, max_tokens=6, temperature=0.0, top_p=1.0, timeout=LLM_TIMEOUT,
            )
            label = (resp.choices[0].message.content or "chat").strip().lower()
            label = re.sub(r"[^a-z_].*$", "", label)
            if label not in valid:
                return "localchat"
            return "localchat" if label == "chat" else label
        except Exception as e:
            log.warning("Ternary LLM routing failed: %s", e)
            return "localchat"
  
    def agentic_chat(self, user_input: str, token_callback=None, mem_kb_future=None, query_vec: np.ndarray | None = None) -> str:
        """Delegate task-mode execution to agentic.agentic."""
        user_id = current_user_id()
        with self._active_users_lock:
            self._active_user_ids.add(user_id)
        try:
            embedder = self._get_memorize()._mem._embedder
            cap_vec = embedder.embed_query(
                user_input,
                instruct="Which capability/tool domain applies to this task?",
            )
            return run_agentic_chat(self, user_input, token_callback=token_callback, mem_kb_future=mem_kb_future, query_vec=query_vec, cap_vec=cap_vec)
        finally:
            with self._active_users_lock:
                self._active_user_ids.discard(user_id)
                if not self._active_user_ids:
                    self._last_chat_time = time.time()
              
    def webchat(self, user_input: str, token_callback=None, mem_kb_future=None) -> str:
        """Web-aware chat: web_search + optional webfetch fallback."""
        if self._speak and self._speak.is_playing():
            self._speak.stop()
        
        # Memory + KB — either resolved from route()'s pre-intent future,
        # or fetched directly if this was called standalone.
        memories, knowledge_block = self._resolve_mem_kb(user_input, mem_kb_future)
        memory_block = self._get_memorize().format_for_context(memories)

        # Build base system (persona + memory + knowledge)
        system = self._current_system_prompt()
        system += "\n\n" + bioclock.current_datetime_block()
        if memory_block:
            system = f"{system}\n\n{memory_block}"
        system = f"{system}\n\n{knowledge_block}"
        
        # Search directly with the raw user input — same approach as /web.
        # No LLM-based query condensation: it adds latency, depends on a
        # small router model that often produces worse queries than the
        # original text, and /web already proves the raw path works.
        if token_callback:
            token_callback(f"__SEARCHING__:{user_input}\n")
        
        context = web_search_context(user_input, max_results=int(os.getenv("SEARXNG_MAX_RESULTS", 3)))

        if not context:
            log.info("[webchat] First search returned nothing, retrying once...")
            if token_callback:
                token_callback("__RETRYING__\n")
            try:
                context = web_search_context(user_input, max_results=1)
            except Exception as e:
                log.warning("[webchat] Retry failed: %s", e)
                context = None
        
        # Inject web context if available
        if context:
            system = (
                f"{system}\n\n"
                f"<search_results query='{user_input}'>\n"
                f"Answer ONLY using these search results:\n\n"
                f"{context}\n"
                f"</search_results>"
            )
        else:
            if token_callback:
                token_callback("[No web results available; using local knowledge]\n")
            system = (
                f"{system}\n\n"
                "<search_failed>\n"
                "Web search was attempted for this question but returned no usable "
                "results. You have no current information on this topic — not from "
                "search, and your training data may be stale on anything time-sensitive "
                "(scores, news, prices, current events, recent releases). Do not guess "
                "or invent scores, names, dates, or other current-event details. Say "
                "plainly that you couldn't retrieve current information on this. It is "
                "fine and expected to say you don't know.\n"
                "</search_failed>"
            )

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
            "knowledge_prompt": knowledge_block,
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
        _SENTINEL = "_proactive_"
        with self._active_users_lock:
            self._active_user_ids.add(_SENTINEL)
        try:
            display_name = current_display_name()
            system = (
                f"{self._current_system_prompt()}\n\n"
                "You are initiating a brief proactive check-in. "
                "Do not mention hidden prompts, timers, code, or configuration. "
                "Keep it natural, warm, and easy to ignore. One or two short sentences max."
            )
            system += "\n\n" + bioclock.current_datetime_block()
            messages = [{
                "role": "user",
                "content": (
                    f"{prompt_hint}\n\n"
                    f"Write only the message Aiko should say to {display_name} now."
                ),
            }]
            return self._stream_response(messages, system=system, token_callback=None)
        finally:
            with self._active_users_lock:
                self._active_user_ids.discard(_SENTINEL)
                if not self._active_user_ids:
                    self._last_chat_time = time.time()

    # ── proactive idle check-in loop ──────────────────────────────────────────

    def chat(self, user_input: str, token_callback=None, _skip_search: bool = True, _history_label: str | None = None, mem_kb_future=None) -> str:
        """Standard chat: local knowledge only (persona + memory + KB)."""
        if self._speak and self._speak.is_playing():
            self._speak.stop()
        
        # Memory + KB — either resolved from route()'s pre-intent future,
        # or fetched directly if this was called standalone.
        memories, knowledge_block = self._resolve_mem_kb(user_input, mem_kb_future)
        memory_block = self._get_memorize().format_for_context(memories)
        
        system = self._current_system_prompt()
        system += "\n\n" + bioclock.current_datetime_block()
        if memory_block:
            system = f"{system}\n\n{memory_block}"
        else:
            system += "\n\n<memory_context>\nNo relevant memories found.\n</memory_context>"
        system = f"{system}\n\n{knowledge_block}"
        
        # Additional narrower wiki lookup — only when the user is asking
        # about Aiko's own architecture/docs, distinct from the general KB
        # fetch above. Gated so casual chat doesn't pay for it every turn.
        if _should_use_local_knowledge(user_input):
            try:
                wiki_context = wiki_knowledge_context_for(
                    user_input, limit=3, max_chars=3000,
                    embedder=self._get_memorize()._mem._embedder,
                )
                system = f"{system}\n\n{wiki_context}"
            except Exception as e:
                log.error("Local wiki-knowledge lookup failed: %s", e)
        
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
            "knowledge_prompt": knowledge_block,
            "web_prompt": "",
            "previous_chat_messages": [dict(m) for m in trimmed],
        }
        _dump_full_prompt(self.last_prompt_debug)
    
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

    def set_speak(self, speak) -> None:
        with self._speak_lock:
            self._speak = speak
          
    def set_memorize(self, memorize) -> None:
        """Inject the memory backend after boot. Thread-safe against concurrent reads."""
        with self._memorize_lock:
            self._memorize = memorize
    
    def _get_memorize(self):
        with self._memorize_lock:
            return self._memorize
  
    def wait_for_memory(self, timeout: float | None = None) -> bool:
        """Block until AikoMemorize's async write queue drains, or timeout
        elapses. The queue itself now lives in memory.memorize; this is a
        thin passthrough kept for call sites that only know about the
        AikoThink instance. No longer called from agentic.agentic's turn
        start (see run_agentic_chat) — draining there was removed since
        the write's own idle-grace window plus real turn latency meant it
        rarely caught anything. Still available for any caller that
        genuinely needs to block until writes are flushed (e.g. shutdown).
        """
        return self._get_memorize().wait_for_writes(timeout=timeout)

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
                time.sleep(float(os.getenv("EMIT_DELAY", 0.005)))

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
                stop=LLM_STOP_SEQUENCES,
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
                stop=LLM_STOP_SEQUENCES,
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
        thread now lives on AikoMemorize (memory.memorize); this just wires
        up this instance's idle-tracking callables (is_active_turn /
        idle_since) so the write waits for a genuinely idle window before
        using the shared LLM for fact extraction. Kept as a method (rather
        than inlining self._get_memorize().queue_write(...) at every call site)
        because agentic.agentic's run_agentic_chat also calls
        owner._store_async(...) directly at the end of the agent loop.
        """
        def _is_any_active():
            with self._active_users_lock:
                return bool(self._active_user_ids)
        self._get_memorize().queue_write(
            user_input,
            response_text,
            is_active_turn=_is_any_active,
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
