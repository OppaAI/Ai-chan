"""
core/think.py

Aiko's chat facade.
  - Routes between single-shot chat and the agentic task loop in core.agentic.
  - Streams llama.cpp response to console + TTS simultaneously.
  - Records daily experience turns and queues long-term memory writes.
  - Owns scheduled-job callbacks and idle learner handoff.
"""

import logging
import os
import json
import warnings

warnings.filterwarnings("ignore")
logging.getLogger("phonemizer").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

from datetime import datetime
from openai import OpenAI
from pathlib import Path
import queue
import re
import threading
import time
import unicodedata

import numpy as np

from core.memorize import AikoMemorize
from core.speak    import AikoSpeak
from core.tools    import deep_search
from core.agentic  import run_agentic_chat
from core.skills   import load_skills
from core.log      import get_logger
from core.schedule import DueJob, ScheduleRunner
from core.experience import append_chat_turn

log = get_logger(__name__)

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
_IDLE_LEARN_SECONDS = int(os.getenv("IDLE_LEARN_SECONDS", 1800))
_MEMORY_WRITE_IDLE_GRACE = float(os.getenv("MEMORY_WRITE_IDLE_GRACE", 3.0))
_MEMORY_WRITE_MAX_WAIT = float(os.getenv("MEMORY_WRITE_MAX_WAIT", 45.0))
# Route task-vs-chat turns semantically by default using the same embedding
# model as memory/RAG. Set ROUTE_MODE=llm to let the local LLM classify
# instead, or ROUTE_MODE=chat to disable autonomous routing.
_ROUTE_ENABLED = os.getenv("ROUTE_ENABLED", "1").lower() in {"1", "true", "yes", "on"}
_ROUTE_MODE = os.getenv("ROUTE_MODE", "semantic").strip().lower()

# Three separate instruct strings, one per embedding context
_ROUTE_INSTRUCT_BINARY = "Does this message ask someone to perform a task or action, or is it just conversation?"
_ROUTE_INSTRUCT_TOOL   = "This is a task request. What kind of task is being asked for?"
_ROUTE_INSTRUCT_SEARCH = "Does answering this require looking up current or external data?"

_SEMANTIC_ROUTE_THRESHOLD = float(os.getenv("ROUTE_SEMANTIC_THRESHOLD", "0.65"))
_SEMANTIC_SEARCH_THRESHOLD = float(os.getenv("SEARCH_SEMANTIC_THRESHOLD", "0.65"))
_SEMANTIC_ROUTE_MIN_GAP = float(os.getenv("ROUTE_MIN_GAP", "0.10"))
_SEMANTIC_TOOL_MIN_GAP = float(os.getenv("ROUTE_TOOL_MIN_GAP", "0.015"))
_SEMANTIC_SEARCH_MIN_GAP = float(os.getenv("ROUTE_SEARCH_MIN_GAP", "0.010"))
_SEMANTIC_LABEL_TOP_K = int(os.getenv("ROUTE_LABEL_TOP_K", "3"))

_PERSONA_PATH = Path(__file__).resolve().parent.parent / "persona" / "soul.md"
_USER_PATH = Path(__file__).resolve().parent.parent / "persona" / "user.md"
_SKILLS_PATH  = Path(__file__).resolve().parent.parent / "persona" / "skills.md"
_SCHEDULE_PATH = Path(__file__).resolve().parent.parent / "persona" / "schedule.md"

def _load_persona() -> str:
    """Read persona and skills definitions."""
    if not _PERSONA_PATH.exists():
        raise FileNotFoundError(f"soul.md not found at {_PERSONA_PATH}")
    persona = _PERSONA_PATH.read_text(encoding="utf-8").strip()
    
    context_blocks = []
    for path in (_USER_PATH, _SKILLS_PATH, _SCHEDULE_PATH):
        if path.exists():
            context_blocks.append(load_skills(path).strip())
    skills_block = "\n\n" + "\n\n".join(context_blocks) if context_blocks else ""

    user_id = os.getenv("USER_ID", "OppaAI")
    today   = datetime.now().strftime("%B %d, %Y")
    return persona.replace("USER_ID_HERE", user_id).replace("TODAY_HERE", today) + skills_block

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

def _load_route_examples() -> tuple[dict, dict, dict]:
    with open(_EXAMPLES_PATH, encoding="utf-8") as f:
        data = json.load(f)
    binary = {k: tuple(v) for k, v in data["binary"].items()}
    tools  = {k: tuple(v) for k, v in data["tools"].items()}
    search = {k: tuple(v) for k, v in data["search"].items()}
    return binary, tools, search

_ROUTE_BINARY_EXAMPLES, _ROUTE_TOOL_EXAMPLES, _ROUTE_SEARCH_EXAMPLES = _load_route_examples()


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
        self._route_chat_classified: str | None = None
        self._semantic_example_cache: dict[int, tuple[list[str], np.ndarray]] = {}
        self._semantic_example_cache_lock = threading.RLock()
        self._active_turn = threading.Event()
        self._reasoning = False
        self._mem_queue  = queue.Queue()
        self._mem_worker = threading.Thread(target=self._mem_write_loop, daemon=True)
        self._mem_worker.start()

        self._last_chat_time = time.time()
        self._idle_learner_thread = threading.Thread(target=self._idle_learner_loop, daemon=True)
        self._idle_learner_thread.start()

        self._reminders = ScheduleRunner(on_due=self.handle_scheduled_job)
        self._reminders.start()

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
        """Main entry point. Uses semantic intent routing."""
        self._last_chat_time = time.time()
        self._active_turn.set()
        try:
            intent = self._route_intent(user_input)
            if intent != "chat":
                log.info("[route] Agent intent=%s for: %r", intent, user_input)
                return self.agentic_chat(user_input, token_callback=token_callback)
            return self.chat(user_input, token_callback=token_callback)
        finally:
            self._last_chat_time = time.time()
            self._active_turn.clear()

    def _route_intent(self, user_input: str) -> str:
        if not _ROUTE_ENABLED or _ROUTE_MODE in {"0", "off", "false", "chat", "disabled"}:
            return "chat"

        if _ROUTE_MODE == "llm":
            label = self._classify_agent_intent(user_input)
            if label != "chat":
                return label
            # still need to check websearch even in llm mode
            self._pending_search_query = user_input if self._needs_websearch(user_input) else None
            return "chat"

        # Stage 1 — semantic
        if self._is_agentic(user_input):
            # Stage 2a
            return self._classify_agentic_tool(user_input)

        # Stage 2b
        self._pending_search_query = user_input if self._needs_websearch(user_input) else None
        if _ROUTE_MODE != "semantic_only" and self._semantic_binary_is_ambiguous(user_input):
            llm_label = self._classify_agent_intent(user_input)
            if llm_label != "chat":
                return llm_label
        return "chat"

    @staticmethod
    def _normalized_vector(vector: list[float]) -> np.ndarray:
        """Return one L2-normalized float32 embedding vector for cosine scoring."""
        arr = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(arr))
        if norm <= 1e-12:
            return arr
        return arr / norm

    @staticmethod
    def _normalized_matrix(vectors: list[list[float]]) -> np.ndarray:
        """Return row-wise L2-normalized float32 embeddings for vectorized cosine."""
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.size == 0:
            return matrix.reshape(0, 0)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.clip(norms, 1e-12, None)

    def _is_agentic(self, user_input: str) -> bool:
        """Stage 1: binary chat vs agentic."""
        try:
            best_label, best_score, gap = self._semantic_best_label(
                user_input, _ROUTE_BINARY_EXAMPLES, _ROUTE_INSTRUCT_BINARY
            )
            log.debug("[route/binary] best=%s score=%.3f gap=%.3f for: %r", best_label, best_score, gap, user_input)
            return best_label == "agentic" and best_score >= _SEMANTIC_ROUTE_THRESHOLD and gap >= _SEMANTIC_ROUTE_MIN_GAP
        except Exception as e:
            log.warning("Binary intent routing failed: %s", e)
            return False

    def _semantic_binary_is_ambiguous(self, user_input: str) -> bool:
        """Return True when chat and agentic are too close for semantic-only routing."""
        try:
            _best_label, best_score, gap = self._semantic_best_label(user_input, _ROUTE_BINARY_EXAMPLES, _ROUTE_INSTRUCT_BINARY)
            return best_score >= _SEMANTIC_ROUTE_THRESHOLD and gap < _SEMANTIC_ROUTE_MIN_GAP
        except Exception as e:
            log.warning("Binary ambiguity check failed: %s", e)
            return False

    def _classify_agentic_tool(self, user_input: str) -> str:
        """Stage 2a: which agentic tool, called only when agentic is confirmed."""
        try:
            best_label, best_score, gap = self._semantic_best_label(
                user_input, _ROUTE_SEARCH_EXAMPLES, _ROUTE_INSTRUCT_TOOL
            )            
            log.debug("[route/tool] best=%s score=%.3f gap=%.3f for: %r", best_label, best_score, gap, user_input)
            if best_score >= _SEMANTIC_ROUTE_THRESHOLD and gap >= _SEMANTIC_TOOL_MIN_GAP:
                return best_label
            # fallback to LLM classifier if score is weak
            return self._classify_agent_intent(user_input)
        except Exception as e:
            log.warning("Tool intent routing failed: %s", e)
            return self._classify_agent_intent(user_input)

    def _needs_websearch(self, user_input: str) -> bool:
        """Stage 2b: does this chat turn need a websearch, called only when chat is confirmed."""
        try:
            best_label, best_score, gap = self._semantic_best_label(
                user_input, _ROUTE_SEARCH_EXAMPLES, _ROUTE_INSTRUCT_SEARCH
            )
            log.debug("[route/search] best=%s score=%.3f gap=%.3f for: %r", best_label, best_score, gap, user_input)
            return best_label == "data" and best_score >= _SEMANTIC_SEARCH_THRESHOLD and gap >= _SEMANTIC_SEARCH_MIN_GAP
        except Exception as e:
            log.warning("Search intent routing failed: %s", e)
            return False

    def _semantic_all_scores(self, user_input: str, examples_by_label: dict, instruct: str) -> dict[str, float]:
        """Return top-k mean cosine score per label for stable close-vector routing."""
        query_vector = self._normalized_vector(
            self._memorize._mem._embedder.embed_query(user_input, instruct=instruct).tolist()
        )
        labels, example_vectors = self._semantic_example_vectors(examples_by_label, instruct)
        if example_vectors.size == 0:
            return {}
        raw_scores = example_vectors @ query_vector
        from collections import defaultdict
        label_scores: dict[str, list[float]] = defaultdict(list)
        for label, score in zip(labels, raw_scores):
            label_scores[label].append(float(score))

        scores: dict[str, float] = {}
        top_k = max(1, _SEMANTIC_LABEL_TOP_K)
        for label, values in label_scores.items():
            strongest = sorted(values, reverse=True)[:top_k]
            scores[label] = sum(strongest) / len(strongest)
        return scores

    def _semantic_best_label(self, user_input: str, examples_by_label: dict, instruct: str) -> tuple[str, float, float]:
        """Return the best semantic label, its score, and the gap to second best."""
        scores = self._semantic_all_scores(user_input, examples_by_label, instruct)
        if not scores:
            return "chat", 0.0, 0.0
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_label, best_score = ranked[0]
        gap = best_score - ranked[1][1] if len(ranked) > 1 else 1.0
        return best_label, best_score, gap

    def _semantic_example_vectors(self, examples_by_label: dict, instruct: str) -> tuple[list[str], np.ndarray]:
        """Return cached embeddings for a static semantic example corpus."""
        cache_key = (id(examples_by_label), instruct)
        cache = getattr(self, "_semantic_example_cache", None)
        cache_lock = getattr(self, "_semantic_example_cache_lock", None)
        if cache is None:
            cache = self._semantic_example_cache = {}
        if cache_lock is None:
            cache_lock = self._semantic_example_cache_lock = threading.RLock()

        with cache_lock:
            cached = cache.get(cache_key)
            if cached is not None:
                return cached

            labels: list[str] = []
            prompts: list[str] = []
            for label, examples in examples_by_label.items():
                labels.extend([label] * len(examples))
                prompts.extend(examples)

            vectors = self._normalized_matrix(
                self._memorize._mem._embedder.embed_queries(prompts, instruct=instruct).tolist()
            )
            cached = (labels, vectors)
            cache[cache_key] = cached
            return cached

    def _classify_agent_intent(self, user_input: str) -> str:
        """Ask the local model for a compact route label when keywords miss."""
        try:
            resp = self._client.chat.completions.create(
                model=self._router_model,
                messages=[{"role": "user", "content": (
                    f"Message: {user_input!r}\n\n"
                    "Output only the best tool label for the task. No explanation.\n"
                    "Labels: [research, reminder, planning, coding, writing, decision, architecture, ongoing_task]\n\n"
                    "Message: 'set a reminder for 9pm'\n"
                    "Label: reminder\n\n"
                    "Message: 'write an email to my landlord'\n"
                    "Label: writing\n\n"
                    "Message: 'debug why asyncio.run() hangs'\n"
                    "Label: coding\n\n"
                    "Message: 'make a plan to learn Japanese'\n"
                    "Label: planning\n\n"
                    "Message: 'search for latest llama.cpp release'\n"
                    "Label: research\n\n"
                    "Message: 'compare ollama vs llama.cpp'\n"
                    "Label: decision\n\n"
                    "Message: 'open soul.md and show the persona block'\n"
                    "Label: architecture\n\n"
                    "Message: 'continue working on the reflection script'\n"
                    "Label: ongoing_task\n\n"
                    "Message: 'what do you think about minimalism'\n"
                    "Label: chat\n\n"
                    "Message: 'give me a roadmap for X'\n"
                    "Label: planning\n\n"
                    "Message: 'help me map out what I need to do before the deadline'\n"
                    "Label: planning\n\n"
                    "Message: 'pick up where we left off on X'\n"
                    "Label: ongoing_task\n\n"
                    "Message: 'which is better for X, option A or option B'\n"
                    "Label: decision\n\n"
                    "Message: 'give me a roadmap for integrating MioTTS into AIVA'\n"
                    "Label: planning\n\n"
                    "Message: 'give me a roadmap for setting up ROS2 on the Jetson'\n"
                    "Label: planning\n\n"
                    "Message: 'pick up where we left off on the memory consolidation refactor'\n"
                    "Label: ongoing_task\n\n"
                    "Message: 'pick up where we left off on the sshfs uid mapping fix'\n"
                    "Label: ongoing_task\n\n"
                    "Message: 'is it weird that I find debugging more satisfying than writing features'\n"
                    "Label: chat\n\n"
                    "Message: 'which quantization level is better for bilingual TTS quality'\n"
                    "Label: decision\n\n"
                    "Label:"
                )}],
                stream=False, max_tokens=6, temperature=0.0, top_p=1.0, top_k=1, timeout=LLM_TIMEOUT,
            )
            label = (resp.choices[0].message.content or "chat").strip().lower()
            label = re.sub(r"[^a-z_].*$", "", label)
            if label in {
                "research", "planning", "writing", "coding",
                "architecture", "decision", "reminder", "ongoing_task",
            }:
                return label
            # LLM returned an unrecognised label — fall back to semantic best guess
            scores = self._semantic_all_scores(user_input, _ROUTE_TOOL_EXAMPLES, _ROUTE_INSTRUCT_BINARY)
            if scores:
                return max(scores, key=scores.__getitem__)
            return "coding"  # last resort: treat as coding task
        except Exception as e:
            log.warning("Intent routing failed: %s", e)
            return "chat"


    def agentic_chat(self, user_input: str, token_callback=None) -> str:
        """Delegate task-mode execution to core.agentic."""
        return run_agentic_chat(self, user_input, token_callback=token_callback)

    def chat(
        self,
        user_input: str,
        token_callback=None,
        _skip_search: bool = False,
        _history_label: str | None = None,
    ) -> str:
        """Standard single-shot conversational turn."""
        if self._speak and self._speak.is_playing():
            self._speak.stop()

        memories     = self._memorize.search(user_input, limit=int(os.getenv("MEMORY_RECALL_LIMIT", 3)))
        memory_block = self._memorize.format_for_context(memories)

        system = self._persona
        if memory_block:
            system = f"{system}\n\n{memory_block}"
        else:
            system += "\n\n<memory_context>\nNo relevant memories found.\n</memory_context>"

        if not _skip_search and self._is_data_intent(user_input):
            try:
                search_query = self._resolve_search_query(user_input)
                if token_callback: token_callback(f"__SEARCHING__:{search_query}")
                
                context = deep_search(search_query, fetch_top=1)
                if context and not context.startswith("[search failed"):
                    system = (
                        f"{system}\n\n"
                        f"<search_results query='{search_query}'>\n"
                        f"Answer using ONLY the information in these search results.\n\n"
                        f"{context}\n"
                        f"</search_results>"
                    )
            except Exception as e:
                log.error(f"Web search step failed: {e}")

        llm_prompt = user_input
        if self._reasoning:
            llm_prompt = f"{user_input}\n\nThink through this carefully. Show reasoning in <think> tags, then answer."

        history_entry = _history_label if _history_label is not None else user_input
        _HISTORY_HARD_CAP = CONTEXT_WINDOW_TURNS * 10

        with self._history_lock:
            self._history.append({"role": "user", "content": history_entry})
            if len(self._history) > _HISTORY_HARD_CAP:
                self._history = self._history[-_HISTORY_HARD_CAP:]
            trimmed = self._history[-(CONTEXT_WINDOW_TURNS * 2):]

        trimmed = self._sanitize_history(trimmed)
        if trimmed and trimmed[-1]["role"] == "user" and llm_prompt != history_entry:
            trimmed = trimmed[:-1] + [{"role": "user", "content": llm_prompt}]

        raw_response = self._stream_response(trimmed, system=system, token_callback=token_callback)

        with self._history_lock:
            self._history.append({"role": "assistant", "content": raw_response})
            if len(self._history) > _HISTORY_HARD_CAP:
                self._history = self._history[-_HISTORY_HARD_CAP:]

        self._record_experience(history_entry, raw_response)
        self._store_async(history_entry, raw_response)
        self._reasoning = False
        return raw_response

    def web_search(self, query: str, token_callback=None) -> str:
        """Explicit /web command path."""
        context = deep_search(query, fetch_top=1)
        if "no results" in context or "failed" in context:
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
        if timeout is None:
            self._mem_queue.join()
            return True
        deadline = time.monotonic() + max(0.0, timeout)
        with self._mem_queue.all_tasks_done:
            while self._mem_queue.unfinished_tasks:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._mem_queue.all_tasks_done.wait(remaining)
        return True

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

    # ── idle learner ──────────────────────────────────────────────────────────

    def _idle_learner_loop(self):
        """Background autonomous learning loop."""
        while True:
            time.sleep(300)  # check every 5 minutes
            if time.time() - self._last_chat_time < _IDLE_LEARN_SECONDS:
                continue  # user has been active recently, don't interrupt
                
            if self._speak and self._speak.is_playing():
                continue
                
            log.info("[learner] Aiko is idle. Starting autonomous learning...")
            try:
                # Pick a gap from history (simplified: just grab a previous noun-heavy user msg)
                with self._history_lock:
                    candidates = [m["content"] for m in self._history if m["role"] == "user" and len(m["content"].split()) > 3]
                
                if not candidates:
                    continue
                    
                topic = candidates[-1] # simplistic: look at last user query
                learned_tag = f"[self-learned:{topic}]"
                if any(learned_tag in (m.get("content") or "") for m in self._history):
                    continue
                if self._memorize.search(learned_tag, limit=1):
                    continue
                
                # Run silent agentic research
                result = self.agentic_chat(f"Research this topic briefly: {topic}")
                
                # Store as self-learned memory
                self._memorize.add([
                    {"role": "system", "content": learned_tag},
                    {"role": "assistant", "content": result[:800]}
                ])
                log.info(f"[learner] Successfully learned about: {topic}")
            except Exception as e:
                log.error(f"[learner] Autonomous learning failed: {e}")

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

        if self._speak:
            self._speak.start_speech_stream()

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
                
                if token_callback and token:
                    token_callback(token)
                
                full_response.append(token)
                
                if self._speak and token:
                    sentence_buffer += token
                    sentences, sentence_buffer = split_stream_sentences(sentence_buffer)
                    for sentence in sentences:
                        self._speak.feed_speech_stream(sentence)

            text = "".join(full_response).strip()
            if text:
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

    def _resolve_search_query(self, user_input: str) -> str:
        """Condense user message into a clean search query.
        Strips conversational framing first; falls back to LLM only if still noisy."""
        # fast pass — strip conversational wrapper
        noise = re.compile(
            r"^(go\s+)?(can\s+you\s+)?(please\s+)?"
            r"(pull\s+up|find\s+out|check|look\s+up|search\s+for|tell\s+me)[\s,]*",
            re.IGNORECASE,
        )
        resolved = noise.sub("", user_input).strip()

        # if still looks like a full sentence (long + has verb framing), ask LLM to condense
        if len(resolved.split()) > 8 or resolved.lower().startswith(("what", "who", "when", "where", "is ", "has ", "did ")):
            return self._llm_resolve_search_query(resolved)

        return resolved or user_input

    def _llm_resolve_search_query(self, user_input: str) -> str:
        """LLM fallback: condense a verbose query into 3-5 search keywords."""
        try:
            resp = self._client.chat.completions.create(
                model=self._router_model,
                messages=[{"role": "user", "content": (
                    f"Message: {user_input!r}\n\n"
                    "Output only a 3-5 word search query. No explanation.\n\n"
                    "Message: 'what's the latest llama.cpp version'\n"
                    "Query: llama.cpp latest stable release\n\n"
                    "Message: 'has NVIDIA released any new Jetson hardware this year'\n"
                    "Query: NVIDIA Jetson new hardware 2025\n\n"
                    "Message: 'what's the Canucks score from last night'\n"
                    "Query: Canucks score last night\n\n"
                    "Message: 'did llama.cpp merge the Vulkan backend yet'\n"
                    "Query: llama.cpp Vulkan backend merged\n\n"
                    "Query:"
                )}],
                stream=False, max_tokens=20, temperature=0.0, top_p=1.0, top_k=1, timeout=LLM_TIMEOUT,
            )
            resolved = (resp.choices[0].message.content or "").strip().split('\n')[0]
            resolved = resolved.strip('*_`()').strip()[:100]
            return resolved or user_input
        except Exception as e:
            log.warning("LLM search query resolution failed: %s", e)
            return user_input

    def _sanitize_history(self, messages: list[dict]) -> list[dict]:
        if not messages: return []
        sanitized = [messages[0]]
        for msg in messages[1:]:
            if msg["role"] == sanitized[-1]["role"]: sanitized[-1] = msg
            else: sanitized.append(msg)
        while sanitized and sanitized[0]["role"] != "user": sanitized.pop(0)
        return sanitized

    def _record_experience(self, user_input: str, response_text: str) -> None:
        try:
            append_chat_turn(user_input, response_text, user_id=os.getenv("USER_ID", "OppaAI"))
        except Exception as e:
            log.warning("Daily experience logging failed: %s", e)

    def _store_async(self, user_input: str, response_text: str) -> None:
        self._mem_queue.put((user_input, response_text))

    def _mem_write_loop(self) -> None:
        while True:
            user_input, response_text = self._mem_queue.get()
            try:
                self._wait_for_memory_write_window()
                self._memorize.add([
                    {"role": "user",      "content": user_input[:500]},
                    {"role": "assistant", "content": response_text[:800]},
                ])
            except Exception as e:
                log.error(f"Async memory write failed: {e}")
            finally:
                self._mem_queue.task_done()

    def _wait_for_memory_write_window(self) -> None:
        """Wait until chat has been idle before using the shared LLM for extraction."""
        deadline = time.monotonic() + max(0.0, _MEMORY_WRITE_MAX_WAIT)
        while True:
            idle_for = time.time() - self._last_chat_time
            if not self._active_turn.is_set() and idle_for >= _MEMORY_WRITE_IDLE_GRACE:
                return
            if (
                _MEMORY_WRITE_MAX_WAIT > 0
                and time.monotonic() >= deadline
                and not self._active_turn.is_set()
            ):
                return
            sleep_for = min(0.5, max(0.05, _MEMORY_WRITE_IDLE_GRACE - idle_for))
            time.sleep(sleep_for)


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
