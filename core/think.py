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
LLM_TIMEOUT  = float(os.getenv("LLM_TIMEOUT", 120))
CONTEXT_WINDOW_TURNS = int(os.getenv("CONTEXT_WINDOW_TURNS", 8))

_BASE_PREDICT    = int(os.getenv("LLM_MAX_TOKENS", os.getenv("BASE_PREDICT", 280)))
_AGENT_MAX_TOKENS = int(os.getenv("AGENT_MAX_TOKENS", _BASE_PREDICT * 4))
_REASONING_SCALE = 3
_IDLE_LEARN_SECONDS = int(os.getenv("IDLE_LEARN_SECONDS", 1800))
_MEMORY_WRITE_IDLE_GRACE = float(os.getenv("MEMORY_WRITE_IDLE_GRACE", 3.0))
_MEMORY_WRITE_MAX_WAIT = float(os.getenv("MEMORY_WRITE_MAX_WAIT", 45.0))
# Route task-vs-chat turns semantically by default using the same embedding
# model as memory/RAG. Set AIKO_ROUTE_MODE=llm to let the local LLM classify
# instead, or AIKO_ROUTE_MODE=chat to disable autonomous routing.
_ROUTE_ENABLED = os.getenv("AIKO_ROUTE_ENABLED", os.getenv("AIKO_ROUTE_LLM", "1")).lower() in {"1", "true", "yes", "on"}
_ROUTE_MODE = os.getenv("AIKO_ROUTE_MODE", "semantic").strip().lower()
_SEMANTIC_ROUTE_THRESHOLD = float(os.getenv("AIKO_ROUTE_SEMANTIC_THRESHOLD", "0.36"))
_SEMANTIC_SEARCH_THRESHOLD = float(os.getenv("AIKO_SEARCH_SEMANTIC_THRESHOLD", "0.36"))

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

_SEMANTIC_ROUTE_EXAMPLES: dict[str, tuple[str, ...]] = {
    "chat": (
        "talk with me normally",
        "answer this casual question conversationally",
        "what is the weather today",
        "what happened in the news today",
    ),
    "reminder": (
        "remind me to do this tomorrow",
        "wake me up at six in the morning",
        "set an alarm for this time",
    ),
    "research": (
        "research this topic and summarize what you find",
        "look into this question and give me sourced findings",
        "compare these options and report the tradeoffs",
    ),
    "planning": (
        "make a step by step plan for this project",
        "turn this goal into an organized checklist",
        "break this task down into next actions",
    ),
    "writing": (
        "draft this message for me",
        "write a note or document from these details",
        "prepare a clean response I can send",
    ),
    "coding": (
        "debug this code problem",
        "explain how this code works and what to change",
        "help implement this programming task",
    ),
    "architecture": (
        "inspect Aiko's own codebase",
        "read the repository and explain how this part works",
        "debug or improve Aiko's architecture",
    ),
    "decision": (
        "help me decide between these choices",
        "evaluate the options and recommend one",
        "score the tradeoffs for this decision",
    ),
    "ongoing_task": (
        "track this task and update progress",
        "continue the task we were working on",
        "manage this ongoing project state",
    ),
}

_SEMANTIC_SEARCH_EXAMPLES: dict[str, tuple[str, ...]] = {
    "chat": (
        "talk with me normally",
        "what do you think about this idea",
        "help me reason through this from what you already know",
        "explain this concept without looking anything up",
        "where should I put this config in my project",
    ),
    "data": (
        "what is the weather today",
        "search for the latest news about this topic",
        "what is the current price of bitcoin",
        "who won the game last night",
        "look up the newest release notes for this library",
        "find current facts and cite the source",
    ),
}

# ── think ─────────────────────────────────────────────────────────────────────

class AikoThink:
    def __init__(self, memorize: AikoMemorize, speak: AikoSpeak | None = None) -> None:
        self._client    = OpenAI(base_url=LLM_BASE_URL, api_key="not-needed")
        self._llm_model = LLM_MODEL
        self._memorize  = memorize
        self._speak     = speak
        self._persona   = _load_persona()
        self._history:  list[dict] = []
        self._history_lock = threading.Lock()
        self._pending_search_query: str | None = None
        self._route_chat_classified: str | None = None
        self._active_turn = threading.Event()
        self._reasoning = False
        self._mem_queue  = queue.Queue()
        self._mem_worker = threading.Thread(target=self._mem_write_loop, daemon=True)
        self._mem_worker.start()

        self._last_chat_time = time.time()
        self._idle_learner_thread = threading.Thread(target=self._idle_learner_loop, daemon=True)
        self._idle_learner_thread.start()

        self._reminders = ScheduleRunner(on_due=self._on_scheduled_job_due)
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
        """Main entry point. Uses keyword + semantic intent routing."""
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
        """Classify whether a turn needs autonomous task mode or normal chat."""
        if not _ROUTE_ENABLED or _ROUTE_MODE in {"0", "off", "false", "chat", "disabled"}:
            return "chat"
        if _ROUTE_MODE == "llm":
            return self._classify_agent_intent(user_input)
        return self._semantic_agent_intent(user_input)

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        denom_a = sum(x * x for x in a) ** 0.5
        denom_b = sum(x * x for x in b) ** 0.5
        if not denom_a or not denom_b:
            return 0.0
        return sum(x * y for x, y in zip(a, b)) / (denom_a * denom_b)

    def _semantic_agent_intent(self, user_input: str) -> str:
        """Classify agentic intent by embedding similarity to route examples."""
        try:
            best_label, best_score = self._semantic_best_label(user_input, _SEMANTIC_ROUTE_EXAMPLES)
            log.debug("[route] Semantic route best=%s score=%.3f for: %r", best_label, best_score, user_input)
            if best_score >= _SEMANTIC_ROUTE_THRESHOLD:
                return best_label
            return "chat"
        except Exception as e:
            log.warning("Semantic intent routing failed: %s", e)
            if _ROUTE_MODE == "semantic_only":
                return "chat"
            return self._classify_agent_intent(user_input)

    def _semantic_best_label(self, user_input: str, examples_by_label: dict[str, tuple[str, ...]]) -> tuple[str, float]:
        """Return the closest semantic label and cosine score for a prompt."""
        prompts = [user_input]
        labels: list[str] = []
        for label, examples in examples_by_label.items():
            labels.extend([label] * len(examples))
            prompts.extend(examples)

        vectors = [self._memorize.embed_text(text) for text in prompts]
        query_vector = vectors[0]
        best_label = "chat"
        best_score = 0.0
        for label, vector in zip(labels, vectors[1:]):
            score = self._cosine_similarity(query_vector, vector)
            if score > best_score:
                best_label = label
                best_score = score
        return best_label, best_score

    def _classify_agent_intent(self, user_input: str) -> str:
        """Ask the local model for a compact route label when keywords miss."""
        try:
            resp = self._client.chat.completions.create(
                model=self._llm_model,
                messages=[{"role": "user", "content": (
                    "Route message. Labels: chat, research, planning, writing, coding, "
                    "architecture, decision, reminder, ongoing_task. "
                    "Use architecture for requests to inspect, read, debug, or improve "
                    "Aiko's own codebase/repository. Reply one label only.\n"
                    f"Message: {user_input!r}"
                )}],
                stream=False, max_tokens=8, temperature=0.0,
            )
            label = (resp.choices[0].message.content or "chat").strip().lower()
            label = re.sub(r"[^a-z_].*$", "", label)
            if label in {
                "research", "planning", "writing", "coding",
                "architecture", "decision", "reminder", "ongoing_task",
            }:
                return label
            self._route_chat_classified = user_input
            return "chat"
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
                search_query = self._build_search_query(user_input)
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

    def _on_scheduled_job_due(self, job: DueJob) -> None:
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
        if self._route_chat_classified == user_input:
            self._route_chat_classified = None
            return False
        if not _ROUTE_ENABLED or _ROUTE_MODE in {"0", "off", "false", "chat", "disabled"}:
            return False
        if _ROUTE_MODE != "llm":
            is_data = self._semantic_data_intent(user_input)
            self._pending_search_query = user_input if is_data else None
            return is_data
        is_data, resolved_query = self._classify_and_resolve(user_input)
        self._pending_search_query = resolved_query if is_data else None
        return is_data

    def _semantic_data_intent(self, user_input: str) -> bool:
        """Classify normal-chat web-search need by embedding similarity."""
        try:
            best_label, best_score = self._semantic_best_label(user_input, _SEMANTIC_SEARCH_EXAMPLES)
            log.debug("[search] Semantic route best=%s score=%.3f for: %r", best_label, best_score, user_input)
            return best_label == "data" and best_score >= _SEMANTIC_SEARCH_THRESHOLD
        except Exception as e:
            log.warning("Semantic search intent routing failed: %s", e)
            if _ROUTE_MODE == "semantic_only":
                return False
            is_data, resolved_query = self._classify_and_resolve(user_input)
            self._pending_search_query = resolved_query if is_data else None
            return is_data

    def _classify_and_resolve(self, user_input: str) -> tuple[bool, str]:
        with self._history_lock:
            last_user = next((m["content"] for m in reversed(self._history) if m["role"] == "user"), "")
        has_context = bool(last_user and last_user != user_input)
        context_block = f'Previous question: "{last_user}"\n' if has_context else ""

        try:
            resp = self._client.chat.completions.create(
                model=self._llm_model,
                messages=[{"role": "user", "content": (
                    f'{context_block}Message: "{user_input}"\n\n'
                    f'Is this asking for factual external data, or conversational?\n'
                    f'If data, resolve pronouns into a search query.\n'
                    f'Reply EXACTLY:\ndata|<search query>\nor:\nsocial|none'
                )}],
                stream=False, max_tokens=32, temperature=0.0,
            )
            answer = resp.choices[0].message.content.strip()
            label, _, rest = answer.partition("|")
            is_data = "data" in label.strip().lower()
            resolved = rest.strip() if (is_data and rest and rest.lower() != "none") else user_input
            return is_data, resolved
        except Exception as e:
            log.warning(f"Intent classification failed: {e}")
            return True, user_input

    def _build_search_query(self, user_input: str) -> str:
        pending = self._pending_search_query
        if pending is not None:
            self._pending_search_query = None
            return pending
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
