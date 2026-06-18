"""
core/think.py

Aiko's cognitive loop.
  - Routes between single-shot chat and agentic task loop.
  - Agentic loop uses tools (web_search, fetch_page) to complete multi-step tasks.
  - Idle learner autonomously researches topics from history in the background.
  - Streams llama.cpp response to console + TTS simultaneously.
  - Stores the turn into long-term memory after each response (background thread).
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

from core.memorize import AikoMemorize
from core.speak    import AikoSpeak
from core.tools    import web_search, fetch_and_extract, deep_search
from core.log      import get_logger

log = get_logger(__name__)

# ── boot labels ───────────────────────────────────────────────────────────────

BOOT_LABELS = {
    'think_start':  'Loading llama.cpp client + persona...',
    'think_warmup': 'Warming up language model...',
}

# ── config ────────────────────────────────────────────────────────────────────

LLAMACPP_BASE_URL = os.getenv("LLAMACPP_BASE_URL", "http://localhost:8080/v1")
LLAMACPP_MODEL    = os.getenv("LLAMACPP_MODEL",    "ministral-3b-instruct")
CONTEXT_WINDOW_TURNS = int(os.getenv("CONTEXT_WINDOW_TURNS", 8))

_BASE_PREDICT    = 280
_REASONING_SCALE = 3

_PERSONA_PATH = Path(__file__).resolve().parent.parent / "persona" / "soul.md"
_SKILLS_PATH  = Path(__file__).resolve().parent.parent / "persona" / "skills.md"

MAX_AGENT_ITER = 8

def _load_persona() -> str:
    """Read persona and skills definitions."""
    if not _PERSONA_PATH.exists():
        raise FileNotFoundError(f"soul.md not found at {_PERSONA_PATH}")
    persona = _PERSONA_PATH.read_text(encoding="utf-8").strip()
    
    skills_block = ""
    if _SKILLS_PATH.exists():
        skills_block = "\n\n" + _SKILLS_PATH.read_text(encoding="utf-8").strip()
        
    user_id = os.getenv("USER_ID", "OppaAI")
    today   = datetime.now().strftime("%B %d, %Y")
    return persona.replace("USER_ID_HERE", user_id).replace("TODAY_HERE", today) + skills_block

# ── intent signals (same as before) ───────────────────────────────────────────

_SOCIAL_SIGNALS = frozenset([
    "wanna", "want to", "would you", "shall we",
    "let's", "lets", "together", "with me", "join me",
    "how are you", "what do you think", "do you like",
])

_FACTUAL_SIGNALS = frozenset([
    "who", "what", "when", "where", "how many", "how much",
    "score", "result", "latest", "news", "current", "today",
    "price", "weather", "won", "win", "lost", "beat",
    "game", "final", "finals", "points", "scored",
    "standing", "ranking", "bitcoin", "crypto", "stock",
    "temperature", "forecast", "match", "series",
])

_FACTUAL_RE = re.compile(r'\b(?:' + '|'.join(re.escape(s) for s in _FACTUAL_SIGNALS) + r')\b', re.IGNORECASE)
_SOCIAL_RE = re.compile(r'\b(?:' + '|'.join(re.escape(s) for s in _SOCIAL_SIGNALS) + r')\b', re.IGNORECASE)

SKILL_TRIGGERS = ["research", "deep dive", "compare", "vs", "which is better", "difference between"]

# ── think ─────────────────────────────────────────────────────────────────────

class AikoThink:
    def __init__(self, memorize: AikoMemorize, speak: AikoSpeak | None = None) -> None:
        self._client    = OpenAI(base_url=LLAMACPP_BASE_URL, api_key="not-needed")
        self._memorize  = memorize
        self._speak     = speak
        self._persona   = _load_persona()
        self._history:  list[dict] = []
        self._history_lock = threading.Lock()
        self._pending_search_query: str | None = None
        self._reasoning = False
        self._mem_queue  = queue.Queue()
        self._mem_worker = threading.Thread(target=self._mem_write_loop, daemon=True)
        self._mem_worker.start()

        self._last_chat_time = time.time()
        self._idle_learner_thread = threading.Thread(target=self._idle_learner_loop, daemon=True)
        self._idle_learner_thread.start()

        self._warmup_thread = threading.Thread(target=self._warmup_llm, daemon=True)
        self._warmup_thread.start()

    def _warmup_llm(self) -> None:
        try:
            self._client.chat.completions.create(
                model=LLAMACPP_MODEL,
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
        """Main entry point. Routes to agentic loop if skill triggered, else normal chat."""
        self._last_chat_time = time.time()
        
        # Simple skill router
        is_skill = any(trigger in user_input.lower() for trigger in SKILL_TRIGGERS)
        
        if is_skill:
            log.info(f"[route] Skill triggered. Entering agentic loop for: {user_input!r}")
            return self.agentic_chat(user_input, token_callback=token_callback)
        return self.chat(user_input, token_callback=token_callback)

    def agentic_chat(self, user_input: str, token_callback=None) -> str:
        """ReAct-style agentic loop with tool calling."""
        tools = [
            {"type": "function", "function": {
                "name": "web_search", "description": "Search the web for information.",
                "parameters": {"type": "object", "properties": {
                    "query": {"type": "string", "description": "The search query."}},
                    "required": ["query"]}}},
            {"type": "function", "function": {
                "name": "fetch_page", "description": "Fetch and extract full text from a specific URL.",
                "parameters": {"type": "object", "properties": {
                    "url": {"type": "string", "description": "The URL to fetch."}},
                    "required": ["url"]}}},
            {"type": "function", "function": {
                "name": "final_answer", "description": "Submit the final comprehensive response to the user.",
                "parameters": {"type": "object", "properties": {
                    "answer": {"type": "string", "description": "The final answer text."}},
                    "required": ["answer"]}}},
        ]

        messages = [
            {"role": "system", "content": self._persona},
            {"role": "user", "content": user_input},
        ]

        final_text = ""

        for step in range(MAX_AGENT_ITER):
            if token_callback:
                token_callback(f"__THINKING__\n")

            try:
                resp = self._client.chat.completions.create(
                    model=LLAMACPP_MODEL, messages=messages, tools=tools,
                    tool_choice="auto", stream=False, max_tokens=1024,
                    temperature=0.3,
                )
                msg = resp.choices[0].message
                messages.append(msg.model_dump(exclude_none=True))
            except Exception as e:
                log.error(f"Agent LLM call failed: {e}")
                break

            if not msg.tool_calls:
                final_text = msg.content or ""
                break

            for call in msg.tool_calls:
                name = call.function.name
                try:
                    args = json.loads(call.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                log.info(f"[agent] step {step} → {name}({args})")
                if token_callback:
                    token_callback(f"__TOOL__:{name}({args})\n")

                if name == "web_search":
                    result = deep_search(args.get("query", ""))
                elif name == "fetch_page":
                    result = fetch_and_extract(args.get("url", ""))
                elif name == "final_answer":
                    final_text = args.get("answer", "")
                    messages.append({
                        "role": "tool", "tool_call_id": call.id,
                        "name": name, "content": "Answer submitted."
                    })
                    break
                else:
                    result = f"[unknown tool: {name}]"

                messages.append({
                    "role": "tool", "tool_call_id": call.id,
                    "name": name, "content": result[:3000],
                })

            if final_text:
                break

        if not final_text:
            final_text = "I got a bit lost trying to complete that task. Here is what I have so far:\n" + (msg.content or "")

        # Emit final text to TTS/Console
        self._emit(final_text, token_callback=token_callback)

        with self._history_lock:
            self._history.append({"role": "user", "content": user_input})
            self._history.append({"role": "assistant", "content": final_text})
        
        self._store_async(user_input, final_text)
        return final_text

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

        self.wait_for_memory()

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
        self._emit(raw_response, token_callback=token_callback)

        with self._history_lock:
            self._history.append({"role": "assistant", "content": raw_response})
            if len(self._history) > _HISTORY_HARD_CAP:
                self._history = self._history[-_HISTORY_HARD_CAP:]

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
    def wait_for_memory(self) -> None: self._mem_queue.join()

    # ── idle learner ──────────────────────────────────────────────────────────

    def _idle_learner_loop(self):
        """Background autonomous learning loop."""
        while True:
            time.sleep(300)  # check every 5 minutes
            if time.time() - self._last_chat_time < 300:
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
                
                # Run silent agentic research
                result = self.agentic_chat(f"Research this topic briefly: {topic}")
                
                # Store as self-learned memory
                self._memorize.add([
                    {"role": "system", "content": f"[self-learned:{topic}]"},
                    {"role": "assistant", "content": result[:800]}
                ])
                log.info(f"[learner] Successfully learned about: {topic}")
            except Exception as e:
                log.error(f"[learner] Autonomous learning failed: {e}")

    # ── internal ──────────────────────────────────────────────────────────────

    def _emit(self, text: str, token_callback=None) -> None:
        if not text: return
        if token_callback and self._speak:
            self._speak.speak_synced(text, on_word=token_callback)
            return
        if token_callback:
            words = text.split(" ")
            for i, word in enumerate(words):
                chunk = word if i == 0 else " " + word
                token_callback(chunk)
                time.sleep(float(os.getenv("EMIT_DELAY", 0.012)))
        else:
            print(f"\nAiko-chan: {text}", flush=True)
            if self._speak:
                self._speak.feed(text)
                self._speak.play_async()

    def _stream_response(self, messages: list[dict], system: str = "", silent: bool = True, token_callback=None) -> str:
        full_response = []
        max_tokens = _BASE_PREDICT * _REASONING_SCALE if self._reasoning else _BASE_PREDICT
        all_messages = [{"role": "system", "content": system}] + messages if system else messages

        try:
            stream = self._client.chat.completions.create(
                model=LLAMACPP_MODEL, messages=all_messages, stream=True,
                max_tokens=max_tokens,
                temperature=float(os.getenv("TEMPERATURE", 0.72)),
                top_p=float(os.getenv("TOP_P", 0.90)),
                stop=["<|im_end|>", "</s>", "[INST]"],
                timeout=float(os.getenv("LLAMACPP_TIMEOUT", 120)),
                extra_body={
                    "repeat_penalty": float(os.getenv("REPEAT_PENALTY", 1.15)),
                    "repeat_last_n":  int(os.getenv("REPEAT_LAST_N", 64)),
                    "top_k":          int(os.getenv("TOP_K", 40)),
                },
            )
            for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                token = (delta.content or "") if delta else ""
                full_response.append(token)
                if not silent and token_callback: token_callback(token)
        except Exception as e:
            msg = f"Stream failed: {e}"
            log.error(msg)
            with self._history_lock:
                if self._history and self._history[-1]["role"] == "user": self._history.pop()
            return ""

        return "".join(full_response)

    def _is_data_intent(self, user_input: str) -> bool:
        if _FACTUAL_RE.search(user_input): return True
        if _SOCIAL_RE.search(user_input): return False
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
                model=LLAMACPP_MODEL,
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

    def _store_async(self, user_input: str, response_text: str) -> None:
        self._mem_queue.put((user_input, response_text))

    def _mem_write_loop(self) -> None:
        while True:
            user_input, response_text = self._mem_queue.get()
            try:
                self._memorize.add([
                    {"role": "user",      "content": user_input[:500]},
                    {"role": "assistant", "content": response_text[:800]},
                ])
            except Exception as e:
                log.error(f"Async memory write failed: {e}")
            finally:
                self._mem_queue.task_done()
