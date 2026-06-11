"""
core/think.py

Aiko's cognitive loop.
  - Drains memory write queue before each recall (prevents write-lag misses)
  - Retrieves relevant memories before each turn
  - Proactively searches the web for data-intent queries before LLM speaks
  - Streams Ollama response to console + TTS simultaneously
  - Stores the turn into long-term memory after each response (background thread)
  - Supports single-shot reasoning mode via set_reasoning(True) / /think command
"""

from email import message
import logging
import os
import warnings

warnings.filterwarnings("ignore")
logging.getLogger("phonemizer").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

from datetime import datetime
from ollama import Client
from pathlib import Path
import queue
import re
import threading
import time

from core.memorize import AikoMemorize
from core.speak    import AikoSpeak
from core.log      import get_logger

log = get_logger(__name__)

# ── boot labels ───────────────────────────────────────────────────────────────

BOOT_LABELS = {
    'think_start':  'Loading Ollama client + persona...',
    'think_warmup': 'Warming up language model...',
}

# ── config ────────────────────────────────────────────────────────────────────

OLLAMA_BASE_URL      = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL         = os.getenv("OLLAMA_MODEL",    "ministral-3:3b-instruct-2512-q4_K_M")
CONTEXT_WINDOW_TURNS = int(os.getenv("CONTEXT_WINDOW_TURNS", 8))

_BASE_PREDICT    = 280   # normal token budget — soul.md says 2–3 sentences default
_REASONING_SCALE = 3     # multiplier applied to num_predict in reasoning mode

_PERSONA_PATH = Path(__file__).resolve().parent.parent / "persona" / "soul.md"


def _load_persona() -> str:
    """Read and return the persona definition from soul.md."""
    if not _PERSONA_PATH.exists():
        raise FileNotFoundError(f"soul.md not found at {_PERSONA_PATH}")
    persona = _PERSONA_PATH.read_text(encoding="utf-8").strip()
    user_id = os.getenv("USER_ID", "OppaAI")
    today   = datetime.now().strftime("%B %d, %Y")
    return persona.replace("USER_ID_HERE", user_id).replace("TODAY_HERE", today)


# ── intent signals ────────────────────────────────────────────────────────────

# Inputs matching ANY of these are immediately classified as social — no search.
_SOCIAL_SIGNALS = frozenset([
    "wanna", "want to", "would you", "do you", "shall we",
    "let's", "lets", "together", "with me", "join me",
    "how are you", "what do you think", "do you like",
    "are you", "can you", "will you",
])

# Inputs matching ANY of these are immediately classified as data — search fires.
_FACTUAL_SIGNALS = frozenset([
    "who", "what", "when", "where", "how many", "how much",
    "score", "result", "latest", "news", "current", "today",
    "price", "weather", "won", "win", "lost", "beat",
    "game", "final", "finals", "points", "scored",
    "standing", "ranking", "bitcoin", "crypto", "stock",
    "temperature", "forecast", "match", "series",
])


# ── think ─────────────────────────────────────────────────────────────────────

class AikoThink:
    """
    Aiko's conversational core.
    speak is injected pre-warmed from wakeup.py.
    LLM warmup starts immediately on init in a background thread.
    wakeup.py calls join_warmup() to block until the model is hot.
    """

    def __init__(self, memorize: AikoMemorize, speak: AikoSpeak | None = None) -> None:
        self._client    = Client(host=OLLAMA_BASE_URL)
        self._memorize  = memorize
        self._speak     = speak
        self._persona   = _load_persona()
        self._history:  list[dict] = []
        self._reasoning = False
        self._mem_queue  = queue.Queue()
        self._mem_worker = threading.Thread(target=self._mem_write_loop, daemon=True)
        self._mem_worker.start()

        self._warmup_thread = threading.Thread(target=self._warmup_llm, daemon=True)
        self._warmup_thread.start()

    def _warmup_llm(self) -> None:
        try:
            self._client.chat(
                model=OLLAMA_MODEL,
                messages=[{"role": "user", "content": "hi"}],
                stream=False,
                keep_alive=-1,
                options={
                    "num_predict": 1,
                    "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", 4096)),
                },
            )
        except Exception as e:
            log.warning("LLM warmup failed — Ollama may not be running: %s", e)

    def join_warmup(self) -> None:
        """Block until LLM warmup completes. Called by wakeup.py before boot finishes."""
        if self._warmup_thread and self._warmup_thread.is_alive():
            self._warmup_thread.join()

    # ── public api ────────────────────────────────────────────────────────────

    def chat(self, user_input: str, token_callback=None) -> str:
        """
        Process one conversational turn and return the full assistant response.

        Flow:
          1. Drain pending memory writes so recall sees the latest facts.
          2. Search long-term memory; inject explicit no-memory signal if empty.
          3. Build system prompt (persona + memory block).
          4. If data intent detected, search web and inject results into system
             prompt before the LLM speaks — LLM never guesses live data.
          5. Stream LLM response silently (full text collected).
          6. Emit display text to token_callback + TTS (word-by-word for feel).
          7. Append to history and enqueue async memory write.

        Reasoning mode (set via set_reasoning(True) / /think command) wraps the
        user prompt with a step-by-step instruction and triples num_predict.
        Auto-resets to False after each turn — single-shot by design.
        """
        self._token_callback = token_callback

        # interrupt any ongoing speech before processing new input
        if self._speak and self._speak.is_playing():
            self._speak.stop()

        # 1. flush pending memory writes so this turn's recall is up to date
        self.wait_for_memory()

        # 2. retrieve relevant long-term memories
        memories     = self._memorize.search(user_input, limit=int(os.getenv("MEMORY_RECALL_LIMIT", 3)))
        memory_block = self._memorize.format_for_context(memories)

        # 3. build system prompt — inject explicit no-memory signal on miss
        system = self._persona
        if memory_block:
            system = f"{system}\n\n{memory_block}"
        else:
            system = (
                f"{system}\n\n"
                "<memory_context>\n"
                "No relevant memories found for this query. "
                "If Oppa asks about something personal or specific to him, "
                "tell him you don't have it stored rather than guessing.\n"
                "</memory_context>"
            )

        # 4. proactive web search — fires before LLM speaks so no guessing
        if self._is_data_intent(user_input):
            from core.tools import web_search_context
            search_query = self._build_search_query(user_input)

            log.debug(f"[intent] DATA — searching: {search_query!r}")

            # notify UI before the blocking search call
            if token_callback:
                token_callback(f"__SEARCHING__:{search_query}")

            context = web_search_context(search_query)
            log.debug(f"[intent] context={'found' if context else 'NONE'}")

            if context:
                system = (
                    f"{system}\n\n"
                    f"<search_results query='{search_query}'>\n"
                    f"Answer using ONLY the information in these search results. "
                    f"Do not add anything from your training data. "
                    f"If the results don't contain the answer, say so plainly.\n\n"
                    f"{context}\n"
                    f"</search_results>"
                )

        # 5. wrap user turn with reasoning instruction if active
        if self._reasoning:
            prompt = (
                f"{user_input}\n\n"
                "Think through this carefully before answering. "
                "Show your reasoning inside <think> tags, then give your final answer."
            )
        else:
            prompt = user_input

        # 6. append user turn
        self._history.append({"role": "user", "content": prompt})

        # 7. trim history to context window
        trimmed = self._history[-(CONTEXT_WINDOW_TURNS * 2):]
        trimmed = self._sanitize_history(trimmed)

        # 8. stream LLM response (silent — display handled in _emit)
        raw_response = self._stream_response(trimmed, system=system)

        # 9. emit to callback + TTS
        self._emit(raw_response)

        # 10. append assistant turn to history
        self._history.append({"role": "assistant", "content": raw_response})

        # 11. persist to memory (background) — store original input, not wrapped prompt
        self._store_async(user_input, raw_response)

        # 12. auto-reset reasoning mode
        self._reasoning = False

        return raw_response

    def web_search(self, query: str, token_callback=None) -> str:
        """
        Run a web search and feed results into a normal chat turn.
        Called by the /web command — no search logic leaks out.
        """
        from core.tools import web_search_context
        context = web_search_context(query)
        if context is None:
            msg = f"[no results for: {query}]"
            if token_callback:
                token_callback(msg)
            return msg
        return self.chat(context, token_callback=token_callback)

    def reset_context(self) -> None:
        """Clear the in-memory conversation history for a fresh session."""
        self._history.clear()

    def last_turn(self) -> tuple[str, str] | None:
        """
        Return the latest complete user/assistant exchange, if one exists.
        Walks history in reverse to find the most recent assistant reply
        and its paired user message.
        Returns (user_text, assistant_text) or None.
        """
        assistant_text: str | None = None

        for message in reversed(self._history):
            role    = message.get("role")
            content = (message.get("content") or "").strip()
            if not content:
                continue
            if assistant_text is None:
                if role == "assistant":
                    assistant_text = content
                continue
            if role == "user":
                return content, assistant_text

        return None

    def set_reasoning(self, enabled: bool) -> None:
        """
        Enable or disable reasoning mode for the next turn only.
        Auto-resets to False after each chat() call.
        """
        self._reasoning = enabled

    def set_speak(self, speak) -> None:
        """Hot-swap the TTS backend. Pass None to silence, speak instance to restore."""
        self._speak = speak

    def wait_for_memory(self) -> None:
        """Block until all enqueued memory writes have been persisted."""
        self._mem_queue.join()

    # ── internal ──────────────────────────────────────────────────────────────

    def _emit(self, text: str) -> None:
        """
        Send display text to token_callback (TUI) or stdout, and feed TTS.
        Streams word-by-word to token_callback for a natural typing feel,
        even though the LLM has already finished (silent mode).
        TTS receives the full text at once for uninterrupted audio.
        """
        if not text:
            return

        if self._token_callback:
            words = text.split(" ")
            for i, word in enumerate(words):
                chunk = word if i == 0 else " " + word
                self._token_callback(chunk)
                time.sleep(0.012)  # ~80 wpm feel
        else:
            print(f"\nAiko-chan: {text}", flush=True)

        if self._speak:
            self._speak.feed(text)
            self._speak.play_async()

    def _stream_response(self, messages: list[dict], system: str = "", silent: bool = True) -> str:
        """
        Stream LLM response to console and TTS simultaneously.
        Console printing is the single source of truth — speak.py is silent.
        TTS skipped if response is a search trigger.

        Tokens are buffered at the start of each response to detect a
        [SEARCH: ...] trigger before any output is committed to the console
        or TTS queue. num_predict is scaled by _REASONING_SCALE when
        reasoning mode is active to budget for the <think> scratchpad.

        Args:
            messages: Full message list (system prompt + trimmed history).
            system:   Optional system prompt to inject.

        Returns:
            The complete response text assembled from all streamed tokens.
        """
        full_response    = []
        tts_started      = False
        buffer           = ""
        is_searching     = False
        buffering_active = True

        # triple token budget in reasoning mode to fit the <think> scratchpad
        num_predict = _BASE_PREDICT * _REASONING_SCALE if self._reasoning else _BASE_PREDICT

        all_messages = (
            [{"role": "system", "content": system}] + messages
            if system else messages
        )

        all_messages = (
            [{"role": "system", "content": system}] + messages
            if system else messages
        )

        try:
            stream = self._client.chat(
                model=OLLAMA_MODEL,
                messages=all_messages,
                stream=True,
                keep_alive=-1,
                options={
                    "num_ctx":        int(os.getenv("OLLAMA_NUM_CTX", 4096)),
                    "temperature":    float(os.getenv("OLLAMA_TEMPERATURE", 0.72)),
                    "repeat_penalty": float(os.getenv("OLLAMA_REPEAT_PENALTY", 1.15)),
                    "repeat_last_n":  int(os.getenv("OLLAMA_REPEAT_LAST_N", 64)),
                    "num_predict":    num_predict,
                    "top_p":          float(os.getenv("OLLAMA_TOP_P", 0.90)),
                    "top_k":          int(os.getenv("OLLAMA_TOP_K", 40)),
                    "tfs_z":          1.0,
                    "stop":           ["<|im_end|>", "</s>", "[INST]"],
                }
            )

            for chunk in stream:
                token = (
                    chunk.message.content
                    if hasattr(chunk, "message")
                    else chunk.get("message", {}).get("content", "")
                ) or ""

                full_response.append(token)

                # live streaming only when silent=False
                if not silent:
                    if self._token_callback:
                        self._token_callback(token)
                    else:
                        if len(full_response) == 1:
                            print("\nAiko-chan: ", end="", flush=True)
                        print(token, end="", flush=True)

            if not silent and not self._token_callback:
                print(flush=True)

        except Exception as e:
            msg = f"Stream failed: {e}"
            log.error(msg)
            
            # remove the dangling user turn so next call doesn't create a consecutive pair
            if self._history and self._history[-1]["role"] == "user":
                self._history.pop()
            if self._token_callback:
                self._token_callback(f"[think] {msg}")
            else:
                print(f"\n[think] {msg}")
            return ""

        return "".join(full_response)

    def _is_data_intent(self, user_input: str) -> bool:
        """
        Classify whether user_input warrants a live web search.

        Priority order:
          1. Social signal fast-path  → False  (no search)
          2. Factual signal fast-path → True   (search)
          3. LLM classifier fallback  → True/False (ambiguous inputs only)

        The LLM fallback is cheap: 1-token answer, temperature=0.0.
        """
        lowered = user_input.lower()

        # fast path: social phrasing — never search
        if any(sig in lowered for sig in _SOCIAL_SIGNALS):
            log.debug(f"[intent] SOCIAL (fast-path): {user_input!r}")
            return False

        # fast path: factual signals — always search
        if any(sig in lowered for sig in _FACTUAL_SIGNALS):
            log.debug(f"[intent] DATA (fast-path): {user_input!r}")
            return True

        # ambiguous — ask the LLM for a 1-token classification
        try:
            resp = self._client.chat(
                model=OLLAMA_MODEL,
                messages=[{
                    "role": "user",
                    "content": (
                        f'Is this message asking for factual external data '
                        f'(news, scores, prices, current events), or is it '
                        f'conversational/social?\n\n'
                        f'Message: "{user_input}"\n\n'
                        f'Reply with exactly one word: data OR social'
                    )
                }],
                stream=False,
                keep_alive=-1,
                options={"num_predict": 4, "temperature": 0.0},
            )
            answer = resp.message.content.strip().lower()
            result = "data" in answer
            log.debug(f"[intent] LLM classified {'DATA' if result else 'SOCIAL'}: {user_input!r}")
            return result
        except Exception as e:
            log.warning(f"Intent gate failed, defaulting to search: {e}")
            return True

    def _build_search_query(self, user_input: str) -> str:
        """
        Build a self-contained search query from user_input.

        For short follow-up messages (≤6 words), resolves pronouns and
        references using the previous user turn as context.
        Example: "How about Game 2?" → "NBA Finals 2026 Game 2 score"

        Falls back to raw user_input on any error or if no prior context exists.
        """
        if len(user_input.split()) <= 6 and self._history:
            last_user = next(
                (m["content"] for m in reversed(self._history)
                 if m["role"] == "user"),
                ""
            )
            if last_user and last_user != user_input:
                try:
                    resp = self._client.chat(
                        model=OLLAMA_MODEL,
                        messages=[{
                            "role": "user",
                            "content": (
                                f'Previous question: "{last_user}"\n'
                                f'Follow-up question: "{user_input}"\n\n'
                                f'Write a single complete search query that resolves '
                                f'any references. Return only the query, nothing else.'
                            )
                        }],
                        stream=False,
                        keep_alive=-1,
                        options={"num_predict": 20, "temperature": 0.0},
                    )
                    resolved = resp.message.content.strip()
                    log.debug(f"[search_query] {user_input!r} → {resolved!r}")
                    return resolved
                except Exception as e:
                    log.warning(f"Query resolution failed: {e}")

        return user_input

    def _sanitize_history(self, messages: list[dict]) -> list[dict]:
        """
        Enforce strict user/assistant alternation.
        Merges consecutive same-role messages (keeps last).
        Strips leading assistant turns — history must start with user.
        """
        if not messages:
            return []

        sanitized = [messages[0]]
        for msg in messages[1:]:
            if msg["role"] == sanitized[-1]["role"]:
                sanitized[-1] = msg
            else:
                sanitized.append(msg)

        while sanitized and sanitized[0]["role"] != "user":
            sanitized.pop(0)

        return sanitized

    def _store_async(self, user_input: str, response_text: str) -> None:
        """
        Enqueue a completed turn for background memory persistence.
        Non-blocking — chat() returns immediately after this call.
        The background worker in _mem_write_loop drains the queue serially.
        """
        self._mem_queue.put((user_input, response_text))

    def _mem_write_loop(self) -> None:
        """
        Serial background worker that drains the memory write queue.
        Runs for the lifetime of the process (daemon thread).
        """
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