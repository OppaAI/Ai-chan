"""
core/think.py

Aiko's cognitive loop.
  - Drains memory write queue before each recall (prevents write-lag misses)
  - Retrieves relevant memories before each turn
  - Proactively searches the web for data-intent queries before LLM speaks
  - Streams llama.cpp response to console + TTS simultaneously
  - Stores the turn into long-term memory after each response (background thread)
  - Supports single-shot reasoning mode via set_reasoning(True) / /think command
"""

import logging
import os
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

_BASE_PREDICT    = 280   # normal token budget — soul.md says 2–3 sentences default
_REASONING_SCALE = 3     # multiplier applied to max_tokens in reasoning mode

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
# NOTE: checked only when no factual signal matches first (see _is_data_intent).
_SOCIAL_SIGNALS = frozenset([
    "wanna", "want to", "would you", "shall we",
    "let's", "lets", "together", "with me", "join me",
    "how are you", "what do you think", "do you like",
])

# Inputs matching ANY of these are immediately classified as data — search fires.
# Matched on word boundaries to avoid "win" matching "winter"/"wind"/"wine", etc.
_FACTUAL_SIGNALS = frozenset([
    "who", "what", "when", "where", "how many", "how much",
    "score", "result", "latest", "news", "current", "today",
    "price", "weather", "won", "win", "lost", "beat",
    "game", "final", "finals", "points", "scored",
    "standing", "ranking", "bitcoin", "crypto", "stock",
    "temperature", "forecast", "match", "series",
])

_FACTUAL_RE = re.compile(
    r'\b(?:' + '|'.join(re.escape(s) for s in _FACTUAL_SIGNALS) + r')\b',
    re.IGNORECASE,
)
_SOCIAL_RE = re.compile(
    r'\b(?:' + '|'.join(re.escape(s) for s in _SOCIAL_SIGNALS) + r')\b',
    re.IGNORECASE,
)


# ── think ─────────────────────────────────────────────────────────────────────

class AikoThink:
    """
    Aiko's conversational core.
    speak is injected pre-warmed from wakeup.py.
    LLM warmup starts immediately on init in a background thread.
    wakeup.py calls join_warmup() to block until the model is hot.
    """

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

        self._warmup_thread = threading.Thread(target=self._warmup_llm, daemon=True)
        self._warmup_thread.start()

    def _warmup_llm(self) -> None:
        try:
            self._client.chat.completions.create(
                model=LLAMACPP_MODEL,
                messages=[{"role": "user", "content": "hi"}],
                stream=False,
                max_tokens=1,
            )
        except Exception as e:
            log.warning("LLM warmup failed — llama-server may not be running: %s", e)

    def join_warmup(self) -> None:
        """Block until LLM warmup completes. Called by wakeup.py before boot finishes."""
        if self._warmup_thread and self._warmup_thread.is_alive():
            self._warmup_thread.join()

    # ── public api ────────────────────────────────────────────────────────────

    def chat(
        self,
        user_input: str,
        token_callback=None,
        _skip_search: bool = False,
        _history_label: str | None = None,
    ) -> str:
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
        user prompt with a step-by-step instruction and triples max_tokens.
        Auto-resets to False after each turn — single-shot by design.

        Args:
            user_input:      The text to process as this turn's input. When
                              called internally from web_search(), this is the
                              full search-results blob, not what Oppa typed.
            token_callback:  Optional per-token/per-chunk display callback.
            _skip_search:    Internal flag. When True, step 4 (proactive web
                              search) is skipped entirely. Set by web_search()
                              to prevent the search-results text from itself
                              triggering a second, recursive search.
            _history_label:  Internal override. When given, this string is
                              stored in conversation history and long-term
                              memory instead of the raw user_input — used by
                              web_search() so history/memory record the
                              original query ("nba scores") rather than the
                              full scraped results blob.
        """
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

        # 4. proactive web search — fires before LLM speaks so no guessing.
        #    Skipped when this turn IS already search-results text (web_search()
        #    path) to prevent recursive searching.
        if not _skip_search and self._is_data_intent(user_input):
            try:
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
            except Exception as e:
                # never let a broken search path crash the whole turn —
                # degrade to no-search-context instead
                log.error(f"Web search step failed, continuing without it: {e}")

        # 5. wrap user turn with reasoning instruction if active.
        #    The wrapper is applied ONLY to the message sent to the LLM —
        #    history stores the clean, unwrapped input so future turns don't
        #    see stale "think step by step" instructions baked into old turns.
        if self._reasoning:
            llm_prompt = (
                f"{user_input}\n\n"
                "Think through this carefully before answering. "
                "Show your reasoning inside <think> tags, then give your final answer."
            )
        else:
            llm_prompt = user_input

        # what gets stored in history/memory: the label if given (original
        # query, for web_search() calls), otherwise the raw user_input —
        # never the reasoning-wrapped or search-results version.
        history_entry = _history_label if _history_label is not None else user_input

        # cap on raw growth of self._history itself — independent of
        # CONTEXT_WINDOW_TURNS, which only controls how much gets sent to the
        # LLM each turn. Without this, self._history grows by 2 entries every
        # turn for the life of the process. Kept generously above the LLM
        # context window so last_turn()/_build_search_query() still have
        # recent-but-not-in-window turns to look back on if needed.
        _HISTORY_HARD_CAP = CONTEXT_WINDOW_TURNS * 10

        with self._history_lock:
            # 6. append user turn (clean label, not the wrapped/blob version)
            self._history.append({"role": "user", "content": history_entry})

            # prevent unbounded growth — truncate the actual list, not just
            # a slice of it, so old turns are freed rather than retained
            if len(self._history) > _HISTORY_HARD_CAP:
                self._history = self._history[-_HISTORY_HARD_CAP:]

            # 7. trim history to context window
            trimmed = self._history[-(CONTEXT_WINDOW_TURNS * 2):]

        trimmed = self._sanitize_history(trimmed)

        # swap the last message's content to the full LLM-facing prompt
        # (reasoning wrapper or full search-results blob) for this call only —
        # history itself keeps the clean label set above.
        if trimmed and trimmed[-1]["role"] == "user" and llm_prompt != history_entry:
            trimmed = trimmed[:-1] + [{"role": "user", "content": llm_prompt}]

        # 8. stream LLM response (silent — display handled in _emit)
        raw_response = self._stream_response(trimmed, system=system, token_callback=token_callback)

        # 9. emit to callback + TTS
        self._emit(raw_response, token_callback=token_callback)

        with self._history_lock:
            # 10. append assistant turn to history
            self._history.append({"role": "assistant", "content": raw_response})
            if len(self._history) > _HISTORY_HARD_CAP:
                self._history = self._history[-_HISTORY_HARD_CAP:]

        # 11. persist to memory (background) — store the clean label, never
        #     the reasoning-wrapped prompt or raw search-results blob
        self._store_async(history_entry, raw_response)

        # 12. auto-reset reasoning mode
        self._reasoning = False

        return raw_response

    def web_search(self, query: str, token_callback=None) -> str:
        """
        Run a web search and feed results into a normal chat turn.
        Called by the /web command — no search logic leaks out.

        Passes _skip_search=True so the search-results text injected as
        user_input doesn't itself trigger a second, recursive search, and
        _history_label=query so history/memory record the original query
        rather than the full scraped results blob.
        """
        from core.tools import web_search_context
        context = web_search_context(query)
        if context is None:
            msg = f"[no results for: {query}]"
            if token_callback:
                token_callback(msg)
            return msg
        return self.chat(
            context,
            token_callback=token_callback,
            _skip_search=True,
            _history_label=query,
        )

    def reset_context(self) -> None:
        """Clear the in-memory conversation history for a fresh session."""
        with self._history_lock:
            self._history.clear()

    def last_turn(self) -> tuple[str, str] | None:
        """
        Return the latest complete user/assistant exchange, if one exists.
        Returns (user_text, assistant_text) or None.
        """
        with self._history_lock:
            history_snapshot = list(self._history)

        users = [
            m["content"].strip() for m in history_snapshot
            if m.get("role") == "user" and (m.get("content") or "").strip()
        ]
        assistants = [
            m["content"].strip() for m in history_snapshot
            if m.get("role") == "assistant" and (m.get("content") or "").strip()
        ]
        if not users or not assistants:
            return None
        return users[-1], assistants[-1]

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

    def _emit(self, text: str, token_callback=None) -> None:
        """
        Send display text to token_callback (TUI) or stdout, and feed TTS.
        Streams word-by-word to token_callback for a natural typing feel,
        even though the LLM has already finished (silent mode).
        TTS receives the full text at once for uninterrupted audio.

        token_callback is taken as a parameter (not read from self) so
        concurrent chat() calls can't crossfire on which callback fires.
        """
        if not text:
            return

        if token_callback:
            words = text.split(" ")
            for i, word in enumerate(words):
                chunk = word if i == 0 else " " + word
                token_callback(chunk)
                time.sleep(float(os.getenv("EMIT_DELAY", 0.012)))  # ~80 wpm feel
        else:
            print(f"\nAiko-chan: {text}", flush=True)

        if self._speak:
            self._speak.feed(text)
            self._speak.play_async()

    def _stream_response(
        self,
        messages: list[dict],
        system: str = "",
        silent: bool = True,
        token_callback=None,
    ) -> str:
        """
        Collect the full LLM response from a stream.
        Live token-by-token printing only happens when silent=False; by
        default (silent=True) the response is gathered in full and replayed
        word-by-word via _emit() afterward, so console/TTS output and the
        actual generation are decoupled.

        max_tokens is scaled by _REASONING_SCALE when reasoning mode is
        active to budget for the <think> scratchpad.

        token_callback is taken as a parameter (not read from self) so
        concurrent chat() calls can't crossfire on which callback fires.

        Args:
            messages: Full message list (system prompt + trimmed history).
            system:   Optional system prompt to inject.

        Returns:
            The complete response text assembled from all streamed tokens.
        """
        full_response = []

        # triple token budget in reasoning mode to fit the <think> scratchpad
        max_tokens = _BASE_PREDICT * _REASONING_SCALE if self._reasoning else _BASE_PREDICT

        all_messages = (
            [{"role": "system", "content": system}] + messages
            if system else messages
        )

        try:
            stream = self._client.chat.completions.create(
                model=LLAMACPP_MODEL,
                messages=all_messages,
                stream=True,
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

                # live streaming only when silent=False
                if not silent:
                    if token_callback:
                        token_callback(token)
                    else:
                        if len(full_response) == 1:
                            print("\nAiko-chan: ", end="", flush=True)
                        print(token, end="", flush=True)

            if not silent and not token_callback:
                print(flush=True)

        except Exception as e:
            msg = f"Stream failed: {e}"
            log.error(msg)

            # remove the dangling user turn so next call doesn't create a consecutive pair
            with self._history_lock:
                if self._history and self._history[-1]["role"] == "user":
                    self._history.pop()
            if token_callback:
                token_callback(f"[think] {msg}")
            else:
                print(f"\n[think] {msg}")
            return ""

        return "".join(full_response)

    def _is_data_intent(self, user_input: str) -> bool:
        """
        Classify whether user_input warrants a live web search.

        Priority order:
          1. Factual signal fast-path (word-boundary match) → True (search)
          2. Social signal fast-path  (word-boundary match) → False (no search)
          3. Combined LLM classifier  → True/False (ambiguous inputs only)

        Factual signals are checked FIRST so a query like "what's the score,
        do you know?" — which contains both a factual cue ("score") and a
        conversational cue ("do you know") — still triggers a search rather
        than being silently swallowed by the social fast-path.

        For ambiguous inputs (no fast-path match), this delegates to
        _classify_and_resolve(), which does intent classification AND query
        resolution in a single LLM call instead of two sequential ones. The
        resolved query is cached on self._pending_search_query so
        _build_search_query() can reuse it without a second call.
        """
        # fast path: factual signals — always search (checked first, see above)
        if _FACTUAL_RE.search(user_input):
            log.debug(f"[intent] DATA (fast-path): {user_input!r}")
            return True

        # fast path: social phrasing — never search
        if _SOCIAL_RE.search(user_input):
            log.debug(f"[intent] SOCIAL (fast-path): {user_input!r}")
            return False

        # ambiguous — single combined LLM call does both jobs at once
        is_data, resolved_query = self._classify_and_resolve(user_input)
        self._pending_search_query = resolved_query if is_data else None
        return is_data

    def _classify_and_resolve(self, user_input: str) -> tuple[bool, str]:
        """
        Single LLM call that both classifies intent AND resolves the query
        (pronoun/reference resolution against the previous turn) in one shot,
        replacing what used to be two sequential LLM calls
        (_is_data_intent's old fallback + _build_search_query's old fallback).

        Returns (is_data_intent, resolved_query). resolved_query is only
        meaningful when is_data_intent is True; it falls back to the raw
        user_input if resolution wasn't needed/possible.
        """
        with self._history_lock:
            last_user = next(
                (m["content"] for m in reversed(self._history)
                 if m["role"] == "user"),
                ""
            )
        has_context = bool(last_user and last_user != user_input)

        context_block = (
            f'Previous question: "{last_user}"\n' if has_context else ""
        )

        try:
            resp = self._client.chat.completions.create(
                model=LLAMACPP_MODEL,
                messages=[{
                    "role": "user",
                    "content": (
                        f'{context_block}'
                        f'Message: "{user_input}"\n\n'
                        f'Is this message asking for factual external data '
                        f'(news, scores, prices, current events), or is it '
                        f'conversational/social?\n\n'
                        f'If it is a data request, also resolve any pronouns '
                        f'or references against the previous question (if given) '
                        f'into a single self-contained search query.\n\n'
                        f'Reply in EXACTLY this format, nothing else:\n'
                        f'data|<search query>\n'
                        f'or:\n'
                        f'social|none'
                    )
                }],
                stream=False,
                max_tokens=32,
                temperature=0.0,
            )
            answer = resp.choices[0].message.content.strip()
            label, _, rest = answer.partition("|")
            label = label.strip().lower()
            rest  = rest.strip()

            is_data = "data" in label
            resolved = rest if (is_data and rest and rest.lower() != "none") else user_input

            log.debug(f"[intent+query] {user_input!r} → is_data={is_data}, query={resolved!r}")
            return is_data, resolved
        except Exception as e:
            log.warning(f"Combined intent/query classification failed, defaulting to search: {e}")
            return True, user_input

    def _build_search_query(self, user_input: str) -> str:
        """
        Build a self-contained search query from user_input.

        If _is_data_intent() already resolved the query via the combined
        classifier (ambiguous-input path), that cached result is reused here
        with no extra LLM call. Otherwise (fast-path matches don't resolve
        queries), falls back to the original behavior: for short follow-up
        messages (≤6 words) with prior history, makes a single LLM call to
        resolve pronouns/references; otherwise returns user_input unchanged.
        """
        pending = self._pending_search_query
        if pending is not None:
            self._pending_search_query = None  # consume — one-shot cache
            return pending

        if len(user_input.split()) <= 6:
            with self._history_lock:
                last_user = next(
                    (m["content"] for m in reversed(self._history)
                     if m["role"] == "user"),
                    ""
                )
            if last_user and last_user != user_input:
                try:
                    resp = self._client.chat.completions.create(
                        model=LLAMACPP_MODEL,
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
                        max_tokens=20,
                        temperature=0.0,
                    )
                    resolved = resp.choices[0].message.content.strip()
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
