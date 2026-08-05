"""
system/wakeup.py

Aiko's boot orchestrator — owns parallel subsystem startup and warmup sequencing.
main.py calls AikoWakeup().boot(...) and receives a BootResult with all live
subsystem references; it never needs to know the startup choreography.

Progress is reported through three injected callbacks so wakeup.py stays
completely UI-ignorant:
    on_loading(key)  — subsystem is starting
    on_done(key)     — subsystem finished successfully
    on_skip(key)     — subsystem skipped (e.g. text mode)

Each module owns its BOOT_LABELS dict; wakeup collects them and exposes
ALL_BOOT_LABELS so the UI(CLI/WebUI) can register display text before boot begins.

Usage:
    result = AikoWakeup().boot(
        on_loading = ui.step_loading,
        on_done    = ui.step_done,
        on_skip    = ui.step_skip,
    )
    think    = result.think
    memorize = result.memorize
    speak    = result.speak
    listen   = result.listen

Flow:
    ┌── init_think ────────────┐   ┌── init_memorize ───┐
    │ AikoThink() — no args    │   │ sqlite-vec+cleanup │   (parallel threads)
    │ boot + warmup            │   │ set mem_ready_evt  │
    │ wait mem_ready_evt       │   └─────────┬──────────┘
    │ set_memorize + idle_lrn  │             │
    │ prewarm semantic cache   │             │
    └───────────┬──────────────┘             │
                └───────────┬────────────────┘
                            ▼
                  join both threads
                  (raise if think failed; memorize None is OK)
                            │
                            ▼
                  construct speak (not yet wired to think)
                            │
                            ▼
                  hand off scheduler startup to system.schedule
                            │
                            ▼
                  voice pipeline (TTS warmup → think.set_speak(speak) → ASR + VAD staged init)
                            │
                            ▼
                  return BootResult

- Parallel phase — init_think and init_memorize run on separate threads at the same time.
- think is constructed with no arguments; memorize and speak both start as None and are
  injected later via set_memorize()/set_speak() once each is actually ready.
- think boots AikoThink, runs warmup, then blocks on mem_ready_evt.wait() until memory is
  done — then injects memory (set_memorize, may be None on failure), starts the idle
  learner (no-ops if memory is None), and prewarms the semantic route/capability caches
  before returning.
- memorize sets up sqlite-vec, runs cleanup, then always signals mem_ready_evt.set() in a
  finally — so think never hangs even if memory boot fails.
- Join point — main thread waits for both think_future/mem_future to finish.
- speak is constructed right after the join, but not wired into think until the voice
  stack is ready.
- Wakeup then hands the live refs to `system.schedule.start_scheduler()`, which owns
  the scheduler thread and all seeded jobs.
- Voice pipeline (sequential) — TTS warmup, then think_ref.set_speak(speak) (or None),
  then ASR staged init (load model → load VAD → join warmup → start barge-in monitor).
- Returns BootResult with all four live subsystem refs.

Failure logging policy — one log line per failure, with traceback + context:
    - _boot_step never logs. It only fires on_skip(key) and re-raises, so the exception
      propagates to whichever subsystem-level except block is actually equipped to say
      what failed and what degraded mode results (e.g. "Aiko will run without voice
      input").
    - Every subsystem-level except block logs exactly once, with log.exception(...) (or
      log.critical(..., exc_info=...) for the one fatal case — AikoThink), which already
      captures the full traceback.
    - This replaces the old pattern where _boot_step logged a traceback, then the outer
      except logged a summary (sometimes a third log.critical on top), turning one real
      failure into 2-3 near-duplicate log entries.
"""

from __future__ import annotations                          # evaluates type annotations later

from collections.abc import Callable                        # for defining boot functions
from concurrent.futures import ThreadPoolExecutor           # for parallel subsystem boot
from dataclasses import dataclass                           # for dataclass to hold subsystem references 
from typing import Any                                      # Any still lives in typing — collections.abc has no equivalent
import threading                                            # for booting up cognition core and memory system in parallel
import json                                                 # for reading the persisted OAuth session token
from pathlib import Path                                    # for locating the stored auth token file

# Must run before the system.* imports below — those modules read secrets
# from os.environ at import time, and this decrypts .env.age into os.environ.
# Idempotent (guarded by _LOADED), so it's a no-op if main.py already ran it —
# this is just a safety net for entrypoints that import this module directly.
from system.config import load_config                       # load secrets and configs before everything start (safety net)
load_config()

from system.log import get_logger                           # pass the logging to universal logger
log = get_logger(__name__)

from cognition.think import BOOT_LABELS as _THINK_LABELS    # for the booting status of cognition core
from cognition.memory.memorize import BOOT_LABELS as _MEM_LABELS      # for the booting status of memory system
from sensory.speak   import BOOT_LABELS as _SPEAK_LABELS    # for the booting status of speaking module
from sensory.listen  import BOOT_LABELS as _LISTEN_LABELS   # for the booting status of listening module

from cognition.memory.memorize import AikoMemorize                    # for initiating memory system
from cognition.think import AikoThink                        # for initiating cognitive core
from sensory.speak import AikoSpeak                          # for initiating speaking module
from sensory.listen import AikoListen                        # for initiating listening module
from system.schedule import (                                # for initiating scheduler system
    start_scheduler,
)

# ── result container ──────────────────────────────────────────────────────────

@dataclass(slots=True, frozen=True)
class BootResult:
    """Holds all live subsystem references produced during boot.

    frozen=True — nothing downstream should be reassigning these refs; if a
    subsystem needs to be swapped out later that should go through an explicit
    method on the owning class, not a silent BootResult mutation.
    """
    think:    AikoThink | None                              # cognition core - None if cognitive system boot failed
    memorize: AikoMemorize | None                           # memory system - None if memory system boot failed
    speak:    AikoSpeak | None                              # speaking module — None if TTS boot failed
    listen:   AikoListen | None                             # listening module — None if ASR/VAD boot failed

type BootCallback = Callable[[str], None]                   # Callback for boot progress: takes step key (string)

# ── helpers ───────────────────────────────────────────────────────────────────

def _stored_display_name() -> str | None:
    """Resolve the display name from the persisted OAuth session
    (~/.aiko/auth_token.json → user.login) without needing an active
    login context. Returns None when no usable token is stored.

    Used as a boot-time fallback so memory pins get the human-readable
    name (e.g. "OppaAI") even before any web session resolves it.
    """
    try:
        token_path = Path.home() / ".aiko" / "auth_token.json"     # persisted OAuth token
        if not token_path.is_file():                                # no stored session → nothing to resolve
            return None
        data = json.loads(token_path.read_text(encoding="utf-8"))   # read token payload
        user = data.get("user") or {}                               # nested user object
        return user.get("login") or None                            # e.g. "OppaAI"
    except Exception:
        return None                                                 # token unreadable → keep fallback behaviour

def _prewarm_semantic_cache(think) -> None:
    """Warm both semantic caches used by first-turn routing/capability
    matching, so the first real user turn never pays an embedding cost.

    Route exemplars (think._semantic_example_vectors): in-memory cache,
    then per-user on-disk npz cache (cognition.reason.cache_vector_path),
    then compute+persist if both miss.

    Capability trigger embeddings (agentic.capability._get_trigger_embedding):
    same three-tier pattern, sharing the same cache_vector_path helper —
    in-memory dict first, then on-disk npz, then compute+persist.

    On a warm disk cache, this whole call is disk loads only, no
    embedding calls. On a cold cache (first boot, or after a trigger/
    exemplar edit), it pays the full embed cost once and persists it.

    Args:
       think: AikoThink instance with a booted embedder (via memorize backend).
    """
    if think._get_memorize() is None:            # gate to skip semantic cache prewarm if memory system is unavailable
        log.info("[wakeup] Skipping semantic cache prewarm — no memory backend.")
        return
    from cognition.think import (
        _ROUTE_TERNARY_EXAMPLES,                # for top-level 3-way routing decision (agentic / webchat / localchat)
        _ROUTE_INSTRUCT_TERNARY,                # the instruction strings of the 3-way routing
    )
    try:
        # Prewarm intent routing cache
        think._semantic_example_vectors(_ROUTE_TERNARY_EXAMPLES, _ROUTE_INSTRUCT_TERNARY)    # prewarm routing cache in designated npz

        # Prewarm capability trigger embeddings (used by agentic_chat -> match_capabilities)
        from agentic.capability import CAPABILITIES, _get_trigger_embedding            # for loading intents and tools from Aiko's capabilities
        embedder = think._get_memorize()._mem._embedder                                # load the pre-embedded semantic vectors from npz files
        for cap in CAPABILITIES.values():                                              # loop through all Aiko's capabilities
            _get_trigger_embedding(cap, embedder)                                      # load all the semantic vectors into cache

        log.info("[wakeup] Semantic exemplar cache warmed (intent + capabilities)")    # log sucess
    except Exception:                                                                  # if error,
        log.exception("[wakeup] Semantic exemplar prewarm failed")                     # log failure — single point, full traceback


# ── wakeup ────────────────────────────────────────────────────────────────────

class AikoWakeup:
    """
    Parallel boot orchestrator for all Aiko cognitive subsystems.

    Boots AikoThink and AikoMemorize concurrently, then stages TTS and ASR
    init sequentially with granular progress reporting per step.
    Each subsystem owns its BOOT_LABELS; ALL_BOOT_LABELS merges them all
    so the UI can register display text before boot begins.
    """

    ALL_BOOT_LABELS: dict[str, str] = {
        **_THINK_LABELS,            # for register AikoThink status
        **_MEM_LABELS,              # for register AikoMemorize status
        **_SPEAK_LABELS,            # for register AikoSpeak status
        **_LISTEN_LABELS,           # for register AikoListen status
        "mcp_client": "Connect to Social MCP server",
    }

    def boot(
        self,
        on_loading: BootCallback,
        on_done:    BootCallback,
        on_skip:    BootCallback,
    ) -> BootResult:
        """
        Execute full boot sequence and return live subsystem references.

        Parallel phase: AikoThink + AikoMemorize boot concurrently.
        Sequential phase: TTS warmup → ASR staged init.
        Barge-in monitor started as the final ASR step so Silero is already
        warm and the VAD thread costs nothing before the first turn.
        """
        mem_ready_evt  = threading.Event()                           # thread-safe boolean flag for blocking until memory system is ready

        # ── parallel boot ─────────────────────────────────────────────────────

        def _boot_step(key: str, fn: Callable[[], Any] | None = None) -> Any:
            """Wrap a boot step with loading/done/skip callbacks.

            Args:
                key: Step identifier for callbacks.
                fn: Callable performing the step work. If None, this is a marker step.

            Returns:
                Result of fn(), or None if fn is None.

            Raises:
                Re-raises any exception from fn() after calling on_skip(). Deliberately
                does NOT log here — the caller's except block is the single point that
                logs (with log.exception / log.critical), since it's the only place that
                knows which subsystem this is and what degraded mode results. Logging
                here too was the source of the old double/triple-log-per-failure bug.
            """
            on_loading(key)                                             # annouce boot step starts
            if fn is None:                                              # if marker step — no work, just progress,
                on_done(key)                                            # annouce the message
                return None                                             # return None for no results
            try:                                                        # attempt to run boot step
                result = fn()                                           # call boot step function
            except Exception:                                           # if error,
                on_skip(key)                                            # annouce boot step skips
                raise                                                   # re-raise for the subsystem-level except to log + handle
            on_done(key)                                                # annouce boot step finishes
            return result                                               # return results of boot step function

        def init_think(memorize_getter):
            """memorize_getter is a zero-arg callable so init_think can pull
            the memorize result lazily, after mem_ready_evt fires — avoids needing
            the memorize future to exist before this closure is defined."""

            think = _boot_step('think_start', lambda: AikoThink())                            # initiate cognitive core
            _boot_step('think_warmup', lambda: (think.start_warmup(), think.join_warmup()))   # start warmup background thread
            _boot_step('think_mem_wait', lambda: mem_ready_evt.wait())                        # block until memorize thread finishes
            _boot_step('think_inject', lambda: (think.set_memorize(memorize_getter()), think.start_idle_learner()))  # inject memory backend to cognitive core and start idle learner thread (no-ops cleanly if memorize is None)
            _boot_step('think_prewarm', lambda: _prewarm_semantic_cache(think))               # load embed exemplars while booting
            return think                                                                      # return the live AutoThink object

        def init_memorize():
            try:
                memorize = _boot_step('mem_embed', lambda: AikoMemorize(silent=True))         # initiate memory system (with logging off to prevent duplicate)

                def _set_display_name():
                    """Pin the resolved display name to the memory backend before
                    any recall happens, so pinned memories can use a
                    human-readable name instead of a raw user_id."""
                    from system.userspace import current_display_name                         # access userspace module
                    display_name = current_display_name()                                     # get the username resolved from OAuth
                    if display_name == memorize.get_user_id():                                # fell back to raw user_id — try stored OAuth session
                        stored = _stored_display_name()
                        if stored:
                            display_name = stored
                    memorize.set_display_name(display_name)                                   # pass the username to memory system
                    if display_name == memorize.get_user_id():                                # if display name fell back to raw user id, log warning
                        log.warning(
                            "[wakeup] No display name for user_id=%s — memory pins "
                            "will use raw user_id until the user logs in.",
                            display_name,
                        )

                _boot_step('mem_display_name', _set_display_name)                             # pass the username to memory system
                _boot_step('mem_cleanup', lambda: memorize.cleanup())                         # prune decayed memories
                _boot_step('mem_ready')                                                       # mark the memory system ready

                return memorize                                                               # return the live AikoMemorize object
            except Exception:                                                                 # if error, log failure once — single point, full traceback
                log.exception("[wakeup] Memory boot failed — Aiko will run without persistent memory.")
                return None                                                                   # return None to indicate failure
            finally:                                                                          # whether success or failure,
                mem_ready_evt.set()                                                           # set memory ready flag to True to trigger any blocked thread

        with ThreadPoolExecutor(max_workers=2) as ex:                                         # start thread pool with 2 worker threads (for loading memory system and cognitive core concurrently)
            mem_future = ex.submit(init_memorize)                                             # start memory system boots on thread 1
            think_future = ex.submit(init_think, lambda: mem_future.result())                 # start cognitive core boots on thread 2

            # memorize's .result() never raises — init_memorize() always returns something
            # (None on failure, logged internally), so no try/except needed here.
            # think's .result() DOES re-raise on failure — caught below so we can still
            # drain mem_future before deciding whether boot failed.
            think_ref: AikoThink | None = None                                                # initiate AikoThink object
            think_exc: Exception | None                                                       # hold exception of cognitive core for chaining
            try:                                                                              # attempt to initiate cognitive core
                think_ref = think_future.result()                                             # block until finishes initiation of cognitive core
            except Exception as exc:                                                          # if error,
                think_exc = exc                                                               # logged once and chained into the raise later
            memorize = mem_future.result()                                                    # grab the results of memory system

        # ── MCP client boot (non-fatal) ─────────────────────────────────────────
        try:
            from agentic.mcp_client.bridge import bootstrap_mcp
            mcp_ok = _boot_step("mcp_client", lambda: bootstrap_mcp())
            if not mcp_ok:
                log.info("[wakeup] MCP client skipped or unavailable — Aiko will run without social posting tools.")
        except Exception:
            log.exception("[wakeup] MCP client boot failed")
            _boot_step("mcp_client", None)  # mark as skip in UI
            on_skip("mcp_client")

        if think_ref is None:                                                                 # if cognitive core returns None value, log error and raise runtime error
            log.critical(                                                                     # single log point: critical severity + full traceback in one line
                "[wakeup] AikoThink boot failed — cannot continue without cognition core.",
                exc_info=think_exc,
            )
            raise RuntimeError("AikoThink boot failed") from think_exc                        # chain from previous error point so callers/tracebacks still see the root cause

        # speak has no boot-time dependency on think or memorize, and nothing
        # inside init_think touches it — safe to construct after the parallel
        # phase instead of before it. Construction itself is non-fatal, same as
        # TTS warmup below — Aiko can run text-only if AikoSpeak() itself blows up.
        try:                                                                                  # attempt to initiate speaking module (sequentially)
            speak = AikoSpeak(silent=True)                                                    # load speaking module with internal logging inhibited
        except Exception:                                                                     # if error,
            log.exception("[wakeup] AikoSpeak construction failed — Aiko will run without voice output.")  # log failure
            speak = None                                                                      # set to None to indicate failure

        # Wakeup now only bootstraps the live subsystems and hands them to
        # system.schedule's scheduler startup helper.
        start_scheduler(
            on_due=think_ref.handle_scheduled_job,
            memorize=memorize,
            think=think_ref,
        )

        # ── voice subsystems ──────────────────────────────────────────────────

        # TTS — non-fatal: Aiko can run text-only if this fails. Gated on speak
        # not already being None (construction above may have failed).
        if speak is not None:                                                                    # gate to skip warmup of TTS model if speaking module boot failed
            try:                                                                                 # attempt the warmup of TTS model
                _boot_step('speak_miotts', lambda: speak.warmup())                               # load/prime the TTS model
                _boot_step('speak_ready')                                                        # log success
            except Exception:                                                                    # if error,
                log.exception("[wakeup] TTS boot failed — Aiko will run without voice output.")  # log failure
                speak = None                                                                     # set hand;e to None to indicate eror

        think_ref.set_speak(speak)                                                               # wires in speaking module only once if TTS model is known to be live or not

        # ASR — staged so each step reports independently; non-fatal. Construction
        # wrapped too, same reasoning as AikoSpeak() above.
        listen: AikoListen | None = None                                                                     # initiate listening module handle to None
        try:                                                                                                 # attempt to load listening module
            listen = AikoListen()                                                                            # load listening module
        except Exception:                                                                                    # if error,
            log.exception("[wakeup] AikoListen construction failed — Aiko will run without voice input.")    # log failure

        if listen is not None:                                                                               # gate to skip loading of ASR/VAD model if listening module boot failed
            try:
                _boot_step('listen_asr', lambda: listen.load_asr())                                          # load ASR model
                _boot_step('listen_silero', lambda: listen.load_vad())                                       # load VAD module
                _boot_step('listen_warmup', lambda: listen.join_warmup())                                    # kick off warmup thread
                _boot_step('listen_ready', lambda: listen.start_barge_in_monitor())                          # load VAD daemon for barge-in monitor — costs ~0 CPU at idle
            except Exception:                                                                                # if error,
                log.exception("[wakeup] ASR/VAD boot failed — Aiko will run without voice input.")           # log failure
                listen = None                                                                                # set handle to None to indicate error

        return BootResult(                                                                        # return the results of the bootup of the 4 modules:
            think    = think_ref,                                                                 # cognitive core
            memorize = memorize,                                                                  # memory system
            speak    = speak,                                                                     # speaking module
            listen   = listen,                                                                    # listening module
        )
