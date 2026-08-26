"""
system/wakeup.py

Aiko's boot orchestrator — owns parallel subsystem startup and warmup sequencing.
The front end (run_webui()/run_session()) calls AikoWakeup().boot(...) and
receives a BootResult with all live subsystem references; it never needs to
know the startup choreography.

Boot is pre-auth safe: no step here requires a logged-in user. Memory opens on
the tempfile-backed guest DB until a real user connects (AikoWeb then calls
switch_user()/set_display_name() per browser session), and user-scoped seeding
(playbooks, schedule jobs) is deferred to the first authenticated connection —
see interface/webui/webui.py's _on_user_active().

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
                  voice pipeline (TTS warmup → think.set_speak(speak) → construct AikoListen)
                            │
                            ▼
                  return BootResult
                  (semantic route/capability cache prewarming deferred to post-auth)

- Parallel phase — init_think and init_memorize run on separate threads at the same time.
- think is constructed with no arguments; memorize and speak both start as None and are
  injected later via set_memorize()/set_speak() once each is actually ready.
- think boots AikoThink, runs warmup, then blocks on mem_ready_evt.wait() until memory is
  done — then injects memory (set_memorize, may be None on failure) and starts the idle
  learner (no-ops if memory is None) before returning.
- memorize sets up sqlite-vec, runs cleanup, then always signals mem_ready_evt.set() in a
  finally — so think never hangs even if memory boot fails.
- Join point — main thread waits for both think_future/mem_future to finish.
- speak is constructed right after the join, but not wired into think until the voice
  stack is ready.
- Wakeup then hands the live refs to `system.schedule.start_scheduler()`, which owns
  the scheduler thread and all seeded jobs.
- Voice pipeline (sequential) — TTS warmup, then think_ref.set_speak(speak) (or None),
  then AikoListen() is constructed (non-fatal). ASR/VAD models load lazily on first
  mic arm via AikoListen.ensure_ready() — not part of boot.
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
from dataclasses import dataclass                           # for dataclass to hold subsystem refs
from typing import Any                                      # Any still lives in typing — collections.abc has no equivalent
import threading                                            # for booting up cognition core and memory system in parallel

# Must run before the system.* imports below — those modules read secrets
# from os.environ at import time, and this decrypts .env.age into os.environ.
# Idempotent (guarded by _LOADED), so it's a no-op if main.py already ran it —
# this is just a safety net for entrypoints that import this module directly.
from system.config import load_config                       # load secrets and configs before everything start (safety net)
load_config()

from system.log import get_logger                           # pass the logging to universal logger
log = get_logger(__name__)

from cognition.think import BOOT_LABELS as _THINK_LABELS    # for the booting status of cognition core
from cognition.memory.memorize import BOOT_LABELS as _MEM_LABELS      # for booting status of memory
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

BootCallback = Callable[[str], None]                        # Callback for boot progress: takes step key (string)

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
        Sequential phase: TTS warmup → construct AikoListen (ASR/VAD models
        themselves load lazily on first mic arm, not here).
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
            on_loading(key)                                             # announce boot step starts
            if fn is None:                                              # if marker step — no work, just progress,
                on_done(key)                                            # announce the message
                return None                                             # return None for no results
            try:                                                        # attempt to run boot step
                result = fn()                                           # call boot step function
            except Exception:                                           # if error,
                on_skip(key)                                            # announce boot step skips
                raise                                                   # re-raise for the subsystem-level except to log + handle
            on_done(key)                                                # announce boot step finishes
            return result                                               # return results of boot step function

        def init_think(memorize_getter: Callable[[], Any]):
            """memorize_getter is a zero-arg callable so init_think can pull
            the memorize result lazily, after mem_ready_evt fires — avoids needing
            the memorize future to exist before this closure is defined."""

            think = _boot_step('think_start', lambda: AikoThink())                            # initiate cognitive core
            _boot_step('think_warmup', lambda: (think.start_warmup(), think.join_warmup()))   # start warmup thread, then block until it finishes
            _boot_step('think_mem_wait', lambda: mem_ready_evt.wait())                        # block until memorize thread finishes
            _boot_step('think_inject', lambda: (think.set_memorize(memorize_getter()), think.start_idle_learner()))  # inject memory backend + start idle learner (no-ops if memorize is None)
            # NOTE: semantic-cache prewarm moved to system/prepare.run_post_auth() —
            # as guest it couldn't persist per-user npz caches anyway, and post-login
            # it loads the real user's existing disk cache instead of recomputing.
            return think                                                                      # return the live AikoThink object

        def init_memorize():
            try:
                memorize = _boot_step('mem_embed', lambda: AikoMemorize(silent=True))        # initialize memory system (with logging suppressed to prevent duplicate)

                # No user-id work here — boot may run before anyone logs in.
                # Display-name pinning + per-user store switching happen when a
                # real identity connects (webui._ws_handler / cli.py); cleanup
                # for the real user runs in webui's post-login hook.

                def _mem_cleanup_if_real_user():
                    """Pruning a throwaway guest tempfile DB that was never
                    written to is pure waste — skip it until a real identity
                    is bound (cleanup() itself no-ops too)."""
                    if memorize.get_user_id() == "guest":
                        log.info("[wakeup] Skipping memory cleanup — guest boot.")
                        return
                    memorize.cleanup()

                _boot_step('mem_cleanup', _mem_cleanup_if_real_user)
                _boot_step('mem_ready')                                                       # mark the memory system ready

                return memorize                                                               # return the live AikoMemorize object
            except Exception:                                                                 # if error, log failure once — single point, full traceback
                log.exception("[wakeup] Memory boot failed — Aiko will run without persistent memory.")
                on_skip('mem_ready')                                                          # resolve the marker step so the UI doesn't hang on it
                return None                                                                   # return None to indicate failure
            finally:                                                                          # whether success or failure,
                mem_ready_evt.set()                                                           # signal memory ready; wake any waiting thread

        with ThreadPoolExecutor(max_workers=2) as ex:                                         # parallel memory and cognition boot
            mem_future = ex.submit(init_memorize)                                             # memory boot on thread 1
            think_future = ex.submit(init_think, lambda: mem_future.result())                 # cognitive core boot on thread 2 (waits for memory)

            # memorize's .result() never raises — init_memorize() always returns something
            # (None on failure, logged internally), so no try/except needed here.
            # think's .result() DOES re-raise on failure — caught below so we can still
            # drain mem_future before deciding whether boot failed.
            think_ref: AikoThink | None = None                                                # will hold AikoThink reference
            think_exc: Exception | None = None                                                # holds exception from cognitive core initialization for error chaining
            try:                                                                              # retrieve cognitive core initialization result
                think_ref = think_future.result()                                             # block until cognitive core initialization completes
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
            # _boot_step already fired on_loading -> on_skip internally
            # before re-raising — nothing further to signal here.

        if think_ref is None:                                                                 # if cognitive core initialization failed (returned None)
            log.critical(                                                                     # single log point: critical severity + full traceback in one line
                "[wakeup] AikoThink boot failed — cannot continue without cognition core.",
                exc_info=think_exc,
            )
            raise RuntimeError("AikoThink boot failed") from think_exc                        # chain from previous error point so callers/tracebacks still see the root cause

        # Live Grasp working-memory hub: now owned by AikoMemorize (wm_* API)
        # — no boot-time install step. think.py injects/records via memorize
        # directly, so the old grasp_hub.install_into_think() monkeypatch
        # (and its signature-drift crash class) is gone entirely.

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
        if speak is not None:                                                                    # skip if AikoSpeak construction failed
            try:                                                                                 # warm up TTS model
                _boot_step('speak_miotts', lambda: speak.warmup())                               # load/cache TTS model
                _boot_step('speak_ready')                                                        # mark ready
            except Exception:                                                                    # if warmup failed,
                log.exception("[wakeup] TTS boot failed — Aiko will run without voice output.")  # log failure
                speak = None                                                                     # set handle to None to indicate error

        think_ref.set_speak(speak)                                                               # inject speak (may be None if TTS boot failed)

        # ASR — construction only; models load lazily on first mic arm via
        # AikoListen.ensure_ready() (see sensory/listen.py). Keeps boot fast
        # and text-mode RAM low. Non-fatal, same as before.
        listen: AikoListen | None = None
        try:
            listen = AikoListen()
        except Exception:
            log.exception("[wakeup] AikoListen construction failed — Aiko will run without voice input.")

        return BootResult(                                                                        # all four subsystem references
            think    = think_ref,                                                                 # cognitive core (always live; fatal if None)
            memorize = memorize,                                                                  # memory system (None on failure)
            speak    = speak,                                                                     # speech synthesis (None if TTS boot failed)
            listen   = listen,                                                                    # speech recognition (None if construction failed)
        )
