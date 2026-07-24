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
                  register deep-study handlers
                  (needs think_ref._client / _llm_model)
                            │
                            ▼
                  scheduler setup
            (ScheduleRunner.start, workspace-knowledge job, social lanes)
                            │
                            ▼
                  voice pipeline
            (TTS warmup → think.set_speak(speak) → ASR + VAD staged init)
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
- speak is constructed right after the join, but NOT wired into think yet (no
  think.set_speak() call here) — think doesn't get a live speak reference until the
  voice pipeline below finishes TTS warmup (or falls back to None on failure). Building
  it early just means it doesn't have to wait on the sequential scheduler-setup phase.
- Deep-study handler registration — runs right after the join, before the scheduler
  starts, because ScheduleRunner needs a handler already registered for the
  deep_study_start/stop jobs seeded into schedule.json, or they'll fire into a void.
  Needs think_ref._client / _llm_model, so it can only happen after think boots.
- Scheduler setup (sequential, single-threaded) — starts the one ScheduleRunner, ensures
  the workspace-knowledge job, registers social lanes.
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

# Must run before the system.* imports below — those modules read secrets
# from os.environ at import time, and this decrypts .env.age into os.environ.
# Idempotent (guarded by _LOADED), so it's a no-op if main.py already ran it —
# this is just a safety net for entrypoints that import this module directly.
from system.config import load_config                       # load secrets and configs before everything start (safety net)
load_config()

from system.log import get_logger                           # pass the logging to universal logger
log = get_logger(__name__)

from cognition.think import BOOT_LABELS as _THINK_LABELS    # for the booting status of cognition core
from memory.memorize import BOOT_LABELS as _MEM_LABELS      # for the booting status of memory system
from sensory.speak   import BOOT_LABELS as _SPEAK_LABELS    # for the booting status of speaking module
from sensory.listen  import BOOT_LABELS as _LISTEN_LABELS   # for the booting status of listening module

from memory.memorize import AikoMemorize                    # for initiating memory system
from cognition.think import AikoThink                       # for initiating cognitive core
from sensory.speak import AikoSpeak                         # for initiating speaking module
from sensory.listen import AikoListen                       # for initiating listening module
from system.schedule import (                               # for initiating scheduler system
    ScheduleRunner,
    register_scheduler,
    register_system_handler,
    ensure_workspace_knowledge_job,
    register_social_handlers,
)
from memory.reflect import generate_and_post                # for loading daily reflection into scheduler
from memory.consolidate import maybe_run_consolidation      # for loading monthly consolidation into scheduler

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
    if think._get_memorize() is None:
        log.info("[wakeup] Skipping semantic cache prewarm — no memory backend.")
        return
    from cognition.think import (
        _ROUTE_TERNARY_EXAMPLES,            # for top-level 3-way routing decision (agentic / webchat / localchat)
        _ROUTE_INSTRUCT_TERNARY,            # the instruction strings of the 3-way routing
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
                    """Pull the cached display name for this user and pin it to the
                    memory backend before any recall happens, so pinned memories are
                    attributed to a human-readable name instead of a raw user_id."""
                    from system.userspace import current_display_name                         # access userspace module
                    display_name = current_display_name()                                     # get the username resolved from OAuth
                    memorize.set_display_name(display_name)                                   # pass the username to memory system
                    if display_name == memorize.get_user_id():                                # if the cached username is the same as user id, log warning
                        log.warning(
                            "[wakeup] No cached display name for user_id=%s — memory pins "
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
            from concurrent.futures import wait, ALL_COMPLETED   # add to top-of-file imports

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

        # ── wire deep_studying into the scheduler's weekday/weekend window ────
        # Must happen before the ScheduleRunner below starts (or at least
        # before its first tick) so the "deep_study_start"/"deep_study_stop"
        # jobs seeded into schedule.json (system.schedule.ensure_deep_study_window_jobs)
        # have a registered handler to call into — otherwise they log
        # "unregistered handler" and silently never fire. Needs AikoThink's
        # LLM client/model, so it can only happen here, after think boots.
        from memory import learn                                                              # access self-learning module

        # NOTE (model-swap possibility): deep-study handlers are wired to think_ref's
        # client/model here because that's what's live at boot. quick_studying (in
        # memory/learn.py's idle_learner_loop) should stay on think's model — it fires
        # during short chat-idle gaps, so any swap latency would be paid on the interactive
        # path. deep_studying runs in scheduled off-hours windows (05:00-18:00 weekdays /
        # 05:00-10:00 weekends by default), which is exactly where an Active/Idle mode
        # split would pay off: Idle mode could tear down TTS/ASR/CV/(maybe embedder) and
        # load a bigger, smarter model just for deep_studying + other off-hour autonomous
        # jobs, then swap back before the window closes. When that mode exists, this call
        # site — not learn.py itself — is where the alternate client/model gets threaded
        # in; register_deep_study_handlers() already accepts client/model as plain params,
        # so no changes needed there.
        learn.register_deep_study_handlers(                                                   # 
            client=think_ref._client,                                                         #
            model=think_ref._llm_model,                                                       #
        )

        if memorize is None:                                                                  # if error during memory system boot,
            log.warning("[wakeup] Memory boot failed — ScheduleRunner starting without system jobs.")  # log warning, not error — Aiko keeps running, just degraded

        # NOTE: this is the ONE ScheduleRunner for the whole app. AikoThink
        # used to also construct its own ScheduleRunner in __init__, which
        # meant two independent daemon threads were both reading and firing
        # the same schedule.json — every due job (reminders, weekly_social,
        # and now the deep_study_start/stop window jobs) would fire twice.
        # That duplicate construction has been removed from cognition/think.py;
        # this is now the only instance, and it's the one registered via
        # register_scheduler() so tools can notify it of newly added jobs.
        scheduler = ScheduleRunner(
            on_due=think_ref.handle_scheduled_job,
            memorize=memorize,
            generate_and_post_fn=generate_and_post,
            consolidate_fn=maybe_run_consolidation,
        )
        register_scheduler(scheduler)  # Allow tools to notify scheduler of new jobs
        scheduler.start()

        # Schedule-driven workspace/knowledge scan. The schedule runner keeps
        # using one sleep-until-next-event loop; the KB scan is represented in
        # schedule.json as a normal interval handler job.
        if memorize is not None:
            try:
                from memory.knowledge import ingest_workspace_knowledge_folder

                register_system_handler(
                    "workspace_knowledge_scan",
                    lambda _memorize: ingest_workspace_knowledge_folder(
                        embedder=_memorize._mem._embedder,
                        user_id=_memorize.get_user_id(),
                    ),
                )
                ensure_workspace_knowledge_job()
                scheduler.notify_new_job()
                log.info("[wakeup] Workspace knowledge scan schedule ensured")
            except Exception:
                log.exception("[wakeup] Workspace knowledge scan schedule failed")

        # Schedule-driven social lanes (weekly postcard, photo inbox, video
        # inbox). register_social_handlers() registers all three handlers
        # with the system handler registry and idempotently seeds their
        # schedule.json jobs (see system/schedule.py). Doesn't depend on
        # memorize the way the workspace-knowledge scan does — the
        # weekly/photo/video handlers are called with memorize but the
        # photo/video ones just absorb and ignore it — but this is kept
        # here, after memory boot, so all "post-scheduler" job seeding
        # happens in one place and any failure here doesn't affect the
        # scheduler start above.
        try:
            register_social_handlers()
            scheduler.notify_new_job()
            log.info("[wakeup] Social handlers registered and schedules ensured")
        except Exception:
            log.exception("[wakeup] Social handler registration failed")

        # ── voice subsystems ──────────────────────────────────────────────────

        # TTS — non-fatal: Aiko can run text-only if this fails. Gated on speak
        # not already being None (construction above may have failed).
        if speak is not None:
            try:
                _boot_step('speak_miotts', lambda: speak.warmup())
                _boot_step('speak_ready')
            except Exception:
                log.exception("[wakeup] TTS boot failed — Aiko will run without voice output.")
                speak = None

        think_ref.set_speak(speak) # wires in speak only once we know if it's live or None

        # ASR — staged so each step reports independently; non-fatal. Construction
        # wrapped too, same reasoning as AikoSpeak() above.
        listen: AikoListen | None = None
        try:
            listen = AikoListen()
        except Exception:
            log.exception("[wakeup] AikoListen construction failed — Aiko will run without voice input.")

        if listen is not None:
            try:
                _boot_step('listen_asr', lambda: listen.load_asr())
                _boot_step('listen_silero', lambda: listen.load_vad())          # also kicks off warmup thread
                _boot_step('listen_warmup', lambda: listen.join_warmup())
                _boot_step('listen_ready', lambda: listen.start_barge_in_monitor())  # VAD daemon — costs ~0 CPU at idle
            except Exception:
                log.exception("[wakeup] ASR/VAD boot failed — Aiko will run without voice input.")
                listen = None

        return BootResult(
            think    = think_ref,
            memorize = memorize,
            speak    = speak,
            listen   = listen,
        )
