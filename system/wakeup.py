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

Aiko's boot orchestrator — owns parallel subsystem startup and warmup sequencing.

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
                            │
                            ▼
              construct speak, think.set_speak(speak)
                            │
                            ▼
                  scheduler setup
            (deep-study handlers, jobs, social lanes)
                            │
                            ▼
                  voice pipeline
            (TTS warmup → ASR + VAD staged init)
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
- speak is constructed and injected via think.set_speak() only after the join — nothing
  in init_think touches speak during boot, so there's no reason to build it earlier.
- Scheduler setup (sequential, single-threaded) — registers deep-study handlers, starts
  the one ScheduleRunner, ensures the workspace-knowledge job, registers social lanes.
- Voice pipeline (sequential) — TTS warmup, then ASR staged init (load model → load VAD →
  join warmup → start barge-in monitor).
- Returns BootResult with all four live subsystem refs.
"""

from __future__ import annotations            # evaluates type annotations later

from dataclasses import dataclass             # for dataclass to hold subsystem references 
from typing import Callable                   # for define boot functions
import threading                              # for booting up cognition core and memory system in parallel

# Must run before the system.* imports below — those modules read secrets
# from os.environ at import time, and this decrypts .env.age into os.environ.
# Idempotent (guarded by _LOADED), so it's a no-op if main.py already ran it —
# this is just a safety net for entrypoints that import this module directly.
from system.config import load_config          # load secrets and configs before everything start (safety net)
load_config()

from system.log import get_logger              # pass the logging to universal logger
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

@dataclass(slots=True)
class BootResult:
    """Holds all live subsystem references produced during boot."""
    think:    AikoThink | None        # cognition core
    memorize: AikoMemorize | None     # memory system
    speak:    AikoSpeak               # speaking module
    listen:   AikoListen              # listening module

type BootCallback = Callable[[str], None]            # type hint for the subsystem callable and none

# ── helpers ───────────────────────────────────────────────────────────────────

def _prewarm_semantic_cache(think) -> None:
    """Embed route and capability exemplars at boot so first-turn latency is cold-free."""
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
    except Exception as e:                                                             # if error,
        log.warning("[wakeup] Semantic exemplar prewarm failed: %s", e)                # log failure


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
        from concurrent.futures import ThreadPoolExecutor            # for managing pool of worker threads
        mem_ready_evt  = threading.Event()                           # thread-safe boolean flag for blocking until memory system is ready
    
        # ── parallel boot ─────────────────────────────────────────────────────
    
        def init_think(memorize_getter):
            """memorize_getter is a zero-arg callable so init_think can pull
            the memorize result lazily, after mem_ready_evt fires — avoids needing
            the memorize future to exist before this closure is defined."""

            on_loading('think_start')                                # announce loading of cognitive core starts
            think = AikoThink()                                      # initiate cognitive core
            on_done('think_start')                                   # announce loading of cognitive core finishes
            
            on_loading('think_warmup')                               # announce warmup of cognitive core starts
            think.start_warmup()                                     # start warmup background thread
            think.join_warmup()                                      # block until warmup thread finishes
            on_done('think_warmup')                                  # announce loading of cognitive core finishes
            
            on_loading('think_mem_wait')                             # announce waiting for memorize thread starts
            mem_ready_evt.wait()                                     # block until memorize thread finishes
            on_done('think_mem_wait')                                # announce waiting for memorize thread finishes
            
            think.set_memorize(memorize_getter())                    # inject memory backend to cognitive core (or None if memory boot failed)
            think.start_idle_learner()                               # start idle learner thread; no-ops cleanly if memorize is None
            
            on_loading('think_prewarm')
            _prewarm_semantic_cache(think)                           # embed exemplars while booting
            on_done('think_prewarm')
            return think
    
        def init_memorize():
            try:
                on_loading('mem_sqlite_vec')
                try:
                    memorize = AikoMemorize(silent=True)
                    from system.userspace import current_display_name
                    display_name = current_display_name()
                    memorize.set_display_name(display_name)
                    if display_name == memorize.get_user_id():
                        log.warning(
                            "[wakeup] No cached display name for user_id=%s — memory pins "
                            "will use raw user_id until the user logs in.",
                            display_name,
                        )
                    on_done('mem_sqlite_vec')
                except Exception:
                    on_skip('mem_sqlite_vec')
                    raise
    
                on_loading('mem_embed')
                on_done('mem_embed')
    
                on_loading('mem_cleanup')
                try:
                    memorize.cleanup()
                    on_done('mem_cleanup')
                except Exception:
                    on_skip('mem_cleanup')
                    raise
    
                on_loading('mem_ready')
                on_done('mem_ready')
                return memorize
            except Exception:
                log.exception("Memory boot failed — Aiko will run without persistent memory.")
                return None          # explicit sentinel, not a silent list-slot
            finally:
                mem_ready_evt.set()
    
        with ThreadPoolExecutor(max_workers=2) as ex:
            mem_future = ex.submit(init_memorize)
            think_future = ex.submit(init_think, lambda: mem_future.result())
            # .result() re-raises any exception the worker thread hit — no
            # silent None left behind unless init_memorize/init_think decided
            # to return None deliberately (as init_memorize does above).
            from concurrent.futures import wait
            done, _ = wait([mem_future, think_future])
            try:
                think_ref = think_future.result()
            except Exception:
                log.exception("AikoThink failed to boot.")
                think_ref = None
            memorize = mem_future.result()   # never raises — already caught internally

        # speak has no boot-time dependency on think or memorize, and nothing
        # inside init_think touches it — safe to construct after the parallel
        # phase instead of before it.
        speak = AikoSpeak(silent=True)
        if think_ref is not None:
            think_ref.set_speak(speak)
        
        # ── wire deep_studying into the scheduler's weekday/weekend window ────
        # Must happen before the ScheduleRunner below starts (or at least
        # before its first tick) so the "deep_study_start"/"deep_study_stop"
        # jobs seeded into schedule.json (system.schedule.ensure_deep_study_window_jobs)
        # have a registered handler to call into — otherwise they log
        # "unregistered handler" and silently never fire. Needs AikoThink's
        # LLM client/model, so it can only happen here, after think boots.
        if think_ref is not None:
            from memory import learn
            learn.register_deep_study_handlers(
                client=think_ref._client,
                model=think_ref._llm_model,
            )
        else:
            log.error("AikoThink failed to boot — deep-study window handlers not registered.")

        if memorize is None:
            log.error("Memory boot failed — ScheduleRunner starting without system jobs.")

        # NOTE: this is the ONE ScheduleRunner for the whole app. AikoThink
        # used to also construct its own ScheduleRunner in __init__, which
        # meant two independent daemon threads were both reading and firing
        # the same schedule.json — every due job (reminders, weekly_social,
        # and now the deep_study_start/stop window jobs) would fire twice.
        # That duplicate construction has been removed from cognition/think.py;
        # this is now the only instance, and it's the one registered via
        # register_scheduler() so tools can notify it of newly added jobs.
        scheduler = ScheduleRunner(
            on_due=think_ref.handle_scheduled_job if think_ref else None,
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
            except Exception as exc:
                log.warning("[wakeup] Workspace knowledge scan schedule failed: %s", exc)

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
        except Exception as exc:
            log.warning("[wakeup] Social handler registration failed: %s", exc)

        # ── voice subsystems ──────────────────────────────────────────────────

        # TTS
        on_loading('speak_miotts')
        speak.warmup()
        on_done('speak_miotts')
        on_loading('speak_ready')
        on_done('speak_ready')

        # ASR — staged so each step reports independently
        listen = AikoListen()

        on_loading('listen_asr')
        listen.load_asr()
        on_done('listen_asr')

        on_loading('listen_silero')
        listen.load_vad()              # also kicks off warmup thread
        on_done('listen_silero')

        on_loading('listen_warmup')
        listen.join_warmup()
        on_done('listen_warmup')

        on_loading('listen_ready')
        listen.start_barge_in_monitor()   # VAD daemon — costs ~0 CPU at idle
        on_done('listen_ready')

        return BootResult(
            think    = think_ref,
            memorize = memorize,
            speak    = speak,
            listen   = listen,
        )
