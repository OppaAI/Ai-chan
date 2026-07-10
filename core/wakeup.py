"""
core/wakeup.py

Aiko's boot orchestrator — owns parallel subsystem startup and warmup sequencing.
main.py calls AikoWakeup().boot(...) and receives a BootResult with all live
subsystem references; it never needs to know the startup choreography.

Progress is reported through three injected callbacks so wakeup.py stays
completely TUI-ignorant:
    on_loading(key)  — subsystem is starting
    on_done(key)     — subsystem finished successfully
    on_skip(key)     — subsystem skipped (e.g. text mode)

Each module owns its BOOT_LABELS dict; wakeup collects them and exposes
ALL_BOOT_LABELS so the TUI can register display text before boot begins.

Usage:
    tui.register_boot_labels(AikoWakeup.ALL_BOOT_LABELS)

    result = AikoWakeup(text_mode=False).boot(
        on_loading = tui.step_loading,
        on_done    = tui.step_done,
        on_skip    = tui.step_skip,
    )
    think    = result.think
    memorize = result.memorize
    speak    = result.speak
    listen   = result.listen
"""

import threading
from dataclasses import dataclass
from typing import Callable

# Must run before the core.* imports below: those modules may read secrets
# from os.environ at their own module level, and load_config() is what
# decrypts .env.age into os.environ. load_config() is idempotent (guarded
# by _LOADED), so this is a no-op if main.py already called it first —
# this is just a safety net for any other entrypoint that imports this
# module directly.
from core.config import load_config
load_config()

from core.log import get_logger
log = get_logger(__name__)

from core.think    import BOOT_LABELS as _THINK_LABELS
from core.memorize import BOOT_LABELS as _MEM_LABELS
from core.speak    import BOOT_LABELS as _SPEAK_LABELS
from core.listen   import BOOT_LABELS as _LISTEN_LABELS

# ── result container ──────────────────────────────────────────────────────────

@dataclass
class BootResult:
    """Holds all live subsystem references produced during boot."""
    think:    object          # AikoThink
    memorize: object          # AikoMemorize
    speak:    object          # AikoSpeak
    listen:   object          # AikoListen


# ── helpers ───────────────────────────────────────────────────────────────────

def _prewarm_semantic_cache(think) -> None:
    """Embed route and search exemplars at boot so first-turn latency is cold-free."""
    from core.think import (
        _ROUTE_TERNARY_EXAMPLES,
        _ROUTE_INSTRUCT_BINARY,
        _ROUTE_INSTRUCT_SEARCH,
        _ROUTE_INSTRUCT_TOOL,
        _ROUTE_TOOL_EXAMPLES,
    )
    try:
        think._semantic_example_vectors(_ROUTE_TERNARY_EXAMPLES, _ROUTE_INSTRUCT_BINARY)
        think._semantic_example_vectors(_ROUTE_TOOL_EXAMPLES, _ROUTE_INSTRUCT_TOOL)
        log.info("[wakeup] Semantic exemplar cache warmed")
    except Exception as e:
        log.warning("[wakeup] Semantic exemplar prewarm failed: %s", e)


# ── wakeup ────────────────────────────────────────────────────────────────────

class AikoWakeup:
    """
    Parallel boot orchestrator for all Aiko cognitive subsystems.

    Boots AikoThink and AikoMemorize concurrently, then stages TTS and ASR
    init sequentially with granular progress reporting per step.
    Each subsystem owns its BOOT_LABELS; ALL_BOOT_LABELS merges them all
    so the TUI can register display text before boot begins.

    Args:
        text_mode: Legacy flag. The CLI now keeps voice subsystems loadable so /voice and /listen can toggle them at runtime.
    """

    ALL_BOOT_LABELS: dict[str, str] = {
        **_THINK_LABELS,
        **_MEM_LABELS,
        **_SPEAK_LABELS,
        **_LISTEN_LABELS,
    }

    def __init__(self, text_mode: bool = False) -> None:
        self._text_mode = text_mode

    def boot(
        self,
        on_loading: Callable[[str], None],
        on_done:    Callable[[str], None],
        on_skip:    Callable[[str], None],
    ) -> BootResult:
        """
        Execute full boot sequence and return live subsystem references.

        Parallel phase: AikoThink + AikoMemorize boot concurrently.
        Sequential phase: TTS warmup → ASR staged init.
        Barge-in monitor started as the final ASR step so Silero is already
        warm and the VAD thread costs nothing before the first turn.

        Args:
            on_loading: Called with a progress key when a subsystem starts.
            on_done:    Called with a progress key when a subsystem finishes.
            on_skip:    Called with a progress key when a subsystem is skipped.

        Returns:
            BootResult with think, memorize, speak, listen references.
        """
        from core.log import get_logger, silent_stderr
        from core.memorize import AikoMemorize

        with silent_stderr():
            from core.speak import AikoSpeak
            from core.think import AikoThink

        speak     = AikoSpeak(silent=True)
        memorize  = [None]
        think_ref = [None]
        mem_ready = threading.Event()

        # ── parallel boot ─────────────────────────────────────────────────────

        def init_think():
            on_loading('think_start')
            think_ref[0] = AikoThink(None, speak=speak)
            on_done('think_start')
            on_loading('think_warmup')
            think_ref[0].join_warmup()
            on_done('think_warmup')
            mem_ready.wait()                        # hold until memorize is ready
            think_ref[0]._memorize = memorize[0]    # inject memory backend
            _prewarm_semantic_cache(think_ref[0])   # embed exemplars while booting

        def init_memorize():
            try:
                on_loading('mem_sqlite_vec')
                memorize[0] = AikoMemorize(silent=True)
                on_done('mem_sqlite_vec')
                on_loading('mem_embed')
                on_done('mem_embed')
                on_loading('mem_cleanup')
                memorize[0].cleanup()
                on_done('mem_cleanup')
                on_loading('mem_ready')
                on_done('mem_ready')
            except Exception:
                log.exception("Memory boot failed — Aiko will run without persistent memory.")
            finally:
                mem_ready.set()

        t1 = threading.Thread(target=init_think,    daemon=True)
        t2 = threading.Thread(target=init_memorize, daemon=True)
        t1.start(); t2.start()
        t1.join();  t2.join()

        # ── wire deep_studying into the scheduler's weekday/weekend window ────
        # Must happen before the ScheduleRunner below starts (or at least
        # before its first tick) so the "deep_study_start"/"deep_study_stop"
        # jobs seeded into schedule.json (core.schedule.ensure_deep_study_window_jobs)
        # have a registered handler to call into — otherwise they log
        # "unregistered handler" and silently never fire. Needs AikoThink's
        # LLM client/model, so it can only happen here, after think boots.
        if think_ref[0] is not None:
            from core import learn
            learn.register_deep_study_handlers(
                client=think_ref[0]._client,
                model=think_ref[0]._llm_model,
            )
        else:
            log.error("AikoThink failed to boot — deep-study window handlers not registered.")

        from core.schedule import ScheduleRunner, register_scheduler, register_system_handler, ensure_workspace_knowledge_job
        from core.reflect import generate_and_post
        from core.consolidate import maybe_run_consolidation

        if memorize[0] is None:
            log.error("Memory boot failed — ScheduleRunner starting without system jobs.")

        # NOTE: this is the ONE ScheduleRunner for the whole app. AikoThink
        # used to also construct its own ScheduleRunner in __init__, which
        # meant two independent daemon threads were both reading and firing
        # the same schedule.json — every due job (reminders, weekly_social,
        # and now the deep_study_start/stop window jobs) would fire twice.
        # That duplicate construction has been removed from core/think.py;
        # this is now the only instance, and it's the one registered via
        # register_scheduler() so tools can notify it of newly added jobs.
        _scheduler = ScheduleRunner(
            on_due=think_ref[0].handle_scheduled_job if think_ref[0] else None,
            memorize=memorize[0],
            generate_and_post_fn=generate_and_post,
            consolidate_fn=maybe_run_consolidation,
        )
        register_scheduler(_scheduler)  # Allow tools to notify scheduler of new jobs
        _scheduler.start()

        # Schedule-driven workspace/knowledge scan. The schedule runner keeps
        # using one sleep-until-next-event loop; the KB scan is represented in
        # schedule.json as a normal interval handler job.
        if memorize[0] is not None:
            try:
                from core.knowledge import ingest_workspace_knowledge_folder

                register_system_handler(
                    "workspace_knowledge_scan",
                    lambda _memorize: ingest_workspace_knowledge_folder(embedder=memorize[0]._mem._embedder),
                )
                ensure_workspace_knowledge_job()
                _scheduler.notify_new_job()
                log.info("[wakeup] Workspace knowledge scan schedule ensured")
            except Exception as exc:
                log.warning("[wakeup] Workspace knowledge scan schedule failed: %s", exc)

        # ── voice subsystems ──────────────────────────────────────────────────

        # TTS
        on_loading('speak_miotts')
        speak.warmup()
        on_done('speak_miotts')
        on_loading('speak_ready')
        on_done('speak_ready')

        # ASR — staged so each step reports independently
        from core.listen import AikoListen
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
            think    = think_ref[0],
            memorize = memorize[0],
            speak    = speak,
            listen   = listen,
        )
