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
    speak    = result.speak    # None in text mode
    listen   = result.listen   # None in text mode
"""

import threading
from dataclasses import dataclass
from typing import Callable
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
    speak:    object | None   # AikoSpeak  — None in text mode
    listen:   object | None   # AikoListen — None in text mode


# ── helpers ───────────────────────────────────────────────────────────────────

def _prewarm_semantic_cache(think) -> None:
    """Embed route and search exemplars at boot so first-turn latency is cold-free."""
    from core.think import _SEMANTIC_ROUTE_EXAMPLES, _SEMANTIC_SEARCH_EXAMPLES
    try:
        think._semantic_example_vectors(_SEMANTIC_ROUTE_EXAMPLES)
        think._semantic_example_vectors(_SEMANTIC_SEARCH_EXAMPLES)
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
        text_mode: When True, TTS and ASR subsystems are skipped entirely.
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
        Sequential phase: TTS warmup → ASR staged init (or skip both).
        Barge-in monitor started as the final ASR step so Silero is already
        warm and the VAD thread costs nothing before the first turn.

        Args:
            on_loading: Called with a progress key when a subsystem starts.
            on_done:    Called with a progress key when a subsystem finishes.
            on_skip:    Called with a progress key when a subsystem is skipped.

        Returns:
            BootResult with think, memorize, speak, listen references.
        """
        from core.silence  import silent_stderr
        from core.memorize import AikoMemorize

        with silent_stderr():
            from core.speak import AikoSpeak
            from core.think import AikoThink

        speak     = AikoSpeak(silent=True) if not self._text_mode else None
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
            except Exception as e:
                log.error("Memory boot failed: %s", e)
            finally:
                mem_ready.set()  # always unblock init_think, even on failure
            if memorize[0] is None:
                return
            from core.dream import start as start_dream_scheduler
            start_dream_scheduler(memorize[0])

        t1 = threading.Thread(target=init_think,    daemon=True)
        t2 = threading.Thread(target=init_memorize, daemon=True)
        t1.start(); t2.start()
        t1.join();  t2.join()

        # ── voice subsystems ──────────────────────────────────────────────────

        listen = None

        if not self._text_mode:
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
        else:
            on_skip('speak_skip')
            on_skip('listen_skip')

        return BootResult(
            think    = think_ref[0],
            memorize = memorize[0],
            speak    = speak,
            listen   = listen,
        )
