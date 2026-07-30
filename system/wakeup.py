"""
system/wakeup.py

Aiko's boot orchestrator — owns parallel subsystem startup and warmup sequencing.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any
import threading

from system.config import load_config
load_config()

from system.log import get_logger
log = get_logger(__name__)

from cognition.think import BOOT_LABELS as _THINK_LABELS
from memory.memorize import BOOT_LABELS as _MEM_LABELS
from sensory.speak   import BOOT_LABELS as _SPEAK_LABELS
from sensory.listen  import BOOT_LABELS as _LISTEN_LABELS

from memory.memorize import AikoMemorize
from cognition.think import AikoThink
from sensory.speak import AikoSpeak
from sensory.listen import AikoListen
from system.schedule import (
    start_scheduler,
)

@dataclass(slots=True, frozen=True)
class BootResult:
    think:    AikoThink | None
    memorize: AikoMemorize | None
    speak:    AikoSpeak | None
    listen:   AikoListen | None

type BootCallback = Callable[[str], None]

def _prewarm_semantic_cache(think) -> None:
    if think._get_memorize() is None:
        log.info("[wakeup] Skipping semantic cache prewarm — no memory backend.")
        return
    from cognition.think import (
        _ROUTE_TERNARY_EXAMPLES,
        _ROUTE_INSTRUCT_TERNARY,
    )
    try:
        think._semantic_example_vectors(_ROUTE_TERNARY_EXAMPLES, _ROUTE_INSTRUCT_TERNARY)

        from agentic.capability import CAPABILITIES, _get_trigger_embedding
        embedder = think._get_memorize()._mem._embedder
        for cap in CAPABILITIES.values():
            _get_trigger_embedding(cap, embedder)

        log.info("[wakeup] Semantic exemplar cache warmed (intent + capabilities)")
    except Exception:
        log.exception("[wakeup] Semantic exemplar prewarm failed")


class AikoWakeup:
    ALL_BOOT_LABELS: dict[str, str] = {
        **_THINK_LABELS,
        **_MEM_LABELS,
        **_SPEAK_LABELS,
        **_LISTEN_LABELS,
        "mcp_client": "Connect to Social MCP server",
    }

    def boot(
        self,
        on_loading: BootCallback,
        on_done:    BootCallback,
        on_skip:    BootCallback,
    ) -> BootResult:
        mem_ready_evt  = threading.Event()

        def _boot_step(key: str, fn: Callable[[], Any] | None = None) -> Any:
            on_loading(key)
            if fn is None:
                on_done(key)
                return None
            try:
                result = fn()
            except Exception:
                on_skip(key)
                raise
            on_done(key)
            return result

        def init_think(memorize_getter):
            think = _boot_step('think_start', lambda: AikoThink())
            _boot_step('think_warmup', lambda: (think.start_warmup(), think.join_warmup()))
            _boot_step('think_mem_wait', lambda: mem_ready_evt.wait())
            _boot_step('think_inject', lambda: (think.set_memorize(memorize_getter()), think.start_idle_learner()))
            _boot_step('think_prewarm', lambda: _prewarm_semantic_cache(think))
            return think

        def init_memorize():
            try:
                memorize = _boot_step('mem_embed', lambda: AikoMemorize(silent=True))

                def _set_display_name():
                    from system.userspace import current_display_name
                    display_name = current_display_name()
                    memorize.set_display_name(display_name)
                    if display_name == memorize.get_user_id():
                        log.warning(
                            "[wakeup] No display name for user_id=%s — memory pins "
                            "will use raw user_id until the user logs in.",
                            display_name,
                        )

                _boot_step('mem_display_name', _set_display_name)
                _boot_step('mem_cleanup', lambda: memorize.cleanup())
                _boot_step('mem_ready')

                return memorize
            except Exception:
                log.exception("[wakeup] Memory boot failed — Aiko will run without persistent memory.")
                return None
            finally:
                mem_ready_evt.set()

        with ThreadPoolExecutor(max_workers=2) as ex:
            mem_future = ex.submit(init_memorize)
            think_future = ex.submit(init_think, lambda: mem_future.result())

            think_ref: AikoThink | None = None
            think_exc: Exception | None = None
            try:
                think_ref = think_future.result()
            except Exception as exc:
                think_exc = exc
            memorize = mem_future.result()

        try:
            from agentic.mcp_client.bridge import bootstrap_mcp
            mcp_ok = _boot_step("mcp_client", lambda: bootstrap_mcp())
            if not mcp_ok:
                log.info("[wakeup] MCP client skipped or unavailable — Aiko will run without social posting tools.")
        except Exception:
            log.exception("[wakeup] MCP client boot failed")
            _boot_step("mcp_client", None)
            on_skip("mcp_client")

        if think_ref is None:
            log.critical(
                "[wakeup] AikoThink boot failed — cannot continue without cognition core.",
                exc_info=think_exc,
            )
            raise RuntimeError("AikoThink boot failed") from think_exc

        try:
            speak = AikoSpeak(silent=True)
        except Exception:
            log.exception("[wakeup] AikoSpeak construction failed — Aiko will run without voice output.")
            speak = None

        start_scheduler(
            on_due=think_ref.handle_scheduled_job,
            memorize=memorize,
            think=think_ref,
        )

        if speak is not None:
            try:
                _boot_step('speak_miotts', lambda: speak.warmup())
                _boot_step('speak_ready')
            except Exception:
                log.exception("[wakeup] TTS boot failed — Aiko will run without voice output.")
                speak = None

        think_ref.set_speak(speak)

        listen: AikoListen | None = None
        try:
            listen = AikoListen()
            try:
                from sensory.listen_native import bind_native_gates
                from sensory import listen as listen_mod
                bind_native_gates(listen_mod)
            except Exception:
                log.exception("[wakeup] native voice gates failed — continuing without them")
        except Exception:
            log.exception("[wakeup] AikoListen construction failed — Aiko will run without voice input.")

        if listen is not None:
            try:
                _boot_step('listen_asr', lambda: listen.load_asr())
                _boot_step('listen_silero', lambda: listen.load_vad())
                _boot_step('listen_warmup', lambda: listen.join_warmup())
                _boot_step('listen_ready', lambda: listen.start_barge_in_monitor())
            except Exception:
                log.exception("[wakeup] ASR/VAD boot failed — Aiko will run without voice input.")
                listen = None

        return BootResult(
            think    = think_ref,
            memorize = memorize,
            speak    = speak,
            listen   = listen,
        )
