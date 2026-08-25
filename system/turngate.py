"""
system/turngate.py

Single process-wide gate shared by orchestrate.py's interactive turn loop
and schedule.py's background scheduler daemon.

Why this exists:
    Aiko runs one model at a time (single llama-server, single AikoMemorize/
    think singleton) — there is no per-user instance. The interactive main
    loop is already single-threaded, so user-vs-user turns are naturally
    serialized via the shared input queue. The scheduler daemon, however,
    runs on its own thread ("aiko-schedule") and fires jobs — some of which
    call directly into the same memorize/think singletons — completely
    unsynchronized with whatever the main loop is doing.

    AIKO_BUSY_LOCK is the one thing both sides take before touching shared
    state (switch_user, inference, memory writes). It makes explicit what
    the main loop already had implicitly (only one turn in flight), and
    extends that same "wait your turn" guarantee to background scheduled
    jobs: a job due mid-turn waits for the turn to finish, and a new turn
    arriving mid-job waits for the job to finish.

    Use an RLock (not a plain Lock) because a single logical "holder" may
    legitimately re-enter — e.g. the scheduler thread nesting a call that
    re-acquires the gate for a sub-step, or a job runner calling into a
    helper that also acquires it defensively.

Usage:
    from system.turngate import AIKO_BUSY_LOCK

    with AIKO_BUSY_LOCK:
        memorize.switch_user(uid)
        ... run the turn / job ...
"""
from __future__ import annotations

import threading

AIKO_BUSY_LOCK = threading.RLock()
