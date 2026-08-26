"""
system/prepare.py

Post-auth initialization — everything that needs a REAL logged-in user and
must never run as guest. wakeup.boot() stays pre-auth safe (guest DB, no
user-space writes); the front end calls run_post_auth() once per uid at the
first authenticated connection, on a background thread so login isn't blocked.

Moved here from interface/webui/webui.py's _on_user_active() so CLI/adapters
can reuse the same post-auth sequence later.

Flow:

    wakeup.boot() ──▶ server up ──▶ first authenticated connect
                                        │
                                        ▼  (background thread)
                              prepare.run_post_auth(uid)
                                        │
                    ┌───────────────────┼────────────────────┐
                    ▼                   ▼                    ▼
          memorize.cleanup()   ensure_playbooks(uid)   bootstrap_non_system_jobs()
          (real store now      (seed user playbook     (Threads/Bluesky/etc.
           open post-login)    definitions)            schedule jobs + watchers)

Idempotency: callers keep their own "already ran" guard (webui's
_user_space_ready set) — this module stays stateless per process.
"""

from __future__ import annotations                            # evaluates type annotations later

from system.log import get_logger                             # assign logging to universal logger
log = get_logger(__name__)


def run_post_auth(uid: str, *, memorize=None, think=None) -> None:
    """Run one-time post-login initialization for a real (non-guest) user.

    Args:
        uid: authenticated user id — guest is rejected (pre-auth contract).
        memorize: live AikoMemorize (cleanup runs on its real per-user store).
        think: live AikoThink (passed through to job bootstrap).
    """
    if not uid or uid == "guest":                                                       # enforce the pre-auth contract,
        log.debug("[prepare] skipping post-auth init for %r", uid or "empty uid")       # nothing to initialize yet
        return

    # ``run_post_auth`` runs in a background thread after the WebSocket login
    # handler returns. ContextVars do not cross that thread boundary, and the
    # live memory facade was constructed during guest-safe boot. Bind both
    # explicitly before *any* user-space operation so cleanup, persona reads,
    # schedules, and cache warmup use USER_SPACE_ROOT/<uid>, not guest.
    from system.userspace import (
        reset_current_display_name,
        reset_current_user_id,
        set_current_display_name,
        set_current_user_id,
    )

    user_token = set_current_user_id(uid)
    display_token = set_current_display_name(uid)
    try:
        if memorize is not None:
            memorize.switch_user(uid)

        # Semantic cache warm under the REAL identity — moved out of wakeup boot:
        # as guest, per-user npz disk caches can't be read or written (see
        # reason.cache_vector_path's guest guard), so every boot recomputed from
        # scratch. Here it loads the user's existing cache files and persists
        # fresh vectors for the next session.
        if think is not None:
            try:
                think.prewarm_caches()
            except Exception:
                log.exception("[prepare] semantic cache prewarm failed")

        # Memory cleanup — boot skipped it for guest; the real per-user store is
        # open now (parity with the old login-gated boot behaviour).
        try:
            if memorize is not None:
                memorize.cleanup()
        except Exception:
            log.exception("[prepare] post-login memory cleanup failed")

        # Static anchor for Grasp relevance scoring — persona + pinned memories
        # become the identity tokens that keep relevant turns resident longer in
        # working memory (activates the previously-dead relevance factor).
        if memorize is not None:
            try:
                texts: list[str] = []
                try:
                    texts.append(memorize.persona_context() or "")
                except Exception:
                    pass
                try:
                    for m in memorize.get_all():
                        if m.get("pinned"):
                            texts.append(m.get("memory") or m.get("text") or "")
                except Exception:
                    pass
                from cognition.memory.grasp import build_anchor_tokens, set_static_anchor_tokens
                anchor = build_anchor_tokens(*texts)
                if anchor:
                    set_static_anchor_tokens(anchor)
                    log.info("[prepare] Grasp static anchor set (%d tokens)", len(anchor))
                else:
                    log.info("[prepare] Grasp static anchor empty — no persona/pinned text yet")
            except Exception:
                log.exception("[prepare] failed to set Grasp static anchor")

        # Playbook seeding — user-scoped playbook definitions under USER_SPACE_ROOT/<uid>/.
        try:
            from agentic.graph_engine import ensure_playbooks
            ensure_playbooks(user_id=uid)
        except Exception:
            log.exception("[prepare] playbook seeding failed")

        # Social schedule jobs (Threads/Bluesky/X/…) + knowledge-folder watcher —
        # seeded per user only after we know who they are.
        try:
            from system.schedule import bootstrap_non_system_jobs
            bootstrap_non_system_jobs(think=think, memorize=memorize)
        except Exception:
            log.exception("[prepare] schedule job bootstrap failed")

        log.info("[prepare] post-auth initialization complete for %s", uid)
    finally:
        reset_current_display_name(display_token)
        reset_current_user_id(user_token)
