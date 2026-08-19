#!/usr/bin/env python3
"""Restore cognition/think.py from origin/dev and apply pre-route gate edits."""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THINK = ROOT / "cognition" / "think.py"


def _excerpt(text: str, needle: str, radius: int = 3) -> str:
    """Return a small context window around needle, or a head/tail sample."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if needle in line:
            start = max(0, i - radius)
            end = min(len(lines), i + radius + 1)
            numbered = [f"{j + 1:>5}| {lines[j]}" for j in range(start, end)]
            return "\n".join(numbered)
    head = "\n".join(f"{j + 1:>5}| {lines[j]}" for j in range(min(8, len(lines))))
    return f"(needle {needle!r} not found)\n{head}"


def main() -> int:
    r = subprocess.run(
        ["git", "show", "origin/dev:cognition/think.py"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if r.returncode != 0:
        r = subprocess.run(
            ["git", "show", "dev:cognition/think.py"],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
    if r.returncode != 0 or not r.stdout.strip():
        print(
            "Could not load cognition/think.py from origin/dev or dev: "
            f"{(r.stderr or '').strip()}",
            file=sys.stderr,
        )
        return 1
    text = r.stdout

    old_route_mid = '''        except Exception as exc:
            log.debug("[route] approval resume pre-check skipped: %s", exc)

        try:
            intent, route_vec = self._route_intent(user_input)
            log.info("[route] intent=%s", intent)
'''
    new_route_mid = '''        except Exception as exc:
            log.debug("[route] approval resume pre-check skipped: %s", exc)

        # Self-assessment *before* quaternary routing so localchat/webchat
        # also get executable soft outcomes (defer / clarify / degrade_chat).
        try:
            from cognition.memory.edge_state import for_identity
            state = for_identity(user_id)
            ok, reason, action = state.should_attempt(user_input, mode="route")
            if not ok:
                log.info("[route] should_attempt action=%s reason=%s", action, reason)
                return self._soft_gate_reply(
                    user_input, action, reason, token_callback=token_callback,
                )
        except Exception as exc:
            log.debug("[route] should_attempt skipped: %s", exc)

        try:
            intent, route_vec = self._route_intent(user_input)
            log.info("[route] intent=%s", intent)
'''
    if old_route_mid not in text:
        print("route insert point not found", file=sys.stderr)
        print(_excerpt(text, "_route_intent"), file=sys.stderr)
        return 1
    text = text.replace(old_route_mid, new_route_mid, 1)

    old_agentic_start = '''    def agentic_chat(self, user_input: str, token_callback=None, mem_kb_future=None, query_vec: np.ndarray | None = None) -> str:
        """Delegate task-mode execution to agentic.agentic.

        Runs a bounded self-assessment gate first (edge_state.should_attempt).
        Critical requests always proceed; discretionary work may degrade to
        chat, defer, or ask for clarification instead of starting the tool loop.
        """
'''
    helper = '''    def _soft_gate_reply(
        self,
        user_input: str,
        action: str,
        reason: str,
        token_callback=None,
        mem_kb_future=None,
        query_vec: np.ndarray | None = None,
    ) -> str:
        """Handle defer / clarify / degrade_chat without starting agentic tools.

        Used by route() (pre-routing) and agentic_chat() (direct agentic entry).
        """
        try:
            from cognition.memory.edge_state import for_identity
            state = for_identity(current_user_id())
            kind = action if action in {"defer", "clarify"} else "stance"
            state.record_self_decision(kind, reason)
            state.persist()
        except Exception:
            pass
        if action == "defer":
            defer_prompt = (
                f"{user_input}\\n\\n"
                "[Internal note — do not mention system details. "
                "You are running low and this is not urgent. "
                "In one short in-character line, say you want to pick this up later "
                "and invite the user to continue when ready.]"
            )
            return self.chat(
                defer_prompt,
                token_callback=token_callback,
                _skip_search=True,
                mem_kb_future=mem_kb_future,
                query_vec=query_vec,
                store_turn=True,
            )
        if action == "clarify":
            clarify_prompt = (
                f"{user_input}\\n\\n"
                "[Internal note — do not mention system details. "
                "You are uncertain what they need. "
                "Ask one concrete clarifying question; do not start a multi-step task.]"
            )
            return self.chat(
                clarify_prompt,
                token_callback=token_callback,
                _skip_search=True,
                mem_kb_future=mem_kb_future,
                query_vec=query_vec,
                store_turn=True,
            )
        return self.chat(
            user_input,
            token_callback=token_callback,
            _skip_search=True,
            mem_kb_future=mem_kb_future,
            query_vec=query_vec,
        )

    def agentic_chat(self, user_input: str, token_callback=None, mem_kb_future=None, query_vec: np.ndarray | None = None) -> str:
        """Delegate task-mode execution to agentic.agentic.

        Runs a bounded self-assessment gate first (edge_state.should_attempt).
        Critical requests always proceed; discretionary work may degrade to
        chat, defer, or ask for clarification instead of starting the tool loop.
        Direct entry (scheduled jobs) still gates here; normal turns are gated
        earlier in route() with mode=route.
        """
'''
    if old_agentic_start not in text:
        print("agentic_chat header not found", file=sys.stderr)
        print(_excerpt(text, "def agentic_chat"), file=sys.stderr)
        return 1
    text = text.replace(old_agentic_start, helper, 1)

    old_gate = '''            # Self-assessment before committing to the agentic tool loop.
            try:
                from cognition.memory.edge_state import for_identity
                state = for_identity(user_id)
                ok, reason, action = state.should_attempt(user_input, mode="agentic")
                if not ok:
                    log.info("[agentic_chat] should_attempt action=%s reason=%s", action, reason)
                    try:
                        state.record_self_decision(action if action in {"defer", "clarify"} else "stance", reason)
                        state.persist()
                    except Exception:
                        pass
                    if action == "defer":
                        defer_prompt = (
                            f"{user_input}\\n\\n"
                            "[Internal note — do not mention system details. "
                            "You are running low and this is not urgent. "
                            "In one short in-character line, say you want to pick this up later "
                            "and invite the user to continue when ready.]"
                        )
                        return self.chat(
                            defer_prompt,
                            token_callback=token_callback,
                            _skip_search=True,
                            mem_kb_future=mem_kb_future,
                            query_vec=query_vec,
                            store_turn=True,
                        )
                    if action == "clarify":
                        clarify_prompt = (
                            f"{user_input}\\n\\n"
                            "[Internal note — do not mention system details. "
                            "You are uncertain what they need. "
                            "Ask one concrete clarifying question; do not start a multi-step task.]"
                        )
                        return self.chat(
                            clarify_prompt,
                            token_callback=token_callback,
                            _skip_search=True,
                            mem_kb_future=mem_kb_future,
                            query_vec=query_vec,
                            store_turn=True,
                        )
                    # degrade_chat (default soft path)
                    return self.chat(
                        user_input,
                        token_callback=token_callback,
                        _skip_search=True,
                        mem_kb_future=mem_kb_future,
                        query_vec=query_vec,
                    )
            except Exception as exc:
                log.debug("[agentic_chat] should_attempt skipped: %s", exc)
'''
    new_gate = '''            # Self-assessment before committing to the agentic tool loop
            # (covers scheduled/direct agentic entry; normal turns already gated in route).
            try:
                from cognition.memory.edge_state import for_identity
                state = for_identity(user_id)
                ok, reason, action = state.should_attempt(user_input, mode="agentic")
                if not ok:
                    log.info("[agentic_chat] should_attempt action=%s reason=%s", action, reason)
                    return self._soft_gate_reply(
                        user_input,
                        action,
                        reason,
                        token_callback=token_callback,
                        mem_kb_future=mem_kb_future,
                        query_vec=query_vec,
                    )
            except Exception as exc:
                log.debug("[agentic_chat] should_attempt skipped: %s", exc)
'''
    if old_gate not in text:
        print("agentic gate body not found", file=sys.stderr)
        print(_excerpt(text, "should_attempt"), file=sys.stderr)
        return 1
    text = text.replace(old_gate, new_gate, 1)

    THINK.write_text(text, encoding="utf-8")
    print(f"Wrote {THINK} ({len(text)} chars)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
