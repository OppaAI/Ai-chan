#!/usr/bin/env python3
"""
Offline semantic + LLM route tracer for think.py cascade.
Mode is driven by ROUTE_MODE in .env (same key think.py reads):
  ROUTE_MODE=semantic      → semantic stages only  (default)
  ROUTE_MODE=llm           → LLM stages only
  ROUTE_MODE=semantic_only → semantic stages only  (no LLM fallback)
  ROUTE_MODE=chat          → routing disabled; tracer notes this and exits

Usage:
  python util/test_route.py [prompt]   # single prompt
  python util/test_route.py            # interactive mode
"""

import os, sys, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.WARNING)

# load .env so ROUTE_MODE (and LLM_BASE_URL etc.) are available before imports
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv optional; env vars may already be set

# ── minimal stubs so we can import without a running stack ────────────────────

from unittest.mock import MagicMock, patch

sys.modules.setdefault("pygame", MagicMock())
sys.modules.setdefault("openai", MagicMock())

from core.memorize import AikoMemorize
from core.think import (
    AikoThink,
    LLM_BASE_URL,
    LLM_MODEL,
    ROUTER_MODEL,
    LLM_TIMEOUT,
    _ROUTE_BINARY_EXAMPLES,
    _ROUTE_TOOL_EXAMPLES,
    _ROUTE_SEARCH_EXAMPLES,
    _SEMANTIC_ROUTE_THRESHOLD,
    _SEMANTIC_SEARCH_THRESHOLD,
    _SEMANTIC_ROUTE_MIN_GAP,
    _SEMANTIC_TOOL_MIN_GAP,
    _SEMANTIC_SEARCH_MIN_GAP,
    _SEMANTIC_LABEL_TOP_K,
)

# ── colour helpers ────────────────────────────────────────────────────────────

CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
MAGENTA= "\033[95m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

def score_bar(score: float, width: int = 20) -> str:
    filled = max(0, min(width, int(round(score * width))))
    color  = GREEN if score >= 0.5 else YELLOW if score >= 0.35 else RED
    return f"{color}{'█' * filled}{'░' * (width - filled)}{RESET} {score:.3f}"

# ── init embedder only (no LLM, no TTS, no schedule) ─────────────────────────

memorize = AikoMemorize()

think = object.__new__(AikoThink)
think._memorize = memorize
think._speak    = None
think._semantic_example_cache      = {}
think._semantic_example_cache_lock = __import__("threading").RLock()
think._pending_search_query        = None
think._route_chat_classified       = None
think._history                     = []
think._history_lock                = __import__("threading").RLock()
think._reasoning                   = False

# ── read ROUTE_MODE (mirrors think.py logic) ─────────────────────────────────

_ROUTE_MODE = os.getenv("ROUTE_MODE", "semantic").strip().lower()
_RUN_SEMANTIC = _ROUTE_MODE not in {"llm", "0", "off", "false", "chat", "disabled"}
# In production, ROUTE_MODE=semantic still falls back to LLM when scores are
# ambiguous, so we always show the LLM path when mode=semantic.
# ROUTE_MODE=semantic_only suppresses the LLM fallback display.
_SHOW_LLM_FALLBACK = _ROUTE_MODE not in {"semantic_only", "chat", "0", "off", "false", "disabled"}

# ── lazy real OpenAI client (fired only when LLM stages are relevant) ─────────

_llm_client = None

def _get_llm_client():
    """Return a real OpenAI-compat client, importing the real openai package."""
    global _llm_client
    if _llm_client is not None:
        return _llm_client
    # pop the MagicMock stub so we get the real package
    sys.modules.pop("openai", None)
    try:
        from openai import OpenAI
        _llm_client = OpenAI(base_url=LLM_BASE_URL, api_key="not-needed")
        return _llm_client
    except Exception as e:
        print(f"\n  {RED}Could not create OpenAI client: {e}{RESET}")
        return None

# ── semantic tracer ───────────────────────────────────────────────────────────

def trace_stage(
    label: str,
    examples_by_label: dict,
    user_input: str,
    threshold: float,
    min_gap: float,
) -> tuple[str, float]:
    scores = think._semantic_all_scores(user_input, examples_by_label)
    if not scores:
        return "chat", 0.0

    best_label  = max(scores, key=scores.get)
    best_score  = scores[best_label]
    sorted_vals = sorted(scores.values(), reverse=True)
    gap         = sorted_vals[0] - sorted_vals[1] if len(sorted_vals) > 1 else 1.0

    print(f"\n  {BOLD}{label}{RESET}")
    print(f"  {'label':<16} {'mean score':<8}  bar")
    print(f"  {'-'*52}")
    for lbl, sc in sorted(scores.items(), key=lambda x: -x[1]):
        marker = f" {BOLD}← best{RESET}" if lbl == best_label else ""
        print(f"  {lbl:<16} {score_bar(sc)}{marker}")

    pass_thresh = best_score >= threshold
    pass_gap    = gap >= min_gap
    verdict     = GREEN + "PASS" + RESET if (pass_thresh and pass_gap) else RED + "FAIL" + RESET
    print(f"\n  threshold={threshold}  gap_min={min_gap}  top_k={_SEMANTIC_LABEL_TOP_K}")
    print(f"  best={best_label}  score={best_score:.3f}  gap={gap:.3f}  [{verdict}]")

    return best_label, best_score

# ── LLM tracer — delegates to think's real methods, no prompt duplication ─────

def _ensure_llm_client() -> bool:
    """Patch a real OpenAI client onto think. Returns False if unavailable."""
    if getattr(think, "_client", None) is not None and not isinstance(think._client, MagicMock):
        return True
    client = _get_llm_client()
    if client is None:
        return False
    think._client       = client
    think._router_model = ROUTER_MODEL
    think._llm_model    = LLM_MODEL
    return True

def trace_llm_router(user_input: str) -> str | None:
    """Call think._classify_agent_intent — the exact production code path."""
    if not _ensure_llm_client():
        return None

    print(f"\n{BOLD}▶ LLM classifier  (ROUTER_MODEL={ROUTER_MODEL}){RESET}")
    import time
    t0 = time.monotonic()
    try:
        result  = think._classify_agent_intent(user_input)
        elapsed = time.monotonic() - t0
        color   = GREEN if result != "chat" else CYAN
        print(f"\n  parsed     : {color}{result}{RESET}")
        print(f"  latency    : {elapsed*1000:.0f} ms")
        return result
    except Exception as e:
        print(f"\n  {RED}LLM call failed: {e}{RESET}")
        return None

def trace_llm_search_classify(user_input: str) -> tuple[bool, str] | None:
    """Call think._classify_and_resolve — the exact production code path."""
    if not _ensure_llm_client():
        return None

    think._history = []

    print(f"\n{BOLD}▶ LLM search classifier  (_classify_and_resolve){RESET}")
    print(f"  {DIM}client={type(think._client).__name__}  model={think._router_model!r}{RESET}")
    import time
    t0 = time.monotonic()
    try:
        # call the LLM directly here instead of delegating to _classify_and_resolve
        # so we can see exactly what's happening without the silent except swallowing it
        resp = think._client.chat.completions.create(
            model=think._router_model,
            messages=[{"role": "user", "content": (
                f"Message: {user_input!r}\n\n"
                "Output only one of these two formats, nothing else:\n"
                "data|<3-5 word search query>\n"
                "social|none\n\n"
                "Message: 'what's the weather in Vancouver'\n"
                "Output: data|current weather Vancouver\n\n"
                "Message: 'who won the NHL game last night'\n"
                "Output: data|NHL game results last night\n\n"
                "Message: 'debug why asyncio.run() hangs'\n"
                "Output: social|none\n\n"
                "Message: 'explain how attention works'\n"
                "Output: social|none\n\n"
                "Message: 'do you think embeddings are good for memory'\n"
                "Output: social|none\n\n"
                "Output:"
            )}],
            stream=False, max_tokens=20, temperature=0.0,
            top_p=1.0, top_k=1, timeout=LLM_TIMEOUT,
        )
        elapsed = time.monotonic() - t0
        raw = resp.choices[0].message.content if resp.choices else None
        print(f"  {DIM}raw response: {raw!r}{RESET}")

        if not raw:
            print(f"  {RED}empty response from model{RESET}")
            print(f"  latency    : {elapsed*1000:.0f} ms")
            return None

        label, _, rest = raw.strip().partition("|")
        is_data = "data" in label.strip().lower()
        resolved = rest.strip().split('\n')[0].strip('*_`()').strip()[:100]
        import re
        resolved = re.sub(r'\{[^}]*\}', '', resolved).strip()
        if not resolved or resolved.lower() in ("none", "<search query>", "<3-5 word search query>"):
            resolved = user_input

        color = GREEN if is_data else CYAN
        print(f"\n  decision   : {color}{'data' if is_data else 'social'}{RESET}")
        if is_data:
            print(f"  query      : {CYAN}{resolved!r}{RESET}")
        print(f"  latency    : {elapsed*1000:.0f} ms")
        return is_data, resolved

    except Exception as e:
        elapsed = time.monotonic() - t0
        print(f"\n  {RED}LLM call failed ({elapsed*1000:.0f} ms): {e!r}{RESET}")
        return None

# ── full trace ────────────────────────────────────────────────────────────────

def trace(prompt: str) -> None:
    print(f"\n{BOLD}{'═'*60}{RESET}")
    print(f"{BOLD}PROMPT:{RESET} {CYAN}{prompt!r}{RESET}")
    print(f"{'═'*60}")

    is_agentic = False

    if _RUN_SEMANTIC:
        # ── Stage 1: semantic binary ──────────────────────────────────────────
        print(f"\n{BOLD}▶ Stage 1  chat vs agentic  (semantic){RESET}")
        best1, score1 = trace_stage(
            "binary classifier",
            _ROUTE_BINARY_EXAMPLES,
            prompt,
            _SEMANTIC_ROUTE_THRESHOLD,
            _SEMANTIC_ROUTE_MIN_GAP,
        )
        scores1     = think._semantic_all_scores(prompt, _ROUTE_BINARY_EXAMPLES)
        sorted1     = sorted(scores1.values(), reverse=True)
        gap1        = sorted1[0] - sorted1[1] if len(sorted1) > 1 else 1.0
        is_agentic  = (
            best1 == "agentic"
            and score1 >= _SEMANTIC_ROUTE_THRESHOLD
            and gap1   >= _SEMANTIC_ROUTE_MIN_GAP
        )

        if is_agentic:
            print(f"\n  {GREEN}→ AGENTIC{RESET}")

            # ── Stage 2a: tool classifier ─────────────────────────────────────
            print(f"\n{BOLD}▶ Stage 2a  which tool  (semantic){RESET}")
            best2, score2 = trace_stage(
                "tool classifier",
                _ROUTE_TOOL_EXAMPLES,
                prompt,
                _SEMANTIC_ROUTE_THRESHOLD,
                _SEMANTIC_TOOL_MIN_GAP,
            )
            scores2 = think._semantic_all_scores(prompt, _ROUTE_TOOL_EXAMPLES)
            sorted2 = sorted(scores2.values(), reverse=True)
            gap2    = sorted2[0] - sorted2[1] if len(sorted2) > 1 else 1.0

            if score2 >= _SEMANTIC_ROUTE_THRESHOLD and gap2 >= _SEMANTIC_TOOL_MIN_GAP:
                print(f"\n  {GREEN}→ ROUTE: agentic_chat  tool={best2}{RESET}")
                if _SHOW_LLM_FALLBACK:
                    print(f"\n  {DIM}(LLM fallback skipped — semantic confident){RESET}")
            else:
                print(f"\n  {YELLOW}→ semantic weak — LLM fallback fires{RESET}")
                if _SHOW_LLM_FALLBACK:
                    llm_label = trace_llm_router(prompt)
                    if llm_label:
                        color = GREEN if llm_label != "chat" else CYAN
                        print(f"\n  {color}→ ROUTE: agentic_chat  tool={llm_label}  (via LLM){RESET}")

        else:
            print(f"\n  {CYAN}→ CHAT{RESET}")

            # ── Stage 2b: websearch classifier ───────────────────────────────
            print(f"\n{BOLD}▶ Stage 2b  websearch needed?  (semantic){RESET}")
            best3, score3 = trace_stage(
                "search classifier",
                _ROUTE_SEARCH_EXAMPLES,
                prompt,
                _SEMANTIC_SEARCH_THRESHOLD,
                _SEMANTIC_SEARCH_MIN_GAP,
            )
            scores3      = think._semantic_all_scores(prompt, _ROUTE_SEARCH_EXAMPLES)
            sorted3      = sorted(scores3.values(), reverse=True)
            gap3         = sorted3[0] - sorted3[1] if len(sorted3) > 1 else 1.0
            needs_search = (
                best3 == "data"
                and score3 >= _SEMANTIC_SEARCH_THRESHOLD
                and gap3   >= _SEMANTIC_SEARCH_MIN_GAP
            )

            if needs_search:
                print(f"\n  {GREEN}→ ROUTE: chat()  websearch=True  query={prompt!r}{RESET}")
            elif score1 >= _SEMANTIC_ROUTE_THRESHOLD and gap1 < _SEMANTIC_ROUTE_MIN_GAP:
                print(f"\n  {YELLOW}→ ROUTE: llm_fallback (binary scores too close){RESET}")
                if _SHOW_LLM_FALLBACK:
                    trace_llm_router(prompt)
            else:
                print(f"\n  {CYAN}→ ROUTE: chat()  websearch=False{RESET}")

            # LLM search resolve (always shown for chat turns when LLM fallback is active)
            if _SHOW_LLM_FALLBACK:
                trace_llm_search_classify(prompt)

    # ── LLM-only mode ────────────────────────────────────────────────────────
    if not _RUN_SEMANTIC:
        print(f"\n{BOLD}▶ Skipping semantic stages (ROUTE_MODE={_ROUTE_MODE}){RESET}")
        think._history = []
        trace_llm_router(prompt)
        think._history = []
        trace_llm_search_classify(prompt)

    print()

# ── entrypoint ────────────────────────────────────────────────────────────────

def main():
    if _ROUTE_MODE in {"chat", "0", "off", "false", "disabled"}:
        print(f"{YELLOW}ROUTE_MODE={_ROUTE_MODE} — routing is disabled in .env; nothing to trace.{RESET}")
        sys.exit(0)

    mode_tag = {
        "semantic":      f"{CYAN}semantic (+ LLM fallback){RESET}",
        "semantic_only": f"{CYAN}semantic-only (no LLM fallback){RESET}",
        "llm":           f"{MAGENTA}LLM-only{RESET}",
    }.get(_ROUTE_MODE, f"{CYAN}{_ROUTE_MODE}{RESET}")

    if len(sys.argv) > 1:
        trace(" ".join(sys.argv[1:]))
        return

    print(f"{BOLD}Aiko route tracer{RESET}  {DIM}(Ctrl+C or empty line to quit){RESET}  ROUTE_MODE={mode_tag}")
    while True:
        try:
            prompt = input(f"\n{BOLD}>{RESET} ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break
        if not prompt:
            break
        trace(prompt)

if __name__ == "__main__":
    main()