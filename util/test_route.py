#!/usr/bin/env python3
"""
Offline semantic route tracer for think.py cascade.
Usage: python util/test_route.py [prompt]
       python util/test_route.py  # interactive mode
"""

import os, sys, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.WARNING)

# ── minimal stubs so we can import without a running LLM ─────────────────────

from unittest.mock import MagicMock, patch

# Stub heavy imports before think.py loads them
sys.modules.setdefault("pygame", MagicMock())
sys.modules.setdefault("openai", MagicMock())

from core.memorize import AikoMemorize
from core.think import (
    AikoThink,
    _ROUTE_BINARY_EXAMPLES,
    _ROUTE_TOOL_EXAMPLES,
    _ROUTE_SEARCH_EXAMPLES,
    _SEMANTIC_ROUTE_THRESHOLD,
    _SEMANTIC_SEARCH_THRESHOLD,
    _SEMANTIC_ROUTE_MIN_GAP,
)

# ── init embedder only (no LLM, no TTS, no schedule) ─────────────────────────

memorize = AikoMemorize()

think = object.__new__(AikoThink)
think._memorize = memorize
think._speak = None
think._semantic_example_cache = {}
think._semantic_example_cache_lock = __import__("threading").RLock()
think._pending_search_query = None
think._route_chat_classified = None

# ── tracer ────────────────────────────────────────────────────────────────────

import numpy as np
from collections import defaultdict

CYAN  = "\033[96m"
GREEN = "\033[92m"
YELLOW= "\033[93m"
RED   = "\033[91m"
BOLD  = "\033[1m"
DIM   = "\033[2m"
RESET = "\033[0m"

def score_bar(score: float, width: int = 20) -> str:
    filled = int(round(score * width))
    filled = max(0, min(width, filled))
    color = GREEN if score >= 0.5 else YELLOW if score >= 0.35 else RED
    return f"{color}{'█' * filled}{'░' * (width - filled)}{RESET} {score:.3f}"

def trace_stage(label: str, examples_by_label: dict, user_input: str, threshold: float) -> tuple[str, float]:
    scores = think._semantic_all_scores(user_input, examples_by_label)
    if not scores:
        return "chat", 0.0

    best_label = max(scores, key=scores.get)
    best_score = scores[best_label]
    sorted_vals = sorted(scores.values(), reverse=True)
    gap = sorted_vals[0] - sorted_vals[1] if len(sorted_vals) > 1 else 1.0

    print(f"\n  {BOLD}{label}{RESET}")
    print(f"  {'label':<16} {'mean score':<8}  bar")
    print(f"  {'-'*52}")
    for lbl, sc in sorted(scores.items(), key=lambda x: -x[1]):
        marker = f" {BOLD}← best{RESET}" if lbl == best_label else ""
        print(f"  {lbl:<16} {score_bar(sc)}{marker}")

    pass_thresh = best_score >= threshold
    pass_gap    = gap >= _SEMANTIC_ROUTE_MIN_GAP
    verdict = GREEN + "PASS" + RESET if (pass_thresh and pass_gap) else RED + "FAIL" + RESET
    print(f"\n  threshold={threshold}  gap_min={_SEMANTIC_ROUTE_MIN_GAP}")
    print(f"  best={best_label}  score={best_score:.3f}  gap={gap:.3f}  [{verdict}]")

    return best_label, best_score

def trace(prompt: str) -> None:
    print(f"\n{BOLD}{'═'*60}{RESET}")
    print(f"{BOLD}PROMPT:{RESET} {CYAN}{prompt!r}{RESET}")
    print(f"{'═'*60}")

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    print(f"\n{BOLD}▶ Stage 1  chat vs agentic{RESET}")
    best1, score1 = trace_stage("binary classifier", _ROUTE_BINARY_EXAMPLES, prompt, _SEMANTIC_ROUTE_THRESHOLD)
    scores1 = think._semantic_all_scores(prompt, _ROUTE_BINARY_EXAMPLES)
    sorted1 = sorted(scores1.values(), reverse=True)
    gap1 = sorted1[0] - sorted1[1] if len(sorted1) > 1 else 1.0
    is_agentic = (best1 == "agentic"
                  and score1 >= _SEMANTIC_ROUTE_THRESHOLD
                  and gap1 >= _SEMANTIC_ROUTE_MIN_GAP)

    if is_agentic:
        print(f"\n  {GREEN}→ AGENTIC{RESET}")

        # ── Stage 2a ──────────────────────────────────────────────────────────
        print(f"\n{BOLD}▶ Stage 2a  which tool{RESET}")
        best2, score2 = trace_stage("tool classifier", _ROUTE_TOOL_EXAMPLES, prompt, _SEMANTIC_ROUTE_THRESHOLD)
        if score2 >= _SEMANTIC_ROUTE_THRESHOLD:
            print(f"\n  {GREEN}→ ROUTE: agentic_chat  tool={best2}{RESET}")
        else:
            print(f"\n  {YELLOW}→ ROUTE: agentic_chat  tool=llm_fallback (score too low){RESET}")
    else:
        print(f"\n  {CYAN}→ CHAT{RESET}")

        # ── Stage 2b ──────────────────────────────────────────────────────────
        print(f"\n{BOLD}▶ Stage 2b  websearch needed?{RESET}")
        best3, score3 = trace_stage("search classifier", _ROUTE_SEARCH_EXAMPLES, prompt, _SEMANTIC_SEARCH_THRESHOLD)
        needs_search = best3 == "data" and score3 >= _SEMANTIC_SEARCH_THRESHOLD

        if needs_search:
            print(f"\n  {GREEN}→ ROUTE: chat()  websearch=True  query={prompt!r}{RESET}")
        else:
            print(f"\n  {CYAN}→ ROUTE: chat()  websearch=False{RESET}")

    print()

# ── entrypoint ────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) > 1:
        trace(" ".join(sys.argv[1:]))
        return

    print(f"{BOLD}Aiko route tracer{RESET}  {DIM}(Ctrl+C or empty line to quit){RESET}")
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