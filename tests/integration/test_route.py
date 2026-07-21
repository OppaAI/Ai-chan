#!/usr/bin/env python3
"""
Offline semantic + LLM route tracer for think.py cascade.
Agentic prompts route to the task loop; Stage 2a prints a dry-run sequence
using the tool names exposed by core/agentic.py.
Mode is driven by ROUTE_MODE in .env (same key think.py reads):
  ROUTE_MODE=semantic      → semantic stages + LLM fallback  (default)
  ROUTE_MODE=llm           → LLM stages only
  ROUTE_MODE=semantic_only → semantic stages only  (no LLM fallback)
  ROUTE_MODE=chat          → routing disabled; tracer notes this and exits

Usage:
  python util/test_route.py                        # interactive REPL
  python util/test_route.py "some prompt"          # single prompt
  python util/test_route.py --suite                # run built-in suite + summary
  python util/test_route.py --file test_prompts.json          # external prompt file
  python util/test_route.py --file test_prompts.json --quiet  # progress lines only
  python util/test_route.py --suite --quiet        # quiet built-in suite
"""

import os, sys, time, json, logging, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.WARNING)

try:
    from system.config import load_config
    load_config()
except ImportError:
    pass

# ── minimal stubs so we can import without a running stack ────────────────────

from unittest.mock import MagicMock
sys.modules.setdefault("pygame", MagicMock())
sys.modules.setdefault("openai", MagicMock())

from memory.memorize import AikoMemorize
from cognition.think import (
    AikoThink,
    LLM_BASE_URL,
    LLM_MODEL,
    ROUTER_MODEL,
    _ROUTE_TERNARY_EXAMPLES,
    _SEMANTIC_ROUTE_MIN_GAP,
    _SEMANTIC_LABEL_TOP_K,
    _ROUTE_INSTRUCT_TERNARY,
    _AGENTIC_ROUTE_RE,
)
from agentic.agentic import tool_schemas

# ── colour helpers ────────────────────────────────────────────────────────────

CYAN    = "\033[96m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
RED     = "\033[91m"
MAGENTA = "\033[95m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
RESET   = "\033[0m"

def score_bar(score: float, width: int = 20) -> str:
    filled = max(0, min(width, int(round(score * width))))
    color  = GREEN if score >= 0.5 else YELLOW if score >= 0.35 else RED
    return f"{color}{'█' * filled}{'░' * (width - filled)}{RESET} {score:.3f}"

def pct_bar(pct: float, width: int = 16) -> str:
    filled = int(round(pct / 100 * width))
    color  = GREEN if pct == 100 else YELLOW if pct >= 50 else RED
    return f"{color}{'█' * filled}{'░' * (width - filled)}{RESET}"

# ── built-in mini suite (fallback when no --file given) ───────────────────────

BUILTIN_SUITE: list[tuple[str, str]] = [
    ("can you ping me when it's 8pm",                                        "agentic"),
    ("schedule checking my email every hour and notify me when abc company writes", "agentic"),
    ("research carrot cake recipes, plan ingredients and steps, then write a report", "agentic"),
    ("put together a message for my team about the deployment delay",          "agentic"),
    ("pull up what's new in the ROS2 Jazzy release and summarize it",          "agentic"),
    ("help me map out what I need to do before the hackathon deadline",        "agentic"),
    ("this keeps segfaulting and I don't know why",                            "agentic"),
    ("refactor the schedule runner to use asyncio instead of threads",         "agentic"),
    ("open think.py and agentic.py, combine architecture routing into coding, and update tests", "agentic"),
    ("should I run the embedder on CPU or offload to GPU for aarch64",         "agentic"),
    ("pick up where we left off on the memory consolidation refactor",         "agentic"),
    ("is it weird that I find debugging more satisfying than writing features", "chat"),
    ("what's the actual difference between a semaphore and a mutex",           "chat"),
    ("do you think embeddings are a good long-term foundation for memory",     "chat"),
    ("walk me through why cosine similarity works for semantic search",        "chat"),
    ("I feel like my routing thresholds are tuned to my test set",             "chat"),
    ("what's the Canucks score from last night",                               "chat+search"),
    ("has NVIDIA released any new Jetson hardware this year",                  "chat+search"),
    ("what's ethereum trading at right now",                                   "chat+search"),
    ("did llama.cpp merge the Vulkan backend yet",                             "chat+search"),
    ("what's the current onnxruntime stable release",                          "chat+search"),
]

# ── load external prompt file ─────────────────────────────────────────────────

def load_prompt_file(path: str) -> list[tuple[str, str]]:
    """
    Load prompts from a JSON file.  Accepts two shapes:
      { "prompts": [ {"prompt": "...", "expected": "..."}, ... ] }
      [ {"prompt": "...", "expected": "..."}, ... ]
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    rows = data.get("prompts", data) if isinstance(data, dict) else data
    if not isinstance(rows, list):
        raise ValueError(f"Cannot parse prompt file {path!r}: expected a list or {{\"prompts\": [...]}}")

    suite = []
    for i, row in enumerate(rows):
        p = row.get("prompt") or row.get("text") or row.get("input")
        e = row.get("expected") or row.get("label") or row.get("route")
        if not p:
            raise ValueError(f"Entry {i} in {path!r} missing 'prompt' field")
        if not e:
            raise ValueError(f"Entry {i} in {path!r} missing 'expected' field")
        suite.append((str(p).strip(), str(e).strip()))

    return suite

# ── result record ─────────────────────────────────────────────────────────────

class RouteResult:
    __slots__ = ("prompt", "expected", "got", "latency_ms", "llm_calls", "passed")

    def __init__(self, prompt: str, expected: str):
        self.prompt     = prompt
        self.expected   = expected
        self.got        = "?"
        self.latency_ms = 0.0
        self.llm_calls  = 0
        self.passed     = False

# ── init embedder only (no LLM, no TTS, no schedule) ─────────────────────────

memorize = AikoMemorize()

think = object.__new__(AikoThink)
think._memorize                    = memorize
think._speak                       = None
think._semantic_example_cache      = {}
think._semantic_example_cache_lock = __import__("threading").RLock()
think._pending_search_query        = None
think._route_chat_classified       = None
think._history                     = []
think._history_lock                = __import__("threading").RLock()
think._reasoning                   = False

# ── read ROUTE_MODE ───────────────────────────────────────────────────────────

_ROUTE_MODE        = os.getenv("ROUTE_MODE", "semantic").strip().lower()
_RUN_SEMANTIC      = _ROUTE_MODE not in {"llm", "llm_only", "0", "off", "false", "chat", "disabled"}
_SHOW_LLM_FALLBACK = _ROUTE_MODE not in {"semantic_only", "chat", "0", "off", "false", "disabled"}

# ── lazy real OpenAI client ───────────────────────────────────────────────────

_llm_client = None

def _get_llm_client():
    global _llm_client
    if _llm_client is not None:
        return _llm_client
    sys.modules.pop("openai", None)
    try:
        from openai import OpenAI
        _llm_client = OpenAI(base_url=LLM_BASE_URL, api_key="not-needed")
        return _llm_client
    except Exception as e:
        print(f"\n  {RED}Could not create OpenAI client: {e}{RESET}")
        return None

# ── semantic tracer ───────────────────────────────────────────────────────────

def trace_ternary_stage(
    user_input: str,
    quiet: bool = False,
) -> tuple[str, dict]:
    """Trace the ternary intent routing (agentic/webchat/localchat)."""
    from cognition import reason
    embedder = think._memorize._mem._embedder
    query_vec = embedder.embed_query(user_input, instruct=_ROUTE_INSTRUCT_TERNARY)
    labels, example_vecs = think._semantic_example_vectors(_ROUTE_TERNARY_EXAMPLES, _ROUTE_INSTRUCT_TERNARY)
    label_scores = reason.label_scores_topk(query_vec, labels, example_vecs, top_k=_SEMANTIC_LABEL_TOP_K)
    
    # Apply agentic mode gate
    if not _AGENTIC_MODE_ON:
        label_scores.pop("agentic", None)
    
    agentic_score = label_scores.get("agentic", 0.0)
    webchat_score = label_scores.get("webchat", 0.0)
    localchat_score = label_scores.get("localchat", 0.0)
    
    ranked = sorted(label_scores.items(), key=lambda kv: kv[1], reverse=True)
    best_label, best_score = ranked[0] if ranked else ("localchat", 0.0)
    gap = best_score - ranked[1][1] if len(ranked) > 1 else 1.0
    
    agentic_threshold = float(os.getenv("ROUTE_AGENTIC_THRESHOLD", "0.65"))
    webchat_threshold = float(os.getenv("ROUTE_WEBCHAT_THRESHOLD", "0.60"))
    
    if not quiet:
        print(f"\n  {BOLD}Ternary Routing Trace{RESET}")
        print(f"  {'label':<12} {'score':<8}  bar")
        print(f"  {'-'*48}")
        for lbl, sc in ranked:
            marker = f" {BOLD}← best{RESET}" if lbl == best_label else ""
            print(f"  {lbl:<12} {score_bar(sc)}{marker}")
        
        # Decision logic
        if best_label == "agentic" and agentic_score >= agentic_threshold and gap >= _SEMANTIC_ROUTE_MIN_GAP:
            verdict = f"{GREEN}→ AGENTIC{RESET}"
        elif best_label == "webchat" and webchat_score >= webchat_threshold and gap >= _SEMANTIC_ROUTE_MIN_GAP:
            verdict = f"{CYAN}→ WEBCHAT{RESET}"
        else:
            ambiguous = (agentic_score >= agentic_threshold or webchat_score >= webchat_threshold) and gap < _SEMANTIC_ROUTE_MIN_GAP
            if ambiguous:
                verdict = f"{YELLOW}→ AMBIGUOUS (gap={gap:.3f} < {_SEMANTIC_ROUTE_MIN_GAP}) → LOCALCHAT{RESET}"
            else:
                verdict = f"{CYAN}→ LOCALCHAT{RESET}"
        
        print(f"\n  threshold_agentic={agentic_threshold}  threshold_webchat={webchat_threshold}  min_gap={_SEMANTIC_ROUTE_MIN_GAP}")
        print(f"  agentic={agentic_score:.3f}  webchat={webchat_score:.3f}  localchat={localchat_score:.3f}  gap={gap:.3f}  [{verdict}]")
    
    return best_label, label_scores


def trace_llm_fallback(user_input: str, result: RouteResult | None = None, quiet: bool = False) -> str | None:
    """Trace LLM-based tie-break when semantic scores are ambiguous."""
    if not _ensure_llm_client():
        return None
    if not quiet:
        print(f"\n{BOLD}▶ LLM classifier  (ROUTER_MODEL={ROUTER_MODEL}){RESET}")
    t0 = time.monotonic()
    try:
        label = think._classify_ternary_intent_llm(user_input, allow_agentic=_AGENTIC_MODE_ON)
        elapsed = time.monotonic() - t0
        if not quiet:
            color = GREEN if label != "localchat" else CYAN
            print(f"\n  parsed     : {color}{label}{RESET}")
            print(f"  latency    : {elapsed*1000:.0f} ms")
        if result is not None:
            result.llm_calls += 1
        return label
    except Exception as e:
        elapsed = time.monotonic() - t0
        if not quiet:
            print(f"\n  {RED}LLM call failed ({elapsed*1000:.0f} ms): {e}{RESET}")
        if result is not None:
            result.llm_calls += 1
        return None

# -- dry-run execution sequence hints -----------------------------------------

_AGENTIC_TOOL_NAMES = {
    schema.get("function", {}).get("name", "")
    for schema in tool_schemas()
}

# ── ternary trace ──────────────────────────────────────────────────────────────

def trace_ternary(user_input: str) -> tuple[str, dict]:
    """Trace the ternary intent routing (agentic/webchat/localchat)."""
    scores = think._semantic_example_vectors(_ROUTE_TERNARY_EXAMPLES, _ROUTE_INSTRUCT_TERNARY)
    # think._route_intent does the actual classification, but we can trace scores here
    from cognition import reason
    embedder = think._memorize._mem._embedder
    query_vec = embedder.embed_query(user_input, instruct=_ROUTE_INSTRUCT_TERNARY)
    labels, example_vecs = scores
    label_scores = reason.label_scores_topk(query_vec, labels, example_vecs, top_k=_SEMANTIC_LABEL_TOP_K)
    
    # Apply agentic mode gate
    if not _AGENTIC_MODE_ON:
        label_scores.pop("agentic", None)
    
    agentic_score = label_scores.get("agentic", 0.0)
    webchat_score = label_scores.get("webchat", 0.0)
    localchat_score = label_scores.get("localchat", 0.0)
    
    ranked = sorted(label_scores.items(), key=lambda kv: kv[1], reverse=True)
    best_label, best_score = ranked[0] if ranked else ("localchat", 0.0)
    gap = best_score - ranked[1][1] if len(ranked) > 1 else 1.0
    
    agentic_threshold = float(os.getenv("ROUTE_AGENTIC_THRESHOLD", "0.65"))
    webchat_threshold = float(os.getenv("ROUTE_WEBCHAT_THRESHOLD", "0.60"))
    
    print(f"\n  {BOLD}Ternary Routing Trace{RESET}")
    print(f"  {'label':<12} {'score':<8}  bar")
    print(f"  {'-'*48}")
    for lbl, sc in ranked:
        marker = f" {BOLD}← best{RESET}" if lbl == best_label else ""
        print(f"  {lbl:<12} {score_bar(sc)}{marker}")
    
    # Decision logic
    if best_label == "agentic" and agentic_score >= agentic_threshold and gap >= _SEMANTIC_ROUTE_MIN_GAP:
        verdict = f"{GREEN}→ AGENTIC{RESET}"
    elif best_label == "webchat" and webchat_score >= webchat_threshold and gap >= _SEMANTIC_ROUTE_MIN_GAP:
        verdict = f"{CYAN}→ WEBCHAT{RESET}"
    else:
        ambiguous = (agentic_score >= agentic_threshold or webchat_score >= webchat_threshold) and gap < _SEMANTIC_ROUTE_MIN_GAP
        if ambiguous:
            verdict = f"{YELLOW}→ AMBIGUOUS (gap={gap:.3f} < {_SEMANTIC_ROUTE_MIN_GAP}) → LOCALCHAT{RESET}"
        else:
            verdict = f"{CYAN}→ LOCALCHAT{RESET}"
    
    print(f"\n  threshold_agentic={agentic_threshold}  threshold_webchat={webchat_threshold}  min_gap={_SEMANTIC_ROUTE_MIN_GAP}")
    print(f"  agentic={agentic_score:.3f}  webchat={webchat_score:.3f}  localchat={localchat_score:.3f}  gap={gap:.3f}  [{verdict}]")
    
    return best_label, label_scores

def print_sequence(route: str, prompt: str) -> None:
    print(f"\n{BOLD}▶ dry-run sequence  (from agentic.agentic tool schemas){RESET}")
    print(f"  {'#':>2}  {'kind':<10} {'tool':<22} query/run")
    print(f"  {'-' * 72}")
    for item in _sequence_for_route(route, prompt):
        query = f"query={item['query']!r}  " if item.get("query") else ""
        print(f"  {item['step']:>2}  {item['kind']:<10} {item['tool']:<22} {query}{item['run']}")

# ── core routing logic (shared between verbose and quiet paths) ───────────────

def _compute_route(prompt: str, result: RouteResult | None, quiet: bool) -> str:
    """Compute the final route label, printing detail when quiet=False."""

    final_route = "localchat"

    if _RUN_SEMANTIC:
        if not quiet:
            print(f"\n{BOLD}▶ Ternary Intent Routing  (semantic){RESET}")
        
        best_label, label_scores = trace_ternary_stage(prompt, quiet)
        
        agentic_score = label_scores.get("agentic", 0.0)
        webchat_score = label_scores.get("webchat", 0.0)
        localchat_score = label_scores.get("localchat", 0.0)
        
        agentic_threshold = float(os.getenv("ROUTE_AGENTIC_THRESHOLD", "0.65"))
        webchat_threshold = float(os.getenv("ROUTE_WEBCHAT_THRESHOLD", "0.60"))
        
        is_agentic = (best_label == "agentic" and 
                      agentic_score >= agentic_threshold and 
                      (agentic_score - webchat_score) >= _SEMANTIC_ROUTE_MIN_GAP)
        is_webchat = (best_label == "webchat" and 
                      webchat_score >= webchat_threshold and 
                      (webchat_score - agentic_score) >= _SEMANTIC_ROUTE_MIN_GAP)
        
        ambiguous = ((agentic_score >= agentic_threshold or webchat_score >= webchat_threshold)
                     and abs(agentic_score - webchat_score) < _SEMANTIC_ROUTE_MIN_GAP)

        if is_agentic:
            final_route = "agentic"
            if not quiet:
                print(f"\n  {GREEN}→ ROUTE: agentic_chat{RESET}")
        
        elif is_webchat:
            final_route = "webchat"
            if not quiet:
                print(f"\n  {CYAN}→ ROUTE: webchat{RESET}")
        
        elif ambiguous:
            if not quiet:
                print(f"\n  {YELLOW}→ AMBIGUOUS - LLM fallback{RESET}")
            if _SHOW_LLM_FALLBACK:
                llm_label = trace_llm_fallback(prompt, result, quiet)
                final_route = "agentic" if llm_label == "agentic" else "localchat"
            else:
                final_route = "localchat"
        
        else:
            final_route = "localchat"
            if not quiet:
                print(f"\n  {CYAN}→ ROUTE: localchat{RESET}")

    # ── LLM-only mode ────────────────────────────────────────────────────────
    if not _RUN_SEMANTIC:
        if not quiet:
            print(f"\n{BOLD}▶ Skipping semantic stages (ROUTE_MODE={_ROUTE_MODE}){RESET}")
        think._history = []
        if _ROUTE_MODE != "llm_only" and _AGENTIC_ROUTE_RE.search(prompt):
            llm_label = "agentic"
            if not quiet:
                print(f"\n  {GREEN}→ ROUTE: agentic_chat  (deterministic task pattern){RESET}")
        else:
            llm_label = trace_llm_fallback(prompt, result, quiet)
        think._history = []

        if llm_label == "agentic":
            final_route = "agentic"
        else:
            final_route = "localchat"

    return final_route

# ── public trace entry point ──────────────────────────────────────────────────

def trace(prompt: str, result: RouteResult | None = None, quiet: bool = False) -> str:
    t_start = time.monotonic()

    if not quiet:
        print(f"\n{BOLD}{'═'*60}{RESET}")
        print(f"{BOLD}PROMPT:{RESET} {CYAN}{prompt!r}{RESET}")
        print(f"{'═'*60}")

    final_route = _compute_route(prompt, result, quiet)

    if not quiet:
        print_sequence(final_route, prompt)

    if result is not None:
        result.latency_ms = (time.monotonic() - t_start) * 1000
        result.got        = final_route
        result.passed     = (final_route == result.expected)

    if not quiet:
        print()

    return final_route

# ── summary table ─────────────────────────────────────────────────────────────

def print_summary(results: list[RouteResult], source_label: str = "") -> None:
    from collections import defaultdict

    total       = len(results)
    passed      = sum(1 for r in results if r.passed)
    total_lat   = sum(r.latency_ms for r in results)
    total_llm   = sum(r.llm_calls  for r in results)
    accuracy    = passed / total * 100 if total else 0.0
    avg_lat     = total_lat / total if total else 0.0
    llm_avg = total_llm / total if total else 0.0

    C_PROMPT  = 50
    C_EXP     = 16
    C_GOT     = 16
    C_LAT     = 8
    C_LLM     = 5
    C_VERDICT = 6

    header = (
        f"{'prompt':<{C_PROMPT}} "
        f"{'expected':<{C_EXP}} "
        f"{'got':<{C_GOT}} "
        f"{'ms':>{C_LAT}} "
        f"{'llm':>{C_LLM}} "
        f"{'result':<{C_VERDICT}}"
    )
    W = len(header)
    SEP = "─" * W

    label_str = f"  [{source_label}]" if source_label else ""

    print(f"\n{BOLD}{'═' * W}{RESET}")
    print(f"{BOLD}RESULTS{label_str}{RESET}")
    print(f"{'═' * W}")
    print(f"  accuracy    {pct_bar(accuracy)} {passed}/{total}  ({accuracy:.1f}%)")
    print(f"  avg latency {avg_lat:>7.0f} ms")
    print(f"  total time  {total_lat:>7.0f} ms")
    print(f"  LLM calls   {total_llm:>4}  ({llm_avg:.2f} per prompt avg)")
    print(f"{'═' * W}")
    print(f"\n{BOLD}{header}{RESET}")
    print(SEP)

    for r in results:
        short   = r.prompt if len(r.prompt) <= C_PROMPT else r.prompt[:C_PROMPT - 1] + "…"
        verdict = f"{GREEN}PASS{RESET}" if r.passed else f"{RED}FAIL{RESET}"
        gc      = GREEN if r.passed else RED
        print(
            f"{short:<{C_PROMPT}} "
            f"{r.expected:<{C_EXP}} "
            f"{gc}{r.got:<{C_GOT}}{RESET} "
            f"{r.latency_ms:>{C_LAT}.0f} "
            f"{r.llm_calls:>{C_LLM}} "
            f"{verdict}"
        )

    print(SEP)

    # ── per-label breakdown ───────────────────────────────────────────────────
    by_label: dict[str, list[RouteResult]] = defaultdict(list)
    for r in results:
        by_label[r.expected].append(r)

    print(f"\n{BOLD}per-label breakdown{RESET}")
    label_header = f"  {'label':<16} {'acc':>4}  bar               n    avg ms  llm/prompt"
    print(label_header)
    print("  " + "─" * (len(label_header) - 2))

    for label in sorted(by_label):
        rows   = by_label[label]
        n      = len(rows)
        n_pass = sum(1 for r in rows if r.passed)
        pct    = n_pass / n * 100
        a_lat  = sum(r.latency_ms for r in rows) / n
        a_llm  = sum(r.llm_calls  for r in rows) / n
        print(
            f"  {label:<16} {pct:>3.0f}%  {pct_bar(pct)}  "
            f"{n:>2}  {a_lat:>6.0f} ms  {a_llm:.2f}"
        )

    # ── failures list ─────────────────────────────────────────────────────────
    failures = [r for r in results if not r.passed]
    if failures:
        print(f"\n{BOLD}{RED}failures ({len(failures)}){RESET}")
        for r in failures:
            short = r.prompt if len(r.prompt) <= 60 else r.prompt[:59] + "…"
            print(f"  {RED}✗{RESET}  {short}")
            print(f"       expected={r.expected}  got={r.got}  {r.latency_ms:.0f} ms  {r.llm_calls} llm")
    else:
        print(f"\n{GREEN}All {total} prompts passed.{RESET}")

    print()

# ── batch runner ──────────────────────────────────────────────────────────────

def _warmup_embedder(suite: list[tuple[str, str]]) -> None:
    if not suite:
        return
    think._semantic_all_scores(suite[0][0], _ROUTE_BINARY_EXAMPLES, _ROUTE_INSTRUCT_BINARY)

def run_suite(suite: list[tuple[str, str]], quiet: bool, source_label: str = "", seed: int | None = None) -> None:
    mode_tag = {
        "semantic":      f"{CYAN}semantic (+ LLM fallback){RESET}",
        "semantic_only": f"{CYAN}semantic-only{RESET}",
        "llm":           f"{MAGENTA}LLM-only{RESET}",
    }.get(_ROUTE_MODE, f"{CYAN}{_ROUTE_MODE}{RESET}")

    if seed is None:
        seed = random.randint(0, 0xFFFF)
    rng = random.Random(seed)
    suite = list(suite)
    rng.shuffle(suite)

    print(f"\n{BOLD}Aiko route tracer{RESET}  ROUTE_MODE={mode_tag}  {len(suite)} prompts  source={source_label or 'built-in'}  seed={seed}")

    _warmup_embedder(suite)
    results = []
    for i, (prompt, expected) in enumerate(suite, 1):
        r = RouteResult(prompt, expected)
        trace(prompt, result=r, quiet=quiet)
        if quiet:
            verdict = f"{GREEN}✓{RESET}" if r.passed else f"{RED}✗{RESET}"
            short   = prompt if len(prompt) <= 56 else prompt[:55] + "…"
            print(f"  {verdict} [{i:>3}/{len(suite)}]  {short:<56}  {r.got:<14}  {r.latency_ms:>6.0f} ms  {r.llm_calls} llm")
        results.append(r)

    print_summary(results, source_label=source_label)

# ── entrypoint ────────────────────────────────────────────────────────────────

def main():
    if _ROUTE_MODE in {"chat", "0", "off", "false", "disabled"}:
        print(f"{YELLOW}ROUTE_MODE={_ROUTE_MODE} — routing is disabled in .env; nothing to trace.{RESET}")
        sys.exit(0)

    # ── parse args properly: flags consume their value so it never leaks ────
    raw            = sys.argv[1:]
    run_suite_flag = False
    quiet          = False
    file_path: str | None = None
    seed:      int | None = None
    cli_prompt_parts: list[str] = []

    i = 0
    while i < len(raw):
        a = raw[i]
        if a == "--suite":
            run_suite_flag = True
        elif a == "--quiet":
            quiet = True
        elif a == "--file":
            i += 1
            if i >= len(raw):
                print(f"{RED}--file requires a path argument{RESET}")
                sys.exit(1)
            file_path = raw[i]
        elif a == "--seed":
            i += 1
            if i >= len(raw):
                print(f"{RED}--seed requires an integer argument{RESET}")
                sys.exit(1)
            try:
                seed = int(raw[i])
            except ValueError:
                print(f"{RED}--seed requires an integer, got {raw[i]!r}{RESET}")
                sys.exit(1)
        elif a.startswith("--"):
            print(f"{YELLOW}unknown flag {a!r} — ignored{RESET}")
        else:
            cli_prompt_parts.append(a)
        i += 1

    if file_path:
        try:
            suite = load_prompt_file(file_path)
        except Exception as e:
            print(f"{RED}Failed to load {file_path!r}: {e}{RESET}")
            sys.exit(1)
        run_suite(suite, quiet=quiet, source_label=os.path.basename(file_path), seed=seed)
        return

    if run_suite_flag:
        run_suite(BUILTIN_SUITE, quiet=quiet, source_label="built-in", seed=seed)
        return

    if cli_prompt_parts:
        trace(" ".join(cli_prompt_parts))
        return

    # interactive REPL
    mode_tag = {
        "semantic":      f"{CYAN}semantic (+ LLM fallback){RESET}",
        "semantic_only": f"{CYAN}semantic-only (no LLM fallback){RESET}",
        "llm":           f"{MAGENTA}LLM-only{RESET}",
    }.get(_ROUTE_MODE, f"{CYAN}{_ROUTE_MODE}{RESET}")

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