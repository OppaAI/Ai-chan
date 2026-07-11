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
    _ROUTE_BINARY_EXAMPLES,
    _ROUTE_TOOL_EXAMPLES,
    _ROUTE_SEARCH_EXAMPLES,
    _SEMANTIC_ROUTE_THRESHOLD,
    _SEMANTIC_SEARCH_THRESHOLD,
    _SEMANTIC_ROUTE_MIN_GAP,
    _SEMANTIC_TOOL_MIN_GAP,
    _SEMANTIC_SEARCH_MIN_GAP,
    _SEMANTIC_LABEL_TOP_K,
    _ROUTE_INSTRUCT_BINARY,
    _ROUTE_INSTRUCT_TOOL,
    _ROUTE_INSTRUCT_SEARCH,
    _AGENTIC_ROUTE_RE,
)
from skills.agentic import tool_schemas

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

def trace_stage(
    label: str,
    examples_by_label: dict,
    user_input: str,
    threshold: float,
    min_gap: float,
    instruct: str,
) -> tuple[str, float]:
    scores = think._semantic_all_scores(user_input, examples_by_label, instruct)
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

    verdict = GREEN + "PASS" + RESET if (best_score >= threshold and gap >= min_gap) else RED + "FAIL" + RESET
    print(f"\n  threshold={threshold}  gap_min={min_gap}  top_k={_SEMANTIC_LABEL_TOP_K}")
    print(f"  best={best_label}  score={best_score:.3f}  gap={gap:.3f}  [{verdict}]")
    return best_label, best_score

# ── LLM tracer ────────────────────────────────────────────────────────────────

def _ensure_llm_client() -> bool:
    if getattr(think, "_client", None) is not None and not isinstance(think._client, MagicMock):
        return True
    client = _get_llm_client()
    if client is None:
        return False
    think._client       = client
    think._router_model = ROUTER_MODEL
    think._llm_model    = LLM_MODEL
    return True

def trace_llm_router(user_input: str, result: RouteResult | None = None, quiet: bool = False) -> str | None:
    if not _ensure_llm_client():
        return None
    if not quiet:
        print(f"\n{BOLD}▶ LLM classifier  (ROUTER_MODEL={ROUTER_MODEL}){RESET}")
    t0 = time.monotonic()
    try:
        label   = think._classify_agent_intent(user_input)
        elapsed = time.monotonic() - t0
        if not quiet:
            color = GREEN if label != "chat" else CYAN
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

def trace_llm_search_resolve(user_input: str, result: RouteResult | None = None, quiet: bool = False) -> str | None:
    if not _ensure_llm_client():
        return None
    if not quiet:
        print(f"\n{BOLD}▶ LLM search query resolver  (_llm_resolve_search_query){RESET}")
    t0 = time.monotonic()
    try:
        query   = think._llm_resolve_search_query(user_input)
        elapsed = time.monotonic() - t0
        if not quiet:
            print(f"\n  query      : {CYAN}{query!r}{RESET}")
            print(f"  latency    : {elapsed*1000:.0f} ms")
        if result is not None:
            result.llm_calls += 1
        return query
    except Exception as e:
        elapsed = time.monotonic() - t0
        if not quiet:
            print(f"\n  {RED}LLM call failed ({elapsed*1000:.0f} ms): {e!r}{RESET}")
        if result is not None:
            result.llm_calls += 1
        return None

# -- dry-run execution sequence hints -----------------------------------------

_AGENTIC_TOOL_NAMES = {
    schema.get("function", {}).get("name", "")
    for schema in tool_schemas()
}

_STEP_TOOLS: dict[str, list[str]] = {
    "research": ["web_search", "fetch_page", "save_note"],
    "planning": ["make_plan", "create_checklist"],
    "writing": ["save_note"],
    "coding": ["repo_search_text", "repo_read_file", "summarize_task_state"],
    "decision": ["make_plan", "summarize_task_state"],
    "schedule": ["schedule_job"],
    "ongoing": ["summarize_task_state", "repo_file_tree"],
}

def _clean_query(prompt: str, max_words: int = 9) -> str:
    words = [
        w.strip(".,;:!?()[]{}\"'").lower()
        for w in prompt.split()
        if w.strip(".,;:!?()[]{}\"'")
    ]
    stop = {
        "a", "an", "and", "are", "as", "at", "be", "can", "for", "from", "has",
        "how", "i", "if", "in", "is", "it", "me", "my", "of", "on", "or", "the",
        "then", "this", "to", "what", "when", "whether", "with", "you",
    }
    terms = [w for w in words if w not in stop]
    return " ".join(terms[:max_words]) or prompt[:80]

def _ranked_agentic_steps(prompt: str) -> list[str]:
    scores = think._semantic_all_scores(prompt, _ROUTE_TOOL_EXAMPLES, _ROUTE_INSTRUCT_TOOL)
    if not scores:
        return ["planning"]
    ranked = [label for label, _score in sorted(scores.items(), key=lambda item: item[1], reverse=True)]
    selected = [
        label for label, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)
        if score >= _SEMANTIC_ROUTE_THRESHOLD
    ]
    return (selected or ranked)[:10]

def _sequence_for_route(route: str, prompt: str) -> list[dict]:
    query = _clean_query(prompt)
    if route == "chat":
        return [{"step": 1, "kind": "chat", "tool": "chat", "query": "", "run": "answer from existing context"}]
    if route == "chat+search":
        return [
            {"step": 1, "kind": "websearch", "tool": "web_search", "query": query, "run": "retrieve current external facts"},
            {"step": 2, "kind": "chat", "tool": "chat", "query": prompt, "run": "answer using search context"},
        ]

    sequence: list[dict] = []
    seen: set[str] = set()
    for label in _ranked_agentic_steps(prompt):
        for tool in _STEP_TOOLS.get(label, []):
            if tool not in _AGENTIC_TOOL_NAMES or tool in seen:
                continue
            seen.add(tool)
            sequence.append({
                "step": len(sequence) + 1,
                "kind": label,
                "tool": tool,
                "query": query if tool in {"web_search", "fetch_page", "repo_search_text", "search_skillsets"} else "",
                "run": "call via core.agentic dispatch",
            })
            if len(sequence) >= 10:
                break
        if len(sequence) >= 10:
            break

    if "final_answer" in _AGENTIC_TOOL_NAMES and len(sequence) < 10:
        sequence.append({
            "step": len(sequence) + 1,
            "kind": "final",
            "tool": "final_answer",
            "query": "",
            "run": "report completed work naturally",
        })
    return sequence or [{"step": 1, "kind": "agentic", "tool": "final_answer", "query": "", "run": "report route decision"}]

def print_sequence(route: str, prompt: str) -> None:
    print(f"\n{BOLD}▶ dry-run sequence  (from skills.agentic tool schemas){RESET}")
    print(f"  {'#':>2}  {'kind':<10} {'tool':<22} query/run")
    print(f"  {'-' * 72}")
    for item in _sequence_for_route(route, prompt):
        query = f"query={item['query']!r}  " if item.get("query") else ""
        print(f"  {item['step']:>2}  {item['kind']:<10} {item['tool']:<22} {query}{item['run']}")

# ── core routing logic (shared between verbose and quiet paths) ───────────────

def _compute_route(prompt: str, result: RouteResult | None, quiet: bool) -> str:
    """Compute the final route label, printing detail when quiet=False."""

    final_route = "chat"

    if _RUN_SEMANTIC:
        if not quiet:
            print(f"\n{BOLD}▶ Stage 1  chat vs agentic  (semantic){RESET}")
            trace_stage("binary classifier", _ROUTE_BINARY_EXAMPLES, prompt,
                        _SEMANTIC_ROUTE_THRESHOLD, _SEMANTIC_ROUTE_MIN_GAP, _ROUTE_INSTRUCT_BINARY)

        scores1 = think._semantic_all_scores(prompt, _ROUTE_BINARY_EXAMPLES, _ROUTE_INSTRUCT_BINARY)
        sorted1 = sorted(scores1.values(), reverse=True)
        gap1    = sorted1[0] - sorted1[1] if len(sorted1) > 1 else 1.0
        best1   = max(scores1, key=scores1.get) if scores1 else "chat"
        score1  = scores1.get(best1, 0.0)

        is_agentic = best1 == "agentic" and score1 >= _SEMANTIC_ROUTE_THRESHOLD and gap1 >= _SEMANTIC_ROUTE_MIN_GAP

        if is_agentic:
            if not quiet:
                print(f"\n  {GREEN}→ AGENTIC{RESET}")
                print(f"\n{BOLD}▶ Stage 2a  likely work steps  (semantic hint only){RESET}")
                trace_stage("step hints", _ROUTE_TOOL_EXAMPLES, prompt,
                            _SEMANTIC_ROUTE_THRESHOLD, _SEMANTIC_TOOL_MIN_GAP, _ROUTE_INSTRUCT_TOOL)

            scores2 = think._semantic_all_scores(prompt, _ROUTE_TOOL_EXAMPLES, _ROUTE_INSTRUCT_TOOL)
            sorted2 = sorted(scores2.values(), reverse=True)
            gap2    = sorted2[0] - sorted2[1] if len(sorted2) > 1 else 1.0
            best2   = max(scores2, key=scores2.get) if scores2 else "coding"
            score2  = scores2.get(best2, 0.0)

            final_route = "agentic"
            if not quiet:
                likely = [
                    label for label, score in sorted(scores2.items(), key=lambda item: item[1], reverse=True)
                    if score >= _SEMANTIC_ROUTE_THRESHOLD
                ]
                hint = " -> ".join(likely[:4]) if likely else best2
                print(f"\n  {GREEN}→ ROUTE: agentic_chat  steps≈{hint}{RESET}")
                print(f"\n  {DIM}(Stage 2a is a hint; agentic.py chooses the actual tool sequence.){RESET}")


        else:
            if not quiet:
                print(f"\n  {CYAN}→ CHAT{RESET}")
                print(f"\n{BOLD}▶ Stage 2b  websearch needed?  (semantic){RESET}")
                trace_stage("search classifier", _ROUTE_SEARCH_EXAMPLES, prompt,
                            _SEMANTIC_SEARCH_THRESHOLD, _SEMANTIC_SEARCH_MIN_GAP, _ROUTE_INSTRUCT_SEARCH)

            scores3 = think._semantic_all_scores(prompt, _ROUTE_SEARCH_EXAMPLES, _ROUTE_INSTRUCT_SEARCH)
            sorted3 = sorted(scores3.values(), reverse=True)
            gap3    = sorted3[0] - sorted3[1] if len(sorted3) > 1 else 1.0
            best3   = max(scores3, key=scores3.get) if scores3 else "no"
            score3  = scores3.get(best3, 0.0)

            needs_search = best3 == "data" and score3 >= _SEMANTIC_SEARCH_THRESHOLD and gap3 >= _SEMANTIC_SEARCH_MIN_GAP
            ambiguous    = score1 >= _SEMANTIC_ROUTE_THRESHOLD and gap1 < _SEMANTIC_ROUTE_MIN_GAP

            if needs_search:
                final_route = "chat+search"
                if not quiet:
                    print(f"\n  {GREEN}→ ROUTE: chat()  websearch=True  query={prompt!r}{RESET}")
            elif ambiguous:
                if not quiet:
                    print(f"\n  {YELLOW}→ ROUTE: llm_fallback (binary scores too close){RESET}")
                if _SHOW_LLM_FALLBACK:
                    llm_label = trace_llm_router(prompt, result, quiet)
                    final_route = "agentic" if llm_label == "agentic" else "chat"
            else:
                final_route = "chat"
                if not quiet:
                    print(f"\n  {CYAN}→ ROUTE: chat()  websearch=False{RESET}")

            # search resolver: only in LLM fallback mode, not every chat turn
            if _SHOW_LLM_FALLBACK and needs_search and _ROUTE_MODE == "llm":
                trace_llm_search_resolve(prompt, result, quiet)

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
            llm_label = trace_llm_router(prompt, result, quiet)
        think._history = []

        if llm_label == "agentic":
            final_route = "agentic"
        else:
            # Production routing gates search semantically before resolving a query.
            if think._needs_websearch(prompt):
                trace_llm_search_resolve(prompt, result, quiet)
                final_route = "chat+search"
            else:
                final_route = "chat"

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