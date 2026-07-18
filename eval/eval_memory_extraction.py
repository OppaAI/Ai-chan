"""
eval/eval_memory_extraction.py
Accuracy eval for memory/memorize.py's fact extraction -- NOT a pytest
suite. Extraction accuracy depends on the real LLM's non-deterministic
phrasing, so this is scored and reported like bench_intent_routing.py,
not asserted pass/fail in CI.

Usage:
    python eval/eval_memory_extraction.py
    python eval/eval_memory_extraction.py --verbose
    python eval/eval_memory_extraction.py --out results/extraction_2026-07-18.json

What it measures, per golden case:
  - precision / recall / F1 of extracted facts against expected facts
    (fuzzy-matched via embedding cosine similarity, since exact string
    match against LLM phrasing is too brittle -- "Oppa's birthday is
    June 3" vs "Oppa was born on June 3rd" should both count as a hit)
  - hedge-leakage: did any fact containing hedging language (might,
    probably, seems...) slip past _HEDGE_RE and get persisted anyway?
  - false-addition rate: facts extracted that don't correspond to
    anything in expected_facts (hallucinated or over-eager extraction)

Run this after any change to _EXTRACT_PROMPT, the hedge filter, or the
underlying extraction model -- compare the printed summary against the
last run's saved JSON to see if accuracy moved.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Golden dataset
# ─────────────────────────────────────────────────────────────────────────────
# Each case: a short conversation, the facts a correct extraction SHOULD
# produce, and facts that must NOT appear (hedged/uncertain statements that
# _HEDGE_RE is supposed to filter out).
#
# Keep conversations realistic and varied: some with 1 fact, some with
# multiple, some with zero (nothing worth remembering), some containing
# hedge language mixed with real facts.

GOLDEN_SET = [
    {
        "id": "simple_single_fact",
        "conversation": [
            {"role": "user", "content": "My birthday is June 3rd, just so you know."},
            {"role": "assistant", "content": "Got it, I'll remember that!"},
        ],
        "expected_facts": ["Oppa's birthday is June 3"],
        "forbidden_facts": [],
    },
    {
        "id": "multiple_facts_one_turn",
        "conversation": [
            {
                "role": "user",
                "content": (
                    "I'm building a robot called Grace, and I also just adopted "
                    "a cat named Max. I hate mushrooms by the way."
                ),
            },
            {"role": "assistant", "content": "That's a lot of exciting news!"},
        ],
        "expected_facts": [
            "Oppa is building a robot called Grace",
            "Oppa has a cat named Max",
            "Oppa dislikes mushrooms",
        ],
        "forbidden_facts": [],
    },
    {
        "id": "hedge_language_should_be_dropped",
        "conversation": [
            {
                "role": "user",
                "content": (
                    "I think I might want to learn piano someday, not sure though. "
                    "But I definitely finished my HVAC certification in 2018."
                ),
            },
            {"role": "assistant", "content": "Nice, HVAC cert is solid!"},
        ],
        "expected_facts": ["Oppa finished his HVAC certification in 2018"],
        "forbidden_facts": ["Oppa might want to learn piano", "Oppa wants to learn piano"],
    },
    {
        "id": "trivial_conversation_nothing_to_extract",
        "conversation": [
            {"role": "user", "content": "hey"},
            {"role": "assistant", "content": "Hey! What's up?"},
        ],
        "expected_facts": [],
        "forbidden_facts": [],
    },
    {
        "id": "deadline_and_project_context",
        "conversation": [
            {
                "role": "user",
                "content": (
                    "Quick heads up, I have a deadline on Friday for the intent "
                    "routing benchmark, and I'm working on it in Aiko-chan."
                ),
            },
            {"role": "assistant", "content": "Noted, good luck with the benchmark!"},
        ],
        "expected_facts": [
            "Oppa has a deadline on Friday for the intent routing benchmark",
        ],
        "forbidden_facts": [],
    },
    {
        "id": "assistant_only_facts_should_not_leak",
        "conversation": [
            {"role": "user", "content": "How's it going Aiko?"},
            {
                "role": "assistant",
                "content": "I've been thinking about our last conversation and feeling quite reflective.",
            },
        ],
        "expected_facts": [],
        "forbidden_facts": ["Aiko feels reflective", "Aiko has been thinking"],
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Fuzzy fact matching -- embedding cosine similarity, not exact string match
# ─────────────────────────────────────────────────────────────────────────────

MATCH_THRESHOLD = 0.80  # cosine similarity above which two facts "count" as the same


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-9
    return float(np.dot(a, b) / denom)


def _match_facts(extracted: list[str], expected: list[str], embed_fn) -> dict:
    """
    Greedy bipartite matching by cosine similarity. Returns matched pairs,
    unmatched expected (misses / recall failures), and unmatched extracted
    (false additions / precision failures).
    """
    if not extracted and not expected:
        return {"matched": [], "missed": [], "false_additions": []}

    extracted_vecs = embed_fn(extracted) if extracted else []
    expected_vecs = embed_fn(expected) if expected else []

    used_extracted = set()
    matched, missed = [], []

    for e_text, e_vec in zip(expected, expected_vecs):
        best_idx, best_score = None, -1.0
        for i, x_vec in enumerate(extracted_vecs):
            if i in used_extracted:
                continue
            score = _cosine(np.array(e_vec), np.array(x_vec))
            if score > best_score:
                best_idx, best_score = i, score
        if best_idx is not None and best_score >= MATCH_THRESHOLD:
            used_extracted.add(best_idx)
            matched.append({"expected": e_text, "extracted": extracted[best_idx], "score": round(best_score, 3)})
        else:
            missed.append(e_text)

    false_additions = [extracted[i] for i in range(len(extracted)) if i not in used_extracted]
    return {"matched": matched, "missed": missed, "false_additions": false_additions}


def _contains_forbidden(extracted: list[str], forbidden: list[str], embed_fn) -> list[str]:
    """Same fuzzy match, but for facts that should NOT have been extracted."""
    if not forbidden or not extracted:
        return []
    result = _match_facts(extracted, forbidden, embed_fn)
    return [m["extracted"] for m in result["matched"]]


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def _load_real_backend():
    """
    Import the real _MemoryBackend + embedder. Fails loudly and early if
    the environment (LLM_BASE_URL, GGUF model, etc.) isn't set up --
    this eval is meant to run on the actual device with real components.
    """
    from memory.memorize import _MemoryBackend

    backend = _MemoryBackend(
        db_path="/tmp/eval_memory_extraction.db",
        llm_base_url="http://localhost:8080/v1",
        model="ministral",
    )
    return backend


def run_eval(verbose: bool = False) -> dict:
    backend = _load_real_backend()

    def embed_fn(texts: list[str]) -> list[list[float]]:
        return backend._embed_batch(texts)

    case_results = []
    total_tp = total_fp = total_fn = 0
    hedge_leaks = 0

    for case in GOLDEN_SET:
        extracted = backend._extract_facts(case["conversation"], display_name="Oppa")
        match = _match_facts(extracted, case["expected_facts"], embed_fn)
        leaked = _contains_forbidden(extracted, case["forbidden_facts"], embed_fn)

        tp = len(match["matched"])
        fp = len(match["false_additions"])
        fn = len(match["missed"])
        total_tp += tp
        total_fp += fp
        total_fn += fn
        hedge_leaks += len(leaked)

        precision = tp / (tp + fp) if (tp + fp) else 1.0
        recall = tp / (tp + fn) if (tp + fn) else 1.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 1.0

        case_results.append({
            "id": case["id"],
            "extracted": extracted,
            "matched": match["matched"],
            "missed": match["missed"],
            "false_additions": match["false_additions"],
            "forbidden_leaked": leaked,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
        })

        if verbose:
            print(f"\n[{case['id']}]")
            print(f"  extracted: {extracted}")
            print(f"  precision={precision:.3f} recall={recall:.3f} f1={f1:.3f}")
            if match["missed"]:
                print(f"  MISSED: {match['missed']}")
            if match["false_additions"]:
                print(f"  FALSE ADDITIONS: {match['false_additions']}")
            if leaked:
                print(f"  HEDGE LEAK (should have been filtered): {leaked}")

    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 1.0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 1.0
    overall_f1 = (
        2 * overall_precision * overall_recall / (overall_precision + overall_recall)
        if (overall_precision + overall_recall) else 1.0
    )

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_cases": len(GOLDEN_SET),
        "overall_precision": round(overall_precision, 3),
        "overall_recall": round(overall_recall, 3),
        "overall_f1": round(overall_f1, 3),
        "hedge_leaks": hedge_leaks,
        "cases": case_results,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Eval memory extraction accuracy against a golden set.")
    parser.add_argument("--verbose", action="store_true", help="Print per-case detail.")
    parser.add_argument("--out", type=str, default=None, help="Optional path to save full JSON results.")
    args = parser.parse_args()

    try:
        summary = run_eval(verbose=args.verbose)
    except Exception as e:
        print(f"ERROR: could not run eval -- is the real embedder/LLM available? ({e})", file=sys.stderr)
        sys.exit(1)

    print("\n" + "=" * 60)
    print(f"Extraction eval — {summary['num_cases']} cases")
    print(f"  precision: {summary['overall_precision']}")
    print(f"  recall:    {summary['overall_recall']}")
    print(f"  f1:        {summary['overall_f1']}")
    print(f"  hedge leaks (should always be 0): {summary['hedge_leaks']}")
    print("=" * 60)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2))
        print(f"Full results saved to {out_path}")


if __name__ == "__main__":
    main()
