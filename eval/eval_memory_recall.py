"""
eval/eval_memory_recall.py
Accuracy eval for memory/memorize.py's search()/recall pipeline -- NOT a
pytest suite. Like eval_memory_extraction.py, this is scored and reported
rather than asserted pass/fail, since recall quality is a spectrum you
want to track over time (e.g. after touching RRF weights, recency decay,
or the quick/wide pass thresholds), not a binary gate.

Usage:
    python eval/eval_memory_recall.py
    python eval/eval_memory_recall.py --verbose
    python eval/eval_memory_recall.py --out results/recall_2026-07-18.json

What it measures:
  - recall@k: for each golden query, does the "must_retrieve" memory
    appear anywhere in the top-k results?
  - mean reciprocal rank (MRR): not just whether it's in top-k, but how
    close to #1 it lands -- useful for catching "technically still
    passes recall@5 but keeps sliding to rank 4" regressions
  - distractor contamination: seeds realistic noise memories alongside
    the target facts, so recall is tested under real competing-candidate
    conditions, not an artificially empty store

This eval seeds its own memory store with a mix of target facts and
distractor noise -- it does NOT run against your live Aiko-chan memory.db.
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Golden dataset: target facts to seed + queries that should retrieve them
# ─────────────────────────────────────────────────────────────────────────────

TARGET_FACTS = [
    "Oppa's birthday is June 3",
    "Oppa is building a robot called Grace",
    "Oppa has a cat named Max",
    "Oppa dislikes mushrooms",
    "Oppa finished his HVAC certification in 2018",
    "Oppa has a deadline on Friday for the intent routing benchmark",
    "Oppa is working on Moonlight Sonata on piano",
    "Oppa lives in Vancouver, BC",
]

# Distractor noise -- realistic-sounding but irrelevant to the queries below,
# seeded alongside targets so recall is tested under real competition.
DISTRACTOR_COUNT = 200

GOLDEN_QUERIES = [
    {"query": "when is my birthday", "must_retrieve": "Oppa's birthday is June 3", "k": 5},
    {"query": "what robot am I building", "must_retrieve": "Oppa is building a robot called Grace", "k": 5},
    {"query": "do I have any pets", "must_retrieve": "Oppa has a cat named Max", "k": 5},
    {"query": "what food do I not like", "must_retrieve": "Oppa dislikes mushrooms", "k": 5},
    {"query": "when did I finish my certification", "must_retrieve": "Oppa finished his HVAC certification in 2018", "k": 5},
    {"query": "what deadline do I have coming up", "must_retrieve": "Oppa has a deadline on Friday for the intent routing benchmark", "k": 5},
    {"query": "what song am I learning on piano", "must_retrieve": "Oppa is working on Moonlight Sonata on piano", "k": 5},
    {"query": "where do I live", "must_retrieve": "Oppa lives in Vancouver, BC", "k": 5},
]


# ─────────────────────────────────────────────────────────────────────────────
# Store seeding
# ─────────────────────────────────────────────────────────────────────────────

def _seed_store(backend, user_id: str) -> dict:
    """
    Seed target facts + distractor noise into a real _MemoryBackend.
    Returns {fact_text: memory_id} for the target facts, so we can check
    whether the RIGHT id came back, not just a text substring match
    (protects against near-duplicate distractor text producing a false
    positive).
    """
    import sqlite_vec

    now = datetime.now(timezone.utc)
    fact_ids = {}

    all_texts = list(TARGET_FACTS) + [
        f"Distractor memory {i}: unrelated synthetic noise for recall eval scale testing."
        for i in range(DISTRACTOR_COUNT)
    ]
    vectors = backend._embed_batch(all_texts)

    for i, (text, vector) in enumerate(zip(all_texts, vectors)):
        mem_id = str(uuid.uuid4())
        created = (now - timedelta(days=i % 60)).isoformat()
        backend._conn.execute(
            """
            INSERT INTO memories (id, user_id, memory, created_at, access_count, last_accessed_at, pinned)
            VALUES (?, ?, ?, ?, 0, 'never', 0)
            """,
            (mem_id, user_id, text, created),
        )
        backend._conn.execute(
            "INSERT INTO memories_vec(id, embedding) VALUES (?, ?)",
            (mem_id, sqlite_vec.serialize_float32(vector.tolist())),
        )
        if text in TARGET_FACTS:
            fact_ids[text] = mem_id

    backend._conn.commit()
    return fact_ids


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────

def _rank_of(results: list[dict], target_id: str) -> int | None:
    """1-indexed rank of target_id in results, or None if absent."""
    for i, r in enumerate(results):
        if r.get("id") == target_id:
            return i + 1
    return None


def run_eval(verbose: bool = False) -> dict:
    from memory.memorize import _MemoryBackend

    backend = _MemoryBackend(
        db_path="/tmp/eval_memory_recall.db",
        llm_base_url="http://localhost:8080/v1",
        model="ministral",
    )
    user_id = "eval_recall_user"
    fact_ids = _seed_store(backend, user_id)

    case_results = []
    hits_at_k = 0
    reciprocal_ranks = []

    for case in GOLDEN_QUERIES:
        target_text = case["must_retrieve"]
        target_id = fact_ids.get(target_text)
        k = case["k"]

        results = backend.search(case["query"], user_id=user_id, limit=k)
        rank = _rank_of(results, target_id) if target_id else None

        hit = rank is not None
        hits_at_k += int(hit)
        reciprocal_ranks.append(1.0 / rank if hit else 0.0)

        case_results.append({
            "query": case["query"],
            "target": target_text,
            "hit": hit,
            "rank": rank,
            "top_results": [r.get("memory", "")[:80] for r in results],
        })

        if verbose:
            status = f"rank {rank}" if hit else "MISS"
            print(f"\n[{case['query']!r}] -> {status}")
            print(f"  target: {target_text}")
            if not hit:
                print(f"  top-{k} returned instead: {[r.get('memory','')[:60] for r in results]}")

    n = len(GOLDEN_QUERIES)
    recall_at_k = hits_at_k / n if n else 1.0
    mrr = sum(reciprocal_ranks) / n if n else 1.0

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_queries": n,
        "distractor_count": DISTRACTOR_COUNT,
        f"recall_at_k": round(recall_at_k, 3),
        "mrr": round(mrr, 3),
        "cases": case_results,
    }

    backend.delete_all(user_id)
    backend._conn.close()
    return summary


def main():
    parser = argparse.ArgumentParser(description="Eval memory recall accuracy against a golden set.")
    parser.add_argument("--verbose", action="store_true", help="Print per-query detail.")
    parser.add_argument("--out", type=str, default=None, help="Optional path to save full JSON results.")
    args = parser.parse_args()

    try:
        summary = run_eval(verbose=args.verbose)
    except Exception as e:
        print(f"ERROR: could not run eval -- is the real embedder/LLM available? ({e})", file=sys.stderr)
        sys.exit(1)

    print("\n" + "=" * 60)
    print(f"Recall eval — {summary['num_queries']} queries, {summary['distractor_count']} distractors")
    print(f"  recall@k: {summary['recall_at_k']}")
    print(f"  MRR:      {summary['mrr']}")
    misses = [c for c in summary["cases"] if not c["hit"]]
    if misses:
        print(f"  MISSED queries: {[m['query'] for m in misses]}")
    print("=" * 60)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2))
        print(f"Full results saved to {out_path}")


if __name__ == "__main__":
    main()
