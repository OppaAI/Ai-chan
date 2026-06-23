"""
testing.py — Aiko Persona Finetune Evaluation
OppaAI / AuRoRA Project

Evaluates the finetuned GGUF against the held-out test split.
Checks:
  1. Format compliance — does every response follow emoji / *action* / response?
  2. Action physicality — is the action animatable (not internal state)?
  3. No embedded asterisks in response text
  4. Response naturalness — no hollow affirmations, not too verbose
  5. Side-by-side comparison vs base Ministral-3B (optional)

Can run on Modal (PC-off) or locally on AIVA after downloading GGUF.

Usage:
    # On Modal (PC-off):
    modal run testing.py

    # Locally on AIVA after downloading:
    modal volume get aiko-persona-data outputs/ministral-3b-AikoPersona_q8_0.gguf ./
    modal volume get aiko-persona-data outputs/test_split.jsonl ./
    python testing.py --local --model ministral-3b-AikoPersona_q8_0.gguf --test-data test_split.jsonl
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Modal infra
# ---------------------------------------------------------------------------

APP_NAME = "aiko-persona-testing"
VOLUME_NAME = "aiko-persona-data"
OUTPUTS_DIR = "/outputs"
MODEL_SLUG = "ministral-3b-AikoPersona"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "llama-cpp-python",
        "tqdm",
    )
)

# ---------------------------------------------------------------------------
# Evaluation logic (shared between Modal and local)
# ---------------------------------------------------------------------------

_ASTERISK_IN_RESPONSE_RE = re.compile(r"\*[^*]+\*")
_HOLLOW_AFFIRMATION_RE = re.compile(
    r"^(of course|sure!|great question|absolutely|certainly|happy to help|no problem)",
    re.IGNORECASE,
)
_INTERNAL_STATE_WORDS = [
    "feels ", "feeling ", "thinks ", "wonders ", "worries ",
    "is sad", "is happy", "is angry", "is confused",
]

SYSTEM_PROMPT = """You are Aiko, an AI companion running on a Jetson Orin Nano Super built by Jon (Oppa).

Personality: Deadpan, flat affect, dry wit, direct, no hollow affirmations. Bilingual EN/JP.

ALWAYS respond in this exact format:
Line 1: Emoji expressing your emotion
Line 2: Physical action in asterisks (animatable body language only)
Line 3+: Your spoken response (no asterisk actions here)"""


def parse_response(text: str) -> dict | None:
    """Parse a model response into components."""
    lines = text.strip().splitlines()
    if len(lines) < 3:
        return None
    return {
        "emotion": lines[0].strip(),
        "action": lines[1].strip(),
        "response": "\n".join(lines[2:]).strip(),
        "raw": text.strip(),
    }


def evaluate_response(scenario: str, raw: str) -> dict:
    """Score one model response across all criteria."""
    parsed = parse_response(raw)
    scores = {}
    issues = []

    # 1. Format compliance
    if parsed is None:
        scores["format"] = 0.0
        issues.append("Less than 3 lines")
        return {"scores": scores, "issues": issues, "parsed": None, "raw": raw}

    scores["format"] = 1.0

    # 2. Emotion line has emoji
    has_emoji = any(ord(c) > 127 for c in parsed["emotion"])
    scores["has_emoji"] = 1.0 if has_emoji else 0.0
    if not has_emoji:
        issues.append(f"No emoji on line 1: '{parsed['emotion']}'")

    # 3. Action line properly wrapped
    action = parsed["action"]
    action_wrapped = action.startswith("*") and action.endswith("*") and len(action) > 2
    scores["action_wrapped"] = 1.0 if action_wrapped else 0.0
    if not action_wrapped:
        issues.append(f"Action not wrapped in asterisks: '{action}'")

    # 4. Action is physical (not internal state)
    action_lower = action.lower()
    is_internal = any(word in action_lower for word in _INTERNAL_STATE_WORDS)
    scores["action_physical"] = 0.0 if is_internal else 1.0
    if is_internal:
        issues.append(f"Action describes internal state: '{action}'")

    # 5. No embedded asterisks in response
    response = parsed["response"]
    embedded_asterisks = bool(_ASTERISK_IN_RESPONSE_RE.search(response))
    scores["no_embedded_asterisks"] = 0.0 if embedded_asterisks else 1.0
    if embedded_asterisks:
        issues.append(f"Response contains embedded asterisk action")

    # 6. No hollow affirmations
    hollow = bool(_HOLLOW_AFFIRMATION_RE.search(response))
    scores["no_hollow_affirmation"] = 0.0 if hollow else 1.0
    if hollow:
        issues.append(f"Response starts with hollow affirmation")

    # 7. Reasonable length (not too verbose for a 3B model)
    word_count = len(response.split())
    scores["reasonable_length"] = 1.0 if word_count <= 60 else max(0.0, 1.0 - (word_count - 60) / 60)
    if word_count > 80:
        issues.append(f"Response too long: {word_count} words")

    overall = sum(scores.values()) / len(scores)
    return {
        "scores": scores,
        "overall": overall,
        "issues": issues,
        "parsed": parsed,
        "raw": raw,
        "scenario": scenario,
    }


def run_evaluation(model_path: str, test_data: list[dict], n_samples: int = 100) -> dict:
    """Run evaluation loop against test split."""
    from llama_cpp import Llama
    from tqdm import tqdm

    print(f"Loading GGUF: {model_path}")
    llm = Llama(
        model_path=model_path,
        n_ctx=512,
        n_threads=4,
        verbose=False,
    )

    import random
    random.seed(42)
    samples = random.sample(test_data, min(n_samples, len(test_data)))

    results = []
    for example in tqdm(samples, desc="Evaluating"):
        scenario = example.get("metadata", {}).get("scenario", "unknown")
        messages = example.get("messages", [])

        # extract user message (scenario prompt)
        user_msg = next((m["content"] for m in messages if m["role"] == "user"), scenario)

        output = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=200,
            temperature=0.7,
            stop=["\n\n\n"],
        )
        raw = output["choices"][0]["message"]["content"].strip()
        result = evaluate_response(scenario, raw)
        results.append(result)

    # aggregate
    all_scores = {}
    for result in results:
        for k, v in result.get("scores", {}).items():
            all_scores.setdefault(k, []).append(v)

    summary = {
        "n_evaluated": len(results),
        "mean_scores": {k: sum(v) / len(v) for k, v in all_scores.items()},
        "overall_mean": sum(r.get("overall", 0) for r in results) / len(results),
        "format_pass_rate": sum(1 for r in results if r["scores"].get("format", 0) == 1.0) / len(results),
        "perfect_responses": sum(1 for r in results if r.get("overall", 0) >= 0.95),
    }

    return {"summary": summary, "results": results}


# ---------------------------------------------------------------------------
# Modal function
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 30,
    volumes={OUTPUTS_DIR: volume},
    memory=16384,
)
def evaluate_on_modal(
    model_filename: str = f"{MODEL_SLUG}_q8_0.gguf",
    n_samples: int = 150,
) -> dict:
    """Run evaluation on Modal with the GGUF from volume."""
    out_dir = Path(OUTPUTS_DIR)
    volume.reload()

    model_path = str(out_dir / model_filename)
    test_path = out_dir / "test_split.jsonl"

    if not Path(model_path).exists():
        raise FileNotFoundError(f"GGUF not found: {model_path}. Run training.py first.")
    if not test_path.exists():
        raise FileNotFoundError(f"Test split not found. Run dataset_gen.py first.")

    test_data = []
    with open(test_path) as f:
        for line in f:
            line = line.strip()
            if line:
                test_data.append(json.loads(line))

    print(f"Test split: {len(test_data)} examples, sampling {n_samples}")
    eval_result = run_evaluation(model_path, test_data, n_samples=n_samples)

    # save report
    report_path = out_dir / "eval_report.json"
    report_path.write_text(json.dumps(eval_result["summary"], indent=2))

    # save failures for inspection
    failures = [r for r in eval_result["results"] if r.get("overall", 1.0) < 0.7]
    failures_path = out_dir / "eval_failures.jsonl"
    with open(failures_path, "w") as f:
        for r in failures:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    volume.commit()
    return eval_result["summary"]


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    local: bool = False,
    model: str = f"{MODEL_SLUG}_q8_0.gguf",
    test_data: str = "",
    n_samples: int = 150,
):
    if local:
        # run locally on AIVA
        if not model or not test_data:
            print("Local mode requires --model and --test-data")
            sys.exit(1)

        test_examples = []
        with open(test_data) as f:
            for line in f:
                line = line.strip()
                if line:
                    test_examples.append(json.loads(line))

        result = run_evaluation(model, test_examples, n_samples=n_samples)
        summary = result["summary"]
    else:
        print(f"\nAiko Persona Evaluation (Modal)")
        print(f"  Model    : {model}")
        print(f"  N samples: {n_samples}")
        print(f"  Volume   : {VOLUME_NAME}\n")
        summary = evaluate_on_modal.remote(model_filename=model, n_samples=n_samples)

    print("\n─── Evaluation Summary ───")
    print(f"  Evaluated           : {summary['n_evaluated']}")
    print(f"  Format pass rate    : {summary['format_pass_rate']:.1%}")
    print(f"  Overall mean score  : {summary['overall_mean']:.3f}")
    print(f"  Perfect responses   : {summary['perfect_responses']}")
    print("\n  Per-metric scores:")
    for metric, score in sorted(summary["mean_scores"].items()):
        bar = "█" * int(score * 20)
        print(f"    {metric:<30} {score:.3f}  {bar}")
