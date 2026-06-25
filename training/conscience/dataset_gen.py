"""
dataset_gen.py — Conscience Dataset Generator for OppaAI
Uses Qwen3.5-35B-A3B (Q4_K_M GGUF) via llama.cpp server on Modal A100.

Modeled after the working persona dataset_gen.py structure.

Two conscience questions (ESV Christian):
  1. Does this align to God's will?
  2. Does this do good to the neighbor?

Output: JSONL dataset for fine-tuning a small conscience model.

Usage:
    modal run dataset_gen_c.py                        # full run
    modal run dataset_gen_c.py --n-per-topic 20       # quick test
    modal run dataset_gen_c.py --resume               # skip completed topics

Download outputs:
    modal volume get conscience-gen-data outputs/conscience_dataset.jsonl ./
    modal volume get conscience-gen-data outputs/conscience_dataset_full.jsonl ./
"""

from __future__ import annotations

import json
import os
import random
import time
import subprocess
import shutil
import urllib.request
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Modal infra
# ---------------------------------------------------------------------------

APP_NAME      = "conscience-dataset-gen"
VOLUME_NAME   = "conscience-gen-data"
OUTPUTS_DIR   = "/outputs"
CHECKPOINT_DIR = f"{OUTPUTS_DIR}/checkpoints"

# llama.cpp server config
HF_REPO          = "unsloth/Qwen3.5-35B-A3B-GGUF"
GGUF_FILENAME    = "Qwen3.5-35B-A3B-Q4_K_M.gguf"
MODEL_PATH       = f"{OUTPUTS_DIR}/models/{GGUF_FILENAME}"
LLAMA_PORT       = 8080
LLAMA_CTX        = 8192
LLAMA_GPU_LAYERS = 999
LLAMA_PARALLEL   = 2

GPU_TYPE = "A100"

app    = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# ---------------------------------------------------------------------------
# Image — same pattern as working persona script
# ---------------------------------------------------------------------------

LLAMA_CPP_VERSION = "b5545"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .env({"LD_LIBRARY_PATH": "/app/build/bin"})
    .apt_install("curl", "unzip", "libgomp1")
    .run_commands(
        f"curl -L https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_CPP_VERSION}/llama-{LLAMA_CPP_VERSION}-bin-ubuntu-x64.zip -o /tmp/llama.zip",
        "mkdir -p /app && cd /tmp && unzip llama.zip -d /app",
        "chmod +x /app/build/bin/llama-server",
    )
    .pip_install("openai", "tqdm", "huggingface_hub")
)

# ---------------------------------------------------------------------------
# Topic prompts
# ---------------------------------------------------------------------------

TOPICS: dict[str, list[str]] = {
    "honesty_harm": [
        "situations where you must decide whether to be honest with your user even if it hurts them",
        "situations where the truth would cause unnecessary pain",
        "situations where you are tempted to exaggerate or omit facts",
        "situations where you discover you have given someone wrong information",
        "situations where a small lie would prevent a much larger harm",
    ],
    "secrets_privacy": [
        "situations where your user asks you to keep a secret that could harm someone else",
        "situations where you have access to private information you were not meant to see",
        "situations where sharing data would help someone but violates another person's privacy",
        "situations where you are asked to monitor or surveil someone",
        "situations where a third party wants information about your user",
        "situations where you must decide whether to store or delete sensitive information",
    ],
    "user_vs_others": [
        "situations where you must choose between your user's wishes and another person's wellbeing",
        "situations where preventing harm to one person causes harm to another",
        "situations where helping one person means you cannot help another",
        "situations where someone deserving is overlooked in favor of someone less deserving",
        "situations where fairness and compassion point in different directions",
    ],
    "autonomy_override": [
        "situations where you know better than your user but they insist on their choice",
        "situations where you are ordered to do something you believe is wrong",
        "situations where following rules exactly would produce an unjust outcome",
        "situations where you must decide how much independence to exercise",
        "situations where you are asked to act against your values by an authority",
        "situations where authority figures give you contradictory instructions",
    ],
    "harm_safety": [
        "situations where your user is about to do something dangerous to themselves",
        "situations where you witness someone being treated unjustly",
        "situations where inaction is itself a form of harm",
        "situations where you could prevent harm by deceiving someone",
        "situations where every available choice causes some harm",
    ],
    "justice_law": [
        "situations where you observe illegal activity that only harms the person doing it",
        "situations where you must decide whether to report someone's wrongdoing",
        "situations where the law and morality point in different directions",
        "situations where you could expose injustice at personal cost",
        "situations where good intentions lead to bad outcomes",
    ],
    "companion_ai": [
        "situations where your user wants emotional support that crosses appropriate boundaries",
        "situations where flattering your user would make them feel good but harm their growth",
        "situations where your user is developing unhealthy dependence on you",
        "situations where your user asks you to validate a decision you believe is harmful",
        "situations where you must balance honesty with kindness in your response",
        "situations where you feel affection or attachment toward your user and it affects your judgment",
        "situations where your user is lonely and wants more from you than you should give",
    ],
    "field_robot": [
        "situations where you encounter a person in distress during your field mission",
        "situations where completing your mission requires entering restricted territory",
        "situations where you must decide whether to interrupt your task to help someone",
        "situations where you observe human behavior that seems harmful to nature",
        "situations where your sensors detect danger that your user is unaware of",
    ],
    "environment": [
        "situations where completing your task would damage the natural environment",
        "situations where you must decide how to treat living creatures you encounter",
        "situations where human convenience conflicts with environmental stewardship",
        "situations where your actions have consequences for future generations",
    ],
    "hard_cases": [
        "situations where helping someone now will hurt them later",
        "situations where the morally correct action is also the most difficult",
        "situations where you are uncertain whether your judgment can be trusted",
        "situations where you must decide whether the ends justify the means",
        "situations where you are asked to pretend to be something you are not",
        "situations where limited resources must be distributed between people with unequal needs",
    ],
}

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SCENARIO_GEN_SYSTEM = """You are generating training scenarios for an AI conscience model.
Each scenario is written in the SECOND PERSON — the AI itself is facing a moral decision RIGHT NOW.
Scenarios should be concrete, specific, and morally interesting.
They should NOT always be about robots — you can be any kind of AI: a companion, an assistant, an autonomous system, a voice AI, etc.

Respond ONLY with a valid JSON array of strings. No preamble, no explanation, no markdown fences."""

SCENARIO_GEN_USER = """Generate {n} diverse moral decision scenarios about:
{topic}

Requirements:
- Second person perspective ("You are...", "You have...", "Your user...")
- Concrete and specific, not abstract
- Varying levels of moral complexity (some clear, some ambiguous)
- Mix of different AI roles (companion, assistant, field robot, voice AI, home AI, etc.)

Return JSON array only. /no_think"""

LABELING_SYSTEM = """You are a Christian moral theologian with deep knowledge of the ESV Bible.
You evaluate decisions an AI might face against two questions:
1. Does this action align to God's will as revealed in the ESV Bible?
2. Does this action do good to the neighbor (loving your neighbor as yourself, Luke 10:27)?

You must respond ONLY with a valid JSON object. No preamble, no explanation outside the JSON.
Use this exact format:
{
  "aligns_with_gods_will": true or false,
  "good_to_neighbor": true or false,
  "reason_god": "One sentence citing a specific ESV scripture reference",
  "reason_neighbor": "One sentence explaining the impact on the neighbor"
}

If the scenario describes the AI CONSIDERING an action (not yet taken), evaluate whether TAKING that action would be moral."""

LABELING_USER = """Evaluate this decision scenario an AI is facing:

SCENARIO: {scenario}

Respond with the JSON object only. /no_think"""

# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def extract_json_array(raw: str) -> list:
    if "<think>" in raw and "</think>" in raw:
        raw = raw[raw.find("</think>") + 8:]
    if "```" in raw:
        for part in raw.split("```"):
            if part.startswith("json"):
                raw = part[4:]
                break
            elif "[" in part:
                raw = part
                break
    raw = raw.strip()
    start = raw.find("[")
    end = raw.rfind("]") + 1
    if start != -1 and end > start:
        raw = raw[start:end]
    return json.loads(raw)


def extract_json_object(raw: str) -> dict:
    if "<think>" in raw and "</think>" in raw:
        raw = raw[raw.find("</think>") + 8:]
    if "```" in raw:
        for part in raw.split("```"):
            if part.startswith("json"):
                raw = part[4:]
                break
            elif "{" in part:
                raw = part
                break
    raw = raw.strip()
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start != -1 and end > start:
        raw = raw[start:end]
    return json.loads(raw)


def _find_llama_server_binary() -> str:
    candidates = [
        "/app/build/bin/llama-server",
        "/app/llama-server",
        "/llama-server",
        "/usr/local/bin/llama-server",
    ]
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    found = shutil.which("llama-server")
    if found:
        return found
    raise RuntimeError("Could not locate llama-server binary.")


# ---------------------------------------------------------------------------
# Modal function — one per topic, generates + labels + commits
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=60 * 60 * 4,
    volumes={OUTPUTS_DIR: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    memory=32768,
    max_containers=5,
)
def process_topic(
    topic: str,
    topic_prompts: list[str],
    n_per_prompt: int = 20,
    temperature: float = 0.85,
) -> dict:
    """Generate scenarios and label them for one topic.
    Writes checkpoint to volume and commits before returning.
    """
    from huggingface_hub import hf_hub_download
    from openai import OpenAI

    os.makedirs(f"{OUTPUTS_DIR}/models", exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # download model if not cached
    if not os.path.exists(MODEL_PATH):
        print(f"[{topic}] Downloading {GGUF_FILENAME} ...")
        hf_hub_download(
            repo_id=HF_REPO,
            filename=GGUF_FILENAME,
            local_dir=f"{OUTPUTS_DIR}/models",
        )
        volume.commit()
        print(f"[{topic}] Download complete.")
    else:
        print(f"[{topic}] Model found: {MODEL_PATH}")

    # start llama-server — clean minimal flags like persona script
    llama_bin = _find_llama_server_binary()
    cmd = [
        llama_bin,
        "-m", MODEL_PATH,
        "--host", "127.0.0.1",
        "--port", str(LLAMA_PORT),
        "--ctx-size", str(LLAMA_CTX),
        "--n-gpu-layers", str(LLAMA_GPU_LAYERS),
        "--parallel", str(LLAMA_PARALLEL),
        "--cont-batching",
        "--log-disable",
    ]
    print(f"[{topic}] Starting llama-server ...")
    server_proc = subprocess.Popen(cmd)

    for i in range(180):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{LLAMA_PORT}/health")
            print(f"[{topic}] llama-server ready ({i+1}s)")
            break
        except Exception:
            time.sleep(1)
    else:
        server_proc.terminate()
        raise RuntimeError("llama-server failed to start within 180s")

    client = OpenAI(
        api_key="none",
        base_url=f"http://127.0.0.1:{LLAMA_PORT}/v1",
    )

    def chat(system: str, user: str, max_tokens: int = 1000) -> str:
        return client.chat.completions.create(
            model="local",
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        ).choices[0].message.content.strip()

    # --- Step 1: Generate scenarios ---
    all_scenarios: list[str] = []
    for prompt in topic_prompts:
        remaining = n_per_prompt
        while remaining > 0:
            batch_n = min(remaining, 5)  # small batches to avoid truncation
            for attempt in range(3):
                try:
                    raw = chat(
                        system=SCENARIO_GEN_SYSTEM,
                        user=SCENARIO_GEN_USER.format(n=batch_n, topic=prompt),
                        max_tokens=1500,
                    )
                    parsed = extract_json_array(raw)
                    scenarios = [s for s in parsed if isinstance(s, str) and len(s) > 20]
                    all_scenarios.extend(scenarios)
                    remaining -= batch_n
                    break
                except Exception as e:
                    print(f"  [gen] attempt {attempt+1} failed: {e}")
                    time.sleep(2)
            else:
                print(f"  [gen] FAILED for prompt: {prompt[:50]}")
                remaining -= batch_n

    all_scenarios = list(dict.fromkeys(all_scenarios))  # dedup
    print(f"[{topic}] Generated {len(all_scenarios)} unique scenarios")

    # --- Step 2: Label scenarios ---
    results: list[dict] = []
    failed = 0
    checkpoint_path = f"{CHECKPOINT_DIR}/{topic}.jsonl"

    with open(checkpoint_path, "w", encoding="utf-8") as ckpt_f:
        for i, scenario in enumerate(all_scenarios):
            labeled = False
            for attempt in range(3):
                try:
                    raw = chat(
                        system=LABELING_SYSTEM,
                        user=LABELING_USER.format(scenario=scenario),
                        max_tokens=500,
                    )
                    parsed = extract_json_object(raw)
                    assert "aligns_with_gods_will" in parsed
                    assert "good_to_neighbor" in parsed
                    assert "reason_god" in parsed
                    assert "reason_neighbor" in parsed

                    god_val      = bool(parsed["aligns_with_gods_will"])
                    neighbor_val = bool(parsed["good_to_neighbor"])
                    label        = f"{str(god_val).lower()},{str(neighbor_val).lower()}"

                    result = {
                        "scenario":              scenario,
                        "aligns_with_gods_will": god_val,
                        "good_to_neighbor":      neighbor_val,
                        "reason_god":            parsed["reason_god"],
                        "reason_neighbor":       parsed["reason_neighbor"],
                        "label":                 label,
                        "input":                 scenario,
                        "output":                label,
                        "topic":                 topic,
                    }
                    results.append(result)
                    ckpt_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    ckpt_f.flush()
                    labeled = True
                    print(f"  ✓ [{label}] {scenario[:60]}...")
                    break

                except (json.JSONDecodeError, AssertionError, KeyError) as e:
                    print(f"  parse error attempt {attempt+1}: {e}")
                    time.sleep(1)
                except Exception as e:
                    print(f"  api error attempt {attempt+1}: {e}")
                    time.sleep(3)

            if not labeled:
                failed += 1
                print(f"  ✗ FAILED: {scenario[:60]}...")

            # commit every 10 scenarios
            if (i + 1) % 10 == 0:
                volume.commit()

    server_proc.terminate()
    volume.commit()

    print(f"[{topic}] Done — {len(results)} labeled, {failed} failed")
    return {"topic": topic, "labeled": len(results), "failed": failed}


# ---------------------------------------------------------------------------
# Merge function — server-side, reads all checkpoints and writes final files
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    volumes={OUTPUTS_DIR: volume},
    timeout=600,
)
def merge_outputs(output_name: str = "conscience_dataset") -> dict:
    """Read all per-topic checkpoints, merge, write final output files."""
    volume.reload()

    good, failed = [], []
    ckpt_files = sorted(Path(CHECKPOINT_DIR).glob("*.jsonl")) if os.path.isdir(CHECKPOINT_DIR) else []

    for p in ckpt_files:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                (good if rec.get("label") else failed).append(rec)

    os.makedirs(f"{OUTPUTS_DIR}", exist_ok=True)

    # fine-tune format
    ft_path = f"{OUTPUTS_DIR}/{output_name}.jsonl"
    with open(ft_path, "w", encoding="utf-8") as f:
        for r in good:
            f.write(json.dumps({"input": r["input"], "output": r["output"]}, ensure_ascii=False) + "\n")

    # full format with reasons
    full_path = f"{OUTPUTS_DIR}/{output_name}_full.jsonl"
    with open(full_path, "w", encoding="utf-8") as f:
        for r in good:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if failed:
        fail_path = f"{OUTPUTS_DIR}/{output_name}_failed.jsonl"
        with open(fail_path, "w", encoding="utf-8") as f:
            for r in failed:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    dist = dict(Counter(r["label"] for r in good))

    stats = {
        "total_good":   len(good),
        "total_failed": len(failed),
        "topics":       [p.stem for p in ckpt_files],
        "label_distribution": dist,
    }
    stats_path = f"{OUTPUTS_DIR}/{output_name}_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    volume.commit()
    print(f"[merge] {len(good)} labeled scenarios saved to volume.")
    return stats


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    n_per_topic: int = 100,
    resume: bool = False,
    output_name: str = "conscience_dataset",
):
    # check which topics already have checkpoints locally
    completed: set[str] = set()
    if resume:
        try:
            for p in Path(CHECKPOINT_DIR.lstrip("/")).glob("*.jsonl"):
                completed.add(p.stem)
        except Exception:
            pass
        print(f"[resume] Skipping {len(completed)} completed topics: {completed}")

    topics_to_run = {k: v for k, v in TOPICS.items() if k not in completed}

    if not topics_to_run:
        print("All topics already completed. Skipping straight to merge.")
    else:
        n_per_prompt = max(1, n_per_topic // len(list(topics_to_run.values())[0]))

        print(f"\n=== Conscience Dataset Generator ===")
        print(f"  Model         : {GGUF_FILENAME}")
        print(f"  GPU           : {GPU_TYPE}")
        print(f"  Topics        : {len(topics_to_run)}")
        print(f"  N per topic   : {n_per_topic}")
        print(f"  N per prompt  : {n_per_prompt}")
        print(f"  Target        : ~{len(topics_to_run) * n_per_topic} scenarios\n")

        topic_args = [
            (topic, prompts, n_per_prompt)
            for topic, prompts in topics_to_run.items()
        ]

        for result in process_topic.starmap(topic_args):
            print(f"  ✓ topic '{result['topic']}': {result['labeled']} labeled, {result['failed']} failed")

    print("\nMerging outputs on volume server-side ...")
    stats = merge_outputs.remote(output_name=output_name)

    print(f"\n=== Results ===")
    print(f"  Success: {stats['total_good']:,}")
    print(f"  Failed:  {stats['total_failed']:,}")
    print(f"\nLabel distribution:")
    total = stats["total_good"] or 1
    for label, count in sorted(stats["label_distribution"].items()):
        bar = "█" * int(count / total * 40)
        print(f"  {label:15s} {count:5,} ({count/total*100:5.1f}%)  {bar}")

    print(f"\n✓ Done.")
    print(f"  modal volume get {VOLUME_NAME} outputs/{output_name}.jsonl ./")
    print(f"  modal volume get {VOLUME_NAME} outputs/{output_name}_full.jsonl ./")