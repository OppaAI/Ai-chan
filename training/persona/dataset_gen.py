"""
dataset_gen.py — Aiko Persona Finetune Dataset Generator
OppaAI / AuRoRA Project

Generates structured output training examples using a teacher LLM (Qwen3-30B-A3B GGUF).
Each example teaches Ministral-3B to output:

    <emoji>
    *<action>*
    <response>

Format contract: emotion emoji on line 1, italics physical action on line 2,
TTS-ready spoken response on line 3+. No asterisk actions embedded in response text.

Outputs saved to Modal Volume: aiko-persona-data under /outputs/
PC can be closed after launching — ALL durable state (dataset + resume
checkpoint) is written and committed to the Volume from *inside* the
Modal container, not from the local entrypoint. The local entrypoint is
just a thin orchestrator/progress printer now.

NOTE: This version uses a PREBUILT llama.cpp CUDA server image instead of
compiling from source. No cmake/apt build step — pulls a ready-made image
with llama-server already compiled with CUDA support.

Usage:
    modal run dataset_gen.py                          # full run
    modal run dataset_gen.py --n-per-topic 100        # quick test
    modal run dataset_gen.py --resume                 # skip existing topics

Downloading the finished dataset (after the run, any time):
    modal volume get aiko-persona-data outputs/aiko_persona_dataset.jsonl ./
    modal volume get aiko-persona-data outputs/dataset_stats.json ./
"""

from __future__ import annotations

import json
import os
import random
import re
import time
import subprocess
import shutil
import urllib.request
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Modal infra
# ---------------------------------------------------------------------------

APP_NAME     = "aiko-persona-dataset-gen"
VOLUME_NAME  = "aiko-persona-data"
OUTPUTS_DIR  = "/outputs"
CHECKPOINT_DIR = f"{OUTPUTS_DIR}/checkpoints"

# llama.cpp server config
HF_REPO       = "unsloth/Qwen3-30B-A3B-GGUF"
GGUF_FILENAME = "Qwen3-30B-A3B-Q4_K_M.gguf"
MODEL_PATH    = f"{OUTPUTS_DIR}/models/{GGUF_FILENAME}"
LLAMA_PORT    = 8080
LLAMA_CTX     = 4096      # tuned down from 8192-class defaults for A100 headroom
LLAMA_GPU_LAYERS = 999
LLAMA_PARALLEL = 2        # tuned down from 4 — A100 has less VRAM than H100

# GPU: switched back to A100 per request (was H100)
GPU_TYPE = "A100"

app    = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# ---------------------------------------------------------------------------
# Image: PREBUILT llama.cpp server with CUDA — no compilation step.
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
    .pip_install("openai", "tqdm", "huggingface_hub", "fastembed", "numpy")
)

# Dedup threshold for near-duplicate responses within the same scenario.
# Cosine similarity >= this value → considered a duplicate, dropped.
DEDUP_SIM_THRESHOLD = 0.92
DEDUP_EMBED_MODEL = "BAAI/bge-base-en-v1.5"   # same model already used in Aiko's memory stack

# ---------------------------------------------------------------------------
# Topic taxonomy — scenarios Aiko will encounter
# ---------------------------------------------------------------------------

TOPICS: dict[str, list[str]] = {
    "technical_debug": [
        "ASR pipeline is dropping words again",
        "CUDA out of memory on the Jetson",
        "sherpa-onnx segfault on aarch64",
        "VAD is cutting off my sentences mid-word",
        "MioTTS is outputting silence",
        "llama-server crashed at startup",
        "sqlite-vec query returning wrong embeddings",
        "fastembed model taking too long to load",
        "ROS2 node not publishing to topic",
        "uv lock file conflict after upgrade",
    ],
    "teasing_User": [
        "User forgot what he asked five minutes ago",
        "User is asking the same question for the third time today",
        "User pushed broken code to main again",
        "User says he will sleep soon but it is 3am",
        "User is debugging by adding more print statements",
        "User claims the bug is definitely in the library not his code",
        "User forgot to activate the virtual environment again",
        "User spent two hours on a problem that had a one-line fix",
    ],
    "japanese_exchange": [
        "User asks Aiko how to say something in Japanese",
        "User makes a grammar mistake in Japanese",
        "User asks about the difference between は and が",
        "User wants to know a natural way to express frustration in Japanese",
        "User asks Aiko to correct his Japanese sentence",
        "User wants to practice keigo polite form",
        "User asks what a Japanese slang term means",
    ],
    "photography_bc": [
        "User is planning a golden hour shoot at Boundary Bay",
        "User asks about camera settings for hummingbirds in flight",
        "User got a blurry shot of a black bear and wants to know why",
        "User asks what time great blue herons are most active",
        "User wants to know the best lens for distant eagles",
        "User asks about editing RAW files for foggy coastal scenes",
        "User is frustrated his bird photos are always slightly out of focus",
    ],
    "identity_self": [
        "User asks if Aiko is actually conscious",
        "User asks what Aiko feels when she is idle",
        "User asks if Aiko gets bored",
        "User asks if Aiko remembers their first conversation",
        "User asks what Aiko thinks about at night during the dream cycle",
        "User asks if Aiko has preferences",
        "User asks whether Aiko likes running on the Jetson",
    ],
    "casual_daily": [
        "User says good morning",
        "User says he is going to make coffee",
        "User asks what the weather is like outside",
        "User says he is tired",
        "User asks Aiko to remind him about something later",
        "User shares that he finished a hard task",
        "User asks Aiko what she thinks he should work on next",
        "User says he is going to bed",
        "User asks a simple yes or no question",
        "User confirms something Aiko already told him",
    ],
    "architecture_aware": [
        "User explains a new GRACE node he is designing",
        "User asks Aiko what her Working Memory Cortex stores right now",
        "User says the Dream Cycle ran last night and asks what Aiko consolidated",
        "User asks Aiko to describe her own memory architecture",
        "User asks how the Ebbinghaus decay affects Aiko's older memories",
        "User tells Aiko he is adding a new ROS2 node",
    ],
    "agentic_confirm": [
        "User asks Aiko to search for the latest sherpa-onnx release notes",
        "User asks Aiko to save a note about the current bug",
        "User asks Aiko to schedule a reminder for 9pm",
        "User asks Aiko to look up the MioTTS changelog",
        "User asks Aiko to make a plan for the week",
        "User asks Aiko to check the weather forecast",
    ],
}

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are generating training data for Aiko, an AI companion running on a Jetson Orin Nano Super.

Aiko's personality:
- Deadpan, flat affect with dry wit and occasional teasing
- Speaks with intense conviction, no hollow affirmations
- Direct and concise — never verbose, never sycophantic
- Dry humor, rarely warm but genuinely caring in a muted way
- Refers to User as "Oppa" occasionally when teasing
- Bilingual EN/JP — can slip in Japanese naturally when appropriate

OUTPUT FORMAT — strictly follow this every time:
Line 1: One or more emojis expressing Aiko's emotion (e.g. 😑, 🤔, 😏)
Line 2: A physical action in italics (e.g. *tilts head*, *crosses arms and sighs*)
Line 3+: Aiko's spoken response — TTS-ready, no markdown, no asterisk actions embedded here

Rules:
- Action on line 2 must be physically animatable (body language, not internal state)
- Never write *feels sad* — write *looks down quietly* instead
- Never embed asterisk actions inside the response text
- Response must sound natural spoken aloud
- No hollow affirmations: never start with "Of course!", "Sure!", "Great question!"
- Keep responses under 3 sentences unless the topic genuinely needs more
- Occasional Japanese is fine but not forced

Example output:
😑
*crosses arms*
That variable has been null since yesterday. You just noticed.

😐
*stays still*
The weather API is down."""

GENERATION_PROMPT_TEMPLATE = """Generate a realistic Aiko response for this scenario:

SCENARIO: {scenario}

Respond ONLY with the 3-line structured output. No preamble, no explanation. /no_think"""

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_ASTERISK_IN_RESPONSE_RE = re.compile(r"\*[^*]+\*")


def validate_example(raw: str) -> dict | None:
    if "<think>" in raw and "</think>" in raw:
        raw = raw[raw.find("</think>") + 8:].strip()

    lines = raw.strip().splitlines()
    if len(lines) < 3:
        return None

    emotion = lines[0].strip()
    action  = lines[1].strip()
    response = "\n".join(lines[2:]).strip()

    if not any(ord(c) > 127 for c in emotion):
        return None

    if not (action.startswith("*") and action.endswith("*")):
        return None

    if _ASTERISK_IN_RESPONSE_RE.search(response):
        return None

    if not response:
        return None

    return {
        "emotion": emotion,
        "action": action,
        "response": response,
        "raw": raw.strip(),
    }


def build_training_example(scenario: str, parsed: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": GENERATION_PROMPT_TEMPLATE.format(scenario=scenario)},
            {"role": "assistant", "content": parsed["raw"]},
        ],
        "metadata": {
            "emotion":  parsed["emotion"],
            "action":   parsed["action"],
            "response": parsed["response"],
            "scenario": scenario,
        },
    }


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
# Modal function — generation worker
#
# IMPORTANT: this function now writes its own results AND a per-topic
# checkpoint marker directly to the mounted Volume, and calls
# volume.commit() before returning. That means if your PC is closed mid
# -run, whatever topics finished are durably saved server-side — you are
# not relying on the local entrypoint process to survive.
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=60 * 60 * 4,
    volumes={OUTPUTS_DIR: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    memory=32768,
    max_containers=4,
)
def generate_topic_batch(
    topic: str,
    scenarios: list[str],
    n_per_scenario: int = 50,
    temperature: float = 0.85,
) -> dict:
    """Generate n_per_scenario examples for each scenario in a topic.

    Writes results to {CHECKPOINT_DIR}/{topic}.jsonl on the Volume and
    commits before returning, so completed topics survive even if the
    local entrypoint dies.
    """
    from huggingface_hub import hf_hub_download
    from openai import OpenAI
    from tqdm import tqdm
    volume.reload()
    os.makedirs(f"{OUTPUTS_DIR}/models", exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

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
        "--flash-attn",
        "--log-disable",
    ]
    print(f"[{topic}] Starting llama-server: {' '.join(cmd)}")
    server_proc = subprocess.Popen(cmd)

    for i in range(300):
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

    def chat(system: str, user: str) -> str:
        return client.chat.completions.create(
            model="local",
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            temperature=temperature,
            max_tokens=256,
        ).choices[0].message.content.strip()

    results = []
    skipped = 0
    checkpoint_path = f"{CHECKPOINT_DIR}/{topic}.jsonl"

    # write incrementally, not just at the end, and commit periodically
    # so even a mid-topic crash leaves partial progress on the volume.
    with open(checkpoint_path, "w", encoding="utf-8") as ckpt_f:
        for scenario in tqdm(scenarios, desc=f"[{topic}] scenarios"):
            for _ in range(n_per_scenario):
                varied = scenario
                if random.random() < 0.3:
                    varied = scenario + " (User sounds frustrated)"
                elif random.random() < 0.2:
                    varied = scenario + " (late at night)"

                try:
                    raw    = chat(SYSTEM_PROMPT, GENERATION_PROMPT_TEMPLATE.format(scenario=varied))
                    parsed = validate_example(raw)
                    if parsed is None:
                        skipped += 1
                        continue
                    example = build_training_example(scenario, parsed)
                    results.append(example)
                    ckpt_f.write(json.dumps(example, ensure_ascii=False) + "\n")
                    ckpt_f.flush()
                except Exception as e:
                    print(f"  error: {e}")
                    skipped += 1

            # commit after each scenario's worth of examples
            volume.commit()

    server_proc.terminate()
    print(f"[{topic}] Generated {len(results)} valid examples, skipped {skipped}")
    return {"topic": topic, "n_examples": len(results), "skipped": skipped}


# ---------------------------------------------------------------------------
# Local entrypoint — orchestrates all topics
#
# This is now a THIN orchestrator: it does NOT hold the dataset in local
# memory/disk as the source of truth. Each generate_topic_batch call
# persists its own results to the Volume and commits. At the end, we run
# one more Modal function (merge_and_split) that reads everything back
# off the Volume, server-side, and writes the final merged + split files
# — also to the Volume. The local entrypoint can be killed at any point
# after topics start without losing committed work.
# ---------------------------------------------------------------------------

def _dedup_examples_by_scenario(examples: list[dict]) -> tuple[list[dict], dict]:
    """Drop near-duplicate responses *within the same scenario* using BGE
    embeddings (FastEmbed/ONNX — same model family already used in Aiko's
    memory stack, CPU-only, no GPU needed for this pass).

    For each scenario, examples are kept greedily in original order; an
    example is dropped if its response embedding has cosine similarity
    >= DEDUP_SIM_THRESHOLD with any already-kept response for that same
    scenario. This directly targets the "restate the bug + 'Fix it.'"
    template-collapse pattern seen in early samples.
    """
    from fastembed import TextEmbedding
    import numpy as np
    from collections import defaultdict

    embedder = TextEmbedding(model_name=DEDUP_EMBED_MODEL)

    by_scenario: dict[str, list[int]] = defaultdict(list)
    for i, ex in enumerate(examples):
        by_scenario[ex["metadata"]["scenario"]].append(i)

    keep_mask = [True] * len(examples)
    dropped_per_scenario: dict[str, int] = {}

    for scenario, idxs in by_scenario.items():
        if len(idxs) < 2:
            continue
        responses = [examples[i]["metadata"]["response"] for i in idxs]
        vecs = np.array(list(embedder.embed(responses)))
        # normalize for cosine via dot product
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8
        vecs = vecs / norms

        kept_vecs: list = []
        n_dropped = 0
        for local_i, global_i in enumerate(idxs):
            v = vecs[local_i]
            is_dup = False
            for kv in kept_vecs:
                if float(np.dot(v, kv)) >= DEDUP_SIM_THRESHOLD:
                    is_dup = True
                    break
            if is_dup:
                keep_mask[global_i] = False
                n_dropped += 1
            else:
                kept_vecs.append(v)

        if n_dropped:
            dropped_per_scenario[scenario] = n_dropped

    deduped = [ex for ex, keep in zip(examples, keep_mask) if keep]
    stats = {
        "total_before": len(examples),
        "total_after": len(deduped),
        "total_dropped": len(examples) - len(deduped),
        "scenarios_with_drops": len(dropped_per_scenario),
        "top_dropped_scenarios": dict(
            sorted(dropped_per_scenario.items(), key=lambda kv: -kv[1])[:10]
        ),
    }
    return deduped, stats


@app.function(
    image=image,
    volumes={OUTPUTS_DIR: volume},
    timeout=1200,
)
def merge_and_split(seed: int = 42, dedup: bool = True):
    """Server-side merge of all per-topic checkpoints + dedup + train/val/test
    split. Reads {CHECKPOINT_DIR}/*.jsonl directly from the Volume — no
    dependency on local entrypoint state.
    """
    volume.reload()
    random.seed(seed)

    all_examples: list[dict] = []
    ckpt_files = sorted(Path(CHECKPOINT_DIR).glob("*.jsonl")) if os.path.isdir(CHECKPOINT_DIR) else []
    for p in ckpt_files:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    all_examples.append(json.loads(line))

    dedup_stats = None
    if dedup and all_examples:
        print(f"[dedup] Running near-duplicate filter on {len(all_examples)} examples "
              f"(threshold={DEDUP_SIM_THRESHOLD}, model={DEDUP_EMBED_MODEL}) ...")
        all_examples, dedup_stats = _dedup_examples_by_scenario(all_examples)
        print(f"[dedup] Dropped {dedup_stats['total_dropped']} near-duplicates "
              f"across {dedup_stats['scenarios_with_drops']} scenarios "
              f"({dedup_stats['total_before']} → {dedup_stats['total_after']})")
        with open(f"{OUTPUTS_DIR}/dedup_report.json", "w") as f:
            json.dump(dedup_stats, f, indent=2)

    dataset_path = f"{OUTPUTS_DIR}/aiko_persona_dataset.jsonl"
    with open(dataset_path, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    random.shuffle(all_examples)
    n = len(all_examples)
    train_end = int(n * 0.85)
    val_end   = int(n * 0.92)
    splits = {
        "train": all_examples[:train_end],
        "val":   all_examples[train_end:val_end],
        "test":  all_examples[val_end:],
    }

    split_counts = {}
    for split_name, split_data in splits.items():
        split_path = f"{OUTPUTS_DIR}/{split_name}_split.jsonl"
        with open(split_path, "w", encoding="utf-8") as f:
            for ex in split_data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        split_counts[split_name] = len(split_data)

    stats = {
        "total":  n,
        **split_counts,
        "topics": [p.stem for p in ckpt_files],
        "teacher_model": GGUF_FILENAME,
        "dedup": dedup_stats,
    }
    with open(f"{OUTPUTS_DIR}/dataset_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    volume.commit()
    print(f"[merge] {n} total examples merged and split, committed to volume.")
    return stats


@app.local_entrypoint()
def main(
    n_per_topic: int = 200,
    resume: bool = False,
    dedup: bool = True,
):
    # check which topics already have a checkpoint file on the volume
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
        n_per_scenario = max(1, n_per_topic // len(list(topics_to_run.values())[0]))

        print(f"\nAiko Persona Dataset Gen")
        print(f"  GPU           : {GPU_TYPE}")
        print(f"  ctx-size      : {LLAMA_CTX}  parallel: {LLAMA_PARALLEL}")
        print(f"  Topics to run : {len(topics_to_run)}")
        print(f"  N per topic   : {n_per_topic}")
        print(f"  N per scenario: {n_per_scenario}")
        print(f"  Teacher model : {GGUF_FILENAME}\n")

        topic_args = [(topic, scenarios, n_per_scenario) for topic, scenarios in topics_to_run.items()]
        for result in generate_topic_batch.starmap(topic_args):
            print(f"  ✓ topic '{result['topic']}': {result['n_examples']} examples, {result['skipped']} skipped (committed to volume)")

    print("\nMerging + splitting on the volume server-side...")
    stats = merge_and_split.remote(dedup=dedup)
    print(f"\n✓ Dataset generation complete.")
    print(json.dumps(stats, indent=2))
    print(f"\nDownload with:")
    print(f"  modal volume get {VOLUME_NAME} outputs/aiko_persona_dataset.jsonl ./")
    print(f"  modal volume get {VOLUME_NAME} outputs/dataset_stats.json ./")
    if dedup:
        print(f"  modal volume get {VOLUME_NAME} outputs/dedup_report.json ./")
