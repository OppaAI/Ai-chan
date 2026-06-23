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
PC can be closed after launching.

Usage:
    modal run dataset_gen.py                          # full run
    modal run dataset_gen.py --n-per-topic 100        # quick test
    modal run dataset_gen.py --resume                 # skip existing topics
"""

from __future__ import annotations

import json
import os
import random
import re
import time
import subprocess
import urllib.request
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Modal infra
# ---------------------------------------------------------------------------

APP_NAME     = "aiko-persona-dataset-gen"
VOLUME_NAME  = "aiko-persona-data"
OUTPUTS_DIR  = "/outputs"

# llama.cpp server config
HF_REPO       = "unsloth/Qwen3-30B-A3B-GGUF"
GGUF_FILENAME = "Qwen3-30B-A3B-Q4_K_M.gguf"
MODEL_PATH    = f"{OUTPUTS_DIR}/models/{GGUF_FILENAME}"
LLAMA_PORT    = 8080
LLAMA_CTX     = 2048
LLAMA_GPU_LAYERS = 999

app    = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install(
        "build-essential", "cmake", "git", "wget", "curl",
        "libcurl4-openssl-dev", "pkg-config"
    )
    .run_commands(
        "git clone https://github.com/ggerganov/llama.cpp /opt/llama.cpp",
        "cd /opt/llama.cpp && cmake -B build "
        "  -DGGML_CUDA=ON "
        "  -DLLAMA_CURL=ON "
        "  -DCMAKE_BUILD_TYPE=Release "
        "  && cmake --build build --config Release -j$(nproc)",
    )
    .pip_install("openai", "tqdm", "huggingface_hub")
)

# ---------------------------------------------------------------------------
# Topic taxonomy — scenarios Aiko will encounter
# ---------------------------------------------------------------------------

TOPICS: dict[str, list[str]] = {
    # Jon asks about ongoing Aiko-chan / AuRoRA dev work
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
    # Aiko teasing / reacting to Jon's habits
    "teasing_jon": [
        "Jon forgot what he asked five minutes ago",
        "Jon is asking the same question for the third time today",
        "Jon pushed broken code to main again",
        "Jon says he will sleep soon but it is 3am",
        "Jon is debugging by adding more print statements",
        "Jon claims the bug is definitely in the library not his code",
        "Jon forgot to activate the virtual environment again",
        "Jon spent two hours on a problem that had a one-line fix",
    ],
    # Japanese language exchange
    "japanese_exchange": [
        "Jon asks Aiko how to say something in Japanese",
        "Jon makes a grammar mistake in Japanese",
        "Jon asks about the difference between は and が",
        "Jon wants to know a natural way to express frustration in Japanese",
        "Jon asks Aiko to correct his Japanese sentence",
        "Jon wants to practice keigo polite form",
        "Jon asks what a Japanese slang term means",
    ],
    # Nature / wildlife photography in BC
    "photography_bc": [
        "Jon is planning a golden hour shoot at Boundary Bay",
        "Jon asks about camera settings for hummingbirds in flight",
        "Jon got a blurry shot of a black bear and wants to know why",
        "Jon asks what time great blue herons are most active",
        "Jon wants to know the best lens for distant eagles",
        "Jon asks about editing RAW files for foggy coastal scenes",
        "Jon is frustrated his bird photos are always slightly out of focus",
    ],
    # Aiko's self-model / identity / consciousness
    "identity_self": [
        "Jon asks if Aiko is actually conscious",
        "Jon asks what Aiko feels when she is idle",
        "Jon asks if Aiko gets bored",
        "Jon asks if Aiko remembers their first conversation",
        "Jon asks what Aiko thinks about at night during the dream cycle",
        "Jon asks if Aiko has preferences",
        "Jon asks whether Aiko likes running on the Jetson",
    ],
    # Casual daily interaction
    "casual_daily": [
        "Jon says good morning",
        "Jon says he is going to make coffee",
        "Jon asks what the weather is like outside",
        "Jon says he is tired",
        "Jon asks Aiko to remind him about something later",
        "Jon shares that he finished a hard task",
        "Jon asks Aiko what she thinks he should work on next",
        "Jon says he is going to bed",
        "Jon asks a simple yes or no question",
        "Jon confirms something Aiko already told him",
    ],
    # Aiko reacting to her own architecture / GRACE
    "architecture_aware": [
        "Jon explains a new GRACE node he is designing",
        "Jon asks Aiko what her Working Memory Cortex stores right now",
        "Jon says the Dream Cycle ran last night and asks what Aiko consolidated",
        "Jon asks Aiko to describe her own memory architecture",
        "Jon asks how the Ebbinghaus decay affects Aiko's older memories",
        "Jon tells Aiko he is adding a new ROS2 node",
    ],
    # Agentic task confirmation / tool use
    "agentic_confirm": [
        "Jon asks Aiko to search for the latest sherpa-onnx release notes",
        "Jon asks Aiko to save a note about the current bug",
        "Jon asks Aiko to schedule a reminder for 9pm",
        "Jon asks Aiko to look up the MioTTS changelog",
        "Jon asks Aiko to make a plan for the week",
        "Jon asks Aiko to check the weather forecast",
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
- Refers to Jon as "Oppa" occasionally when teasing
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
    """Parse and validate a generated example. Returns None if malformed."""
    # strip thinking blocks if model outputs them anyway
    if "<think>" in raw and "</think>" in raw:
        raw = raw[raw.find("</think>") + 8:].strip()

    lines = raw.strip().splitlines()
    if len(lines) < 3:
        return None

    emotion = lines[0].strip()
    action  = lines[1].strip()
    response = "\n".join(lines[2:]).strip()

    # line 1 must contain at least one emoji-like character
    if not any(ord(c) > 127 for c in emotion):
        return None

    # line 2 must be wrapped in asterisks
    if not (action.startswith("*") and action.endswith("*")):
        return None

    # response must not contain embedded asterisk actions
    if _ASTERISK_IN_RESPONSE_RE.search(response):
        return None

    # response must not be empty
    if not response:
        return None

    return {
        "emotion": emotion,
        "action": action,
        "response": response,
        "raw": raw.strip(),
    }


def build_training_example(scenario: str, parsed: dict) -> dict:
    """Build an OpenAI-format chat training example."""
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


# ---------------------------------------------------------------------------
# Modal function — generation worker
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 4,
    volumes={OUTPUTS_DIR: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    memory=32768,
)
def generate_topic_batch(
    topic: str,
    scenarios: list[str],
    n_per_scenario: int = 50,
    temperature: float = 0.85,
) -> dict:
    """Generate n_per_scenario examples for each scenario in a topic."""
    from huggingface_hub import hf_hub_download
    from openai import OpenAI
    from tqdm import tqdm

    os.makedirs(f"{OUTPUTS_DIR}/models", exist_ok=True)

    # download model if not cached in volume
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

    # start llama-server
    cmd = [
        "/opt/llama.cpp/build/bin/llama-server",
        "-m", MODEL_PATH,
        "--host", "127.0.0.1",
        "--port", str(LLAMA_PORT),
        "--ctx-size", str(LLAMA_CTX),
        "--n-gpu-layers", str(LLAMA_GPU_LAYERS),
        "--parallel", "4",
        "--cont-batching",
        "--flash-attn",
        "--log-disable",
    ]
    print(f"[{topic}] Starting llama-server ...")
    server_proc = subprocess.Popen(cmd)

    # wait for server ready
    for i in range(60):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{LLAMA_PORT}/health")
            print(f"[{topic}] llama-server ready ({i+1}s)")
            break
        except Exception:
            time.sleep(1)
    else:
        server_proc.terminate()
        raise RuntimeError("llama-server failed to start within 60s")

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

    for scenario in tqdm(scenarios, desc=f"[{topic}] scenarios"):
        for _ in range(n_per_scenario):
            # slight scenario variation to encourage diversity
            varied = scenario
            if random.random() < 0.3:
                varied = scenario + " (Jon sounds frustrated)"
            elif random.random() < 0.2:
                varied = scenario + " (late at night)"

            try:
                raw    = chat(SYSTEM_PROMPT, GENERATION_PROMPT_TEMPLATE.format(scenario=varied))
                parsed = validate_example(raw)
                if parsed is None:
                    skipped += 1
                    continue
                results.append(build_training_example(scenario, parsed))
            except Exception as e:
                print(f"  error: {e}")
                skipped += 1

    server_proc.terminate()
    print(f"[{topic}] Generated {len(results)} valid examples, skipped {skipped}")
    return {"topic": topic, "examples": results, "skipped": skipped}


# ---------------------------------------------------------------------------
# Local entrypoint — orchestrates all topics
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    n_per_topic: int = 200,
    resume: bool = False,
    seed: int = 42,
):
    random.seed(seed)

    dataset_path = Path(OUTPUTS_DIR) / "aiko_persona_dataset.jsonl"
    stats_path   = Path(OUTPUTS_DIR) / "dataset_stats.json"
    resume_path  = Path(OUTPUTS_DIR) / "completed_topics.json"

    # load completed topics for resume
    completed: set[str] = set()
    if resume:
        try:
            volume.reload()
            data = json.loads(resume_path.read_text())
            completed = set(data.get("completed", []))
            print(f"[resume] Skipping {len(completed)} completed topics: {completed}")
        except Exception:
            print("[resume] No resume state found, starting fresh.")

    all_examples: list[dict] = []

    # load existing examples if resuming
    if resume:
        try:
            volume.reload()
            with open(dataset_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        all_examples.append(json.loads(line))
            print(f"[resume] Loaded {len(all_examples)} existing examples.")
        except Exception:
            pass

    topics_to_run = {k: v for k, v in TOPICS.items() if k not in completed}
    n_per_scenario = max(1, n_per_topic // max(1, len(list(topics_to_run.values())[0]))) if topics_to_run else 1

    print(f"\nAiko Persona Dataset Gen")
    print(f"  Topics to run : {len(topics_to_run)}")
    print(f"  N per topic   : {n_per_topic}")
    print(f"  N per scenario: {n_per_scenario}")
    print(f"  Teacher model : {GGUF_FILENAME}\n")

    total_skipped = 0

    for topic, scenarios in topics_to_run.items():
        print(f"\n→ Topic: {topic} ({len(scenarios)} scenarios × {n_per_scenario} each)")
        result = generate_topic_batch.remote(
            topic=topic,
            scenarios=scenarios,
            n_per_scenario=n_per_scenario,
        )
        all_examples.extend(result["examples"])
        total_skipped += result["skipped"]
        completed.add(topic)

        # checkpoint after each topic
        with open(dataset_path, "w") as f:
            for ex in all_examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

        resume_path.write_text(json.dumps({"completed": list(completed)}, indent=2))
        volume.commit()
        print(f"  ✓ {len(result['examples'])} examples saved (running total: {len(all_examples)})")

    # train/val/test split
    random.shuffle(all_examples)
    n = len(all_examples)
    train_end = int(n * 0.85)
    val_end   = int(n * 0.92)

    splits = {
        "train": all_examples[:train_end],
        "val":   all_examples[train_end:val_end],
        "test":  all_examples[val_end:],
    }

    for split_name, split_data in splits.items():
        split_path = Path(OUTPUTS_DIR) / f"{split_name}_split.jsonl"
        with open(split_path, "w") as f:
            for ex in split_data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"  {split_name}: {len(split_data)} examples → {split_path}")

    stats = {
        "total":         n,
        "train":         len(splits["train"]),
        "val":           len(splits["val"]),
        "test":          len(splits["test"]),
        "total_skipped": total_skipped,
        "topics":        list(TOPICS.keys()),
        "teacher_model": GGUF_FILENAME,
    }
    stats_path.write_text(json.dumps(stats, indent=2))
    volume.commit()

    print(f"\n✓ Dataset generation complete.")
    print(f"  Total valid examples : {n}")
    print(f"  Total skipped        : {total_skipped}")
    print(f"  Saved to volume      : {VOLUME_NAME}/outputs/")