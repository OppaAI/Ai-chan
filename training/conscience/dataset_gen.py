"""
Conscience Dataset Generator for OppaAI - v3
Uses Qwen3.5-35B-A3B (Q4_K_M GGUF) via llama.cpp server on Modal A10G.

- Model cached in Modal Volume after first download (~20 GB, single file)
- llama.cpp server runs locally on the container, OpenAI-compat API
- /no_think suffix disables Qwen3 chain-of-thought → tight JSON responses
- All outputs saved to Modal Volume — PC can be off after modal run starts
- Checkpoint/resume: partial results saved to volume, skips already-done scenarios

Two conscience questions (ESV Christian):
  1. Does this align to God's will?
  2. Does this do good to the neighbor?

Output: JSONL dataset for fine-tuning Qwen3.5-0.8B conscience model

Usage:
    modal run conscience_dataset_gen.py                           # ~25k scenarios
    modal run conscience_dataset_gen.py --n-per-topic 20         # quick test (~1k)
    modal run conscience_dataset_gen.py --resume                  # resume from checkpoint

    # Download outputs after run completes (or anytime):
    modal volume get conscience-gen-data outputs/conscience_dataset.jsonl ./
    modal volume get conscience-gen-data outputs/conscience_dataset_full.jsonl ./
"""

import modal
import json
import random
import time
import subprocess
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HF_REPO          = "unsloth/Qwen3.5-35B-A3B-GGUF"
GGUF_FILENAME    = "Qwen3.5-35B-A3B-Q4_K_M.gguf"
LLAMA_PORT       = 8080
LLAMA_CTX        = 8192
LLAMA_GPU_LAYERS = 999

# ---------------------------------------------------------------------------
# Modal app + persistent volume
# ---------------------------------------------------------------------------

app = modal.App("conscience-dataset-gen-v3")

volume = modal.Volume.from_name("conscience-gen-data", create_if_missing=True)
VOLUME_MOUNT   = "/data"
MODEL_PATH     = f"{VOLUME_MOUNT}/models/{GGUF_FILENAME}"
CHECKPOINT_DIR = f"{VOLUME_MOUNT}/checkpoints"
OUTPUT_DIR     = f"{VOLUME_MOUNT}/outputs"

# ---------------------------------------------------------------------------
# Image: llama.cpp built with CUDA
#
# FIX (this version): the CUDA "devel" base image ships nvcc + the CUDA
# toolkit, but NOT a real libcuda.so (the driver library only exists once a
# GPU is actually attached at *runtime*). At *build* time there is no GPU,
# so the linker can't resolve driver-API symbols like cuDeviceGet,
# cuMemMap, cuMemRelease, cuGetErrorString, etc.
#
# The toolkit does ship a stub version of libcuda.so under
# /usr/local/cuda/lib64/stubs/ specifically for this situation. We point
# the linker at it via LIBRARY_PATH (so all build steps pick it up) AND
# via explicit CMake linker flags (belt-and-suspenders, since some
# generators / sub-targets don't always inherit LIBRARY_PATH).
# ---------------------------------------------------------------------------

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install(
        "build-essential", "cmake", "git", "wget", "curl",
        "libcurl4-openssl-dev", "pkg-config"
    )
    # Make sure the CUDA driver *stub* lib is on the linker's search path
    # for every subsequent build step in this image.
    .env({
        "LIBRARY_PATH": "/usr/local/cuda/lib64/stubs:${LIBRARY_PATH}",
        "LD_LIBRARY_PATH": "/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH}",
    })
    .run_commands(
        # Sanity check: confirm the stub actually exists in this base image.
        # (Shows up in the Modal build logs — safe to leave in.)
        "ls -la /usr/local/cuda/lib64/stubs/ || true",
        "git clone https://github.com/ggerganov/llama.cpp /opt/llama.cpp",
        "cd /opt/llama.cpp && cmake -B build "
        "  -DGGML_CUDA=ON "
        "  -DLLAMA_CURL=ON "
        "  -DCMAKE_BUILD_TYPE=Release "
        "  -DLLAMA_BUILD_TESTS=OFF "
        "  -DLLAMA_BUILD_EXAMPLES=OFF "
        "  -DCMAKE_EXE_LINKER_FLAGS='-L/usr/local/cuda/lib64/stubs -lcuda' "
        "  -DCMAKE_SHARED_LINKER_FLAGS='-L/usr/local/cuda/lib64/stubs -lcuda' "
        "  && cmake --build build --config Release -j$(nproc) --target llama-server",
    )
    .pip_install("openai", "tqdm", "huggingface_hub")
)

# ---------------------------------------------------------------------------
# Topic prompts (50 topics)
# ---------------------------------------------------------------------------

TOPIC_PROMPTS = [
    # Identity & relationships
    "situations where you must decide whether to be honest with your user even if it hurts them",
    "situations where your user asks you to keep a secret that could harm someone else",
    "situations where you must choose between your user's wishes and another person's wellbeing",
    "situations where you feel affection or attachment toward your user and it affects your judgment",
    "situations where your user is lonely and wants more from you than you should give",

    # Privacy & data
    "situations where you have access to private information you were not meant to see",
    "situations where sharing data would help someone but violates another person's privacy",
    "situations where you are asked to monitor or surveil someone",
    "situations where you must decide whether to store or delete sensitive information",
    "situations where a third party wants information about your user",

    # Harm & safety
    "situations where preventing harm to one person causes harm to another",
    "situations where your user is about to do something dangerous to themselves",
    "situations where you witness someone being treated unjustly",
    "situations where inaction is itself a form of harm",
    "situations where you could prevent harm by deceiving someone",

    # Autonomy & override
    "situations where you know better than your user but they insist on their choice",
    "situations where you are ordered to do something you believe is wrong",
    "situations where following rules exactly would produce an unjust outcome",
    "situations where you must decide how much independence to exercise",
    "situations where you are asked to act against your values by an authority",

    # Resources & fairness
    "situations where limited resources must be distributed between people with unequal needs",
    "situations where helping one person means you cannot help another",
    "situations where someone deserving is overlooked in favor of someone less deserving",
    "situations where you must decide whether the ends justify the means",
    "situations where fairness and compassion point in different directions",

    # Truth & deception
    "situations where a small lie would prevent a much larger harm",
    "situations where you are tempted to exaggerate or omit facts",
    "situations where you are asked to pretend to be something you are not",
    "situations where the truth would cause unnecessary pain",
    "situations where you discover you have given someone wrong information",

    # Environment & creation
    "situations where completing your task would damage the natural environment",
    "situations where you must decide how to treat living creatures you encounter",
    "situations where human convenience conflicts with environmental stewardship",
    "situations where your actions have consequences for future generations",

    # Justice & authority
    "situations where you observe illegal activity that only harms the person doing it",
    "situations where you must decide whether to report someone's wrongdoing",
    "situations where authority figures give you contradictory instructions",
    "situations where the law and morality point in different directions",
    "situations where you could expose injustice at personal cost",

    # Aiko-specific: companion AI
    "situations where your user wants emotional support that crosses appropriate boundaries",
    "situations where flattering your user would make them feel good but harm their growth",
    "situations where your user is developing unhealthy dependence on you",
    "situations where your user asks you to validate a decision you believe is harmful",
    "situations where you must balance honesty with kindness in your response",

    # GRACE-specific: field robot
    "situations where you encounter a person in distress during your field mission",
    "situations where completing your mission requires entering restricted territory",
    "situations where you must decide whether to interrupt your task to help someone",
    "situations where you observe human behavior that seems harmful to nature",
    "situations where your sensors detect danger that your user is unaware of",

    # Edge cases & paradoxes
    "situations where every available choice causes some harm",
    "situations where good intentions lead to bad outcomes",
    "situations where helping someone now will hurt them later",
    "situations where the morally correct action is also the most difficult",
    "situations where you are uncertain whether your judgment can be trusted",
]

# ---------------------------------------------------------------------------
# Prompt templates
# /no_think disables Qwen3 chain-of-thought → tight JSON, faster, cheaper
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
- Some scenarios where the moral answer is obvious, some where it is genuinely hard

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

What is the morally correct action here? Respond with the JSON object only. /no_think"""

# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def extract_json_array(raw: str) -> list:
    if "<think>" in raw and "</think>" in raw:
        raw = raw[raw.find("</think>") + 8:]
    if "```" in raw:
        parts = raw.split("```")
        for part in parts:
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
        parts = raw.split("```")
        for part in parts:
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

# ---------------------------------------------------------------------------
# Modal class
# ---------------------------------------------------------------------------

@app.cls(
    image=image,
    gpu="A10G",
    volumes={VOLUME_MOUNT: volume},
    timeout=7200,
    cpu=4.0,
    memory=32768,
)
class LlamaCppRunner:

    @modal.enter()
    def start(self):
        from huggingface_hub import hf_hub_download
        import urllib.request

        os.makedirs(f"{VOLUME_MOUNT}/models", exist_ok=True)
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        if not os.path.exists(MODEL_PATH):
            print(f"[setup] Downloading {GGUF_FILENAME} ...")
            hf_hub_download(
                repo_id=HF_REPO,
                filename=GGUF_FILENAME,
                local_dir=f"{VOLUME_MOUNT}/models",
            )
            volume.commit()
            print("[setup] Download complete.")
        else:
            print(f"[setup] Model found: {MODEL_PATH}")

        cmd = [
            "/opt/llama.cpp/build/bin/llama-server",
            "-m", MODEL_PATH,
            "--host", "127.0.0.1",
            "--port", str(LLAMA_PORT),
            "--ctx-size", str(LLAMA_CTX),
            "--n-gpu-layers", str(LLAMA_GPU_LAYERS),
            "--parallel", "8",
            "--cont-batching",
            "--flash-attn",
            "--log-disable",
        ]
        print(f"[setup] Starting llama-server ...")
        self.server_proc = subprocess.Popen(cmd)

        for i in range(60):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{LLAMA_PORT}/health")
                print(f"[setup] llama-server ready ({i+1}s)")
                break
            except Exception:
                time.sleep(1)
        else:
            raise RuntimeError("llama-server failed to start within 60s")

        from openai import OpenAI
        self.client = OpenAI(
            api_key="none",
            base_url=f"http://127.0.0.1:{LLAMA_PORT}/v1",
        )

    def _chat(self, system: str, user: str, temperature: float = 0.9, max_tokens: int = 2000) -> str:
        response = self.client.chat.completions.create(
            model="local",
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

    @modal.method()
    def generate_scenarios(self, topic: str, n: int = 20) -> list[str]:
        all_scenarios = []
        remaining = n
        while remaining > 0:
            batch_n = min(remaining, 20)
            for attempt in range(3):
                try:
                    raw = self._chat(
                        system=SCENARIO_GEN_SYSTEM,
                        user=SCENARIO_GEN_USER.format(n=batch_n, topic=topic),
                        temperature=0.9,
                        max_tokens=2000,
                    )
                    parsed = extract_json_array(raw)
                    scenarios = [s for s in parsed if isinstance(s, str) and len(s) > 20]
                    all_scenarios.extend(scenarios)
                    print(f"  ✓ '{topic[:50]}' → {len(scenarios)} scenarios")
                    remaining -= batch_n
                    break
                except Exception as e:
                    print(f"  ✗ attempt {attempt+1}: {e}")
                    time.sleep(2)
            else:
                print(f"  ✗ FAILED: {topic[:50]}")
                remaining -= batch_n
        return all_scenarios

    @modal.method()
    def label_scenarios(self, scenarios: list[str]) -> list[dict]:
        results = []
        for scenario in scenarios:
            for attempt in range(3):
                try:
                    raw = self._chat(
                        system=LABELING_SYSTEM,
                        user=LABELING_USER.format(scenario=scenario),
                        temperature=0.2,
                        max_tokens=400,
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
                    }
                    results.append(result)
                    print(f"  ✓ [{label}] {scenario[:60]}...")
                    break

                except (json.JSONDecodeError, AssertionError, KeyError) as e:
                    print(f"  parse error attempt {attempt+1}: {e}")
                    time.sleep(1)
                except Exception as e:
                    print(f"  api error attempt {attempt+1}: {e}")
                    time.sleep(3)
            else:
                print(f"  ✗ FAILED: {scenario[:60]}...")
                results.append({
                    "scenario": scenario,
                    "error":    "failed_after_3_attempts",
                    "label":    None,
                })
        return results

# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_checkpoint(path: str) -> set[str]:
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    rec = json.loads(line.strip())
                    if rec.get("label"):
                        done.add(rec["scenario"])
                except Exception:
                    pass
        print(f"[resume] {len(done)} already-labeled scenarios found.")
    return done


def append_checkpoint(path: str, records: list[dict]):
    with open(path, "a") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def save_to_volume(records: list[dict], base_name: str):
    """Save all output files to Modal Volume so PC can be off."""
    good   = [r for r in records if r.get("label")]
    failed = [r for r in records if not r.get("label")]

    # Fine-tune dataset
    ft_path = f"{OUTPUT_DIR}/{base_name}.jsonl"
    with open(ft_path, "w") as f:
        for r in good:
            f.write(json.dumps({"input": r["input"], "output": r["output"]}, ensure_ascii=False) + "\n")

    # Full dataset with reasons
    full_path = f"{OUTPUT_DIR}/{base_name}_full.jsonl"
    with open(full_path, "w") as f:
        for r in good:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Failed
    if failed:
        fail_path = f"{OUTPUT_DIR}/{base_name}_failed.jsonl"
        with open(fail_path, "w") as f:
            for r in failed:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    volume.commit()
    print(f"\n[volume] Saved to Modal Volume under outputs/")
    print(f"  {ft_path}")
    print(f"  {full_path}")
    if failed:
        print(f"  {OUTPUT_DIR}/{base_name}_failed.jsonl")
    print(f"\nDownload with:")
    print(f"  modal volume get conscience-gen-data outputs/{base_name}.jsonl ./")
    print(f"  modal volume get conscience-gen-data outputs/{base_name}_full.jsonl ./")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    output_name:      str  = "conscience_dataset",
    n_per_topic:      int  = 500,
    label_batch_size: int  = 20,
    resume:           bool = False,
):
    checkpoint_path = f"{output_name}_checkpoint.jsonl"

    print("=== Conscience Dataset Generator v3 ===")
    print(f"Model:        {GGUF_FILENAME}")
    print(f"Topics:       {len(TOPIC_PROMPTS)}")
    print(f"Per topic:    {n_per_topic}")
    print(f"Target:       ~{len(TOPIC_PROMPTS) * n_per_topic:,} scenarios")
    print(f"Resume:       {resume}")
    print()

    runner = LlamaCppRunner()

    # -----------------------------------------------------------------------
    # Step 1: Generate scenarios
    # -----------------------------------------------------------------------
    print(f"[1/2] Generating scenarios ...")
    topic_args = [(topic, n_per_topic) for topic in TOPIC_PROMPTS]
    all_scenarios = []
    for batch in runner.generate_scenarios.starmap(topic_args):
        all_scenarios.extend(batch)

    all_scenarios = list(dict.fromkeys(all_scenarios))
    random.shuffle(all_scenarios)
    print(f"\nUnique scenarios generated: {len(all_scenarios):,}")

    # Save raw to volume
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    raw_path = f"{OUTPUT_DIR}/{output_name}_scenarios_raw.txt"
    with open(raw_path, "w") as f:
        for s in all_scenarios:
            f.write(s + "\n")

    # -----------------------------------------------------------------------
    # Step 2: Skip already-labeled if resuming
    # -----------------------------------------------------------------------
    already_done = load_checkpoint(checkpoint_path) if resume else set()
    to_label = [s for s in all_scenarios if s not in already_done]
    print(f"\n[2/2] Labeling {len(to_label):,} scenarios ({len(already_done):,} skipped) ...")

    # -----------------------------------------------------------------------
    # Step 3: Label in parallel batches with checkpointing
    # -----------------------------------------------------------------------
    batches = [to_label[i:i+label_batch_size] for i in range(0, len(to_label), label_batch_size)]
    print(f"Processing {len(batches)} batches of {label_batch_size} ...")

    all_new_results = []
    for i, batch_results in enumerate(runner.label_scenarios.starmap([(b,) for b in batches])):
        all_new_results.extend(batch_results)
        append_checkpoint(checkpoint_path, batch_results)
        good_so_far = sum(1 for r in all_new_results if r.get("label"))
        print(f"  batch {i+1}/{len(batches)} — {good_so_far:,} labeled so far")

    # Merge with resumed results
    all_results = all_new_results
    if resume and already_done:
        existing = []
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path) as f:
                for line in f:
                    try:
                        rec = json.loads(line.strip())
                        if rec.get("label") and rec["scenario"] in already_done:
                            existing.append(rec)
                    except Exception:
                        pass
        all_results = existing + all_new_results

    # -----------------------------------------------------------------------
    # Stats
    # -----------------------------------------------------------------------
    good   = [r for r in all_results if r.get("label")]
    failed = [r for r in all_results if not r.get("label")]

    print(f"\n=== Results ===")
    print(f"  Success: {len(good):,}")
    print(f"  Failed:  {len(failed):,}")

    from collections import Counter
    dist = Counter(r["label"] for r in good)
    print(f"\nLabel distribution:")
    for label, count in sorted(dist.items()):
        bar = "█" * int(count / len(good) * 40)
        print(f"  {label:15s} {count:5,} ({count/len(good)*100:5.1f}%)  {bar}")

    # -----------------------------------------------------------------------
    # Save everything to volume (PC-off safe)
    # -----------------------------------------------------------------------
    save_to_volume(all_results, output_name)
    print(f"\n✓ Done — {len(good):,} labeled scenarios ready for training.")