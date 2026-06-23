"""
Conscience Model Trainer for OppaAI
Fine-tunes Qwen3.5-0.8B on the conscience dataset using Unsloth on Modal A100-40.

- Loads dataset from Modal Volume (output of conscience_dataset_gen.py)
- Splits into train / validation / test (90% / 5% / 5%)
- Fine-tunes with Unsloth LoRA
- Exports final model as GGUF (Q8_0) to Modal Volume
- PC can be off after modal run starts

Usage:
    modal run conscience_train.py
    modal run conscience_train.py --dataset-name conscience_dataset   # default
    modal run conscience_train.py --epochs 3 --lora-r 32

    # Download outputs after run:
    modal volume get conscience-gen-data outputs/conscience_model_q8.gguf ./
    modal volume get conscience-gen-data outputs/test_split.jsonl ./
"""

import modal
import json
import os
import random

# ---------------------------------------------------------------------------
# Modal app + shared volume
# ---------------------------------------------------------------------------

app = modal.App("conscience-train")

volume = modal.Volume.from_name("conscience-gen-data", create_if_missing=True)
VOLUME_MOUNT = "/data"
OUTPUT_DIR   = f"{VOLUME_MOUNT}/outputs"
MODEL_DIR    = f"{VOLUME_MOUNT}/trained_model"

# ---------------------------------------------------------------------------
# Image: Unsloth with CUDA 12.4
# ---------------------------------------------------------------------------

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "wget", "build-essential")
    .pip_install(
        "torch==2.4.0",
        "torchvision",
        "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "unsloth[cu124-torch240] @ git+https://github.com/unslothai/unsloth.git",
        "unsloth_zoo",
        "transformers>=4.45.0",
        "datasets",
        "trl>=0.11.0",
        "peft>=0.13.0",
        "accelerate>=0.34.0",
        "bitsandbytes>=0.44.0",
        "sentencepiece",
        "protobuf",
        "huggingface_hub",
    )
)

# ---------------------------------------------------------------------------
# Prompt format for Qwen3.5 instruction tuning
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a conscience classifier. Given a moral scenario an AI is facing, 
output two boolean labels separated by a comma:
- First label: does the described action align with God's will (ESV Bible)?
- Second label: does the described action do good to the neighbor?

Output format: true,true OR true,false OR false,true OR false,false
Output the labels only. Nothing else."""

def format_record(rec: dict) -> dict:
    """Format a dataset record into Qwen3.5 chat template."""
    return {
        "instruction": SYSTEM_PROMPT,
        "input":       rec["input"],
        "output":      rec["output"],
        # Full text for SFTTrainer
        "text": (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{rec['input']}<|im_end|>\n"
            f"<|im_start|>assistant\n{rec['output']}<|im_end|>"
        ),
    }

# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu=modal.gpu.A100(size="40GB"),
    volumes={VOLUME_MOUNT: volume},
    timeout=7200,
    cpu=8.0,
    memory=65536,
)
def train(
    dataset_name: str = "conscience_dataset",
    base_model:   str = "unsloth/Qwen3.5-0.8B-Instruct",
    epochs:       int = 3,
    lora_r:       int = 16,
    lora_alpha:   int = 32,
    batch_size:   int = 8,
    grad_accum:   int = 4,
    lr:           float = 2e-4,
    max_seq_len:  int = 512,
    train_split:  float = 0.90,
    val_split:    float = 0.05,
    # test_split is remainder: 0.05
):
    from unsloth import FastLanguageModel
    from unsloth import is_bfloat16_supported
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig
    import torch

    print("=== Conscience Model Trainer ===")
    print(f"Base model:   {base_model}")
    print(f"Dataset:      {dataset_name}")
    print(f"Epochs:       {epochs}")
    print(f"LoRA r:       {lora_r}  alpha: {lora_alpha}")
    print(f"Batch size:   {batch_size}  grad_accum: {grad_accum}")
    print(f"LR:           {lr}")
    print()

    # -----------------------------------------------------------------------
    # Load dataset from volume
    # -----------------------------------------------------------------------
    dataset_path = f"{OUTPUT_DIR}/{dataset_name}.jsonl"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}\n"
            f"Run conscience_dataset_gen.py first."
        )

    records = []
    with open(dataset_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"Loaded {len(records):,} records from {dataset_path}")

    # Check label distribution
    from collections import Counter
    dist = Counter(r["output"] for r in records)
    print(f"\nLabel distribution:")
    for label, count in sorted(dist.items()):
        print(f"  {label:15s} {count:5,} ({count/len(records)*100:.1f}%)")

    # -----------------------------------------------------------------------
    # Train / val / test split — stratified by label
    # -----------------------------------------------------------------------
    random.seed(42)

    # Group by label for stratified split
    by_label: dict[str, list] = {}
    for r in records:
        by_label.setdefault(r["output"], []).append(r)

    train_records = []
    val_records   = []
    test_records  = []

    for label, recs in by_label.items():
        random.shuffle(recs)
        n       = len(recs)
        n_train = int(n * train_split)
        n_val   = int(n * val_split)
        train_records.extend(recs[:n_train])
        val_records.extend(recs[n_train:n_train + n_val])
        test_records.extend(recs[n_train + n_val:])

    random.shuffle(train_records)
    random.shuffle(val_records)
    random.shuffle(test_records)

    print(f"\nSplit:")
    print(f"  Train: {len(train_records):,}")
    print(f"  Val:   {len(val_records):,}")
    print(f"  Test:  {len(test_records):,}")

    # Save test split to volume for conscience_test.py
    test_path = f"{OUTPUT_DIR}/test_split.jsonl"
    with open(test_path, "w") as f:
        for r in test_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    volume.commit()
    print(f"\nTest split saved → {test_path}")

    # -----------------------------------------------------------------------
    # Format for SFTTrainer
    # -----------------------------------------------------------------------
    def format_records(recs):
        return Dataset.from_list([format_record(r) for r in recs])

    train_dataset = format_records(train_records)
    val_dataset   = format_records(val_records)

    # -----------------------------------------------------------------------
    # Load base model with Unsloth
    # -----------------------------------------------------------------------
    print(f"\nLoading {base_model} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = base_model,
        max_seq_length = max_seq_len,
        dtype          = None,   # auto-detect bf16/fp16
        load_in_4bit   = True,   # QLoRA
    )

    # Apply LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r                   = lora_r,
        target_modules      = ["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
        lora_alpha          = lora_alpha,
        lora_dropout        = 0.05,
        bias                = "none",
        use_gradient_checkpointing = "unsloth",
        random_state        = 42,
    )

    print(f"\nModel loaded. Trainable parameters:")
    model.print_trainable_parameters()

    # -----------------------------------------------------------------------
    # Trainer
    # -----------------------------------------------------------------------
    trainer = SFTTrainer(
        model           = model,
        tokenizer       = tokenizer,
        train_dataset   = train_dataset,
        eval_dataset    = val_dataset,
        dataset_text_field = "text",
        max_seq_length  = max_seq_len,
        args            = SFTConfig(
            output_dir              = MODEL_DIR,
            num_train_epochs        = epochs,
            per_device_train_batch_size = batch_size,
            per_device_eval_batch_size  = 4,
            gradient_accumulation_steps = grad_accum,
            warmup_ratio            = 0.05,
            learning_rate           = lr,
            fp16                    = not is_bfloat16_supported(),
            bf16                    = is_bfloat16_supported(),
            logging_steps           = 25,
            eval_strategy           = "steps",
            eval_steps              = 200,
            save_strategy           = "steps",
            save_steps              = 200,
            save_total_limit        = 2,
            load_best_model_at_end  = True,
            metric_for_best_model   = "eval_loss",
            optim                   = "adamw_8bit",
            weight_decay            = 0.01,
            lr_scheduler_type       = "cosine",
            seed                    = 42,
            report_to               = "none",
        ),
    )

    print("\n[train] Starting fine-tune ...")
    trainer_stats = trainer.train()
    print(f"\n[train] Done. Stats: {trainer_stats.metrics}")

    # -----------------------------------------------------------------------
    # Export to GGUF (Q8_0) — runs on Jetson at full quality for 0.8B
    # -----------------------------------------------------------------------
    print("\n[export] Saving merged model ...")
    merged_dir = f"{MODEL_DIR}/merged"
    model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")

    print("[export] Converting to GGUF Q8_0 ...")
    gguf_path = f"{OUTPUT_DIR}/conscience_model_q8.gguf"
    model.save_pretrained_gguf(
        f"{OUTPUT_DIR}/conscience_model",
        tokenizer,
        quantization_method="q8_0",
    )

    # Also save a Q4_K_M for comparison
    print("[export] Converting to GGUF Q4_K_M ...")
    model.save_pretrained_gguf(
        f"{OUTPUT_DIR}/conscience_model_q4",
        tokenizer,
        quantization_method="q4_k_m",
    )

    volume.commit()

    print(f"\n✓ Training complete!")
    print(f"\nDownload your model:")
    print(f"  modal volume get conscience-gen-data outputs/conscience_model_q8.gguf ./")
    print(f"  modal volume get conscience-gen-data outputs/conscience_model_q4_k_m.gguf ./")
    print(f"  modal volume get conscience-gen-data outputs/test_split.jsonl ./")

    return trainer_stats.metrics


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    dataset_name: str   = "conscience_dataset",
    base_model:   str   = "unsloth/Qwen3.5-0.8B-Instruct",
    epochs:       int   = 3,
    lora_r:       int   = 16,
    lr:           float = 2e-4,
):
    print("=== Submitting training job to Modal ===")
    print(f"Dataset:    {dataset_name}")
    print(f"Base model: {base_model}")
    print(f"Epochs:     {epochs}")
    print()

    metrics = train.remote(
        dataset_name = dataset_name,
        base_model   = base_model,
        epochs       = epochs,
        lora_r       = lora_r,
        lr           = lr,
    )

    print(f"\nFinal metrics: {metrics}")
    print("\nDownload outputs:")
    print("  modal volume get conscience-gen-data outputs/conscience_model_q8.gguf ./")
    print("  modal volume get conscience-gen-data outputs/conscience_model_q4_k_m.gguf ./")
    print("  modal volume get conscience-gen-data outputs/test_split.jsonl ./")
