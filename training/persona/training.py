"""
training.py — Aiko Persona Finetune Training
OppaAI / AuRoRA Project

Fine-tunes Ministral-3B-Instruct on the structured output persona dataset.
Uses LoRA via Unsloth for fast aarch64-compatible training.
Exports merged GGUF for direct use on the Jetson (llama-server).

Target behavior after training:
  Every response follows the 3-line format:
    <emoji>
    *<physical action>*
    <TTS-ready response>

Outputs saved to Modal Volume: aiko-persona-data under /outputs/

Usage:
    modal run training.py
    modal run training.py --lora-r 32 --epochs 3
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Modal infra
# ---------------------------------------------------------------------------

APP_NAME = "aiko-persona-training"
VOLUME_NAME = "aiko-persona-data"
OUTPUTS_DIR = "/outputs"

# Ministral-3B — the model running on AuRoRA Jetson
STUDENT_MODEL = "mistralai/Ministral-3B-Instruct"
MODEL_SLUG = "ministral-3b-AikoPersona"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git",
        "torch>=2.3.0",
        "transformers>=4.43.0",
        "datasets>=2.20.0",
        "trl>=0.9.0",
        "peft>=0.11.0",
        "huggingface_hub>=0.23.0",
        "bitsandbytes>=0.43.0",
        "llama-cpp-python",
        "tqdm",
        "accelerate",
    )
)

# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=60 * 90,   # 90 minutes
    volumes={OUTPUTS_DIR: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    memory=40960,
)
def train(
    lora_r: int = 16,
    lora_alpha: int = 32,
    epochs: int = 3,
    batch_size: int = 4,
    grad_accum: int = 4,
    lr: float = 2e-4,
    max_seq_len: int = 512,
    warmup_ratio: float = 0.05,
):
    import torch
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig
    from unsloth import FastLanguageModel
    from tqdm import tqdm

    out_dir = Path(OUTPUTS_DIR)
    volume.reload()

    # -----------------------------------------------------------------------
    # Load dataset
    # -----------------------------------------------------------------------
    train_path = out_dir / "train_split.jsonl"
    val_path = out_dir / "val_split.jsonl"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training data not found at {train_path}. Run dataset_gen.py first."
        )

    def load_jsonl(path: Path) -> list[dict]:
        examples = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))
        return examples

    train_data = load_jsonl(train_path)
    val_data = load_jsonl(val_path)
    print(f"Loaded {len(train_data)} train / {len(val_data)} val examples")

    # -----------------------------------------------------------------------
    # Format for SFT — extract chat messages and format as text
    # -----------------------------------------------------------------------
    def format_example(example: dict) -> dict:
        """Convert chat messages to a single training text string."""
        messages = example["messages"]
        # Build text: system + user turn + assistant turn
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                parts.append(f"<s>[INST] <<SYS>>\n{content}\n<</SYS>>\n\n")
            elif role == "user":
                parts.append(f"{content} [/INST] ")
            elif role == "assistant":
                parts.append(f"{content}</s>")
        return {"text": "".join(parts)}

    train_dataset = Dataset.from_list([format_example(ex) for ex in train_data])
    val_dataset = Dataset.from_list([format_example(ex) for ex in val_data])

    # -----------------------------------------------------------------------
    # Load model with Unsloth
    # -----------------------------------------------------------------------
    print(f"Loading {STUDENT_MODEL} with Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=STUDENT_MODEL,
        max_seq_length=max_seq_len,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    print(f"LoRA rank={lora_r}, alpha={lora_alpha}")
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # -----------------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------------
    adapter_path = out_dir / "lora_adapter"
    merged_path = out_dir / "merged_model"

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=SFTConfig(
            output_dir=str(adapter_path),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            warmup_ratio=warmup_ratio,
            learning_rate=lr,
            fp16=False,
            bf16=True,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            save_steps=100,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            report_to="none",
            dataset_text_field="text",
            max_seq_length=max_seq_len,
            packing=False,
        ),
    )

    print("\nStarting training...")
    trainer_stats = trainer.train()
    print(f"\nTraining complete.")
    print(f"  Steps     : {trainer_stats.global_step}")
    print(f"  Train loss: {trainer_stats.training_loss:.4f}")
    print(f"  Time      : {trainer_stats.metrics['train_runtime']:.1f}s")

    # -----------------------------------------------------------------------
    # Save adapter
    # -----------------------------------------------------------------------
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    volume.commit()
    print(f"LoRA adapter saved to {adapter_path}")

    # -----------------------------------------------------------------------
    # Merge and export GGUF
    # -----------------------------------------------------------------------
    print("\nMerging LoRA into base model...")
    model.save_pretrained_merged(
        str(merged_path),
        tokenizer,
        save_method="merged_16bit",
    )

    print("Exporting GGUF variants...")
    for quant in ["q8_0", "q4_k_m"]:
        gguf_name = f"{MODEL_SLUG}_{quant}.gguf"
        gguf_path = out_dir / gguf_name
        model.save_pretrained_gguf(
            str(out_dir / MODEL_SLUG),
            tokenizer,
            quantization_method=quant,
        )
        # unsloth names it slightly differently, find and rename
        candidates = list(out_dir.glob(f"{MODEL_SLUG}*{quant}*.gguf"))
        if candidates:
            candidates[0].rename(gguf_path)
        print(f"  ✓ {gguf_name}")

    volume.commit()

    # -----------------------------------------------------------------------
    # Save training metadata
    # -----------------------------------------------------------------------
    meta = {
        "base_model": STUDENT_MODEL,
        "model_slug": MODEL_SLUG,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "epochs": epochs,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "lr": lr,
        "max_seq_len": max_seq_len,
        "train_examples": len(train_data),
        "val_examples": len(val_data),
        "train_loss": trainer_stats.training_loss,
        "train_steps": trainer_stats.global_step,
        "runtime_seconds": trainer_stats.metrics["train_runtime"],
        "gguf_q8": f"{MODEL_SLUG}_q8_0.gguf",
        "gguf_q4": f"{MODEL_SLUG}_q4_k_m.gguf",
    }
    (out_dir / "training_meta.json").write_text(json.dumps(meta, indent=2))
    volume.commit()

    print(f"\n✓ Training pipeline complete.")
    print(f"  Q8  GGUF : {out_dir}/{MODEL_SLUG}_q8_0.gguf")
    print(f"  Q4K GGUF : {out_dir}/{MODEL_SLUG}_q4_k_m.gguf")
    print(f"  Metadata : {out_dir}/training_meta.json")
    return meta


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    lora_r: int = 16,
    lora_alpha: int = 32,
    epochs: int = 3,
    batch_size: int = 4,
    grad_accum: int = 4,
    lr: float = 2e-4,
):
    print(f"\nAiko Persona Finetuning")
    print(f"  Base model  : {STUDENT_MODEL}")
    print(f"  LoRA r/alpha: {lora_r}/{lora_alpha}")
    print(f"  Epochs      : {epochs}")
    print(f"  Volume      : {VOLUME_NAME}\n")

    meta = train.remote(
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        epochs=epochs,
        batch_size=batch_size,
        grad_accum=grad_accum,
        lr=lr,
    )
    print("\nFinal metadata:")
    print(json.dumps(meta, indent=2))
