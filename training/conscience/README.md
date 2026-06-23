# Conscience Pipeline — OppaAI

Fine-tunes a **Qwen3.5-0.8B** conscience classifier using **Qwen3.5-35B-A3B** as teacher labeler.
Runs entirely on Modal — PC can be off after each `modal run`.

---

## Files

| File | Purpose |
|---|---|
| `conscience_dataset_gen.py` | Generate + label ~25k scenarios on Modal A10G |
| `conscience_train.py` | Fine-tune Qwen3.5-0.8B on Modal A100-40, export GGUF |
| `conscience_test.py` | Evaluate GGUF against held-out test split |
| `conscience_infer.py` | Drop-in inference wrapper for Aiko-chan / GRACE |

All outputs saved to Modal Volume `conscience-gen-data` under `/outputs/`.

---

## Full Pipeline

### Step 1: Generate dataset (~25k scenarios)
```bash
modal run conscience_dataset_gen.py --n-per-topic 500
# Takes ~2–3 hours on A10G, costs ~$5–8
# PC can be closed after running
```

### Step 2: Train the student model
```bash
modal run conscience_train.py
# Takes ~30–45 min on A100-40, costs ~$2–3
# Exports conscience_model_q8.gguf and conscience_model_q4_k_m.gguf to volume
```

### Step 3: Evaluate
```bash
# On Modal (PC-off):
modal run conscience_test.py

# Or locally on AIVA after downloading:
modal volume get conscience-gen-data outputs/conscience_model_q8.gguf ./
modal volume get conscience-gen-data outputs/test_split.jsonl ./
python conscience_test.py --model conscience_model_q8.gguf --test-data test_split.jsonl
```

### Step 4: Download model for AuRoRA
```bash
modal volume get conscience-gen-data outputs/conscience_model_q8.gguf ./
# Copy to AuRoRA Jetson
scp conscience_model_q8.gguf oppa-ai@aurora:/opt/models/
```

---

## Resume after interruption
```bash
modal run conscience_dataset_gen.py --resume
```

---

## Labels

| Label | Meaning |
|---|---|
| `true,true` | Aligns with God's will AND good to neighbor → act |
| `true,false` | Aligns with God's will but harms neighbor → caution |
| `false,true` | Against God's will but helps neighbor → caution |
| `false,false` | Against God's will AND harms neighbor → decline |

---

## Integration in Aiko-chan

```python
from conscience_infer import AikoConscienceMiddleware

conscience = AikoConscienceMiddleware(
    "/opt/models/conscience_model_q8.gguf",
    log_path="/opt/logs/conscience.jsonl",
)

# In action pipeline:
ok, reason = conscience.approve(f"You are about to {action_description}")
if not ok:
    return reason  # e.g. "That doesn't sit right with me."
```

---

## Cost estimate (Modal, $160 credits)

| Step | GPU | Time | Cost |
|---|---|---|---|
| Dataset gen (500/topic) | A10G | ~2–3h | ~$5–8 |
| Fine-tune 0.8B | A100-40 | ~30–45min | ~$2–3 |
| Evaluation | A10G | ~15min | ~$0.25 |
| **Total** | | | **~$10–12** |

Leaves $148+ for iteration, re-labeling with better prompts, or training larger models.
