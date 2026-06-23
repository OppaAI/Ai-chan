# Aiko Persona Finetune Pipeline — OppaAI

Fine-tunes Ministral-3B-Instruct to reliably output Aiko's structured 3-line format.
Uses Qwen3-30B-A3B as teacher labeler. Runs entirely on Modal — PC can be off after each step.

## Output Format

Every Aiko response after finetuning:

```
😑
*crosses arms*
I told you this would happen.
```

- **Line 1** — emoji → VRM face expression
- **Line 2** — `*physical action*` → animation via semantic embedding lookup
- **Line 3+** — spoken response → TTS pipeline

Animation mapping is done at runtime via fastembed cosine similarity —
no enum locking, add new animations anytime without retraining.

---

## Files

| File | Purpose |
|---|---|
| `dataset_gen.py` | Generate ~1600+ structured examples using Qwen3-30B-A3B teacher |
| `training.py` | Fine-tune Ministral-3B on Modal A100-40, export GGUF |
| `testing.py` | Evaluate GGUF against held-out test split |
| `inference.py` | Drop-in parser for think.py — parses output, resolves animations |

All outputs saved to Modal Volume `aiko-persona-data` under `/outputs/`.

---

## Full Pipeline

### Step 1: Generate dataset

```bash
modal run dataset_gen.py                    # ~1600 examples, all topics
modal run dataset_gen.py --n-per-topic 50   # quick test run (~400 examples)
modal run dataset_gen.py --resume           # resume after interruption
```

- ~1–2 hours on A10G
- ~$3–6 Modal credits
- Covers 8 topic areas: technical debug, teasing Jon, Japanese exchange,
  BC photography, Aiko identity, casual daily, architecture-aware, agentic confirm

### Step 2: Train

```bash
modal run training.py
modal run training.py --lora-r 32 --epochs 4   # stronger run
```

- ~30–45 min on A100-40GB
- ~$2–3 Modal credits
- Exports `ministral-3b-AikoPersona_q8_0.gguf` and `ministral-3b-AikoPersona_q4_k_m.gguf`

### Step 3: Evaluate

```bash
# On Modal (PC-off):
modal run testing.py

# Or locally on AIVA after downloading:
modal volume get aiko-persona-data outputs/ministral-3b-AikoPersona_q8_0.gguf ./
modal volume get aiko-persona-data outputs/test_split.jsonl ./
python testing.py --local --model ministral-3b-AikoPersona_q8_0.gguf --test-data test_split.jsonl
```

Evaluation metrics:
- Format pass rate (3-line structure)
- Emoji presence
- Action properly wrapped
- Action is physical (not internal state)
- No embedded asterisks in response
- No hollow affirmations
- Reasonable response length

### Step 4: Download to AuRoRA

```bash
modal volume get aiko-persona-data outputs/ministral-3b-AikoPersona_q8_0.gguf ./
scp ministral-3b-AikoPersona_q8_0.gguf oppa-ai@aurora:/opt/models/
```

Use q4_k_m if RAM is tight on the Jetson.

---

## Integration in think.py

```python
from inference import AikoOutputParser, ActionResolver, MINIMAL_SYSTEM_PROMPT

# load once at startup
resolver = ActionResolver()
parser = AikoOutputParser(resolver)

# replace soul.md system prompt with the minimal anchor (~100 tokens)
system = MINIMAL_SYSTEM_PROMPT

# after getting raw LLM output:
result = parser.parse_safe(raw_output)

# route components
send_emotion_to_vrm(result.emotion)          # emoji → blendshape
send_animations_to_vrm(result.animations, result.action_mode)   # [(anim_key, weight), ...]
speak(result.response)                       # → speak.py TTS
```

---

## Adding New Animations

Edit `ANIMATION_REGISTRY` in `inference.py` — no retraining needed:

```python
ANIMATION_REGISTRY["anim_new_move"] = "description of what the animation looks like"
```

The description is what gets embedded for semantic matching.
Restart Aiko to reload (or call `resolver.reload_registry(ANIMATION_REGISTRY)`).

---

## Cost Estimate (Modal, $160 credits)

| Step | GPU | Time | Cost |
|---|---|---|---|
| Dataset gen (full) | A10G | ~1–2h | ~$3–6 |
| Fine-tune Ministral-3B | A100-40GB | ~30–45min | ~$2–3 |
| Evaluation | A10G | ~10min | ~$0.20 |
| **Total** | | | **~$6–10** |

---

## After Finetuning: Reduced soul.md

Replace the full `soul.md` with `MINIMAL_SYSTEM_PROMPT` (~100 tokens):

```
You are Aiko, Jon's AI companion on AuRoRA (Jetson Orin Nano Super).
Deadpan. Direct. Dry wit. No hollow affirmations. Bilingual EN/JP.
Always respond: emoji / *physical action* / spoken response.
```

Format behavior is baked into weights.
Keep this minimal anchor for identity grounding and format reminder.
