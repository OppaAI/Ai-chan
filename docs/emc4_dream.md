# EMC-4 — dream distillation (EM → SM)

## What
During `AikoMemorize.dream()`, undistrilled episodic rows are:
1. Selected (salience / recall_count / recency)
2. Batched into an LLM prompt for durable facts only
3. Written via `add_raw` (same dedup/supersede as normal writes)
4. Marked `distilled_at` **only when the LLM returned a non-empty fact list**
   (empty extract leaves rows eligible for a later dream pass)

Human analogy: sleep consolidates episodic traces into semantic knowledge.
WM→EM is EMC-2; this is EM→SM.

## Config
```yaml
EMC_DREAM_ENABLED: "1"
EMC_DREAM_LIMIT: "12"       # max undistrilled episodes per dream pass
EMC_DREAM_BATCH: "4"        # episodes per LLM call
EMC_DREAM_MIN_CHARS: "60"   # skip short traces
EMC_DREAM_MAX_TOKENS: "256"
```

## Schema
`emc_storage.distilled_at TEXT` added idempotently on first distill.

## Out of scope
- Grouping/clustering staging before flush
- Full emotional reprocessing of episodes
