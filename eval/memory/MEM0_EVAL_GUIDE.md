# Memory Benchmarks & Eval Guide

## Committed Fixes

### 1. Memory JSON Parse Bug (b65b12b)
- Added `parse_json_array()` to salvage truncated LLM output
- Bumped `_EXTRACT_MAX_TOKENS`: 128 → 512
- Cleaned dead imports

### 2. KB Graph Import Timing (25a46fd)
- Deferred `cognition.knowledge` import in `_open_conn()`
- Import runs at request time, not module load
- Prevents fallback to wrong DB when age binary not ready at import time

## Running the mem0 Eval

### Setup

```bash
git clone https://github.com/mem0ai/memory-benchmarks.git
cd memory-benchmarks
cp /home/oppa-ai/jetson/eval/memory/aiko_mem0_adapter.py benchmarks/common/
```

### Environment Variables

```bash
export AIKO_ROOT=/home/oppa-ai/jetson
export SQLITE_MEMORY_PATH=/tmp/aiko_bench_memory.db
export LLM_BASE_URL=http://localhost:8080/v1
export EXTRACT_MODEL=ministral   # or your extraction model
export OPENAI_API_KEY=sk-...     # for gpt-4o judge/answerer
```

### Patch Runner

Edit `benchmarks/locomo/run.py`:

```python
from benchmarks.common.aiko_mem0_adapter import AikoMemClient as Mem0Client
```

### Execute Benchmark

```bash
python -m benchmarks.locomo.run \
  --project-name aiko-locomo \
  --judge-model gpt-4o \
  --answerer-model gpt-4o
```

## LLM Configuration

| Component | Default | Notes |
|-----------|---------|-------|
| **Extraction** | Local llama-server | Uses `LLM_BASE_URL` + `EXTRACT_MODEL` |
| **Judge** | gpt-4o | Requires `OPENAI_API_KEY` |
| **Answerer** | gpt-4o | Requires `OPENAI_API_KEY` |

### Override Models

```bash
python -m benchmarks.locomo.run \
  --project-name aiko-locomo \
  --judge-model gpt-4o \
  --answerer-model gpt-4o \
  --provider openai|anthropic|azure
```# Memory Benchmarks & Eval Guide

## Committed Fixes

### 1. Memory JSON Parse Bug (b65b12b)
- Added `parse_json_array()` to salvage truncated LLM output
- Bumped `_EXTRACT_MAX_TOKENS`: 128 → 512
- Cleaned dead imports

### 2. KB Graph Import Timing (25a46fd)
- Deferred `cognition.knowledge` import in `_open_conn()`
- Import runs at request time, not module load
- Prevents fallback to wrong DB when age binary not ready at import time

## Running the mem0 Eval

### Setup

```bash
git clone https://github.com/mem0ai/memory-benchmarks.git
cd memory-benchmarks
cp /home/oppa-ai/jetson/eval/memory/aiko_mem0_adapter.py benchmarks/common/
```

### Environment Variables

```bash
export AIKO_ROOT=/home/oppa-ai/jetson
export SQLITE_MEMORY_PATH=/tmp/aiko_bench_memory.db
export LLM_BASE_URL=http://localhost:8080/v1
export EXTRACT_MODEL=ministral   # or your extraction model
export OPENAI_API_KEY=sk-...     # for gpt-4o judge/answerer
```

### Patch Runner

Edit `benchmarks/locomo/run.py`:

```python
from benchmarks.common.aiko_mem0_adapter import AikoMemClient as Mem0Client
```

### Execute Benchmark

```bash
python -m benchmarks.locomo.run \
  --project-name aiko-locomo \
  --judge-model gpt-4o \
  --answerer-model gpt-4o
```

## LLM Configuration

| Component | Default | Notes |
|-----------|---------|-------|
| **Extraction** | Local llama-server | Uses `LLM_BASE_URL` + `EXTRACT_MODEL` |
| **Judge** | gpt-4o | Requires `OPENAI_API_KEY` |
| **Answerer** | gpt-4o | Requires `OPENAI_API_KEY` |

### Override Models

```bash
python -m benchmarks.locomo.run \
  --project-name aiko-locomo \
  --judge-model gpt-4o \
  --answerer-model gpt-4o \
  --provider openai|anthropic|azure
```