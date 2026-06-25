[← Back to README](../README.md)
# Aiko-chan 愛子ちゃん — Installation Guide

This guide installs the current Aiko-chan stack: Python 3.12 + `uv`, SearXNG in Docker, a local OpenAI-compatible LLM endpoint (usually llama.cpp `llama-server`), sqlite-vec memory, MioTTS, SenseVoice ASR, the curses TUI, and the optional browser WebUI/VRM frontend.

---

## Table of Contents

1. [System Requirements](#1-system-requirements)
2. [Python 3.12 via pyenv](#2-python-312-via-pyenv)
3. [uv Package Manager](#3-uv-package-manager)
4. [Clone the Repo](#4-clone-the-repo)
5. [Docker & SearXNG](#5-docker--searxng)
6. [Local LLM Server](#6-local-llm-server)
7. [MioTTS Server](#7-miotts-server)
8. [Configure Environment](#8-configure-environment)
9. [Install Python Dependencies](#9-install-python-dependencies)
10. [Jetson Orin Nano Notes](#10-jetson-orin-nano-notes)
11. [Verify the Full Stack](#11-verify-the-full-stack)
12. [Run Aiko-chan](#12-run-aiko-chan)

---

## 1. System Requirements

| Requirement | Minimum | Notes |
|---|---|---|
| OS | Ubuntu 22.04 / 24.04 | Also works on WSL2 for text-only/testing use |
| Python | **3.12.x** | `pyproject.toml` requires `>=3.12,<3.13` |
| RAM | 8 GB | 16 GB recommended |
| GPU/accelerator | optional but recommended | Jetson Orin Nano is the target constrained device |
| Storage | 20 GB free | LLM, ASR, TTS, and embedding models are large |
| Docker | 24.x+ | Required for SearXNG |
| Local LLM server | OpenAI-compatible `/v1` API | `LLM_BASE_URL` defaults to `http://localhost:8080/v1` |

---

## 2. Python 3.12 via pyenv

```bash
sudo apt update
sudo apt install -y make build-essential libssl-dev zlib1g-dev \
  libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
  libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
  libffi-dev liblzma-dev git

curl https://pyenv.run | bash

echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
source ~/.bashrc

pyenv install 3.12.13
pyenv global 3.12.13
python --version
```

---

## 3. uv Package Manager

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.cargo/env"
uv --version
```

`uv` is the supported dependency manager for this repo.

---

## 4. Clone the Repo

```bash
git clone https://github.com/OppaAI/Aiko-chan.git
cd Aiko-chan
cp .env.example .env
```

The repo contains `docker-compose.yml`, `searxng/` config, Python source, `webui/static/`, skills, persona files, and docs.

---

## 5. Docker & SearXNG

```bash
sudo apt remove -y docker docker-engine docker.io containerd runc 2>/dev/null || true
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER"
newgrp docker

docker --version
docker compose version
```

Start SearXNG from the project root:

```bash
docker compose up -d
docker compose ps
```

Expected service/container: `aiko_searxng` running and listening at `http://localhost:8081`.

---

## 6. Local LLM Server

Aiko's current chat runtime uses the OpenAI Python client against a local OpenAI-compatible endpoint. The default environment values are:

```dotenv
LLM_BASE_URL=http://localhost:8080/v1
LLM_MODEL=ministral
```

A common setup is llama.cpp `llama-server` with a local GGUF model. Example:

```bash
# Example only: adjust paths, GPU layers, context, and alias for your hardware/model.
llama-server \
  -m /path/to/Ministral-3-3B-Reasoning-Q4_K_M.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  --alias ministral \
  --jinja \
  -c 4096
```

Verify the endpoint:

```bash
curl http://localhost:8080/v1/models
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"ministral","messages":[{"role":"user","content":"Say hello."}],"max_tokens":32}'
```

> Older Ollama-specific instructions are no longer the primary path for the current codebase. You can still use any backend that exposes an OpenAI-compatible `/v1` API and matches `LLM_BASE_URL`/`LLM_MODEL`.

---

## 7. MioTTS Server

MioTTS is an external HTTP synthesis service. Aiko calls `MIOTTS_API_URL` and plays the returned audio through `sounddevice`.

```dotenv
MIOTTS_API_URL=http://localhost:8001
MIOTTS_MODEL=MioTTS-0.4B-Q4_K_M
MIOTTS_PRESET=aiko_flat
```

Typical MioTTS flow:

1. Start the local OpenAI-compatible model server that serves the MioTTS model.
2. Start MioTTS's `run_server.py` wrapper, pointed at that model server.
3. Verify the API before launching Aiko.

```bash
curl http://localhost:8001/health
uv run python core/speak.py --devices
uv run python core/speak.py --wait "Hello, I'm Aiko."
```

If you do not need voice, run Aiko with `--text`; wakeup will skip TTS and ASR.

---

## 8. Configure Environment

Edit `.env`. At minimum, check these values:

```dotenv
# Identity
AI_NAME=Aiko
USER_ID=OppaAI

# Local OpenAI-compatible LLM
LLM_BASE_URL=http://localhost:8080/v1
LLM_MODEL=ministral
LLM_TIMEOUT=120
LLM_MAX_TOKENS=280

# Memory
SQLITE_MEMORY_PATH=/home/oppa-ai/.aiko/memory.db
FASTEMBED_CACHE_PATH=/home/oppa-ai/.cache/fastembed
EMBED_MODEL=BAAI/bge-base-en-v1.5
MEMORY_RECALL_LIMIT=5

# Web search
SEARXNG_URL=http://localhost:8081
SEARXNG_SECRET=<your_secret_from_searxng_settings.yml>

# Voice output
MIOTTS_API_URL=http://localhost:8001
MIOTTS_PRESET=aiko_flat

# Voice input
ASR_DEVICE=cpu
ASR_LANGUAGE=auto
ASR_MODEL=csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17

# Routing and agentic mode (current names read by core/think.py)
ROUTE_ENABLED=1
ROUTE_MODE=semantic
ROUTE_SEMANTIC_THRESHOLD=0.36
SEARCH_SEMANTIC_THRESHOLD=0.36

# Scheduling/workspace
TIMEZONE=America/Vancouver
WORKSPACE_ROOT=workspace
SCHEDULE_PATH=workspace/schedule.json
SCHEDULE_POLL_SECONDS=15
```

If your `.env` still has only `AIKO_ROUTE_*` keys, add the `ROUTE_*` keys above; the current code reads `ROUTE_ENABLED`, `ROUTE_MODE`, and `ROUTE_SEMANTIC_THRESHOLD`.

---

## 9. Install Python Dependencies

```bash
uv sync
```

This installs dependencies from `pyproject.toml`/`uv.lock`, including `openai`, `sqlite-vec`, `fastembed`, `sherpa-onnx`, `silero-vad`, `sounddevice`, `soundfile`, `websockets`, `torch`, and `torchaudio`.

For browser frontend asset experiments, the repo also has a `package.json` with `three` and `@pixiv/three-vrm`; the checked-in `webui/static/` files are served directly by Python, so `npm install` is not required for normal runtime.

---

## 10. Jetson Orin Nano Notes

- The current `pyproject.toml` uses PyPI dependencies plus an ONNX Runtime CUDA nightly index for `onnxruntime-gpu`.
- `ASR_DEVICE=cpu` is the documented SenseVoice setting in `.env.example` because CUDA EP availability can vary by JetPack/JP version.
- Keep `SQLITE_MEMORY_PATH` and `FASTEMBED_CACHE_PATH` on persistent storage, not `/tmp`.
- If audio output is silent, inspect devices and set `MIOTTS_DEVICE` or the system default sink:

```bash
uv run python core/speak.py --devices
pactl list short sinks
pactl set-default-sink <sink_name>
```

- Watch memory pressure with `jtop` when LLM, MioTTS, ASR, and embeddings are warm.

---

## 11. Verify the Full Stack

```bash
# SearXNG
curl "http://localhost:8081/search?q=test&format=json" \
  -H "X-Forwarded-For: 127.0.0.1"

# Local OpenAI-compatible LLM
curl http://localhost:8080/v1/models

# MioTTS (voice mode only)
curl http://localhost:8001/health

# Python imports
uv run python -c "import sqlite_vec, fastembed, sherpa_onnx, silero_vad, sounddevice, websockets; print('OK')"

# Skill registry and agentic tool schemas
uv run python -c "from core.skills import list_skillsets; print(list_skillsets())"
uv run python -c "from core.agentic import tool_schemas; print([s['function']['name'] for s in tool_schemas()])"

# Memory backend
uv run python -c "from core.memorize import AikoMemorize; m=AikoMemorize(silent=True); print(m.get_all()[:1])"
```

---

## 12. Run Aiko-chan

```bash
# Default: curses TUI, full voice if services are available
uv run python main.py

# Browser WebUI + VRM frontend
uv run python main.py --webui

# Keyboard-only, skips ASR and TTS
uv run python main.py --text

# Keyboard input but keep TTS available
uv run python main.py --no-asr

# Debug memory hits each turn
uv run python main.py --debug

# Wipe memories and exit
uv run python main.py --clear-mem
```

In-app commands include `/quit`, `/reset`, `/memory`, `/clear`, `/remember`, `/think <question>`, `/web <query>`, `/voice`, `/listen`, and `/help`.
