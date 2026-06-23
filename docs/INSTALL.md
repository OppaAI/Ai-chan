[← Back to README](../README.md)
# Aiko-chan 愛子ちゃん — Installation Guide

This guide walks through installing every component of the Aiko-chan stack from scratch. Follow the sections in order; each one is a prerequisite for the next.

---

## Table of Contents

1. [System Requirements](#1-system-requirements)
2. [Python 3.12 via pyenv](#2-python-312-via-pyenv)
3. [uv Package Manager](#3-uv-package-manager)
4. [Docker & Docker Compose](#4-docker--docker-compose)
5. [SearXNG (via Docker)](#5-searxng-via-docker)
6. [Ollama](#6-ollama)
7. [Pull a Chat Model](#7-pull-a-chat-model)
8. [MioTTS Server](#8-miotts-server)
9. [Clone the Repo & Configure Environment](#9-clone-the-repo--configure-environment)
10. [Install Python Dependencies](#10-install-python-dependencies)
11. [Jetson Orin Nano — Extra Steps](#11-jetson-orin-nano--extra-steps)
12. [Verify the Full Stack](#12-verify-the-full-stack)
13. [Run Aiko-chan](#13-run-aiko-chan)

---

## 1. System Requirements

| Requirement | Minimum | Notes |
|---|---|---|
| OS | Ubuntu 22.04 / 24.04 | Also works on WSL2 |
| Python | **3.12.x** | `pyproject.toml` is pinned `>=3.12,<3.13` |
| RAM | 8 GB | 16 GB recommended for comfort |
| GPU VRAM | 4 GB | 8 GB for smooth local LLM inference |
| Storage | 20 GB free | Models are large |
| Docker | 24.x+ | Required for SearXNG |

> **Jetson Orin Nano:** See [Section 11](#11-jetson-orin-nano--extra-steps) for board-specific wheel overrides before running `uv sync`.

---

## 2. Python 3.12 via pyenv

Aiko-chan requires Python **3.12**. Using `pyenv` avoids polluting your system Python.

```bash
# Install pyenv dependencies
sudo apt update
sudo apt install -y make build-essential libssl-dev zlib1g-dev \
  libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
  libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
  libffi-dev liblzma-dev git

# Install pyenv
curl https://pyenv.run | bash

# Add pyenv to your shell (bash example — adjust for zsh/fish)
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
source ~/.bashrc

# Install Python 3.12 and set it as the local default
pyenv install 3.12.13
pyenv global 3.12.13

# Confirm
python --version   # should print Python 3.12.x
```

---

## 3. uv Package Manager

`uv` is the only supported package manager for this project.

```bash
# Install uv (official installer)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Reload your shell so the uv binary is on PATH
source $HOME/.cargo/env   # or open a new terminal

# Confirm
uv --version
```

---

## 4. Docker & Docker Compose

SearXNG run as Docker containers.

```bash
# Remove any old Docker installs
sudo apt remove -y docker docker-engine docker.io containerd runc 2>/dev/null || true

# Install Docker Engine via the official convenience script
curl -fsSL https://get.docker.com | sudo sh

# Add your user to the docker group (avoids needing sudo for every docker command)
sudo usermod -aG docker $USER
newgrp docker   # apply group change without logging out

# Confirm Docker
docker --version

# Docker Compose is included with Docker Engine as a plugin
docker compose version
```

---

## 5. SearXNG (via Docker)

SearXNG is also managed by the project's `docker-compose.yml`. The `searxng/` directory inside the repo holds its configuration, so **clone the repo first** (Section 10) before starting the stack.

Once the repo is cloned, from the project root:

```bash
# Start SearXNG together
docker compose up -d

# Confirm both containers are running
docker compose ps
```

Expected output:

```
NAME        STATUS
searxng     running
```

SearXNG is available at the URL you set in `SEARXNG_URL` (default **http://localhost:8081**).

---

## 6. Ollama

Ollama serves the local LLM used by Aiko's `think.py` core.

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# The installer registers a systemd service; confirm it is running
systemctl status ollama

# If not running, start it manually
ollama serve
```

> On a **Jetson**, Ollama detects CUDA automatically. Verify with `ollama run llama3.2 "hello"` and check that GPU utilisation rises in `jtop`.

---

## 7. Pull a Chat Model

Pull the model referenced in your `.env` as `OLLAMA_MODEL`. The default in `.env.example` is a Ministral 3B reasoning GGUF:

```bash
ollama pull hf.co/unsloth/Ministral-3-3B-Reasoning-2512-GGUF:UD-Q4_K_XL
```

For a more capable model (recommended for grounded search answers — 7B+):

```bash
ollama pull mistral:7b-instruct
# or
ollama pull qwen2.5:7b
```

Confirm the model loads:

```bash
ollama run mistral:7b-instruct "Say hello."
```

---

## 8. MioTTS Server

MioTTS is an external HTTP TTS server. Aiko calls it at `MIOTTS_API_URL` (default **http://localhost:8001**).

> If you do not need voice output, set `MIOTTS_API_URL` to an unreachable address and run Aiko in `--text` mode. The client fails gracefully.

**Option A — Run MioTTS via Docker (recommended):**

```bash
docker run -d --name miotts \
  -p 8001:8001 \
  # replace with the actual MioTTS image when available
  miotts/miotts-server
```

**Option B — Run MioTTS from source:**

Follow the MioTTS project's own README. Once running, verify the health endpoint:

```bash
curl http://localhost:8001/health
# Expected: {"status":"ok"} or similar
```

---

## 9. Clone the Repo & Configure Environment

```bash
# Clone
git clone https://github.com/OppaAI/Aiko-chan.git
cd Aiko-chan

# Copy the example env file
cp .env.example .env
```

Open `.env` in your editor and set at minimum:

```dotenv
# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=hf.co/unsloth/Ministral-3-3B-Reasoning-2512-GGUF:UD-Q4_K_XL

# SearXNG
SEARXNG_URL=http://localhost:8081
SEARXNG_SECRET=<your_secret_from_searxng_settings.yml>

# MioTTS (leave blank or point to a dummy URL to disable voice)
MIOTTS_API_URL=http://localhost:8001

# ASR — SenseVoice via sherpa-onnx + Silero VAD
ASR_DEVICE=cpu
ASR_LANGUAGE=auto
ASR_MODEL=csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17
```

The `SEARXNG_SECRET` must match the `secret_key` value inside `searxng/settings.yml`.

---

## 10. Install Python Dependencies

```bash
# From the project root with Python 3.12 active
uv sync
```

`uv` reads `pyproject.toml` and `uv.lock`, creates a virtual environment in `.venv/`, and installs all dependencies.

> **If `uv sync` fails on wheel not found:** Check the local wheel paths in `pyproject.toml` for `torch` and `ctranslate2`. On a standard x86 machine, comment out or replace those path overrides with regular PyPI versions. On a Jetson, see Section 12.

---

## 11. Jetson Orin Nano — Extra Steps

The Jetson requires board-specific wheels for PyTorch that are not on PyPI.

### 11a. Download Jetson AI Lab wheels

Visit [https://forums.developer.nvidia.com/t/pytorch-for-jetson](https://forums.developer.nvidia.com/t/pytorch-for-jetson) and download the `.whl` files for:

- `torch` (JetPack 6.x compatible)
- `torchaudio`
- `torchvision`
- `ctranslate2` (Jetson build)

Place the wheels in a local directory, e.g. `~/wheels/`.

### 11b. Update pyproject.toml wheel paths

In `pyproject.toml`, confirm the path overrides for `torch` and `ctranslate2` point to your downloaded `.whl` files:

```toml
[tool.uv.sources]
torch = { path = "/home/oppa-ai/wheels/torch-<version>-cp310-linux_aarch64.whl" }
ctranslate2 = { path = "/home/oppa-ai/wheels/ctranslate2-<version>-cp310-linux_aarch64.whl" }
```

### 11c. Install JetPack dev libraries (if not already present)

```bash
sudo apt install -y libopenblas-dev libopenmpi-dev
```

### 11d. Run uv sync

```bash
uv sync
```

### 11e. PulseAudio default sink (for TTS audio output)

If audio output is silent after install, fix the PulseAudio default sink:

```bash
# List available sinks
pactl list short sinks

# Set the default (replace <sink_name> with your output device)
pactl set-default-sink <sink_name>

# To make it persist across reboots, add to /etc/pulse/default.pa:
echo "set-default-sink <sink_name>" | sudo tee -a /etc/pulse/default.pa
```

---

## 12. Verify the Full Stack

Run this checklist before launching Aiko for the first time:

```bash
# 1. SearXNG
curl "http://localhost:8081/search?q=test&format=json" \
  -H "X-Forwarded-For: 127.0.0.1"
# Expected: JSON search results

# 2. Ollama
curl http://localhost:11434/api/tags
# Expected: JSON list of pulled models

# 3. MioTTS (if using voice)
curl http://localhost:8001/health
# Expected: {"status":"ok"} or similar

# 4. Python environment
uv run python -c "import sqlite_vec; import sherpa_onnx; import silero_vad; print('deps OK')"
# Expected: deps OK

# 5. ASR model cache (first voice launch downloads SenseVoice files if absent)
uv run python -c "from core.listen import ASR_MODEL; print(ASR_MODEL)"
# Expected: csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17 or your override
```

---

## 13. Run Aiko-chan

```bash
# Full voice mode (ASR input + TTS output)
uv run python main.py

# Text-only mode (keyboard input, no ASR/TTS)
uv run python main.py --text

# Text mode with memory debug output
uv run python main.py --text --debug

# Wipe all stored memories and exit
uv run python main.py --clear-mem
```

On first launch Aiko will:
1. Connect to SQLite-vec and initialise the memory collection if it does not exist.
2. Warm up the Ollama model in the background.
3. Warm up MioTTS in the background (voice mode only).
4. Open the curses TUI.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `uv sync` fails — wheel not found | Jetson wheel path wrong | Update path in `pyproject.toml` → Section 12b |
| SearXNG returns 403 | Wrong `SEARXNG_SECRET` | Match secret in `.env` and `searxng/settings.yml` |
| Ollama model not found | Model not pulled | `ollama pull <model>` → Section 8 |
| No TTS audio on Jetson | Wrong PulseAudio sink | `pactl set-default-sink` → Section 11e |
| ASR import/model error | Missing sherpa-onnx/SenseVoice deps or model cache | Run `uv sync`, then launch once online or set `HF_HUB_OFFLINE=0` |
| LLM ignores search results | Model too small (< 7B) | Pull a 7B+ model → Section 8 |
| `curses` import error | Running inside a non-TTY | Run in a real terminal, not a pipe |
