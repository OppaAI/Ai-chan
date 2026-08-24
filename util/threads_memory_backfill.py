#!/usr/bin/env python3
"""Backfill Threads reply-log exchanges into Aiko's long-term memory.

The Threads reply monitor (interface/mcp_server/social/services/threads.py)
archives every triggered exchange to logs/threads/YYYY-MM-DD.jsonl but,
before interaction memory existed, nothing was written to the memory store.
This tool replays those archives through AikoMemorize.add() so fact
extraction runs exactly like a live turn.

Pairing model (see monitor's _append_reply_log events):
    kind=reply|mention  -> a triggered comment (reply_id = comment id)
    kind=aiko_reply     -> Aiko's response (in_reply_to = comment id)

Only exchanges whose triggering comment came from the owner account
(THREADS_USERNAME, default oppa.ai.bot) are ingested.

Idempotent: ingested aiko_reply ids are recorded in a state file next to
the logs and skipped on re-run, so this is safe to run repeatedly.

Usage:
    # Normal box with llama-server up:
    python util/threads_memory_backfill.py

    # With ollama instead (auto-falls back if :8080 refuses):
    LLM_BASE_URL=http://127.0.0.1:11434/v1 \
    EXTRACT_MODEL=hf.co/unsloth/Ministral-3-14B-Instruct-2512-GGUF:UD-Q4_K_XL \
    AIKO_USER_ID=github_205369547 \
    python util/threads_memory_backfill.py --embedder auto

Dry-run first to inspect pairing without touching memory:
    python util/threads_memory_backfill.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

DEFAULT_LOG_DIR = Path("logs/threads")
DEFAULT_OWNER = "oppa.ai.bot"
DEFAULT_DISPLAY_NAME = "OppaAI"


class OllamaHarrierEmbedder:
    """Drop-in for HarrierEmbedder._embed_texts backed by ollama /api/embed.

    Only used when the regular llama-server embedding endpoint is not
    reachable. The harrier GGUF must be pulled in ollama first.
    """

    def __init__(self, base_url: str, model: str, batch_size: int, timeout: float) -> None:
        import requests

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.batch_size = max(1, batch_size)
        self.timeout = timeout
        self._session = requests.Session()

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        out = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            resp = self._session.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": batch},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            embs = resp.json().get("embeddings") or []
            if len(embs) != len(batch):
                raise RuntimeError(f"ollama returned {len(embs)} embeddings for {len(batch)} inputs")
            arr = np.asarray(embs, dtype=np.float32)
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            out.append(arr / norms)
        return np.vstack(out) if out else np.zeros((0, 1), dtype=np.float32)


def patch_embedder(mode: str) -> None:
    """Redirect HarrierEmbedder HTTP calls when llama-server is unavailable."""
    if mode == "harrier":
        return

    from cognition.memory import vecstore

    original = vecstore.HarrierEmbedder._embed_texts

    def probe_harrier(self) -> bool:
        try:
            self._session.get(f"{self.base_url}/health", timeout=3)
            return True
        except Exception:
            return False

    def ollama_embed_texts(self, texts: list[str]) -> np.ndarray:
        shim = OllamaHarrierEmbedder(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            model=os.getenv("OLLAMA_EMBED_MODEL", "hf.co/mykor/harrier-oss-v1-270m-GGUF:q8_0"),
            batch_size=self.batch_size,
            timeout=self.timeout,
        )
        return shim.embed_batch(texts)

    if mode == "ollama":
        vecstore.HarrierEmbedder._embed_texts = ollama_embed_texts
        return

    # auto: keep harrier when the server answers, else fall back to ollama.
    def auto_embed_texts(self, texts: list[str]) -> np.ndarray:
        if probe_harrier(self):
            return original(self, texts)
        return ollama_embed_texts(self, texts)

    vecstore.HarrierEmbedder._embed_texts = auto_embed_texts


def load_events(path: Path) -> list[dict]:
    events = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def pair_exchanges(events: list[dict], owner: str) -> list[tuple[dict, dict]]:
    triggers = {
        str(e.get("reply_id")): e
        for e in events
        if e.get("kind") in {"reply", "mention"}
        and str(e.get("username") or "").lstrip("@").casefold() == owner
        and e.get("reply_id")
        and e.get("text")
    }
    pairs = []
    for event in events:
        if event.get("kind") != "aiko_reply":
            continue
        trigger = triggers.get(str(event.get("in_reply_to")))
        if trigger and event.get("text"):
            pairs.append((trigger, event))
    return pairs


def _one_line(text: str) -> str:
    return " ".join(str(text or "").split())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "logs",
        nargs="*",
        type=Path,
        help="JSONL log files (default: every logs/threads/*.jsonl, oldest first)",
    )
    parser.add_argument("--user-id", default=os.getenv("AIKO_USER_ID", ""), help="Target Aiko user id")
    parser.add_argument("--display-name", default=DEFAULT_DISPLAY_NAME)
    parser.add_argument("--owner", default=os.getenv("THREADS_USERNAME", DEFAULT_OWNER))
    parser.add_argument(
        "--embedder",
        choices=["auto", "harrier", "ollama"],
        default="auto",
        help="Embedding backend; 'auto' tries llama-server then falls back to ollama",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=None,
        help="Ingested-reply id cache (default: <first log dir>/.backfilled_ids)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show pairings without writing memory")
    parser.add_argument("--limit", type=int, default=0, help="Ingest at most N exchanges (0 = all)")
    args = parser.parse_args()

    paths = args.logs or sorted(DEFAULT_LOG_DIR.glob("*.jsonl"))
    if not paths:
        parser.error(f"no log files found at {DEFAULT_LOG_DIR}/")

    owner = args.owner.lstrip("@").casefold()
    state_path = args.state_file or paths[0].parent / ".backfilled_ids"
    done: set[str] = set()
    if state_path.exists():
        done = {line.strip() for line in state_path.read_text(encoding="utf-8").splitlines() if line.strip()}

    all_pairs: list[tuple[dict, dict]] = []
    for path in paths:
        all_pairs.extend(pair_exchanges(load_events(path), owner))
    pending = [(t, a) for t, a in all_pairs if str(a.get("reply_id")) not in done]
    if args.limit > 0:
        pending = pending[: args.limit]

    print(f"{len(all_pairs)} owner exchange(s) found, {len(pending)} not yet ingested.")
    if args.dry_run:
        for trigger, aiko in pending:
            ts = str(trigger.get("timestamp") or "")[:19]
            print(f"\n[{ts}] {trigger.get('username')}: {str(trigger.get('text'))[:140]}")
            print(f"  -> Aiko: {str(aiko.get('text'))[:140]}")
        return 0
    if not pending:
        return 0

    patch_embedder(args.embedder)

    from cognition.memory.memorize import AikoMemorize

    memorize = AikoMemorize(silent=True)
    extracted = raw_fallback = failed = 0
    with state_path.open("a", encoding="utf-8") as state:
        for trigger, aiko in pending:
            comment = str(trigger.get("text") or "").strip()
            reply = str(aiko.get("text") or "").strip()
            timestamp = str(trigger.get("timestamp") or "")
            day = timestamp[:10] or "undated"
            messages = [
                {"role": "user", "content": f"[Threads {timestamp}] {args.display_name} said: {comment[:2000]}"},
                {"role": "assistant", "content": f"Aiko replied: {reply[:2000]}"},
            ]
            try:
                # _mem.add returns fact ids ([] when extraction judged the
                # exchange to hold nothing memorable); the public add() hides
                # that from callers, so go one layer down to know the truth.
                ids = memorize._mem.add(messages, user_id=args.user_id or None, display_name=args.display_name)
                if ids:
                    memorize._maybe_clear_search_cache()
                    memorize._mem._invalidate_entity_importance(args.user_id)
                    extracted += 1
                    ok = True
                else:
                    # Substantive exchange, zero facts — keep it as one raw
                    # episodic row so the day isn't lossy in recall.
                    digest = (
                        f"[Threads {day}] Exchange with {args.display_name} — "
                        f"they said: {_one_line(comment)[:300]} | "
                        f"Aiko replied: {_one_line(reply)[:300]}"
                    )
                    mem_id = memorize.add_raw(digest, user_id=args.user_id or None)
                    if mem_id:
                        raw_fallback += 1
                        ok = True
                    else:
                        ok = False
            except Exception as exc:
                print(f"  ! failed for reply {aiko.get('reply_id')}: {exc}", file=sys.stderr)
                ok = False
            if ok:
                state.write(f"{aiko.get('reply_id')}\n")
                state.flush()
                time.sleep(0.2)  # gentle pace over SSHFS-backed stores
            else:
                failed += 1

    print(f"Done. extracted={extracted} raw_fallback={raw_fallback} failed={failed} state={state_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
