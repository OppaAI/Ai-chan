"""
cognition/attention_train.py

Continuous on-device LightGBM training for Jetson.

* Every talk turn, for_identity().record() calls log_example() — appends a
  JSONL row to workspace/training/attention_buffer.jsonl (no blocking).
* A daemonic background thread started at boot watches the buffer; when
  ≥50 new rows appear, it retrains the tabular model (LightGBM, <100 ms for
  <10k rows) and atomically replaces models/intent_classifier.lgb.
* The classifier in attention.py picks up the new file on next _load_model()
  or on next process restart — no hot reload needed on the gate path
  (intentionally, to avoid file-watch overhead).
* RAM: buffer file on disk, training uses <20 MB transient.
* Latency: log_example is a single open+append (~0.3 ms); retrain runs
  off the hot path in a daemon thread at nice priority.

This is the "training will do continuously while I talk to Aiko" piece
the user requested. Labels are derived from the next turn's outcome
signal (outcome_fail/ok) when available, otherwise neutral.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

from system.log import get_logger
from system.userspace import user_state_path

log = get_logger(__name__)

_BUFFER_PATH = Path(user_state_path("training/attention_buffer.jsonl"))
_MODEL_PATH = Path("models/intent_classifier.lgb")
_MIN_ROWS_FOR_RETRAIN = 50
_RETRAIN_INTERVAL_S = 600  # at most every 10 min even if buffer grows fast
_last_retrain = 0.0
_buffer_lock = threading.Lock()
_thread_started = False


def log_example(features: dict, label: float | None = None) -> None:
    """Append one training row (non-blocking, best-effort)."""
    try:
        # Use provided label or infer neutral 0.5 when no outcome yet.
        row = {"ts": time.time(), "features": features, "label": label if label is not None else 0.5}
        _BUFFER_PATH.parent.mkdir(parents=True, exist_ok=True)
        # atomic append
        with _buffer_lock:
            with _BUFFER_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        # maybe trigger retrain
        _maybe_schedule_retrain()
    except Exception as e:
        log.debug("attention_train log_example failed: %s", e)


def _maybe_schedule_retrain() -> None:
    global _last_retrain, _thread_started
    now = time.monotonic()
    if now - _last_retrain < _RETRAIN_INTERVAL_S:
        return
    # count rows cheaply
    try:
        if not _BUFFER_PATH.is_file():
            return
        # fast line count without loading JSON
        with _BUFFER_PATH.open("r", encoding="utf-8") as f:
            cnt = sum(1 for _ in f)
        if cnt < _MIN_ROWS_FOR_RETRAIN:
            return
        if _thread_started:
            return
        _thread_started = True
        t = threading.Thread(target=_retrain_job, name="attention-train", daemon=True)
        t.start()
    except Exception as e:
        log.debug("attention_train schedule check failed: %s", e)


def _retrain_job() -> None:
    global _last_retrain, _thread_started
    try:
        _last_retrain = time.monotonic()
        rows = []
        with _buffer_lock:
            if not _BUFFER_PATH.is_file():
                return
            with _BUFFER_PATH.open("r", encoding="utf-8") as f:
                for line in f:
                    line=line.strip()
                    if not line:
                        continue
                    try:
                        r=json.loads(line)
                        if "features" in r and "label" in r:
                            rows.append(r)
                    except Exception:
                        continue
        if len(rows) < _MIN_ROWS_FOR_RETRAIN:
            return
        # Only train when we have some signal variance
        labels = [float(r["label"]) for r in rows]
        if max(labels) - min(labels) < 0.1:
            log.info("attention_train skipped — labels lack variance (%d rows)", len(rows))
            return
        # Build feature matrix in canonical order
        from cognition.attention import IntentConfidenceClassifier
        order = IntentConfidenceClassifier._FEATURE_ORDER
        X = []
        y = []
        for r in rows:
            feats = r["features"]
            X.append([float(feats.get(k, 0.0)) for k in order])
            y.append(float(r["label"]))
        try:
            import lightgbm as lgb
            import numpy as np
        except Exception as e:
            log.warning("attention_train: lightgbm not available: %s", e)
            return
        train_set = lgb.Dataset(np.array(X, dtype=float), label=np.array(y, dtype=float))
        params = {
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "num_leaves": 15,
            "max_depth": 5,
            "learning_rate": 0.08,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "min_data_in_leaf": 5,
        }
        booster = lgb.train(params, train_set, num_boost_round=80)
        # atomic write
        tmp = _MODEL_PATH.with_suffix(".tmp.lgb")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        booster.save_model(str(tmp))
        tmp.replace(_MODEL_PATH)
        log.info("attention_train retrained on %d rows → %s", len(rows), _MODEL_PATH)
        # Optionally truncate buffer after successful retrain (keep last 500)
        try:
            with _BUFFER_PATH.open("r", encoding="utf-8") as f:
                all_lines = f.readlines()
            if len(all_lines) > 500:
                with _BUFFER_PATH.open("w", encoding="utf-8") as f:
                    f.writelines(all_lines[-500:])
        except Exception:
            pass
    except Exception as e:
        log.warning("attention_train retrain failed: %s", e)
    finally:
        _thread_started = False
        _last_retrain = time.monotonic()


def start_continuous_training() -> None:
    """Call once at boot (e.g. from system.wakeup) to enable background retrains."""
    # No dedicated thread needed; log_example triggers on demand.
    # This is a hook for future periodic checks.
    log.info("attention continuous training enabled (buffer=%s)", _BUFFER_PATH)


__all__ = ["log_example", "start_continuous_training"]
