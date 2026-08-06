"""Phase 19 unit tests — copy into repo tests/ and adapt imports if needed."""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Allow importing patches when run from artifacts tree
_PATCHES = Path(__file__).resolve().parents[1] / "patches"
if str(_PATCHES) not in sys.path:
    sys.path.insert(0, str(_PATCHES))

from arousal_and_filter import (  # noqa: E402
    apply_neg_hard_filter,
    arousal_rank_bonus,
    infer_arousal_score,
)


def test_infer_arousal_high():
    assert infer_arousal_score("He panicked about the emergency glitch!!") == 2


def test_infer_arousal_mid():
    assert infer_arousal_score("Oppa is anxious about the deadline") == 1


def test_infer_arousal_low():
    assert infer_arousal_score("A calm quiet evening, routine work") == -1


def test_infer_arousal_neutral():
    assert infer_arousal_score("Oppa prefers dark mode") == 0


def test_arousal_rank_bonus_respects_flag(monkeypatch):
    monkeypatch.setenv("MEMORY_AROUSAL_ENABLED", "0")
    assert arousal_rank_bonus(2) == 0.0
    monkeypatch.setenv("MEMORY_AROUSAL_ENABLED", "1")
    monkeypatch.setenv("MEMORY_AROUSAL_RANK_WEIGHT", "0.08")
    assert abs(arousal_rank_bonus(2) - 0.08) < 1e-9


def test_neg_hard_filter_drops_unsolicited(monkeypatch):
    monkeypatch.setenv("MEMORY_NEG_HARD_FILTER", "1")
    monkeypatch.setenv("MEMORY_NEG_HARD_THRESHOLD", "-1")
    mems = [
        {"id": "1", "memory": "Oppa lost his wallet", "valence_score": -2, "entities": []},
        {"id": "2", "memory": "Oppa prefers dark mode", "valence_score": 0, "entities": []},
    ]
    out = apply_neg_hard_filter(mems, query="what do you know about my settings")
    ids = [m["id"] for m in out]
    assert "1" not in ids
    assert "2" in ids


def test_neg_hard_filter_keeps_on_overlap(monkeypatch):
    monkeypatch.setenv("MEMORY_NEG_HARD_FILTER", "1")
    mems = [
        {"id": "1", "memory": "Oppa lost his wallet last week", "valence_score": -2, "entities": ["wallet"]},
    ]
    out = apply_neg_hard_filter(mems, query="did I lose my wallet")
    assert len(out) == 1


def test_neg_hard_filter_disabled(monkeypatch):
    monkeypatch.setenv("MEMORY_NEG_HARD_FILTER", "0")
    mems = [{"id": "1", "memory": "bad day", "valence_score": -2, "entities": []}]
    assert len(apply_neg_hard_filter(mems, query="hello")) == 1
