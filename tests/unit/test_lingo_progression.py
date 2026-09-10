"""Tests for the lesson/test/final progression engine + earlier bugfixes."""
import importlib

import pytest

# NOTE: `interface.webui.lingo.__init__` re-exports the FastAPI `router`
# object, so plain `from ... import router` binds the APIRouter, not the
# module. import_module reliably returns the module itself.
router = importlib.import_module("interface.webui.lingo.router")
from interface.webui.lingo import models as lingo_models


def test_deck_sort_key_numeric_order():
    ids = ["openjlpt-n5-lesson-10", "openjlpt-n5-lesson-2", "openjlpt-n5-lesson-1"]
    assert sorted(ids, key=router._deck_sort_key) == [
        "openjlpt-n5-lesson-1",
        "openjlpt-n5-lesson-2",
        "openjlpt-n5-lesson-10",
    ]


def test_deck_sort_key_basics_first():
    ids = ["openjlpt-grammar-n5", "n5-grammar-basics"]
    assert sorted(ids, key=router._deck_sort_key) == [
        "n5-grammar-basics",
        "openjlpt-grammar-n5",
    ]


def test_normalize_strips_spaces_case_and_wave_dash():
    assert router._normalize_test_answer("  ねこ ") == "ねこ"
    assert router._normalize_test_answer("〜ます") == "ます"
    assert router._normalize_test_answer("Taberu") == "taberu"


def test_accepted_answers_cover_reading_and_grammar_variants():
    card = {"front": "水", "back": "water", "reading": "みず"}
    accepted = router._accepted_test_answers(card)
    assert "みず" in accepted
    assert "水" in accepted

    gcard = {"front": "〜が（but）", "back": "but; however", "reading": ""}
    gaccepted = router._accepted_test_answers(gcard)
    # Full pattern, bare particle, and paren-stripped forms all pass.
    assert router._normalize_test_answer("〜が（but）") in gaccepted
    assert "が" in gaccepted


def test_word_of_day_allows_missing_date():
    # Cache-hit path rebuilds the model without `date`; must not 500.
    w = lingo_models.WordOfDayResponse(card_id=1, hiragana="ねこ", meaning="cat")
    assert w.date == ""


def test_grade_and_advance_pass_moves_lesson(monkeypatch):
    monkeypatch.setattr(router, "_stock_lesson_cards", lambda *a, **k: None)
    awarded = []
    monkeypatch.setattr(router, "_award_xp", lambda uid, n: awarded.append(n) or 99)
    monkeypatch.setattr(router, "record_practice", lambda uid: None)
    monkeypatch.setattr(
        router,
        "_track_progress",
        lambda uid, track: lingo_models.LessonProgress(
            level="N5", track=track, current_lesson=2, lessons_total=56
        ),
    )
    from interface.webui.lingo import lingo_store

    advanced = []
    monkeypatch.setattr(
        lingo_store, "set_lesson_progress", lambda uid, t, l, c: advanced.append(c) or c
    )
    # router looks set_lesson_progress up on lingo_store at call time via
    # `from .lingo_store import set_lesson_progress` INSIDE the function, so
    # patching the attribute works.
    orig_grade = router._grade_and_advance

    cards = {
        "c0": {"front": "ねこ", "back": "cat", "reading": "ねこ"},
        "c1": {"front": "水", "back": "water", "reading": "みず"},
    }
    answers = [lingo_models.TestAnswer(qid="c0", answer="ねこ"), lingo_models.TestAnswer(qid="c1", answer="みず")]
    resp = orig_grade("u1", "vocab", "N5", cards, answers,
                      is_final=False, lesson_pos=1, lessons_total=56)
    assert isinstance(resp, lingo_models.TestSubmitResponse)
    assert resp.passed and resp.correct == 2 and resp.total == 2
    assert resp.xp == 4
    assert awarded == [4]
    assert advanced == [2]


def test_grade_final_pass_levels_up(monkeypatch):
    monkeypatch.setattr(router, "_stock_lesson_cards", lambda *a, **k: None)
    monkeypatch.setattr(router, "_award_xp", lambda uid, n: n)
    monkeypatch.setattr(router, "record_practice", lambda uid: None)
    monkeypatch.setattr(
        router,
        "_track_progress",
        lambda uid, track: lingo_models.LessonProgress(
            level="N5", track=track, current_lesson=57, lessons_total=56,
            final_unlocked=True,
        ),
    )
    moved = []
    monkeypatch.setattr(router, "update_last_level", lambda uid, lvl: moved.append(lvl))
    monkeypatch.setattr(router, "_ordered_track_decks", lambda track, level: [])

    cards = {"openjlpt-n5-lesson-1#0": {"front": "ねこ", "back": "cat", "reading": "ねこ"}}
    answers = [lingo_models.TestAnswer(qid="openjlpt-n5-lesson-1#0", answer="ねこ")]
    resp = router._grade_and_advance("u1", "vocab", "N5", cards, answers, is_final=True)
    assert resp.passed
    assert resp.new_level == "N4"
    assert moved == ["N4"]
    assert resp.progress.level == "N4"


def test_get_xp_starts_at_zero():
    data = router.get_xp("definitely-not-a-real-user-12345")
    assert data["xp"] == 0
    assert data["level"] == 1
    assert data["next_level_xp"] == 100


def test_clean_tts_text_skips_pattern_markers():
    assert router._clean_tts_text("〜です") == "です"
    assert router._clean_tts_text("〜ます") == "ます"
    assert router._clean_tts_text("〜が（but）") == "が"
    assert router._clean_tts_text("〜たことがある") == "たことがある"
    assert router._clean_tts_text("ねこ") == "ねこ"
