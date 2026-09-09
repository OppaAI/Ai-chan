import csv
import sqlite3
import time

from interface.webui.lingo import import_bank, import_openjlpt, lingo_store


def _write_csv(path, fieldnames, row):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def _materials_connection():
    con = sqlite3.connect(":memory:")
    con.execute(
        "CREATE TABLE jlpt_cards (id TEXT PRIMARY KEY, level TEXT, kind TEXT, "
        "front TEXT, reading TEXT, back TEXT, note TEXT, source TEXT)"
    )
    con.execute(
        "CREATE TABLE vocab_pool (front TEXT, back TEXT, reading TEXT, kind TEXT, "
        "level TEXT, used_count INTEGER, created_at REAL, source TEXT, "
        "UNIQUE(front, back))"
    )
    con.execute(
        "CREATE TABLE courses (id TEXT PRIMARY KEY, title TEXT, level TEXT, "
        "kind TEXT, cards_json TEXT)"
    )
    con.execute(
        "CREATE TABLE grammar_decks (id TEXT PRIMARY KEY, title TEXT, level TEXT, "
        "kind TEXT, cards_json TEXT)"
    )
    return con


def test_ensure_csvs_has_hard_timeout(monkeypatch, tmp_path):
    def slow_urlopen(*_args, **_kwargs):
        time.sleep(0.1)
        raise AssertionError("slow request should be abandoned")

    monkeypatch.setattr(import_openjlpt, "CONTENT", tmp_path)
    monkeypatch.setattr(import_openjlpt, "LEVELS", ("N5",))
    monkeypatch.setattr(import_openjlpt, "FETCH_TIMEOUT", 0.01)
    monkeypatch.setattr(import_openjlpt, "FETCH_DEADLINE", 0.015)
    monkeypatch.setattr(import_openjlpt, "urlopen", slow_urlopen)

    started = time.monotonic()
    failures = import_openjlpt.ensure_csvs()

    assert time.monotonic() - started < 0.08
    assert set(failures) == {"vocab-n5.csv", "grammar-n5.csv"}
    assert "/main/" not in import_openjlpt.RAW
    assert import_openjlpt.OPENJLPT_COMMIT in import_openjlpt.RAW


def test_partial_fetch_failure_still_imports_available_csvs(monkeypatch, tmp_path):
    _write_csv(
        tmp_path / "vocab-n5.csv",
        ["word", "reading", "meanings", "example_ja", "example_en"],
        {
            "word": "猫",
            "reading": "ねこ",
            "meanings": "cat",
            "example_ja": "猫です。",
            "example_en": "It is a cat.",
        },
    )
    _write_csv(
        tmp_path / "grammar-n5.csv",
        ["pattern", "meaning", "formation", "example_ja", "example_en"],
        {
            "pattern": "〜です",
            "meaning": "to be",
            "formation": "Noun + です",
            "example_ja": "猫です。",
            "example_en": "It is a cat.",
        },
    )
    failures = {"vocab-n4.csv": "request timeout reached"}
    monkeypatch.setattr(import_openjlpt, "CONTENT", tmp_path)
    monkeypatch.setattr(import_openjlpt, "ensure_csvs", lambda: failures)

    con = _materials_connection()
    stats = import_openjlpt.import_openjlpt_into(con)

    assert stats["jlpt_cards"] == 1
    assert stats["vocab_pool"] == 1
    assert stats["courses"] == 1
    assert stats["grammar"] == 1
    assert stats["fetch_failures"] == failures
    assert stats["fetch_limited"] is True


def test_immediate_network_timeout_is_reported_as_fetch_limit(monkeypatch, tmp_path):
    def timed_out(*_args, **_kwargs):
        raise TimeoutError

    monkeypatch.setattr(import_openjlpt, "CONTENT", tmp_path)
    monkeypatch.setattr(import_openjlpt, "LEVELS", ("N5",))
    monkeypatch.setattr(import_openjlpt, "urlopen", timed_out)

    failures = import_openjlpt.ensure_csvs()

    assert failures == {
        "vocab-n5.csv": "request timeout reached",
        "grammar-n5.csv": "request timeout reached",
    }


def test_fetch_limit_allows_curated_fallback(monkeypatch):
    curated_calls = []
    monkeypatch.setattr(
        import_openjlpt,
        "import_openjlpt_into",
        lambda _con: {"vocab_pool": 1, "fetch_limited": True},
    )
    monkeypatch.setattr(
        import_bank,
        "import_curated_into",
        lambda _con: curated_calls.append(True) or {"vocab_pool": 1},
    )

    lingo_store._seed_materials(sqlite3.connect(":memory:"))

    assert curated_calls == [True]
