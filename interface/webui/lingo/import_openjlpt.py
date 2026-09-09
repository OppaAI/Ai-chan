"""OpenJLPT → materials.db. Verified lists only (no LLM). CC BY-SA 4.0.
https://github.com/evanclan/OpenJLPT
"""
from __future__ import annotations
import csv, json, logging, sqlite3, time
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Any
from urllib.request import urlopen

log = logging.getLogger(__name__)
CONTENT = Path(__file__).parent / "content" / "openjlpt"
LEVELS = ("N5", "N4", "N3", "N2", "N1")
OPENJLPT_COMMIT = "c42fd9fa3777bfc1775446f7c418d549dfd6e4cf"
RAW = f"https://raw.githubusercontent.com/evanclan/OpenJLPT/{OPENJLPT_COMMIT}/data/csv"
SRC = "openjlpt"
FETCH_TIMEOUT = 3.0
FETCH_DEADLINE = 10.0


def ensure_csvs() -> dict[str, str]:
    CONTENT.mkdir(parents=True, exist_ok=True)
    failures: dict[str, str] = {}
    deadline = time.monotonic() + FETCH_DEADLINE

    def fetch(
        name: str,
        timeout: float,
        result: Queue[tuple[bytes | None, Exception | None]],
    ) -> None:
        try:
            data = urlopen(f"{RAW}/{name}", timeout=timeout).read()
            result.put((data, None))
        except Exception as exc:
            result.put((None, exc))

    for lv in LEVELS:
        for kind in ("vocab", "grammar"):
            name = f"{kind}-{lv.lower()}.csv"
            path = CONTENT / name
            if path.is_file():
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                failures[name] = "overall fetch deadline reached"
                log.warning("OpenJLPT fetch deadline reached before %s", name)
                continue

            result: Queue[tuple[bytes | None, Exception | None]] = Queue(maxsize=1)
            timeout = min(FETCH_TIMEOUT, remaining)
            worker = Thread(target=fetch, args=(name, timeout, result), daemon=True)
            worker.start()
            worker.join(timeout=timeout)
            if worker.is_alive():
                if time.monotonic() >= deadline:
                    failures[name] = "overall fetch deadline reached"
                    log.warning("OpenJLPT fetch deadline reached during %s", name)
                else:
                    failures[name] = "request timeout reached"
                    log.warning("OpenJLPT fetch timed out: %s", name)
                continue
            if time.monotonic() > deadline:
                failures[name] = "overall fetch deadline reached"
                log.warning("OpenJLPT fetch deadline reached during %s", name)
                continue

            data, error = result.get_nowait()
            if error is not None:
                reason = getattr(error, "reason", None)
                if isinstance(error, TimeoutError) or isinstance(reason, TimeoutError):
                    failures[name] = "request timeout reached"
                    log.warning("OpenJLPT fetch timed out: %s", name)
                else:
                    failures[name] = f"{type(error).__name__}: {error}"
                    log.warning("OpenJLPT fetch failed %s: %s", name, error)
                continue
            assert data is not None
            path.write_bytes(data)
            log.info("OpenJLPT fetched %s (%d)", name, len(data))
    return failures


def _kind(w: str) -> str:
    if w and all("\u3040" <= c <= "\u309f" or c in "ー・" for c in w):
        return "hiragana"
    if w and all("\u30a0" <= c <= "\u30ff" or c in "ー・" for c in w):
        return "katakana"
    return "kanji"


def _vocab() -> dict[str, list]:
    out = {lv: [] for lv in LEVELS}
    for lv in LEVELS:
        p = CONTENT / f"vocab-{lv.lower()}.csv"
        if not p.is_file():
            continue
        with p.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                w = (row.get("word") or "").strip()
                m = (row.get("meanings") or "").strip()
                if not w or not m:
                    continue
                r = (row.get("reading") or "").strip() or w
                note = " / ".join(
                    x for x in (
                        (row.get("example_ja") or "").strip(),
                        (row.get("example_en") or "").strip(),
                    ) if x
                )
                out[lv].append({
                    "front": w, "reading": r,
                    "back": m.split(";")[0].strip() or m,
                    "kind": _kind(w), "note": note,
                })
    return out


def import_openjlpt_into(con: sqlite3.Connection) -> dict[str, Any]:
    fetch_failures = ensure_csvs()
    fetch_limited = any(
        reason in {"request timeout reached", "overall fetch deadline reached"}
        for reason in fetch_failures.values()
    )
    stats: dict[str, Any] = {
        "jlpt_cards": 0,
        "vocab_pool": 0,
        "courses": 0,
        "grammar": 0,
        "fetch_failures": fetch_failures,
        "fetch_limited": fetch_limited,
    }
    cols = {r[1] for r in con.execute("PRAGMA table_info(vocab_pool)").fetchall()}
    if "source" not in cols:
        con.execute(
            "ALTER TABLE vocab_pool ADD COLUMN source TEXT NOT NULL DEFAULT 'spawn'"
        )
        cols.add("source")
    banks = _vocab()
    for level, items in banks.items():
        for i, it in enumerate(items, 1):
            front, back = it["front"], it["back"]
            reading, kind, note = it["reading"], it["kind"], it["note"]
            cid = f"oj-{level.lower()}-v-{i:04d}"
            before = con.total_changes
            con.execute(
                "INSERT OR IGNORE INTO jlpt_cards"
                "(id,level,kind,front,reading,back,note,source) VALUES(?,?,?,?,?,?,?,?)",
                (cid, level, "vocab" if kind == "kanji" else kind,
                 front, reading, back, note, SRC),
            )
            if con.total_changes > before:
                stats["jlpt_cards"] += 1
            cur = con.execute(
                "INSERT OR IGNORE INTO vocab_pool"
                "(front,back,reading,kind,level,used_count,created_at,source)"
                " VALUES(?,?,?,?,?,0,?,?)",
                (front, back, reading, kind, level, time.time(), SRC),
            )
            if cur.rowcount:
                stats["vocab_pool"] += 1
    chunk = 12
    for level, items in banks.items():
        for i in range(0, len(items), chunk):
            part = items[i:i + chunk]
            n = i // chunk + 1
            cards = [{"front": x["front"], "back": x["back"],
                      "reading": x["reading"], "note": x.get("note", "")}
                     for x in part]
            con.execute(
                "INSERT OR REPLACE INTO courses(id,title,level,kind,cards_json)"
                " VALUES(?,?,?,?,?)",
                (f"openjlpt-{level.lower()}-lesson-{n}",
                 f"{level} Lesson {n} (OpenJLPT)", level, "course",
                 json.dumps(cards, ensure_ascii=False)),
            )
            stats["courses"] += 1
    for level in LEVELS:
        p = CONTENT / f"grammar-{level.lower()}.csv"
        if not p.is_file():
            continue
        cards = []
        with p.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                pat = (row.get("pattern") or "").strip()
                mean = (row.get("meaning") or "").strip()
                if not pat or not mean:
                    continue
                note = " · ".join(
                    x for x in (
                        (row.get("formation") or "").strip(),
                        (row.get("example_ja") or "").strip(),
                        (row.get("example_en") or "").strip(),
                    ) if x
                )
                cards.append({"front": pat, "back": mean, "reading": "", "note": note})
        if cards:
            con.execute(
                "INSERT OR REPLACE INTO grammar_decks"
                "(id,title,level,kind,cards_json) VALUES(?,?,?,?,?)",
                (f"openjlpt-grammar-{level.lower()}",
                 f"{level} Grammar (OpenJLPT)", level, "grammar",
                 json.dumps(cards, ensure_ascii=False)),
            )
            stats["grammar"] += 1
    log.info("OpenJLPT import: %s", stats)
    return stats
