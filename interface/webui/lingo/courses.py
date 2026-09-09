"""Structured JLPT courses (e.g. N1 Lesson 1) + grammar review decks.

Starter content is curated; can be expanded later. Cards are static so Learn
Courses never waits on the LLM.
"""
from __future__ import annotations

from typing import Dict, List, Optional

# course_id -> {id, title, level, lesson, cards:[{front,back,reading,note}]}
_COURSES: Dict[str, dict] = {}
_GRAMMAR: Dict[str, dict] = {}


def _add_course(course_id: str, title: str, level: str, lesson: int, cards: list) -> None:
    _COURSES[course_id] = {
        "id": course_id,
        "title": title,
        "level": level,
        "lesson": lesson,
        "kind": "course",
        "cards": cards,
    }


def _add_grammar(gid: str, title: str, level: str, cards: list) -> None:
    _GRAMMAR[gid] = {
        "id": gid,
        "title": title,
        "level": level,
        "kind": "grammar",
        "cards": cards,
    }


# --- N1 sample lessons (expand over time) ---
_add_course("n1-lesson-1", "N1 Lesson 1 — 抽象・論理", "N1", 1, [
    {"front": "抽象的", "back": "abstract", "reading": "ちゅうしょうてき", "note": "abstract / conceptual"},
    {"front": "論理的", "back": "logical", "reading": "ろんりてき", "note": ""},
    {"front": "前提", "back": "premise; assumption", "reading": "ぜんてい", "note": ""},
    {"front": "示唆", "back": "implication; suggestion", "reading": "がんい", "note": ""},
    {"front": "矛盾", "back": "contradiction", "reading": "むじゅん", "note": ""},
    {"front": "整合性", "back": "consistency", "reading": "せいごうせい", "note": ""},
    {"front": "客観的", "back": "objective", "reading": "きゃっかんてき", "note": ""},
    {"front": "主観的", "back": "subjective", "reading": "しゅかんてき", "note": ""},
])

_add_course("n1-lesson-2", "N1 Lesson 2 — 社会・制度", "N1", 2, [
    {"front": "制度", "back": "system; institution", "reading": "せいど", "note": ""},
    {"front": "政策", "back": "policy", "reading": "せいさく", "note": ""},
    {"front": "規制", "back": "regulation", "reading": "きせい", "note": ""},
    {"front": "改革", "back": "reform", "reading": "かいかく", "note": ""},
    {"front": "格差", "back": "gap; disparity", "reading": "かくさ", "note": ""},
    {"front": "世論", "back": "public opinion", "reading": "よろん", "note": ""},
])

_add_course("n2-lesson-1", "N2 Lesson 1 — 仕事・連絡", "N2", 1, [
    {"front": "連絡", "back": "contact; get in touch", "reading": "れんらく", "note": ""},
    {"front": "報告", "back": "report", "reading": "ほうこく", "note": ""},
    {"front": "会議", "back": "meeting", "reading": "かいぎ", "note": ""},
    {"front": "部署", "back": "department (company)", "reading": "ぶしょ", "note": ""},
])

# --- Grammar decks ---
_add_grammar("grammar-n1-1", "N1 Grammar — 〜ざるを得ない / 〜かねない", "N1", [
    {"front": "〜ざるを得ない", "back": "cannot help but…; have no choice but to…", "reading": "ざるをえない",
     "note": "V-nai stem + ざるを得ない"},
    {"front": "〜かねない", "back": "might (undesirable outcome)", "reading": "かねない",
     "note": "V-masu stem + かねない"},
    {"front": "〜ないではいられない", "back": "cannot help doing…", "reading": "ないではいられない", "note": ""},
    {"front": "〜极み / 〜極まる", "back": "extremely…", "reading": "きわまる", "note": "N1 formal"},
])

_add_grammar("grammar-n2-1", "N2 Grammar — 〜わけだ / 〜はずだ", "N2", [
    {"front": "〜わけだ", "back": "no wonder; that explains…", "reading": "わけだ", "note": ""},
    {"front": "〜はずだ", "back": "should / expected to…", "reading": "はずだ", "note": ""},
    {"front": "〜ことになる", "back": "it has been decided that…", "reading": "ことになる", "note": ""},
])

_add_grammar("grammar-n3-1", "N3 Grammar — 〜そうだ / 〜ようだ", "N3", [
    {"front": "〜そうだ（様態）", "back": "looks like…", "reading": "そうだ", "note": "V-masu stem + そうだ"},
    {"front": "〜ようだ", "back": "seems / appears…", "reading": "ようだ", "note": ""},
    {"front": "〜らしい", "back": "apparently; typical of…", "reading": "らしい", "note": ""},
])


def list_courses(level: Optional[str] = None) -> List[dict]:
    out = []
    for c in _COURSES.values():
        if level and c["level"] != level:
            continue
        out.append({
            "id": c["id"], "title": c["title"], "level": c["level"],
            "lesson": c["lesson"], "kind": "course",
            "card_count": len(c["cards"]),
        })
    return sorted(out, key=lambda x: (x["level"], x["lesson"]))


def get_course(course_id: str) -> Optional[dict]:
    return _COURSES.get(course_id)


def list_grammar(level: Optional[str] = None) -> List[dict]:
    out = []
    for g in _GRAMMAR.values():
        if level and g["level"] != level:
            continue
        out.append({
            "id": g["id"], "title": g["title"], "level": g["level"],
            "kind": "grammar", "card_count": len(g["cards"]),
        })
    return sorted(out, key=lambda x: x["level"], reverse=True)


def get_grammar(gid: str) -> Optional[dict]:
    return _GRAMMAR.get(gid)
