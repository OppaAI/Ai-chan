"""Embedded curated JLPT CSV packs."""
from __future__ import annotations

from . import pack_n5, pack_n4, pack_n3, pack_n2, pack_n1

BANK_CSV = {
    "N5": pack_n5.CSV,
    "N4": pack_n4.CSV,
    "N3": pack_n3.CSV,
    "N2": pack_n2.CSV,
    "N1": pack_n1.CSV,
}

GRAMMAR_JSON = r"""[
  {"id": "grammar-n5-1", "title": "N5 Grammar · Particles & Polite", "level": "N5", "kind": "grammar",
   "cards": [
    {"front": "〜です", "back": "is/am/are (polite)", "reading": "", "note": "X は Y です"},
    {"front": "〜ます", "back": "polite verb ending", "reading": "", "note": "食べます"},
    {"front": "〜か", "back": "question particle", "reading": "", "note": "ねこですか"},
    {"front": "〜の", "back": "possessive / noun link", "reading": "", "note": "私の本"},
    {"front": "〜に", "back": "to / at (time/place)", "reading": "", "note": "学校に行く"},
    {"front": "〜を", "back": "object particle", "reading": "", "note": "水を飲む"},
    {"front": "〜が", "back": "subject particle", "reading": "", "note": "ねこがいる"},
    {"front": "〜は", "back": "topic particle", "reading": "", "note": "私は学生です"},
    {"front": "〜も", "back": "also / too", "reading": "", "note": "私も行く"},
    {"front": "〜へ", "back": "to (direction)", "reading": "", "note": "日本へ行く"}
  ]},
  {"id": "grammar-n4-1", "title": "N4 Grammar · Obligation & Experience", "level": "N4", "kind": "grammar",
   "cards": [
    {"front": "〜なければならない", "back": "must; have to", "reading": "", "note": ""},
    {"front": "〜てはいけない", "back": "must not", "reading": "", "note": ""},
    {"front": "〜てもいい", "back": "may; allowed to", "reading": "", "note": ""},
    {"front": "〜たことがある", "back": "have done before", "reading": "", "note": ""},
    {"front": "〜ようになる", "back": "come to be able to", "reading": "", "note": ""},
    {"front": "〜すぎる", "back": "too much", "reading": "", "note": ""}
  ]},
  {"id": "grammar-n3-1", "title": "N3 Grammar · Appearances", "level": "N3", "kind": "grammar",
   "cards": [
    {"front": "〜そうだ（様態）", "back": "looks like…", "reading": "そうだ", "note": ""},
    {"front": "〜ようだ", "back": "seems / appears…", "reading": "ようだ", "note": ""},
    {"front": "〜らしい", "back": "apparently; typical of…", "reading": "らしい", "note": ""},
    {"front": "〜にとって", "back": "for (someone)", "reading": "", "note": ""},
    {"front": "〜について", "back": "about; concerning", "reading": "", "note": ""},
    {"front": "〜によって", "back": "by means of; depending on", "reading": "", "note": ""}
  ]},
  {"id": "grammar-n2-1", "title": "N2 Grammar · Formal Connectors", "level": "N2", "kind": "grammar",
   "cards": [
    {"front": "〜かねない", "back": "might (undesirable)", "reading": "かねない", "note": ""},
    {"front": "〜ざるを得ない", "back": "cannot help but", "reading": "ざるをえない", "note": ""},
    {"front": "〜つつある", "back": "be in the process of", "reading": "つつある", "note": ""},
    {"front": "〜に伴い", "back": "along with", "reading": "にともない", "note": ""},
    {"front": "〜一方だ", "back": "increasingly", "reading": "いっぽうだ", "note": ""}
  ]},
  {"id": "grammar-n1-1", "title": "N1 Grammar · Advanced Forms", "level": "N1", "kind": "grammar",
   "cards": [
    {"front": "〜極まる", "back": "extremely…", "reading": "きわまる", "note": ""},
    {"front": "〜ずにはおかない", "back": "bound to…", "reading": "", "note": ""},
    {"front": "〜ないではいられない", "back": "cannot help doing", "reading": "", "note": ""},
    {"front": "〜とはいえ", "back": "though; nevertheless", "reading": "", "note": ""},
    {"front": "〜ならでは", "back": "unique to…", "reading": "", "note": ""},
    {"front": "〜をよそに", "back": "ignoring; despite", "reading": "", "note": ""},
    {"front": "〜を皮切りに", "back": "starting with", "reading": "をかわきりに", "note": ""},
    {"front": "〜に即して", "back": "in accordance with", "reading": "にそくして", "note": ""}
  ]}
]"""
