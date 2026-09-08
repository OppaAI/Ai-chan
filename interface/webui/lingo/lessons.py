"""
Lingo lesson decks (static curated content for Learn mode).

Unlike SRS cards (auto-extracted from conversation into the per-user DB),
these decks are fixed study material: kana charts, starter words, daily
phrases, and N5 kanji. Served read-only via GET /lessons and
GET /lessons/{deck_id} — no SRS writes, no auth side effects.
"""
from typing import Dict, List

# Each card: (front, back, reading). Reading feeds the app's speak button.
_DECKS: Dict[str, dict] = {}


def _kana_deck(deck_id: str, title: str, subtitle: str, rows: List[tuple]) -> None:
    cards = [{"front": kana, "back": roma, "reading": kana} for kana, roma in rows]
    _DECKS[deck_id] = {
        "id": deck_id, "title": title, "subtitle": subtitle,
        "kind": "kana", "cards": cards,
    }


_kana_deck("hiragana", "Hiragana", "46 basic characters", [
    ("あ", "a"), ("い", "i"), ("う", "u"), ("え", "e"), ("お", "o"),
    ("か", "ka"), ("き", "ki"), ("く", "ku"), ("け", "ke"), ("こ", "ko"),
    ("さ", "sa"), ("し", "shi"), ("す", "su"), ("せ", "se"), ("そ", "so"),
    ("た", "ta"), ("ち", "chi"), ("つ", "tsu"), ("て", "te"), ("と", "to"),
    ("な", "na"), ("に", "ni"), ("ぬ", "nu"), ("ね", "ne"), ("の", "no"),
    ("は", "ha"), ("ひ", "hi"), ("ふ", "fu"), ("へ", "he"), ("ほ", "ho"),
    ("ま", "ma"), ("み", "mi"), ("む", "mu"), ("め", "me"), ("も", "mo"),
    ("や", "ya"), ("ゆ", "yu"), ("よ", "yo"),
    ("ら", "ra"), ("り", "ri"), ("る", "ru"), ("れ", "re"), ("ろ", "ro"),
    ("わ", "wa"), ("を", "wo"), ("ん", "n"),
])

_kana_deck("katakana", "Katakana", "46 basic characters", [
    ("ア", "a"), ("イ", "i"), ("ウ", "u"), ("エ", "e"), ("オ", "o"),
    ("カ", "ka"), ("キ", "ki"), ("ク", "ku"), ("ケ", "ke"), ("コ", "ko"),
    ("サ", "sa"), ("シ", "shi"), ("ス", "su"), ("セ", "se"), ("ソ", "so"),
    ("タ", "ta"), ("チ", "chi"), ("ツ", "tsu"), ("テ", "te"), ("ト", "to"),
    ("ナ", "na"), ("ニ", "ni"), ("ヌ", "nu"), ("ネ", "ne"), ("ノ", "no"),
    ("ハ", "ha"), ("ヒ", "hi"), ("フ", "fu"), ("ヘ", "he"), ("ホ", "ho"),
    ("マ", "ma"), ("ミ", "mi"), ("ム", "mu"), ("メ", "me"), ("モ", "mo"),
    ("ヤ", "ya"), ("ユ", "yu"), ("ヨ", "yo"),
    ("ラ", "ra"), ("リ", "ri"), ("ル", "ru"), ("レ", "re"), ("ロ", "ro"),
    ("ワ", "wa"), ("ヲ", "wo"), ("ン", "n"),
])


def _word_deck(deck_id: str, title: str, subtitle: str, kind: str, rows: List[tuple]) -> None:
    cards = [{"front": f, "back": b, "reading": r} for f, b, r in rows]
    _DECKS[deck_id] = {
        "id": deck_id, "title": title, "subtitle": subtitle,
        "kind": kind, "cards": cards,
    }


_word_deck("words", "Starter Words", "20 everyday nouns", "words", [
    ("水", "water", "みず"), ("本", "book", "ほん"),
    ("人", "person", "ひと"), ("学校", "school", "がっこう"),
    ("先生", "teacher", "せんせい"), ("友達", "friend", "ともだち"),
    ("家族", "family", "かぞく"), ("食べ物", "food", "たべもの"),
    ("飲み物", "drink", "のみもの"), ("車", "car", "くるま"),
    ("電車", "train", "でんしゃ"), ("駅", "station", "えき"),
    ("病院", "hospital", "びょういん"), ("銀行", "bank", "ぎんこう"),
    ("天気", "weather", "てんき"), ("雨", "rain", "あめ"),
    ("雪", "snow", "ゆき"), ("海", "sea", "うみ"),
    ("山", "mountain", "やま"), ("空", "sky", "そら"),
])

_word_deck("phrases", "Daily Phrases", "12 phrases that always work", "phrases", [
    ("おはようございます", "good morning", "おはようございます"),
    ("こんにちは", "hello, good afternoon", "こんにちは"),
    ("こんばんは", "good evening", "こんばんは"),
    ("おやすみなさい", "good night", "おやすみなさい"),
    ("ありがとうございます", "thank you", "ありがとうございます"),
    ("すみません", "excuse me, sorry", "すみません"),
    ("はじめまして", "nice to meet you", "はじめまして"),
    ("よろしくおねがいします", "please treat me well", "よろしくおねがいします"),
    ("お元気ですか", "how are you?", "おげんきですか"),
    ("元気です", "I'm fine", "げんきです"),
    ("いくらですか", "how much is it?", "いくらですか"),
    ("どこですか", "where is it?", "どこですか"),
])

_word_deck("kanji", "N5 Kanji", "15 first kanji with readings", "kanji", [
    ("日", "day, sun — ニチ / ひ", "ひ"),
    ("本", "book, origin — ホン / もと", "ほん"),
    ("人", "person — ジン / ひと", "ひと"),
    ("水", "water — スイ / みず", "みず"),
    ("火", "fire — カ / ひ", "ひ"),
    ("木", "tree — モク / き", "き"),
    ("金", "gold, money — キン / かね", "かね"),
    ("土", "earth — ド / つち", "つち"),
    ("月", "moon, month — ゲツ / つき", "つき"),
    ("年", "year — ネン / とし", "とし"),
    ("時", "time — ジ / とき", "とき"),
    ("分", "minute — ブン / ふん", "ふん"),
    ("上", "up, above — ジョウ / うえ", "うえ"),
    ("下", "down, below — カ / した", "した"),
    ("中", "middle, inside — チュウ / なか", "なか"),
])


def list_decks() -> List[dict]:
    """Deck metadata without cards (for the lesson picker)."""
    return [
        {"id": d["id"], "title": d["title"], "subtitle": d["subtitle"],
         "kind": d["kind"], "card_count": len(d["cards"])}
        for d in _DECKS.values()
    ]


def get_deck(deck_id: str) -> dict | None:
    """Full deck with cards, or None for unknown ids."""
    return _DECKS.get(deck_id)
