"""
Vocabulary extraction and memory integration.

Split out of the old lingo_srs.py per lingo_architecture.md: this module
owns Japanese text parsing (VocabExtractor) and the bridge into Aiko's
long-term memory (MemoryBridge). It depends on srs.py for the
LingoVocabCard/ReviewGrade types but has no knowledge of SQLite or the
FastAPI router, so it can be tested (and reused) on its own.

=====================================================================
BUGFIX PASS:
  1. _hiragana_to_romaji() only handled single hiragana characters, so
     any word with youon (combined sounds like きゃ/しゅ/ちょ) or small
     kana (ぁぃぅぇぉ) produced garbled romaji (e.g. "きゃ" -> "kya" was
     actually rendered as "ki" + "や" = "kiya"). Rewrote as a proper
     longest-match tokenizer over an expanded mapping table, including
     small tsu (っ) for consonant gemination and the long vowel mark (ー).
  2. VocabExtractor accepted an `llm_client` in __init__ but never
     actually called it -- unknown kanji just silently returned None
     from _kanji_lookup(), so anything outside the ~15-entry
     COMMON_READINGS cache was dropped and never added to the SRS
     queue. Implemented the fallback lookup.
=====================================================================
"""

import json
import logging
import re
from typing import Optional, List, Dict

from .srs import LingoVocabCard, ReviewGrade

log = logging.getLogger(__name__)


class VocabExtractor:
    # Function words and verb-ending fragments that the regex segmenter
    # splits off as bare hiragana runs. Carding them produced the
    # meaningless "[hiragana: は]" style quiz questions.
    _FUNCTION_WORDS = frozenset({
        "は", "が", "を", "に", "へ", "と", "で", "も", "の", "ね", "よ",
        "か", "な", "や", "わ", "ぞ", "ぜ", "さ", "て", "って", "たら",
        "れば", "ても", "でも", "から", "まで", "より", "だけ", "しか",
        "ばかり", "けれど", "けれども", "けど", "のに", "ので", "ため",
        "みたい", "とか", "など", "なんて", "ながら", "たり", "だり",
        "そうだ", "ようだ", "らしい", "べき", "はず", "わけ", "ところ",
        "まま", "ごと", "たび", "ごろ", "ころ", "あたり", "について",
        "によって", "に対して", "として", "ない", "ます", "です",
    })

    """
    Parse Japanese text and extract vocabulary for SRS cards.

    Strategy:
    1. Segment Japanese text into words (use MeCab-like heuristic or regex)
    2. Classify each token: kanji word, hiragana verb/adj, etc.
    3. Match against known dictionary (fallback: LLM lookup)
    4. Create card with reading + meaning
    """

    # Common kanji readings cache (subset for performance; full dict loaded on demand)
    COMMON_READINGS = {
        "日": {"hiragana": ["ひ", "か"], "meaning": "day, sun"},
        "本": {"hiragana": ["ほん"], "meaning": "book, origin"},
        "人": {"hiragana": ["ひと", "にん"], "meaning": "person"},
        "学": {"hiragana": ["がく"], "meaning": "study, learning"},
        "校": {"hiragana": ["こう"], "meaning": "school"},
        "生": {"hiragana": ["せい", "い"], "meaning": "life, birth"},
        "先": {"hiragana": ["せん"], "meaning": "before, ahead"},
        "中": {"hiragana": ["なか", "ちゅう"], "meaning": "middle, inside"},
        "大": {"hiragana": ["だい", "おお"], "meaning": "large, big"},
        "小": {"hiragana": ["しょう", "ちいさ"], "meaning": "small"},
        "朝": {"hiragana": ["あさ"], "meaning": "morning"},
        "子": {"hiragana": ["こ", "し"], "meaning": "child, offspring"},
        "食": {"hiragana": ["しょく", "た"], "meaning": "eat, food"},
        "べ": {"hiragana": ["べ"], "meaning": "eat (verb suffix)"},
        "水": {"hiragana": ["みず"], "meaning": "water"},
    }

    # Cache of successful LLM lookups so we don't re-ask the model for the
    # same kanji in every conversation turn.
    _llm_lookup_cache: Dict[str, Optional[dict]] = {}

    def __init__(self, llm_client=None, llm_model: Optional[str] = None):
        """
        llm_client: optional client exposing `.chat.completions.create(...)`
                    (e.g. Aiko's `think._client`), used as a fallback for
                    kanji not present in COMMON_READINGS.
        llm_model:  model name to use for the fallback lookup. If omitted,
                    the fallback is skipped even when llm_client is set,
                    since we don't want to guess at a model name.
        """
        self.llm_client = llm_client
        self.llm_model = llm_model

    def extract_vocab(self, japanese_text: str, context: str = "") -> List[LingoVocabCard]:
        """
        Parse Japanese text and extract vocabulary cards.
        Returns list of LingoVocabCard (ready to add to SRS).
        """
        if not japanese_text:
            return []

        # Remove furigana markup (e.g., "漢字{かんじ}" → "漢字")
        clean_text = re.sub(r'\{[^}]+\}', '', japanese_text)
        # Remove actions/emoji (already done in sanitize, but double-check)
        clean_text = re.sub(r'\*[^*]*\*', '', clean_text)

        cards = []
        seen = set()  # Dedup by (hiragana, meaning) tuple

        # Segment by kanji clusters and hiragana words
        tokens = self._segment_tokens(clean_text)

        for token in tokens:
            if not token or not self._is_japanese(token):
                continue
            if token in self._FUNCTION_WORDS:
                continue
            # Single-kana runs are particles/endings split off by the
            # segmenter, never standalone vocab. Single KANJI still go
            # through lookup (they can be real words like 日/水).
            if len(token) <= 1 and not self._has_kanji(token):
                continue
            # Hiragana runs have no spaces: greedily match known words
            # inside them (e.g. ねこがすきです -> ねこ + すき + です),
            # skipping particles/endings between matches.
            if self._is_hiragana(token) and token not in self._HIRAGANA_MEANINGS:
                self._scan_hiragana_run(token, context, seen, cards)
                continue

            # Try to extract vocab from token
            vocab = self._extract_token(token, context)
            if vocab:
                key = (vocab.hiragana, vocab.meaning)
                if key not in seen:
                    cards.append(vocab)
                    seen.add(key)

        log.debug(f"Extracted {len(cards)} vocab cards from: {japanese_text[:50]}")
        return cards

    def _segment_tokens(self, text: str) -> List[str]:
        """
        Segment Japanese text into tokens (rough morphological split).
        Simple heuristic: split on particle-like hiragana; keep kanji clusters.
        """
        # Patterns: kanji clusters, hiragana words, special chars
        pattern = r'[\u4e00-\u9fff]+|[\u3040-\u309f]+|[\u30a0-\u30ff]+'
        return re.findall(pattern, text)

    def _extract_token(self, token: str, context: str = "") -> Optional[LingoVocabCard]:
        """
        Attempt to extract vocab from single token.
        Tries kanji lookup, then hiragana lookup, then LLM fallback.
        """
        # All hiragana: only the curated common words become cards.
        # Anything else (particles, endings, fragments) is skipped — the
        # old generic "[hiragana: ...]" fallback filled the SRS queue and
        # Word of the Day with meaningless questions.
        if self._is_hiragana(token):
            if token in self._HIRAGANA_MEANINGS:
                return self._hiragana_card(token, context)
            return None

        # Kanji present: try lookup
        if self._has_kanji(token):
            card = self._kanji_lookup(token)
            if card:
                card.context = context
                return card

        return None

    def _kanji_lookup(self, token: str) -> Optional[LingoVocabCard]:
        """Look up kanji token in cache; fallback to LLM if needed"""
        # Exact match in cache
        if token in self.COMMON_READINGS:
            reading_info = self.COMMON_READINGS[token]
            hiragana = reading_info["hiragana"][0]  # Primary reading
            return LingoVocabCard(
                kanji=token,
                hiragana=hiragana,
                romaji=self._hiragana_to_romaji(hiragana),
                meaning=reading_info["meaning"],
                pos="noun"
            )

        # Use the LLM fallback for anything outside the small static cache
        # above. Without this, essentially all intermediate/advanced
        # vocabulary was silently dropped and never made it into the SRS
        # queue.
        return self._llm_lookup(token)

    def _llm_lookup(self, token: str) -> Optional[LingoVocabCard]:
        if token in self._llm_lookup_cache:
            cached = self._llm_lookup_cache[token]
            if not cached:
                return None
            return LingoVocabCard(
                kanji=token,
                hiragana=cached["hiragana"],
                romaji=self._hiragana_to_romaji(cached["hiragana"]),
                meaning=cached["meaning"],
                pos=cached.get("pos", "unknown"),
            )

        if not self.llm_client or not self.llm_model:
            return None

        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[{
                    "role": "system",
                    "content": (
                        "You are a Japanese dictionary. Given a single word or "
                        "kanji compound, respond with ONLY a JSON object with "
                        "keys 'hiragana' (the primary reading, hiragana only), "
                        "'meaning' (a short English gloss), and 'pos' "
                        "(part of speech: noun/verb/adjective/adverb/particle/"
                        "unknown). If you don't recognize the word, respond "
                        "with {\"hiragana\": \"\", \"meaning\": \"\", \"pos\": \"unknown\"}."
                    )
                }, {
                    "role": "user",
                    "content": token
                }],
                response_format={"type": "json_object"},
                timeout=5.0,
            )
            content = response.choices[0].message.content
            data = json.loads(content)
            if data.get("pos") == "particle":
                self._llm_lookup_cache[token] = None
                return None
            hiragana = (data.get("hiragana") or "").strip()
            meaning = (data.get("meaning") or "").strip()

            if not hiragana or not meaning:
                self._llm_lookup_cache[token] = None
                return None

            self._llm_lookup_cache[token] = {
                "hiragana": hiragana,
                "meaning": meaning,
                "pos": data.get("pos", "unknown"),
            }
            return LingoVocabCard(
                kanji=token,
                hiragana=hiragana,
                romaji=self._hiragana_to_romaji(hiragana),
                meaning=meaning,
                pos=data.get("pos", "unknown"),
            )
        except Exception as e:
            log.warning(f"LLM vocab lookup failed for '{token}': {e}")
            # Don't cache failures from transient errors (timeouts, etc.) --
            # only cache confirmed "the model doesn't know this word" results.
            return None

    # Curated common kana-only words with real glosses. Only these (plus
    # kanji lookups) may become cards — everything else hiragana-only is a
    # particle, ending, or fragment and is skipped.
    _HIRAGANA_MEANINGS = {
        "です": "is, am, are (polite)",
        "ます": "does, go (polite ending)",
        "ある": "to exist, to have",
        "いる": "to be, to exist (animate)",
        "する": "to do, to make",
        "なる": "to become",
        "ねこ": "cat",
        "いぬ": "dog",
        "とり": "bird",
        "さかな": "fish",
        "ごはん": "cooked rice, meal",
        "こと": "thing, matter",
        "もの": "thing",
        "ところ": "place",
        "これ": "this one",
        "それ": "that one",
        "あれ": "that one over there",
        "ここ": "here",
        "そこ": "there",
        "あそこ": "over there",
        "だれ": "who",
        "なに": "what",
        "いつ": "when",
        "どう": "how",
        "とても": "very",
        "たくさん": "a lot, many",
        "すこし": "a little",
        "もっと": "more",
        "もう": "already",
        "まだ": "yet, still",
        "ぜひ": "by all means",
        "きょう": "today",
        "あした": "tomorrow",
        "きのう": "yesterday",
        "すき": "liking, fondness",
        "きらい": "dislike",
        "おおきい": "big",
        "ちいさい": "small",
        "あたらしい": "new",
        "おいしい": "delicious",
        "たのしい": "fun, enjoyable",
        "こんにちは": "hello, good afternoon",
        "ありがとう": "thank you",
        "ありがとうございます": "thank you (polite)",
        "おはよう": "good morning",
        "おはようございます": "good morning (polite)",
        "すみません": "excuse me, sorry",
        "おやすみ": "good night",
        "さようなら": "goodbye",
        "はじめまして": "nice to meet you",
        "おねがいします": "please (request)",
    }

    def _scan_hiragana_run(self, run: str, context: str, seen: set, cards: list) -> None:
        """Greedy longest-match of curated words inside a hiragana run."""
        keys = sorted(self._HIRAGANA_MEANINGS, key=len, reverse=True)
        i = 0
        while i < len(run):
            match = next((k for k in keys if run.startswith(k, i)), None)
            if match is None:
                i += 1
                continue
            vocab = self._hiragana_card(match, context)
            key = (vocab.hiragana, vocab.meaning)
            if key not in seen:
                cards.append(vocab)
                seen.add(key)
            i += len(match)

    def _hiragana_card(self, token: str, context: str = "") -> LingoVocabCard:
        """Create card for hiragana token (verb/adjective)"""
        meanings = self._HIRAGANA_MEANINGS
        return LingoVocabCard(
            hiragana=token,
            romaji=self._hiragana_to_romaji(token),
            meaning=meanings[token],
            context=context,
            pos="verb" if token.endswith("う") or token in ["する", "ある", "いる"] else "auxiliary"
        )

    @staticmethod
    def _is_japanese(text: str) -> bool:
        """Check if text contains any Japanese characters"""
        return bool(re.search(r'[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]', text))

    @staticmethod
    def _is_hiragana(text: str) -> bool:
        """Check if text is all hiragana"""
        return bool(re.match(r'^[\u3040-\u309f]+$', text))

    @staticmethod
    def _has_kanji(text: str) -> bool:
        """Check if text contains kanji"""
        return bool(re.search(r'[\u4e00-\u9fff]', text))

    # Full hiragana romanization table + longest-match tokenizer.
    # Base single-kana mapping (used both directly and to build youon combos).
    _ROMAJI_BASE = {
        "あ": "a", "い": "i", "う": "u", "え": "e", "お": "o",
        "か": "ka", "き": "ki", "く": "ku", "け": "ke", "こ": "ko",
        "が": "ga", "ぎ": "gi", "ぐ": "gu", "げ": "ge", "ご": "go",
        "さ": "sa", "し": "shi", "す": "su", "せ": "se", "そ": "so",
        "ざ": "za", "じ": "ji", "ず": "zu", "ぜ": "ze", "ぞ": "zo",
        "た": "ta", "ち": "chi", "つ": "tsu", "て": "te", "と": "to",
        "だ": "da", "ぢ": "ji", "づ": "zu", "で": "de", "ど": "do",
        "な": "na", "に": "ni", "ぬ": "nu", "ね": "ne", "の": "no",
        "は": "ha", "ひ": "hi", "ふ": "fu", "へ": "he", "ほ": "ho",
        "ば": "ba", "び": "bi", "ぶ": "bu", "べ": "be", "ぼ": "bo",
        "ぱ": "pa", "ぴ": "pi", "ぷ": "pu", "ぺ": "pe", "ぽ": "po",
        "ま": "ma", "み": "mi", "む": "mu", "め": "me", "も": "mo",
        "や": "ya", "ゆ": "yu", "よ": "yo",
        "ら": "ra", "り": "ri", "る": "ru", "れ": "re", "ろ": "ro",
        "わ": "wa", "ゐ": "wi", "ゑ": "we", "を": "wo", "ん": "n",
        # Small vowels (used standalone in loanwords, e.g. ふぁ = "fa")
        "ぁ": "a", "ぃ": "i", "ぅ": "u", "ぇ": "e", "ぉ": "o",
        "ー": "",  # long vowel mark: extends the previous vowel, drop here
    }

    # Youon (combined sounds): consonant kana + small ya/yu/yo.
    _YOUON_SMALL = {"ゃ": "ya", "ゅ": "yu", "ょ": "yo"}
    _YOUON_STEMS = {
        "き": "ky", "ぎ": "gy", "し": "sh", "じ": "j", "ち": "ch", "ぢ": "j",
        "に": "ny", "ひ": "hy", "び": "by", "ぴ": "py", "み": "my", "り": "ry",
    }

    @classmethod
    def _build_romaji_table(cls) -> Dict[str, str]:
        table = dict(cls._ROMAJI_BASE)
        for stem_kana, stem_romaji in cls._YOUON_STEMS.items():
            for small_kana, small_romaji in cls._YOUON_SMALL.items():
                # e.g. き + ゃ -> "kya", し + ゅ -> "shu"
                vowel = small_romaji[1:]  # strip leading 'y' -> "a"/"u"/"o"
                table[stem_kana + small_kana] = stem_romaji + vowel
        return table

    @staticmethod
    def _hiragana_to_romaji(hiragana: str) -> str:
        """
        Convert hiragana to romaji using a longest-match tokenizer so that
        youon (きゃ, しゅ, ちょ, ...), small tsu (っ, gemination), and the
        long vowel mark (ー) are all handled correctly instead of being
        transliterated character-by-character.
        """
        table = VocabExtractor._build_romaji_table()
        result = []
        i = 0
        n = len(hiragana)
        while i < n:
            char = hiragana[i]

            # Small tsu (っ): doubles the consonant of the following mora.
            if char == "っ" and i + 1 < n:
                nxt = hiragana[i + 1]
                # Look ahead for a 2-char youon combo after the っ too.
                two_char = hiragana[i + 1:i + 3]
                following_romaji = table.get(two_char) or table.get(nxt)
                if following_romaji:
                    first_consonant = following_romaji[0]
                    if first_consonant not in "aeiou":
                        result.append(first_consonant)
                    i += 1
                    continue
                # Unknown following mora; just skip the sokuon marker.
                i += 1
                continue

            # Long vowel mark: repeat the last vowel of what we've built.
            if char == "ー":
                if result and result[-1]:
                    result.append(result[-1][-1])
                i += 1
                continue

            # Try a 2-character youon combo first (きゃ, しゅ, ちょ, ...).
            two_char = hiragana[i:i + 2]
            if two_char in table:
                result.append(table[two_char])
                i += 2
                continue

            # Fall back to single character.
            result.append(table.get(char, char))
            i += 1

        return "".join(result)


class MemoryBridge:
    """
    Integration layer: feed vocabulary reviews into Aiko's long-term memory system.

    Valence mapping:
    - Correct (grade >= GOOD): +1 valence (learning success)
    - Mistake (grade < GOOD): -1 valence (reinforces correction)
    """

    def __init__(self, aiko_instance=None):
        """aiko_instance: Aiko's main class (for access to _memory)"""
        self.aiko = aiko_instance

    def record_vocab_event(
        self,
        user_id: str,
        card: LingoVocabCard,
        grade: ReviewGrade,
        is_correct: bool
    ):
        """Log vocabulary review to Aiko's episodic memory."""
        if not self.aiko or not hasattr(self.aiko, '_memory'):
            log.warning("Aiko memory not available; skipping memory bridge")
            return

        from datetime import datetime

        valence = 1 if is_correct else -1
        event_text = (
            f"JJ reviewed '{card.hiragana}' ({card.meaning}) in Lingo. "
            f"Grade: {grade.name}. "
            f"{'Correct—building retention.' if is_correct else 'Corrected—reinforcing pattern.'}"
        )

        try:
            # Insert episodic memory (scene-based)
            scene_id = f"lingo_vocab_{card.id}_{datetime.utcnow().timestamp()}"
            self.aiko._memory.insert_episodic(
                user_id=user_id,
                scene_id=scene_id,
                content=event_text,
                valence=valence,
                tags=["lingo", "vocabulary", "srs", card.hiragana],
                source="lingo_srs"
            )
            log.info(f"Vocab event logged to LTM: {event_text[:60]}")
        except Exception as e:
            log.error(f"Failed to log vocab event to memory: {e}")


__all__ = ["VocabExtractor", "MemoryBridge"]
