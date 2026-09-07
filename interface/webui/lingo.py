import os
import json
import uuid
import logging
import re
import time
import html
from pathlib import Path
from typing import Optional, List, Dict
from functools import lru_cache
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from system.userspace import set_current_user_id, reset_current_user_id

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/english", tags=["lingo"])

# Audio storage setup
STATIC_DIR = Path(__file__).parent / "static"
AUDIO_DIR = STATIC_DIR / "lingo_audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# We'll resolve this dynamically from auth module
AUDIO_BASE_URL = "https://aiko.ide-chroma.ts.net/lingo_audio"

# ============================================================================
# Protocol Definition: Tag vocabulary and constants for Lingo response format
# ============================================================================
class LingoTags:
    """
    Engram vocabulary: standardized tag names for Lingo conversation protocol.
    This centralizes the protocol definition and reduces string literal duplication.
    """
    MISTAKE = "MISTAKE"
    FEEDBACK = "FEEDBACK"
    SUGGESTION = "SUGGESTION"
    REPLY_JP = "REPLY_JP"
    REPLY_EN = "REPLY_EN"
    FINISHED = "FINISHED"
    
    ALL = [MISTAKE, FEEDBACK, SUGGESTION, REPLY_JP, REPLY_EN, FINISHED]
    START_TAGS = [REPLY_JP, REPLY_EN, FINISHED]


# ============================================================================
# Validation and Sanitization Utilities
# ============================================================================

def is_valid_text(text: Optional[str], min_length: int = 1) -> bool:
    """
    Engram validation: check if text is present, non-empty after stripping,
    and not a placeholder. Centralizes empty-check logic to avoid inconsistencies.
    """
    if not text:
        return False
    stripped = text.strip()
    return len(stripped) >= min_length and stripped not in ["...", "*", ""]


def is_japanese_text(text: str, min_ja_chars: int = 2) -> bool:
    """
    Cognition check: validate that text contains meaningful Japanese content.
    Detects hiragana, katakana, and kanji—avoids spurious single-character matches.
    """
    if not text:
        return False
    
    ja_count = 0
    for c in text:
        code = ord(c)
        # Hiragana range: 0x3040-0x309F
        # Katakana range: 0x30A0-0x30FF
        # Kanji (CJK Unified): 0x4E00-0x9FFF
        if (0x3040 <= code <= 0x309F or
            0x30A0 <= code <= 0x30FF or
            0x4E00 <= code <= 0x9FFF):
            ja_count += 1
    
    return ja_count >= min_ja_chars


def sanitize_text(text: Optional[str]) -> str:
    """
    Active cognition: remove roleplay actions (*smiles*) and emoji from text.
    Must happen BEFORE audio generation to prevent TTS from synthesizing meta-text.
    """
    if not text:
        return ""
    
    # Remove actions: *anything*
    t = re.sub(r'\*[^*]*\*', '', text)
    
    # Remove emoji ranges (Unicode blocks for miscellaneous symbols, emoticons, etc)
    # Emoticons: U+1F600-U+1F64F
    # Miscellaneous Symbols and Pictographs: U+1F300-U+1F5FF
    # Transport and Map Symbols: U+1F680-U+1F6FF
    # Miscellaneous Symbols: U+2600-U+27BF
    t = re.sub(r'[\u2600-\u27bf\U0001f300-\U0001f5ff\U0001f680-\U0001f6ff\U0001f600-\U0001f64f]', '', t)
    
    return t.strip()


def unwrap_nested_dict(data: Dict, expected_keys: set) -> Dict:
    """
    Flatten neuronal pathway: unwrap one level of dict nesting if top-level values
    are dicts but their keys match expected linguistic fields (japanese, english, etc).
    Logs discarded branches for debugging.
    """
    for key in list(data.keys()):
        val = data[key]
        if not isinstance(val, dict):
            continue
        
        # Try to find expected linguistic fields
        for field in expected_keys:
            if field in val and isinstance(val[field], str):
                data[key] = val[field]
                log.debug(f"Unwrapped {key}.{field}")
                break
        else:
            # No expected field found; try first string value
            for sub_val in val.values():
                if isinstance(sub_val, str) and sub_val.strip():
                    data[key] = sub_val
                    log.debug(f"Unwrapped {key} (first string value)")
                    break
    
    return data


# ============================================================================
# Request/Response Models
# ============================================================================

class TranslateRequest(BaseModel):
    text: str = Field(..., max_length=500, description="English text to translate")


class StartRequest(BaseModel):
    level: str = Field(..., regex="^(beginner|intermediate|advanced)$")


class DialogueHistoryEntry(BaseModel):
    speaker: str  # 'user' or 'aiko'
    text: str


class RespondRequest(BaseModel):
    text: str = Field(..., max_length=500)
    history: Optional[List[DialogueHistoryEntry]] = []


class TranslationResult(BaseModel):
    register: str
    text: str
    audioUrl: Optional[str] = None


class TranslateResponse(BaseModel):
    translations: List[TranslationResult]


class ConversationResponse(BaseModel):
    japaneseText: str
    englishTranslation: str
    audioUrl: Optional[str] = None
    isFinished: bool = False
    isCorrect: bool = True
    feedback: Optional[str] = None
    suggestion: Optional[str] = None


# ============================================================================
# Audio Generation with Atomic Writes and Uniqueness
# ============================================================================

def generate_lingo_audio(text: str) -> Optional[str]:
    """
    Synthesize Japanese text using AikoSpeak and return a public URL.
    Uses atomic writes (temp → rename) to prevent corruption on concurrent access.
    """
    from interface.webui import auth

    if not is_valid_text(text):
        return None

    # Filter out Latin/English characters and Romaji completely.
    clean_text = re.sub(r'[a-zA-Z]', '', text)
    # Remove common Romaji leftovers like empty parentheses: "こんにちは ( )"
    clean_text = re.sub(r'\(\s*\)', '', clean_text)
    # Remove any extra spaces left behind
    clean_text = re.sub(r'\s{2,}', ' ', clean_text).strip()

    if not is_valid_text(clean_text):
        return None

    if not auth.aiko_web_instance or not auth.aiko_web_instance._speak:
        return None

    try:
        wav_bytes = auth.aiko_web_instance._speak._synthesize(clean_text)
        if not wav_bytes:
            return None

        # Generate unique filename with timestamp for collision avoidance
        # and safe atomic writes
        audio_id = f"{uuid.uuid4()}_{int(time.time() * 1000)}"
        filename = f"{audio_id}.wav"
        filepath = AUDIO_DIR / filename
        
        # Write atomically: temp file → rename (POSIX-atomic operation)
        temp_path = filepath.with_suffix('.tmp')
        temp_path.write_bytes(wav_bytes)
        temp_path.replace(filepath)

        base_url = getattr(auth, "REDIRECT_BASE", "https://aiko.ide-chroma.ts.net").rstrip("/")
        return f"{base_url}/lingo_audio/{filename}"
    except Exception as e:
        log.error(f"Failed to generate audio for Lingo: {e}")
        return None


# ============================================================================
# JSON Parsing with Robust Tag Extraction
# ============================================================================

def parse_lingo_json(content: str) -> dict:
    """
    Robustly parse JSON from LLM response and handle common nesting issues.
    Extracts nested dicts and validates structure before returning.
    """
    if not content:
        raise ValueError("Empty response from LLM")

    content = content.strip()

    def _extract(text: str) -> dict:
        """Inner cascade: try direct JSON, then markdown fence, then raw dict."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try markdown code fence
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Try raw dict extraction
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                # Fix common formatting issues: missing commas between quoted strings
                fixed = re.sub(r'("\s*\n\s*")', '",\n"', match.group(1))
                try:
                    return json.loads(fixed)
                except json.JSONDecodeError:
                    pass

        raise ValueError(f"Could not parse JSON from: {text[:200]}")

    data = _extract(content)

    if isinstance(data, dict):
        # Unwrap nested structures
        expected_keys = {
            'text', 'message', 'japanese', 'english', 'feedback', 'suggestion',
            'register', 'japaneseText', 'englishTranslation'
        }
        data = unwrap_nested_dict(data, expected_keys)

    return data


# ============================================================================
# Conversation History Building
# ============================================================================

def build_conversation_history(entries: List[DialogueHistoryEntry]) -> List[dict]:
    """
    Engram reconstruction: rebuild dialogue chain respecting strict role alternation.
    Merges consecutive same-role messages and seeds with user if first is assistant.
    Critical for llama-server, which requires alternating user/assistant turns.
    """
    if not entries:
        return []
    
    pending = []
    for entry in entries:
        text = entry.text.strip()
        if not is_valid_text(text):
            continue
        
        role = "assistant" if entry.speaker == "aiko" else "user"
        
        # Ensure first message is always user
        if not pending:
            if role == "assistant":
                pending.append({"role": "user", "content": "Let's practice Japanese."})
        
        # Merge consecutive same-role messages (chat API requirement)
        if pending and pending[-1]["role"] == role:
            pending[-1]["content"] += "\n" + text
        else:
            pending.append({"role": role, "content": text})
    
    return pending


# ============================================================================
# Session & Auth
# ============================================================================

async def get_lingo_session(request: Request) -> dict:
    """Engram retrieval: session context for user identity and auth gate."""
    from interface.webui import auth
    try:
        session = await auth.require_session(request)
        return await auth.require_accepted_session(session)
    except Exception as e:
        log.warning(f"Lingo session auth failed: {e} — falling back to app owner")
        owner = os.getenv("AIKO_USER_ID", "OppaAI")
        if not owner or owner == "":
            log.error("AIKO_USER_ID not set and auth failed; aborting request")
            raise HTTPException(status_code=500, detail="Session unavailable")
        return {"user_id": owner, "username": owner}


# ============================================================================
# API Endpoints
# ============================================================================

@router.post("/translate", response_model=TranslateResponse)
async def translate(request: TranslateRequest, session: dict = Depends(get_lingo_session)):
    """Translate English to Japanese in multiple registers."""
    from interface.webui import auth
    if not auth.aiko_web_instance or not auth.aiko_web_instance._think:
        raise HTTPException(status_code=503, detail="Aiko's brain is not ready")

    think = auth.aiko_web_instance._think
    uid = session.get("user_id")
    token = set_current_user_id(uid)
    try:
        system_prompt = (
            "You are a Japanese translation expert. "
            "Output ONLY valid JSON. Your response must be a JSON object with a single 'translations' key. "
            "That key must contain an array of objects, each with 'register' and 'text' keys."
        )
        # Escape user text to prevent prompt injection
        escaped_text = html.escape(request.text)
        user_prompt = (
            f"Translate the following English text to Japanese in 3-5 different registers: {escaped_text}"
        )

        response = think._client.chat.completions.create(
            model=think._llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            timeout=120.0
        )

        content = response.choices[0].message.content
        data = parse_lingo_json(content)
        translations_raw = data.get("translations", [])

        results = []
        for item in translations_raw:
            text_jp = item.get("text", "")
            results.append(TranslationResult(
                register=item.get("register", "Normal"),
                text=text_jp,
                audioUrl=None
            ))
        return TranslateResponse(translations=results)
    except Exception as e:
        log.exception("Lingo translation failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        reset_current_user_id(token)


@router.post("/conversation/start")
async def conversation_start(request: StartRequest, session: dict = Depends(get_lingo_session)):
    """Start a Japanese conversation at the specified level."""
    from interface.webui import auth
    if not auth.aiko_web_instance or not auth.aiko_web_instance._think:
        raise HTTPException(status_code=503, detail="Aiko's brain is not ready")

    think = auth.aiko_web_instance._think
    uid = session.get("user_id")

    async def event_generator():
        token = set_current_user_id(uid)
        try:
            base_prompt = think._current_system_prompt("Start Japanese conversation")
            system_prompt = (
                f"{base_prompt}\n\n"
                "ACTIVATE SKILL: JAPANESE_TUTOR\n"
                "You are in 'Lingo App Mode'. Start a Japanese conversation at {request.level} level. "
                "Output your response exactly like this:\n"
                f"{LingoTags.REPLY_JP}: <Japanese sentences>\n"
                f"{LingoTags.REPLY_EN}: <English translation>\n"
                f"{LingoTags.FINISHED}: False"
            )
            user_prompt = f"Start a Japanese conversation at {request.level} level. Speak 1-3 sentences in Japanese."

            response = think._client.chat.completions.create(
                model=think._llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                stream=True,
                timeout=120.0
            )

            full_content = ""
            is_streaming_jp = False
            try:
                for chunk in response:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        full_content += delta

                        # More robust parsing for streaming
                        clean_buf = full_content.replace("**", "")
                        if not is_streaming_jp and f"{LingoTags.REPLY_JP}:" in clean_buf:
                            is_streaming_jp = True
                            parts = clean_buf.split(f"{LingoTags.REPLY_JP}:", 1)
                            if len(parts) > 1:
                                text = parts[1]
                                # Check if next tag is already here
                                stop_tags = [f"{LingoTags.REPLY_EN}:", f"{LingoTags.FINISHED}:"]
                                for tag in stop_tags:
                                    if tag in text:
                                        text = text.split(tag, 1)[0]
                                        is_streaming_jp = False
                                if text:
                                    yield json.dumps({"type": "delta", "text": text}) + "\n"
                            continue

                        if is_streaming_jp:
                            clean_delta = delta.replace("**", "")
                            stop_tags = [f"{LingoTags.REPLY_EN}:", f"{LingoTags.FINISHED}:"]
                            found_stop = False
                            for tag in stop_tags:
                                if tag in clean_delta:
                                    text = clean_delta.split(tag, 1)[0]
                                    if text:
                                        yield json.dumps({"type": "delta", "text": text}) + "\n"
                                    is_streaming_jp = False
                                    found_stop = True
                                    break
                            if not found_stop:
                                yield json.dumps({"type": "delta", "text": clean_delta}) + "\n"

            except Exception as e:
                log.exception("Chunk iteration failed during start stream")
                yield json.dumps({"type": "error", "message": str(e)}) + "\n"
                return

            # Final structured data extraction using robust regex
            data = {"isCorrect": True, "isFinished": False}

            flat_content = full_content.replace("**", "").replace("`", "")

            # Extract tags with flexible regex (handles indentation)
            tag_regex = r"([A-Z_]+)\s*:\s*(.*?)(?=[A-Z_]+\s*:|$)"
            found_pairs = re.findall(tag_regex, flat_content, re.DOTALL | re.IGNORECASE)

            for tag, val in found_pairs:
                tag = tag.strip().upper()
                val = val.strip()  # CRITICAL: strip all extracted values
                if tag == LingoTags.FINISHED.upper():
                    data["isFinished"] = val.lower() in ["true", "yes", "1"]
                elif tag == LingoTags.REPLY_JP.upper():
                    data["japanese"] = val
                elif tag == LingoTags.REPLY_EN.upper():
                    data["english"] = val

            # Fallback: if no tags found and content is substantial, treat as Japanese
            if not data.get("japanese") and not any(f"{t}:" in flat_content.upper() for t in LingoTags.START_TAGS):
                data["japanese"] = full_content.strip()

            # Smart Fallback for Japanese
            if not data.get("japanese"):
                jp_match = re.search(
                    r"([\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff].*?)(?:\n|$|REPLY_EN:|English:)",
                    flat_content,
                    re.DOTALL
                )
                if jp_match:
                    candidate = jp_match.group(1).strip()
                    if is_japanese_text(candidate):
                        data["japanese"] = candidate

            # Smart Fallback for English: look for sentence-like patterns
            if not data.get("english"):
                en_match = re.search(
                    r"([A-Z][a-zA-Z\s,.'\"?!-]{10,}[.!?])",
                    flat_content
                )
                if en_match:
                    candidate = en_match.group(1).strip()
                    if len(candidate.split()) >= 2:
                        data["english"] = candidate

            # Generate audio only if we have valid Japanese
            audio_url = None
            if is_valid_text(data.get("japanese")):
                audio_url = generate_lingo_audio(data["japanese"])

            yield json.dumps({
                "type": "final",
                "isCorrect": True,
                "japanese": data.get("japanese") or "...",
                "english": data.get("english") or "Translation unavailable",
                "isFinished": data.get("isFinished", False),
                "audioUrl": audio_url
            }) + "\n"
        except Exception as e:
            log.exception("Lingo start stream failed")
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"
        finally:
            reset_current_user_id(token)

    response = StreamingResponse(event_generator(), media_type="text/event-stream")
    response.headers["X-Accel-Buffering"] = "no"
    response.headers["Cache-Control"] = "no-cache"
    return response


@router.post("/conversation/respond_stream")
async def conversation_respond_stream(request: RespondRequest, session: dict = Depends(get_lingo_session)):
    """Stream a Japanese tutor response to user input with corrections and feedback."""
    from interface.webui import auth
    if not auth.aiko_web_instance or not auth.aiko_web_instance._think:
        raise HTTPException(status_code=503, detail="Aiko's brain is not ready")

    think = auth.aiko_web_instance._think
    uid = session.get("user_id")

    async def event_generator():
        token = set_current_user_id(uid)
        try:
            # Combine Aiko's real persona with the Japanese Tutor skill
            base_prompt = think._current_system_prompt(request.text)
            # Force the protocol instructions to be the highest priority
            system_prompt = (
                "You are Aiko, a helpful Japanese tutor. Maintain your unique personality but "
                "you MUST suppress all emojis and roleplay actions (like *smiles*). "
                "You are in 'Lingo App Mode' and must output ONLY these tags in order:\n\n"
                f"{LingoTags.MISTAKE}: <True/False>\n"
                f"{LingoTags.FEEDBACK}: <English explanation or empty>\n"
                f"{LingoTags.SUGGESTION}: <Corrected Japanese or empty>\n"
                f"{LingoTags.REPLY_JP}: <Next Japanese conversation turn. NO ROMAJI. NO ACTIONS.>\n"
                f"{LingoTags.REPLY_EN}: <Complete literal English translation of {LingoTags.REPLY_JP} and {LingoTags.SUGGESTION}>\n"
                f"{LingoTags.FINISHED}: <False>\n\n"
                "Do not include any other text."
            )

            # Build conversation history respecting strict role alternation
            messages = [{"role": "system", "content": system_prompt}]
            history_messages = build_conversation_history(request.history)
            messages.extend(history_messages)

            # Append current user input
            user_text = request.text.strip()
            if messages[-1]["role"] == "user":
                messages[-1]["content"] += "\n" + user_text
            else:
                messages.append({"role": "user", "content": user_text})

            response = think._client.chat.completions.create(
                model=think._llm_model,
                messages=messages,
                stream=True,
                timeout=120.0,
                temperature=0.3  # Lower temperature for strict protocol compliance
            )

            full_content = ""
            is_streaming_jp = False
            try:
                for chunk in response:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        full_content += delta

                        # Determine target streaming tag based on whether there's a mistake
                        clean_buf = full_content.replace("**", "")
                        target_tag = (
                            f"{LingoTags.SUGGESTION}:"
                            if f"{LingoTags.MISTAKE}: TRUE" in clean_buf.upper()
                            else f"{LingoTags.REPLY_JP}:"
                        )

                        if not is_streaming_jp:
                            if target_tag in clean_buf:
                                is_streaming_jp = True
                                parts = clean_buf.split(target_tag, 1)
                                if len(parts) > 1:
                                    text = parts[1]
                                    # Check for stop tags
                                    stop_tags = [
                                        f"{LingoTags.REPLY_EN}:",
                                        f"{LingoTags.FINISHED}:",
                                        f"{LingoTags.REPLY_JP}:"
                                    ]
                                    if target_tag in stop_tags:
                                        stop_tags.remove(target_tag)
                                    for tag in stop_tags:
                                        if tag in text:
                                            text = text.split(tag, 1)[0]
                                            is_streaming_jp = False
                                    if text:
                                        yield json.dumps({"type": "delta", "text": text}) + "\n"
                            elif (not any(t in clean_buf.upper() for t in [LingoTags.MISTAKE, LingoTags.REPLY_JP, LingoTags.REPLY_EN])
                                  and len(clean_buf) > 10):
                                # Fallback: if no tags yet, stream raw text
                                is_streaming_jp = True
                                yield json.dumps({"type": "delta", "text": delta.replace("**", "")}) + "\n"
                            continue

                        if is_streaming_jp:
                            clean_delta = delta.replace("**", "")
                            stop_tags = [
                                f"{LingoTags.REPLY_EN}:",
                                f"{LingoTags.FINISHED}:",
                                f"{LingoTags.REPLY_JP}:",
                                f"{LingoTags.SUGGESTION}:"
                            ]
                            if target_tag in stop_tags:
                                stop_tags.remove(target_tag)
                            found_stop = False
                            for tag in stop_tags:
                                if tag in clean_delta:
                                    text = clean_delta.split(tag, 1)[0]
                                    if text:
                                        yield json.dumps({"type": "delta", "text": text}) + "\n"
                                    is_streaming_jp = False
                                    found_stop = True
                                    break
                            if not found_stop:
                                yield json.dumps({"type": "delta", "text": clean_delta}) + "\n"

            except Exception as e:
                log.exception("Chunk iteration failed during respond stream")
                yield json.dumps({"type": "error", "message": str(e)}) + "\n"
                return

            # Final structured data extraction
            data = {"isCorrect": True, "isFinished": False}

            flat_content = full_content.replace("**", "").replace("`", "")

            # Extract tags with flexible regex
            tag_regex = r"([A-Z_]+)\s*:\s*(.*?)(?=[A-Z_]+\s*:|$)"
            found_pairs = re.findall(tag_regex, flat_content, re.DOTALL | re.IGNORECASE)

            for tag, val in found_pairs:
                tag = tag.strip().upper()
                val = val.strip()  # CRITICAL: strip all extracted values
                if tag == LingoTags.MISTAKE.upper():
                    data["isCorrect"] = val.lower() not in ["true", "yes", "y", "1"]
                elif tag == LingoTags.FINISHED.upper():
                    data["isFinished"] = val.lower() in ["true", "yes", "y", "1"]
                elif tag == LingoTags.REPLY_JP.upper():
                    data["japanese"] = val
                elif tag == LingoTags.REPLY_EN.upper():
                    data["english"] = val
                elif tag == LingoTags.FEEDBACK.upper():
                    data["feedback"] = val
                elif tag == LingoTags.SUGGESTION.upper():
                    data["suggestion"] = val

            # Fallback: if no tags found, treat whole thing as Japanese
            if not data.get("japanese") and not any(f"{t}:" in flat_content.upper() for t in LingoTags.ALL):
                data["japanese"] = full_content.strip()

            # Smart Fallback for Japanese
            if not data.get("japanese"):
                jp_match = re.search(
                    r"([\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff].*?)(?:\n|REPLY_EN:|English:|$)",
                    flat_content,
                    re.DOTALL
                )
                if jp_match:
                    candidate = jp_match.group(1).strip()
                    if is_japanese_text(candidate):
                        data["japanese"] = candidate

            # Smart Fallback for English
            if not data.get("english"):
                en_match = re.search(r"([A-Z][a-zA-Z\s,.'\"?!-]{10,}[.!?])", flat_content)
                if en_match:
                    candidate = en_match.group(1).strip()
                    if len(candidate.split()) >= 2:
                        data["english"] = candidate

            # Determine final display text and audio generation
            is_mistake = not data.get("isCorrect", True)

            if is_mistake:
                feedback = data.get("feedback") or "Grammar/usage error."
                suggestion_clean = sanitize_text(data.get("suggestion") or "")
                final_jp = f"Mistake: {feedback}"
                if suggestion_clean:
                    final_jp += f"\nSuggestion: {suggestion_clean}"
                audio_text = suggestion_clean or feedback
                english_text = data.get("english") or "Please correct your Japanese ♡"
            else:
                final_jp = sanitize_text(data.get("japanese") or "...")
                audio_text = final_jp
                english_text = data.get("english") or "Perfect! Let's keep talking."

            # Generate audio only if valid text
            audio_url = None
            if is_valid_text(audio_text):
                audio_url = generate_lingo_audio(audio_text)

            yield json.dumps({
                "type": "final",
                "isCorrect": data.get("isCorrect", True),
                "feedback": data.get("feedback"),
                "suggestion": data.get("suggestion"),
                "japanese": final_jp,
                "english": english_text,
                "isFinished": data.get("isFinished", False),
                "audioUrl": audio_url
            }) + "\n"
        except Exception as e:
            log.exception("Lingo respond stream failed")
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"
        finally:
            reset_current_user_id(token)

    response = StreamingResponse(event_generator(), media_type="text/event-stream")
    response.headers["X-Accel-Buffering"] = "no"
    response.headers["Cache-Control"] = "no-cache"
    return response


@router.post("/conversation/hint", response_model=ConversationResponse)
async def conversation_hint(session: dict = Depends(get_lingo_session)):
    """Provide a hint for what to say next in the conversation."""
    from interface.webui import auth
    if not auth.aiko_web_instance or not auth.aiko_web_instance._think:
        raise HTTPException(status_code=503, detail="Aiko's brain is not ready")

    think = auth.aiko_web_instance._think
    uid = session.get("user_id")
    token = set_current_user_id(uid)
    try:
        system_prompt = (
            "You are Aiko, teaching Japanese through conversation. "
            "Suggest a concise response for the student to say in Japanese and provide an English translation. "
            "Keep the Japanese response under 30 words. Japanese ONLY in 'japanese' field. "
            "Output ONLY valid JSON with 'japanese' and 'english' keys."
        )
        response = think._client.chat.completions.create(
            model=think._llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Hint please"}
            ],
            response_format={"type": "json_object"},
            timeout=120.0
        )
        data = parse_lingo_json(response.choices[0].message.content)
        jp = data.get("japanese") or data.get("japaneseText") or ""
        en = data.get("english") or data.get("englishTranslation") or "Translation unavailable"
        
        audio_url = None
        if is_valid_text(str(jp)):
            audio_url = generate_lingo_audio(str(jp))
        
        return ConversationResponse(
            japaneseText=str(jp),
            englishTranslation=str(en),
            audioUrl=audio_url,
            isCorrect=True
        )
    except Exception as e:
        log.exception("Lingo conversation hint failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        reset_current_user_id(token)


@router.get("/tts")
async def get_tts(text: str, session: dict = Depends(get_lingo_session)):
    """
    Text-to-speech endpoint with input validation and DOS protection.
    Max 200 chars to prevent abuse.
    """
    # Input validation: max 200 characters (roughly 30 seconds of audio)
    if not text or len(text) > 200:
        raise HTTPException(
            status_code=400,
            detail="Text length must be 1–200 characters"
        )
    
    url = generate_lingo_audio(text)
    if not url:
        raise HTTPException(status_code=503, detail="TTS generation failed")
    return {"audioUrl": url}


@router.post("/conversation/stop")
async def conversation_stop(session: dict = Depends(get_lingo_session)):
    """Stop the current conversation session."""
    return {"success": True}
