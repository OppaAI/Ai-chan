import os
import json
import uuid
import logging
from pathlib import Path
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from system.userspace import set_current_user_id, reset_current_user_id

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/english", tags=["lingo"])

# Audio storage setup
STATIC_DIR = Path(__file__).parent / "static"
AUDIO_DIR = STATIC_DIR / "lingo_audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# We'll resolve this dynamically from auth module
AUDIO_BASE_URL = "https://aiko.ide-chroma.ts.net/lingo_audio"

class TranslateRequest(BaseModel):
    text: str

class StartRequest(BaseModel):
    level: str

class DialogueHistoryEntry(BaseModel):
    speaker: str
    text: str

class RespondRequest(BaseModel):
    text: str
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

def generate_lingo_audio(text: str) -> Optional[str]:
    """Synthesize Japanese text using AikoSpeak and return a public URL."""
    from interface.webui import auth

    if not text or not text.strip():
        log.warning("Audio generation skipped: empty text")
        return None

    # Filter out English characters to ensure she only speaks Japanese
    import re
    # Keep Japanese characters, punctuation, and digits. Remove latin letters.
    # [a-zA-Z] matches basic English letters.
    clean_text = re.sub(r'[a-zA-Z]', '', text).strip()

    if not clean_text:
        log.warning(f"Audio generation skipped: no Japanese text found in '{text}'")
        return None

    if not auth.aiko_web_instance or not auth.aiko_web_instance._speak:
        log.warning("Audio generation failed: AikoSpeak instance not available")
        return None

    try:
        # AikoSpeak._synthesize returns WAV bytes
        wav_bytes = auth.aiko_web_instance._speak._synthesize(text)
        if not wav_bytes:
            return None

        audio_id = str(uuid.uuid4())
        filename = f"{audio_id}.wav"
        filepath = AUDIO_DIR / filename
        filepath.write_bytes(wav_bytes)

        # Try to use REDIRECT_BASE from auth if available
        base_url = getattr(auth, "REDIRECT_BASE", "https://aiko.ide-chroma.ts.net").rstrip("/")
        return f"{base_url}/lingo_audio/{filename}"
    except Exception as e:
        log.error(f"Failed to generate audio for Lingo: {e}")
        return None

async def get_lingo_session(request: Request):
    """Helper to get session from cookie or fallback to owner for the app."""
    from interface.webui import auth
    try:
        # require_session and require_accepted_session are dependencies
        session = await auth.require_session(request)
        return await auth.require_accepted_session(session)
    except Exception:
        # Fallback for the Android app which doesn't have OAuth cookies yet
        # Default to OppaAI or the configured owner
        owner = os.getenv("AIKO_USER_ID", "OppaAI")
        return {"user_id": owner, "username": owner}

def parse_lingo_json(content: str):
    """Robustly parse JSON from LLM response and handle common nesting issues."""
    if not content:
        raise ValueError("Empty response from LLM")

    content = content.strip()

    def _extract(text):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            import re
            # Try to find JSON block in markdown
            match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass

            # Last ditch: try to find anything between { and }
            match = re.search(r"(\{.*\})", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    # If still failing, try to fix common errors
                    fixed = re.sub(r'("\s*\n\s*")', '",\n"', match.group(1))
                    try:
                        return json.loads(fixed)
                    except Exception:
                        pass
            raise

    data = _extract(content)

    # Post-process to ensure we don't have dicts where strings should be
    # (handles cases like {"japanese": {"message": "..."}} or {"japanese": {"greeting": "..."}})
    if isinstance(data, dict):
        # Specific fix for 'translations' key in translate endpoint
        if 'translations' in data and isinstance(data['translations'], list):
            for i, item in enumerate(data['translations']):
                if isinstance(item, dict):
                    for k, v in item.items():
                        if isinstance(v, dict):
                            data['translations'][i][k] = str(list(v.values())[0]) if v else ""

        for key in list(data.keys()):
            val = data[key]
            if isinstance(val, dict):
                # If it's a dict, try to find the first string value inside it
                # This covers 'message', 'text', 'greeting', 'content', etc.
                found_string = None
                # Priority keys
                for p_key in ['text', 'message', 'japanese', 'english', 'feedback', 'suggestion']:
                    if p_key in val and isinstance(val[p_key], str):
                        found_string = val[p_key]
                        break

                if not found_string:
                    for sub_val in val.values():
                        if isinstance(sub_val, str) and sub_val.strip():
                            found_string = sub_val
                            break

                if found_string:
                    data[key] = found_string
                elif len(val) == 1:
                    data[key] = str(list(val.values())[0])

    return data

@router.post("/translate", response_model=TranslateResponse)
async def translate(request: TranslateRequest, session: dict = Depends(get_lingo_session)):
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
        user_prompt = f"Translate the following English text to Japanese in 3-5 different registers (e.g., Casual, Polite, Respective, etc.): {request.text}"

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
        log.debug(f"Lingo translate response: {content}")

        data = parse_lingo_json(content)
        translations_raw = data.get("translations", [])

        results = []
        for item in translations_raw:
            text_jp = item.get("text", "")
            results.append(TranslationResult(
                register=item.get("register", "Normal"),
                text=text_jp,
                audioUrl=generate_lingo_audio(text_jp)
            ))
        return TranslateResponse(translations=results)
    except Exception as e:
        log.exception("Lingo translation failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        reset_current_user_id(token)

@router.post("/conversation/start", response_model=ConversationResponse)
async def conversation_start(request: StartRequest, session: dict = Depends(get_lingo_session)):
    from interface.webui import auth
    think = auth.aiko_web_instance._think
    uid = session.get("user_id")
    token = set_current_user_id(uid)
    try:
        system_prompt = (
            "You are Aiko, teaching Japanese through conversation. "
            "Output ONLY valid JSON. Your response must be a JSON object with 'japanese' and 'english' keys. "
            "The 'japanese' value must be in Japanese ONLY (no English). "
            "DO NOT include any text before or after the JSON block."
        )
        user_prompt = (
            f"Start a Japanese conversation at {request.level} level. Speak 1-3 sentences in Japanese. "
            "Example response format: {\"japanese\": \"こんにちは、お元気ですか？\", \"english\": \"Hello, how are you?\"}"
        )

        # Retry logic for conversation start
        max_retries = 2
        content = ""
        for attempt in range(max_retries + 1):
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
            log.info(f"Lingo conversation start response (attempt {attempt}): {content}")

            try:
                data = parse_lingo_json(content)
                if data.get("japanese") or data.get("japaneseText"):
                    break
            except Exception as e:
                if attempt == max_retries:
                    raise
                log.warning(f"Lingo JSON parse failed on attempt {attempt}: {e}")

        data = parse_lingo_json(content)
        japanese_text = data.get("japanese") or data.get("japaneseText") or ""
        english_text = data.get("english") or data.get("englishTranslation") or ""
        return ConversationResponse(
            japaneseText=str(japanese_text),
            englishTranslation=str(english_text) or "Translation unavailable",
            audioUrl=generate_lingo_audio(str(japanese_text)),
            isFinished=data.get("finished", data.get("isFinished", False))
        )
    except Exception as e:
        log.exception("Lingo conversation start failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        reset_current_user_id(token)

@router.post("/conversation/respond", response_model=ConversationResponse)
async def conversation_respond(request: RespondRequest, session: dict = Depends(get_lingo_session)):
    from interface.webui import auth
    think = auth.aiko_web_instance._think
    uid = session.get("user_id")
    token = set_current_user_id(uid)
    try:
        system_prompt = (
            "You are Aiko, a Japanese teacher. Maintain a natural roleplay conversation with the student. "
            "When the student responds, check their Japanese for grammar, spelling, or unnatural usage. "
            "1. If there is a mistake: set 'isCorrect' to false. Provide feedback in 'feedback' (explaining the error in English) and the corrected version in 'suggestion' (Japanese ONLY). Set 'japanese' to the feedback text. "
            "2. If it is correct and natural: set 'isCorrect' to true. Provide the NEXT conversation turn in 'japanese' (Japanese ONLY) and 'english' (English translation) that naturally continues the roleplay. "
            "Output ONLY valid JSON with keys: 'isCorrect', 'feedback', 'suggestion', 'japanese', 'english', 'finished'."
        )

        # Build message list from history, ensuring roles alternate strictly (User -> Assistant -> User)
        # and merging consecutive same-role messages to avoid LLM backend errors.
        messages = [{"role": "system", "content": system_prompt}]

        pending_history = []
        if request.history:
            for entry in request.history:
                role = "assistant" if entry.speaker == "aiko" else "user"
                if not pending_history:
                    # First message after system SHOULD be User
                    if role == "user":
                        pending_history.append({"role": role, "content": entry.text})
                    else:
                        # If Aiko started, skip her first message or it would follow system directly
                        continue
                else:
                    last = pending_history[-1]
                    if last["role"] == role:
                        # Merge consecutive same-role messages
                        last["content"] += "\n" + entry.text
                    else:
                        pending_history.append({"role": role, "content": entry.text})

        # Add current user input
        if pending_history and pending_history[-1]["role"] == "user":
            pending_history[-1]["content"] += "\n" + request.text
        else:
            pending_history.append({"role": "user", "content": request.text})

        messages.extend(pending_history)

        # Retry logic for conversation respond
        max_retries = 2
        content = ""
        for attempt in range(max_retries + 1):
            response = think._client.chat.completions.create(
                model=think._llm_model,
                messages=messages,
                response_format={"type": "json_object"},
                timeout=120.0
            )

            content = response.choices[0].message.content
            log.info(f"Lingo conversation respond response (attempt {attempt}): {content}")

            try:
                data = parse_lingo_json(content)
                # Success if we have either a correction flow or a continuation flow
                if data.get("isCorrect") is not None or data.get("japanese") or data.get("japaneseText"):
                    break
            except Exception as e:
                if attempt == max_retries:
                    raise
                log.warning(f"Lingo JSON parse failed on attempt {attempt}: {e}")

        data = parse_lingo_json(content)
        is_correct = data.get("isCorrect", True)

        if not is_correct:
            # Mistake mode: Speak the feedback or suggestion
            feedback = data.get("feedback", "")
            suggestion = data.get("suggestion", "")
            japanese_text = f"Mistake: {feedback}\nSuggestion: {suggestion}"
            return ConversationResponse(
                japaneseText=japanese_text,
                englishTranslation="Please correct your response ♡",
                audioUrl=generate_lingo_audio(suggestion), # Speak the correct version
                isFinished=False,
                isCorrect=False,
                feedback=feedback,
                suggestion=suggestion
            )

        # Success mode: Continue conversation
        japanese_text = data.get("japanese") or data.get("japaneseText") or ""
        english_text = data.get("english") or data.get("englishTranslation") or ""
        return ConversationResponse(
            japaneseText=str(japanese_text),
            englishTranslation=str(english_text) or "Translation unavailable",
            audioUrl=generate_lingo_audio(str(japanese_text)),
            isFinished=data.get("finished", data.get("isFinished", False)),
            isCorrect=True
        )
    except Exception as e:
        log.exception("Lingo conversation response failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        reset_current_user_id(token)

@router.post("/conversation/hint", response_model=ConversationResponse)
async def conversation_hint(session: dict = Depends(get_lingo_session)):
    from interface.webui import auth
    think = auth.aiko_web_instance._think
    uid = session.get("user_id")
    token = set_current_user_id(uid)
    try:
        system_prompt = (
            "You are Aiko, teaching Japanese through conversation. "
            "Suggest a concise response for the student to say in Japanese and provide an English translation. "
            "Keep the Japanese response under 30 words so it doesn't get cut off. "
            "The 'japanese' value must be in Japanese ONLY (no English). "
            "Output ONLY valid JSON. Your response must be a JSON object with 'japanese' and 'english' keys."
        )
        user_prompt = "Give me a hint for what I should say next in Japanese."

        # Retry logic for conversation hint
        max_retries = 2
        content = ""
        for attempt in range(max_retries + 1):
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
            log.info(f"Lingo conversation hint response (attempt {attempt}): {content}")

            try:
                data = parse_lingo_json(content)
                if data.get("japanese") or data.get("japaneseText"):
                    break
            except Exception as e:
                if attempt == max_retries:
                    raise
                log.warning(f"Lingo JSON parse failed on attempt {attempt}: {e}")

        data = parse_lingo_json(content)
        japanese_text = data.get("japanese") or data.get("japaneseText") or ""
        english_text = data.get("english") or data.get("englishTranslation") or ""
        return ConversationResponse(
            japaneseText=str(japanese_text),
            englishTranslation=str(english_text) or "Translation unavailable",
            audioUrl=generate_lingo_audio(str(japanese_text))
        )
    except Exception as e:
        log.exception("Lingo conversation hint failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        reset_current_user_id(token)

@router.post("/conversation/stop")
async def conversation_stop(session: dict = Depends(get_lingo_session)):
    return {"success": True}
