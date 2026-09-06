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

class RespondRequest(BaseModel):
    text: str

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

def generate_lingo_audio(text: str) -> Optional[str]:
    """Synthesize Japanese text using AikoSpeak and return a public URL."""
    from interface.webui import auth
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
    # (handles cases like {"japanese": {"message": "..."}})
    if isinstance(data, dict):
        for key in list(data.keys()):
            val = data[key]
            if isinstance(val, dict):
                # If it's a dict with a single key like 'message' or 'text', flatten it
                if len(val) == 1:
                    data[key] = list(val.values())[0]
                elif 'text' in val:
                    data[key] = val['text']
                elif 'message' in val:
                    data[key] = val['message']
                elif 'japanese' in val:
                    data[key] = val['japanese']

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
            "Output ONLY valid JSON. Your response must be a JSON object with 'japanese' and 'english' keys."
        )
        user_prompt = f"Start a Japanese conversation at {request.level} level. Speak 1-3 sentences in Japanese."

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
        log.debug(f"Lingo conversation start response: {content}")

        data = parse_lingo_json(content)
        japanese_text = data.get("japanese", "")
        english_text = data.get("english", "")
        return ConversationResponse(
            japaneseText=japanese_text,
            englishTranslation=english_text,
            audioUrl=generate_lingo_audio(japanese_text),
            isFinished=data.get("finished", False)
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
            "You are Aiko, teaching Japanese through conversation. "
            "Output ONLY valid JSON. Your response must be a JSON object with 'japanese' and 'english' keys."
        )

        response = think._client.chat.completions.create(
            model=think._llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.text}
            ],
            response_format={"type": "json_object"},
            timeout=120.0
        )

        content = response.choices[0].message.content
        log.debug(f"Lingo conversation respond response: {content}")

        data = parse_lingo_json(content)
        japanese_text = data.get("japanese", "")
        english_text = data.get("english", "")
        return ConversationResponse(
            japaneseText=japanese_text,
            englishTranslation=english_text,
            audioUrl=generate_lingo_audio(japanese_text),
            isFinished=data.get("finished", False)
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
            "Suggest a response for the student to say in Japanese and provide an English translation. "
            "Output ONLY valid JSON. Your response must be a JSON object with 'japanese' and 'english' keys."
        )
        user_prompt = "Give me a hint for what I should say next in Japanese."

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
        log.debug(f"Lingo conversation hint response: {content}")

        data = parse_lingo_json(content)
        japanese_text = data.get("japanese", "")
        english_text = data.get("english", "")
        return ConversationResponse(
            japaneseText=japanese_text,
            englishTranslation=english_text,
            audioUrl=generate_lingo_audio(japanese_text)
        )
    except Exception as e:
        log.exception("Lingo conversation hint failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        reset_current_user_id(token)

@router.post("/conversation/stop")
async def conversation_stop(session: dict = Depends(get_lingo_session)):
    return {"success": True}
