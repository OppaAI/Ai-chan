import os
import json
import uuid
import logging
from pathlib import Path
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
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
        return None

    # Filter out English characters to ensure she only speaks Japanese
    import re
    clean_text = re.sub(r'[a-zA-Z]', '', text).strip()

    if not clean_text:
        return None

    if not auth.aiko_web_instance or not auth.aiko_web_instance._speak:
        return None

    try:
        wav_bytes = auth.aiko_web_instance._speak._synthesize(clean_text)
        if not wav_bytes:
            return None

        audio_id = str(uuid.uuid4())
        filename = f"{audio_id}.wav"
        filepath = AUDIO_DIR / filename
        filepath.write_bytes(wav_bytes)

        base_url = getattr(auth, "REDIRECT_BASE", "https://aiko.ide-chroma.ts.net").rstrip("/")
        return f"{base_url}/lingo_audio/{filename}"
    except Exception as e:
        log.error(f"Failed to generate audio for Lingo: {e}")
        return None

async def get_lingo_session(request: Request):
    """Helper to get session from cookie or fallback to owner for the app."""
    from interface.webui import auth
    try:
        session = await auth.require_session(request)
        return await auth.require_accepted_session(session)
    except Exception:
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
            match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass

            match = re.search(r"(\{.*\})", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    fixed = re.sub(r'("\s*\n\s*")', '",\n"', match.group(1))
                    try:
                        return json.loads(fixed)
                    except Exception:
                        pass
            raise

    data = _extract(content)

    if isinstance(data, dict):
        if 'translations' in data and isinstance(data['translations'], list):
            for i, item in enumerate(data['translations']):
                if isinstance(item, dict):
                    for k, v in item.items():
                        if isinstance(v, dict):
                            data['translations'][i][k] = str(list(v.values())[0]) if v else ""

        for key in list(data.keys()):
            val = data[key]
            if isinstance(val, dict):
                found_string = None
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
        user_prompt = f"Translate the following English text to Japanese in 3-5 different registers: {request.text}"

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
    from interface.webui import auth
    if not auth.aiko_web_instance or not auth.aiko_web_instance._think:
        raise HTTPException(status_code=503, detail="Aiko's brain is not ready")

    think = auth.aiko_web_instance._think
    uid = session.get("user_id")

    async def event_generator():
        token = set_current_user_id(uid)
        try:
            system_prompt = (
                "You are Aiko, teaching Japanese through conversation. "
                "Output your response exactly like this:\n"
                "REPLY_JP: <Japanese sentences>\n"
                "REPLY_EN: <English translation>\n"
                "FINISHED: False"
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
            for chunk in response:
                delta = chunk.choices[0].delta.content
                if delta:
                    full_content += delta

                    # More robust parsing for streaming
                    clean_buf = full_content.replace("**", "")
                    if not is_streaming_jp and "REPLY_JP:" in clean_buf:
                        is_streaming_jp = True
                        # Start from after the tag
                        parts = clean_buf.split("REPLY_JP:", 1)
                        if len(parts) > 1:
                            text = parts[1]
                            # If next tag is already here, stop
                            stop_tags = ["REPLY_EN:", "FINISHED:"]
                            for tag in stop_tags:
                                if tag in text:
                                    text = text.split(tag, 1)[0]
                                    is_streaming_jp = False
                            if text:
                                yield json.dumps({"type": "delta", "text": text}) + "\n"
                        continue

                    if is_streaming_jp:
                        clean_delta = delta.replace("**", "")
                        # Check if this delta contains a stop tag
                        stop_tags = ["REPLY_EN:", "FINISHED:"]
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

            # Final structured data extraction using robust regex
            data = {"isCorrect": True}

            tags = ["REPLY_JP", "REPLY_EN", "FINISHED"]
            flat_content = full_content.replace("**", "").replace("`", "")

            import re
            tag_regex = r"([A-Z_]+)\s*:\s*(.*?)(?=\n[A-Z_]+\s*:|[A-Z_]+\s*:|$)"
            found_tags = re.findall(tag_regex, flat_content, re.DOTALL | re.IGNORECASE)

            for k, v in found_tags:
                k = k.strip().upper()
                v = v.strip()
                if k == "FINISHED": data["isFinished"] = v.lower() in ["true", "yes", "1"]
                elif k in ["REPLY_JP", "JAPANESE"]: data["japanese"] = v
                elif k in ["REPLY_EN", "ENGLISH"]: data["english"] = v

            # Fallback: if she just sent raw text without any tags, treat the whole thing as Japanese
            if not any(f"{t}:" in flat_content.upper() for t in tags) and not data.get("japanese"):
                data["japanese"] = full_content.strip()

            # Smart Fallback for Japanese
            if not data.get("japanese"):
                jp_match = re.search(r"([\u3040-\u30ff\u4e00-\u9fff].*?)(?:\n|$|REPLY_EN:|English:)", flat_content, re.DOTALL)
                if jp_match:
                    data["japanese"] = jp_match.group(1).strip()

            # Smart Fallback for English
            if data.get("english"):
                jp_chars = len(re.findall(r"[\u3040-\u30ff\u4e00-\u9fff]", data["english"]))
                if jp_chars > len(data["english"]) / 2:
                    data["english"] = None

            if not data.get("english"):
                en_match = re.search(r"([a-zA-Z][a-zA-Z\s,.'\"?!-]{10,})", flat_content)
                if en_match:
                    data["english"] = en_match.group(1).strip()

            audio_url = generate_lingo_audio(data.get("japanese"))
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

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@router.post("/conversation/respond_stream")
async def conversation_respond_stream(request: RespondRequest, session: dict = Depends(get_lingo_session)):
    from interface.webui import auth
    if not auth.aiko_web_instance or not auth.aiko_web_instance._think:
        raise HTTPException(status_code=503, detail="Aiko's brain is not ready")

    think = auth.aiko_web_instance._think
    uid = session.get("user_id")

    async def event_generator():
        token = set_current_user_id(uid)
        try:
            # Combine Aiko's real persona with Lingo instructions
            base_prompt = think._current_system_prompt(request.text)
            system_prompt = (
                f"{base_prompt}\n\n"
                "--- LINGO INSTRUCTIONS ---\n"
                "You are acting as a Japanese teacher. Maintain a natural roleplay conversation. "
                "When the student responds, check their Japanese for grammar, spelling, or unnatural usage. "
                "Format your response EXACTLY like this:\n"
                "MISTAKE: <True/False>\n"
                "FEEDBACK: <Explanation in English of the mistake, or empty if no mistake>\n"
                "SUGGESTION: <Corrected Japanese version of what the student said, or empty if no mistake>\n"
                "REPLY_JP: <Your NEXT Japanese conversation turn to keep the dialogue going>\n"
                "REPLY_EN: <English translation of your next turn>\n"
                "FINISHED: <True if the conversation is naturally over (e.g. they said goodbye), otherwise False>"
            )

            # Strict role alternation for llama-server
            messages = [{"role": "system", "content": system_prompt}]

            # Build pending list from history
            pending = []
            if request.history:
                for entry in request.history:
                    # Filter out empty or placeholder text
                    text = entry.text.strip()
                    if not text or text in ["...", "*"]:
                        continue

                    role = "assistant" if entry.speaker == "aiko" else "user"
                    if not pending:
                        if role == "user":
                            pending.append({"role": role, "content": text})
                        else:
                            pending.append({"role": "user", "content": "I'm ready to talk Japanese."})
                            pending.append({"role": "assistant", "content": text})
                    else:
                        if pending[-1]["role"] == role:
                            pending[-1]["content"] += "\n" + text
                        else:
                            pending.append({"role": role, "content": text})

            # Current user input
            user_text = request.text.strip()
            if pending and pending[-1]["role"] == "user":
                pending[-1]["content"] += "\n" + user_text
            else:
                pending.append({"role": "user", "content": user_text})

            messages.extend(pending)

            response = think._client.chat.completions.create(
                model=think._llm_model,
                messages=messages,
                stream=True,
                timeout=120.0,
                temperature=0.7 # Higher temperature for more variety
            )

            full_content = ""
            is_streaming_jp = False
            for chunk in response:
                delta = chunk.choices[0].delta.content
                if delta:
                    full_content += delta

                    # Filter for streaming Japanese
                    # We want to stream either SUGGESTION: (if mistake) or REPLY_JP: (if correct)
                    clean_buf = full_content.replace("**", "")
                    target_tag = "SUGGESTION:" if "MISTAKE: TRUE" in clean_buf.upper() else "REPLY_JP:"

                    if not is_streaming_jp:
                        if target_tag in clean_buf:
                            is_streaming_jp = True
                            parts = clean_buf.split(target_tag, 1)
                            if len(parts) > 1:
                                text = parts[1]
                                stop_tags = ["REPLY_EN:", "FINISHED:", "REPLY_JP:"]
                                if target_tag in stop_tags: stop_tags.remove(target_tag)
                                for tag in stop_tags:
                                    if tag in text:
                                        text = text.split(tag, 1)[0]
                                        is_streaming_jp = False
                                if text:
                                    yield json.dumps({"type": "delta", "text": text}) + "\n"
                        elif not any(t in clean_buf.upper() for t in ["MISTAKE:", "REPLY_JP:", "REPLY_EN:"]) and len(clean_buf) > 10:
                            # Fallback: if it's just raw text, start streaming it
                            is_streaming_jp = True
                            yield json.dumps({"type": "delta", "text": delta.replace("**", "")}) + "\n"
                        continue

                    if is_streaming_jp:
                        clean_delta = delta.replace("**", "")
                        stop_tags = ["REPLY_EN:", "FINISHED:", "REPLY_JP:", "SUGGESTION:"]
                        if target_tag in stop_tags: stop_tags.remove(target_tag)
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

            # Final structured data extraction using robust regex
            data = {"isCorrect": True}

            # Map of internal keys to possible LLM tag variations
            tags = ["MISTAKE", "FEEDBACK", "SUGGESTION", "REPLY_JP", "REPLY_EN", "FINISHED"]

            # Remove markdown bolding and backticks
            flat_content = full_content.replace("**", "").replace("`", "")

            import re
            # Extract all tag-like patterns: TAG_NAME: content
            # The lookahead ensures we stop before the next tag
            tag_regex = r"([A-Z_]+)\s*:\s*(.*?)(?=\n[A-Z_]+\s*:|[A-Z_]+\s*:|$)"
            found_tags = re.findall(tag_regex, flat_content, re.DOTALL | re.IGNORECASE)

            for k, v in found_tags:
                k = k.strip().upper()
                v = v.strip()
                if k == "MISTAKE": data["isCorrect"] = v.lower() not in ["true", "yes", "y", "1"]
                elif k == "FINISHED": data["isFinished"] = v.lower() in ["true", "yes", "y", "1"]
                elif k in ["REPLY_JP", "JAPANESE"]: data["japanese"] = v
                elif k in ["REPLY_EN", "ENGLISH"]: data["english"] = v
                elif k == "FEEDBACK": data["feedback"] = v
                elif k == "SUGGESTION": data["suggestion"] = v

            # Fallback: if she just sent raw text without any tags, treat the whole thing as Japanese
            if not any(f"{t}:" in flat_content.upper() for t in tags) and not data.get("japanese"):
                data["japanese"] = full_content.strip()
                data["isCorrect"] = True

            # Smart Fallback for Japanese: search for actual Japanese characters if tag is missing
            if not data.get("japanese") and data.get("isCorrect"):
                jp_match = re.search(r"([\u3040-\u30ff\u4e00-\u9fff].*?)(?:\n|$|REPLY_EN:|English:)", flat_content, re.DOTALL)
                if jp_match:
                    data["japanese"] = jp_match.group(1).strip()

            # Smart Fallback for English: ensure English field doesn't accidentally contain Japanese
            if data.get("english"):
                # If 'english' contains lots of Japanese characters, it's probably a wrong match
                jp_chars = len(re.findall(r"[\u3040-\u30ff\u4e00-\u9fff]", data["english"]))
                if jp_chars > len(data["english"]) / 2:
                    data["english"] = None # Reset it to search again

            if not data.get("english"):
                # Search for a block of English text (basic letters and spaces)
                en_match = re.search(r"([a-zA-Z][a-zA-Z\s,.'\"?!-]{10,})", flat_content)
                if en_match:
                    data["english"] = en_match.group(1).strip()

            # Determine final display text and audio text
            is_mistake = not data.get("isCorrect", True)

            if is_mistake:
                feedback = data.get("feedback") or "Grammar/usage error."
                suggestion = data.get("suggestion") or ""
                final_jp = f"Mistake: {feedback}"
                if suggestion:
                    final_jp += f"\nSuggestion: {suggestion}"
                audio_text = suggestion or feedback
            else:
                final_jp = data.get("japanese") or "..."
                audio_text = final_jp

            audio_url = generate_lingo_audio(audio_text)

            yield json.dumps({
                "type": "final",
                "isCorrect": data.get("isCorrect", True),
                "feedback": data.get("feedback"),
                "suggestion": data.get("suggestion"),
                "japanese": final_jp,
                "english": data.get("english") or "Please correct your Japanese ♡",
                "isFinished": data.get("isFinished", False),
                "audioUrl": audio_url
            }) + "\n"
        except Exception as e:
            log.exception("Lingo streaming failed")
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"
        finally:
            reset_current_user_id(token)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@router.post("/conversation/hint", response_model=ConversationResponse)
async def conversation_hint(session: dict = Depends(get_lingo_session)):
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
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": "Hint please"}],
            response_format={"type": "json_object"},
            timeout=120.0
        )
        data = parse_lingo_json(response.choices[0].message.content)
        jp = data.get("japanese") or data.get("japaneseText") or ""
        en = data.get("english") or data.get("englishTranslation") or "Translation unavailable"
        return ConversationResponse(
            japaneseText=str(jp),
            englishTranslation=str(en),
            audioUrl=generate_lingo_audio(str(jp)),
            isCorrect=True
        )
    except Exception as e:
        log.exception("Lingo conversation hint failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        reset_current_user_id(token)

@router.get("/tts")
async def get_tts(text: str, session: dict = Depends(get_lingo_session)):
    url = generate_lingo_audio(text)
    if not url:
        raise HTTPException(status_code=500, detail="TTS generation failed")
    return {"audioUrl": url}

@router.post("/conversation/stop")
async def conversation_stop(session: dict = Depends(get_lingo_session)):
    return {"success": True}
