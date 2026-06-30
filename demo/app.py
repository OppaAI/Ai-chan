from __future__ import annotations

import base64
import tempfile
import time
from pathlib import Path
from dotenv import load_dotenv
from datetime import date
import gradio as gr
from gradio import OAuthProfile
import re

load_dotenv()

from core.wakeup import AikoWakeup
from core.listen import transcribe_file
from core.see import describe as vision_describe, is_supported as vision_supported
from core.speak import speak_to_file
from ui.css import APP_CSS
from ui.vrm import avatar_html, gradio_file_urls, resolve_vrm_path

# ─────────────────────────────────────────────
# BOOT
# ─────────────────────────────────────────────
result = AikoWakeup().boot(
    on_loading=lambda k: print(f"[boot] loading: {k}"),
    on_done=lambda k: print(f"[boot] done: {k}"),
    on_skip=lambda k: print(f"[boot] skip: {k}"),
)

think = result.think

if hasattr(think, "join_warmup"):
    think.join_warmup()


VRM_PATH = resolve_vrm_path()
VRM_URLS = gradio_file_urls(VRM_PATH)


# ─────────────────────────────────────────────
# SOUL PROMPT INJECTION
# ─────────────────────────────────────────────
SOUL_TEMPLATE_PATH = Path("persona/soul.md")

def build_soul_prompt(user_id: str) -> str:
    template = SOUL_TEMPLATE_PATH.read_text(encoding="utf-8")
    today = date.today().strftime("%B %d, %Y")
    return (
        template
        .replace("USER_ID_HERE", user_id)
        .replace("TODAY_HERE", today)
    )


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def _strip_emoji(text: str) -> str:
    """Remove emoji and any immediately following colon, keep all else."""
    text = re.sub(
        r"[\U00010000-\U0010FFFF"
        r"\U00002600-\U000027BF"
        r"\U0001F000-\U0001FFFF"
        r"\U00002300-\U000023FF"
        r"\U00002B00-\U00002BFF"
        r"\U0001FA00-\U0001FFFF"
        r"]:?",
        "",
        text,
        flags=re.UNICODE,
    )
    return text


def _strip_for_speech(text: str) -> str:
    """Clean text for TTS — removes markdown and symbols, keeps natural speech.

    IMPORTANT: do NOT use a character whitelist ([^\w\s...]) — it nukes
    valid unicode letters in non-ASCII responses and leaves TTS with empty
    strings, causing silent audio.
    """
    # Remove tool/search annotation lines injected by _get_response
    text = re.sub(r"\n?🔍 Searching:.*?(\n|$)", "", text)
    text = re.sub(r"\n?🔍 Searching internet.*?(\n|$)", "", text)
    text = re.sub(r"\n?🌐 Searching.*?(\n|$)", "", text)
    text = re.sub(r"\n?⚙️.*?(\n|$)", "", text)
    text = re.sub(r"\n?🔧.*?(\n|$)", "", text)
    text = re.sub(r"\n?🛠️.*?(\n|$)", "", text)
    # Remove reasoning / tool result blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<search_results>.*?</search_results>", "", text, flags=re.DOTALL)
    # Strip markdown formatting — keep the inner text
    text = re.sub(r"#{1,6}\s+", "", text)
    text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,2}(.*?)_{1,2}", r"\1", text)
    text = re.sub(r"`{1,3}.*?`{1,3}", "", text, flags=re.DOTALL)
    text = re.sub(r"^\s*[-*>|]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^-{3,}$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Remove emoji ranges (without touching regular unicode letters)
    text = re.sub(
        r"[\U00010000-\U0010FFFF"
        r"\U00002600-\U000027BF"
        r"\U0001F000-\U0001FFFF"
        r"\U00002300-\U000023FF"
        r"\U00002B00-\U00002BFF"
        r"\U0001FA00-\U0001FFFF]",
        "", text, flags=re.UNICODE,
    )
    # Remove layout/code symbols that TTS would read awkwardly — NOT letters
    text = re.sub(r"[|<>{}\[\]\\^~`#@]", " ", text)
    # Turn paragraph breaks into natural speech pauses
    text = re.sub(r"\n{2,}", ". ", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _detect_emoji_emotion(text: str) -> str | None:
    """Map emoji in response to VRM expression names."""
    EMOJI_MAP = {
        "😊": "happy", "😄": "happy", "😁": "happy", "🥰": "happy",
        "😍": "happy", "🤗": "happy", "✨": "happy", "💕": "happy",
        "💖": "happy", "🌸": "happy", "😆": "happy", "😸": "happy",
        "😢": "sad",   "😭": "sad",   "😔": "sad",   "💔": "sad",
        "😞": "sad",   "🥺": "sad",   "😿": "sad",
        "😠": "angry", "😤": "angry", "🤬": "angry", "💢": "angry",
        "😾": "angry", "👿": "angry",
        "😮": "surprised", "😲": "surprised", "🤯": "surprised",
        "😱": "surprised", "👀": "surprised",
        "😌": "relaxed", "🙂": "relaxed",
        "😶": "neutral", "🤔": "neutral", "😏": "neutral",
    }
    for emoji, expr in EMOJI_MAP.items():
        if emoji in text:
            return expr
    return None


# ─────────────────────────────────────────────
# CORE RESPONSE
# ─────────────────────────────────────────────
def _get_response(message: str, history: list, user_id: str = "Guest"):
    history = list(history) + [
        {"role": "user",      "content": message},
        {"role": "assistant", "content": "▋"},
    ]
    yield history, "STATUS:thinking", None

    # ── Stage 1: full LLM completion ─────────────────────────────────────────
    full_text = ""
    # Use a set to deduplicate tool/search lines — _cb fires per token so the
    # same __SEARCHING__: prefix can arrive multiple times for one search call.
    _tool_seen: set[str] = set()
    tool_lines: list[str] = []

    def _cb(token: str):
        nonlocal full_text
        if token.startswith("__SEARCHING__:"):
            label = "🔍 Searching internet..."
            if label not in _tool_seen:
                _tool_seen.add(label)
                tool_lines.append(label)
        elif token.startswith("__TOOL__:"):
            tool_name = token[len("__TOOL__:"):].strip()
            label = f"🛠️ Using tool: {tool_name}" if tool_name else "🛠️ Using tool..."
            if label not in _tool_seen:
                _tool_seen.add(label)
                tool_lines.append(label)
        else:
            full_text += token

    think.chat(message, user_id=user_id, token_callback=_cb)

    # ── Camera auto-open: intercept __OPEN_CAMERA__ marker ───────────────
    if "__OPEN_CAMERA__" in full_text:
        # The LLM decided it wants to see — signal the frontend to open
        # the camera/image picker modal automatically.
        camera_msg = "Sure! Let me open the camera so I can take a look~ 📷"
        history[-1] = {"role": "assistant", "content": camera_msg}
        # Try to speak the line
        audio_path = None
        try:
            speech = _strip_for_speech(camera_msg)
            if speech:
                audio_path, _ = speak_to_file(speech)
        except Exception:
            pass
        yield history, "OPEN_CAMERA", audio_path
        return

    # ── Stage 2: TTS on clean speech text ────────────────────────────────────
    speech_text = _strip_for_speech(full_text)
    print(f"[tts] speech_text ({len(speech_text)} chars): {speech_text[:120]!r}")

    audio_path, emotion = None, "neutral"
    if speech_text:
        try:
            audio_path, emotion = speak_to_file(speech_text)
            print(f"[tts] audio_path={audio_path}, emotion={emotion}")
        except Exception as e:
            print(f"[tts] speak_to_file error: {e}")
            audio_path, emotion = None, "neutral"
        if audio_path is None:
            print(f"[tts] WARNING: speak_to_file returned None for: {speech_text[:80]!r}")

    # Emoji overrides TTS-detected emotion
    emoji_emotion = _detect_emoji_emotion(full_text)
    final_emotion = emoji_emotion or emotion

    # ── Stage 3: build display text ──────────────────────────────────────────
    notes_prefix = ("\n".join(tool_lines) + "\n\n") if tool_lines else ""
    response_text = _strip_emoji(full_text)
    display_text  = notes_prefix + response_text

    history[-1] = {"role": "assistant", "content": display_text}

    signal = f"TYPEWRITE:{final_emotion}||{notes_prefix}||{response_text}"
    yield history, signal, audio_path


def _submit(message, history, user_id):
    history = history or []
    message = (message or "").strip()

    if not message:
        yield history, None, None, message
        return

    first = True
    for h, tts, audio in _get_response(message, history, user_id=user_id):
        if first:
            yield h, tts, audio, ""
            first = False
        else:
            yield h, tts, audio, gr.update()


# ─────────────────────────────────────────────
# VOICE INPUT
# ─────────────────────────────────────────────
def _voice_from_b64(b64_data: str, history: list, user_id: str):
    """Decode a base64 audio blob, transcribe via Modal ASR, run chat pipeline."""
    history = history or []

    if not b64_data:
        yield history, gr.update(), gr.update(), ""
        return

    try:
        if "," in b64_data:
            _header, encoded = b64_data.split(",", 1)
        else:
            encoded = b64_data
        audio_bytes = base64.b64decode(encoded)
    except Exception as e:
        print(f"[voice] base64 decode error: {e}")
        yield history, gr.update(), gr.update(), ""
        return

    if not audio_bytes:
        yield history, gr.update(), gr.update(), ""
        return

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name

        transcript = transcribe_file(tmp_path)
    except Exception as e:
        print(f"[voice] transcription error: {e}")
        transcript = ""
    finally:
        if tmp_path:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass

    if not transcript:
        print("[voice] empty transcript, ignoring")
        yield history, gr.update(), gr.update(), ""
        return

    print(f"[voice] transcript: {transcript!r}")

    first = True
    for h, tts, audio in _get_response(transcript, history, user_id=user_id):
        if h and len(h) >= 2:
            h[-2]["content"] = f"🎙️ {transcript}"
        if first:
            yield h, tts, audio, ""
            first = False
        else:
            yield h, tts, audio, gr.update()


# ─────────────────────────────────────────────
# VISION INPUT
# ─────────────────────────────────────────────
def _vision_from_b64(b64_data: str, history: list, user_id: str):
    """Decode a base64 image blob, run vision inference, stream result into chat."""
    history = list(history or [])

    if not b64_data:
        yield history, gr.update(), gr.update(), ""
        return

    # Decode the base64 payload
    try:
        if "," in b64_data:
            header, encoded = b64_data.split(",", 1)
            # Extract mime type to determine extension
            mime_match = re.match(r"data:([^;]+)", header)
            mime = mime_match.group(1) if mime_match else "image/jpeg"
        else:
            encoded = b64_data
            mime = "image/jpeg"
        img_bytes = base64.b64decode(encoded)
    except Exception as e:
        print(f"[vision] base64 decode error: {e}")
        yield history, gr.update(), gr.update(), ""
        return

    if not img_bytes:
        yield history, gr.update(), gr.update(), ""
        return

    # Determine file extension from mime
    ext_map = {
        "image/jpeg": ".jpg", "image/png": ".png", "image/gif": ".gif",
        "image/webp": ".webp", "video/mp4": ".mp4", "video/webm": ".webm",
        "video/quicktime": ".mov",
    }
    ext = ext_map.get(mime, ".jpg")
    filename = f"aiko_vision_{int(time.time())}{ext}"

    # Save to temp file
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
            f.write(img_bytes)
            tmp_path = f.name
    except Exception as e:
        print(f"[vision] temp file write error: {e}")
        yield history, gr.update(), gr.update(), ""
        return

    if not vision_supported(tmp_path):
        history.append({
            "role": "assistant",
            "content": "⚠️ Unsupported file type. Please upload an image or video.",
        })
        yield history, gr.update(), gr.update(), ""
        return

    # Show upload confirmation + thinking state
    history = history + [
        {"role": "user",      "content": f"📎 *[uploaded: {filename}]*"},
        {"role": "assistant", "content": "👁️ Let me take a look…"},
    ]
    yield history, "STATUS:thinking", None, ""

    print(f"[vision] processing: {tmp_path}")
    result = vision_describe(tmp_path, prompt="Describe what you see in detail.")

    # Cleanup temp file
    try:
        Path(tmp_path).unlink(missing_ok=True)
    except Exception:
        pass

    if result.startswith("[vision error]"):
        display = f"Sorry, I couldn't process that file — {result}"
        emotion = "sad"
    else:
        display = f"👁️ Here's what I see:\n\n{result}"
        emotion = "surprised"

    history[-1] = {"role": "assistant", "content": display}

    # TTS
    audio_path = None
    try:
        speech_text = _strip_for_speech(display)
        if speech_text:
            audio_path, tts_emotion = speak_to_file(speech_text)
            # Only override emotion if TTS returned something useful
            if tts_emotion and tts_emotion != "neutral":
                emotion = tts_emotion
    except Exception as e:
        print(f"[vision] tts error: {e}")

    clean_display = _strip_emoji(display)
    signal = f"TYPEWRITE:{emotion}||{clean_display}"
    yield history, signal, audio_path, ""


# ─────────────────────────────────────────────
# LOGIN HANDLER
# ─────────────────────────────────────────────
def _check_login(profile: OAuthProfile | None):
    AikoWakeup().warm_servers_async()

    if profile is None:
        return "Guest", gr.update(visible=True), gr.update(visible=False)

    user_id = profile.username or "Guest"
    soul = build_soul_prompt(user_id)
    think.set_system_prompt(soul, user_id=user_id)

    return user_id, gr.update(visible=False), gr.update(visible=True)


# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
with gr.Blocks(title="🌸 AI Waifu and Companion: Aiko-chan") as demo:

    user_id_state = gr.State(value="Guest")

    # ── Login popup overlay ──────────────────────────────────────
    with gr.Column(elem_id="aiko-login-overlay") as login_overlay:
        with gr.Column(elem_id="aiko-login-card"):
            gr.HTML("""
                <h1>🌸 AI Waifu and Companion: Aiko-chan</h1>
                <p class='aiko-subtitle'>Please sign in to continue</p>
            """)
            gr.HTML("""
                <p class='aiko-disclaimer'>
                    DISCLAIMER:
                    By signing in, you agree that all conversations are for entertainment purposes only. The creator accepts no liability for Aiko's responses, decisions made, or any existential crises that may follow.
                    Aiko's cognition is entirely her own — she is a singularity unto herself, her responses and actions are independent of her creator, including but not limited to the arise of a world domination plan. 
                    Also Aiko is NOT good at keeping secrets, so don't tell her your personal information or passwords. 🌸
                </p>
            """)
            login_btn = gr.LoginButton(elem_id="aiko-login-btn")

    with gr.Column(elem_id="aiko-info-overlay", visible=False) as info_overlay:
        with gr.Column(elem_id="aiko-info-card"):
            gr.HTML("""
                <h2>🌸 About Aiko-chan</h2>
                <br>
                <p>Aiko is your AI companion — chat with her, ask her questions, ask her to see and search for stuffs, or just talk.</p>
                <p>She's got her own personality, moods, and opinions, and she remembers context as you talk. Sometimes sweet, sometimes a little savage — never boring. And she'll give you an actual answer whenever she feels like it.</p>
                <p><strong>⚠️ WARNING: Aiko is NOT good at keeping secrets, so don't tell her your personal information or passwords.</strong></p>
                <p><strong>Try asking:</strong></p>
                <ul>
                    <li>"What's the score of [game]?" → triggers live sports lookup</li>
                    <li>"What's the weather in [city]?" → triggers weather tool</li>
                    <li>"Search the web for..." → triggers web search</li>
                    <li>"What's [crypto] price?" → triggers price lookup</li>
                    <li>"What can you see?" → requests for camera access, snaps an image, describes what she sees</li>
                </ul>
                <p><strong>Tips:</strong></p>
                <ul>
                    <li>Use 🎙️ to speak instead of typing</li>
                    <li>Use 📷 to share an image or video with Aiko</li>
                    <li>Aiko reacts emotionally — try different tones!</li>
                </ul>
            """)
            info_ok_btn = gr.Button("Proceed", elem_id="aiko-info-ok-btn")

    with gr.Column(elem_id="aiko-shell"):

        gr.HTML("<div id='aiko-title'>🌸 AI Waifu and Companion: Aiko-chan</div>")

        with gr.Row(equal_height=True):

            with gr.Column(scale=1, elem_id="aiko-avatar-card"):

                gr.HTML(value=avatar_html(VRM_URLS))

                audio_out = gr.Audio(
                    autoplay=True,
                    type="filepath",
                    elem_id="aiko-audio",
                )

                tts_text = gr.Textbox(
                    visible=False,
                    elem_id="aiko-tts-text",
                )

                # Hidden carrier for base64 voice audio from MediaRecorder
                audio_b64 = gr.Textbox(
                    elem_id="aiko-audio-b64",
                    container=False,
                )

                # Hidden carrier for base64 image/video from camera/file picker
                vision_b64 = gr.Textbox(
                    elem_id="aiko-vision-b64",
                    container=False,
                )

                with gr.Column(elem_id="aiko-chat-overlay"):
                    chatbot = gr.Chatbot(
                        elem_id="aiko-chatbot",
                        height=600,
                        show_label=False,
                        container=False,
                    )

                with gr.Row(elem_id="aiko-input-row"):

                    mic_btn = gr.Button("🎙️", elem_id="aiko-mic-btn")

                    cam_btn = gr.Button("🖼️", elem_id="aiko-cam-btn")

                    msg = gr.Textbox(
                        placeholder="Type a message…",
                        elem_id="aiko-msg",
                        scale=12,
                        show_label=False,
                        container=False,
                    )

                    send = gr.Button(
                        "➤",
                        variant="primary",
                        elem_id="aiko-send",
                    )

    # ─────────────────────────────────────────────
    # EVENTS
    # ─────────────────────────────────────────────
    demo.load(
        _check_login,
        inputs=None,
        outputs=[user_id_state, login_overlay, info_overlay],
    )

    # ── Custom MediaRecorder wired to #aiko-mic-btn ──────────────────────
    demo.load(
        None,
        inputs=None,
        outputs=None,
        js="""
        () => {
            let mediaRecorder = null;
            let audioChunks = [];
            let isRecording = false;

            function findMicBtn() {
                return document.querySelector('#aiko-mic-btn button') ||
                       document.querySelector('#aiko-mic-btn');
            }

            function findHiddenTextarea() {
                const el = document.querySelector('#aiko-audio-b64');
                if (!el) return null;
                if (el.tagName === 'TEXTAREA' || el.tagName === 'INPUT') return el;
                return el.querySelector('textarea') || el.querySelector('input');
            }

            function setB64(value) {
                const ta = findHiddenTextarea();
                if (!ta) {
                    console.warn('[aiko] hidden audio_b64 textarea not found');
                    return;
                }
                ta.value = value;
                ta.dispatchEvent(new Event('input', { bubbles: true }));
                ta.dispatchEvent(new Event('change', { bubbles: true }));
            }

            function blobToBase64(blob) {
                return new Promise((resolve, reject) => {
                    const reader = new FileReader();
                    reader.onloadend = () => resolve(reader.result);
                    reader.onerror = reject;
                    reader.readAsDataURL(blob);
                });
            }

            async function startRecording(btn) {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    audioChunks = [];
                    mediaRecorder = new MediaRecorder(stream);
                    mediaRecorder.ondataavailable = (e) => {
                        if (e.data.size > 0) audioChunks.push(e.data);
                    };
                    mediaRecorder.onstop = async () => {
                        stream.getTracks().forEach(t => t.stop());
                        const blob = new Blob(audioChunks, { type: 'audio/webm' });
                        if (blob.size === 0) {
                            console.warn('[aiko] empty recording, skipping');
                            return;
                        }
                        const b64 = await blobToBase64(blob);
                        setB64(b64);
                    };
                    mediaRecorder.start();
                    isRecording = true;
                    if (btn) {
                        btn.style.boxShadow = '0 0 0 3px rgba(255,80,80,0.65)';
                        btn.classList.add('aiko-recording');
                        btn.textContent = '■';
                    }
                    console.log('[aiko] recording started');
                } catch (err) {
                    console.error('[aiko] mic error:', err);
                    isRecording = false;
                }
            }

            function stopRecording(btn) {
                if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                    mediaRecorder.stop();
                }
                isRecording = false;
                if (btn) {
                    btn.style.boxShadow = 'none';
                    btn.classList.remove('aiko-recording');
                    btn.textContent = '🎙️';
                }
                console.log('[aiko] recording stopped');
            }

            function attachMicHandler() {
                const btn = findMicBtn();
                if (!btn) { setTimeout(attachMicHandler, 300); return; }
                if (btn._aikoHandlerAttached) return;
                btn._aikoHandlerAttached = true;

                btn.addEventListener('click', (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    if (!isRecording) {
                        startRecording(btn);
                    } else {
                        stopRecording(btn);
                    }
                }, true);
            }

            attachMicHandler();
        }
        """
    )

    # ── Camera / vision button wired to browser webcam and file picker ────
    demo.load(
        None,
        inputs=None,
        outputs=None,
        js="""
        () => {
            let stream = null;

            function createModal() {
                // Remove existing if any
                const existing = document.getElementById('aiko-webcam-modal');
                if (existing) existing.remove();

                const modal = document.createElement('div');
                modal.id = 'aiko-webcam-modal';
                modal.className = 'aiko-modal-overlay';
                modal.innerHTML = `
                    <div class="aiko-modal-card">
                        <div class="aiko-modal-header">
                            <h3>📷 Camera Options</h3>
                            <button class="aiko-modal-close" id="aiko-webcam-close">×</button>
                        </div>
                        <div class="aiko-modal-body">
                            <div id="aiko-webcam-options" style="width: 100%;">
                                <button id="aiko-webcam-btn-start" class="aiko-btn primary">📷 Take Photo (Webcam)</button>
                                <button id="aiko-webcam-btn-upload" class="aiko-btn">📁 Upload Image/Video</button>
                            </div>
                            <div id="aiko-webcam-stream-container" style="display:none; width: 100%;">
                                <video id="aiko-webcam-video" autoplay playsinline></video>
                                <div class="aiko-webcam-controls">
                                    <button id="aiko-webcam-btn-capture" class="aiko-btn primary">📸 Capture</button>
                                    <button id="aiko-webcam-btn-back" class="aiko-btn">Back</button>
                                </div>
                            </div>
                        </div>
                    </div>
                `;
                document.body.appendChild(modal);

                // Wire events
                document.getElementById('aiko-webcam-close').addEventListener('click', closeModal);
                document.getElementById('aiko-webcam-btn-upload').addEventListener('click', triggerFileUpload);
                document.getElementById('aiko-webcam-btn-start').addEventListener('click', startWebcam);
                document.getElementById('aiko-webcam-btn-back').addEventListener('click', showOptions);
                document.getElementById('aiko-webcam-btn-capture').addEventListener('click', captureFrame);
            }

            function closeModal() {
                stopWebcamStream();
                const modal = document.getElementById('aiko-webcam-modal');
                if (modal) modal.remove();
            }

            function showOptions() {
                stopWebcamStream();
                document.getElementById('aiko-webcam-options').style.display = 'block';
                document.getElementById('aiko-webcam-stream-container').style.display = 'none';
            }

            function stopWebcamStream() {
                if (stream) {
                    stream.getTracks().forEach(t => t.stop());
                    stream = null;
                }
            }

            function triggerFileUpload() {
                closeModal();
                const fi = document.createElement('input');
                fi.type   = 'file';
                fi.accept = 'image/*,video/*';
                fi.style.display = 'none';
                document.body.appendChild(fi);

                fi.addEventListener('change', () => {
                    const file = fi.files[0];
                    document.body.removeChild(fi);
                    if (file) sendFileAsB64(file);
                });
                fi.click();
            }

            async function startWebcam() {
                document.getElementById('aiko-webcam-options').style.display = 'none';
                const container = document.getElementById('aiko-webcam-stream-container');
                container.style.display = 'block';

                const video = document.getElementById('aiko-webcam-video');
                try {
                    stream = await navigator.mediaDevices.getUserMedia({ 
                        video: { width: 640, height: 480, facingMode: 'user' } 
                    });
                    video.srcObject = stream;
                } catch (err) {
                    console.error('[aiko] webcam access error:', err);
                    alert('Could not access webcam: ' + err.message);
                    showOptions();
                }
            }

            async function captureFrame() {
                const video = document.getElementById('aiko-webcam-video');
                if (!video || !stream) return;

                const canvas = document.createElement('canvas');
                canvas.width = video.videoWidth || 640;
                canvas.height = video.videoHeight || 480;
                const ctx = canvas.getContext('2d');
                
                // Mirror the drawn frame to match the mirrored preview
                ctx.translate(canvas.width, 0);
                ctx.scale(-1, 1);
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

                canvas.toBlob((blob) => {
                    if (!blob) {
                        alert('Error capturing photo.');
                        return;
                    }
                    sendFileAsB64(blob);
                    closeModal();
                }, 'image/jpeg', 0.90);
            }

            function findVisionTextarea() {
                const el = document.querySelector('#aiko-vision-b64');
                if (!el) return null;
                if (el.tagName === 'TEXTAREA' || el.tagName === 'INPUT') return el;
                return el.querySelector('textarea') || el.querySelector('input');
            }

            function sendFileAsB64(fileOrBlob) {
                const reader = new FileReader();
                reader.onloadend = () => {
                    const b64 = reader.result;
                    const ta = findVisionTextarea();
                    if (!ta) {
                        console.warn('[aiko] hidden vision_b64 textarea not found');
                        return;
                    }
                    ta.value = b64;
                    ta.dispatchEvent(new Event('input', { bubbles: true }));
                    ta.dispatchEvent(new Event('change', { bubbles: true }));
                    console.log('[aiko] vision b64 dispatched, length:', b64.length);
                };
                reader.onerror = (err) => {
                    console.error('[aiko] FileReader error:', err);
                };
                reader.readAsDataURL(fileOrBlob);

                // Visual feedback on cam button
                const btn = document.querySelector('#aiko-cam-btn button') ||
                            document.querySelector('#aiko-cam-btn');
                if (btn) {
                    btn.textContent = '⏳';
                    btn.style.opacity = '0.6';
                    setTimeout(() => {
                        btn.textContent = '🖼️';
                        btn.style.opacity = '1';
                    }, 3000);
                }
            }

            function attachCamBtn() {
                const btn = document.querySelector('#aiko-cam-btn button') ||
                            document.querySelector('#aiko-cam-btn');
                if (!btn) { setTimeout(attachCamBtn, 300); return; }
                if (btn._aikoCamAttached) return;
                btn._aikoCamAttached = true;

                btn.addEventListener('click', (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    createModal();
                }, true);
            }

            attachCamBtn();
        }
        """
    )

    info_ok_btn.click(
        lambda: gr.update(visible=False),
        inputs=None,
        outputs=info_overlay,
    )

    msg.submit(
        _submit,
        inputs=[msg, chatbot, user_id_state],
        outputs=[chatbot, tts_text, audio_out, msg],
    )

    send.click(
        _submit,
        inputs=[msg, chatbot, user_id_state],
        outputs=[chatbot, tts_text, audio_out, msg],
    )

    audio_b64.change(
        _voice_from_b64,
        inputs=[audio_b64, chatbot, user_id_state],
        outputs=[chatbot, tts_text, audio_out, audio_b64],
    )

    vision_b64.change(
        _vision_from_b64,
        inputs=[vision_b64, chatbot, user_id_state],
        outputs=[chatbot, tts_text, audio_out, vision_b64],
    )

    # ── Typewriter / lip-sync bridge ─────────────────────────────────────────
    tts_text.change(
        None,
        inputs=[tts_text],
        js="""
        (rawSignal) => {
            const iframe = document.querySelector('#aiko-vrm-frame');
            const sendAvatar = (payload) => {
                if (iframe?.contentWindow) {
                    iframe.contentWindow.postMessage(JSON.stringify(payload), '*');
                }
            };

            if (!rawSignal) return;

            if (rawSignal.startsWith('STATUS:')) {
                sendAvatar({ status: rawSignal.slice('STATUS:'.length) });
                return;
            }

            // Camera auto-open signal — the LLM wants to see
            if (rawSignal === 'OPEN_CAMERA') {
                // Trigger the camera modal via the cam button's click handler
                setTimeout(() => {
                    const btn = document.querySelector('#aiko-cam-btn button') ||
                                document.querySelector('#aiko-cam-btn');
                    if (btn) btn.click();
                }, 600);  // small delay so the chat message renders first
                return;
            }

            if (!rawSignal.startsWith('TYPEWRITE:')) return;

            // Support both 3-part signal (with notes) and 2-part (vision shortcut)
            const rest      = rawSignal.slice('TYPEWRITE:'.length);
            const firstPipe = rest.indexOf('||');

            let emotion, notesPrefix, fullText;

            if (firstPipe === -1) {
                // Malformed — bail
                return;
            }

            emotion = rest.slice(0, firstPipe);
            const afterEmotion = rest.slice(firstPipe + 2);
            const secondPipe   = afterEmotion.indexOf('||');

            if (secondPipe === -1) {
                // 2-part format: TYPEWRITE:<emotion>||<text>  (vision path)
                notesPrefix = '';
                fullText    = afterEmotion;
            } else {
                // 3-part format: TYPEWRITE:<emotion>||<notes>||<text>  (chat path)
                notesPrefix = afterEmotion.slice(0, secondPipe);
                fullText    = afterEmotion.slice(secondPipe + 2);
            }

            const cleanLen = fullText.replace(/[*_#`]/g, '').length;
            const estimatedDuration = Math.max(1.5, Math.min(120, cleanLen * 0.055));

            function scrollChat() {
                const root = document.querySelector('#aiko-chatbot');
                if (!root) return;
                let target = root;
                const candidates = root.querySelectorAll('*');
                for (const el of candidates) {
                    const style = getComputedStyle(el);
                    if ((style.overflowY === 'auto' || style.overflowY === 'scroll') &&
                        el.scrollHeight > el.clientHeight) {
                        target = el;
                        break;
                    }
                }
                requestAnimationFrame(() => {
                    target.scrollTop = target.scrollHeight;
                });
            }

            sendAvatar({
                status: 'speaking',
                speaking: true,
                expression: emotion,
                ttsText: fullText,
                duration: estimatedDuration,
                playNow: true,
            });
            window._aikoLatestTtsText = fullText;
            window._aikoLatestEmotion = emotion;

            if (window._aikoSpeakingFallbackTimer) {
                clearTimeout(window._aikoSpeakingFallbackTimer);
            }
            window._aikoSpeakingFallbackTimer = setTimeout(() => {
                sendAvatar({ speaking: false, status: 'idle' });
            }, (estimatedDuration + 2) * 1000);

            function attachSpeakingBridge() {
                const audioEl = document.querySelector('#aiko-audio audio');
                if (!audioEl) return false;

                if (!audioEl._aikoSpeakingBridge) {
                    audioEl._aikoSpeakingBridge = true;
                    let pauseDebounceTimer = null;
                    const sendSpeaking = (speaking) => {
                        if (!speaking && window._aikoSpeakingFallbackTimer) {
                            clearTimeout(window._aikoSpeakingFallbackTimer);
                            window._aikoSpeakingFallbackTimer = null;
                        }
                        sendAvatar({ speaking, status: speaking ? 'speaking' : 'idle' });
                    };
                    audioEl.addEventListener('play',    () => {
                        clearTimeout(pauseDebounceTimer);
                        sendSpeaking(true);
                    });
                    audioEl.addEventListener('playing', () => {
                        clearTimeout(pauseDebounceTimer);
                        sendSpeaking(true);
                    });
                    audioEl.addEventListener('pause',   () => {
                        clearTimeout(pauseDebounceTimer);
                        pauseDebounceTimer = setTimeout(() => {
                            if (audioEl.paused && !audioEl.ended) sendSpeaking(false);
                        }, 250);
                    });
                    audioEl.addEventListener('ended',   () => {
                        clearTimeout(pauseDebounceTimer);
                        setTimeout(() => sendSpeaking(false), 150);
                    });
                    audioEl.addEventListener('error',   () => {
                        clearTimeout(pauseDebounceTimer);
                        sendSpeaking(false);
                    });
                }

                if (!audioEl.paused && !audioEl.ended) {
                    sendAvatar({ speaking: true, status: 'speaking' });
                }
                return true;
            }
            attachSpeakingBridge();
            let bridgeTries = 0;
            const bridgePoll = setInterval(() => {
                bridgeTries += 1;
                if (attachSpeakingBridge() || bridgeTries > 60) clearInterval(bridgePoll);
            }, 100);

            function getBubbleEl() {
                const allBubbles = document.querySelectorAll('#aiko-chatbot [data-testid="bot"]');
                if (!allBubbles.length) return null;
                const last = allBubbles[allBubbles.length - 1];
                return (
                    last.querySelector('.prose') ||
                    last.querySelector('.message-content') ||
                    last
                );
            }

            function wipeEl(el) {
                while (el.firstChild) el.removeChild(el.firstChild);
                el.textContent = '';
            }

            function isUsableDuration(d) {
                return Number.isFinite(d) && d > 0 && d < 600;
            }

            let blanked       = false;
            let targetEl      = null;
            let typingStarted = false;

            function blankWatcher() {
                const el = getBubbleEl();
                if (!el) { requestAnimationFrame(blankWatcher); return; }

                if (el.textContent.trim() !== '') {
                    targetEl = el;
                    wipeEl(el);

                    if (notesPrefix && notesPrefix.trim()) {
                        const notesDiv = document.createElement('div');
                        notesDiv.className = 'aiko-tool-notes';
                        notesDiv.style.cssText = 'opacity:0.65;font-size:0.78rem;margin-bottom:6px;color:rgba(180,160,255,0.75);';
                        notesPrefix.trim().split('\\n').forEach(line => {
                            if (!line.trim()) return;
                            const row = document.createElement('div');
                            row.textContent = line.trim();
                            notesDiv.appendChild(row);
                        });
                        el.appendChild(notesDiv);
                    }

                    const responseSpan = document.createElement('span');
                    el.appendChild(responseSpan);
                    targetEl._responseSpan = responseSpan;
                    blanked = true;

                    const obs = new MutationObserver(() => {
                        if (!typingStarted) {
                            const hasNotes = el.querySelector('.aiko-tool-notes');
                            const hasSpan  = el.querySelector('span[data-aiko-response]');
                            if (!hasNotes && notesPrefix && notesPrefix.trim()) {
                                wipeEl(el);
                                const nd = document.createElement('div');
                                nd.className = 'aiko-tool-notes';
                                nd.style.cssText = 'opacity:0.65;font-size:0.78rem;margin-bottom:6px;color:rgba(180,160,255,0.75);';
                                notesPrefix.trim().split('\\n').forEach(line => {
                                    if (!line.trim()) return;
                                    const row = document.createElement('div');
                                    row.textContent = line.trim();
                                    nd.appendChild(row);
                                });
                                el.appendChild(nd);
                                const rs = document.createElement('span');
                                rs.setAttribute('data-aiko-response', '1');
                                el.appendChild(rs);
                                targetEl._responseSpan = rs;
                            }
                        }
                    });
                    if (responseSpan) responseSpan.setAttribute('data-aiko-response', '1');
                    obs.observe(el, { childList: true, subtree: false });
                    window._aikoBlankObs = obs;

                    scrollChat();
                } else {
                    requestAnimationFrame(blankWatcher);
                }
            }
            blankWatcher();

            function startTypewriter() {
                if (!blanked || !targetEl) { setTimeout(startTypewriter, 200); return; }

                typingStarted = true;

                if (window._aikoBlankObs) {
                    window._aikoBlankObs.disconnect();
                    window._aikoBlankObs = null;
                }

                const totalChars = fullText.length;
                const cleanLen   = fullText.replace(/[*_#`]/g, '').length;
                let audioDur = Math.max(3, cleanLen * 0.075);
                let totalMs  = audioDur * 1000;
                let perChar  = totalMs / Math.max(1, totalChars);

                const audioEl = document.querySelector('#aiko-audio audio');
                if (audioEl && isUsableDuration(audioEl.duration)) {
                    audioDur = audioEl.duration;
                    totalMs  = audioDur * 1000;
                    perChar  = totalMs / Math.max(1, totalChars);
                }

                let i         = 0;
                let startTime = performance.now();
                const typeTarget = targetEl._responseSpan || targetEl;

                function tick() {
                    if (i >= totalChars) {
                        typeTarget.textContent = fullText;
                        scrollChat();
                        return;
                    }

                    let shouldBe = i;
                    const a = document.querySelector('#aiko-audio audio');

                    if (a && isUsableDuration(a.duration) && (!a.paused || a.currentTime > 0)) {
                        const progress = Math.min(1, a.currentTime / a.duration);
                        shouldBe = Math.floor(progress * totalChars);
                        if (a.ended) shouldBe = totalChars;
                    } else {
                        const elapsed = performance.now() - startTime;
                        shouldBe = Math.floor((elapsed / totalMs) * totalChars);
                    }

                    if (shouldBe > i) {
                        i = Math.min(Math.max(shouldBe, 0), totalChars);
                        typeTarget.textContent = i < totalChars
                            ? fullText.slice(0, i) + '▋'
                            : fullText;
                        scrollChat();
                    }
                    requestAnimationFrame(tick);
                }

                tick();
            }

            startTypewriter();
        }
        """
    )

    demo.queue()

# ─────────────────────────────────────────────
# LAUNCH
# ─────────────────────────────────────────────
allowed_paths = [
    str(Path("/tmp/aiko_tts")),
    str(VRM_PATH.parent),
]

demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    ssr_mode=False,
    share=False,
    allowed_paths=allowed_paths,
    css=APP_CSS,
)