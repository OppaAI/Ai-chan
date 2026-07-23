/**
 * webui.js
 * Real-time chat UI with WebSocket bridge to Aiko backend.
 * Handles voice I/O (mic capture via pcm-worklet + VAD, TTS playback with mouth sync),
 * WebSocket message routing (chat, token streaming, vitals, expressions, visemes),
 * initialization status tracking (step progress), and mic/text input modes.
 *
 * Core flows:
 *   - mic capture: AudioWorklet → VAD frame → server (if speech detected)
 *   - TTS playback: binary WAV frames → decode → analyser RMS → lip-sync blendshapes
 *   - chat: text input or voice transcription → user_input message → token streaming
 *   - gestures: server sends expression/viseme/pose → window.aikoSetX() → vrm.js
 */

// ── DOM refs ──────────────────────────────────────────────────────────────
const initPanel = document.getElementById('init-panel');
const chatPanel = document.getElementById('chat-panel');
const toolStatus = document.getElementById('tool-status');
const content = document.getElementById('content');
const allOnline = document.getElementById('all-online');
const input = document.getElementById('user-input');
const micBtn = document.getElementById('mic-btn');
const sendBtn = document.getElementById('send-btn');
const voiceSt = document.getElementById('voice-status');
const clock = document.getElementById('panel-clock');
const wsDot = document.getElementById('ws-dot');
const wsLabel = document.getElementById('ws-label');
const vadDot = document.getElementById('vad-dot');
const vadStatus = document.getElementById('vad-status');

const bootProgressFill = document.getElementById('boot-progress-fill');
const bootProgressMsg = document.getElementById('boot-progress-msg');

const vTok = document.getElementById('v-tok');
const vToks = document.getElementById('v-toks');
const vRam = document.getElementById('v-ram');
const vUp = document.getElementById('v-up');
const vMode = document.getElementById('v-mode');

const AUTO_MIC = false;
let autoListenRequested = false;

// ── viewport height fix (mobile browser toolbar collapse/expand) ─────────
// dvh units are unreliable on Firefox Android; visualViewport tracks the
// true visible area after the toolbar/keyboard resizes it.
function setAppHeight() {
  const h = window.visualViewport ? window.visualViewport.height : window.innerHeight;
  document.documentElement.style.setProperty('--app-height', `${h}px`);
}
setAppHeight();
if (window.visualViewport) {
  window.visualViewport.addEventListener('resize', setAppHeight);
  window.visualViewport.addEventListener('scroll', setAppHeight);
} else {
  window.addEventListener('resize', setAppHeight);
}
window.addEventListener('orientationchange', () => setTimeout(setAppHeight, 100));

// ── clock ─────────────────────────────────────────────────────────────────
function tickClock() {
  const now = new Date();
  clock.textContent = now.toLocaleString('en-CA', {
    month: 'short', day: '2-digit', year: 'numeric',
    hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: true,
  });
}
tickClock();
setInterval(tickClock, 1000);

// ── VAD init ──────────────────────────────────────────────────────────────
// Browser runs a lightweight energy-RMS gate (vad.js) only — no model to load.
// Backend Silero VAD is the authoritative speech/silence check.
initVAD().then((status) => {
  vadDot.className = 'dot on';
  vadStatus.textContent = 'vad ready';
  vadStatus.className = 'ready';
}).catch(err => {
  vadStatus.textContent = 'vad failed';
  console.error('[vad] init error:', err);
});

// ── step / init tracking ──────────────────────────────────────────────────
let bootDone = 0, bootTotal = 0;
let bootKeys = {};

function handleStep(msg) {
  const key = msg.key, state = msg.state;
  const label = msg.label || key;
  if (state === 'loading') {
    bootProgressMsg.textContent = label;
  } else if (['done', 'skip', 'error'].includes(state)) {
    if (!bootKeys[key]) {
      bootKeys[key] = true;
      bootTotal++;
    }
    bootDone++;
  }
  const total = bootTotal || 1;
  const pct = Math.min(100, Math.round(100 * bootDone / total));
  bootProgressFill.style.width = pct + '%';
  if (bootTotal > 0 && bootDone >= bootTotal) allOnline.classList.add('show');
}

// ── phase switch ──────────────────────────────────────────────────────────
let chatPhaseActive = false;

function switchToChat() {
  chatPhaseActive = true;
  initPanel.classList.add('hidden');
  chatPanel.classList.add('show');
  input.focus();
}

// ── chat rendering ────────────────────────────────────────────────────────
let streamEl = null;

function addMessage(sender, text) {
  flushStream();
  const div = document.createElement('div');
  if (sender === 'you') {
    div.className = 'msg msg-you';
    div.innerHTML = `<span class="msg-prefix">${esc(window.currentUsername || 'You')}: </span>${esc(text)}`;
  } else if (sender === 'aiko') {
    div.className = 'msg msg-aiko';
    div.innerHTML = `<span class="msg-prefix">Aiko: </span>${esc(text)}`;
  } else {
    div.className = 'msg msg-sys';
    div.textContent = `  ◈  ${text}`;
  }
  chatPanel.insertBefore(div, toolStatus);
  scrollBottom();
}

function appendToken(text) {
  if (!streamEl) {
    const div = document.createElement('div');
    div.className = 'msg msg-aiko';
    streamEl = document.createElement('span');
    streamEl.className = 'cursor';
    div.innerHTML = '<span class="msg-prefix">Aiko: </span>';
    div.appendChild(streamEl);
    chatPanel.insertBefore(div, toolStatus);
  }
  streamEl.textContent += text;
  scrollBottom();
}

function flushStream() {
  if (streamEl) { streamEl.classList.remove('cursor'); streamEl = null; }
  toolStatus.textContent = '';
}

function esc(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function scrollBottom() { content.scrollTop = content.scrollHeight; }

// ── vitals ────────────────────────────────────────────────────────────────
function applyVitals(v) {
  vTok.textContent = `${(v.tokens || 0).toLocaleString()} tok`;
  vToks.textContent = v.tok_s > 0 ? `${v.tok_s} t/s` : '— t/s';
  vRam.textContent = `RAM ${v.ram || '—'}`;
  vUp.textContent = `↑ ${v.uptime || '—'}`;
  vMode.textContent = (v.asr ? '🎤 ASR' : '⌨ TXT') + '  ' + (v.tts ? '🔊 TTS' : '🔇 TTS');

  if (AUTO_MIC && wsReady() && !v.asr && !autoListenRequested) {
    autoListenRequested = true;
    ws.send(JSON.stringify({ type: 'user_input', text: '/listen' }));
  }
}

// ── voice status ──────────────────────────────────────────────────────────
const VOICE_LABELS = {
  waiting: '⏸  waiting for Aiko…',
  listening: '🎤  listening…',
  transcribing: '⚙  transcribing…',
  idle: '',
};
function applyVoice(status) {
  voiceSt.textContent = VOICE_LABELS[status] ?? '';
  voiceSt.className = status === 'idle' ? '' : status;
}

// ── TTS playback (binary WAV frames from server) ──────────────────────────
let ttsContext = null;
let ttsQueue = [];
let ttsPlaying = false;
let ttsAnalyser = null;
let ttsAnalyserData = null;
let ttsAnalyserConnected = false;
let ttsMouthLoop = false;
let ttsMouthLevel = 0;

function getTtsContext() {
  if (!ttsContext) ttsContext = new AudioContext();
  return ttsContext;
}

function getTtsAnalyser() {
  const ctx = getTtsContext();
  if (!ttsAnalyser) {
    ttsAnalyser = ctx.createAnalyser();
    ttsAnalyser.fftSize = 1024;
    ttsAnalyser.smoothingTimeConstant = 0.35;
    ttsAnalyserData = new Float32Array(ttsAnalyser.fftSize);
  }
  if (!ttsAnalyserConnected) {
    ttsAnalyser.connect(ctx.destination);
    ttsAnalyserConnected = true;
  }
  return ttsAnalyser;
}

function startMouthAnalyserLoop() {
  if (ttsMouthLoop) return;
  ttsMouthLoop = true;
  const tick = () => {
    if (!ttsMouthLoop) return;
    let target = 0;
    if (ttsPlaying && ttsAnalyser && ttsAnalyserData) {
      ttsAnalyser.getFloatTimeDomainData(ttsAnalyserData);
      let sum = 0;
      for (let i = 0; i < ttsAnalyserData.length; i++) {
        const v = ttsAnalyserData[i];
        sum += v * v;
      }
      const rms = Math.sqrt(sum / ttsAnalyserData.length);
      target = Math.max(0, Math.min(1, (rms - 0.012) * 9.5));
    }

    // Fast attack, slower release. Pauses from punctuation naturally drop
    // the analyser RMS, closing the mouth instead of racing ahead of audio.
    const coeff = target > ttsMouthLevel ? 0.65 : 0.28;
    ttsMouthLevel += (target - ttsMouthLevel) * coeff;
    if (window.aikoSetMouthOpen) window.aikoSetMouthOpen(ttsMouthLevel);

    if (!ttsPlaying && ttsMouthLevel < 0.01) {
      ttsMouthLoop = false;
      ttsMouthLevel = 0;
      if (window.aikoSetMouthOpen) window.aikoSetMouthOpen(0);
      return;
    }
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

async function enqueueTtsAudio(arrayBuffer) {
  ttsQueue.push(arrayBuffer);
  if (!ttsPlaying) playNextTts();
}

let ttsCurrentSource = null;

async function playNextTts() {
  const buf = ttsQueue.shift();
  if (!buf) { ttsPlaying = false; window.aikoIsSpeaking = false; return; }
  ttsPlaying = true;
  window.aikoIsSpeaking = true;          // exposed for vad.js to read
  try {
    const ctx = getTtsContext();
    const audioBuffer = await ctx.decodeAudioData(buf.slice(0));
    const analyser = getTtsAnalyser();
    const src = ctx.createBufferSource();
    src.buffer = audioBuffer;
    src.connect(analyser);
    src.onended = playNextTts;
    ttsCurrentSource = src;
    src.start();
    startMouthAnalyserLoop();
  } catch (err) {
    console.error('[tts] decode/play failed:', err);
    playNextTts();
  }
}

function stopTtsPlayback() {
  ttsQueue = [];
  if (ttsCurrentSource) {
    try { ttsCurrentSource.onended = null; ttsCurrentSource.stop(); } catch (e) {}
    ttsCurrentSource = null;
  }
  ttsPlaying = false;
  window.aikoIsSpeaking = false;
}
window.stopTtsPlayback = stopTtsPlayback;

// ── mic capture ───────────────────────────────────────────────────────────
// Opened/closed by server mic.start / mic.stop messages.
// Frames go through processVADFrame() in vad.js. By default browser VAD gates
// network audio; WEBUI_BROWSER_VAD_GATE=0 asks the browser to stream raw PCM
// for diagnostics so server-side VAD can be tested.
let micStream = null;
let micContext = null;
let micSource = null;
let micWorklet = null;
let micFirstFrameSeen = false;
let micStreamingEnabled = false;
let browserVadGate = true;
let micCommandSeq = 0;
let micSecureContextWarned = false;

async function startMic() {
  if (micContext) return true;
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    console.error('[mic] microphone requires localhost or HTTPS');
    if (!micSecureContextWarned) {
      micSecureContextWarned = true;
      const uiPort = location.port || '8787';
      const localUrl = 'http://localhost:' + uiPort + '/';
      const secureUrl = 'https://' + location.hostname + ':' + uiPort + '/';
      addMessage('sys', 'Microphone blocked — browsers only allow mic access on localhost or HTTPS. Open ' + localUrl + ' on this machine, or restart with WEBUI_HTTPS=1 and use ' + secureUrl + '.');
    }
    micBtn.classList.remove('on');
    return false;
  }
  try {
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: false },
    });
    micContext = new AudioContext({ sampleRate: 16000 });
    // Resume if suspended (happens when AudioContext is created outside a user
    // gesture, e.g. from a WebSocket message handler on a reconnected session).
    if (micContext.state === 'suspended') {
      await micContext.resume();
      console.log('[mic] AudioContext was suspended — resumed');
    }
    micSource = micContext.createMediaStreamSource(micStream);

    // ── Serialised VAD processing queue ───────────────────────────────────
    // Energy VAD is synchronous math, but keeping this queue ensures frames
    // are always sent to the server in strict arrival order.
    let _vadQueue = Promise.resolve();
    function pushVADFrame(frame) {
      _vadQueue = _vadQueue.then(() => processVADFrame(frame, ws, browserVadGate)).catch(e => console.error('[mic] VAD error:', e));
    }

    // ── AudioWorklet (preferred) ──────────────────────────────────────────
    // Falls back to ScriptProcessorNode if the worklet module cannot be
    // loaded (e.g. MIME-type issues with the static file server).
    let awok = false;
    try {
      await micContext.audioWorklet.addModule('./pcm-worklet.js');
      micWorklet = new AudioWorkletNode(micContext, 'pcm-capture-processor');
      micFirstFrameSeen = false;
      micWorklet.port.onmessage = (e) => {
        if (!micFirstFrameSeen) {
          micFirstFrameSeen = true;
          console.log('[mic] AudioWorklet is sending PCM frames');
        }
        if (wsReady() && micStreamingEnabled) {
          pushVADFrame(new Float32Array(e.data));
        }
      };
      micSource.connect(micWorklet);
      micWorklet.connect(micContext.destination);  // required so the audio graph processes real samples
      awok = true;
      console.log('[mic] using AudioWorklet capture');
    } catch (awErr) {
      console.warn('[mic] AudioWorklet failed, falling back to ScriptProcessorNode:', awErr);
    }

    // ── ScriptProcessorNode fallback ──────────────────────────────────────
    if (!awok) {
      const bufSize = 2048;  // 2048 samples = 128 ms at 16 kHz — supported everywhere
      const frameSamples = 512;
      let _spBuf = new Float32Array(0);
      const spNode = micContext.createScriptProcessor(bufSize, 1, 1);
      spNode.onaudioprocess = (e) => {
        if (!wsReady() || !micStreamingEnabled) return;
        const input = e.inputBuffer.getChannelData(0);
        // Accumulate until we have full 512-sample frames
        let combined = new Float32Array(_spBuf.length + input.length);
        combined.set(_spBuf);
        combined.set(input, _spBuf.length);
        _spBuf = combined;
        while (_spBuf.length >= frameSamples) {
          const frame = _spBuf.slice(0, frameSamples);
          _spBuf = _spBuf.slice(frameSamples);
          pushVADFrame(frame);
        }
      };
      micSource.connect(spNode);
      spNode.connect(micContext.destination);  // required by spec (silent output)
      micWorklet = spNode;  // reuse micWorklet ref for cleanup
      console.log('[mic] using ScriptProcessorNode capture');
    }

    vadDot.className = 'dot on';
    vadStatus.textContent = 'mic ready';
    vadStatus.className = 'ready';
    micBtn.classList.add('on');
    return true;
  } catch (err) {
    console.error('[mic] getUserMedia/AudioWorklet failed:', err);
    addMessage('sys', 'Microphone access failed — check browser permissions.');
    micBtn.classList.remove('on');
    return false;
  }
}

function stopMic() {
  micCommandSeq++;
  micStreamingEnabled = false;
  if (window.resetVADState) window.resetVADState();
  if (micWorklet) {
    if (micWorklet.port) micWorklet.port.onmessage = null;
    micWorklet.disconnect();
    micWorklet = null;
  }
  if (micSource) { micSource.disconnect(); micSource = null; }
  if (micContext) { micContext.close(); micContext = null; }
  if (micStream) { micStream.getTracks().forEach(t => t.stop()); micStream = null; }
  vadDot.className = 'dot on';
  vadStatus.textContent = 'vad ready';
  vadStatus.className = 'ready';
  micBtn.classList.remove('on');
}

// ── text input ────────────────────────────────────────────────────────────
function submitInput() {
  const text = input.value.trim();
  if (!text || !wsReady()) return;
  ws.send(JSON.stringify({ type: 'user_input', text }));
  input.value = '';
}

input.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submitInput(); }
});
sendBtn.addEventListener('click', submitInput);
micBtn.addEventListener('click', async () => {
  if (!wsReady()) {
    addMessage('sys', 'WebSocket bridge is offline. Cannot toggle voice mode.');
    return;
  }

  const asrEnabled = vMode.textContent.includes('ASR');

  if (micContext) {
    // Mic is already open: close it and disable ASR so the backend stops
    // waiting for browser voice frames.
    stopMic();
    if (asrEnabled) ws.send(JSON.stringify({ type: 'user_input', text: '/listen' }));
  } else {
    // Mic is closed: open it first. If ASR was already on at startup, do
    // not send /listen because that would toggle ASR off right as the
    // browser starts capturing.
    const ok = await startMic();
    if (!ok) return;
    if (!asrEnabled) ws.send(JSON.stringify({ type: 'user_input', text: '/listen' }));
  }
  input.focus();
});

// ── WebSocket ─────────────────────────────────────────────────────────────
let ws = null;

function wsReady() { return ws && ws.readyState === WebSocket.OPEN; }

function websocketURL() {
  const params = new URLSearchParams(location.search);
  const wsHost = params.get("ws_host") || location.hostname;
  const wsPortParam = params.get("ws");
  const protoOverride = (params.get("ws_proto") || "").toLowerCase();
  const wsProto = protoOverride === "ws" || protoOverride === "wss"
    ? protoOverride + ":"
    : location.protocol === "https:" ? "wss:" : "ws:";

  // If accessed via Tailscale (.ts.net), use path-based routing (no port)
  if (wsHost.endsWith(".ts.net")) {
    return wsProto + "//" + wsHost + "/ws";
  }

  // If custom ws port is specified explicitly, connect there
  if (wsPortParam) {
    return wsProto + "//" + wsHost + ":" + wsPortParam + "/";
  }

  // Default to path-based routing on same port as page
  const portPart = location.port ? ":" + location.port : "";
  return wsProto + "//" + wsHost + portPart + "/ws";
}

function connectWS() {
  const wsUrl = websocketURL();
  ws = new WebSocket(wsUrl);
  ws.binaryType = 'arraybuffer';

  ws.onopen = () => {
    wsDot.className = 'dot on';
    wsLabel.textContent = 'ws connected';
    if (AUTO_MIC) startMic();
  };

  ws.onmessage = (e) => {
    if (e.data instanceof ArrayBuffer) {
      enqueueTtsAudio(e.data);
      return;
    }
    let msg;
    try { msg = JSON.parse(e.data); } catch (_) { return; }

    // Only real conversation content should force an early switch out of the
    // splash screen (this fallback exists for a browser reconnecting mid-chat
    // after boot already completed, since no further 'phase' broadcast will
    // ever arrive for that new connection). 'vitals' must NOT be in this list:
    // spin_loop() broadcasts vitals every ~250ms starting the instant the
    // browser connects — well before AikoWakeup().boot() has actually finished
    // loading subsystems — which was hiding the splash almost immediately
    // while the real multi-minute boot silently continued underneath.
    if (!chatPhaseActive && ['chat', 'token'].includes(msg.type)) {
      switchToChat();
    }

    switch (msg.type) {
      case 'step': handleStep(msg); break;
      case 'phase': if (msg.value === 'chat') switchToChat(); break;
      case 'chat': addMessage(msg.sender, msg.text); break;
      case 'token': appendToken(msg.text); break;
      case 'commit': flushStream(); break;
      case 'tool': toolStatus.textContent = msg.status ? `  ⚙  ${msg.status}` : ''; break;
      case 'vitals': applyVitals(msg); break;
      case 'voice': applyVoice(msg.status); break;
      case 'mic':
        if (msg.action === 'start') {
          const seq = ++micCommandSeq;
          browserVadGate = msg.browser_vad_gate !== false;
          startMic().then((ok) => {
            if (!ok || seq !== micCommandSeq) return;
            if (window.resetVADState) window.resetVADState();
            micStreamingEnabled = true;
            vadDot.className = 'dot vad';
            vadStatus.textContent = browserVadGate ? 'vad active' : 'raw mic';
            vadStatus.className = 'active';
          });
        } else if (msg.action === 'stop') {
          micCommandSeq++;
          micStreamingEnabled = false;
          if (window.resetVADState) window.resetVADState();
          vadDot.className = 'dot on';
          vadStatus.textContent = 'mic ready';
          vadStatus.className = 'ready';
        }
        break;
      case 'expression': if (window.aikoSetExpression) window.aikoSetExpression(msg.name, msg.intensity ?? 1.0); break;
      case 'viseme': if (window.aikoSetViseme) window.aikoSetViseme(msg.viseme, msg.weight ?? 1.0); break;
      case 'pose': if (window.aikoSetPose) window.aikoSetPose(msg.name, msg.active); break;
    }
  };

  ws.onclose = () => {
    wsDot.className = 'dot';
    wsLabel.textContent = 'ws offline';
    stopMic();
    if (wsUrl.startsWith("wss:")) {
      toolStatus.textContent = "  ws offline: open " + wsUrl.replace("wss:", "https:") + " once to accept the WSS certificate";
    } else {
      toolStatus.textContent = "  ws offline: " + wsUrl;
    }
    setTimeout(connectWS, 3000);
  };
  ws.onerror = () => {
    console.error('[ws] connection failed:', wsUrl);
    ws.close();
  };
}

// ── OAuth Login ──────────────────────────────────────────────────────────
const authOverlay = document.getElementById('auth-overlay');
const authStatus = document.getElementById('auth-status');

async function checkAuth() {
  try {
    const res = await fetch('/api/auth/me', { credentials: 'include' });
    if (res.ok) {
      let data = {};
      try { data = await res.json(); } catch (_) { /* no body / non-JSON */ }
      // Displayed chat-label username (falls back to 'you' if session has none).
      window.currentUsername = data.username || 'You';
      const aiNameEl = document.getElementById('vrm-ai-name');
      if (aiNameEl) aiNameEl.textContent = data.ai_name || 'Aiko';
      const userNameEl = document.getElementById('vrm-user-name');
      if (userNameEl) userNameEl.textContent = window.currentUsername;
      hideAuthOverlay();
      // If the backend reports the user hasn't accepted the current terms
      // version, gate on the terms modal before opening the WebSocket.
      if (data.accepted_terms === false) {
        showTermsOverlay();
      } else {
        connectWS();
      }
      return true;
    }
  } catch (_) { }
  return false;
}

function hideAuthOverlay() {
  authOverlay.classList.add('hidden');
  setTimeout(() => authOverlay.style.display = 'none', 600);
}

function setAuthStatus(msg) {
  authStatus.textContent = msg;
}

function loginGitHub() {
  window.location.href = '/auth/github/login';
}

function loginPatreon() {
  window.location.href = '/auth/patreon/login';
}

// ── Terms / guidelines modal ───────────────────────────────────────────────
// FIX: previously nothing in this file referenced these elements at all, so
// the checkbox never enabled the Continue button and clicking it did nothing.
const termsOverlay = document.getElementById('terms-overlay');
const termsCheckbox = document.getElementById('terms-checkbox');
const termsContinueBtn = document.getElementById('terms-continue');

function showTermsOverlay() {
  termsOverlay.style.display = 'flex';
  termsOverlay.classList.remove('hidden');
}

function hideTermsOverlay() {
  termsOverlay.classList.add('hidden');
  setTimeout(() => termsOverlay.style.display = 'none', 600);
}

termsCheckbox.addEventListener('change', () => {
  termsContinueBtn.disabled = !termsCheckbox.checked;
});

termsContinueBtn.addEventListener('click', async () => {
  if (!termsCheckbox.checked) return;
  termsContinueBtn.disabled = true;
  try {
    await fetch('/api/auth/accept-terms', {
      method: 'POST',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ accepted: true }),
    });  }
    catch (err) {
    console.error('[terms] failed to record acceptance:', err);
  }
  hideTermsOverlay();
  connectWS();
});

// Load config and check auth
fetch('/api/auth/config')
  .then(r => {
    if (!r.ok) throw new Error('Failed to load auth config');
    return r.json();
  })
  .then(cfg => {
    window.OAUTH_CONFIG = cfg;
    return checkAuth();
  })
  .then(authenticated => {
    // checkAuth() already hides the login overlay and either opens the
    // terms modal or connects the WebSocket when authenticated — only the
    // "not authenticated" branch needs handling here.
    if (!authenticated) {
      authOverlay.classList.remove('hidden');
      setAuthStatus('Authentication required. Please log in.');
    }
  })
  .catch(err => {
    console.error('[auth] initialization error:', err);
    authOverlay.classList.remove('hidden');
    setAuthStatus('Failed to load authentication system.');
  });
