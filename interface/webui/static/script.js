/**
 * script.js
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
 *
 *      sets AIKO_TTS_STARTED_AT on TTS start + AIKO_BARGE_ECHO_GUARD_MS on mic start
 *      so vad.js can ignore self-echo barge for BARGE_IN_ECHO_GUARD_MS after TTS begins.
 *
 * UI extras:
 *   - theme switch (style.css light ⇄ style-dark.css dark), persisted in localStorage
 *   - shoujo-mode flourishes: chat bubbles, typing indicator, emotion badge,
 *     floating emotion particles (hidden by style-dark.css in dark mode)
 */

// ── DOM refs ──────────────────────────────────────────────────────────────
const initPanel = document.getElementById('init-panel');
const chatPanel = document.getElementById('chat-panel');
const toolStatus = document.getElementById('tool-status');
const content = document.getElementById('content');
const allOnline = document.getElementById('all-online');
const input = document.getElementById('user-input');
const cameraBtn = document.getElementById('camera-btn');
const micBtn = document.getElementById('mic-btn');
const sendBtn = document.getElementById('send-btn');
const voiceSt = document.getElementById('voice-status');
const clock = document.getElementById('panel-clock');
const wsDot = document.getElementById('ws-dot');
const wsLabel = document.getElementById('ws-label');
const vadDot = document.getElementById('vad-dot');
const vadStatus = document.getElementById('vad-status');

const emotionBadge = document.getElementById('emotion-badge');
const emotionEmoji = document.getElementById('emotion-emoji');
const emotionText = document.getElementById('emotion-text');

const bootProgressFill = document.getElementById('boot-progress-fill');
const bootProgressMsg = document.getElementById('boot-progress-msg');

const vTok = document.getElementById('v-tok');
const vToks = document.getElementById('v-toks');
const vRam = document.getElementById('v-ram');
const vUp = document.getElementById('v-up');
const vMode = document.getElementById('v-mode');

const AUTO_MIC = false;
let autoListenRequested = false;

// Default barge-in off until server mic.start sets it (S0).
window.AIKO_BARGE_IN_ENABLED = false;

// ── theme switch (style.css light ⇄ style-dark.css dark) ──────────────────
// index.html's inline <head> script has already applied the saved choice to
// the <link disabled> flags to avoid a flash; here we just keep the toggle
// button icon in sync and handle clicks.
const THEME_KEY = 'aiko-theme';
const themeToggleBtn = document.getElementById('theme-toggle');

function applyTheme(theme) {
  const dark = theme === 'dark';
  document.documentElement.dataset.theme = dark ? 'dark' : 'light';
  const lightLink = document.getElementById('theme-style-light');
  const darkLink = document.getElementById('theme-style-dark');
  if (lightLink && darkLink) {
    lightLink.disabled = dark;
    darkLink.disabled = !dark;
  }
  if (themeToggleBtn) {
    const sunIcon = themeToggleBtn.querySelector('.icon-sun');
    const moonIcon = themeToggleBtn.querySelector('.icon-moon');
    if (sunIcon) sunIcon.style.display = dark ? '' : 'none';
    if (moonIcon) moonIcon.style.display = dark ? 'none' : '';
    themeToggleBtn.title = dark ? 'Switch to light mode' : 'Switch to dark mode';
  }
}

function initTheme() {
  let saved = null;
  try { saved = localStorage.getItem(THEME_KEY); } catch (_) { /* storage blocked */ }
  applyTheme(saved === 'dark' ? 'dark' : 'light');
}

function toggleTheme() {
  const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
  try { localStorage.setItem(THEME_KEY, next); } catch (_) { /* storage blocked */ }
  applyTheme(next);
}

initTheme();
if (themeToggleBtn) themeToggleBtn.addEventListener('click', toggleTheme);

// ── viewport height fix (mobile browser toolbar collapse/expand) ─────────
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
let streamDiv = null;
let streamRawText = '';
let streamExprApplied = null;  // expression name applied once per stream turn
let sourcesRow = null;
let filesRow = null;
let typingIndicator = null;

// ── Emotion Particles (shoujo mode; hidden by style-dark.css) ────────────
const EMOJI_PARTICLES = {
  happy:    ['🌸', '✨', '💗', '💖', '🌟'],
  angry:    ['💢', '🔥', '⚡', '💥'],
  sorrow:   ['💧', '😢', '💔', '🌧️'],
  surprised:['❗', '✨', '💫', '🌟'],
  fun:      ['🎉', '🎈', '✨', '🌈', '🎀'],
  neutral:  ['💭', '✦', '·'],
};

function spawnEmotionParticles(exprName) {
  const particles = EMOJI_PARTICLES[exprName] || EMOJI_PARTICLES.neutral;
  const count = Math.min(6, particles.length);
  for (let i = 0; i < count; i++) {
    const el = document.createElement('div');
    el.textContent = particles[i % particles.length];
    el.style.cssText = `
      position: fixed;
      pointer-events: none;
      font-size: ${14 + Math.random() * 10}px;
      z-index: 100;
      left: ${20 + Math.random() * 60}%;
      top: ${20 + Math.random() * 40}%;
      opacity: 0;
      animation: emotionParticle 1.5s ease-out forwards;
      animation-delay: ${i * 0.1}s;
    `;
    document.body.appendChild(el);
    setTimeout(() => el.remove(), 2000);
  }
}

// Inject keyframes if not present
if (!document.getElementById('shoujo-animations')) {
  const style = document.createElement('style');
  style.id = 'shoujo-animations';
  style.textContent = `
    @keyframes emotionParticle {
      0% { opacity: 0; transform: translateY(0) scale(0.5); }
      30% { opacity: 0.8; transform: translateY(-20px) scale(1.1); }
      100% { opacity: 0; transform: translateY(-60px) scale(0.8); }
    }
  `;
  document.head.appendChild(style);
}

function updateEmotionBadge(exprName) {
  if (!emotionBadge) return;
  const emojiMap = {
    happy: '🌸', angry: '💢', sorrow: '💧', surprised: '✨', fun: '🎀', neutral: '💭'
  };
  emotionBadge.style.display = 'flex';
  emotionEmoji.textContent = emojiMap[exprName] || '💗';
  emotionText.textContent = exprName;
}

// ── Typing Indicator ──────────────────────────────────────────────────────
function showTypingIndicator() {
  if (typingIndicator) return;
  const row = document.createElement('div');
  row.className = 'typing-row';
  row.id = 'typing-indicator';

  const avatar = document.createElement('div');
  avatar.className = 'msg-avatar aiko';
  avatar.textContent = '🤖';

  const bubble = document.createElement('div');
  bubble.className = 'typing-bubble';
  for (let i = 0; i < 3; i++) {
    const dot = document.createElement('div');
    dot.className = 'typing-dot';
    bubble.appendChild(dot);
  }

  row.appendChild(avatar);
  row.appendChild(bubble);
  chatPanel.insertBefore(row, toolStatus);
  typingIndicator = row;
  scrollBottom();
}

function hideTypingIndicator() {
  if (typingIndicator) {
    typingIndicator.remove();
    typingIndicator = null;
  }
}

function ensureAuxRow(kind) {
  let row = kind === 'sources' ? sourcesRow : filesRow;
  if (row && row.parentNode) return row;
  row = document.createElement('div');
  row.className = kind === 'sources' ? 'sources-row' : 'files-row';
  chatPanel.insertBefore(row, toolStatus);
  if (kind === 'sources') sourcesRow = row;
  else filesRow = row;
  return row;
}

function clearAuxRows() {
  if (sourcesRow) { sourcesRow.remove(); sourcesRow = null; }
  if (filesRow) { filesRow.remove(); filesRow = null; }
}

function faviconUrl(domain) {
  if (!domain) return '';
  return 'https://www.google.com/s2/favicons?domain=' + encodeURIComponent(domain) + '&sz=32';
}

function renderSources(items) {
  if (!Array.isArray(items) || !items.length) return;
  const row = ensureAuxRow('sources');
  row.innerHTML = '';
  const label = document.createElement('span');
  label.className = 'sources-label';
  label.textContent = 'sources';
  row.appendChild(label);
  for (const it of items.slice(0, 8)) {
    const a = document.createElement('a');
    a.className = 'source-chip';
    let href = it.url || '#';
    try {
      const u = new URL(href);
      if (u.protocol !== 'http:' && u.protocol !== 'https:') {
        href = '#';
      }
    } catch {
      href = '#';
    }
    a.href = href;
    a.target = '_blank';
    a.rel = 'noopener noreferrer';
    a.title = it.title || it.url || '';
    const img = document.createElement('img');
    img.className = 'source-favicon';
    img.alt = '';
    img.src = faviconUrl(it.domain || '');
    img.referrerPolicy = 'no-referrer';
    const name = document.createElement('span');
    name.textContent = it.domain || (it.title || 'link').slice(0, 24);
    a.appendChild(img);
    a.appendChild(name);
    row.appendChild(a);
  }
  scrollBottom();
}

function renderFiles(items) {
  if (!Array.isArray(items) || !items.length) return;
  const row = ensureAuxRow('files');
  row.innerHTML = '';
  const label = document.createElement('span');
  label.className = 'files-label';
  label.textContent = 'files';
  row.appendChild(label);
  for (const it of items) {
    const chip = document.createElement('button');
    chip.className = 'file-chip';
    const path = it.path || '';
    const name = it.label || path || 'file';
    chip.title = path || name;
    chip.textContent = name;
    chip.disabled = !path;
    chip.addEventListener('click', function () {
      if (navigator.clipboard && path) {
        navigator.clipboard.writeText(path).then(function () {
          chip.classList.add('copied');
          setTimeout(function () { chip.classList.remove('copied'); }, 900);
        }).catch(function () {});
      }
    });
    row.appendChild(chip);
  }
  scrollBottom();
}


const EMOJI_EXPRESSIONS = {
  '😊': 'happy', '😄': 'happy', '😁': 'happy', '😆': 'happy', '🥰': 'happy', '😍': 'happy', '🙂': 'happy', '😋': 'happy', '🌸': 'happy', '✨': 'happy', '❤️': 'happy', '💖': 'happy',
  '😒': 'angry', '😡': 'angry', '😠': 'angry', '😤': 'angry', '🤬': 'angry', '💢': 'angry',
  '😭': 'sorrow', '😢': 'sorrow', '🥺': 'sorrow', '☹️': 'sorrow', '🙁': 'sorrow', '😔': 'sorrow', '😞': 'sorrow', '💧': 'sorrow',
  '😮': 'surprised', '😯': 'surprised', '😲': 'surprised', '😳': 'surprised', '🤯': 'surprised', '😱': 'surprised', '⁉️': 'surprised', '❓': 'surprised',
  '😜': 'fun', '🤪': 'fun', '😏': 'fun', '😈': 'fun', '🙃': 'fun', '😉': 'fun',
  '😐': 'neutral', '😑': 'neutral', '😶': 'neutral', '🤖': 'neutral', '😴': 'neutral', '🤔': 'neutral', '💭': 'neutral'
};

function esc(s) {
  // Previous implementation replaced &, <, >, and " with themselves
  // (identity regex substitutions — no-ops), leaving only the apostrophe
  // actually escaped. Since this feeds innerHTML in addMessage() and
  // renderAikoContent() for both the user's own echoed text and Aiko's
  // responses, any "<", ">", "&", or '"' in chat text — including an ASR
  // transcript, tool output, or a search-result summary — rendered as
  // live HTML. Using textContent to build the escaped string sidesteps
  // hand-writing an entity map entirely and can't get this wrong again.
  const div = document.createElement('div');
  div.textContent = s || '';
  return div.innerHTML;
}

function parseMarkdown(text) {
  if (!text) return '';
  let safe = esc(text);
  safe = safe.replace(/`([^`]+)`/g, '<code>$1</code>');
  safe = safe.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  safe = safe.replace(/\*([^*]+)\*/g, '<em>$1</em>');
  safe = safe.replace(/_([^_]+)_/g, '<em>$1</em>');
  safe = safe.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  safe = safe.replace(/\n/g, '<br>');
  return safe;
}

function isControlTokenChunk(s) {
  // Server status/search/source control lines must not enter the bubble.
  return /^\s*__(?:STATUS|SEARCHING|RETRYING|SOURCES|REPLACE)(?:__:|\b)/.test(s || '');
}

function stripControlLines(text) {
  return (text || '')
    .split('\n')
    .filter((line) => !isControlTokenChunk(line) && !/^\s*__\w+__/.test(line))
    .join('\n');
}

/**
 * Parse Aiko message structure.
 * soft=true (while streaming): keep body as dialogue only — no *()[] / ACTION
 * split that causes half-sentence / colour-box flicker mid-token.
 * soft=false (commit / final): full parse for non-verbal + action boxes.
 */
function parseAikoMessage(rawText, soft = false) {
  let text = stripControlLines(rawText || '');
  let emoji = null;
  let action = null;

  // Remove --- separators first
  text = text.replace(/^---+\s*$/gm, '');

  // 1. Parse EMOTION: prefix if present (legacy format) or leading emoji only (new format)
  const legacyEmotionMatch = text.match(/^\s*EMOTION:\s*([\u{1F300}-\u{1FAFF}\u{2600}-\u{27BF}\u{2300}-\u{23FF}\u{2B00}-\u{2BFF}\u{FE00}-\u{FE0F}]|[\uD83C-\uDBFF][\uDC00-\uDFFF]|[a-zA-Z_-]+)\s*/u);
  if (legacyEmotionMatch) {
    emoji = legacyEmotionMatch[1];
    text = text.replace(/^\s*EMOTION:\s*[\u{1F300}-\u{1FAFF}\u{2600}-\u{27BF}\u{2300}-\u{23FF}\u{2B00}-\u{2BFF}\u{FE00}-\u{FE0F}]|[\uD83C-\uDBFF][\uDC00-\uDFFF]|[a-zA-Z_-]+\s*/u, '');
  } else {
    const directEmojiRegex = /^\s*([\u{1F300}-\u{1FAFF}\u{2600}-\u{27BF}\u{2300}-\u{23FF}\u{2B00}-\u{2BFF}\u{FE00}-\u{FE0F}]|[\uD83C-\uDBFF][\uDC00-\uDFFF])\s*:?\s*/u;
    const directEmojiMatch = text.match(directEmojiRegex);
    if (directEmojiMatch) {
      emoji = directEmojiMatch[1];
      text = text.replace(directEmojiRegex, '');
    }
  }

  // 2. Parse ACTION: only when complete (not soft / mid-stream)
  if (!soft) {
    const actionMatch = text.match(/(?:^|\n|\s*)ACTION:\s*([^\n]+)/i);
    if (actionMatch) {
      action = actionMatch[1].trim();
      text = text.replace(/(?:^|\n|\s*)ACTION:\s*[^\n]+\n?/i, '\n');
    }
  }

  // 3. Check for emoji header format (e.g., "😊: hello") if emoji not set yet
  const emojiHeaderRegex = /^\s*(?:([\u{1F300}-\u{1FAFF}\u{2600}-\u{27BF}\u{2300}-\u{23FF}\u{2B00}-\u{2BFF}\u{FE00}-\u{FE0F}]|[\uD83C-\uDBFF][\uDC00-\uDFFF])|:([a-zA-Z0-9_-]+):)?\s*:\s*/u;
  const match = text.match(emojiHeaderRegex);
  if (match && !emoji) {
    emoji = match[1] || match[2] || null;
    text = text.replace(emojiHeaderRegex, '');
  }

  const nonVerbalParts = [];

  if (!soft) {
    text = text.replace(/\*([^*]+)\*/g, (m, p1) => {
      nonVerbalParts.push(`*${p1.trim()}*`);
      return '';
    });

    text = text.replace(/\(([^)]+)\)/g, (m, p1) => {
      nonVerbalParts.push(`(${p1.trim()})`);
      return '';
    });

    text = text.replace(/\[([^\]]+)\]/g, (m, p1) => {
      // keep markdown links [text](url) — only strip bare [stage directions]
      return m;
    });
    // Bare [direction] without following (
    text = text.replace(/\[([^\]\n]+)\](?!\()/g, (m, p1) => {
      nonVerbalParts.push(`[${p1.trim()}]`);
      return '';
    });
  }

  // Ensure all --- lines and multiple blank lines are removed
  text = text.replace(/^---+\s*$/gm, '');
  text = text.replace(/\n{3,}/g, '\n\n').trim();

  const nonVerbalText = soft ? '' : nonVerbalParts.join(' ').trim();
  const dialogueText = soft
    ? text.replace(/\s{2,}/g, ' ').trim()
    : text.replace(/\s{2,}/g, ' ').trim();

  return { emoji, action, nonVerbalText, dialogueText };
}

function renderAikoContent(container, parsed, showCursor = false) {
  container.replaceChildren();

  const prefixSpan = document.createElement('span');
  prefixSpan.className = 'msg-prefix';
  prefixSpan.textContent = 'Aiko: ';
  container.appendChild(prefixSpan);

  if (parsed.nonVerbalText) {
    const nvDiv = document.createElement('div');
    nvDiv.className = 'msg-non-verbal';
    nvDiv.innerHTML = parseMarkdown(parsed.nonVerbalText);
    container.appendChild(nvDiv);
  }

  if (parsed.action && parsed.action.toLowerCase() !== 'none') {
    const actionDiv = document.createElement('div');
    actionDiv.className = 'msg-action';
    const cleanAction = parsed.action.replace(/^ACTION:\s*/i, '').trim();
    actionDiv.innerHTML = parseMarkdown(cleanAction);
    container.appendChild(actionDiv);
  }

  const dialSpan = document.createElement('span');
  dialSpan.className = 'msg-dialogue';
  dialSpan.innerHTML = parseMarkdown(parsed.dialogueText || '');

  if (showCursor) {
    const cursorSpan = document.createElement('span');
    cursorSpan.className = 'cursor';
    dialSpan.appendChild(cursorSpan);
  }

  container.appendChild(dialSpan);
}

// ── Bubble Message Rendering ──────────────────────────────────────────────
function createMessageRow(sender) {
  const row = document.createElement('div');
  row.className = 'msg-row' + (sender === 'you' ? ' user' : '');

  const avatar = document.createElement('div');
  avatar.className = 'msg-avatar ' + (sender === 'aiko' ? 'aiko' : 'user');
  avatar.textContent = sender === 'aiko' ? '🤖' : '🌸';

  const bubble = document.createElement('div');
  bubble.className = 'msg-bubble ' + (sender === 'aiko' ? 'aiko' : 'user');

  row.appendChild(avatar);
  row.appendChild(bubble);
  return { row, bubble };
}

function addMessage(sender, text) {
  flushStream();
  hideTypingIndicator();
  let insertEl;
  if (sender === 'you') {
    const { row, bubble } = createMessageRow('you');
    const prefix = document.createElement('span');
    prefix.className = 'msg-prefix';
    prefix.textContent = window.currentUsername || 'You';
    bubble.appendChild(prefix);
    const body = document.createElement('div');
    body.innerHTML = esc(text).replace(/\n/g, '<br>');
    bubble.appendChild(body);
    insertEl = row;
  } else if (sender === 'aiko') {
    const { row, bubble } = createMessageRow('aiko');
    const parsed = parseAikoMessage(text);
    if (parsed.emoji && window.aikoSetExpression) {
      const exprName = EMOJI_EXPRESSIONS[parsed.emoji] || parsed.emoji;
      window.aikoSetExpression(exprName, 1.0);
      spawnEmotionParticles(exprName);
      updateEmotionBadge(exprName);
    }
    renderAikoContent(bubble, parsed, false);
    insertEl = row;
  } else {
    const div = document.createElement('div');
    div.className = 'msg msg-sys';
    div.textContent = `  ◈  ${text}`;
    insertEl = div;
  }
  chatPanel.insertBefore(insertEl, toolStatus);
  scrollBottom();
}

function appendToken(text) {
  if (text == null || text === '') return;
  // Drop pure control chunks (status/search) so they never typewrite into the bubble.
  if (isControlTokenChunk(text) && !streamRawText) return;
  if (!streamDiv) {
    hideTypingIndicator();
    const { row, bubble } = createMessageRow('aiko');
    streamDiv = bubble;
    streamRawText = '';
    streamExprApplied = null;
    chatPanel.insertBefore(row, toolStatus);
  }
  streamRawText += text;
  // Soft parse while streaming: single dialogue line + cursor, no action/nv boxes.
  const parsed = parseAikoMessage(streamRawText, true);
  if (parsed.emoji && window.aikoSetExpression) {
    const exprName = EMOJI_EXPRESSIONS[parsed.emoji] || parsed.emoji;
    if (exprName && exprName !== streamExprApplied) {
      streamExprApplied = exprName;
      window.aikoSetExpression(exprName, 1.0);
    }
  }
  renderAikoContent(streamDiv, parsed, true);
  scrollBottom();
}

function flushStream() {
  if (streamDiv) {
    // Full parse only when the turn is complete.
    const parsed = parseAikoMessage(streamRawText, false);
    if (parsed.emoji && window.aikoSetExpression) {
      const exprName = EMOJI_EXPRESSIONS[parsed.emoji] || parsed.emoji;
      if (exprName && exprName !== streamExprApplied) {
        streamExprApplied = exprName;
        window.aikoSetExpression(exprName, 1.0);
      }
    }
    renderAikoContent(streamDiv, parsed, false);
    streamDiv = null;
    streamRawText = '';
    streamExprApplied = null;
  }
  toolStatus.textContent = '';
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
  if (status === 'waiting' && chatPhaseActive) showTypingIndicator();
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
  window.aikoIsSpeaking = true;
  window.AIKO_TTS_STARTED_AT = performance.now();  // S3 echo guard
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
let micStream = null;
let micContext = null;
let micSource = null;
let micWorklet = null;
let micFirstFrameSeen = false;
let micStreamingEnabled = false;
let browserVadGate = true;
let micCommandSeq = 0;
let micSecureContextWarned = false;

let micStartPromise = null;

async function startMic() {
  if (micContext) return true;
  if (micStartPromise) return micStartPromise;   // <- dedupe concurrent callers
  micStartPromise = _startMicInner().finally(() => { micStartPromise = null; });
  return micStartPromise;
}

async function _startMicInner() {
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
    if (micContext.state === 'suspended') {
      await micContext.resume();
      console.log('[mic] AudioContext was suspended — resumed');
    }
    micSource = micContext.createMediaStreamSource(micStream);

    let _vadQueue = Promise.resolve();
    function pushVADFrame(frame) {
      _vadQueue = _vadQueue.then(() => processVADFrame(frame, ws, browserVadGate)).catch(e => console.error('[mic] VAD error:', e));
    }

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
      micWorklet.connect(micContext.destination);
      awok = true;
      console.log('[mic] using AudioWorklet capture');
    } catch (awErr) {
      console.warn('[mic] AudioWorklet failed, falling back to ScriptProcessorNode:', awErr);
    }

    if (!awok) {
      const bufSize = 2048;
      const frameSamples = 512;
      let _spBuf = new Float32Array(0);
      const spNode = micContext.createScriptProcessor(bufSize, 1, 1);
      spNode.onaudioprocess = (e) => {
        if (!wsReady() || !micStreamingEnabled) return;
        const input = e.inputBuffer.getChannelData(0);
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
      spNode.connect(micContext.destination);
      micWorklet = spNode;
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
  clearAuxRows();
  flushStream();
  showTypingIndicator();
  ws.send(JSON.stringify({ type: 'user_input', text }));
  input.value = '';
}

async function captureImage() {
  if (!wsReady()) {
    addMessage('sys', 'WebSocket bridge is offline. Cannot use the camera.');
    return;
  }
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    addMessage('sys', 'This browser does not support camera capture.');
    return;
  }

  cameraBtn.disabled = true;
  toolStatus.textContent = '  📷  opening camera…';
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    const video = document.createElement('video');
    video.srcObject = stream;
    video.playsInline = true;
    await video.play();
    await new Promise(resolve => setTimeout(resolve, 250));
    const canvas = document.createElement('canvas');
    const maxWidth = 1024;
    const scale = Math.min(1, maxWidth / video.videoWidth);
    canvas.width = Math.max(1, Math.round(video.videoWidth * scale));
    canvas.height = Math.max(1, Math.round(video.videoHeight * scale));
    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
    const image = canvas.toDataURL('image/jpeg', 0.85);
    const text = input.value.trim();
    input.value = '';
    ws.send(JSON.stringify({ type: 'image_input', image, text }));
    addMessage('user', text ? `📷 Camera image — ${text}` : '📷 Camera image');
  } catch (err) {
    console.error('[camera] capture failed:', err);
    addMessage('sys', 'Camera access was unavailable. Allow camera permission and try again.');
  } finally {
    if (stream) stream.getTracks().forEach(track => track.stop());
    cameraBtn.disabled = false;
    input.focus();
  }
}

input.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submitInput(); }
});
sendBtn.addEventListener('click', submitInput);
cameraBtn.addEventListener('click', captureImage);
micBtn.addEventListener('click', async () => {
  if (!wsReady()) {
    addMessage('sys', 'WebSocket bridge is offline. Cannot toggle voice mode.');
    return;
  }

  const asrEnabled = vMode.textContent.includes('ASR');

  if (micContext) {
    stopMic();
    if (asrEnabled) ws.send(JSON.stringify({ type: 'user_input', text: '/listen' }));
  } else {
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

  if (wsHost.endsWith(".ts.net")) {
    return wsProto + "//" + wsHost + "/ws";
  }

  if (wsPortParam) {
    return wsProto + "//" + wsHost + ":" + wsPortParam + "/";
  }

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

    if (!chatPhaseActive && ['chat', 'token'].includes(msg.type)) {
      switchToChat();
    }

    switch (msg.type) {
      case 'step': handleStep(msg); break;
      case 'phase': if (msg.value === 'chat') switchToChat(); break;
      case 'chat': addMessage(msg.sender, msg.text); break;
      case 'vision':
        if (msg.status === 'working') toolStatus.textContent = '  👁  analyzing camera image…';
        else if (msg.status === 'done') toolStatus.textContent = '';
        else if (msg.status === 'error') { toolStatus.textContent = ''; addMessage('sys', msg.message); }
        break;
      case 'token': appendToken(msg.text); break;
      case 'sources': renderSources(msg.items || []); break;
      case 'files': renderFiles(msg.items || []); break;
      case 'meta':
        if (msg.emotion && window.aikoSetExpression) {
          const expr = msg.emotion === 'neutral' ? 'neutral' : msg.emotion;
          window.aikoSetExpression(expr, 0.9);
          spawnEmotionParticles(expr);
          updateEmotionBadge(expr);
        }
        break;
      case 'commit': flushStream(); hideTypingIndicator(); break;
      case 'tool': toolStatus.textContent = msg.status ? `  ⚙  ${msg.status}` : ''; break;
      case 'vitals': applyVitals(msg); break;
      case 'voice': applyVoice(msg.status); break;
      case 'mic':
        if (msg.action === 'start') {
          const seq = ++micCommandSeq;
          browserVadGate = msg.browser_vad_gate !== false;
          // S0: master barge-in switch from server (BARGE_IN_ENABLED)
          window.AIKO_BARGE_IN_ENABLED = !!msg.barge_in_enabled;
          // S3: echo guard window from server (BARGE_IN_ECHO_GUARD_MS)
          window.AIKO_BARGE_ECHO_GUARD_MS = msg.echo_guard_ms ?? 450;
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
      case 'expression':
        if (window.aikoSetExpression) {
          window.aikoSetExpression(msg.name, msg.intensity ?? 1.0);
          spawnEmotionParticles(msg.name);
          updateEmotionBadge(msg.name);
        }
        break;
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
      window.currentUsername = data.username || 'You';
      const aiNameEl = document.getElementById('vrm-ai-name');
      if (aiNameEl) aiNameEl.textContent = data.ai_name || 'Aiko';
      const userNameEl = document.getElementById('vrm-user-name');
      if (userNameEl) userNameEl.textContent = window.currentUsername;
      hideAuthOverlay();
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
