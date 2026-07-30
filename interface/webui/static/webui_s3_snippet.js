// S3 — fold into webui.js:
//
// 1) In playNextTts(), when setting window.aikoIsSpeaking = true:
//      window.AIKO_TTS_STARTED_AT = performance.now();
//
// 2) In mic:start handler next to AIKO_BARGE_IN_ENABLED:
//      window.AIKO_BARGE_ECHO_GUARD_MS = msg.echo_guard_ms ?? 450;
//
// vad.js already honors both. Defaults work even without (2).
