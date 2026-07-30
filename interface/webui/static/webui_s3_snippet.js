// S3 — folded into webui.js (this file is historical reference only).
//
// 1) playNextTts(): window.AIKO_TTS_STARTED_AT = performance.now();
// 2) mic:start:     window.AIKO_BARGE_ECHO_GUARD_MS = msg.echo_guard_ms ?? 450;
// 3) webui.py mic broadcast includes echo_guard_ms from BARGE_IN_ECHO_GUARD_MS.
//
// vad.js already honors both. Defaults work even without server payload (450ms).
