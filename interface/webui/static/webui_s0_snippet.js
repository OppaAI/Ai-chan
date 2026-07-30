// Add next to browserVadGate assignment in the mic:start handler:
//   browserVadGate = msg.browser_vad_gate !== false;
//   window.AIKO_BARGE_IN_ENABLED = !!msg.barge_in_enabled;
//
// Until this is merged into webui.js, vad.js defaults barge-in OFF unless
// the server has already set the flag (safe with BARGE_IN_ENABLED=0).
