/**
 * vad.js
 * Browser-side VAD between pcm-worklet.js and the WebSocket.
 * Energy-gate only. Backend Silero is authoritative.
 *
 * S0: barge only if window.AIKO_BARGE_IN_ENABLED
 * S1: PRE_SPEECH_BUFS ~700ms
 * S3: echo guard after TTS start + stricter barge confirm / energy
 */

const SILENCE_TIMEOUT = 1200;
const PRE_SPEECH_BUFS = 22;

const ENERGY_START_RMS = 0.008;
const ENERGY_END_RMS = 0.005;
const ENERGY_MIN_FRAMES = 2;

// S3 barge: need more sustained energy than plain speech-onset
const BARGE_IN_CONFIRM_FRAMES = 4;
const BARGE_RMS_MULT = 1.5;  // barge threshold = speech start * this
const DEFAULT_ECHO_GUARD_MS = 450;

let _noiseFloor = 0.015;

let _speaking = false;
let _silTimer = null;
let _preBuf = [];
let _energyHits = 0;
let _vadEpoch = 0;
let _lastBargeSent = 0;
let _bargeHits = 0;

function _bargeInEnabled() {
    return window.AIKO_BARGE_IN_ENABLED === true || window.AIKO_BARGE_IN_ENABLED === 1
        || window.AIKO_BARGE_IN_ENABLED === "1";
}

function _echoGuardMs() {
    const v = window.AIKO_BARGE_ECHO_GUARD_MS;
    if (typeof v === "number" && v >= 0) return v;
    if (typeof v === "string" && v !== "" && !Number.isNaN(Number(v))) return Number(v);
    return DEFAULT_ECHO_GUARD_MS;
}

function _inEchoGuard() {
    // Set by webui.js when TTS playback starts
    const t0 = window.AIKO_TTS_STARTED_AT;
    if (typeof t0 !== "number" || !t0) return false;
    return (performance.now() - t0) < _echoGuardMs();
}

async function initVAD() {
    _resetState();
    return { mode: 'energy', ready: true, fallback: false };
}

function resetVADState() {
    _vadEpoch++;
    _resetState();
}

function _resetState() {
    _preBuf = [];
    _speaking = false;
    _energyHits = 0;
    _bargeHits = 0;
    _lastBargeSent = 0;
    if (_silTimer) { clearTimeout(_silTimer); _silTimer = null; }
}

async function processVADFrame(frame, ws, gate = true) {
    const epoch = _vadEpoch;
    if (!_canSend(ws, epoch)) return;
    processEnergyVADFrame(frame, ws, epoch, gate);
}

function _calcThresholds() {
    const startThresh = Math.max(ENERGY_START_RMS, _noiseFloor * 2.2);
    // ENERGY_END_RMS was declared but never actually applied here — endThresh
    // had no floor at all, only the 0.5 ceiling. In practice this rarely
    // bites (the 0.5 ceiling only engages once _noiseFloor exceeds ~0.33
    // RMS, i.e. near-clipping ambient noise), but it's still a real gap:
    // add the floor back so endThresh can't collapse toward 0 either.
    const endThresh = Math.max(ENERGY_END_RMS, Math.min(_noiseFloor * 1.5, 0.5));
    return { startThresh, endThresh };
}

function _fireBarge(ws) {
    if (!_bargeInEnabled()) return;
    if (!window.aikoIsSpeaking) return;
    if (_inEchoGuard()) return;
    const now = performance.now();
    if (now - _lastBargeSent <= 300) return;
    _lastBargeSent = now;
    _bargeHits = 0;
    if (window.stopTtsPlayback) window.stopTtsPlayback();
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'barge_in' }));
    }
}

function _maybeBargeIn(ws, rms, startThresh) {
    if (!_bargeInEnabled()) return;
    if (!window.aikoIsSpeaking) return;
    if (_inEchoGuard()) {
        _bargeHits = 0;
        return;
    }
    const bargeThresh = startThresh * BARGE_RMS_MULT;
    if (rms < bargeThresh) {
        _bargeHits = 0;
        return;
    }
    _bargeHits++;
    if (_bargeHits < BARGE_IN_CONFIRM_FRAMES) return;
    _fireBarge(ws);
}

function processEnergyVADFrame(frame, ws, epoch = _vadEpoch, gate = true) {
    if (!_canSend(ws, epoch)) return;
    if (!gate) {
        ws.send(frame.buffer.slice(0));
    }
    const rms = _rms(frame);

    const { startThresh } = _calcThresholds();

    if (!_speaking) {
        if (rms < _noiseFloor) {
            _noiseFloor = rms;
        } else {
            _noiseFloor += (rms - _noiseFloor) * 0.001;
        }
    }

    const { endThresh } = _calcThresholds();

    if (!_speaking && rms >= startThresh) {
        _energyHits++;
        if (_energyHits < ENERGY_MIN_FRAMES) {
            if (gate) _pushPreSpeech(frame);
            return;
        }

        _speaking = true;
        _energyHits = 0;
        if (_silTimer) { clearTimeout(_silTimer); _silTimer = null; }
        if (!_canSend(ws, epoch)) return;

        // Onset barge: still requires echo guard + higher energy path via _maybeBargeIn
        _maybeBargeIn(ws, rms, startThresh);

        console.log(`[vad] speech START  rms=${rms.toFixed(5)}  floor=${_noiseFloor.toFixed(5)}  start≥${startThresh.toFixed(5)}`);
        ws.send(JSON.stringify({ type: 'vad', event: 'start' }));
        if (gate) {
            for (const buf of _preBuf) {
                if (!_canSend(ws, epoch)) return;
                ws.send(buf);
            }
            if (!_canSend(ws, epoch)) return;
            ws.send(frame.buffer.slice(0));
        }
        _preBuf = [];
        return;
    }

    if (_speaking) {
        if (gate) {
            if (!_canSend(ws, epoch)) return;
            ws.send(frame.buffer.slice(0));
        }

        _maybeBargeIn(ws, rms, startThresh);

        if (rms > endThresh) {
            if (_silTimer) { clearTimeout(_silTimer); _silTimer = null; }
            return;
        }

        if (!_silTimer) {
            _silTimer = setTimeout(() => {
                _silTimer = null;
                if (!_canSend(ws, epoch)) return;
                _speaking = false;
                _energyHits = 0;
                if (!_canSend(ws, epoch)) return;
                console.log(`[vad] speech END  floor=${_noiseFloor.toFixed(5)}`);
                ws.send(JSON.stringify({ type: 'vad', event: 'end' }));
            }, SILENCE_TIMEOUT);
        }
        return;
    }

    _energyHits = 0;
    if (gate) _pushPreSpeech(frame);
}

function _canSend(ws, epoch) {
    return epoch === _vadEpoch && ws && ws.readyState === WebSocket.OPEN;
}

function _rms(frame) {
    let sum = 0;
    for (let i = 0; i < frame.length; i++) sum += frame[i] * frame[i];
    return Math.sqrt(sum / frame.length);
}

function _pushPreSpeech(frame) {
    _preBuf.push(frame.buffer.slice(0));
    if (_preBuf.length > PRE_SPEECH_BUFS) _preBuf.shift();
}
