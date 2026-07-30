/**
 * vad.js
 * Browser-side VAD between pcm-worklet.js and the WebSocket.
 * Energy-gate only. Browser VAD is a coarse "is this worth sending" filter;
 * the backend runs Silero VAD as the authoritative check on whatever this
 * forwards (see listen.py _record()).
 *
 * Barge-in: only when window.AIKO_BARGE_IN_ENABLED is true (set from server
 * mic-start / config). Default false until the server says otherwise so a
 * stale tab cannot cut TTS after BARGE_IN_ENABLED=0.
 */

// -- tunables -----------------------------------------------------------------

const SILENCE_TIMEOUT = 1200;   // ms of silence before utterance ends
const PRE_SPEECH_BUFS = 10;     // ~320 ms of context kept before speech starts

const ENERGY_START_RMS = 0.008;
const ENERGY_END_RMS = 0.005;
const ENERGY_MIN_FRAMES = 2;

let _noiseFloor = 0.015;

// -- state --------------------------------------------------------------------

let _speaking = false;
let _silTimer = null;
let _preBuf = [];
let _energyHits = 0;
let _vadEpoch = 0;
let _lastBargeSent = 0;
let _bargeHits = 0;

const BARGE_IN_CONFIRM_FRAMES = 2;

function _bargeInEnabled() {
    // Server sets window.AIKO_BARGE_IN_ENABLED on mic start. Default off.
    return window.AIKO_BARGE_IN_ENABLED === true || window.AIKO_BARGE_IN_ENABLED === 1
        || window.AIKO_BARGE_IN_ENABLED === "1";
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
    const endThresh = Math.min(_noiseFloor * 1.5, 0.5);
    return { startThresh, endThresh };
}

function _maybeBargeIn(ws, rms, startThresh) {
    if (!_bargeInEnabled()) return;
    if (!window.aikoIsSpeaking) return;
    if (rms < startThresh) {
        _bargeHits = 0;
        return;
    }
    _bargeHits++;
    if (_bargeHits < BARGE_IN_CONFIRM_FRAMES) return;
    const now = performance.now();
    if (now - _lastBargeSent <= 300) return;
    _lastBargeSent = now;
    _bargeHits = 0;
    if (window.stopTtsPlayback) window.stopTtsPlayback();
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'barge_in' }));
    }
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

        // Onset barge (only if enabled)
        if (_bargeInEnabled() && window.aikoIsSpeaking) {
            const now = performance.now();
            if (now - _lastBargeSent > 300) {
                _lastBargeSent = now;
                if (window.stopTtsPlayback) window.stopTtsPlayback();
                ws.send(JSON.stringify({ type: 'barge_in' }));
            }
        }

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
