/**
 * vad.js
 * Browser-side VAD between pcm-worklet.js and the WebSocket.
 * Energy-gate only. Browser VAD is a coarse "is this worth sending" filter;
 * the backend runs Silero VAD as the authoritative check on whatever this
 * forwards (see listen.py _record()).
 *
 * Default flow (browser VAD gate on):
 *   pcm-worklet -> Float32Array frame -> processVADFrame(frame, ws, true)
 *     silence  -> kept locally, not sent to the server
 *     speech   -> ws.send(binary frame)
 *     on start -> ws.send({type:'vad', event:'start'}) + pre-speech context
 *     on end   -> ws.send({type:'vad', event:'end'})
 *
 * Diagnostic flow (browser VAD gate off):
 *   processVADFrame(frame, ws, false) forwards every PCM frame so server-side
 *   VAD can be evaluated in isolation.
 */

// -- tunables -----------------------------------------------------------------

const SILENCE_TIMEOUT = 1200;   // ms of silence before utterance ends
const PRE_SPEECH_BUFS = 10;     // ~320 ms of context kept before speech starts

// Energy VAD tunables — values below are your tuned optimum. Conservative
// enough to avoid streaming normal room tone.
// NOTE: If voice input never transcribes (mic blinks, says "listening" but
// nothing happens), lower ENERGY_START_RMS. Check console for "[vad]" RMS logs.
const ENERGY_START_RMS = 0.008;
const ENERGY_END_RMS = 0.005;
const ENERGY_MIN_FRAMES = 2;

// Adaptive noise tracking: running minimum RMS when not speaking, used as end-of-speech floor.
let _noiseFloor = 0.015;

// -- state --------------------------------------------------------------------

let _speaking = false;
let _silTimer = null;
let _preBuf = [];     // circular pre-speech context
let _energyHits = 0;
let _vadEpoch = 0;

// -- init ---------------------------------------------------------------------

/**
 * No model loading needed for energy VAD — kept as an async function so
 * callers (index.html) that await initVAD() don't need to change.
 */
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
    if (_silTimer) { clearTimeout(_silTimer); _silTimer = null; }
}

// -- main entry point ---------------------------------------------------------

/**
 * Process one PCM frame from pcm-worklet.js.
 * Sends binary frames + VAD sentinel JSON messages over `ws`.
 * @param {Float32Array} frame  - PCM samples at 16 kHz mono
 * @param {WebSocket}    ws     - live WebSocket to Jetson
 * @param {boolean}      gate   - true: browser VAD gates network audio;
 *                                false: diagnostic raw PCM passthrough
 */
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

function processEnergyVADFrame(frame, ws, epoch = _vadEpoch, gate = true) {
    if (!_canSend(ws, epoch)) return;
    if (!gate) {
        ws.send(frame.buffer.slice(0));
    }
    const rms = _rms(frame);

    // Adaptive noise floor: track the minimum RMS when not speaking
    if (!_speaking) {
        if (rms < _noiseFloor) {
            _noiseFloor = rms;
        } else {
            // Slowly decay up so we track changes in background noise
            _noiseFloor += (rms - _noiseFloor) * 0.001;
        }
    }

    const { startThresh, endThresh } = _calcThresholds();

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
        console.log(`[vad] speech START  rms=${rms.toFixed(5)}  floor=${_noiseFloor.toFixed(5)}`);
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
