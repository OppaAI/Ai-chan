/**
 * vad.js
 * Browser-side VAD between pcm-worklet.js and the WebSocket.
 * Silero ONNX is preferred when its browser assets are present. If ONNX Runtime
 * Web or silero_vad.onnx is missing, this falls back to a simple energy gate so
 * microphone input still works instead of silently dropping every frame.
 *
 * Flow:
 *   pcm-worklet -> Float32Array frame -> processVADFrame(frame, ws)
 *     silence  -> dropped (never leaves the device)
 *     speech   -> ws.send(binary frame)
 *     on start -> ws.send({type:'vad', event:'start'})
 *     on end   -> ws.send({type:'vad', event:'end'})
 *
 * The server (webui.py) uses 'start'/'end' sentinels to gate _mic_active and
 * signal listen.py to skip its own VAD pass.
 */

// -- tunables -----------------------------------------------------------------

const VAD_THRESHOLD    = 0.5;    // Silero speech probability cutoff (0-1)
const SILENCE_TIMEOUT  = 1200;   // ms of silence before utterance ends
const PRE_SPEECH_BUFS  = 10;     // ~320 ms of context kept before speech starts

// Energy fallback tunables. Conservative enough to avoid streaming normal room
// tone, but intentionally simple so missing optional assets do not break input.
const ENERGY_START_RMS  = 0.018;
const ENERGY_END_RMS    = 0.010;
const ENERGY_MIN_FRAMES = 3;

// -- state --------------------------------------------------------------------

let _session     = null;   // ort.InferenceSession
let _h           = null;   // GRU hidden state tensor  [2, 1, 64]
let _c           = null;   // GRU cell  state tensor   [2, 1, 64]
let _srTensor    = null;
let _vadMode     = 'energy';
let _speaking    = false;
let _silTimer    = null;
let _preBuf      = [];     // circular pre-speech context
let _energyHits  = 0;
let _vadEpoch    = 0;

// -- init ---------------------------------------------------------------------

/**
 * Load the Silero VAD ONNX model from the same static dir as this script.
 * Returns a small status object used by index.html for the visible VAD label.
 */
async function initVAD() {
    _resetState();

    if (typeof ort === 'undefined') {
        console.warn('[vad] ONNX Runtime Web missing; using energy VAD fallback');
        return { mode: _vadMode, ready: true, fallback: true };
    }

    // Serve WASM files from the same static dir, not CDN -- works offline/LAN.
    ort.env.wasm.wasmPaths = './';
    ort.env.wasm.numThreads = 1;   // single-threaded safer on mobile

    try {
        _srTensor = new ort.Tensor('int64', BigInt64Array.from([16000n]), [1]);
        _session = await ort.InferenceSession.create('./silero_vad.onnx', {
            executionProviders: ['wasm'],
        });
        _vadMode = 'silero';
        _resetState();
        console.log('[vad] Silero VAD loaded');
        return { mode: _vadMode, ready: true, fallback: false };
    } catch (err) {
        _session = null;
        _vadMode = 'energy';
        _resetState();
        console.warn('[vad] failed to load Silero model; using energy VAD fallback:', err);
        return { mode: _vadMode, ready: true, fallback: true };
    }
}

function resetVADState() {
    _vadEpoch++;
    _resetState();
}

function _resetState() {
    if (typeof ort !== 'undefined') {
        const z64 = new Float32Array(2 * 1 * 64);
        _h = new ort.Tensor('float32', z64.slice(), [2, 1, 64]);
        _c = new ort.Tensor('float32', z64.slice(), [2, 1, 64]);
    } else {
        _h = null;
        _c = null;
    }
    _preBuf = [];
    _speaking = false;
    _energyHits = 0;
    if (_silTimer) { clearTimeout(_silTimer); _silTimer = null; }
}

// -- main entry point ---------------------------------------------------------

/**
 * Process one 512-sample Float32Array frame from pcm-worklet.js.
 * Sends binary frames + VAD sentinel JSON messages over `ws`.
 * @param {Float32Array} frame  - 512 samples at 16 kHz mono
 * @param {WebSocket}    ws     - live WebSocket to Jetson
 */
async function processVADFrame(frame, ws) {
    const epoch = _vadEpoch;
    if (!_canSend(ws, epoch)) return;
    if (!_session || _vadMode !== 'silero') {
        processEnergyVADFrame(frame, ws, epoch);
        return;
    }

    const input = new ort.Tensor('float32', frame, [1, frame.length]);
    let out;
    try {
        out = await _session.run({ input, sr: _srTensor, h: _h, c: _c });
    } catch (err) {
        console.error('[vad] inference error:', err);
        return;
    }

    if (!_canSend(ws, epoch)) return;

    _h = out.hn;
    _c = out.cn;
    const prob = out.output.data[0];

    if (prob >= VAD_THRESHOLD) {
        if (_silTimer) { clearTimeout(_silTimer); _silTimer = null; }

        if (!_speaking) {
            _speaking = true;
            if (!_canSend(ws, epoch)) return;
            ws.send(JSON.stringify({ type: 'vad', event: 'start' }));
            for (const buf of _preBuf) {
                if (!_canSend(ws, epoch)) return;
                ws.send(buf);
            }
            _preBuf = [];
        }

        if (!_canSend(ws, epoch)) return;
        ws.send(frame.buffer.slice(0));
    } else {
        if (_speaking) {
            if (!_canSend(ws, epoch)) return;
            ws.send(frame.buffer.slice(0));

            if (!_silTimer) {
                _silTimer = setTimeout(() => {
                    _silTimer = null;
                    if (!_canSend(ws, epoch)) return;
                    _speaking = false;
                    _resetState();
                    if (!_canSend(ws, epoch)) return;
                    ws.send(JSON.stringify({ type: 'vad', event: 'end' }));
                }, SILENCE_TIMEOUT);
            }
        } else {
            _pushPreSpeech(frame);
        }
    }
}

function processEnergyVADFrame(frame, ws, epoch = _vadEpoch) {
    if (!_canSend(ws, epoch)) return;
    const rms = _rms(frame);

    if (!_speaking && rms >= ENERGY_START_RMS) {
        _energyHits++;
        if (_energyHits < ENERGY_MIN_FRAMES) {
            _pushPreSpeech(frame);
            return;
        }

        _speaking = true;
        _energyHits = 0;
        if (_silTimer) { clearTimeout(_silTimer); _silTimer = null; }
        if (!_canSend(ws, epoch)) return;
        ws.send(JSON.stringify({ type: 'vad', event: 'start' }));
        for (const buf of _preBuf) {
            if (!_canSend(ws, epoch)) return;
            ws.send(buf);
        }
        _preBuf = [];
        if (!_canSend(ws, epoch)) return;
        ws.send(frame.buffer.slice(0));
        return;
    }

    if (_speaking) {
        if (!_canSend(ws, epoch)) return;
        ws.send(frame.buffer.slice(0));

        if (rms > ENERGY_END_RMS) {
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
                ws.send(JSON.stringify({ type: 'vad', event: 'end' }));
            }, SILENCE_TIMEOUT);
        }
        return;
    }

    _energyHits = 0;
    _pushPreSpeech(frame);
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
