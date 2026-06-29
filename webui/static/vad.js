/**
 * vad.js
 * Browser-side Silero VAD via ONNX Runtime Web (WASM).
 * Sits between pcm-worklet.js and the WebSocket — only speech frames
 * are sent over the network, silence and ambient noise are dropped locally.
 *
 * Flow:
 *   pcm-worklet → Float32Array frame → processVADFrame(frame, ws)
 *     silence  → dropped (never leaves the device)
 *     speech   → ws.send(binary frame)
 *     on start → ws.send({type:'vad', event:'start'})
 *     on end   → ws.send({type:'vad', event:'end'})
 *
 * The server (aiko_web.py) uses 'start'/'end' sentinels to gate
 * _mic_active and signal listen.py to skip its own VAD pass.
 */

// ── tunables ──────────────────────────────────────────────────────────────────

const VAD_THRESHOLD    = 0.5;    // speech probability cutoff (0–1)
const SILENCE_TIMEOUT  = 1200;   // ms of silence before utterance ends
const PRE_SPEECH_BUFS  = 10;     // ~320 ms of context kept before speech starts
                                 // prevents clipped word-initial consonants

// ── state ─────────────────────────────────────────────────────────────────────

let _session     = null;   // ort.InferenceSession
let _h           = null;   // GRU hidden state tensor  [2, 1, 64]
let _c           = null;   // GRU cell  state tensor   [2, 1, 64]
let _speaking    = false;
let _silTimer    = null;
let _preBuf      = [];     // circular pre-speech context

const _SR = new ort.Tensor('int64', BigInt64Array.from([16000n]), [1]);

// ── init ──────────────────────────────────────────────────────────────────────

/**
 * Load the Silero VAD ONNX model from the same static dir as this script.
 * Call once after the page loads; processVADFrame() is a no-op until done.
 */
async function initVAD() {
    // serve WASM files from the same static dir, not CDN — works offline/LAN
    ort.env.wasm.wasmPaths = './';
    ort.env.wasm.numThreads = 1;   // single-threaded safer on mobile

    try {
        _session = await ort.InferenceSession.create('./silero_vad.onnx', {
            executionProviders: ['wasm'],
        });
        _resetState();
        console.log('[vad] Silero VAD loaded');
    } catch (err) {
        console.error('[vad] failed to load model:', err);
    }
}

function _resetState() {
    const z64 = new Float32Array(2 * 1 * 64);
    _h = new ort.Tensor('float32', z64.slice(), [2, 1, 64]);
    _c = new ort.Tensor('float32', z64.slice(), [2, 1, 64]);
    _preBuf  = [];
    _speaking = false;
    if (_silTimer) { clearTimeout(_silTimer); _silTimer = null; }
}

// ── main entry point ──────────────────────────────────────────────────────────

/**
 * Process one 512-sample Float32Array frame from pcm-worklet.js.
 * Sends binary frames + VAD sentinel JSON messages over `ws`.
 * @param {Float32Array} frame  - 512 samples at 16 kHz mono
 * @param {WebSocket}    ws     - live WebSocket to Jetson
 */
async function processVADFrame(frame, ws) {
    if (!_session || !ws || ws.readyState !== WebSocket.OPEN) return;

    // ── run inference ─────────────────────────────────────────────────────────
    const input = new ort.Tensor('float32', frame, [1, frame.length]);
    let out;
    try {
        out = await _session.run({ input, sr: _SR, h: _h, c: _c });
    } catch (err) {
        console.error('[vad] inference error:', err);
        return;
    }

    // update recurrent state for next frame
    _h = out.hn;
    _c = out.cn;
    const prob = out.output.data[0];

    // ── speech detected ───────────────────────────────────────────────────────
    if (prob >= VAD_THRESHOLD) {
        if (_silTimer) { clearTimeout(_silTimer); _silTimer = null; }

        if (!_speaking) {
            _speaking = true;
            ws.send(JSON.stringify({ type: 'vad', event: 'start' }));

            // flush pre-speech context so we don't clip the first phoneme
            for (const buf of _preBuf) ws.send(buf);
            _preBuf = [];
        }

        ws.send(frame.buffer.slice(0));   // live speech frame (copy, not transfer)

    // ── silence ───────────────────────────────────────────────────────────────
    } else {
        if (_speaking) {
            // still pad a little past end-of-speech before declaring done
            ws.send(frame.buffer.slice(0));

            if (!_silTimer) {
                _silTimer = setTimeout(() => {
                    _silTimer  = null;
                    _speaking  = false;
                    _resetState();   // reset GRU state for next utterance
                    ws.send(JSON.stringify({ type: 'vad', event: 'end' }));
                }, SILENCE_TIMEOUT);
            }
        } else {
            // accumulate pre-speech context ring buffer
            _preBuf.push(frame.buffer.slice(0));
            if (_preBuf.length > PRE_SPEECH_BUFS) _preBuf.shift();
        }
    }
}
