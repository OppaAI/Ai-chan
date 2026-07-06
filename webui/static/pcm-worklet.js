/**
 * pcm-worklet.js
 * AudioWorklet processor running on the audio render thread.
 * Accumulates Web Audio's 128-sample render quanta into 512-sample PCM frames
 * (matching listen.py _CHUNK_SAMPLES_VAD) and posts each full frame to main thread
 * as a transferable Float32Array buffer for VAD processing.
 */

class PCMCaptureProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this._FRAME_SAMPLES = 512;
        this._buf = new Float32Array(this._FRAME_SAMPLES);
        this._fill = 0;
    }

    process(inputs) {
        const input = inputs[0];
        if (!input || !input[0]) return true;
        const channel = input[0]; // mono — first channel only

        for (let i = 0; i < channel.length; i++) {
            this._buf[this._fill++] = channel[i];
            if (this._fill === this._FRAME_SAMPLES) {
                // copy out before transferring, so we can keep reusing this._buf
                const out = this._buf.slice(0);
                this.port.postMessage(out.buffer, [out.buffer]);
                this._fill = 0;
            }
        }
        return true;
    }
}

registerProcessor('pcm-capture-processor', PCMCaptureProcessor);
