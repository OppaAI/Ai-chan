// pcm-worklet.js
// Runs on the audio render thread. Web Audio delivers 128-sample render
// quanta; we accumulate them into 512-sample frames (matching listen.py's
// _CHUNK_SAMPLES_VAD) and post each full frame to the main thread as a
// transferable Float32Array buffer.

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