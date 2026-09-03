/**
 * Microphone capture that produces exactly what the pipeline expects:
 * 16 kHz, mono, 16-bit PCM WAV.
 *
 * WHY NOT MediaRecorder
 * ---------------------
 * `MediaRecorder` gives WebM/Opus. Opus is a lossy perceptual codec, and the
 * detail it discards first is precisely what this pipeline measures: jitter is
 * cycle-to-cycle variation in glottal period, shimmer is cycle-to-cycle
 * variation in amplitude, and both are sub-perceptual. Measuring them on
 * decoded Opus would produce numbers about the codec.
 *
 * So this captures raw Float32 PCM from the Web Audio API, downsamples to
 * 16 kHz, and writes the WAV header itself. Nothing is transcoded anywhere,
 * and the server needs no ffmpeg.
 *
 * WHY THE DOWNSAMPLING IS AVERAGED, NOT PICKED
 * --------------------------------------------
 * Browsers usually capture at 44.1 or 48 kHz. Taking every Nth sample would
 * alias: energy above 8 kHz folds back down into the band where F0 and the
 * first formants live. Averaging each output sample over its input window is a
 * crude box-filter decimation — not as good as a designed anti-alias filter,
 * but it attenuates the fold-back instead of inviting it, and it is honest
 * about being simple.
 *
 * BROWSER AUDIO PROCESSING IS TURNED OFF
 * --------------------------------------
 * echoCancellation, noiseSuppression and autoGainControl are all disabled.
 * Every one of them is a time-varying filter on the signal. AGC in particular
 * would flatten exactly the intensity variation `intensity_rms_cv` measures,
 * and noise suppression is a spectral gate that mangles low-energy voiced
 * frames. Leaving them on would mean measuring the browser.
 */

const TARGET_RATE = 16000;

export class Recorder {
  constructor() {
    this.stream = null;
    this.context = null;
    this.source = null;
    this.processor = null;
    this.chunks = [];
    this.inputRate = 0;
    this.recording = false;
  }

  /**
   * Ask for the microphone and wire up the capture graph.
   * @returns {Promise<void>}
   */
  async start() {
    if (this.recording) return;

    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      },
    });

    this.context = new (window.AudioContext || window.webkitAudioContext)();
    this.inputRate = this.context.sampleRate;
    this.source = this.context.createMediaStreamSource(this.stream);

    // ScriptProcessorNode is deprecated in favour of AudioWorklet, but it is
    // supported everywhere and needs no separate module file — which matters
    // for something served off a homeserver with no build step. The buffer is
    // large (4096) because this only accumulates samples; nothing here is
    // latency-sensitive.
    this.processor = this.context.createScriptProcessor(4096, 1, 1);
    this.chunks = [];

    this.processor.onaudioprocess = (event) => {
      if (!this.recording) return;
      // Copy: the event buffer is reused by the browser on the next callback.
      this.chunks.push(new Float32Array(event.inputBuffer.getChannelData(0)));
    };

    this.source.connect(this.processor);
    // ScriptProcessor only fires while connected to a destination. Routing it
    // through a silent gain node means it runs without the subject hearing
    // themselves echoed back, which is distracting and changes how people talk.
    const mute = this.context.createGain();
    mute.gain.value = 0;
    this.processor.connect(mute);
    mute.connect(this.context.destination);

    this.recording = true;
  }

  /**
   * Stop capture, release the microphone, and encode.
   * @returns {Blob} A 16 kHz mono 16-bit PCM WAV.
   */
  stop() {
    this.recording = false;

    if (this.processor) {
      this.processor.disconnect();
      this.processor.onaudioprocess = null;
    }
    if (this.source) this.source.disconnect();
    if (this.stream) this.stream.getTracks().forEach((t) => t.stop());
    if (this.context) this.context.close();

    const merged = mergeChunks(this.chunks);
    const resampled = downsample(merged, this.inputRate, TARGET_RATE);
    this.chunks = [];
    return encodeWav(resampled, TARGET_RATE);
  }

  /** Seconds captured so far. @returns {number} Duration. */
  get elapsed() {
    if (!this.inputRate) return 0;
    return this.chunks.reduce((n, c) => n + c.length, 0) / this.inputRate;
  }
}

/**
 * Concatenate the captured buffers.
 * @param {Float32Array[]} chunks Captured buffers.
 * @returns {Float32Array} One contiguous buffer.
 */
function mergeChunks(chunks) {
  const total = chunks.reduce((n, c) => n + c.length, 0);
  const out = new Float32Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.length;
  }
  return out;
}

/**
 * Box-filter decimation to the target rate.
 * @param {Float32Array} input Samples at inputRate.
 * @param {number} inputRate Source rate in Hz.
 * @param {number} outputRate Target rate in Hz.
 * @returns {Float32Array} Samples at outputRate.
 */
function downsample(input, inputRate, outputRate) {
  if (outputRate === inputRate) return input;
  if (outputRate > inputRate) {
    throw new Error(
      `cannot upsample ${inputRate} Hz to ${outputRate} Hz; ` +
      `record on a device that captures at ${outputRate} Hz or above`
    );
  }

  const ratio = inputRate / outputRate;
  const outLength = Math.floor(input.length / ratio);
  const out = new Float32Array(outLength);

  for (let i = 0; i < outLength; i++) {
    const start = Math.floor(i * ratio);
    const end = Math.min(Math.floor((i + 1) * ratio), input.length);
    let sum = 0;
    for (let j = start; j < end; j++) sum += input[j];
    out[i] = end > start ? sum / (end - start) : 0;
  }
  return out;
}

/**
 * Write a 16-bit PCM mono WAV.
 * @param {Float32Array} samples Samples in [-1, 1].
 * @param {number} rate Sample rate in Hz.
 * @returns {Blob} The WAV file.
 */
function encodeWav(samples, rate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  const writeString = (offset, text) => {
    for (let i = 0; i < text.length; i++) view.setUint8(offset + i, text.charCodeAt(i));
  };

  writeString(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);        // PCM header size
  view.setUint16(20, 1, true);         // format: PCM
  view.setUint16(22, 1, true);         // channels: mono
  view.setUint32(24, rate, true);
  view.setUint32(28, rate * 2, true);  // byte rate
  view.setUint16(32, 2, true);         // block align
  view.setUint16(34, 16, true);        // bits per sample
  writeString(36, "data");
  view.setUint32(40, samples.length * 2, true);

  // Clamp before scaling. A sample at exactly 1.0 would otherwise wrap to the
  // most negative 16-bit value, putting a click in the waveform — and a click
  // is a large amplitude excursion, which is what shimmer measures.
  let offset = 44;
  for (let i = 0; i < samples.length; i++) {
    const clamped = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff, true);
    offset += 2;
  }

  return new Blob([view], { type: "audio/wav" });
}
