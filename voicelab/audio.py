"""Accepting, validating and measuring an uploaded recording.

One job: take the bytes the browser posted, satisfy yourself they are the WAV
this pipeline can measure, store them, and extract the acoustic features once.

Why the browser sends WAV and not what MediaRecorder produces
-------------------------------------------------------------
`MediaRecorder` gives you WebM/Opus. Opus is a *lossy perceptual* codec: it
throws away exactly the fine-grained waveform detail that this pipeline
measures. Jitter is cycle-to-cycle variation in glottal period, and shimmer is
cycle-to-cycle variation in amplitude -- both are sub-perceptual, both are the
first thing a perceptual codec discards, and both carry the largest weights in
the signal. Measuring jitter on decoded Opus would produce a number, and the
number would be about the codec.

So `static/recorder.js` captures raw Float32 PCM from the Web Audio API,
downsamples to 16 kHz, and writes a 16-bit PCM WAV in the browser. Nothing is
transcoded here or anywhere else, and no ffmpeg is needed on the server.

Sample-rate mismatches are refused rather than resampled, for the same reason
the parent project refuses them: resampling changes period lengths, and jitter
*is* a period-length measurement.
"""

from __future__ import annotations

import io
import struct
import wave
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

import config
from voicelab.dsp import settings as dsp_settings
from voicelab.dsp.acoustic_features import extract_features
from voicelab.dsp.audio_preprocess import preprocess


class BadRecording(ValueError):
    """Raised when an upload is not a usable recording."""


@dataclass(frozen=True)
class StoredRecording:
    """A recording that has been written to disk and measured.

    Attributes:
        filename: Name within the audio directory.
        duration_sec: Length of the stored audio.
        sample_rate: Sample rate as stored.
        features: Extracted acoustic features, or None when extraction failed.
        error: Why extraction failed, when it did.
    """

    filename: str
    duration_sec: float
    sample_rate: int
    features: Optional[Dict[str, float]]
    error: Optional[str]


def decode_wav(raw: bytes) -> Tuple[np.ndarray, int]:
    """Decode a 16-bit PCM mono WAV into a float waveform.

    Args:
        raw: The uploaded bytes.

    Returns:
        ``(samples, sample_rate)`` with samples in [-1, 1].

    Raises:
        BadRecording: If the bytes are not a WAV this pipeline can read.

    Note:
        Normalisation is by the format's full scale (32768), not by the
        sample's own peak. Peak-normalising would make every recording equally
        loud and destroy the intensity features outright.
    """
    try:
        with wave.open(io.BytesIO(raw), "rb") as handle:
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            rate = handle.getframerate()
            frames = handle.readframes(handle.getnframes())
    except (wave.Error, EOFError, struct.error) as exc:
        raise BadRecording(f"not a readable WAV file: {exc}") from exc

    if width != 2:
        raise BadRecording(
            f"expected 16-bit samples, got {width * 8}-bit"
        )
    if channels != 1:
        raise BadRecording(f"expected mono audio, got {channels} channels")
    if rate != dsp_settings.VOICE_SAMPLE_RATE_HZ:
        raise BadRecording(
            f"expected {dsp_settings.VOICE_SAMPLE_RATE_HZ} Hz, got {rate} Hz. "
            "Resampling is refused rather than performed, because it changes "
            "the period lengths this pipeline measures."
        )

    samples = np.frombuffer(frames, dtype="<i2").astype(np.float64) / 32768.0
    if samples.size == 0:
        raise BadRecording("recording contained no audio")
    return samples, rate


def store(raw: bytes, username: str, label: str, audio_dir: Optional[Path] = None) -> StoredRecording:
    """Validate, save and measure one uploaded recording.

    Args:
        raw: The uploaded bytes.
        username: Whose recording, used in the filename.
        label: The recording's label, used in the filename.
        audio_dir: Destination directory. Defaults to ``config.AUDIO_DIR``.

    Returns:
        A :class:`StoredRecording`.

    Raises:
        BadRecording: If the upload is not usable at all. Note that a
            *measurable* failure -- audio that decodes but that the extractor
            cannot get features from -- is not an error here. It is stored with
            its reason recorded, because how often that happens on real
            recordings is one of the things this lab exists to find out.
    """
    if len(raw) > config.MAX_UPLOAD_BYTES:
        raise BadRecording("recording is too large")

    samples, rate = decode_wav(raw)
    duration = len(samples) / float(rate)

    if duration < dsp_settings.VOICE_MIN_DURATION_SEC:
        raise BadRecording(
            f"recording is {duration:.1f} s; at least "
            f"{dsp_settings.VOICE_MIN_DURATION_SEC:.0f} s is needed for a "
            "usable pitch and rhythm measurement"
        )

    directory = Path(audio_dir or config.AUDIO_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    safe_user = "".join(c for c in username if c.isalnum() or c in "-_")[:32]
    safe_label = "".join(c for c in label if c.isalnum() or c in "-_")[:16]
    filename = f"{safe_user}-{safe_label}-{stamp}.wav"
    (directory / filename).write_bytes(raw)

    features: Optional[Dict[str, float]] = None
    error: Optional[str] = None
    try:
        audio = preprocess(samples, rate)
        extracted = extract_features(audio)
        if not extracted.is_usable:
            error = "extractor could not measure this recording (too little voiced speech)"
        else:
            features = extracted.to_dict()
    except Exception as exc:  # noqa: BLE001 - report, do not lose the sample
        error = f"{type(exc).__name__}: {exc}"

    return StoredRecording(
        filename=filename,
        duration_sec=round(duration, 2),
        sample_rate=rate,
        features=features,
        error=error,
    )
