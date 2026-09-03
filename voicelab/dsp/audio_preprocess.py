"""Prepare a raw waveform for acoustic analysis.

One job: framing, energy, and the voiced/silent decision. No feature is
extracted here and no interpretation is made -- this module hands
``acoustic_features.py`` a clean, framed signal and a mask saying which frames
carry voice.

What this module does NOT do, ever
----------------------------------
No transcription. No speech-to-text. No phoneme recognition, no word spotting,
no language identification, no keyword detection. Nothing in this file or
anywhere else in ``voice_pipeline/`` looks at *what* was said. The only
question the whole package answers is *how* the voice was produced. That is not
an implementation detail; it is the constraint that makes a voice feature
acceptable in a welfare system at all, and it is why the module boundary is
drawn here.

Pipeline position:
    ``ingestion/voice_loader`` -> **audio_preprocess** ->
    ``voice_pipeline/acoustic_features``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from voicelab.dsp import settings

# ASSUMPTION: standard speech pre-emphasis coefficient. Applied only to the
# copy used for pitch estimation -- see the note in :func:`preprocess`.
PRE_EMPHASIS_COEFFICIENT: float = 0.97


@dataclass(frozen=True)
class PreprocessedAudio:
    """A framed, analysis-ready waveform.

    Attributes:
        waveform: DC-removed signal, trimmed of leading/trailing silence.
            Amplitude is untouched otherwise, so intensity features remain
            meaningful.
        emphasised: Pre-emphasised copy, used for pitch estimation only.
        sample_rate: Sample rate in Hz.
        frames: 2D array of shape (n_frames, frame_length) over ``waveform``.
        emphasised_frames: Same framing applied to ``emphasised``.
        frame_rms: RMS energy per frame, shape (n_frames,).
        voiced_mask: Boolean array, True where the frame carries voice.
        frame_times: Start time of each frame in seconds.
        hop_length: Hop between frames in samples.
        frame_length: Frame length in samples.
    """

    waveform: np.ndarray
    emphasised: np.ndarray
    sample_rate: int
    frames: np.ndarray
    emphasised_frames: np.ndarray
    frame_rms: np.ndarray
    voiced_mask: np.ndarray
    frame_times: np.ndarray
    hop_length: int
    frame_length: int

    @property
    def duration_sec(self) -> float:
        """Duration of the trimmed waveform in seconds."""
        return float(len(self.waveform)) / float(self.sample_rate)

    @property
    def voiced_fraction(self) -> float:
        """Fraction of frames classified as voiced."""
        return float(self.voiced_mask.mean()) if self.voiced_mask.size else 0.0


def remove_dc_offset(waveform: np.ndarray) -> np.ndarray:
    """Subtract the mean, removing any DC bias.

    Args:
        waveform: Input signal.

    Returns:
        Zero-mean signal.

    Note:
        A DC offset shifts every RMS measurement upward and biases the
        autocorrelation at all lags, so this runs before anything else.
    """
    return waveform - float(np.mean(waveform)) if waveform.size else waveform


def pre_emphasis(waveform: np.ndarray, coefficient: float = PRE_EMPHASIS_COEFFICIENT) -> np.ndarray:
    """Apply a first-order high-pass pre-emphasis filter.

    Args:
        waveform: Input signal.
        coefficient: Pre-emphasis coefficient.

    Returns:
        Filtered signal of the same length.

    Note:
        Pre-emphasis flattens the spectral tilt of voiced speech, which makes
        the autocorrelation peak at the pitch period sharper. It also changes
        the amplitude envelope, which is why it is applied to a *copy* and the
        intensity features are computed from the unfiltered signal.
    """
    if waveform.size == 0:
        return waveform
    out = np.empty_like(waveform)
    out[0] = waveform[0]
    out[1:] = waveform[1:] - coefficient * waveform[:-1]
    return out


def frame_signal(
    waveform: np.ndarray, frame_length: int, hop_length: int
) -> np.ndarray:
    """Split a signal into overlapping frames.

    Args:
        waveform: Input signal.
        frame_length: Frame length in samples.
        hop_length: Hop between frame starts in samples.

    Returns:
        Array of shape (n_frames, frame_length). Returns an empty
        (0, frame_length) array when the signal is shorter than one frame.
    """
    if waveform.size < frame_length:
        return np.empty((0, frame_length), dtype=np.float64)
    n_frames = 1 + (waveform.size - frame_length) // hop_length
    indices = np.arange(frame_length)[None, :] + (
        hop_length * np.arange(n_frames)[:, None]
    )
    return waveform[indices]


def frame_energy(frames: np.ndarray) -> np.ndarray:
    """Compute RMS energy per frame.

    Args:
        frames: Framed signal, shape (n_frames, frame_length).

    Returns:
        RMS per frame, shape (n_frames,). Empty input gives an empty array.
    """
    if frames.size == 0:
        return np.empty(0, dtype=np.float64)
    return np.sqrt(np.mean(np.square(frames), axis=1))


def voiced_frames(
    frame_rms: np.ndarray, silence_fraction: float = settings.VOICE_SILENCE_RMS_FRACTION
) -> np.ndarray:
    """Classify frames as voiced or silent by relative energy.

    Args:
        frame_rms: RMS per frame.
        silence_fraction: Threshold as a fraction of the recording's peak
            frame RMS. Frames at or below it are silent.

    Returns:
        Boolean mask, True for voiced frames.

    Design note:
        The threshold is *relative* to the recording's own peak rather than an
        absolute level. Check-ins will be recorded on different handsets at
        different distances, and an absolute threshold would classify a quiet
        recording as entirely silent and a loud one as entirely voiced. The
        cost of the relative choice is that a recording containing nothing but
        background noise will have some of that noise classified as voice; the
        minimum-duration and voiced-fraction checks downstream are what catch
        that case.
    """
    if frame_rms.size == 0:
        return np.zeros(0, dtype=bool)
    peak = float(np.max(frame_rms))
    if peak <= 0:
        return np.zeros_like(frame_rms, dtype=bool)
    return frame_rms > (peak * silence_fraction)


def trim_silence(
    waveform: np.ndarray,
    sample_rate: int,
    frame_length: int,
    hop_length: int,
    silence_fraction: float = settings.VOICE_SILENCE_RMS_FRACTION,
) -> np.ndarray:
    """Remove leading and trailing silence, keeping internal pauses intact.

    Args:
        waveform: Input signal.
        sample_rate: Sample rate in Hz.
        frame_length: Frame length in samples.
        hop_length: Hop in samples.
        silence_fraction: Silence threshold as a fraction of peak frame RMS.

    Returns:
        Trimmed signal. Returns the input unchanged when no voiced frame is
        found.

    Design note:
        Internal pauses are deliberately preserved -- pause ratio is one of the
        features being measured, and trimming internal silence would destroy
        it. Only the dead air at the start and end, which is an artefact of
        when the person pressed record, is removed.
    """
    frames = frame_signal(waveform, frame_length, hop_length)
    mask = voiced_frames(frame_energy(frames), silence_fraction)
    if not mask.any():
        return waveform
    first, last = int(np.argmax(mask)), int(len(mask) - 1 - np.argmax(mask[::-1]))
    start = first * hop_length
    end = min(len(waveform), last * hop_length + frame_length)
    return waveform[start:end]


def preprocess(
    waveform: np.ndarray, sample_rate: int = settings.VOICE_SAMPLE_RATE_HZ
) -> PreprocessedAudio:
    """Run the full preprocessing chain on one waveform.

    Args:
        waveform: Raw mono signal in [-1, 1].
        sample_rate: Sample rate in Hz.

    Returns:
        A :class:`PreprocessedAudio` ready for feature extraction.

    Steps, in order:
        1. Remove DC offset.
        2. Trim leading and trailing silence (internal pauses kept).
        3. Frame at the configured frame/hop length.
        4. Compute per-frame RMS from the *unfiltered* signal.
        5. Classify frames voiced/silent.
        6. Produce a pre-emphasised copy and frame it, for pitch estimation.

    Note:
        Steps 4 and 6 use different signals on purpose. Intensity must come
        from the untouched waveform or pre-emphasis distorts it; pitch
        estimation benefits from pre-emphasis. Computing both from one signal
        would mean choosing which of the two features to degrade.
    """
    frame_length = int(round(settings.VOICE_FRAME_LENGTH_MS * sample_rate / 1000.0))
    hop_length = int(round(settings.VOICE_HOP_LENGTH_MS * sample_rate / 1000.0))

    signal = remove_dc_offset(np.asarray(waveform, dtype=np.float64))
    signal = trim_silence(signal, sample_rate, frame_length, hop_length)

    frames = frame_signal(signal, frame_length, hop_length)
    rms = frame_energy(frames)
    mask = voiced_frames(rms)

    emphasised = pre_emphasis(signal)
    emphasised_frames = frame_signal(emphasised, frame_length, hop_length)

    times = np.arange(len(frames), dtype=np.float64) * hop_length / float(sample_rate)

    return PreprocessedAudio(
        waveform=signal,
        emphasised=emphasised,
        sample_rate=sample_rate,
        frames=frames,
        emphasised_frames=emphasised_frames,
        frame_rms=rms,
        voiced_mask=mask,
        frame_times=times,
        hop_length=hop_length,
        frame_length=frame_length,
    )
