"""Extract acoustic features from a preprocessed waveform.

One job: measure eight properties of *how* a voice was produced.

    f0_mean_hz                       average pitch
    f0_sd_hz                         pitch variability across the utterance
    speaking_rate_syllables_per_sec  syllable nuclei per second
    pause_ratio                      share of the recording that is silence
    intensity_rms_mean               average loudness (context only)
    intensity_rms_sd                 loudness variability (context only)
    intensity_rms_cv                 loudness variability relative to level
    jitter_local_pct                 cycle-to-cycle variation in pitch period
    shimmer_local_pct                cycle-to-cycle variation in amplitude

None of these is a word, a phoneme, or anything derived from one. There is no
transcription, speech-to-text, phoneme recognition, keyword spotting or
language identification anywhere in this module or this package. The
information this file can extract is bounded by construction: from an
autocorrelation lag, a period mark and an RMS envelope, the content of speech
simply is not recoverable. That is the point.

Pipeline position:
    ``voice_pipeline/audio_preprocess`` -> **acoustic_features** ->
    ``voice_pipeline/voice_baseline`` -> ``voice_pipeline/voice_stress_signal``
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.signal import find_peaks

from voicelab.dsp import settings
from voicelab.dsp.audio_preprocess import PreprocessedAudio

# Minimum voiced frames before pitch statistics are reported at all. Below
# this, F0 mean and SD are NaN rather than a number computed from two frames.
# ASSUMPTION.
MIN_VOICED_FRAMES_FOR_PITCH: int = 5
# Minimum period marks in a voiced run before that run contributes to jitter
# and shimmer. ASSUMPTION: fewer than five cycles gives a perturbation
# estimate dominated by its own endpoints.
MIN_CYCLES_FOR_PERTURBATION: int = 5


@dataclass(frozen=True)
class AcousticFeatures:
    """The eight extracted acoustic measurements for one sample.

    Attributes:
        f0_mean_hz: Mean fundamental frequency over voiced frames.
        f0_sd_hz: Standard deviation of F0 over voiced frames.
        speaking_rate_syllables_per_sec: Estimated syllable nuclei per second.
        pause_ratio: Fraction of frames classified as silence.
        intensity_rms_mean: Mean RMS over voiced frames. Context only -- not
            scale-invariant, so never compared across recordings.
        intensity_rms_sd: Standard deviation of RMS over voiced frames.
            Context only, for the same reason.
        intensity_rms_cv: ``intensity_rms_sd / intensity_rms_mean``. This is
            the scale-invariant loudness-variability measure that IS compared
            across recordings.
        jitter_local_pct: Mean absolute period-to-period difference, as a
            percentage of mean period.
        shimmer_local_pct: Mean absolute amplitude difference between
            consecutive cycles, as a percentage of mean amplitude.
        voiced_frame_count: How many frames were voiced (a quality indicator,
            not a stress feature).
        analysed_duration_sec: Duration analysed after silence trimming.
    """

    f0_mean_hz: float
    f0_sd_hz: float
    speaking_rate_syllables_per_sec: float
    pause_ratio: float
    intensity_rms_mean: float
    intensity_rms_sd: float
    intensity_rms_cv: float
    jitter_local_pct: float
    shimmer_local_pct: float
    voiced_frame_count: int
    analysed_duration_sec: float

    def to_dict(self) -> Dict[str, float]:
        """Return all fields as a plain dictionary."""
        return asdict(self)

    def feature_vector(self) -> Dict[str, float]:
        """Return the features compared against a personal baseline.

        Returns:
            Mapping restricted to ``settings.VOICE_COMPARISON_FEATURE_NAMES``.

        Two kinds of field are excluded. The quality indicators
        (``voiced_frame_count``, ``analysed_duration_sec``) describe the
        recording rather than the voice. The absolute intensity measures
        (``intensity_rms_mean``, ``intensity_rms_sd``) are excluded because
        they are not scale-invariant: recording level varies with handset,
        gain and distance from the mouth, so comparing them across check-ins
        measures the recording setup, not the person. Their ratio
        (``intensity_rms_cv``) is scale-invariant and is used in their place.
        """
        full = self.to_dict()
        return {name: full[name] for name in settings.VOICE_COMPARISON_FEATURE_NAMES}

    @property
    def is_usable(self) -> bool:
        """Whether enough voiced material was found to trust the measurements."""
        return (
            self.voiced_frame_count >= MIN_VOICED_FRAMES_FOR_PITCH
            and not np.isnan(self.f0_mean_hz)
        )


def _parabolic_peak(values: np.ndarray, index: int) -> float:
    """Refine a discrete peak location by fitting a parabola to its neighbours.

    Args:
        values: The sequence the peak was found in.
        index: Integer index of the peak.

    Returns:
        Sub-sample peak position. Falls back to ``index`` at the array edges.

    Note:
        This matters more than it looks. Jitter is a measurement of *tiny*
        differences between consecutive periods; at 16 kHz one sample is
        already ~0.9% of a 125 Hz period, so integer-resolution lags would put
        a quantisation floor right on top of the quantity being measured.
    """
    if index <= 0 or index >= len(values) - 1:
        return float(index)
    left, centre, right = values[index - 1], values[index], values[index + 1]
    denominator = left - 2.0 * centre + right
    if abs(denominator) < 1e-12:
        return float(index)
    return float(index) + 0.5 * (left - right) / denominator


def estimate_f0_per_frame(
    audio: PreprocessedAudio,
    f0_min: float = settings.VOICE_F0_MIN_HZ,
    f0_max: float = settings.VOICE_F0_MAX_HZ,
    voicing_threshold: float = settings.VOICE_AUTOCORR_VOICING_THRESHOLD,
) -> np.ndarray:
    """Estimate fundamental frequency for every frame by autocorrelation.

    Args:
        audio: Preprocessed audio.
        f0_min: Lowest F0 to search for, in Hz.
        f0_max: Highest F0 to search for, in Hz.
        voicing_threshold: Minimum normalised autocorrelation peak for a frame
            to count as voiced.

    Returns:
        Array of F0 per frame in Hz, NaN for frames judged unvoiced.

    Method:
        Per frame: mean-remove, autocorrelate, normalise by the zero-lag value,
        search the lag range implied by ``[f0_min, f0_max]`` for the largest
        peak, refine it parabolically, and convert lag to frequency. A frame
        whose best normalised peak falls below ``voicing_threshold`` is
        reported as unvoiced.

    Why autocorrelation rather than a learned pitch tracker:
        It is a closed-form signal-processing operation with no trained
        parameters, so its behaviour is fully inspectable and it cannot drift.
        In a system that must be defensible under scrutiny, a pitch estimate
        nobody can explain is a liability. It is also cheap enough to run on
        every check-in without a queue.
    """
    frames = audio.emphasised_frames
    if frames.size == 0:
        return np.empty(0, dtype=np.float64)

    sample_rate = audio.sample_rate
    min_lag = max(2, int(np.floor(sample_rate / f0_max)))
    max_lag = min(frames.shape[1] - 1, int(np.ceil(sample_rate / f0_min)))
    if max_lag <= min_lag:
        return np.full(frames.shape[0], np.nan)

    f0 = np.full(frames.shape[0], np.nan, dtype=np.float64)
    for i in range(frames.shape[0]):
        if not audio.voiced_mask[i]:
            continue
        frame = frames[i] - frames[i].mean()
        energy = float(np.dot(frame, frame))
        if energy <= 0:
            continue
        correlation = np.correlate(frame, frame, mode="full")[len(frame) - 1 :]
        correlation = correlation / energy

        window = correlation[min_lag : max_lag + 1]
        if window.size == 0:
            continue
        best = int(np.argmax(window))
        if window[best] < voicing_threshold:
            continue
        refined_lag = _parabolic_peak(correlation, min_lag + best)
        if refined_lag > 0:
            f0[i] = sample_rate / refined_lag
    return f0


def _voiced_runs(mask: np.ndarray, min_length: int = 3) -> List[Tuple[int, int]]:
    """Find contiguous runs of True in a boolean mask.

    Args:
        mask: Boolean array.
        min_length: Shortest run to report.

    Returns:
        List of (start, end) index pairs, end exclusive.
    """
    runs: List[Tuple[int, int]] = []
    start: int | None = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start >= min_length:
                runs.append((start, i))
            start = None
    if start is not None and len(mask) - start >= min_length:
        runs.append((start, len(mask)))
    return runs


def _period_marks(
    signal: np.ndarray, sample_rate: int, f0_hz: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Locate successive glottal cycles and their peak amplitudes.

    Args:
        signal: A contiguous voiced segment of the waveform.
        sample_rate: Sample rate in Hz.
        f0_hz: Estimated fundamental frequency of the segment.

    Returns:
        Tuple of (period lengths in samples, peak amplitude per cycle). Both
        arrays are empty when fewer than two cycles could be marked.

    Method:
        Starting from the largest excursion in the first period-and-a-half,
        step forward looking for the next largest excursion in the window
        ``[0.7T, 1.4T]`` ahead. That search window is what makes the tracker
        robust: it cannot lock onto a harmonic (too close) or skip a cycle
        (too far), because both fall outside the window.

    Assumption:
        This is peak-picking on the acoustic waveform, not electroglottographic
        or inverse-filtered glottal closure detection. It measures
        cycle-to-cycle variation in the signal as recorded, which is the same
        quantity the check-in device can actually capture. It is a genuine
        cycle-level measurement -- not a frame-level approximation -- but it is
        not a clinical-grade jitter measurement, and it is not presented as
        one anywhere in the system.
    """
    if not np.isfinite(f0_hz) or f0_hz <= 0:
        return np.empty(0), np.empty(0)
    period = sample_rate / f0_hz
    if signal.size < 2 * period:
        return np.empty(0), np.empty(0)

    magnitude = np.abs(signal)
    search_end = int(min(len(signal), 1.5 * period))
    index = int(np.argmax(magnitude[:search_end]))

    marks: List[int] = [index]
    amplitudes: List[float] = [float(magnitude[index])]
    while True:
        low = int(index + 0.7 * period)
        high = int(index + 1.4 * period)
        if high >= len(signal):
            break
        offset = int(np.argmax(magnitude[low:high]))
        index = low + offset
        marks.append(index)
        amplitudes.append(float(magnitude[index]))

    if len(marks) < 2:
        return np.empty(0), np.empty(0)
    return np.diff(np.array(marks, dtype=np.float64)), np.array(amplitudes, dtype=np.float64)


def compute_perturbation(
    audio: PreprocessedAudio, f0_per_frame: np.ndarray
) -> Tuple[float, float]:
    """Compute local jitter and shimmer across all voiced runs.

    Args:
        audio: Preprocessed audio.
        f0_per_frame: Per-frame F0 estimates.

    Returns:
        Tuple of (jitter_local_pct, shimmer_local_pct). Both NaN when no
        voiced run yielded enough cycles.

    Definitions used:
        jitter_local  = mean(|T_i - T_{i-1}|) / mean(T) x 100
        shimmer_local = mean(|A_i - A_{i-1}|) / mean(A) x 100

        These are the standard relative "local" definitions. Shimmer is
        expressed as a percentage of mean amplitude rather than in decibels;
        both conventions are in use, and the percentage form keeps it on the
        same footing as jitter for the deviation calculation downstream.

    Note:
        Cycles are marked on the *unfiltered* waveform, not the pre-emphasised
        copy, because pre-emphasis alters the amplitude of each cycle and would
        corrupt shimmer directly.
    """
    voiced = audio.voiced_mask & np.isfinite(f0_per_frame)
    runs = _voiced_runs(voiced, min_length=3)
    if not runs:
        return float("nan"), float("nan")

    jitter_values: List[float] = []
    shimmer_values: List[float] = []

    for start, end in runs:
        segment_f0 = float(np.nanmedian(f0_per_frame[start:end]))
        first_sample = start * audio.hop_length
        last_sample = min(len(audio.waveform), (end - 1) * audio.hop_length + audio.frame_length)
        segment = audio.waveform[first_sample:last_sample]

        periods, amplitudes = _period_marks(segment, audio.sample_rate, segment_f0)
        if periods.size < MIN_CYCLES_FOR_PERTURBATION:
            continue

        mean_period = float(np.mean(periods))
        if mean_period > 0:
            jitter_values.append(100.0 * float(np.mean(np.abs(np.diff(periods)))) / mean_period)

        mean_amplitude = float(np.mean(amplitudes))
        if mean_amplitude > 0:
            shimmer_values.append(
                100.0 * float(np.mean(np.abs(np.diff(amplitudes)))) / mean_amplitude
            )

    jitter = float(np.mean(jitter_values)) if jitter_values else float("nan")
    shimmer = float(np.mean(shimmer_values)) if shimmer_values else float("nan")
    return jitter, shimmer


def estimate_speaking_rate(audio: PreprocessedAudio) -> float:
    """Estimate syllable nuclei per second from the energy envelope.

    Args:
        audio: Preprocessed audio.

    Returns:
        Syllables per second over the analysed duration, or NaN if the
        recording is too short to judge.

    Method:
        Syllable nuclei appear as local maxima in the smoothed frame-energy
        envelope. Peaks are counted with a minimum separation
        (``VOICE_MIN_SYLLABLE_SEPARATION_MS``) and a minimum prominence
        relative to the envelope's own range.

    Assumption:
        Energy-envelope peak counting is a well-established proxy for syllable
        rate, but it is a proxy: it will merge two syllables spoken without an
        energy dip and will split one with a strong internal dip. It is used
        here because the alternative -- counting actual syllables -- would
        require phonetic decoding of the speech, which this system is
        forbidden from doing. The proxy is compared only against the same
        person's own baseline computed the same way, so the systematic bias
        cancels.
    """
    if audio.frame_rms.size < 3 or audio.duration_sec <= 0:
        return float("nan")

    envelope = audio.frame_rms.astype(np.float64)
    # Three-frame moving average, to suppress single-frame energy spikes.
    kernel = np.ones(3) / 3.0
    smoothed = np.convolve(envelope, kernel, mode="same")

    span = float(np.max(smoothed) - np.min(smoothed))
    if span <= 0:
        return float("nan")

    min_distance_frames = max(
        1,
        int(round(settings.VOICE_MIN_SYLLABLE_SEPARATION_MS / settings.VOICE_HOP_LENGTH_MS)),
    )
    peaks, _ = find_peaks(
        smoothed,
        distance=min_distance_frames,
        prominence=span * settings.VOICE_SYLLABLE_PROMINENCE_FRACTION,
    )
    return float(len(peaks)) / audio.duration_sec


def extract_features(audio: PreprocessedAudio) -> AcousticFeatures:
    """Extract all eight acoustic features from one preprocessed sample.

    Args:
        audio: Preprocessed audio.

    Returns:
        An :class:`AcousticFeatures` record. Individual fields are NaN when
        there was not enough voiced material to compute them; the caller
        checks ``is_usable`` rather than receiving a fabricated number.
    """
    f0 = estimate_f0_per_frame(audio)
    voiced_f0 = f0[np.isfinite(f0)]

    if voiced_f0.size >= MIN_VOICED_FRAMES_FOR_PITCH:
        f0_mean = float(np.mean(voiced_f0))
        f0_sd = float(np.std(voiced_f0, ddof=1)) if voiced_f0.size > 1 else 0.0
    else:
        f0_mean, f0_sd = float("nan"), float("nan")

    voiced_rms = audio.frame_rms[audio.voiced_mask]
    if voiced_rms.size:
        rms_mean = float(np.mean(voiced_rms))
        rms_sd = float(np.std(voiced_rms, ddof=1)) if voiced_rms.size > 1 else 0.0
    else:
        rms_mean, rms_sd = float("nan"), float("nan")
    rms_cv = rms_sd / rms_mean if rms_mean and rms_mean > 0 else float("nan")

    pause_ratio = (
        float(1.0 - audio.voiced_mask.mean()) if audio.voiced_mask.size else float("nan")
    )
    jitter, shimmer = compute_perturbation(audio, f0)

    return AcousticFeatures(
        f0_mean_hz=f0_mean,
        f0_sd_hz=f0_sd,
        speaking_rate_syllables_per_sec=estimate_speaking_rate(audio),
        pause_ratio=pause_ratio,
        intensity_rms_mean=rms_mean,
        intensity_rms_sd=rms_sd,
        intensity_rms_cv=rms_cv,
        jitter_local_pct=jitter,
        shimmer_local_pct=shimmer,
        voiced_frame_count=int(voiced_f0.size),
        analysed_duration_sec=audio.duration_sec,
    )
