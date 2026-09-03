"""Tests for the voice lab.

Run with:

    python -m unittest discover -s tests

They cover the parts that would silently produce wrong *results* rather than
visible errors: the WAV contract, the leak-free baseline ordering, and the
separation arithmetic. A broken login is obvious the moment you open the app; a
baseline that quietly includes the sample it is scoring is not.
"""

from __future__ import annotations

import io
import sys
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voicelab import analysis, audio  # noqa: E402
from voicelab.dsp import settings as dsp_settings  # noqa: E402

RATE = dsp_settings.VOICE_SAMPLE_RATE_HZ


def make_wav(seconds: float = 8.0, rate: int = RATE, channels: int = 1,
             width: int = 2, f0: float = 130.0) -> bytes:
    """Build a WAV of a crude voiced tone.

    Args:
        seconds: Duration.
        rate: Sample rate.
        channels: Channel count.
        width: Sample width in bytes.
        f0: Fundamental frequency.

    Returns:
        WAV file bytes.
    """
    t = np.arange(int(seconds * rate)) / rate
    # A few harmonics with an amplitude envelope, so the extractor has
    # something periodic to lock onto rather than a pure sine.
    signal = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in (1, 2, 3, 4))
    signal *= 0.4 * (1 + 0.3 * np.sin(2 * np.pi * 3.0 * t))
    pcm = (np.clip(signal / 2.2, -1, 1) * 32767).astype("<i2")

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(width)
        handle.setframerate(rate)
        handle.writeframes(pcm.tobytes())
    return buffer.getvalue()


class TestWavContract(unittest.TestCase):
    """The pipeline's input contract is refused, never repaired."""

    def test_correct_wav_decodes(self) -> None:
        samples, rate = audio.decode_wav(make_wav(4.0))
        self.assertEqual(rate, RATE)
        self.assertAlmostEqual(len(samples) / rate, 4.0, places=2)

    def test_wrong_sample_rate_is_refused_not_resampled(self) -> None:
        """Resampling changes period lengths, and jitter is a period measure."""
        with self.assertRaises(audio.BadRecording) as caught:
            audio.decode_wav(make_wav(4.0, rate=44100))
        self.assertIn("44100", str(caught.exception))

    def test_stereo_is_refused(self) -> None:
        with self.assertRaises(audio.BadRecording):
            audio.decode_wav(make_wav(4.0, channels=2))

    def test_eight_bit_is_refused(self) -> None:
        with self.assertRaises(audio.BadRecording):
            audio.decode_wav(make_wav(4.0, width=1))

    def test_non_wav_bytes_are_refused(self) -> None:
        with self.assertRaises(audio.BadRecording):
            audio.decode_wav(b"this is not a wav file at all")

    def test_normalisation_is_by_full_scale_not_peak(self) -> None:
        """Peak-normalising would destroy the intensity features."""
        quiet = make_wav(4.0)
        samples, _ = audio.decode_wav(quiet)
        # A quarter-scale signal must stay at a quarter scale, not be
        # stretched to fill [-1, 1].
        self.assertLess(float(np.max(np.abs(samples))), 0.95)

    def test_too_short_is_rejected_on_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(audio.BadRecording):
                audio.store(make_wav(1.0), "someone", "neutral", audio_dir=Path(tmp))

    def test_stored_file_lands_on_disk_and_is_measured(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stored = audio.store(make_wav(8.0), "someone", "neutral", audio_dir=Path(tmp))
            self.assertTrue((Path(tmp) / stored.filename).exists())
            self.assertEqual(stored.sample_rate, RATE)
            # The synthetic tone is periodic enough to measure; if this ever
            # fails, the extractor has changed and the lab's results with it.
            self.assertIsNotNone(stored.features, stored.error)
            self.assertIn("f0_mean_hz", stored.features)


def recording(label: str, when: str, **features: float) -> dict:
    """Build a recording row for the analysis.

    Args:
        label: 'neutral' or a case label.
        when: ISO timestamp; ordering is by this.
        **features: Feature overrides.

    Returns:
        A row shaped like the ones `app.py` passes in.
    """
    base = {
        "f0_mean_hz": 130.0,
        "f0_sd_hz": 18.0,
        "speaking_rate_syllables_per_sec": 3.5,
        "pause_ratio": 0.30,
        "intensity_rms_cv": 0.35,
        "jitter_local_pct": 0.60,
        "shimmer_local_pct": 4.0,
    }
    base.update(features)
    return {"label": label, "created_at": when, "features": base, "extract_error": None}


class TestLeakFreeBaseline(unittest.TestCase):
    """A recording is scored only against strictly earlier ones."""

    def test_baseline_consumes_the_first_n_neutrals(self) -> None:
        needed = dsp_settings.VOICE_BASELINE_MIN_SAMPLES
        recs = [recording("neutral", f"2026-01-0{i + 1}") for i in range(needed)]
        result = analysis.analyse_subject("solo", recs)
        self.assertEqual(result.baseline_count, needed)
        self.assertEqual(result.control_scores, [])

    def test_extra_neutrals_become_controls(self) -> None:
        needed = dsp_settings.VOICE_BASELINE_MIN_SAMPLES
        recs = [recording("neutral", f"2026-01-{i + 1:02d}") for i in range(needed + 2)]
        result = analysis.analyse_subject("solo", recs)
        self.assertEqual(result.baseline_count, needed)
        self.assertEqual(len(result.control_scores), 2)

    def test_a_subject_with_too_few_neutrals_is_reported_not_scored(self) -> None:
        recs = [recording("neutral", "2026-01-01"), recording("sad", "2026-01-02")]
        result = analysis.analyse_subject("solo", recs)
        self.assertIsNone(result.separation)
        self.assertTrue(any("baseline never became reliable" in s for s in result.skipped))

    def test_unmeasured_recordings_are_reported_not_dropped(self) -> None:
        recs = [recording("neutral", f"2026-01-{i + 1:02d}") for i in range(5)]
        recs.append({
            "label": "sad", "created_at": "2026-01-09",
            "features": None, "extract_error": "too little voiced material",
        })
        result = analysis.analyse_subject("solo", recs)
        self.assertTrue(any("too little voiced material" in s for s in result.skipped))

    def test_a_case_matching_baseline_scores_near_zero(self) -> None:
        """Sanity floor: identical input must not produce a deviation."""
        recs = [recording("neutral", f"2026-01-{i + 1:02d}") for i in range(5)]
        recs.append(recording("sad", "2026-01-09"))
        result = analysis.analyse_subject("solo", recs)
        self.assertTrue(result.all_case_scores)
        self.assertLess(max(result.all_case_scores), 5.0)

    def test_a_case_departing_in_the_expected_direction_scores_high(self) -> None:
        recs = [
            recording("neutral", f"2026-01-{i + 1:02d}",
                      f0_mean_hz=130 + i, jitter_local_pct=0.6 + 0.02 * i)
            for i in range(5)
        ]
        recs.append(recording(
            "sad", "2026-01-09",
            f0_mean_hz=190.0, jitter_local_pct=2.5, shimmer_local_pct=9.0,
            speaking_rate_syllables_per_sec=5.2, pause_ratio=0.10,
        ))
        result = analysis.analyse_subject("solo", recs)
        self.assertGreater(max(result.all_case_scores), 40.0)


class TestAuc(unittest.TestCase):
    """The separation metric."""

    def test_perfect_separation(self) -> None:
        self.assertEqual(analysis.auc([1, 2, 3], [4, 5, 6]), 1.0)

    def test_perfect_inversion(self) -> None:
        self.assertEqual(analysis.auc([4, 5, 6], [1, 2, 3]), 0.0)

    def test_ties_count_a_half(self) -> None:
        # Matters because a saturating signal produces real ties at 0 and 100.
        self.assertEqual(analysis.auc([5, 5], [5, 5]), 0.5)

    def test_empty_group_gives_none(self) -> None:
        self.assertIsNone(analysis.auc([], [1, 2]))
        self.assertIsNone(analysis.auc([1, 2], []))


class TestFeatureWeightsContract(unittest.TestCase):
    """Guardrails on the file people will be editing."""

    def test_weights_sum_to_one(self) -> None:
        total = sum(dsp_settings.VOICE_FEATURE_WEIGHTS.values())
        self.assertAlmostEqual(total, 1.0, places=9)

    def test_every_compared_feature_has_a_weight_and_a_direction(self) -> None:
        for name in dsp_settings.VOICE_COMPARISON_FEATURE_NAMES:
            self.assertIn(name, dsp_settings.VOICE_FEATURE_WEIGHTS)
            self.assertIn(name, dsp_settings.VOICE_FEATURE_DIRECTIONS)

    def test_directions_are_plus_or_minus_one(self) -> None:
        for name, direction in dsp_settings.VOICE_FEATURE_DIRECTIONS.items():
            self.assertIn(direction, (1, -1), f"{name} has direction {direction}")

    def test_absolute_intensity_is_never_compared(self) -> None:
        """Recording level varies with the setup, not the person."""
        for name in ("intensity_rms_mean", "intensity_rms_sd"):
            self.assertNotIn(name, dsp_settings.VOICE_COMPARISON_FEATURE_NAMES)


if __name__ == "__main__":
    unittest.main()
