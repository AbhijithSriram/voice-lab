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


def paired(label: str, when: str, session: str, easy_pause: float,
           hard_pause: float, **features: float) -> list:
    """One complete sitting: an automatic half and an effortful half.

    Args:
        label: Label both halves carry.
        when: ISO timestamp prefix; the two halves are ordered under it.
        session: Session id shared by both halves.
        easy_pause: ``pause_ratio`` for the automatic prompt.
        hard_pause: ``pause_ratio`` for the effortful prompt.
        **features: Applied to both halves.

    Returns:
        Two recording rows.
    """
    out = []
    for load, pause in (("automatic", easy_pause), ("effortful", hard_pause)):
        rec = recording(label, f"{when}-{load}", pause_ratio=pause, **features)
        rec["session_id"] = session
        rec["load"] = load
        out.append(rec)
    return out


class TestPerLoadBaselines(unittest.TestCase):
    """A baseline pools like with like, and the load is part of "like"."""

    def _neutrals(self, load: str, session_prefix: str, pause: float,
                  n: int = 4, spread: float = 0.004) -> list:
        """Neutral recordings of one load, tightly clustered."""
        out = []
        for i in range(n):
            rec = recording("neutral", f"2026-01-{i + 1:02d}-{load}",
                            pause_ratio=pause + spread * i)
            rec["session_id"] = f"{session_prefix}{i}"
            rec["load"] = load
            out.append(rec)
        return out

    def test_a_distant_load_does_not_inflate_another_loads_scale(self) -> None:
        """The 0.004-against-0.647 problem, as an assertion.

        Sustained phonation barely pauses; counting pauses constantly. Pooled,
        the IQR across both is not the subject's variability but the gap
        between two tasks, and it becomes the divisor for every later z-score.
        Kept apart, a genuine departure on one load still registers.
        """
        recs = self._neutrals("automatic", "a", 0.30)
        recs += self._neutrals("phonation", "p", 0.004)

        # Deviating *downwards*, which is the direction the absolute signal can
        # see. VOICE_FEATURE_DIRECTIONS gives pause_ratio -1, and
        # voice_stress_signal takes max(0, z), so a recording that pauses MORE
        # than baseline contributes exactly nothing to the score. See
        # TestArousalClamp below -- that is a property of the parent pipeline,
        # not of this test.
        case = recording("sad", "2026-02-01", pause_ratio=0.15)
        case["session_id"], case["load"] = "c1", "automatic"

        result = analysis.analyse_subject("p", recs + [case])
        scores = result.case_scores.get("sad", [])
        self.assertEqual(len(scores), 1)

        # 0.15 against a baseline centred near 0.306 whose spread is
        # thousandths saturates this feature, so it contributes its entire
        # weight and nothing else moved: 100 * 0.12.
        #
        # Pooled with the phonation neutrals the scale would have been the IQR
        # across 0.004 and 0.30 -- roughly fifty times larger -- the z would
        # have landed well under saturation, and the same recording would have
        # scored around a third of this. That difference is the whole reason
        # loads are kept apart.
        saturated = dsp_settings.VOICE_FEATURE_WEIGHTS["pause_ratio"] * 100.0
        self.assertAlmostEqual(scores[0], saturated, places=1)

    def test_each_load_builds_its_own_baseline(self) -> None:
        """Three neutrals of one load is a baseline; one each of three is not."""
        one_each = []
        for load, pause in (("automatic", 0.30), ("phonation", 0.004), ("effortful", 0.60)):
            rec = recording("neutral", f"2026-01-01-{load}", pause_ratio=pause)
            rec["session_id"], rec["load"] = "s1", load
            one_each.append(rec)
        result = analysis.analyse_subject("p", one_each)
        # No load reached the minimum, so nothing is scorable and each load
        # says so rather than quietly borrowing another's baseline.
        self.assertEqual(result.control_scores, [])
        self.assertEqual(len(result.skipped), 3)

    def test_untagged_recordings_are_kept_apart_not_dropped(self) -> None:
        """Free prompts and pre-migration samples get their own bucket."""
        recs = [recording("neutral", f"2026-01-0{i}") for i in range(1, 6)]
        result = analysis.analyse_subject("p", recs)
        self.assertTrue(result.control_scores)
        self.assertEqual(result.sessions_paired, 0)


class TestSensitivity(unittest.TestCase):
    """Scoring the corpus a second time without jitter and shimmer."""

    def _corpus(self) -> dict:
        recs = []
        for i in range(5):
            recs.append(recording("neutral", f"2026-01-0{i + 1}",
                                  pause_ratio=0.30 + 0.004 * i))
        # Departs in pause_ratio ONLY. Give the case an inflated jitter and the
        # dropped pair becomes a large contributor, so removing it lowers the
        # score for a reason that has nothing to do with renormalisation.
        recs.append(recording("sad", "2026-02-01", pause_ratio=0.15))
        return {"p": recs}

    def test_the_headline_run_carries_a_sensitivity_section(self) -> None:
        report = analysis.run_analysis(self._corpus())
        self.assertIn("sensitivity", report)
        self.assertEqual(report["sensitivity"]["dropped"],
                         list(analysis.SENSITIVITY_DROP))

    def test_the_reduced_run_does_not_recurse(self) -> None:
        """Without this guard the second run spawns a third, and so on."""
        report = analysis.run_analysis(self._corpus(),
                                       drop_features=analysis.SENSITIVITY_DROP)
        self.assertNotIn("sensitivity", report)

    def test_dropping_features_does_not_mutate_the_input(self) -> None:
        corpus = self._corpus()
        before = len(corpus["p"][0]["features"])
        analysis.run_analysis(corpus)
        self.assertEqual(len(corpus["p"][0]["features"]), before)

    def test_the_remaining_weight_is_renormalised_not_lost(self) -> None:
        """The mechanism the whole approach rests on.

        Nothing re-weights anything by hand. compute_voice_stress_signal skips
        features it cannot read and divides by the weight it actually used, so
        removing jitter and shimmer leaves 0.62 and the survivors scale up to
        fill it. If that stopped being true, every sensitivity number would be
        silently deflated by 38%.
        """
        kept = [n for n in dsp_settings.VOICE_COMPARISON_FEATURE_NAMES
                if n not in analysis.SENSITIVITY_DROP]
        remaining = sum(dsp_settings.VOICE_FEATURE_WEIGHTS[n] for n in kept)
        self.assertAlmostEqual(remaining, 0.62, places=6)
        # Must clear the floor below which the signal refuses to emit at all.
        self.assertGreater(remaining, 0.5)

    def test_a_pause_only_departure_scores_higher_without_the_dropped_pair(self) -> None:
        """Same departure, larger share of a smaller weighted set."""
        report = analysis.run_analysis(self._corpus())
        full = report["summary"]["case_mean"]
        reduced = report["sensitivity"]["without_dropped"]["case_mean"]
        # pause_ratio alone saturates: 100 * 0.12 / 1.00 with everything,
        # 100 * 0.12 / 0.62 once the pair is gone.
        self.assertAlmostEqual(full, 12.0, places=1)
        self.assertAlmostEqual(reduced, 19.35, places=1)


class TestArousalClamp(unittest.TestCase):
    """What the absolute signal can and cannot see, stated as tests.

    Not a complaint about the parent pipeline -- it was built for pressured
    speech and does that correctly. These exist so nobody reads a page of 0.0
    scores as "no sadness detected" when it means "sadness moves these features
    the way this signal discards".
    """

    def _neutrals(self, **fixed: float) -> list:
        return [
            recording("neutral", f"2026-01-0{i}",
                      pause_ratio=0.30 + 0.004 * i, **fixed)
            for i in range(1, 6)
        ]

    def test_more_pausing_than_baseline_scores_zero(self) -> None:
        """Sadness lengthens pauses. The signal cannot represent that."""
        case = recording("sad", "2026-02-01", pause_ratio=0.60)
        result = analysis.analyse_subject("p", self._neutrals() + [case])
        self.assertEqual(result.case_scores["sad"], [0.0])

    def test_the_feature_table_still_records_the_direction(self) -> None:
        """The score discards it; the per-feature table must not.

        This is the reason feature_z exists and is not clamped -- it is the
        only place a departure running against the strain convention survives.
        """
        case = recording("sad", "2026-02-01", pause_ratio=0.60)
        result = analysis.analyse_subject("p", self._neutrals() + [case])
        self.assertLess(result.feature_z["pause_ratio"]["case"], 0.0)


class TestLoadContrast(unittest.TestCase):
    """The paired easy/hard design."""

    WARMUP = dsp_settings.VOICE_BASELINE_MIN_SAMPLES + 2

    def _warmup(self) -> list:
        """Neutral sittings enough to give both paired loads a baseline."""
        recs = []
        for i in range(self.WARMUP):
            recs += paired("neutral", f"2026-01-{i + 1:02d}", f"w{i}",
                           0.300 + 0.004 * i, 0.360 + 0.004 * i)
        return recs

    def _sad(self, easy: float, hard: float) -> list:
        return paired("sad", "2026-03-01", "d1", easy, hard)

    def _sad_response(self, easy: float, hard: float) -> float:
        result = analysis.analyse_subject("p", self._warmup() + self._sad(easy, hard))
        return result.load_response["sad"][0]

    def test_a_complete_sitting_is_paired(self) -> None:
        result = analysis.analyse_subject("p", self._warmup() + self._sad(0.30, 0.40))
        self.assertEqual(result.sessions_paired, self.WARMUP + 1)
        self.assertEqual(result.sessions_unpaired, 0)

    def test_a_half_sitting_is_counted_not_scored(self) -> None:
        """One half is not a pair, and must not be silently treated as one."""
        half = paired("sad", "2026-03-01", "d1", 0.30, 0.40)[:1]
        result = analysis.analyse_subject("p", self._warmup() + half)
        self.assertEqual(result.sessions_paired, self.WARMUP)
        self.assertEqual(result.sessions_unpaired, 1)

    def test_halves_under_different_labels_do_not_pair(self) -> None:
        """A pair split across two moods is not a pair."""
        easy = paired("neutral", "2026-03-01", "x1", 0.30, 0.40)[0]
        hard = paired("sad", "2026-03-01", "x1", 0.30, 0.40)[1]
        result = analysis.analyse_subject("p", self._warmup() + [easy, hard])
        self.assertEqual(result.sessions_unpaired, 2)

    def test_recordings_without_a_session_are_ignored_by_the_contrast(self) -> None:
        """Samples predating the paired design must not corrupt it."""
        recs = [recording("neutral", f"2026-01-0{i}") for i in range(1, 6)]
        result = analysis.analyse_subject("p", recs)
        self.assertEqual(result.sessions_paired, 0)
        self.assertEqual(result.load_response, {})

    def test_a_wider_gap_when_sad_produces_a_positive_shift(self) -> None:
        """The whole hypothesis, in one assertion."""
        result = analysis.analyse_subject("p", self._warmup() + self._sad(0.30, 0.52))
        contrast = result.load_contrast_dict()
        self.assertIsNotNone(contrast["shift"])
        self.assertGreater(contrast["shift"], 0.0)

    def test_the_contrast_sees_a_change_in_the_gap(self) -> None:
        self.assertGreater(self._sad_response(0.30, 0.55), self._sad_response(0.30, 0.35))

    def test_a_common_shift_moves_it_far_less_than_a_gap_change(self) -> None:
        """The property the design is built on, stated as it actually holds.

        With one shared baseline the centre cancelled algebraically and a
        common shift vanished exactly. Per-load centres remove the task effect
        exactly instead, and leave ``d * (1/scale_hard - 1/scale_easy)`` of a
        common shift behind. That residue must stay small against a real change
        in the gap, or the design has lost the robustness it was chosen for.
        """
        base = self._sad_response(0.30, 0.45)
        shifted = self._sad_response(0.40, 0.55)   # +0.10 on both, gap unchanged
        widened = self._sad_response(0.30, 0.55)   # +0.10 on the hard half only
        self.assertLess(abs(shifted - base), abs(widened - base) / 3.0)

    def test_load_directions_are_not_the_strain_directions(self) -> None:
        """The bug this test exists to prevent, stated as a contract.

        VOICE_FEATURE_DIRECTIONS describes pressured speech: faster, less
        pausing. Cognitive load does the opposite. Scoring the contrast with
        the strain table inverts the sign, so a textbook positive result
        reports as a large negative one.
        """
        for name, load_dir in analysis.LOAD_DIRECTIONS.items():
            strain_dir = dsp_settings.VOICE_FEATURE_DIRECTIONS[name]
            self.assertEqual(
                load_dir, -strain_dir,
                f"{name}: load and strain directions must stay opposed",
            )

    def test_the_positive_control_warns_when_the_prompts_do_not_separate(self) -> None:
        """A hard prompt that draws no more pausing means the chain is deaf."""
        recs = []
        for i in range(self.WARMUP):
            recs += paired("neutral", f"2026-01-{i + 1:02d}", f"w{i}", 0.30, 0.30)
        report = analysis.run_analysis({"p": recs})
        self.assertTrue(report["load_contrast"]["control_check"].startswith("WARNING"))

    def test_the_positive_control_passes_on_a_real_prompt_gap(self) -> None:
        """The regression this replaced.

        The old control read the mean neutral *contrast*, which is zero by
        construction -- neutral recordings are what the per-load baselines are
        built from. It reported a false alarm against 122 real recordings whose
        raw separation was large and clean (median pause_ratio 0.26 counting
        against 0.49 naming animals). The control now reads raw values.
        """
        recs = []
        for i in range(self.WARMUP):
            recs += paired("neutral", f"2026-01-{i + 1:02d}", f"w{i}", 0.26, 0.49)
        report = analysis.run_analysis({"p": recs})
        check = report["load_contrast"]["control_check"]
        self.assertTrue(check.startswith("passed"), check)
        self.assertEqual(
            report["load_contrast"]["control_detail"]["subjects_pausing_more_on_the_hard_prompt"], 1)


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
