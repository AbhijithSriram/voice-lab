"""Does the pipeline separate a person's sad speech from their neutral speech?

That is the whole question this lab exists to answer, and this module is where
it is answered.

The method
----------
Everything below runs **once per prompt load**, not once per subject. A
baseline pools a person's neutral recordings to learn what their voice normally
does, which only means anything if those recordings are of the same task: a
sustained vowel, counting and category fluency differ in ``pause_ratio`` by two
orders of magnitude, so a baseline mixing them measures the difference between
three tasks and calls it one person's variability. Free prompts and anything
recorded before loads existed are grouped separately and scored among
themselves, never folded into a tagged baseline.

Within one load, for each subject, in strict chronological order:

1. Take their **neutral** recordings. The first
   ``VOICE_BASELINE_MIN_SAMPLES`` build the personal baseline and produce no
   score -- exactly as in the parent pipeline, where a check-in is compared
   against strictly earlier check-ins.
2. Score every remaining neutral recording against that baseline. These are the
   **controls**: same person, same instruction, so whatever they score is the
   pipeline's noise floor for that person.
3. Score every **labelled** recording (sad, or any other label) against the
   same baseline. These are the **cases**.
4. Compare. If the pipeline works, cases score higher than controls.

Why controls matter more than they look
---------------------------------------
Without them, a mean sad score of 45 means nothing: 45 against what? A person's
neutral recordings scored against their own neutral baseline should sit near
zero. If they do not -- if a subject's controls average 40 -- then the pipeline
is measuring recording conditions, mood drift, or a microphone, and any
separation you see on top of that is not evidence.

What is reported
----------------
Per subject and pooled:

- mean control score, mean case score, and the separation between them
- AUC, which asks: pick one control and one case at random, how often does the
  case score higher? 0.5 is a coin flip, 1.0 is perfect separation. It needs no
  threshold, which is what makes it the right metric while the thresholds are
  still being tuned.
- **the per-feature directional z-score table** -- for each acoustic feature,
  how far cases and controls each sat from baseline. This is the part to read
  when the headline number disappoints, because it says *which* feature moved
  and *which way*, and a feature moving the opposite way from
  ``VOICE_FEATURE_DIRECTIONS`` is a finding, not a failure.

The paired load contrast
------------------------
The protocol above measures a *level*: how far a sad recording sits from a
neutral baseline. That level is only comparable if recording conditions hold
still, and they do not -- subjects record on their own phones, in their own
rooms, at their own hours.

So each sitting also captures a pair: an **automatic** prompt (counting,
weekdays -- overlearned, no retrieval) and an **effortful** one (naming
animals -- category fluency, reliably impaired in depression). The measurement
is the gap between them::

    delta  = z(effortful) - z(automatic)          within one sitting
    signal = mean(delta | sad) - mean(delta | neutral)

Because both halves are scored against the same baseline, the baseline centre
cancels and only its scale survives. Anything that shifts both halves together
-- a different phone, a noisier room, tiredness, caffeine, the subject's own
habitual speaking rate, their language -- cancels with it. A difference is a
far more robust thing to measure than a level when conditions cannot be
controlled, and here they cannot.

It doubles as a **positive control**, but the control must be read off **raw**
feature values, not off this contrast. Neutral recordings are what the per-load
baselines are built from, so their z-scores centre on zero however the prompts
behave -- a control computed from them can only ever fail. ``_positive_control``
therefore compares each subject's median raw ``pause_ratio`` on the hard prompt
against the easy one, and asks whether the larger effect, whose existence is not
in question, is visible at all. If it is not, nothing subtler the pipeline
reports about mood can be believed.

The contrast covers rate and pause structure (0.30 of the feature weight). It
says nothing about jitter and shimmer (0.38), which are laryngeal rather than
cognitive -- that is what the sustained-vowel prompt is for, and why it is not
part of the pair.

A caution about what a good result would mean
---------------------------------------------
Subjects are asked to *speak as they would when sad*. That is performed
affect, not felt affect. A pipeline that separates performed sadness has shown
it can detect the acoustic correlates people produce when portraying sadness,
which is a real and useful result -- and is not the same as detecting sadness.
Do not let the number outgrow the design.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from voicelab.dsp import settings as dsp_settings
from voicelab.dsp.voice_baseline import (
    VoiceBaseline,
    build_baseline,
    update_baseline,
)
from voicelab.dsp.voice_stress_signal import compute_voice_stress_signal

# The label that builds the baseline. Everything else is a case.
BASELINE_LABEL = "neutral"

# The two prompt loads the contrast is built from. 'phonation' is deliberately
# not one of them: a sustained vowel carries jitter and shimmer, which are
# laryngeal rather than cognitive and do not belong on a load axis.
LOAD_EASY = "automatic"
LOAD_HARD = "effortful"

# Where free prompts and pre-migration recordings go. They get a baseline among
# themselves if there are enough, and never contaminate a tagged one.
UNTAGGED_LOAD = "untagged"

# How each feature moves when a task gets cognitively harder.
#
# These are NOT ``VOICE_FEATURE_DIRECTIONS``, and reusing those would silently
# invert the result. That table encodes *pressured* speech -- strain makes you
# faster (speaking_rate +1) and makes you pause less (pause_ratio -1).
# Cognitive load does the opposite: retrieval under load lengthens pauses and
# slows delivery. A load contrast scored with the strain directions reports a
# large negative number for a textbook positive result.
#
# Only the two features whose load direction is actually established are
# scored. f0 and the intensity coefficient of variation move under load too,
# but not with a sign that can be asserted up front, and jitter and shimmer are
# laryngeal rather than cognitive -- the sustained vowel carries those. Every
# feature's raw delta is still reported in the table; this map only decides
# what the headline number is built from.
LOAD_DIRECTIONS: Dict[str, int] = {
    "pause_ratio": +1,                       # load lengthens pauses
    "speaking_rate_syllables_per_sec": -1,   # load slows delivery
}

# Share of the *load-relevant* weight a sitting must supply before its contrast
# is scored. Measured against the weight of LOAD_DIRECTIONS rather than the
# whole feature set, which would be unreachable by construction.
MIN_CONTRAST_WEIGHT_FRACTION = 0.5

# Dropped in the sensitivity run. Not a claim that these features are useless --
# a claim that on THIS corpus they are not measuring what they name. Across 41
# sustained-vowel recordings the median jitter was 8.03% with a maximum of
# 25.99%, against a clinical norm under about 1%; 76% sat above 3%. Values that
# high are not eleven damaged larynxes, they are a period detector failing. At
# 16 kHz one sample is 62.5 us, so for a 200 Hz voice 1% jitter is a period
# difference smaller than a single sample -- the quantisation floor alone lands
# above the clinical threshold before any noise is added.
#
# Together these carry 0.38 of the feature weight, so "what are they worth here"
# is worth answering before anyone re-records a corpus at 44.1 kHz.
SENSITIVITY_DROP: Tuple[str, ...] = ("jitter_local_pct", "shimmer_local_pct")


@dataclass
class SubjectResult:
    """One subject's outcome.

    Attributes:
        username: Who.
        baseline_count: Recordings consumed building the baseline.
        control_scores: Scores for held-out neutral recordings.
        case_scores: Scores for labelled recordings, keyed by label.
        feature_z: Mean directional z per feature, for controls and cases.
        skipped: Recordings that could not be scored, with the reason.
    """

    username: str
    baseline_count: int = 0
    control_scores: List[float] = field(default_factory=list)
    case_scores: Dict[str, List[float]] = field(default_factory=dict)
    feature_z: Dict[str, Dict[str, float]] = field(default_factory=dict)
    skipped: List[str] = field(default_factory=list)
    # Paired-load results, keyed by label: one load-response value per complete
    # sitting, and the per-feature deltas behind them.
    load_response: Dict[str, List[float]] = field(default_factory=dict)
    load_feature_delta: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)
    # Median RAW pause_ratio per load over this subject's neutral recordings.
    # The positive control reads this, not the z-scores: see run_analysis.
    load_raw_pause: Dict[str, float] = field(default_factory=dict)
    sessions_paired: int = 0
    sessions_unpaired: int = 0

    @property
    def all_case_scores(self) -> List[float]:
        """Every case score, across all non-neutral labels."""
        return [s for scores in self.case_scores.values() for s in scores]

    @property
    def separation(self) -> Optional[float]:
        """Mean case score minus mean control score, or None if either is empty."""
        cases, controls = self.all_case_scores, self.control_scores
        if not cases or not controls:
            return None
        return float(np.mean(cases) - np.mean(controls))

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable form."""
        cases, controls = self.all_case_scores, self.control_scores
        return {
            "username": self.username,
            "baseline_count": self.baseline_count,
            "control_n": len(controls),
            "control_mean": round(float(np.mean(controls)), 2) if controls else None,
            "case_n": len(cases),
            "case_mean": round(float(np.mean(cases)), 2) if cases else None,
            "separation": round(self.separation, 2) if self.separation is not None else None,
            "auc": auc(controls, cases),
            "by_label": {
                label: {
                    "n": len(scores),
                    "mean": round(float(np.mean(scores)), 2) if scores else None,
                }
                for label, scores in sorted(self.case_scores.items())
            },
            "feature_z": self.feature_z,
            "load_contrast": self.load_contrast_dict(),
            "skipped": self.skipped,
        }

    def load_contrast_dict(self) -> Dict[str, Any]:
        """The paired-load result for this subject.

        Returns:
            Per-label mean load response, the neutral-to-case shift, and the
            AUC separating individual sittings. ``shift`` is the headline:
            how much further the hard prompt sat from the easy one when the
            subject was sad than when they were neutral.
        """
        neutral = self.load_response.get(BASELINE_LABEL, [])
        cases = [v for label, vals in self.load_response.items()
                 if label != BASELINE_LABEL for v in vals]
        return {
            "sessions_paired": self.sessions_paired,
            "sessions_unpaired": self.sessions_unpaired,
            "by_label": {
                label: {"n": len(vals), "mean": round(float(np.mean(vals)), 3)}
                for label, vals in sorted(self.load_response.items())
            },
            "neutral_mean": round(float(np.mean(neutral)), 3) if neutral else None,
            "case_mean": round(float(np.mean(cases)), 3) if cases else None,
            "shift": round(float(np.mean(cases) - np.mean(neutral)), 3)
            if neutral and cases else None,
            "auc": auc(neutral, cases),
        }


def auc(controls: Sequence[float], cases: Sequence[float]) -> Optional[float]:
    """Probability that a random case outranks a random control.

    Args:
        controls: Control scores.
        cases: Case scores.

    Returns:
        The AUC in [0, 1], or None when either group is empty. Ties count a
        half, which is the standard convention and matters here because a
        saturating signal produces real ties at 0 and at 100.

    Note:
        Computed directly rather than via sklearn, because the counts in this
        lab are small enough that the pairwise definition is both exact and
        clearer than a curve.
    """
    if not controls or not cases:
        return None
    wins = 0.0
    for case in cases:
        for control in controls:
            if case > control:
                wins += 1.0
            elif case == control:
                wins += 0.5
    return round(wins / (len(cases) * len(controls)), 3)


def _directional_z(
    features: Dict[str, float], baseline: VoiceBaseline
) -> Dict[str, float]:
    """Per-feature directional z-score against a baseline.

    Args:
        features: The comparison feature vector for one recording.
        baseline: The subject's baseline.

    Returns:
        Mapping of feature name to ``direction * (value - centre) / scale``.
        Positive means "moved the way the settings expect under strain".
        Unlike the signal itself this is **not** clamped at zero, because a
        feature moving the wrong way is precisely what this table is for.
    """
    out: Dict[str, float] = {}
    for name in dsp_settings.VOICE_COMPARISON_FEATURE_NAMES:
        value = features.get(name)
        centre = baseline.centre.get(name)
        scale = baseline.scale.get(name)
        if value is None or centre is None or scale is None:
            continue
        if not math.isfinite(value) or not math.isfinite(centre) or not scale:
            continue
        direction = dsp_settings.VOICE_FEATURE_DIRECTIONS.get(name, 1)
        out[name] = float(direction * (value - centre) / scale)
    return out


def _mean_z(
    recordings: Sequence[Dict[str, Any]], baseline: VoiceBaseline
) -> Dict[str, float]:
    """Mean directional z per feature over several recordings.

    Args:
        recordings: Recordings sharing a load level within one sitting.
        baseline: The subject's baseline.

    Returns:
        Feature name to mean directional z. A sitting normally holds one
        recording per load, but a subject who records the easy prompt twice
        should not have the second one ignored.
    """
    stacked: Dict[str, List[float]] = {}
    for rec in recordings:
        for name, value in _undirected_z(_comparison_vector(rec["features"]), baseline).items():
            stacked.setdefault(name, []).append(value)
    return {name: float(np.mean(vals)) for name, vals in stacked.items() if vals}


def _load_response(delta: Dict[str, float]) -> Optional[float]:
    """Collapse a per-feature load delta to one number.

    Args:
        delta: Feature name to (hard z - easy z).

    Returns:
        The weighted mean delta over :data:`LOAD_DIRECTIONS`, signed so that
        positive means "the hard prompt moved this the way cognitive load
        moves it", or None when too little of that weight was measurable.

    Note:
        Weighted by ``VOICE_FEATURE_WEIGHTS`` so pause ratio and speaking rate
        keep their relative standing from the rest of the pipeline, but signed
        by :data:`LOAD_DIRECTIONS` rather than ``VOICE_FEATURE_DIRECTIONS``,
        which describe a different phenomenon in the opposite direction.
    """
    possible = sum(dsp_settings.VOICE_FEATURE_WEIGHTS.get(n, 0.0) for n in LOAD_DIRECTIONS)
    total = 0.0
    available = 0.0
    for name, direction in LOAD_DIRECTIONS.items():
        value = delta.get(name)
        weight = dsp_settings.VOICE_FEATURE_WEIGHTS.get(name)
        if value is None or weight is None or not math.isfinite(value):
            continue
        total += weight * direction * value
        available += weight
    if not possible or available < MIN_CONTRAST_WEIGHT_FRACTION * possible:
        return None
    return float(total / available)


def _contrast_sessions(
    recordings: Sequence[Dict[str, Any]],
    baselines: Dict[str, VoiceBaseline],
    result: SubjectResult,
) -> None:
    """Score every complete sitting's easy/hard contrast onto ``result``.

    Args:
        recordings: One subject's usable recordings.
        baselines: Reliable baseline per load. Both halves' loads must be
            present or the sitting cannot be scored.
        result: Mutated in place.

    Note:
        The measurement is ``z_hard - z_easy`` within one sitting, where each
        half is scored against **its own load's** baseline::

            (hard - centre_hard) / scale_hard  -  (easy - centre_easy) / scale_easy

        Each z therefore reads "how unusual was this recording for this task,
        for this person", and the difference reads "did the hard task fall
        further from its own norm today than the easy one did from its own".

        This is a deliberate trade against the earlier single-baseline form,
        where the shared centre cancelled algebraically and any nuisance
        shifting both halves equally vanished exactly. With per-load centres
        that cancellation becomes approximate -- a common shift ``d`` leaves
        ``d * (1/scale_hard - 1/scale_easy)`` behind. It is still the better
        deal, because the task effect it now removes exactly is far the larger
        term: counting and category fluency differ in ``pause_ratio`` by two
        orders of magnitude, while the two scales are of similar size, so what
        is left of the nuisance is second order.

        A sitting counts only when both halves carry the same label. A pair
        split across two moods is not a pair.
    """
    sessions: Dict[Tuple[str, str], Dict[str, List[Dict[str, Any]]]] = {}
    for rec in recordings:
        session_id = rec.get("session_id")
        load = rec.get("load")
        if not session_id or load not in (LOAD_EASY, LOAD_HARD):
            continue
        sessions.setdefault((str(session_id), str(rec["label"])), {}).setdefault(load, []).append(rec)

    for (_session_id, label), halves in sorted(sessions.items()):
        easy, hard = halves.get(LOAD_EASY), halves.get(LOAD_HARD)
        if not easy or not hard:
            result.sessions_unpaired += 1
            continue

        baseline_easy = baselines.get(LOAD_EASY)
        baseline_hard = baselines.get(LOAD_HARD)
        if baseline_easy is None or baseline_hard is None:
            result.sessions_unpaired += 1
            continue

        z_easy = _mean_z(easy, baseline_easy)
        z_hard = _mean_z(hard, baseline_hard)
        delta = {
            name: z_hard[name] - z_easy[name]
            for name in z_hard
            if name in z_easy
        }
        response = _load_response(delta)
        if response is None:
            result.sessions_unpaired += 1
            continue

        result.sessions_paired += 1
        result.load_response.setdefault(label, []).append(round(response, 3))
        for name, value in delta.items():
            result.load_feature_delta.setdefault(name, {}).setdefault(label, []).append(value)


def _undirected_z(
    features: Dict[str, float], baseline: VoiceBaseline
) -> Dict[str, float]:
    """Per-feature z against a baseline, with no direction applied.

    Args:
        features: The comparison feature vector for one recording.
        baseline: The subject's baseline.

    Returns:
        Mapping of feature name to ``(value - centre) / scale``.

    Note:
        The contrast needs raw signs. Applying a direction here would bake the
        strain convention into a difference that is about cognitive load, and
        the per-feature table would then report a feature moving "the wrong
        way" when it had done exactly what load predicts.
    """
    out: Dict[str, float] = {}
    for name in dsp_settings.VOICE_COMPARISON_FEATURE_NAMES:
        value = features.get(name)
        centre = baseline.centre.get(name)
        scale = baseline.scale.get(name)
        if value is None or centre is None or scale is None:
            continue
        if not math.isfinite(value) or not math.isfinite(centre) or not scale:
            continue
        out[name] = float((value - centre) / scale)
    return out


def _comparison_vector(features: Dict[str, float]) -> Dict[str, float]:
    """Reduce a stored feature dict to the comparison subset.

    Args:
        features: All extracted features, as stored.

    Returns:
        Only the scale-invariant features the baseline comparison uses.
    """
    return {
        name: features[name]
        for name in dsp_settings.VOICE_COMPARISON_FEATURE_NAMES
        if name in features
    }


def _protocol_for_load(
    username: str,
    recordings: Sequence[Dict[str, Any]],
    result: SubjectResult,
    load: str,
    control_z: List[Dict[str, float]],
    case_z: List[Dict[str, float]],
) -> Optional[VoiceBaseline]:
    """Run the leak-free protocol within a single load.

    Args:
        username: Who.
        recordings: This subject's recordings **of one load**, any order.
        result: Mutated in place with scores and skips.
        load: The load these recordings share, for skip messages.
        control_z: Accumulator for control directional z-scores.
        case_z: Accumulator for case directional z-scores.

    Returns:
        The settled baseline for this load, or None when too few neutrals.

    Note:
        One baseline per load, rather than one per subject. A subject's first
        three neutrals in the old scheme could be a sustained vowel, counting
        and category fluency, whose ``pause_ratio`` values differ by two orders
        of magnitude -- 0.004 against 0.647 in the first real sitting recorded
        on this server. The IQR across that is not this person's variability,
        it is the difference between three tasks, and every later z-score was
        divided by it. Comparing like with like is the entire point of a
        personal baseline; the load is part of "like".
    """
    ordered = sorted(recordings, key=lambda r: str(r.get("created_at", "")))
    neutral = [r for r in ordered if r["label"] == BASELINE_LABEL]
    cases = [r for r in ordered if r["label"] != BASELINE_LABEL]

    baseline = VoiceBaseline(pseudonym_id=username)
    history: List[Dict[str, float]] = []

    for rec in neutral:
        vector = _comparison_vector(rec["features"])
        # Score first, then absorb: compared only against strictly earlier ones.
        if baseline.is_reliable:
            outcome = compute_voice_stress_signal(vector, baseline)
            if outcome.is_reliable:
                result.control_scores.append(round(float(outcome.signal), 2))
                control_z.append(_directional_z(vector, baseline))
        else:
            result.baseline_count += 1

        history.append(vector)
        stamp = str(rec.get("created_at") or "")
        if len(history) <= dsp_settings.VOICE_BASELINE_MIN_SAMPLES:
            baseline = build_baseline(username, history, last_updated=stamp)
        else:
            baseline = update_baseline(baseline, vector, sample_date=stamp)

    if not baseline.is_reliable:
        result.skipped.append(
            f"load {load!r}: baseline never became reliable "
            f"({baseline.sample_count} usable neutral recording(s), "
            f"{dsp_settings.VOICE_BASELINE_MIN_SAMPLES} needed) - "
            f"{len(cases)} case recording(s) at this load unscored"
        )
        return None

    for rec in cases:
        vector = _comparison_vector(rec["features"])
        outcome = compute_voice_stress_signal(vector, baseline)
        if not outcome.is_reliable:
            result.skipped.append(
                f"{rec['label']} {rec.get('created_at', '')} ({load}): {outcome.reason}"
            )
            continue
        result.case_scores.setdefault(rec["label"], []).append(
            round(float(outcome.signal), 2)
        )
        case_z.append(_directional_z(vector, baseline))

    return baseline


def analyse_subject(username: str, recordings: Sequence[Dict[str, Any]]) -> SubjectResult:
    """Run the protocol for one subject, one load at a time.

    Args:
        username: Who.
        recordings: That subject's recordings, each with ``label``,
            ``created_at``, ``load`` and a parsed ``features`` dict. Order does
            not matter; this function sorts.

    Returns:
        A :class:`SubjectResult`.

    Note:
        Free prompts and anything recorded before the paired design carry no
        load. They are grouped under :data:`UNTAGGED_LOAD` and scored among
        themselves if there are enough of them, rather than being dropped or --
        worse -- folded into the baseline of a task they do not resemble.
    """
    result = SubjectResult(username=username)

    usable = []
    for rec in recordings:
        if not rec.get("features"):
            result.skipped.append(
                f"{rec.get('label', '?')} {rec.get('created_at', '')}: "
                f"{rec.get('extract_error') or 'no features extracted'}"
            )
            continue
        usable.append(rec)

    by_load: Dict[str, List[Dict[str, Any]]] = {}
    for rec in usable:
        by_load.setdefault(str(rec.get("load") or UNTAGGED_LOAD), []).append(rec)

    # Raw medians for the positive control, before any baseline touches them.
    for load, recs in by_load.items():
        pauses = [
            r["features"]["pause_ratio"] for r in recs
            if r["label"] == BASELINE_LABEL and r["features"].get("pause_ratio") is not None
        ]
        if pauses:
            result.load_raw_pause[load] = float(np.median(pauses))

    control_z: List[Dict[str, float]] = []
    case_z: List[Dict[str, float]] = []
    baselines: Dict[str, VoiceBaseline] = {}
    for load, recs in sorted(by_load.items()):
        baseline = _protocol_for_load(username, recs, result, load, control_z, case_z)
        if baseline is not None:
            baselines[load] = baseline

    _contrast_sessions(usable, baselines, result)

    result.feature_z = {
        name: {
            "control": round(float(np.mean([z[name] for z in control_z if name in z])), 3)
            if any(name in z for z in control_z) else None,
            "case": round(float(np.mean([z[name] for z in case_z if name in z])), 3)
            if any(name in z for z in case_z) else None,
        }
        for name in dsp_settings.VOICE_COMPARISON_FEATURE_NAMES
    }
    return result


def _drop_features(
    by_subject: Dict[str, List[Dict[str, Any]]], drop: Sequence[str]
) -> Dict[str, List[Dict[str, Any]]]:
    """Copy the corpus with some features removed from every recording.

    Args:
        by_subject: Recordings grouped by username.
        drop: Feature names to remove.

    Returns:
        A new mapping; the input is not modified.

    Note:
        Removing a feature from the vector is the whole mechanism, and it is why
        this costs no change to the vendored DSP.
        ``compute_voice_stress_signal`` skips any feature it cannot read and
        divides by ``available_weight`` at the end, so whatever remains is
        renormalised over its own weight automatically. Dropping jitter and
        shimmer leaves 0.62, comfortably above the 0.5 floor below which the
        signal refuses to emit a number at all.
    """
    dropped = set(drop)
    out: Dict[str, List[Dict[str, Any]]] = {}
    for name, recordings in by_subject.items():
        rows = []
        for rec in recordings:
            features = rec.get("features")
            rows.append({
                **rec,
                "features": (
                    {k: v for k, v in features.items() if k not in dropped}
                    if features else features
                ),
            })
        out[name] = rows
    return out


def _sensitivity(
    by_subject: Dict[str, List[Dict[str, Any]]], full: Dict[str, Any]
) -> Dict[str, Any]:
    """Score the corpus again without :data:`SENSITIVITY_DROP` and compare.

    Args:
        by_subject: Recordings grouped by username.
        full: The report produced from the complete feature set.

    Returns:
        Both sets of headline figures, per-subject AUCs, and a plain verdict.

    Note:
        The load contrast is unaffected and deliberately not reported here:
        :data:`LOAD_DIRECTIONS` covers only ``pause_ratio`` and
        ``speaking_rate``, so removing jitter and shimmer cannot move it. This
        compares the absolute score, which is where their 0.38 of the weight
        lands.
    """
    reduced = run_analysis(by_subject, drop_features=SENSITIVITY_DROP)

    def headline(report: Dict[str, Any]) -> Dict[str, Any]:
        summary = report["summary"]
        return {
            key: summary[key] for key in
            ("subjects_scored", "control_n", "case_n", "control_mean",
             "case_mean", "separation", "pooled_auc", "mean_within_subject_auc")
        }

    with_all, without = headline(full), headline(reduced)
    a, b = with_all["mean_within_subject_auc"], without["mean_within_subject_auc"]
    if a is None or b is None:
        verdict = "not enough scored subjects to compare"
    elif b > a + 0.05:
        verdict = (
            f"dropping them separates BETTER ({a} -> {b}). On this corpus they "
            f"are costing the score, which is what their measured values predict."
        )
    elif a > b + 0.05:
        verdict = (
            f"they are contributing ({a} with, {b} without) despite the "
            f"implausible values. Understand why before removing them."
        )
    else:
        verdict = (
            f"no material difference ({a} with, {b} without). They carry 0.38 of "
            f"the weight and change nothing."
        )

    return {
        "dropped": list(SENSITIVITY_DROP),
        "reason": (
            "Median sustained-vowel jitter on this corpus sits far above the "
            "clinical norm, and 16 kHz cannot resolve jitter to that precision "
            "in any case. This asks what the two features are worth here."
        ),
        "with_all_features": with_all,
        "without_dropped": without,
        "per_subject": [
            {"username": x["username"], "auc_with": x["auc"], "auc_without": y["auc"]}
            for x, y in zip(full["subjects"], reduced["subjects"])
            if x["auc"] is not None or y["auc"] is not None
        ],
        "verdict": verdict,
    }


def run_analysis(
    by_subject: Dict[str, List[Dict[str, Any]]],
    drop_features: Sequence[str] = (),
) -> Dict[str, Any]:
    """Analyse every subject and pool the result.

    Args:
        by_subject: Recordings grouped by username.
        drop_features: Features to exclude from scoring entirely. Used by the
            sensitivity run; empty for the headline analysis.

    Returns:
        A JSON-serialisable report: per-subject rows, a pooled summary, the
        per-feature table, a copy of the settings used, and -- on the headline
        run only -- a ``sensitivity`` section scoring the corpus a second time
        without :data:`SENSITIVITY_DROP`.
    """
    if drop_features:
        by_subject = _drop_features(by_subject, drop_features)
    subjects = [analyse_subject(name, recs) for name, recs in sorted(by_subject.items())]
    scored = [s for s in subjects if s.control_scores and s.all_case_scores]

    pooled_controls = [x for s in scored for x in s.control_scores]
    pooled_cases = [x for s in scored for x in s.all_case_scores]

    # Per-subject AUCs averaged, alongside the pooled AUC. They answer
    # different questions: pooled AUC can be flattered by between-subject
    # variation in overall score level, while the mean of within-subject AUCs
    # cannot. When they disagree, the within-subject figure is the honest one,
    # because the pipeline only ever compares a person to themselves.
    per_subject_aucs = [
        a for a in (auc(s.control_scores, s.all_case_scores) for s in scored)
        if a is not None
    ]

    feature_rows = []
    for name in dsp_settings.VOICE_COMPARISON_FEATURE_NAMES:
        controls = [s.feature_z[name]["control"] for s in scored
                    if s.feature_z.get(name, {}).get("control") is not None]
        cases = [s.feature_z[name]["case"] for s in scored
                 if s.feature_z.get(name, {}).get("case") is not None]
        feature_rows.append({
            "feature": name,
            "direction": dsp_settings.VOICE_FEATURE_DIRECTIONS.get(name),
            "weight": dsp_settings.VOICE_FEATURE_WEIGHTS.get(name),
            "control_z": round(float(np.mean(controls)), 3) if controls else None,
            "case_z": round(float(np.mean(cases)), 3) if cases else None,
            "shift": round(float(np.mean(cases) - np.mean(controls)), 3)
            if controls and cases else None,
        })
    # Largest positive shift first: the features actually carrying the result.
    feature_rows.sort(key=lambda r: (r["shift"] is None, -(r["shift"] or 0)))

    # --- paired load contrast, pooled -----------------------------------
    contrast_neutral = [v for s in subjects for v in s.load_response.get(BASELINE_LABEL, [])]
    contrast_cases = [
        v for s in subjects for label, vals in s.load_response.items()
        if label != BASELINE_LABEL for v in vals
    ]
    contrast_subject_aucs = [
        a for a in (
            auc(
                s.load_response.get(BASELINE_LABEL, []),
                [v for lbl, vals in s.load_response.items() if lbl != BASELINE_LABEL for v in vals],
            )
            for s in subjects
        ) if a is not None
    ]
    contrast_rows = []
    for name in dsp_settings.VOICE_COMPARISON_FEATURE_NAMES:
        per_label = {}
        for s in subjects:
            for label, vals in s.load_feature_delta.get(name, {}).items():
                per_label.setdefault(label, []).extend(vals)
        neutral = per_label.get(BASELINE_LABEL, [])
        cases = [v for lbl, vals in per_label.items() if lbl != BASELINE_LABEL for v in vals]
        contrast_rows.append({
            "feature": name,
            "weight": dsp_settings.VOICE_FEATURE_WEIGHTS.get(name),
            "neutral_delta": round(float(np.mean(neutral)), 3) if neutral else None,
            "case_delta": round(float(np.mean(cases)), 3) if cases else None,
            "shift": round(float(np.mean(cases) - np.mean(neutral)), 3)
            if neutral and cases else None,
        })
    contrast_rows.sort(key=lambda r: (r["shift"] is None, -(r["shift"] or 0)))

    report = {
        "subjects": [s.to_dict() for s in subjects],
        "load_contrast": {
            "sessions_paired": sum(s.sessions_paired for s in subjects),
            "sessions_unpaired": sum(s.sessions_unpaired for s in subjects),
            "neutral_n": len(contrast_neutral),
            "case_n": len(contrast_cases),
            "neutral_mean": round(float(np.mean(contrast_neutral)), 3) if contrast_neutral else None,
            "case_mean": round(float(np.mean(contrast_cases)), 3) if contrast_cases else None,
            "shift": round(float(np.mean(contrast_cases) - np.mean(contrast_neutral)), 3)
            if contrast_neutral and contrast_cases else None,
            "pooled_auc": auc(contrast_neutral, contrast_cases),
            "mean_within_subject_auc": round(float(np.mean(contrast_subject_aucs)), 3)
            if contrast_subject_aucs else None,
            "features": contrast_rows,
            **_positive_control(subjects),
        },
        "summary": {
            "subjects_total": len(subjects),
            "subjects_scored": len(scored),
            "control_n": len(pooled_controls),
            "case_n": len(pooled_cases),
            "control_mean": round(float(np.mean(pooled_controls)), 2) if pooled_controls else None,
            "case_mean": round(float(np.mean(pooled_cases)), 2) if pooled_cases else None,
            "separation": round(float(np.mean(pooled_cases) - np.mean(pooled_controls)), 2)
            if pooled_controls and pooled_cases else None,
            "pooled_auc": auc(pooled_controls, pooled_cases),
            "mean_within_subject_auc": round(float(np.mean(per_subject_aucs)), 3)
            if per_subject_aucs else None,
        },
        "features": feature_rows,
        "settings": settings_snapshot(),
    }
    # Only the headline run spawns the comparison; the reduced run must not
    # recurse into another one.
    if not drop_features:
        report["sensitivity"] = _sensitivity(by_subject, report)
    return report


def _positive_control(subjects: Sequence[SubjectResult]) -> Dict[str, Any]:
    """Check the chain resolves cognitive load at all, on raw values.

    Args:
        subjects: Every analysed subject.

    Returns:
        ``control_check`` and ``control_detail`` for the report.

    Note:
        This reads **raw** median ``pause_ratio`` per load, not the contrast's
        z-scores. An earlier version checked that the mean neutral contrast was
        far from zero, which was wrong by construction: neutral recordings are
        what the per-load baselines are built from, so their z-scores centre on
        zero whatever the prompts do, and the check could only ever fail. It
        reported a false alarm against data whose raw separation was in fact
        large and clean.

        The real question is whether naming animals produces more pausing than
        counting, per subject, before any normalisation touches it.
    """
    agree = 0
    total = 0
    gaps: List[float] = []
    for subject in subjects:
        easy = subject.load_raw_pause.get(LOAD_EASY)
        hard = subject.load_raw_pause.get(LOAD_HARD)
        if easy is None or hard is None:
            continue
        total += 1
        gaps.append(hard - easy)
        if hard > easy:
            agree += 1

    if not total:
        return {"control_check": "no subject has both prompts yet", "control_detail": None}

    detail = {
        "subjects": total,
        "subjects_pausing_more_on_the_hard_prompt": agree,
        "median_raw_gap": round(float(np.median(gaps)), 3),
    }
    if agree * 2 >= total and float(np.median(gaps)) > 0.02:
        return {
            "control_check": (
                f"passed - the hard prompt draws more pausing than the easy one in "
                f"{agree}/{total} subjects (median gap {np.median(gaps):+.3f})"
            ),
            "control_detail": detail,
        }
    return {
        "control_check": (
            f"WARNING: the hard prompt does not reliably draw more pausing than the "
            f"easy one ({agree}/{total} subjects, median gap {np.median(gaps):+.3f}). "
            f"The chain is not resolving a large known effect, so nothing subtler "
            f"it reports can be believed."
        ),
        "control_detail": detail,
    }


def settings_snapshot() -> Dict[str, Any]:
    """Capture the tunable settings this run used.

    Returns:
        The subset of ``voicelab.dsp.settings`` that changes results, so a
        stored analysis stays interpretable after the settings are edited.
    """
    return {
        "VOICE_BASELINE_MIN_SAMPLES": dsp_settings.VOICE_BASELINE_MIN_SAMPLES,
        "VOICE_DEVIATION_SATURATION_SD": dsp_settings.VOICE_DEVIATION_SATURATION_SD,
        "VOICE_BASELINE_EMA_ALPHA": dsp_settings.VOICE_BASELINE_EMA_ALPHA,
        "VOICE_F0_MIN_HZ": dsp_settings.VOICE_F0_MIN_HZ,
        "VOICE_F0_MAX_HZ": dsp_settings.VOICE_F0_MAX_HZ,
        "VOICE_AUTOCORR_VOICING_THRESHOLD": dsp_settings.VOICE_AUTOCORR_VOICING_THRESHOLD,
        "VOICE_SILENCE_RMS_FRACTION": dsp_settings.VOICE_SILENCE_RMS_FRACTION,
        "VOICE_FEATURE_DIRECTIONS": dict(dsp_settings.VOICE_FEATURE_DIRECTIONS),
        "VOICE_FEATURE_WEIGHTS": dict(dsp_settings.VOICE_FEATURE_WEIGHTS),
    }


def summarise(report: Dict[str, Any]) -> str:
    """One line describing a run, for the analysis history list.

    Args:
        report: Output of :func:`run_analysis`.

    Returns:
        A short summary string.
    """
    s = report["summary"]
    contrast = report.get("load_contrast", {})
    if s["separation"] is None:
        return f"{s['subjects_total']} subject(s), nothing scorable yet"
    line = (
        f"{s['subjects_scored']}/{s['subjects_total']} subjects scored - "
        f"separation {s['separation']:+.1f} pts, "
        f"within-subject AUC {s['mean_within_subject_auc']}"
    )
    if contrast.get("shift") is not None:
        line += (
            f"; load contrast {contrast['shift']:+.2f} over "
            f"{contrast['sessions_paired']} paired sitting(s)"
        )
    return line
