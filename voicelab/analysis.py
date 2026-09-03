"""Does the pipeline separate a person's sad speech from their neutral speech?

That is the whole question this lab exists to answer, and this module is where
it is answered.

The method
----------
For each subject, in strict chronological order:

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
            "skipped": self.skipped,
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


def analyse_subject(username: str, recordings: Sequence[Dict[str, Any]]) -> SubjectResult:
    """Run the leak-free protocol for one subject.

    Args:
        username: Who.
        recordings: That subject's recordings, each with ``label``,
            ``created_at`` and a parsed ``features`` dict. Order does not
            matter; this function sorts.

    Returns:
        A :class:`SubjectResult`.
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

    usable.sort(key=lambda r: str(r.get("created_at", "")))

    neutral = [r for r in usable if r["label"] == BASELINE_LABEL]
    cases = [r for r in usable if r["label"] != BASELINE_LABEL]

    baseline = VoiceBaseline(pseudonym_id=username)
    history: List[Dict[str, float]] = []
    control_z: List[Dict[str, float]] = []
    case_z: List[Dict[str, float]] = []

    # Neutral recordings, in order: the first few build the baseline, the rest
    # are scored against it. The build/update split below is copied from
    # ``voice_pipeline/pipeline.py`` in the parent project so that a result
    # measured here is a result about that pipeline -- a batch median over the
    # first N vectors, then an EMA update per vector after that.
    for rec in neutral:
        vector = _comparison_vector(rec["features"])

        # Score first, then absorb: a recording is compared only against
        # strictly earlier ones.
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
            f"baseline never became reliable: {baseline.sample_count} usable "
            f"neutral recording(s), {dsp_settings.VOICE_BASELINE_MIN_SAMPLES} needed"
        )
        return result

    for rec in cases:
        vector = _comparison_vector(rec["features"])
        outcome = compute_voice_stress_signal(vector, baseline)
        if not outcome.is_reliable:
            result.skipped.append(
                f"{rec['label']} {rec.get('created_at', '')}: {outcome.reason}"
            )
            continue
        result.case_scores.setdefault(rec["label"], []).append(
            round(float(outcome.signal), 2)
        )
        case_z.append(_directional_z(vector, baseline))

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


def run_analysis(by_subject: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Analyse every subject and pool the result.

    Args:
        by_subject: Recordings grouped by username.

    Returns:
        A JSON-serialisable report: per-subject rows, a pooled summary, the
        per-feature table, and a copy of the settings used.
    """
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

    return {
        "subjects": [s.to_dict() for s in subjects],
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
    if s["separation"] is None:
        return f"{s['subjects_total']} subject(s), nothing scorable yet"
    return (
        f"{s['subjects_scored']}/{s['subjects_total']} subjects scored - "
        f"separation {s['separation']:+.1f} pts, "
        f"within-subject AUC {s['mean_within_subject_auc']}"
    )
