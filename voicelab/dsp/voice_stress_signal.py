"""Reduce a check-in to one number: how far it departs from the person's baseline.

One job, and a strict output contract. **This is the only thing that leaves
the voice pipeline.** A single 0-100 deviation value per (person, snapshot),
plus the metadata needed to say whether it can be trusted.

Nothing downstream ever receives audio, per-feature acoustic values, or the
personal baseline. The scoring path, the officer dashboard, the alert rules and
the recommendation engine all see exactly one number and a presence flag. That
narrowness is the module's purpose: it makes it structurally impossible for a
welfare officer's screen to display "your pitch was 12% higher than usual",
which would be both unactionable and intrusive.

What the number is, and what it is not
--------------------------------------
It is a **deviation from the person's own recent norm**, on the same 0-100
scale as every other behavioral signal. It is not a stress score, not a
diagnosis, not a measure of psychological state, and not comparable between
people. Two people with a value of 60 have each departed similarly far from
their own baselines; nothing follows about which of them is more distressed.

Pipeline position:
    ``voice_pipeline/voice_baseline`` -> **voice_stress_signal** ->
    ``behavioral_engine/behavioral_signals`` (as an optional input)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

from voicelab.dsp import settings
from voicelab.dsp.voice_baseline import VoiceBaseline

# Fail at import time rather than trusting the config to stay consistent.
_WEIGHT_TOTAL = sum(settings.VOICE_FEATURE_WEIGHTS.values())
if abs(_WEIGHT_TOTAL - 1.0) > 1e-9:
    raise ValueError(f"VOICE_FEATURE_WEIGHTS sums to {_WEIGHT_TOTAL}, not 1.0")
if set(settings.VOICE_FEATURE_WEIGHTS) != set(settings.VOICE_COMPARISON_FEATURE_NAMES):
    raise ValueError(
        "VOICE_FEATURE_WEIGHTS and VOICE_COMPARISON_FEATURE_NAMES disagree: "
        f"{sorted(set(settings.VOICE_FEATURE_WEIGHTS) ^ set(settings.VOICE_COMPARISON_FEATURE_NAMES))}"
    )
if set(settings.VOICE_FEATURE_DIRECTIONS) != set(settings.VOICE_COMPARISON_FEATURE_NAMES):
    raise ValueError("VOICE_FEATURE_DIRECTIONS does not cover the comparison features")

# Minimum share of the weighted feature set that must be computable before a
# signal is emitted at all. ASSUMPTION: below half the weight, the value would
# rest on two or three features and should not be presented as a reading.
MIN_AVAILABLE_WEIGHT: float = 0.50


@dataclass
class VoiceStressResult:
    """The single value that leaves the voice pipeline, plus its provenance.

    Attributes:
        pseudonym_id: Whose check-in this is.
        sample_date: Date of the check-in.
        signal: The 0-100 deviation value, or NaN when it could not be
            computed.
        is_reliable: Whether the baseline and the sample were both good enough
            for the value to be used. False means the value must be discarded
            by the caller, not shown with a caveat.
        available_weight: Share of the weighted feature set that contributed.
        contributing_features: Per-feature directional z-scores, retained
            **inside this module only** for debugging and unit tests. It is
            never serialised into an API response; see the module docstring.
        reason: Why the result is unreliable, when it is.
    """

    pseudonym_id: str
    sample_date: pd.Timestamp | None
    signal: float
    is_reliable: bool
    available_weight: float = 0.0
    contributing_features: Dict[str, float] = field(default_factory=dict)
    reason: str = ""

    def to_signal_row(self) -> Dict[str, object]:
        """Return the row that crosses the module boundary.

        Returns:
            Mapping with only ``pseudonym_id``, ``sample_date`` and
            ``voice_stress_signal``. Deliberately excludes the per-feature
            breakdown -- that is the whole point of this module.
        """
        return {
            "pseudonym_id": self.pseudonym_id,
            "sample_date": self.sample_date,
            settings.VOICE_SIGNAL_NAME: self.signal,
        }


def compute_voice_stress_signal(
    feature_vector: Mapping[str, float],
    baseline: VoiceBaseline,
    sample_date: pd.Timestamp | None = None,
) -> VoiceStressResult:
    """Compare one check-in against its owner's baseline.

    Args:
        feature_vector: Comparison features from
            ``AcousticFeatures.feature_vector()``.
        baseline: The person's baseline, built from *earlier* check-ins only.
        sample_date: Date of this check-in.

    Returns:
        A :class:`VoiceStressResult`.

    Method:
        For each comparison feature:
            ``z = direction * (value - centre) / scale``
        Departures in the non-concerning direction are clamped to zero, then
        each z is scaled to 0-100 saturating at
        ``VOICE_DEVIATION_SATURATION_SD``, and the results are combined with
        the configured weights, renormalised over whichever features were
        actually computable.

    Why only one direction counts:
        The same reasoning as in the behavioral engine. Speaking more slowly
        than usual is not something this system acts on, and allowing it to
        contribute a negative value would let one ordinary feature cancel out a
        genuine departure elsewhere in the same blend.

    Reliability:
        The result is unreliable -- and must be discarded, not shown with a
        caveat -- when the baseline has fewer than
        ``VOICE_BASELINE_MIN_SAMPLES`` contributing check-ins, or when less
        than :data:`MIN_AVAILABLE_WEIGHT` of the feature weight could be
        computed.
    """
    if not baseline.is_reliable:
        return VoiceStressResult(
            pseudonym_id=baseline.pseudonym_id,
            sample_date=sample_date,
            signal=float("nan"),
            is_reliable=False,
            reason=(
                f"personal baseline has {baseline.sample_count} check-in(s); "
                f"{settings.VOICE_BASELINE_MIN_SAMPLES} are required"
            ),
        )

    contributions: Dict[str, float] = {}
    weighted_sum = 0.0
    available_weight = 0.0

    for name in settings.VOICE_COMPARISON_FEATURE_NAMES:
        value = float(feature_vector.get(name, np.nan))
        centre = float(baseline.centre.get(name, np.nan))
        scale = float(baseline.scale.get(name, np.nan))
        if not (np.isfinite(value) and np.isfinite(centre) and np.isfinite(scale)) or scale <= 0:
            continue

        direction = settings.VOICE_FEATURE_DIRECTIONS[name]
        z = direction * (value - centre) / scale
        contributions[name] = float(z)

        concerning = max(0.0, z)
        scaled = min(1.0, concerning / settings.VOICE_DEVIATION_SATURATION_SD)
        weight = settings.VOICE_FEATURE_WEIGHTS[name]
        weighted_sum += weight * scaled
        available_weight += weight

    if available_weight < MIN_AVAILABLE_WEIGHT:
        return VoiceStressResult(
            pseudonym_id=baseline.pseudonym_id,
            sample_date=sample_date,
            signal=float("nan"),
            is_reliable=False,
            available_weight=available_weight,
            contributing_features=contributions,
            reason=(
                f"only {available_weight:.0%} of the acoustic feature weight could "
                f"be measured on this recording"
            ),
        )

    signal = settings.SIGNAL_MAX * weighted_sum / available_weight
    return VoiceStressResult(
        pseudonym_id=baseline.pseudonym_id,
        sample_date=sample_date,
        signal=float(np.clip(signal, settings.SIGNAL_MIN, settings.SIGNAL_MAX)),
        is_reliable=True,
        available_weight=available_weight,
        contributing_features=contributions,
    )


def signals_to_frame(
    results: Sequence[VoiceStressResult],
    snapshot_dates: Sequence[pd.Timestamp],
) -> pd.DataFrame:
    """Align per-check-in signals to the pipeline's snapshot dates.

    Args:
        results: Reliable voice-stress results, any order.
        snapshot_dates: The feature pipeline's snapshot dates.

    Returns:
        DataFrame with ``pseudonym_id``, ``snapshot_date`` and
        ``voice_stress_signal``, containing a row only where a person had a
        usable check-in on or before that snapshot. Absent rows are the normal
        case and are handled by the behavioral engine's presence flag.

    Design note:
        A snapshot takes the most recent check-in **at or before** it. Using a
        later check-in would leak information from after the snapshot into a
        training row, which would inflate every model metric. A check-in older
        than one snapshot interval is not carried forward, because a
        three-month-old voice reading is not evidence about today and
        presenting it as one would be dishonest about how fresh the input is.
    """
    usable = [r for r in results if r.is_reliable and np.isfinite(r.signal)]
    if not usable:
        return pd.DataFrame(
            columns=["pseudonym_id", "snapshot_date", settings.VOICE_SIGNAL_NAME]
        )

    frame = pd.DataFrame([r.to_signal_row() for r in usable])
    frame["sample_date"] = pd.to_datetime(frame["sample_date"])

    max_age = pd.Timedelta(days=settings.SNAPSHOT_INTERVAL_DAYS)
    rows: List[Dict[str, object]] = []
    for pid, group in frame.groupby("pseudonym_id"):
        group = group.sort_values("sample_date")
        for snapshot in snapshot_dates:
            eligible = group[
                (group["sample_date"] <= snapshot)
                & (group["sample_date"] >= snapshot - max_age)
            ]
            if eligible.empty:
                continue
            rows.append(
                {
                    "pseudonym_id": pid,
                    "snapshot_date": snapshot,
                    settings.VOICE_SIGNAL_NAME: float(
                        eligible.iloc[-1][settings.VOICE_SIGNAL_NAME]
                    ),
                }
            )
    return pd.DataFrame(
        rows, columns=["pseudonym_id", "snapshot_date", settings.VOICE_SIGNAL_NAME]
    )
