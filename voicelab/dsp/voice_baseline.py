"""Build and maintain each person's own acoustic baseline.

One job: hold what this person's voice normally does, so a later check-in can
be compared against themselves rather than against anyone else.

Why the baseline must be personal
---------------------------------
This is not a stylistic preference; it is what makes the voice feature work at
all. Measured on the project's synthetic corpus, the correlation between mean
pitch and the injected strain level is:

    pooled across all speakers   +0.04
    within each speaker          +0.98

Pooled, the signal is invisible -- habitual pitch varies enormously between
people, and that between-speaker variation completely swamps the within-speaker
change. The same measurement, referenced to each person's own history, recovers
it almost perfectly. Any voice feature compared against a population norm would
be measuring who someone is rather than how they are doing, which is both
useless and exactly the kind of profiling this system must not do.

What is stored, and what is not
-------------------------------
A baseline holds summary statistics over the seven scale-invariant comparison
features: a centre and a scale per feature, plus a sample count and a
timestamp. It does not hold audio, does not hold per-sample feature history,
and cannot be used to reconstruct any recording. Raw audio is discarded
immediately after feature extraction
(``settings.RETENTION_RAW_AUDIO_DAYS = 0``).

Pipeline position:
    ``voice_pipeline/acoustic_features`` -> **voice_baseline** ->
    ``voice_pipeline/voice_stress_signal``
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np

from voicelab.dsp import settings

# Converts an interquartile range to a standard-deviation equivalent for
# normally distributed data (IQR = 1.349 sigma).
IQR_TO_SD: float = 1.349
# Converts a mean absolute deviation to a standard-deviation equivalent for
# normally distributed data (MAD = sigma * sqrt(2/pi) = 0.798 sigma).
MAD_TO_SD: float = 1.2533

# Floor on the per-feature scale, as a fraction of the centre. A person whose
# first few check-ins happen to be nearly identical would otherwise get a
# near-zero scale, turning any ordinary variation into a huge deviation.
# ASSUMPTION.
MIN_SCALE_FRACTION_OF_CENTRE: float = 0.02


@dataclass
class VoiceBaseline:
    """One person's acoustic baseline.

    Attributes:
        pseudonym_id: Owner of the baseline. Never a raw personnel id.
        centre: Per-feature central value.
        scale: Per-feature standard-deviation-equivalent spread.
        sample_count: How many check-ins have contributed.
        last_updated: ISO date of the most recent contributing sample.
    """

    pseudonym_id: str
    centre: Dict[str, float] = field(default_factory=dict)
    scale: Dict[str, float] = field(default_factory=dict)
    sample_count: int = 0
    last_updated: str | None = None

    @property
    def is_reliable(self) -> bool:
        """Whether enough check-ins back this baseline to compare against it.

        Returns:
            True once ``sample_count`` reaches
            ``settings.VOICE_BASELINE_MIN_SAMPLES``. Below that the baseline
            exists but must not be used to produce a signal -- comparing a
            person against one or two of their own recordings would generate
            confident-looking noise.
        """
        return self.sample_count >= settings.VOICE_BASELINE_MIN_SAMPLES

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serialisable form, for database storage."""
        return {
            "pseudonym_id": self.pseudonym_id,
            "centre": dict(self.centre),
            "scale": dict(self.scale),
            "sample_count": self.sample_count,
            "last_updated": self.last_updated,
        }

    def to_json(self) -> str:
        """Return the baseline as a JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "VoiceBaseline":
        """Rebuild a baseline from its stored form.

        Args:
            payload: Output of :meth:`to_dict`.

        Returns:
            The reconstructed baseline.
        """
        return cls(
            pseudonym_id=str(payload["pseudonym_id"]),
            centre={k: float(v) for k, v in dict(payload.get("centre", {})).items()},
            scale={k: float(v) for k, v in dict(payload.get("scale", {})).items()},
            sample_count=int(payload.get("sample_count", 0)),
            last_updated=payload.get("last_updated"),  # type: ignore[arg-type]
        )


def _floor_scale(scale: float, centre: float) -> float:
    """Apply the minimum-scale floor to one feature.

    Args:
        scale: Computed spread.
        centre: Computed centre, used to set a proportional floor.

    Returns:
        The scale, raised to the floor if it fell below it. Returns NaN
        unchanged.
    """
    if not np.isfinite(scale) or not np.isfinite(centre):
        return scale
    floor = abs(centre) * MIN_SCALE_FRACTION_OF_CENTRE
    return max(scale, floor)


def build_baseline(
    pseudonym_id: str,
    feature_vectors: Sequence[Mapping[str, float]],
    last_updated: str | None = None,
    feature_names: Sequence[str] = settings.VOICE_COMPARISON_FEATURE_NAMES,
) -> VoiceBaseline:
    """Build a baseline from a batch of a person's feature vectors.

    Args:
        pseudonym_id: Owner of the baseline.
        feature_vectors: One mapping per contributing check-in, as returned by
            ``AcousticFeatures.feature_vector()``.
        last_updated: ISO date of the most recent contributing sample.
        feature_names: Features to summarise.

    Returns:
        A :class:`VoiceBaseline`. Features with no finite observations get NaN
        centre and scale, which the signal calculation then skips.

    Method:
        Centre is the **median** and scale is **IQR / 1.349** (a
        standard-deviation equivalent), not mean and SD. With three to five
        check-ins, a single unusual recording -- a bad connection, a cold, a
        noisy room -- would move a mean substantially and inflate an SD enough
        to mask every subsequent change. Median and IQR are far less affected
        by one outlier, which matters most precisely when the sample is
        smallest.
    """
    centre: Dict[str, float] = {}
    scale: Dict[str, float] = {}

    for name in feature_names:
        values = np.array(
            [float(v.get(name, np.nan)) for v in feature_vectors], dtype=np.float64
        )
        values = values[np.isfinite(values)]
        if values.size == 0:
            centre[name] = float("nan")
            scale[name] = float("nan")
            continue
        median = float(np.median(values))
        if values.size < 2:
            spread = float("nan")
        else:
            iqr = float(np.percentile(values, 75) - np.percentile(values, 25))
            spread = iqr / IQR_TO_SD
        centre[name] = median
        scale[name] = _floor_scale(spread, median)

    return VoiceBaseline(
        pseudonym_id=pseudonym_id,
        centre=centre,
        scale=scale,
        sample_count=len(feature_vectors),
        last_updated=last_updated,
    )


def update_baseline(
    baseline: VoiceBaseline,
    feature_vector: Mapping[str, float],
    sample_date: str | None = None,
    alpha: float = settings.VOICE_BASELINE_EMA_ALPHA,
    feature_names: Sequence[str] = settings.VOICE_COMPARISON_FEATURE_NAMES,
) -> VoiceBaseline:
    """Fold one new check-in into an existing baseline.

    Args:
        baseline: The existing baseline.
        feature_vector: The new check-in's comparison features.
        sample_date: ISO date of the new sample.
        alpha: Exponential-moving-average factor. Higher adapts faster.
        feature_names: Features to update.

    Returns:
        A new :class:`VoiceBaseline`; the input is not mutated.

    Method:
        Centre is updated by exponential moving average. Scale is updated by an
        EMA over the absolute deviation from the *previous* centre, converted
        to a standard-deviation equivalent (MAD x 1.2533).

    Why the baseline is allowed to drift at all:
        A person's voice changes over months and years for reasons that have
        nothing to do with welfare. A frozen baseline would slowly turn ordinary
        ageing into a rising stress reading. The trade-off is real and worth
        stating plainly: a baseline that adapts will eventually absorb a slow,
        sustained deterioration and stop flagging it. That is precisely why the
        voice signal is one optional input among eight and never a
        determination on its own, and why sustained trends are the job of
        ``post_model_analytics/trend_engine.py``, which works on the risk score
        and does not adapt.
    """
    centre = dict(baseline.centre)
    scale = dict(baseline.scale)

    for name in feature_names:
        value = float(feature_vector.get(name, np.nan))
        if not np.isfinite(value):
            continue
        old_centre = centre.get(name, float("nan"))
        old_scale = scale.get(name, float("nan"))

        if not np.isfinite(old_centre):
            centre[name] = value
            scale[name] = _floor_scale(abs(value) * MIN_SCALE_FRACTION_OF_CENTRE, value)
            continue

        deviation = abs(value - old_centre)
        new_centre = (1.0 - alpha) * old_centre + alpha * value
        mad = deviation if not np.isfinite(old_scale) else (
            (1.0 - alpha) * (old_scale / MAD_TO_SD) + alpha * deviation
        )
        centre[name] = new_centre
        scale[name] = _floor_scale(mad * MAD_TO_SD, new_centre)

    return VoiceBaseline(
        pseudonym_id=baseline.pseudonym_id,
        centre=centre,
        scale=scale,
        sample_count=baseline.sample_count + 1,
        last_updated=sample_date or baseline.last_updated,
    )


def build_baselines_for_people(
    features_by_person: Mapping[str, Sequence[Mapping[str, float]]],
    last_updated_by_person: Mapping[str, str] | None = None,
) -> Dict[str, VoiceBaseline]:
    """Build baselines for many people at once.

    Args:
        features_by_person: Mapping of pseudonym to that person's feature
            vectors, oldest first.
        last_updated_by_person: Optional mapping of pseudonym to the ISO date
            of their most recent contributing sample.

    Returns:
        Mapping of pseudonym to baseline, including unreliable ones -- the
        caller checks ``is_reliable`` rather than finding people silently
        missing.
    """
    dates = last_updated_by_person or {}
    return {
        pid: build_baseline(pid, vectors, dates.get(pid))
        for pid, vectors in features_by_person.items()
    }
