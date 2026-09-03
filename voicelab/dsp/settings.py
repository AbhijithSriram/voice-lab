"""Every tunable constant in the acoustic pipeline, in one file.

**This is the file you edit when iterating.** Nothing else in `voicelab/dsp/`
holds a number you would want to change; they hold the arithmetic that uses
these. Change a value here, re-run the analysis from the admin dashboard, and
compare separation before and after.

Vendored from `pwiews` (`backend/config/settings.py`, the voice block) so this
app stands alone and can be tuned without touching the hackathon repo. When a
setting here earns its keep, copy it back.

The convention from the parent project is kept: every constant carries either
`SOURCE:` (traceable to a published figure) or `ASSUMPTION:` (a choice this
project made). Do not blur the two — the whole point of the exercise is to
find out which assumptions survive contact with real voices.

WHAT THIS PIPELINE DOES NOT DO
------------------------------
No transcription, speech-to-text, phoneme recognition, keyword spotting or
language identification. It measures *how* a voice sounds, never *what* was
said, and that is why subjects may speak in any language they like. Nothing
downstream of `acoustic_features.extract_features` sees anything but numbers.
"""

from __future__ import annotations

from typing import Dict, Final, Tuple

# ---------------------------------------------------------------------------
# Signal scale
# ---------------------------------------------------------------------------
SIGNAL_MIN: Final[float] = 0.0
SIGNAL_MAX: Final[float] = 100.0
VOICE_SIGNAL_NAME: Final[str] = "voice_stress_signal"

# Used only by `voice_stress_signal.signals_to_frame`, which aligns signals to
# the parent project's monthly snapshot dates. This lab does not call it. The
# constant is kept so the vendored DSP files stay byte-identical to the ones in
# pwiews apart from their import lines -- if they diverge, results measured here
# stop being evidence about the pipeline there.
SNAPSHOT_INTERVAL_DAYS: Final[int] = 30

# ---------------------------------------------------------------------------
# Audio format and framing
# ---------------------------------------------------------------------------
# The browser records at exactly this rate and encodes 16-bit PCM WAV, so no
# resampling ever happens. That matters more than it looks: jitter is a
# period-length measurement, and resampling changes period lengths.
VOICE_SAMPLE_RATE_HZ: Final[int] = 16000
VOICE_FRAME_LENGTH_MS: Final[int] = 32       # ASSUMPTION: standard 32 ms frame.
VOICE_HOP_LENGTH_MS: Final[int] = 10         # ASSUMPTION: standard 10 ms hop.
VOICE_MIN_DURATION_SEC: Final[float] = 3.0   # Below this, sample is rejected.
VOICE_MAX_DURATION_SEC: Final[float] = 60.0  # Above this, sample is truncated.

# ---------------------------------------------------------------------------
# Pitch tracking
# ---------------------------------------------------------------------------
# SOURCE: adult human speaking F0 spans roughly 70-300 Hz (typical adult male
# ~85-180 Hz, adult female ~165-255 Hz); this range covers both.
#
# TUNING NOTE: if your subjects include voices near the edges of this range,
# widening it costs a little accuracy in the middle (more room for the
# autocorrelation to find a wrong peak) and gains coverage at the ends. Check
# f0_mean_hz in the per-sample feature table before widening on a hunch.
VOICE_F0_MIN_HZ: Final[float] = 70.0
VOICE_F0_MAX_HZ: Final[float] = 300.0

# Autocorrelation peak height above which a frame counts as voiced.
# ASSUMPTION. Raise it if noisy rooms are producing spurious pitch; lower it
# if quiet speakers are losing most of their frames.
VOICE_AUTOCORR_VOICING_THRESHOLD: Final[float] = 0.35

# ---------------------------------------------------------------------------
# Silence, pauses and speaking rate
# ---------------------------------------------------------------------------
# Silence threshold as a fraction of the sample's peak RMS. ASSUMPTION.
VOICE_SILENCE_RMS_FRACTION: Final[float] = 0.10
VOICE_MIN_PAUSE_MS: Final[int] = 150         # ASSUMPTION: pauses >=150 ms count.

# Syllable nuclei are energy-envelope peaks at least this far apart, with at
# least this prominence relative to the envelope range. ASSUMPTIONS.
VOICE_MIN_SYLLABLE_SEPARATION_MS: Final[int] = 100
VOICE_SYLLABLE_PROMINENCE_FRACTION: Final[float] = 0.15

# ---------------------------------------------------------------------------
# Personal baseline
# ---------------------------------------------------------------------------
# How many prior samples before a personal baseline is trusted. ASSUMPTION.
#
# TUNING NOTE: this is the single most consequential number in the experiment.
# At 3, a subject needs 4 recordings before the 4th produces any signal at all;
# with 5 baseline recordings you get 2 scored. Lowering it to 2 buys you more
# scored samples per subject at the cost of a noisier baseline. The lab's
# analysis reports how many samples each subject actually scored, so you can
# see what this is costing you.
VOICE_BASELINE_MIN_SAMPLES: Final[int] = 3

# Exponential-moving-average factor for updating a baseline. ASSUMPTION.
# Only relevant over long collection periods; with a day or two of recording
# it barely matters.
VOICE_BASELINE_EMA_ALPHA: Final[float] = 0.30

# Deviation, in personal-baseline SDs, at which the signal saturates at 100.
# ASSUMPTION.
#
# TUNING NOTE: if almost every "sad" sample pins at 100, this is too low and
# you are throwing away the ranking. If nothing gets above 40, it is too high.
VOICE_DEVIATION_SATURATION_SD: Final[float] = 3.0

# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
# Every acoustic measurement extracted and stored per sample.
VOICE_ACOUSTIC_FEATURE_NAMES: Final[Tuple[str, ...]] = (
    "f0_mean_hz",
    "f0_sd_hz",
    "speaking_rate_syllables_per_sec",
    "pause_ratio",
    "intensity_rms_mean",
    "intensity_rms_sd",
    "intensity_rms_cv",
    "jitter_local_pct",
    "shimmer_local_pct",
)

# The subset actually compared against a personal baseline. It excludes the two
# absolute intensity measures, which are NOT scale-invariant: recording level
# varies with handset, microphone gain and distance from the mouth, none of
# which this app controls, so comparing absolute loudness across recordings
# measures the setup rather than the person. Their self-normalising ratio
# (intensity_rms_cv = sd / mean) is scale-invariant and is used instead. The
# excluded pair is still extracted and stored, as recording-quality context.
#
# TUNING NOTE: this is a hard constraint, not a preference. In a browser-based
# collection app the recording setup varies far more than it did in the
# synthetic corpus. Do not add intensity_rms_mean here.
VOICE_COMPARISON_FEATURE_NAMES: Final[Tuple[str, ...]] = (
    "f0_mean_hz",
    "f0_sd_hz",
    "speaking_rate_syllables_per_sec",
    "pause_ratio",
    "intensity_rms_cv",
    "jitter_local_pct",
    "shimmer_local_pct",
)

# ---------------------------------------------------------------------------
# Directions and weights -- THE MAIN TUNING SURFACE
# ---------------------------------------------------------------------------
# +1 means "higher than baseline is the direction of concern", -1 the reverse.
# A departure in the non-concerning direction is clamped to zero.
#
# These came from the pressured-speech literature and were then *validated
# against synthetic audio that was generated using the same directions* --
# which proves the DSP recovers what was injected, and proves nothing about
# real speech. That circularity is exactly what this app exists to break.
#
# IMPORTANT: sadness is not the same as pressure/stress. Much of the literature
# on sad or depressed speech reports the OPPOSITE of several of these:
#
#   pressured / anxious speech          sad / depressed speech
#   ------------------------            ----------------------
#   F0 raised                           F0 lowered, flatter
#   speaking rate faster                speaking rate slower
#   pause ratio lower                   pause ratio HIGHER
#
# So when you record subjects "speaking as they would when sad", expect
# speaking_rate and pause_ratio to fight the current directions. Do not treat
# a negative result as a broken pipeline until you have looked at the
# per-feature z-score table the analysis prints -- it will tell you which
# features moved and which way, and that table is the actual finding.
#
# Jitter and shimmer are the two most likely to hold their direction across
# both states, because both reflect loss of fine laryngeal control rather than
# a communicative choice, and neither is under much conscious control.
VOICE_FEATURE_DIRECTIONS: Final[Dict[str, int]] = {
    "f0_mean_hz": +1,                       # raised pitch
    "f0_sd_hz": +1,                         # see note below
    "speaking_rate_syllables_per_sec": +1,  # faster
    "pause_ratio": -1,                      # less pausing
    "intensity_rms_cv": +1,                 # less even loudness
    "jitter_local_pct": +1,                 # more cycle-to-cycle perturbation
    "shimmer_local_pct": +1,
}

# f0_sd_hz is the one direction set from measurement behaviour rather than the
# literature. The intuitive value is -1 (a flatter contour under pressure); it
# is +1 because F0 standard deviation measured *in Hz* is confounded -- it
# scales with mean F0 and with cycle-level perturbation, both of which rise,
# and that swamps any flattening. A semitone-denominated measure would behave
# differently. It carries the smallest weight for exactly this reason.

# Weights combining per-feature deviations into the single signal.
# Must sum to 1.0; voice_stress_signal.py asserts this at import time.
# ALL ASSUMPTIONS.
VOICE_FEATURE_WEIGHTS: Final[Dict[str, float]] = {
    "f0_mean_hz": 0.18,
    "f0_sd_hz": 0.06,
    "speaking_rate_syllables_per_sec": 0.18,
    "pause_ratio": 0.12,
    "intensity_rms_cv": 0.08,
    "jitter_local_pct": 0.18,
    "shimmer_local_pct": 0.20,
}
