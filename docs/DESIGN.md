# Voice Lab — what it is, what it collects, and how it is analysed

Internal design note for the SIH26186 team. Written to get everyone onto the
same page about what this app is actually measuring, what we changed in the
parent pipeline and why, and which of our results would and would not survive a
hostile question.

Canonical location: `docs/DESIGN.md` in the `voice-lab` repo. If this document
and the code disagree, the code is right and this file is stale — fix it.

---

## 1. Why this exists

The parent project (`pwiews`, SIH26186) is a welfare-monitoring system for CRPF
personnel. One of its inputs is a **voice stress signal**: a number derived from
how somebody sounds, compared against how that same person usually sounds.

That pipeline was built and tuned against a *synthetic corpus*. Nobody has ever
run it on real human voices. Voice Lab exists to do exactly one thing:

> **Collect labelled voice samples from real people, run the parent pipeline
> over them unmodified, and find out whether it separates a person's low-mood
> speech from their ordinary speech.**

It is a measuring instrument pointed at our own pipeline. A negative result is a
real result and is the second most useful thing this can produce. The most
useful is a specific, named reason the pipeline fails.

**Live at:** `https://voicelab.abhijith-sriram.in`

> **Where we are (6 September 2026).** The first collection round is in: 122
> recordings from 11 subjects, none of which the extractor failed on. The
> collection works; the pipeline does not separate mood yet, and two specific
> reasons are now measured rather than guessed. **Section 11 has the numbers** —
> read it before quoting anything from this document as a result.

---

## 2. The question, stated precisely

Not *"can we detect sadness from voice"*. The question we can actually answer
with 20–30 volunteers is narrower:

> For a given person, does the pipeline score their self-labelled low-mood
> recordings higher than their own ordinary recordings, when both are compared
> against a baseline built only from earlier ordinary recordings?

Three things are load-bearing in that sentence:

- **For a given person.** Never across people. Habitual pitch and pace vary
  enormously between individuals and swamp any within-person change. The
  vendored baseline module records the measurement: on the synthetic corpus,
  mean pitch against injected strain correlated **+0.04 pooled across speakers**
  and **+0.98 within each speaker**.
- **Self-labelled.** The subject tells us how they felt. That is context, not
  ground truth. See §8.
- **Only from earlier recordings.** A baseline that includes the sample it is
  scoring leaks, and would manufacture a result. The analysis enforces strict
  chronological ordering.

### 2.1 Stress is not sadness, and this matters

The parent pipeline detects **pressured speech** — the acoustic profile of
*acute, high-arousal* stress: faster, higher-pitched, less pausing.

Sadness and "heaviness" are negative valence but **low arousal**: slower, lower,
*more* pausing. Same valence, opposite arousal.

They are not the same construct, and on the two features that matter most for
pace they move in **opposite directions**. §7.2 covers what that does to our
numbers. Anyone presenting this work needs to be able to say that sentence out
loud before a judge says it for them.

---

## 3. What we collect

### 3.1 Per subject

| Field | Notes |
|---|---|
| `username` | Chosen by the subject. Not a real name unless they choose one. |
| `password_hash` | Werkzeug PBKDF2. Never the password itself. |
| `display_name` | Optional. |
| `notes` | Free text. Age band, first language, "I have a cold today". Optional by design — this is a voice experiment, not a profile. |
| `is_admin` | Set only at signup, by typing the admin code. |

### 3.2 Per recording

| Field | Notes |
|---|---|
| `filename` | The WAV on disk. |
| `label` | `neutral` or `sad`. The subject picks "Normal" or "Low" in the UI. |
| `intensity` | Self-reported 1–3 on low-mood recordings only. |
| `language` | Free text, subject-supplied. |
| `session_id` | Groups the three recordings of one round. |
| `prompt_id` | Which prompt — and therefore which **load** (§4). |
| `duration_sec`, `sample_rate` | Recording-quality context. |
| `features_json` | The nine extracted acoustic measures, computed once at upload. |
| `extract_error` | Set when the extractor could not measure the sample. Such samples are **kept and reported, never silently dropped** — a pipeline that fails on 30% of real recordings is the single most important thing this lab could tell us. |

### 3.3 What we do **not** collect

**There is no transcription anywhere in this system.** What a subject says is
never converted to text and never analysed. That is why they can speak in any
language, and it is the strongest privacy claim we have. Do not weaken it by
adding ASR "just to check".

### 3.4 Audio format, and why it is not MP3

`MediaRecorder` produces WebM/Opus. Opus is a **lossy perceptual** codec: it
discards exactly the fine waveform detail this pipeline measures. Jitter is
cycle-to-cycle variation in glottal period and shimmer is cycle-to-cycle
variation in amplitude — both sub-perceptual, both the first thing a perceptual
codec throws away, and together they carry **0.38 of the feature weight**.
Measuring jitter on decoded Opus produces a number, and the number is about the
codec.

So `static/recorder.js` captures raw Float32 PCM from the Web Audio API,
downsamples to 16 kHz, and writes 16-bit PCM WAV **in the browser**. Nothing is
transcoded anywhere, and the server needs no `ffmpeg`.

Sample-rate mismatches are **refused, not resampled**. Resampling changes period
lengths, and jitter *is* a period-length measurement.

> **Read section 11.4 before relying on this.** The argument above is about
> *lossy coding* and it holds. It does not establish that 16 kHz is enough
> **resolution** for perturbation measures, and the first real corpus says it is
> not: one sample at 16 kHz is 62.5 µs, which is larger than the period
> difference 1% jitter represents in a 200 Hz voice.

### 3.5 Where it lives

```
/srv/nas/voice-lab/voicelab.sqlite3    the database
/srv/nas/voice-lab/audio/*.wav         the recordings
```

On the 596 GB `/srv` partition, not the 46 GB root filesystem, and outside the
code checkout so `git pull` can never touch it. Local ext4 — which is the only
reason SQLite is safe there. **If this ever moves to NFS or CIFS, the database
must come back to local disk**; SQLite's locking cannot be trusted over either.

> **Note a divergence from the parent project.** `voice_baseline.py` states that
> raw audio is discarded immediately after feature extraction
> (`RETENTION_RAW_AUDIO_DAYS = 0`). That constant does not exist in this lab's
> settings and the claim is **not true here** — we keep every WAV on purpose, so
> measurements can be re-run when the pipeline changes. That is right for a lab
> and would be wrong in deployment. Do not copy this behaviour into `pwiews`.

---

## 4. The recording protocol

A **round** is three recordings made back to back in one sitting, all under one
mood label. The three are fixed and the subject does not choose them:

| # | Prompt | Load | What it is for |
|---|---|---|---|
| 1 | Hold a steady "aaah" for 8+ seconds | `phonation` | Jitter and shimmer (0.38 of the weight) |
| 2 | Count one to twenty at a comfortable pace | `automatic` | The **easy anchor** |
| 3 | Name as many animals as you can | `effortful` | The **hard probe** |

**Every subject records 4 ordinary-day ("Normal") rounds first**, then low-mood
rounds whenever one genuinely occurs.

### 4.1 Why these three, specifically

**Sustained vowel.** Jitter and shimmer come from clinical voice assessment,
where they are measured on sustained phonation. On connected speech they pick up
intonation sweeps, voicing onsets and coarticulation instead of the larynx. This
is not theoretical — the first real sitting on our server measured **4.32%
jitter on counting and 1.26% on the sustained vowel** from the same person
minutes apart. Clinical norms are under about 1%. The connected-speech number is
not a measurement of that person's voice.

**Automatic speech.** Counting and weekday lists are *overlearned sequences* —
in neurology, literally "automatic speech", preserved even in severe aphasia.
They require no retrieval, which makes them the true floor of cognitive load and
a stable anchor.

**Category fluency.** A standard neuropsychological measure, reliably impaired in
depression, and **emotionally inert**. The pauses are the mechanism rather than a
proxy: as retrieval gets harder, inter-word intervals lengthen directly.

### 4.2 What we deliberately rejected

An earlier proposal for the hard prompt was *"tell me about the last time you
called your mother"*. It was rejected because it is harder **and** emotionally
loaded — homesickness, estrangement, bereavement. A lengthened pause could not
be attributed to cognitive difficulty rather than reactivity to the topic. For a
system aimed at personnel posted far from home that confound is the whole
subject matter, and it is not a question any welfare tool should be putting to a
bereaved subject.

The prompt also said **"count slowly"** until we looked at real data: it scored
`pause_ratio` **0.647** against category fluency's **0.608**, i.e. the *easy*
prompt was pausing more than the hard one, because the wording instructed the
pauses it was meant to measure.

Three free prompts ("describe your room", …) were removed entirely. They carried
no load, could not pair, and pooled three unrelated tasks into one baseline. Two
of the first three sittings ever recorded on this server used them and counted
for nothing.

---

## 5. The original pipeline

Three vendored modules, byte-identical to `pwiews` apart from import lines. If
they diverge, results measured here stop being evidence about the pipeline
there.

```
acoustic_features  ->  voice_baseline  ->  voice_stress_signal
```

### 5.1 Features

Nine are extracted and stored; **seven** are compared against baseline:

```
f0_mean_hz                        0.18   +1
f0_sd_hz                          0.06   +1
speaking_rate_syllables_per_sec   0.18   +1
pause_ratio                       0.12   -1
intensity_rms_cv                  0.08   +1
jitter_local_pct                  0.18   +1
shimmer_local_pct                 0.20   +1
                                  ----
                                  1.00
```

`intensity_rms_mean` and `intensity_rms_sd` are extracted but **never
compared** — absolute loudness varies with handset, microphone gain and distance
from the mouth, none of which we control, so comparing it measures the setup
rather than the person. Their self-normalising ratio (`cv = sd / mean`) is
scale-invariant and is used instead. This is a hard constraint, not a
preference.

The direction column is `+1` for "higher than baseline is the direction of
concern". Note `pause_ratio` is `-1` — pressured speech pauses *less*.

### 5.2 Personal baseline

Centre is the **median**, scale is **IQR / 1.349** (an SD equivalent) — not mean
and SD. With three to five samples one bad recording would move a mean
substantially and inflate an SD enough to hide everything after it. Scale is
floored at 2% of centre so a subject whose first few recordings happen to be
near-identical does not get a near-zero divisor.

Needs `VOICE_BASELINE_MIN_SAMPLES = 3`. After that it updates by exponential
moving average (`alpha = 0.30`).

### 5.3 The signal

Per feature:

```
z          = direction * (value - centre) / scale
concerning = max(0, z)                       <-- note this
scaled     = min(1, concerning / 3.0)
```

then a weighted mean over available features, times 100. Needs at least 50% of
the feature weight measurable or it returns "not reliable" rather than a number.

**`max(0, z)` is the single most consequential line in the pipeline for us.**
See §7.2.

---

## 6. What we changed, and why

Everything in this section is *our* work, not the parent pipeline's. The
vendored DSP is untouched.

### 6.1 Per-load baselines

A baseline pools a person's ordinary recordings to learn what their voice
normally does — which means something only if those recordings are **the same
task**.

Our first real subject's three ordinary recordings were a sustained vowel,
counting and category fluency, with `pause_ratio` **0.004, 0.647 and 0.608**.
Pooled, the spread that becomes the divisor for every later z-score is not that
person's variability at all. It is the distance between three different tasks,
and it is enormous.

So **each load carries its own baseline**. Free prompts and pre-migration
recordings go to an `untagged` bucket and are scored among themselves, so they
neither vanish nor contaminate a task they do not resemble.

Consequence for data collection: `VOICE_BASELINE_MIN_SAMPLES` is per load. Four
ordinary **rounds**, not four recordings.

### 6.2 The paired cognitive-load contrast

The absolute score measures a *level*: how far a low-mood recording sits from
baseline. That is only comparable if recording conditions hold still, and they
do not — subjects use their own phones, in their own rooms, at their own hours.

So each round also yields a **contrast**:

```
delta  = z(effortful) - z(automatic)          within one round
signal = mean(delta | low) - mean(delta | ordinary)
```

Each half is scored against its own load's baseline, so each `z` reads "how
unusual was this recording for this task, for this person", and the difference
reads "did the hard task fall further from its own norm today than the easy one
did from its own".

**Why a difference is better than a level here:** anything that shifts both
halves together — different phone, noisier room, tiredness, caffeine, habitual
speaking rate, language — largely cancels.

> **Honest caveat.** With a single shared baseline the centre cancelled
> algebraically and a common shift vanished *exactly*. With per-load centres the
> cancellation is approximate, leaving `d * (1/scale_hard - 1/scale_easy)`. We
> took that trade deliberately: the task effect now removed exactly is two
> orders of magnitude larger than the residue. There is a test asserting the
> residue stays small rather than pretending it is zero.

**It is also a positive control.** The easy/hard gap is a large effect whose
existence is not in doubt. If ordinary rounds show no gap, the measurement chain
is not resolving cognitive load at all, and nothing subtler it says about
sadness can be believed. The report states this outright in `control_check`.

### 6.3 Load directions ≠ strain directions

Scoring the contrast with `VOICE_FEATURE_DIRECTIONS` **inverts it**. That table
describes pressured speech (faster, less pausing); cognitive load does the
reverse. We found this the hard way — a textbook positive result scored −0.667.

```python
LOAD_DIRECTIONS = {
    "pause_ratio": +1,                       # load lengthens pauses
    "speaking_rate_syllables_per_sec": -1,   # load slows delivery
}
```

Only these two, because they are the only features whose response to cognitive
load can be asserted up front. Every feature's raw delta is still reported. A
contract test keeps the two tables opposed so this cannot regress.

### 6.4 Rounds, and the UI rebuild

`session_id` groups the three recordings of a round; a round is one visit to the
record page. The UI was rebuilt to ask **one question at a time** and fix the
running order server-side, after the first version let subjects assemble
sittings that counted for nothing. The words *load*, *baseline*, *session* and
*contrast* do not appear anywhere a subject can read them.

---

## 7. How to read the results page

### 7.1 The absolute analysis

- **Controls** — held-out ordinary recordings, scored against the subject's own
  ordinary baseline. This is the **noise floor**. If a subject's controls average
  40, the pipeline is measuring recording conditions and any separation on top of
  that is not evidence.
- **Cases** — low-mood recordings against the same baseline.
- **AUC** — pick one control and one case at random; how often does the case
  score higher? 0.5 is a coin flip. Needs no threshold, which is why it is the
  right metric while thresholds are untuned.
- **Pooled vs mean within-subject AUC** — when they disagree, **the
  within-subject figure is the honest one**, because the pipeline only ever
  compares a person to themselves.

### 7.2 Expect the absolute score to under-report, and know why

`concerning = max(0, z)` discards every departure running against the strain
direction. With `pause_ratio: -1` and `speaking_rate: +1`, and sadness moving
both the other way, **0.30 of the feature weight is structurally invisible for
exactly the recordings this lab collects.** A low-mood sample that pauses far
more than baseline contributes nothing and can score `0.0`.

That is correct behaviour for a pipeline built to detect pressured speech. It is
not obvious from a results page full of zeros, so:

- The **per-feature z table is not clamped.** It is the only place a departure
  running against the strain convention survives. **A negative shift there is a
  finding, not a bug.**
- The load contrast does not clamp either, which is now its strongest
  justification.

---

## 8. Known limitations — say these before someone else does

1. **Portrayed, not felt.** Subjects are asked to record when they feel low, and
   the UI suggests musical mood induction, but a student recording on demand is
   not a jawan under sustained stress. Acted emotional speech is measurably more
   exaggerated and more separable than genuine emotion, so any AUC we report is
   optimistic. Nearly every public emotion-speech corpus has this problem; state
   it rather than hide it.
2. **Arousal mismatch.** §2.1 and §7.2. We are collecting low-arousal sadness and
   scoring it with a high-arousal stress convention.
3. **Self-report is not ground truth.** No clinical instrument (PHQ-9 or
   similar) is administered. `intensity` is context.
4. **Small n.** 20–30 students, one demographic, mostly one region, mostly young.
   Nothing here generalises to a CRPF population.
5. **No field validation.** Nothing in this lab tests the deployed system.
6. **Ordering effects.** Every round runs in the same fixed order, so any
   warm-up or fatigue effect is baked in identically everywhere. It cancels in
   the within-subject contrast; it does not cancel in the absolute score.

---

## 9. Running it

```
code     /home/abhijith/voice-lab        (systemd unit: voicelab)
data     /srv/nas/voice-lab
server   gunicorn, 1 worker / 8 threads, bound to 127.0.0.1:5000
public   https://voicelab.abhijith-sriram.in  (existing cloudflared tunnel)
```

**One worker is a correctness decision, not a throughput one.** `db.py` opens a
fresh SQLite connection per call with the default rollback journal and a 5 s busy
timeout. Several *processes* writing that database is how you get "database is
locked" in the middle of a demo; several threads in one process is not.

```bash
sudo systemctl restart voicelab      # after a git pull
journalctl -u voicelab -f            # logs
sudo bash deploy/install.sh          # idempotent full install
python -m unittest discover -s tests # 36 tests
```

`VOICELAB_BEHIND_PROXY=1` is required. Without it Flask builds `http://` URLs,
the browser calls the page insecure, and `getUserMedia` refuses the
microphone — the app looks broken in a way that has nothing to do with audio.

---

## 10. Open questions for the team

1. **Do we add a high-arousal label?** "Tense / on edge" alongside "low". If the
   two separate in *opposite* directions from the same baseline, that is a far
   stronger result than a single AUC — and it directly tests §2.1.
2. **Do we record whether mood induction was used?** One checkbox. If induced
   and spontaneous low-mood samples differ, we would have demonstrated *why*
   acted corpora overstate performance.
3. **Backups.** `/srv/nas/voice-lab` has no automated backup. `db.py` argues
   recordings are never deleted because an experiment whose samples can vanish
   cannot be reproduced — that argument applies to disk failure too.
4. **Cloudflare Access on `/admin`.** The admin code is currently the entire
   access model on a public hostname.
5. **Do we report the absolute score at all**, given §7.2, or lead with the
   contrast and present the absolute number only as a documented null?

---

## 11. First results — 6 September 2026

First real corpus, collected over two days from volunteers recording on their
own phones. Numbers below are from the analysis run of 6 September; regenerate
with the **Analyse** button rather than trusting these once more data lands.

### 11.1 What was collected

```
registered users        16
users who recorded      11
recordings             122
extraction failures      0
durations          7.2 - 45.1 s
languages          86 unspecified, Tamil 18, Hindi 12, English 6
```

| subject | ordinary rounds | low rounds | status |
|---|---|---|---|
| `djk` | 4 | 6 | scorable |
| `overthinker234` | 4 | 1 | scorable |
| `sha` | 4 | 1 | scorable |
| `circinus` | 4 | 0 | one low round from scorable |
| `smp` | 4 | 0 | one low round from scorable |
| `talkative` | 4 | 0 | one low round from scorable |
| `hahaha` | 2 | 1 | low round unusable — baseline incomplete |
| `jayesh`, `sabari` | 2 | 0 | needs 2 more ordinary rounds |
| `abhijith_sriram`, `farzana006` | 1 | 0 | needs 3 more |

**Only three subjects are scorable, and `djk` supplies 18 of the 24 cases.**
Every result below is constrained by that before it is constrained by anything
else.

### 11.2 What works

**Feature extraction succeeded on 122 of 122 recordings.** Eleven people, their
own handsets, their own rooms, four languages, no `ffmpeg`, no failures. This
was a genuine risk — the vendored DSP had only ever seen a synthetic corpus —
and it is now answered.

**The prompt design produces the cognitive-load effect it was built for.**
Median raw values over ordinary recordings:

| prompt | `pause_ratio` | rate (syll/s) |
|---|---|---|
| sustained vowel | 0.005 | 0.87 |
| counting | 0.260 | 1.95 |
| naming animals | 0.490 | 1.31 |

Naming animals draws close to **double** the pausing of counting, and the
direction holds in **10 of 11 subjects** (median gap +0.212). The positive
control passes.

### 11.3 The pipeline does not separate mood yet

```
subjects scored              3
controls / cases          9 / 24
control mean / case mean   29.36 / 31.46
separation                 +2.09
mean within-subject AUC    0.432
  djk                      0.074
  overthinker234           0.444
  sha                      0.778
```

Below chance — but three values spanning 0.07 to 0.78 are not a measurement.
**Do not report an AUC from this round.** The honest statement is that the study
is not yet powered to answer its own question.

### 11.4 Jitter and shimmer are not measuring voices

Across 41 sustained-vowel recordings:

```
median jitter    8.03%
range            0.51% - 25.99%
above 3%         76% of recordings
```

Clinical norm on sustained phonation is **under about 1%**; above 3% indicates
pathology. Values above 20% are not physiologically possible from someone
holding a steady note — that is a period detector failing, not eleven damaged
larynxes. `djk`, who has the most data and the most inverted AUC, sits at
**22.24%**.

Part of this is structural and was predictable. **At 16 kHz one sample is
62.5 µs.** A 200 Hz voice has a 5000 µs period, so 1% jitter is a period
difference of about 50 µs — *smaller than a single sample*. The quantisation
floor alone lands near 1.25%, above the clinical threshold, and worsens with
pitch. Clinical perturbation work uses 44.1 kHz for exactly this reason.

§3.4 defends 16 kHz against *lossy coding*, which it does correctly. It does not
establish that 16 kHz suffices for perturbation measures — and it does not.

Quantisation alone does not explain a 0.51–25.99% spread, so period detection is
failing outright on many recordings as well.

**These two features carry 0.38 of the feature weight.**

#### The sensitivity run

Rather than edit the weights, `run_analysis` now scores the corpus a second time
with the pair removed (`SENSITIVITY_DROP`). No vendored code changes — the
features are simply absent from the vector, and `compute_voice_stress_signal`
divides by the weight it actually used, so the surviving five renormalise over
0.62 by themselves.

| | with all 7 | without the pair |
|---|---|---|
| mean within-subject AUC | 0.432 | **0.506** |
| separation | +2.09 | +0.94 |
| `djk` | 0.074 | 0.130 |
| `overthinker234` | 0.444 | 0.500 |
| `sha` | 0.778 | 0.889 |

**All three subjects improve.** With n=3 that consistency carries more weight
than the size of the change.

Two conclusions, and the second matters as much as the first:

1. The two features are **actively costing accuracy** on this corpus.
2. **0.506 is still a coin flip.** Removing them stops the harm; it does not
   produce a working detector. They are *a* problem, not *the* problem.

### 11.5 The one consistent mood finding

Per subject, low-mood minus ordinary, raw medians over the two speech prompts:

| subject | Δ `pause_ratio` | Δ rate | Δ f0 |
|---|---|---|---|
| `djk` | −0.036 | +0.258 | **−6.5 Hz** |
| `hahaha` | +0.023 | +1.249 | **−16.7 Hz** |
| `overthinker234` | +0.075 | −0.293 | **−1.3 Hz** |
| `sha` | −0.179 | +0.163 | **−30.3 Hz** |

**f0 falls when low in 4 of 4** — three clearly, one negligibly, none the other
way. That is the only consistent signal in the corpus.

**This contradicts what §2.1 led us to expect.** We predicted low-arousal
sadness: slower, more pausing. `pause_ratio` splits 2 up / 2 down, and speaking
rate goes *faster* in 3 of 4. Whatever subjects produce when they mark
themselves "Low", it is not the psychomotor-retardation profile. Anyone
presenting this must not assert the textbook pattern — our own data does not
show it.

f0 tracking itself looks sound: 107–252 Hz across subjects, male and female
ranges where expected, no sign of octave doubling.

### 11.6 A bug this round exposed

The dashboard was printing `WARNING: the pipeline is not resolving a large known
effect`. **That was a false alarm, and the fault was ours.** The control tested
whether ordinary sittings showed an easy/hard gap *in z-scores* — but ordinary
recordings are what the per-load baselines are built from, so their z-scores
centre on zero however the prompts behave. It could only ever fail.

Introducing per-load baselines (§6.1) and keeping that control (§6.2) were each
defensible; together they were not. It now reads raw values and passes at 10/11
subjects.

The lesson generalises: **a control derived from the same normalisation it is
meant to check is not a control.**

### 11.7 What the next round needs

1. **More low-mood rounds, before anything else.** `circinus`, `smp` and
   `talkative` have complete baselines and no low round — three messages would
   double the scorable n from 3 to 6. Nothing in the code moves the result as
   much.
2. **Chase the two-round subjects.** `hahaha` has already recorded a low round
   sitting unusable because their baseline stops at 2.
3. **Decide about 44.1 kHz for the sustained vowel only.** It would make jitter
   measurable; it would also break the byte-identical property with `pwiews`
   settings and require re-recording. Not worth doing until the n problem is
   fixed.
4. **Do not touch `VOICE_FEATURE_DIRECTIONS` yet.** Flipping signs to chase a
   result across three subjects is how you fit noise. Revisit at n ≥ 10.
