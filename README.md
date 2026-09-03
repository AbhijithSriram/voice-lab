# voice lab

A standalone app for collecting **labelled voice samples from real people** and
testing whether the pwiews acoustic pipeline can tell them apart.

It has nothing to do with CRPF, welfare, or SIH26186. It is one question:

> Given a person's neutral speech as a baseline, does the pipeline score their
> sad speech higher — and by how much?

The pipeline in `voicelab/dsp/` is a **vendored copy** of
`personnel-welfare-intelligence/backend/voice_pipeline/`, byte-identical apart
from its import lines. That is deliberate: if the two diverge, a result measured
here stops being evidence about the pipeline there. Only
`voicelab/dsp/settings.py` is meant to be edited.

---

## Why this exists

The pwiews voice pipeline was validated against synthetic audio that was
generated using the same acoustic directions the detector looks for. That proves
the DSP recovers what was injected — a real and necessary check — and proves
nothing about real speech. This app breaks that circle.

**Read this before interpreting any result:** subjects are asked to *speak as
they would when sad*. That is performed affect, not felt affect. A good result
shows the pipeline can detect the acoustic correlates people produce when
portraying sadness. That is genuinely useful and it is not the same as detecting
sadness. Do not let the number outgrow the design.

---

## Run it

```bash
pip install -r requirements.txt
python run.py
```

Then open `http://127.0.0.1:5000`. Sign up with the admin code (default
`let-me-in`) to get the dashboard; sign up without it to be a subject.

### On a homeserver, behind Cloudflare Tunnel

```bash
export VOICELAB_SECRET_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")
export VOICELAB_ADMIN_CODE='something-only-you-know'
export VOICELAB_BEHIND_PROXY=1
export VOICELAB_HOST=0.0.0.0
python run.py
```

Then point the tunnel at `http://localhost:5000`.

**HTTPS is not optional.** `getUserMedia` only works in a secure context, so the
microphone will silently refuse to open over plain http on anything but
localhost. Cloudflare Tunnel gives you https for free, which is why
`VOICELAB_BEHIND_PROXY=1` matters — without it Flask builds `http://` URLs
behind the tunnel and the browser treats the page as insecure.

| Variable | Default | What it does |
|---|---|---|
| `VOICELAB_SECRET_KEY` | random per start | Session signing. Unset means everyone is logged out on restart. |
| `VOICELAB_ADMIN_CODE` | `let-me-in` | Typed on the signup form to get an admin account. |
| `VOICELAB_BEHIND_PROXY` | `0` | Set to `1` behind Cloudflare Tunnel or any reverse proxy. |
| `VOICELAB_DATA_DIR` | `./data` | Where the database and audio live. |
| `VOICELAB_HOST` / `VOICELAB_PORT` | `127.0.0.1` / `5000` | Bind address. |

The dashboard shows a warning banner while the first two are unset.

---

## How to run the collection

**1. Get neutral samples first.** Each subject needs more than
`VOICE_BASELINE_MIN_SAMPLES` (default 3) neutral recordings before anything can
be scored — the first 3 build their baseline and produce no score at all. Ask
for **5 or 6** neutral samples per person, not 4. The extras are what become
*controls*, and without controls the sad scores mean nothing.

**2. Then get sad samples.** Ask people to talk the way they would when they are
down. Everyone does this differently, which is fine — every comparison is
against that person's own baseline, never against anyone else.

**3. Spread it out.** Samples recorded back-to-back in one sitting share a room,
a microphone position and a mood. Samples across two days are a much harder and
much more honest test. Subjects can log in and record whenever they like.

**4. Keep the device constant per person.** Different microphones change the
measurements. If someone records neutral on a laptop and sad on a phone, you
have measured the phone. The signup form has a notes field — ask people to say
what they are recording on.

**5. Any language.** The pipeline never converts speech to text, so it does not
care. Fixed prompts (everyone reads the same words) and free prompts (talk about
anything) are both collected, because it is not obvious in advance which
separates better — and finding that out is itself a result.

---

## Reading the results

Hit **Run analysis** on the dashboard. It re-reads the stored features every
time, so you can edit settings and re-run without re-recording anything.

| Number | What it means |
|---|---|
| **Control mean** | Neutral recordings scored against the same person's neutral baseline. **Should be near zero.** If it is not, the pipeline is measuring rooms and microphones, and everything else on the page is standing on sand. |
| **Case mean** | Sad recordings against the same baseline. |
| **Separation** | Case mean − control mean. The headline. |
| **Within-subject AUC** | Pick one control and one case from the *same person* at random: how often does the case score higher? 0.5 is a coin flip. **This is the honest number.** |
| **Pooled AUC** | Same question ignoring who is who. Can be flattered by some people simply scoring higher overall — when the two disagree, believe the within-subject one. |
| **Per-feature shift** | How far each acoustic feature moved, and in which direction. |

### The per-feature table is the actual finding

When the headline number disappoints, that table tells you *why*. And there is a
specific thing to expect:

**Sadness is not stress.** The current directions in `settings.py` came from the
pressured/anxious-speech literature. Much of the sad/depressed-speech literature
reports the opposite for three of the seven features:

| | pressured speech | sad speech |
|---|---|---|
| F0 | raised | lowered, flatter |
| Speaking rate | faster | slower |
| Pause ratio | lower | **higher** |

So `speaking_rate_syllables_per_sec` and `pause_ratio` may well shift *negative*.
That is a finding, not a bug — it says the pipeline is tuned for a different
affective state than the one you collected. Jitter and shimmer are the two most
likely to hold their direction across both, because both reflect loss of fine
laryngeal control rather than a communicative choice.

If that is what you see, the honest options are, in order:

1. **Report it.** "Our pipeline is tuned for acute stress; on sad speech two
   features invert, which is consistent with the literature" is a stronger thing
   to say to a judge than a number.
2. Flip those two directions and re-run — but then be clear you are now
   detecting sadness, not stress, and that you tuned on your own test set.
3. Add a second direction set and pick per use case.

Option 2 has a real cost: with 20–30 subjects, flipping directions after seeing
the result *is* fitting to your test set. If you do it, collect a fresh batch of
subjects afterwards and check the number holds.

---

## What to tune

Everything is in `voicelab/dsp/settings.py`. In rough order of impact:

- **`VOICE_BASELINE_MIN_SAMPLES`** (3) — the most consequential number. Lower it
  to 2 and every subject gains a scored control at the cost of a noisier
  baseline.
- **`VOICE_FEATURE_DIRECTIONS`** — see above.
- **`VOICE_FEATURE_WEIGHTS`** — must sum to 1.0; the module asserts it at import.
  Move weight toward whichever features actually shifted.
- **`VOICE_DEVIATION_SATURATION_SD`** (3.0) — if nearly every sad sample pins at
  100 you are throwing away the ranking; if nothing exceeds 40 it is too high.
- **`VOICE_F0_MIN_HZ` / `MAX_HZ`** — only if a subject's `f0_mean_hz` looks
  wrong in their detail page.

Every run stores a copy of the settings it used, so old results stay
interpretable after you change them.

---

## What it does with recordings

Audio files are kept. The parent project deletes raw audio immediately after
measuring it (`RETENTION_RAW_AUDIO_DAYS = 0`) and that is right for a
deployment — but a lab that discards its inputs cannot re-run its analysis, and
re-running the analysis is the entire point here. Recordings live in
`data/audio/` and `data/` is gitignored.

Tell your subjects that. They are lending you their voices for an experiment,
and "we keep the recording so we can re-measure it" is a different deal from
what the pwiews privacy policy describes.

---

## Layout

```
voice-lab/
├── run.py                     entry point
├── config.py                  deployment settings from the environment
├── voicelab/
│   ├── app.py                 Flask routes: signup, record, admin
│   ├── db.py                  SQLite schema and helpers
│   ├── audio.py               WAV validation, storage, feature extraction
│   ├── analysis.py            the experiment: baselines, controls, cases, AUC
│   ├── dsp/                   vendored acoustic pipeline
│   │   └── settings.py        ← the file you edit
│   ├── templates/
│   └── static/recorder.js     browser-side 16 kHz mono WAV encoder
└── data/                      database + audio (gitignored)
```

### Why the browser encodes WAV instead of using MediaRecorder

`MediaRecorder` produces WebM/Opus. Opus is a lossy *perceptual* codec, and what
it discards first is exactly what this pipeline measures — jitter is
cycle-to-cycle variation in glottal period, shimmer is cycle-to-cycle variation
in amplitude, and both are sub-perceptual. Measuring them on decoded Opus would
produce numbers about the codec. So `recorder.js` captures raw Float32 PCM from
the Web Audio API, box-filter downsamples to 16 kHz, and writes the WAV header
itself. Nothing is transcoded, and the server needs no ffmpeg.

The browser's own `echoCancellation`, `noiseSuppression` and `autoGainControl`
are all switched off, for the same reason: every one is a time-varying filter,
and AGC in particular would flatten the exact intensity variation
`intensity_rms_cv` measures.

Server-side, a sample-rate mismatch is **refused rather than resampled** —
resampling changes period lengths, and jitter *is* a period-length measurement.

---

## Honest limits of this setup

- **Access control is one shared admin code.** Anybody who has it is an admin.
  Fine for a lab of people you know; not fine for anything else.
- **No consent record beyond signing up.** If this were going anywhere near an
  ethics process it would need one.
- **Performed affect, not felt affect.** Said above; worth saying twice.
- **20–30 subjects is small.** With 5 neutral and 3 sad each you get roughly 2
  controls and 3 cases per person. That is enough to see a large effect and not
  enough to trust a small one. Treat a within-subject AUC of 0.6 as "no signal
  found", not as "weak signal found".
- **Flask's development server.** Genuinely fine at this scale; put waitress in
  front of `voicelab.app:create_app()` if it ever isn't.
