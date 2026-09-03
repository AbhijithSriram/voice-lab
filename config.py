"""Deployment configuration for the voice lab.

Everything here is read from the environment with a development default, so
running locally needs no setup and running on a homeserver behind Cloudflare
needs no code change.

Set these before exposing it:

    VOICELAB_SECRET_KEY   session signing key. Generate with
                          `python -c "import secrets; print(secrets.token_hex(32))"`
    VOICELAB_ADMIN_CODE   the code someone types on the signup form to get an
                          admin account. Anyone who knows it becomes an admin,
                          so treat it like a password.
    VOICELAB_DATA_DIR     where the database and audio live. Default ./data
    VOICELAB_BEHIND_PROXY set to 1 when running behind Cloudflare Tunnel or any
                          reverse proxy, so Flask reads X-Forwarded-Proto and
                          builds https:// URLs instead of http://
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = Path(os.environ.get("VOICELAB_DATA_DIR", BASE_DIR / "data"))
AUDIO_DIR = DATA_DIR / "audio"
DB_PATH = DATA_DIR / "voicelab.sqlite3"

# A generated key means sessions do not survive a restart, which is correct for
# a dev default: it fails visibly (everyone is logged out) rather than silently
# shipping a known key to a public host.
SECRET_KEY = os.environ.get("VOICELAB_SECRET_KEY") or secrets.token_hex(32)
SECRET_KEY_IS_EPHEMERAL = "VOICELAB_SECRET_KEY" not in os.environ

ADMIN_CODE = os.environ.get("VOICELAB_ADMIN_CODE", "let-me-in")
ADMIN_CODE_IS_DEFAULT = "VOICELAB_ADMIN_CODE" not in os.environ

BEHIND_PROXY = os.environ.get("VOICELAB_BEHIND_PROXY", "0") == "1"

HOST = os.environ.get("VOICELAB_HOST", "127.0.0.1")
PORT = int(os.environ.get("VOICELAB_PORT", "5000"))

# Upload ceiling. A 60 s 16 kHz 16-bit mono WAV is about 1.9 MB; 8 MB leaves
# room without letting anybody post a film.
MAX_UPLOAD_BYTES = 8 * 1024 * 1024

# Recording length the browser enforces. The lower bound is the pipeline's
# own minimum (VOICE_MIN_DURATION_SEC); below it a sample is rejected outright.
MIN_RECORD_SEC = 6
MAX_RECORD_SEC = 45
