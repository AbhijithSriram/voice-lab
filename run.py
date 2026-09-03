"""Start the voice lab.

    python run.py

For a homeserver behind Cloudflare Tunnel, set the environment first:

    VOICELAB_SECRET_KEY=<random hex>  VOICELAB_ADMIN_CODE=<your code>
    VOICELAB_BEHIND_PROXY=1  VOICELAB_HOST=0.0.0.0  python run.py

Flask's development server is what this uses. For twenty or thirty people
recording occasionally that is genuinely fine; if it ever needs to be more,
put waitress or gunicorn in front of `voicelab.app:create_app()`.
"""

from __future__ import annotations

import config
from voicelab.app import create_app

app = create_app()

if __name__ == "__main__":
    print(f"voice lab on http://{config.HOST}:{config.PORT}")
    if config.SECRET_KEY_IS_EPHEMERAL:
        print("  ! VOICELAB_SECRET_KEY unset - sessions reset on restart")
    if config.ADMIN_CODE_IS_DEFAULT:
        print(f"  ! VOICELAB_ADMIN_CODE unset - default code is '{config.ADMIN_CODE}'")
    print(f"  data in {config.DATA_DIR}")
    app.run(host=config.HOST, port=config.PORT, debug=False)
