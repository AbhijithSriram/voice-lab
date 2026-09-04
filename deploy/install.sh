#!/usr/bin/env bash
#
# Install the voice lab as a system service on homeserver.
#
#   sudo bash /home/abhijith/voice-lab/deploy/install.sh
#
# Idempotent: safe to re-run. It touches exactly four things --
#
#   /etc/voice-lab.env            created once, never overwritten
#   /etc/systemd/system/voicelab.service
#   /etc/cloudflared/config.yml   ONE ingress rule appended, backed up first
#   /srv/nas/voice-lab           the data directory (recordings + database)
#
# It does not touch cockpit, chat, ammas-codex or locomotion-transitops.

set -euo pipefail

APP_USER=abhijith
APP_DIR=/home/abhijith/voice-lab
# Recordings and the database, together, on the 596 GB /srv partition
# (/dev/nvme0n1p6). Two reasons they stay together rather than splitting the
# audio off: / has about 22 GB free and recordings are the thing that grows,
# and the `recordings` rows point at those files by name -- separated across
# two filesystems, no single snapshot is guaranteed consistent and a restore
# has two sources to reconcile. Safe for sqlite because /srv is local ext4;
# on an NFS or CIFS mount the database would have to stay local, because
# sqlite's locking is not reliable over either.
DATA_DIR=/srv/nas/voice-lab
ENV_FILE=/etc/voice-lab.env
HOSTNAME_FQDN=voicelab.abhijith-sriram.in
PORT=5000
CF_CONFIG=/etc/cloudflared/config.yml

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31mFAILED: %s\033[0m\n' "$*" >&2; exit 1; }

[ "$(id -u)" -eq 0 ] || die "run this with sudo"
[ -d "$APP_DIR" ]    || die "$APP_DIR does not exist -- clone the repo first"

# ---------------------------------------------------------------- 1. packages
# Test the capability, do not assume the package. `python3 -m venv` already
# works on this box, and an apt-get that cannot resolve a name would abort the
# whole install under `set -e` for a dependency that was never missing.
if python3 -m venv --help >/dev/null 2>&1; then
    say "python venv support present"
else
    say "Installing python venv support"
    apt-get update -qq
    apt-get install -y python3-venv python3-dev         || apt-get install -y "python$(python3 -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")')-venv" python3-dev         || die "could not install venv support"
fi

# ------------------------------------------------------------------- 2. venv
# Built as the app user: a root-owned venv inside abhijith's home is a trap
# that only shows up the first time he tries to pip install something.
say "Building the virtualenv (as $APP_USER)"
install -d -o "$APP_USER" -g "$APP_USER" "$DATA_DIR"
sudo -u "$APP_USER" bash -c "
    set -e
    cd '$APP_DIR'
    [ -d .venv ] || python3 -m venv .venv
    .venv/bin/pip install --quiet --upgrade pip wheel
    .venv/bin/pip install --quiet -r requirements.txt
    .venv/bin/python -c 'import flask, numpy, scipy, gunicorn; print(\"deps OK\")'
"

# -------------------------------------------------------------- 3. secrets
# Generated here, on the box. Never in the repo, never in the chat log that
# produced this file.
if [ ! -f "$ENV_FILE" ]; then
    say "Creating $ENV_FILE"
    SECRET=$(python3 -c 'import secrets; print(secrets.token_hex(32))')
    ADMIN=$(python3 -c 'import secrets; print(secrets.token_urlsafe(12))')
    cat > "$ENV_FILE" <<ENV
# Voice lab runtime secrets. Read by systemd, which is not a shell:
# do not quote values, do not use \$substitution.
VOICELAB_SECRET_KEY=$SECRET
VOICELAB_ADMIN_CODE=$ADMIN
VOICELAB_BEHIND_PROXY=1
VOICELAB_DATA_DIR=$DATA_DIR
ENV
    chown root:root "$ENV_FILE"
    chmod 600 "$ENV_FILE"
    NEW_ENV=1
else
    say "$ENV_FILE already exists -- left untouched"
    NEW_ENV=0
fi

# -------------------------------------------------------------- 4. systemd
say "Installing the service"
install -m 0644 "$APP_DIR/deploy/voicelab.service" /etc/systemd/system/voicelab.service
systemctl daemon-reload
systemctl enable voicelab >/dev/null
systemctl restart voicelab

sleep 3
systemctl is-active --quiet voicelab || {
    journalctl -u voicelab -n 30 --no-pager
    die "voicelab did not stay up (log above)"
}
curl -fsS -o /dev/null "http://127.0.0.1:$PORT/login" || die "nothing answering on 127.0.0.1:$PORT"
say "Service is up and answering on 127.0.0.1:$PORT"

# ------------------------------------------------------------ 5. cloudflared
if grep -q "$HOSTNAME_FQDN" "$CF_CONFIG"; then
    say "Tunnel already routes $HOSTNAME_FQDN -- config left untouched"
else
    say "Adding one ingress rule to the existing tunnel"
    BACKUP="$CF_CONFIG.bak.$(date +%Y%m%d-%H%M%S)"
    cp -a "$CF_CONFIG" "$BACKUP"
    echo "    backup: $BACKUP"

    # Insert immediately before the catch-all. Cloudflare matches ingress rules
    # top to bottom and http_status:404 matches everything, so a rule appended
    # after it would be dead -- and the four existing hostnames keep their
    # order and their precedence either way.
    awk -v host="$HOSTNAME_FQDN" -v port="$PORT" '
        !done && /^[[:space:]]*-[[:space:]]*service:[[:space:]]*http_status:404/ {
            print "  - hostname: " host
            print "    service: http://localhost:" port
            print "    originRequest:"
            print "      # 8 MB WAV uploads, with feature extraction inline."
            print "      connectTimeout: 30s"
            done = 1
        }
        { print }
        END { if (!done) exit 3 }
    ' "$BACKUP" > "$CF_CONFIG" || die "no catch-all rule found -- $CF_CONFIG restored from $BACKUP"

    if ! cloudflared tunnel ingress validate --config "$CF_CONFIG"; then
        cp -a "$BACKUP" "$CF_CONFIG"
        die "ingress validation failed -- $CF_CONFIG rolled back, tunnel untouched"
    fi

    say "Reloading cloudflared"
    systemctl restart cloudflared
    sleep 3
    systemctl is-active --quiet cloudflared || {
        cp -a "$BACKUP" "$CF_CONFIG"
        systemctl restart cloudflared
        die "cloudflared would not start -- config rolled back and restarted"
    }
fi

say "Done"
echo
echo "  https://$HOSTNAME_FQDN"
echo
if [ "$NEW_ENV" = "1" ]; then
    echo "  Admin code (type it on the signup form to create an admin account):"
    echo
    echo "      $(grep '^VOICELAB_ADMIN_CODE=' "$ENV_FILE" | cut -d= -f2-)"
    echo
    echo "  It is in $ENV_FILE. Anyone who has it is an admin, so treat it"
    echo "  like a password. Change it there, then: sudo systemctl restart voicelab"
    echo
fi
echo "  Logs:    journalctl -u voicelab -f"
echo "  Restart: sudo systemctl restart voicelab"
echo
