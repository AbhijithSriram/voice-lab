"""The Flask application: signup, recording, and the admin dashboard.

Three surfaces:

    /signup, /login       anybody can create a subject account
    /record               a subject records, labels and submits samples
    /admin                subjects, sample counts, and the analysis

On the admin code: signup accepts an optional code, and anybody who types the
right one gets an admin account. That is the whole access model. It suits a
lab of twenty or thirty people you know, and it is stated plainly rather than
dressed up -- see `backend`-style notes in README.md for what would have to
change before this held anything that mattered.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

from flask import (
    Flask,
    abort,
    flash,
    g,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    session,
    url_for,
)
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.security import check_password_hash, generate_password_hash

import config
from voicelab import analysis, audio, db
from voicelab.dsp import settings as dsp_settings

# Labels a subject may submit. 'neutral' is special -- it builds the baseline.
# Extend this list to run a different experiment; the analysis treats anything
# that is not 'neutral' as a case and reports each label separately.
LABELS = [
    ("neutral", "Neutral — how you normally speak"),
    ("sad", "Sad — speak how you would when you are down"),
]

LABEL_VALUES = {value for value, _ in LABELS}


def _now() -> str:
    """Current UTC timestamp, ISO 8601."""
    return datetime.now(timezone.utc).isoformat()


def login_required(view: Callable) -> Callable:
    """Require any signed-in user.

    Args:
        view: The view function.

    Returns:
        The wrapped view.
    """

    @wraps(view)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        if not g.user:
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)

    return wrapped


def admin_required(view: Callable) -> Callable:
    """Require a signed-in admin.

    Args:
        view: The view function.

    Returns:
        The wrapped view. Non-admins get 403 rather than a redirect, so a
        subject who guesses the URL is told no rather than bounced somewhere
        that looks like it might have worked.
    """

    @wraps(view)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        if not g.user:
            return redirect(url_for("login", next=request.path))
        if not g.user["is_admin"]:
            abort(403)
        return view(*args, **kwargs)

    return wrapped


def create_app() -> Flask:
    """Build the application.

    Returns:
        The configured Flask app.
    """
    app = Flask(__name__)
    app.config["SECRET_KEY"] = config.SECRET_KEY
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_BYTES

    if config.BEHIND_PROXY:
        # Cloudflare Tunnel terminates TLS, so without this Flask builds
        # http:// URLs and the browser refuses getUserMedia on what it then
        # treats as an insecure page.
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

    db.init_db()

    @app.before_request
    def load_user() -> None:
        """Attach the signed-in user to ``g`` for every request."""
        user_id = session.get("user_id")
        g.user = (
            db.query_one("SELECT * FROM users WHERE id = ?", (user_id,))
            if user_id
            else None
        )

    @app.context_processor
    def template_globals() -> Dict[str, Any]:
        """Values every template can use."""
        return {
            "user": g.get("user"),
            "min_record_sec": config.MIN_RECORD_SEC,
            "max_record_sec": config.MAX_RECORD_SEC,
            "sample_rate": dsp_settings.VOICE_SAMPLE_RATE_HZ,
        }

    # ---------------------------------------------------------------- auth

    @app.route("/")
    def index() -> Any:
        """Send people where they belong."""
        if not g.user:
            return redirect(url_for("login"))
        return redirect(url_for("admin") if g.user["is_admin"] else url_for("record"))

    @app.route("/signup", methods=["GET", "POST"])
    def signup() -> Any:
        """Create a subject (or admin) account."""
        # Not `== "GET"`: werkzeug routes HEAD to this view too, and a HEAD
        # falling through to the POST branch is handled as an empty form --
        # answering an uptime check with 400 on a form it never submitted.
        if request.method != "POST":
            return render_template("signup.html")

        username = (request.form.get("username") or "").strip().lower()
        password = request.form.get("password") or ""
        display_name = (request.form.get("display_name") or "").strip()
        notes = (request.form.get("notes") or "").strip()
        admin_code = (request.form.get("admin_code") or "").strip()

        if len(username) < 3 or not username.replace("_", "").replace("-", "").isalnum():
            flash("Username must be at least 3 characters, letters/numbers/-/_ only.")
            return render_template("signup.html"), 400
        if len(password) < 8:
            flash("Password must be at least 8 characters.")
            return render_template("signup.html"), 400
        if db.query_one("SELECT id FROM users WHERE username = ?", (username,)):
            flash("That username is taken.")
            return render_template("signup.html"), 400

        is_admin = 1 if (admin_code and admin_code == config.ADMIN_CODE) else 0
        if admin_code and not is_admin:
            flash("That admin code is not right.")
            return render_template("signup.html"), 400

        user_id = db.execute(
            "INSERT INTO users (username, password_hash, is_admin, display_name, notes, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                username,
                generate_password_hash(password),
                is_admin,
                display_name or username,
                notes,
                _now(),
            ),
        )
        session["user_id"] = user_id
        return redirect(url_for("index"))

    @app.route("/login", methods=["GET", "POST"])
    def login() -> Any:
        """Sign in."""
        # See signup(): HEAD must render the form, not fail an empty login.
        if request.method != "POST":
            return render_template("login.html")

        username = (request.form.get("username") or "").strip().lower()
        password = request.form.get("password") or ""
        row = db.query_one("SELECT * FROM users WHERE username = ?", (username,))

        # Same message either way: which usernames exist is not something a
        # login form should tell a stranger.
        if not row or not check_password_hash(row["password_hash"], password):
            flash("Wrong username or password.")
            return render_template("login.html"), 403

        session.clear()
        session["user_id"] = row["id"]
        nxt = request.args.get("next")
        return redirect(nxt if nxt and nxt.startswith("/") else url_for("index"))

    @app.route("/logout", methods=["POST"])
    def logout() -> Any:
        """Sign out."""
        session.clear()
        return redirect(url_for("login"))

    # ------------------------------------------------------------ subject

    @app.route("/record")
    @login_required
    def record() -> Any:
        """The recording screen."""
        prompts = db.query("SELECT * FROM prompts WHERE is_active = 1 ORDER BY kind, id")
        mine = db.query(
            "SELECT label, COUNT(*) AS n FROM recordings WHERE user_id = ? GROUP BY label",
            (g.user["id"],),
        )
        counts = {row["label"]: row["n"] for row in mine}
        # The automatic prompt this subject first used becomes their anchor,
        # and the others are locked out. The contrast cancels nuisances inside
        # a sitting, but comparing one sitting's gap against another's assumes
        # both measured from the same easy end. Counting on Monday and weekdays
        # on Tuesday puts the task difference straight back, one level up.
        anchor = db.query_one(
            "SELECT r.prompt_id FROM recordings r JOIN prompts p ON p.id = r.prompt_id"
            " WHERE r.user_id = ? AND p.load = 'automatic'"
            " ORDER BY r.created_at LIMIT 1",
            (g.user["id"],),
        )
        pinned_automatic = anchor["prompt_id"] if anchor else None
        recent = db.query(
            "SELECT * FROM recordings WHERE user_id = ? ORDER BY created_at DESC LIMIT 10",
            (g.user["id"],),
        )
        return render_template(
            "record.html",
            prompts=prompts,
            labels=LABELS,
            counts=counts,
            recent=recent,
            pinned_automatic=pinned_automatic,
            baseline_needed=dsp_settings.VOICE_BASELINE_MIN_SAMPLES,
        )

    @app.route("/api/recordings", methods=["POST"])
    @login_required
    def upload_recording() -> Any:
        """Accept one recording.

        Body: multipart form with ``audio`` (a WAV file), ``label``,
        optional ``prompt_id``, ``session_id``, ``intensity``, ``language``.
        """
        upload = request.files.get("audio")
        if upload is None:
            return jsonify({"error": "no audio in request"}), 400

        label = (request.form.get("label") or "").strip().lower()
        if label not in LABEL_VALUES:
            return jsonify({"error": f"unknown label {label!r}"}), 400

        try:
            stored = audio.store(upload.read(), g.user["username"], label)
        except audio.BadRecording as exc:
            return jsonify({"error": str(exc)}), 400

        prompt_id = request.form.get("prompt_id") or None
        intensity = request.form.get("intensity")
        # Minted by the browser once per visit to /record and sent back with
        # every upload from that visit. The pairing it enables is the whole
        # point of the load contrast, so it is bounded and stored verbatim
        # rather than parsed -- an unrecognised value costs one session, while
        # trusting client text into the database costs more than that.
        session_id = (request.form.get("session_id") or "").strip()[:64] or None
        db.execute(
            "INSERT INTO recordings (user_id, prompt_id, label, intensity, language,"
            " session_id, filename, duration_sec, sample_rate, created_at, features_json,"
            " extract_error)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                g.user["id"],
                int(prompt_id) if prompt_id and prompt_id.isdigit() else None,
                label,
                int(intensity) if intensity and intensity.isdigit() else None,
                (request.form.get("language") or "").strip()[:40] or None,
                session_id,
                stored.filename,
                stored.duration_sec,
                stored.sample_rate,
                _now(),
                json.dumps(stored.features) if stored.features else None,
                stored.error,
            ),
        )

        total = db.query_one(
            "SELECT COUNT(*) AS n FROM recordings WHERE user_id = ? AND label = ?",
            (g.user["id"], label),
        )
        return jsonify(
            {
                "ok": True,
                "duration_sec": stored.duration_sec,
                "measured": stored.features is not None,
                # Surfaced to the subject, gently. A recording the extractor
                # could not measure is kept, but they should know to try again
                # rather than believe they have contributed a usable sample.
                "note": stored.error,
                "label": label,
                "label_count": total["n"] if total else 0,
            }
        ), 201

    # -------------------------------------------------------------- admin

    @app.route("/admin")
    @admin_required
    def admin() -> Any:
        """Subjects and their sample counts."""
        subjects = db.query(
            """
            SELECT u.id, u.username, u.display_name, u.notes, u.created_at, u.is_admin,
                   COUNT(r.id) AS total,
                   SUM(CASE WHEN r.label = 'neutral' THEN 1 ELSE 0 END) AS neutral_n,
                   SUM(CASE WHEN r.label != 'neutral' THEN 1 ELSE 0 END) AS case_n,
                   -- The r.id guard matters: without it the LEFT JOIN's null
                   -- row for a subject with no recordings counts as one
                   -- unmeasured recording.
                   SUM(CASE WHEN r.id IS NOT NULL AND r.features_json IS NULL
                            THEN 1 ELSE 0 END) AS unmeasured_n
            FROM users u
            LEFT JOIN recordings r ON r.user_id = u.id
            GROUP BY u.id
            ORDER BY total DESC, u.username
            """
        )
        runs = db.query(
            "SELECT id, created_at, summary FROM analyses ORDER BY created_at DESC LIMIT 10"
        )
        return render_template(
            "admin.html",
            subjects=subjects,
            runs=runs,
            baseline_needed=dsp_settings.VOICE_BASELINE_MIN_SAMPLES,
            insecure_secret=config.SECRET_KEY_IS_EPHEMERAL,
            default_admin_code=config.ADMIN_CODE_IS_DEFAULT,
        )

    @app.route("/admin/subject/<int:user_id>")
    @admin_required
    def subject_detail(user_id: int) -> Any:
        """Everything held about one subject."""
        subject = db.query_one("SELECT * FROM users WHERE id = ?", (user_id,))
        if not subject:
            abort(404)
        recordings = db.query(
            "SELECT r.*, p.text AS prompt_text FROM recordings r"
            " LEFT JOIN prompts p ON p.id = r.prompt_id"
            " WHERE r.user_id = ? ORDER BY r.created_at",
            (user_id,),
        )
        for rec in recordings:
            rec["features"] = json.loads(rec["features_json"]) if rec["features_json"] else None
        result = analysis.analyse_subject(subject["username"], recordings).to_dict()
        return render_template(
            "subject.html",
            subject=subject,
            recordings=recordings,
            result=result,
            feature_names=list(dsp_settings.VOICE_COMPARISON_FEATURE_NAMES),
        )

    @app.route("/admin/audio/<path:filename>")
    @admin_required
    def admin_audio(filename: str) -> Any:
        """Serve one stored recording, for listening back."""
        return send_from_directory(config.AUDIO_DIR, filename)

    @app.route("/admin/analyse", methods=["POST"])
    @admin_required
    def run_analysis_route() -> Any:
        """Run the analysis over every subject and store the result."""
        # LEFT JOIN, not JOIN: a recording whose prompt was later deactivated
        # still belongs in the absolute analysis. It drops out of the load
        # contrast on its own, because `load` comes back NULL.
        rows = db.query(
            "SELECT u.username, r.label, r.created_at, r.features_json, r.extract_error,"
            " r.session_id, p.load"
            " FROM recordings r"
            " JOIN users u ON u.id = r.user_id"
            " LEFT JOIN prompts p ON p.id = r.prompt_id"
            " ORDER BY u.username, r.created_at"
        )
        by_subject: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            by_subject.setdefault(row["username"], []).append(
                {
                    "label": row["label"],
                    "created_at": row["created_at"],
                    "extract_error": row["extract_error"],
                    "session_id": row["session_id"],
                    "load": row["load"],
                    "features": json.loads(row["features_json"])
                    if row["features_json"]
                    else None,
                }
            )

        report = analysis.run_analysis(by_subject)
        summary = analysis.summarise(report)
        db.execute(
            "INSERT INTO analyses (created_at, settings_json, results_json, summary)"
            " VALUES (?, ?, ?, ?)",
            (_now(), json.dumps(report["settings"]), json.dumps(report), summary),
        )
        return redirect(url_for("results"))

    @app.route("/admin/results")
    @app.route("/admin/results/<int:run_id>")
    @admin_required
    def results(run_id: Optional[int] = None) -> Any:
        """Show one analysis run, most recent by default."""
        if run_id is None:
            row = db.query_one("SELECT * FROM analyses ORDER BY created_at DESC LIMIT 1")
        else:
            row = db.query_one("SELECT * FROM analyses WHERE id = ?", (run_id,))
        if not row:
            flash("No analysis has been run yet.")
            return redirect(url_for("admin"))

        runs = db.query("SELECT id, created_at, summary FROM analyses ORDER BY created_at DESC")
        return render_template(
            "results.html",
            run=row,
            report=json.loads(row["results_json"]),
            runs=runs,
        )

    @app.route("/admin/export.json")
    @admin_required
    def export_json() -> Any:
        """Every recording's metadata and features, for offline analysis."""
        rows = db.query(
            "SELECT u.username, r.label, r.intensity, r.language, r.filename,"
            " r.duration_sec, r.created_at, r.features_json, r.extract_error,"
            " p.text AS prompt_text, p.kind AS prompt_kind"
            " FROM recordings r JOIN users u ON u.id = r.user_id"
            " LEFT JOIN prompts p ON p.id = r.prompt_id"
            " ORDER BY u.username, r.created_at"
        )
        for row in rows:
            row["features"] = json.loads(row.pop("features_json") or "null")
        return jsonify(
            {
                "exported_at": _now(),
                "settings": analysis.settings_snapshot(),
                "recordings": rows,
            }
        )

    @app.errorhandler(403)
    def forbidden(_exc: Any) -> Any:
        """Plain 403."""
        return render_template("error.html", code=403,
                               message="That page is for the admin account."), 403

    @app.errorhandler(413)
    def too_large(_exc: Any) -> Any:
        """Upload over the size ceiling."""
        return jsonify({"error": "recording too large"}), 413

    return app
