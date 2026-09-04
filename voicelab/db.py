"""SQLite schema and access helpers.

One job: own the database. Nothing else in the app opens a connection.

The schema is small enough to read in one sitting:

    users       one row per person who signed up, plus the admin(s)
    prompts     the fixed bank of things to read or talk about
    recordings  one row per submitted sample, with its label
    analyses    one row per analysis run, so results are comparable over time

Recordings are never deleted by the app. An experiment where samples can vanish
is an experiment whose results cannot be reproduced.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT    NOT NULL UNIQUE,
    password_hash TEXT    NOT NULL,
    is_admin      INTEGER NOT NULL DEFAULT 0,
    display_name  TEXT,
    -- Free text. Age band, first language, anything the subject wants to note.
    -- Optional by design: this is a voice experiment, not a profile.
    notes         TEXT,
    created_at    TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS prompts (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    kind      TEXT NOT NULL,          -- 'fixed' or 'free'
    text      TEXT NOT NULL,
    -- Cognitive load, and the axis the paired design turns on:
    --   'phonation'  a sustained vowel; carries jitter and shimmer
    --   'automatic'  overlearned speech (counting, weekdays); no retrieval
    --   'effortful'  verbal fluency; retrieval under load
    --   NULL         free prompts, kept for variety, excluded from the contrast
    -- Orthogonal to `kind`: 'fixed' vs 'free' is about whether everyone says
    -- the same words, `load` is about how hard it is to produce them.
    load      TEXT,
    is_active INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS recordings (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       INTEGER NOT NULL,
    prompt_id     INTEGER,
    -- The label. 'neutral' builds the baseline; anything else is compared
    -- against it. Kept as free text rather than an enum so a later experiment
    -- can add 'tired' or 'rushed' without a migration.
    label         TEXT    NOT NULL,
    -- The subject's own 0-4 rating of how strongly they felt it. Self-report,
    -- so it is context, not ground truth.
    intensity     INTEGER,
    language      TEXT,
    -- One sitting. The paired design compares an 'effortful' recording against
    -- an 'automatic' one made minutes apart in the same mood, so the two have
    -- to be identifiable as a pair. Grouping by timestamp proximity was the
    -- alternative and it breaks the first time somebody takes a phone call
    -- halfway through and comes back twenty minutes later.
    session_id    TEXT,
    filename      TEXT    NOT NULL,
    duration_sec  REAL,
    sample_rate   INTEGER,
    created_at    TEXT    NOT NULL,
    -- Extracted acoustic features, JSON, filled in at upload time so the
    -- analysis does not have to re-open every WAV on every run.
    features_json TEXT,
    -- Set when the extractor could not measure the recording. Such samples are
    -- kept and reported, never silently dropped: a pipeline that fails on 30%
    -- of real recordings is the most important thing this lab could tell you.
    extract_error TEXT,
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (prompt_id) REFERENCES prompts(id)
);

CREATE INDEX IF NOT EXISTS idx_recordings_user ON recordings(user_id, created_at);
-- The session index is created in _migrate, not here. On a database that
-- predates session_id, CREATE TABLE IF NOT EXISTS leaves the old table alone,
-- so this script would try to index a column that does not exist yet and
-- init_db would raise -- inside create_app(), which means the service does not
-- start at all.

CREATE TABLE IF NOT EXISTS analyses (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at   TEXT NOT NULL,
    -- A copy of the settings the run used, so a result stays interpretable
    -- after the settings change. Without this, comparing two runs tells you
    -- nothing.
    settings_json TEXT NOT NULL,
    results_json  TEXT NOT NULL,
    summary       TEXT
);
"""

# The starting prompt bank.
#
# Fixed prompts are the same words for every subject and every label, so a
# difference between two recordings of the same prompt is not a difference in
# what was said. Free prompts let people talk naturally, which produces more
# honest prosody but less comparability. Both are collected because it is not
# obvious in advance which will separate better -- and finding that out is
# itself a result.
# The prompt bank. Each row is (kind, text, load).
#
# The three loads are the experiment, not decoration:
#
#   phonation   Jitter and shimmer are defined on sustained phonation and carry
#               0.38 of the feature weight. Measured on connected speech they
#               pick up intonation and voicing onsets instead of the larynx.
#   automatic   Overlearned sequences need no retrieval at all -- in neurology
#               these survive severe aphasia. This is the true floor of
#               cognitive load, and the anchor the effortful prompt is
#               measured against.
#   effortful   Category fluency is a standard neuropsychological measure,
#               reliably impaired in depression, and emotionally inert. The
#               pauses are the mechanism rather than a proxy: as retrieval gets
#               harder, inter-word intervals lengthen directly.
#
# The effortful prompt is deliberately NOT autobiographical. "The last time you
# called your mother" is harder *and* emotionally loaded -- homesickness,
# estrangement, bereavement -- so a lengthened pause could not be attributed to
# cognitive difficulty rather than reactivity to the topic. For a system aimed
# at personnel posted away from home that confound is not hypothetical, and the
# question is one no welfare tool should be putting to a bereaved subject.
DEFAULT_PROMPTS = [
    (
        "fixed",
        "Take a breath, then hold a steady “aaah” for as long as it stays "
        "comfortable — at least eight seconds. One steady note, not a tune.",
        "phonation",
    ),
    (
        "fixed",
        "Count from one to twenty at a comfortable, even pace, in any language "
        "you like. Do not draw it out — speak as you normally would.",
        "automatic",
    ),
    (
        "fixed",
        "Name as many animals as you can, out loud, until you are asked to stop. "
        "Any language. If you run out, keep trying — the pauses are the point.",
        "effortful",
    ),
]

# Deliberately gone, and deactivated rather than deleted because recordings
# point at them:
#
#   "Say the days of the week"      a second automatic prompt, and a subject
#   "Read this aloud: 'The train'"  can only ever use one -- their anchor has
#                                   to stay fixed across sittings or the gaps
#                                   are not comparable. The reading prompt also
#                                   required English text, which quietly
#                                   contradicts "any language you like".
#
#   the three free prompts          they carry no load, so they cannot pair and
#                                   cannot contribute to the contrast. Worse,
#                                   they pooled into one 'untagged' baseline
#                                   spanning three unrelated tasks -- the exact
#                                   mixing that per-load baselines exist to
#                                   prevent. Offering them meant a subject
#                                   could record a full sitting that counted
#                                   for nothing, and one did.


def connect(path: Optional[Path] = None) -> sqlite3.Connection:
    """Open a connection with row access by name and foreign keys on.

    Args:
        path: Database file. Defaults to ``config.DB_PATH``.

    Returns:
        The connection.
    """
    target = Path(path or config.DB_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(target)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _columns(conn: sqlite3.Connection, table: str) -> set:
    """Column names on a table.

    Args:
        conn: Open connection.
        table: Table name.

    Returns:
        Set of column names.
    """
    return {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}


def _migrate(conn: sqlite3.Connection) -> None:
    """Bring an existing database up to the current schema.

    Args:
        conn: Open connection, inside a transaction.

    Note:
        ``CREATE TABLE IF NOT EXISTS`` in :data:`SCHEMA` does nothing to a table
        that already exists, so columns added after a database was first
        created have to be added here. Both are nullable with no default, which
        is what makes them safe to add to a populated table: recordings made
        before the paired design simply have ``session_id IS NULL`` and drop out
        of the contrast analysis rather than corrupting it.
    """
    if "load" not in _columns(conn, "prompts"):
        conn.execute("ALTER TABLE prompts ADD COLUMN load TEXT")
    if "session_id" not in _columns(conn, "recordings"):
        conn.execute("ALTER TABLE recordings ADD COLUMN session_id TEXT")
    # Only now is the column guaranteed to exist, on a fresh database and an
    # upgraded one alike.
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_recordings_session"
        " ON recordings(user_id, session_id)"
    )


def _sync_prompts(conn: sqlite3.Connection) -> None:
    """Make the prompts table match :data:`DEFAULT_PROMPTS`.

    Args:
        conn: Open connection, inside a transaction.

    Note:
        The bank is defined in code and this table is a cache of it, so a
        prompt whose wording changes is matched by text and re-inserted rather
        than edited in place.

        Prompts no longer in the bank are **deactivated, never deleted**.
        ``recordings.prompt_id`` points at them, and an analysis that cannot
        say which prompt produced a sample is an analysis of nothing --
        particularly here, where the prompt's load level is the independent
        variable.
    """
    wanted = {text: (kind, load) for kind, text, load in DEFAULT_PROMPTS}
    existing = {
        row["text"]: row for row in conn.execute("SELECT id, text, kind, load, is_active FROM prompts")
    }

    for text, (kind, load) in wanted.items():
        row = existing.get(text)
        if row is None:
            conn.execute(
                "INSERT INTO prompts (kind, text, load, is_active) VALUES (?, ?, ?, 1)",
                (kind, text, load),
            )
        elif (row["kind"], row["load"], row["is_active"]) != (kind, load, 1):
            conn.execute(
                "UPDATE prompts SET kind = ?, load = ?, is_active = 1 WHERE id = ?",
                (kind, load, row["id"]),
            )

    for text, row in existing.items():
        if text not in wanted and row["is_active"]:
            conn.execute("UPDATE prompts SET is_active = 0 WHERE id = ?", (row["id"],))


def init_db(path: Optional[Path] = None) -> None:
    """Create the schema, migrate an older one, and sync the prompt bank.

    Args:
        path: Database file. Defaults to ``config.DB_PATH``.
    """
    conn = connect(path)
    with conn:
        conn.executescript(SCHEMA)
        _migrate(conn)
        _sync_prompts(conn)
    conn.close()


def query(sql: str, params: Iterable[Any] = (), path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Run a SELECT and return plain dicts.

    Args:
        sql: The statement.
        params: Bound parameters.
        path: Database file.

    Returns:
        List of row dicts.
    """
    conn = connect(path)
    try:
        return [dict(row) for row in conn.execute(sql, tuple(params))]
    finally:
        conn.close()


def query_one(sql: str, params: Iterable[Any] = (), path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Run a SELECT expecting at most one row.

    Args:
        sql: The statement.
        params: Bound parameters.
        path: Database file.

    Returns:
        The row dict, or None.
    """
    rows = query(sql, params, path)
    return rows[0] if rows else None


def execute(sql: str, params: Iterable[Any] = (), path: Optional[Path] = None) -> int:
    """Run an INSERT/UPDATE/DELETE.

    Args:
        sql: The statement.
        params: Bound parameters.
        path: Database file.

    Returns:
        ``lastrowid`` for inserts, otherwise the number of rows changed.
    """
    conn = connect(path)
    try:
        with conn:
            cursor = conn.execute(sql, tuple(params))
            return cursor.lastrowid if cursor.lastrowid else cursor.rowcount
    finally:
        conn.close()
