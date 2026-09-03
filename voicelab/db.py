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
DEFAULT_PROMPTS = [
    ("fixed", "Please count slowly from one to twenty, in any language you like."),
    ("fixed", "Say the days of the week, twice, at whatever pace feels natural."),
    ("fixed", "Read this aloud: 'The train leaves at seven in the morning and arrives late in the evening.'"),
    ("free", "Describe what you did yesterday, in as much detail as you like."),
    ("free", "Describe the room you are sitting in right now."),
    ("free", "Talk about a place you would like to travel to, and why."),
]


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


def init_db(path: Optional[Path] = None) -> None:
    """Create the schema and seed the prompt bank if it is empty.

    Args:
        path: Database file. Defaults to ``config.DB_PATH``.
    """
    conn = connect(path)
    with conn:
        conn.executescript(SCHEMA)
        existing = conn.execute("SELECT COUNT(*) AS n FROM prompts").fetchone()["n"]
        if existing == 0:
            conn.executemany(
                "INSERT INTO prompts (kind, text, is_active) VALUES (?, ?, 1)",
                DEFAULT_PROMPTS,
            )
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
