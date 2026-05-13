# core/database.py
# ─────────────────────────────────────────────────────────────
# All SQLite interactions live here.
# Raw SQL is hidden behind clean function signatures so the
# pipeline code never has to think about cursor management.
#
# Functions:
#   init_db()              → opens connection, ensures table exists
#   insert_violation()     → inserts a new row, returns row id
#   update_plate()         → updates plate text & confidence for a row
#   finalize_unresolved()  → marks still-scanning rows as "Unreadable"
#   get_violation_count()  → returns total violations for a video
# ─────────────────────────────────────────────────────────────

import os
import sqlite3

from core.config import DB_DIR, DB_PATH

# SQL to create the violations table if it doesn't already exist
_CREATE_TABLE_SQL = '''
    CREATE TABLE IF NOT EXISTS violations (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp       DATETIME,
        video_source    TEXT,
        video_second    INTEGER,
        license_plate   TEXT,
        confidence      REAL
    )
'''


def init_db():
    """Open (or create) the violations database and ensure the schema exists.
    
    Returns:
        (conn, cur): A (connection, cursor) tuple ready for use.
    """
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(_CREATE_TABLE_SQL)
    conn.commit()
    return conn, cur


def insert_violation(cur, timestamp: str, video_source: str, video_second: int) -> int:
    """Insert a new violation row with a placeholder plate value.
    
    The plate is updated later once ALPR produces a result.
    Returns the auto-incremented row id.
    """
    cur.execute(
        '''INSERT INTO violations (timestamp, video_source, video_second, license_plate, confidence)
           VALUES (?, ?, ?, ?, ?)''',
        (timestamp, video_source, video_second, "Scanning...", 0.0)
    )
    return cur.lastrowid


def update_plate(cur, db_id: int, plate_text: str, confidence: float):
    """Overwrite the plate text and confidence for an existing violation row."""
    cur.execute(
        "UPDATE violations SET license_plate = ?, confidence = ? WHERE id = ?",
        (plate_text, confidence, db_id)
    )


def finalize_unresolved(cur, db_id: int):
    """Mark a row as 'Unreadable' if the plate was never successfully read.
    
    Only updates rows that are still in the 'Scanning...' state.
    """
    cur.execute(
        "UPDATE violations SET license_plate = 'Unreadable' "
        "WHERE id = ? AND license_plate = 'Scanning...'",
        (db_id,)
    )


def get_violation_count(cur, video_source: str) -> int:
    """Return the total number of violations recorded for a given video file."""
    cur.execute(
        'SELECT COUNT(*) FROM violations WHERE video_source = ?',
        (video_source,)
    )
    return cur.fetchone()[0]
