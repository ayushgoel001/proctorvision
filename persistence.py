"""Built-in SQLite persistence for monitoring sessions and review events."""

import sqlite3
from pathlib import Path

SESSION_FIELDS = {
    "status",
    "started_at_utc",
    "ended_at_utc",
    "calibration_completed_at_utc",
    "calibration_details_json",
    "average_fps",
    "failure_reason",
}

EVENT_RESOLUTION_FIELDS = {
    "status",
    "resolved_monotonic",
    "resolved_at_utc",
    "duration_seconds",
    "updated_at_utc",
}


SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    status TEXT NOT NULL CHECK (
        status IN ('CREATED', 'CALIBRATING', 'RUNNING', 'STOPPED', 'FAILED')
    ),
    created_at_utc TEXT NOT NULL,
    started_at_utc TEXT,
    ended_at_utc TEXT,
    video_source TEXT NOT NULL,
    calibration_completed_at_utc TEXT,
    calibration_details_json TEXT,
    event_count INTEGER NOT NULL DEFAULT 0 CHECK (event_count >= 0),
    average_fps REAL,
    failure_reason TEXT
);

CREATE TABLE IF NOT EXISTS events (
    event_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    detector_source TEXT NOT NULL,
    source_state TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('CONFIRMED', 'RESOLVED')),
    started_monotonic REAL NOT NULL,
    confirmed_monotonic REAL NOT NULL,
    resolved_monotonic REAL,
    started_at_utc TEXT NOT NULL,
    confirmed_at_utc TEXT NOT NULL,
    resolved_at_utc TEXT,
    duration_seconds REAL,
    confidence REAL,
    bounding_box_json TEXT,
    evidence_path TEXT,
    metadata_json TEXT NOT NULL,
    created_at_utc TEXT NOT NULL,
    updated_at_utc TEXT NOT NULL,
    PRIMARY KEY (session_id, event_id),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_events_session_id ON events(session_id);
CREATE INDEX IF NOT EXISTS idx_events_session_type ON events(session_id, event_type);
"""


class PersistenceError(RuntimeError):
    pass


class RecordNotFoundError(PersistenceError):
    pass


class SQLiteRepository:
    """Small repository for both tables using one transactional connection."""

    def __init__(self, database_path, initialize_schema=True):
        self.database_path = Path(database_path).resolve()
        self.connection = None
        try:
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(str(self.database_path))
            self.connection.row_factory = sqlite3.Row
            self.connection.execute("PRAGMA foreign_keys = ON")
            enabled = self.connection.execute("PRAGMA foreign_keys").fetchone()[0]
            if enabled != 1:
                raise PersistenceError("SQLite foreign-key enforcement could not be enabled.")
            if initialize_schema:
                self.connection.executescript(SCHEMA)
        except (OSError, sqlite3.Error, PersistenceError) as exc:
            if self.connection is not None:
                self.connection.close()
                self.connection = None
            if isinstance(exc, PersistenceError):
                raise
            raise PersistenceError(
                f"Failed to initialize SQLite database at {self.database_path}: {exc}"
            ) from exc

    def close(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def insert_session(self, record):
        sql = """
            INSERT INTO sessions (
                session_id, status, created_at_utc, started_at_utc,
                ended_at_utc, video_source, calibration_completed_at_utc,
                calibration_details_json, event_count, average_fps, failure_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        values = (
            record["session_id"],
            record["status"],
            record["created_at_utc"],
            record.get("started_at_utc"),
            record.get("ended_at_utc"),
            record["video_source"],
            record.get("calibration_completed_at_utc"),
            record.get("calibration_details_json"),
            record.get("event_count", 0),
            record.get("average_fps"),
            record.get("failure_reason"),
        )
        try:
            with self.connection:
                self.connection.execute(sql, values)
        except sqlite3.Error as exc:
            raise PersistenceError(f"Failed to create session: {exc}") from exc

    def update_session(self, session_id, **fields):
        self._validate_fields(fields, SESSION_FIELDS, "session")
        assignments = ", ".join(f"{name} = ?" for name in fields)
        values = [*fields.values(), session_id]
        try:
            with self.connection:
                cursor = self.connection.execute(
                    f"UPDATE sessions SET {assignments} WHERE session_id = ?",
                    values,
                )
                if cursor.rowcount != 1:
                    raise RecordNotFoundError(f"Session not found: {session_id}")
        except RecordNotFoundError:
            raise
        except sqlite3.Error as exc:
            raise PersistenceError(f"Failed to update session {session_id}: {exc}") from exc

    def fetch_session(self, session_id):
        try:
            row = self.connection.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            return dict(row) if row is not None else None
        except sqlite3.Error as exc:
            raise PersistenceError(f"Failed to read session {session_id}: {exc}") from exc

    def list_sessions(self, limit=None, offset=0):
        limit, offset = self._validate_pagination(limit, offset)
        sql = "SELECT * FROM sessions ORDER BY created_at_utc DESC, session_id DESC"
        parameters = ()
        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            parameters = (limit, offset)
        try:
            rows = self.connection.execute(sql, parameters).fetchall()
            return [dict(row) for row in rows]
        except sqlite3.Error as exc:
            raise PersistenceError(f"Failed to list sessions: {exc}") from exc

    def insert_event(self, record):
        sql = """
            INSERT INTO events (
                event_id, session_id, event_type, detector_source, source_state,
                status, started_monotonic, confirmed_monotonic, resolved_monotonic,
                started_at_utc, confirmed_at_utc, resolved_at_utc, duration_seconds,
                confidence, bounding_box_json, evidence_path, metadata_json,
                created_at_utc, updated_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        values = tuple(
            record.get(name)
            for name in (
                "event_id",
                "session_id",
                "event_type",
                "detector_source",
                "source_state",
                "status",
                "started_monotonic",
                "confirmed_monotonic",
                "resolved_monotonic",
                "started_at_utc",
                "confirmed_at_utc",
                "resolved_at_utc",
                "duration_seconds",
                "confidence",
                "bounding_box_json",
                "evidence_path",
                "metadata_json",
                "created_at_utc",
                "updated_at_utc",
            )
        )
        try:
            with self.connection:
                self.connection.execute(sql, values)
                cursor = self.connection.execute(
                    "UPDATE sessions SET event_count = event_count + 1 "
                    "WHERE session_id = ?",
                    (record["session_id"],),
                )
                if cursor.rowcount != 1:
                    raise RecordNotFoundError(
                        f"Session not found: {record['session_id']}"
                    )
        except RecordNotFoundError:
            raise
        except sqlite3.Error as exc:
            raise PersistenceError(f"Failed to persist event: {exc}") from exc

    def update_event_resolution(self, session_id, event_id, **fields):
        self._validate_fields(fields, EVENT_RESOLUTION_FIELDS, "event resolution")
        assignments = ", ".join(f"{name} = ?" for name in fields)
        values = [*fields.values(), session_id, event_id]
        try:
            with self.connection:
                cursor = self.connection.execute(
                    f"UPDATE events SET {assignments} "
                    "WHERE session_id = ? AND event_id = ?",
                    values,
                )
                if cursor.rowcount != 1:
                    raise RecordNotFoundError(
                        f"Event {event_id} was not found in session {session_id}."
                    )
        except RecordNotFoundError:
            raise
        except sqlite3.Error as exc:
            raise PersistenceError(f"Failed to resolve event {event_id}: {exc}") from exc

    def fetch_event(self, session_id, event_id):
        try:
            row = self.connection.execute(
                "SELECT * FROM events WHERE session_id = ? AND event_id = ?",
                (session_id, event_id),
            ).fetchone()
            return dict(row) if row is not None else None
        except sqlite3.Error as exc:
            raise PersistenceError(
                f"Failed to read event {event_id} in session {session_id}: {exc}"
            ) from exc

    def list_events(self, session_id, limit=None, offset=0):
        limit, offset = self._validate_pagination(limit, offset)
        sql = (
            "SELECT * FROM events WHERE session_id = ? "
            "ORDER BY confirmed_at_utc, event_id"
        )
        parameters = [session_id]
        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            parameters.extend((limit, offset))
        try:
            rows = self.connection.execute(sql, parameters).fetchall()
            return [dict(row) for row in rows]
        except sqlite3.Error as exc:
            raise PersistenceError(
                f"Failed to list events for session {session_id}: {exc}"
            ) from exc

    def check_health(self):
        try:
            row = self.connection.execute("SELECT 1").fetchone()
            table_rows = self.connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' AND name IN ('sessions', 'events')"
            ).fetchall()
            tables = {table_row[0] for table_row in table_rows}
            return (
                row is not None
                and row[0] == 1
                and tables == {"sessions", "events"}
            )
        except sqlite3.Error as exc:
            raise PersistenceError("SQLite readiness check failed.") from exc

    @staticmethod
    def _validate_fields(fields, allowed_fields, record_name):
        if not fields:
            raise ValueError(f"At least one {record_name} field is required.")
        invalid = set(fields) - allowed_fields
        if invalid:
            raise ValueError(
                f"Unsupported {record_name} fields: {sorted(invalid)}"
            )

    @staticmethod
    def _validate_pagination(limit, offset):
        offset = int(offset)
        if offset < 0:
            raise ValueError("offset cannot be negative.")
        if limit is None:
            return None, offset
        limit = int(limit)
        if limit < 1:
            raise ValueError("limit must be at least 1.")
        return limit, offset
