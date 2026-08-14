"""Monitoring-session domain model and lifecycle service."""

import math
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path

from serialization import dumps_json, loads_json

CONFIRMED_EVENT_STATUS = "CONFIRMED"
RESOLVED_EVENT_STATUS = "RESOLVED"


class SessionStatus(str, Enum):
    CREATED = "CREATED"
    CALIBRATING = "CALIBRATING"
    RUNNING = "RUNNING"
    STOPPED = "STOPPED"
    FAILED = "FAILED"


TERMINAL_SESSION_STATUSES = {SessionStatus.STOPPED, SessionStatus.FAILED}

VALID_TRANSITIONS = {
    SessionStatus.CREATED: {SessionStatus.CALIBRATING, SessionStatus.FAILED},
    SessionStatus.CALIBRATING: {
        SessionStatus.RUNNING,
        SessionStatus.STOPPED,
        SessionStatus.FAILED,
    },
    SessionStatus.RUNNING: {SessionStatus.STOPPED, SessionStatus.FAILED},
    SessionStatus.STOPPED: set(),
    SessionStatus.FAILED: set(),
}


class SessionError(RuntimeError):
    pass


class SessionNotFoundError(SessionError):
    pass


class EventNotFoundError(SessionError):
    pass


class InvalidSessionTransitionError(SessionError):
    pass


@dataclass(frozen=True, slots=True)
class MonitoringSession:
    session_id: str
    status: SessionStatus
    created_at_utc: datetime
    started_at_utc: datetime | None
    ended_at_utc: datetime | None
    video_source: str
    calibration_completed_at_utc: datetime | None
    calibration_details: dict | None
    event_count: int
    average_fps: float | None
    failure_reason: str | None


@dataclass(frozen=True, slots=True)
class PersistedEvent:
    event_id: str
    session_id: str
    event_type: str
    detector_source: str
    source_state: str
    status: str
    started_monotonic: float
    confirmed_monotonic: float
    resolved_monotonic: float | None
    started_at_utc: datetime
    confirmed_at_utc: datetime
    resolved_at_utc: datetime | None
    duration_seconds: float | None
    confidence: float | None
    bounding_box: tuple[int, int, int, int] | None
    evidence_path: str | None
    metadata: dict


class SessionService:
    """Validate lifecycle transitions and translate domain data for persistence."""

    def __init__(self, repository, clock=None, id_factory=None):
        self.repository = repository
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.id_factory = id_factory or (lambda: uuid.uuid4().hex)

    def create_session(self, video_source):
        video_source = str(video_source).strip()
        if not video_source:
            raise ValueError("video_source cannot be empty.")
        now = self._utc_now()
        session_id = str(self.id_factory())
        self.repository.insert_session(
            {
                "session_id": session_id,
                "status": SessionStatus.CREATED.value,
                "created_at_utc": now.isoformat(),
                "video_source": video_source,
            }
        )
        return self.get_session(session_id)

    def start_calibration(self, session_id):
        return self._transition(
            session_id,
            SessionStatus.CALIBRATING,
            started_at_utc=self._utc_now().isoformat(),
        )

    def mark_running(self, session_id, calibration_details=None):
        now = self._utc_now()
        return self._transition(
            session_id,
            SessionStatus.RUNNING,
            calibration_completed_at_utc=now.isoformat(),
            calibration_details_json=(
                dumps_json(calibration_details)
                if calibration_details is not None
                else None
            ),
        )

    def stop_session(self, session_id, average_fps=None):
        return self._transition(
            session_id,
            SessionStatus.STOPPED,
            ended_at_utc=self._utc_now().isoformat(),
            average_fps=self._validate_fps(average_fps),
        )

    def fail_session(self, session_id, reason, average_fps=None):
        reason = str(reason).strip()
        if not reason:
            raise ValueError("A failure reason is required.")
        return self._transition(
            session_id,
            SessionStatus.FAILED,
            ended_at_utc=self._utc_now().isoformat(),
            average_fps=self._validate_fps(average_fps),
            failure_reason=reason[:2000],
        )

    def record_confirmed_event(self, session_id, event, evidence_path):
        event_status = getattr(event.status, "value", event.status)
        if event_status != CONFIRMED_EVENT_STATUS:
            raise ValueError("Only confirmed ReviewEvent objects can be inserted.")
        session = self.get_session(session_id)
        if session.status != SessionStatus.RUNNING:
            raise SessionError(
                f"Cannot persist an event while session {session_id} is "
                f"{session.status.value}."
            )

        evidence_path_value = str(evidence_path).strip()
        if not evidence_path_value:
            raise ValueError("A confirmed event requires an evidence path.")
        evidence_path = Path(evidence_path_value).as_posix()
        confirmed_at_utc = self._utc_now()
        confirmation_duration = max(0.0, event.confirmed_at - event.started_at)
        started_at_utc = confirmed_at_utc - timedelta(
            seconds=confirmation_duration
        )
        confidence = None if event.confidence is None else float(event.confidence)
        if confidence is not None and not math.isfinite(confidence):
            raise ValueError("Event confidence must be finite.")
        record = {
            "event_id": event.event_id,
            "session_id": session_id,
            "event_type": event.event_type,
            "detector_source": event.detector_source,
            "source_state": event.source_state,
            "status": CONFIRMED_EVENT_STATUS,
            "started_monotonic": event.started_at,
            "confirmed_monotonic": event.confirmed_at,
            "resolved_monotonic": None,
            "started_at_utc": started_at_utc.isoformat(),
            "confirmed_at_utc": confirmed_at_utc.isoformat(),
            "resolved_at_utc": None,
            "duration_seconds": confirmation_duration,
            "confidence": confidence,
            "bounding_box_json": (
                dumps_json(event.bounding_box)
                if event.bounding_box is not None
                else None
            ),
            "evidence_path": evidence_path,
            "metadata_json": dumps_json(event.metadata),
            "created_at_utc": confirmed_at_utc.isoformat(),
            "updated_at_utc": confirmed_at_utc.isoformat(),
        }
        self.repository.insert_event(record)
        return self._event_from_record(record)

    def resolve_event(self, session_id, event):
        event_status = getattr(event.status, "value", event.status)
        if (
            event_status != RESOLVED_EVENT_STATUS
            or event.resolved_at is None
        ):
            raise ValueError("A resolved ReviewEvent with resolved_at is required.")
        persisted = self.get_event(session_id, event.event_id)
        resolution_utc = self._utc_now()
        duration_seconds = max(0.0, event.resolved_at - event.started_at)
        self.repository.update_event_resolution(
            session_id,
            event.event_id,
            status=RESOLVED_EVENT_STATUS,
            resolved_monotonic=event.resolved_at,
            resolved_at_utc=resolution_utc.isoformat(),
            duration_seconds=duration_seconds,
            updated_at_utc=resolution_utc.isoformat(),
        )
        return replace(
            persisted,
            status=RESOLVED_EVENT_STATUS,
            resolved_monotonic=event.resolved_at,
            resolved_at_utc=resolution_utc,
            duration_seconds=duration_seconds,
        )

    def get_session(self, session_id):
        record = self.repository.fetch_session(session_id)
        if record is None:
            raise SessionNotFoundError(f"Session not found: {session_id}")
        return self._session_from_record(record)

    def list_sessions(self, limit=None, offset=0):
        return [
            self._session_from_record(record)
            for record in self.repository.list_sessions(limit, offset)
        ]

    def get_event(self, session_id, event_id):
        record = self.repository.fetch_event(session_id, event_id)
        if record is None:
            raise EventNotFoundError(
                f"Event {event_id} was not found in session {session_id}."
            )
        return self._event_from_record(record)

    def list_events(self, session_id, limit=None, offset=0):
        self.get_session(session_id)
        return [
            self._event_from_record(record)
            for record in self.repository.list_events(session_id, limit, offset)
        ]

    def _transition(self, session_id, target_status, **fields):
        current = self.get_session(session_id)
        if target_status not in VALID_TRANSITIONS[current.status]:
            raise InvalidSessionTransitionError(
                f"Invalid session transition: {current.status.value} -> "
                f"{target_status.value}."
            )
        self.repository.update_session(
            session_id,
            status=target_status.value,
            **fields,
        )
        return self.get_session(session_id)

    def _utc_now(self):
        value = self.clock()
        if not isinstance(value, datetime) or value.tzinfo is None:
            raise ValueError("Session clock must return a timezone-aware datetime.")
        return value.astimezone(timezone.utc)

    @staticmethod
    def _validate_fps(value):
        if value is None:
            return None
        value = float(value)
        if value < 0:
            raise ValueError("average_fps cannot be negative.")
        return value

    @staticmethod
    def _parse_datetime(value):
        return datetime.fromisoformat(value) if value is not None else None

    @classmethod
    def _session_from_record(cls, record):
        return MonitoringSession(
            session_id=record["session_id"],
            status=SessionStatus(record["status"]),
            created_at_utc=cls._parse_datetime(record["created_at_utc"]),
            started_at_utc=cls._parse_datetime(record["started_at_utc"]),
            ended_at_utc=cls._parse_datetime(record["ended_at_utc"]),
            video_source=record["video_source"],
            calibration_completed_at_utc=cls._parse_datetime(
                record["calibration_completed_at_utc"]
            ),
            calibration_details=loads_json(record["calibration_details_json"]),
            event_count=int(record["event_count"]),
            average_fps=record["average_fps"],
            failure_reason=record["failure_reason"],
        )

    @classmethod
    def _event_from_record(cls, record):
        bounding_box = loads_json(record["bounding_box_json"])
        return PersistedEvent(
            event_id=record["event_id"],
            session_id=record["session_id"],
            event_type=record["event_type"],
            detector_source=record["detector_source"],
            source_state=record["source_state"],
            status=record["status"],
            started_monotonic=float(record["started_monotonic"]),
            confirmed_monotonic=float(record["confirmed_monotonic"]),
            resolved_monotonic=(
                float(record["resolved_monotonic"])
                if record["resolved_monotonic"] is not None
                else None
            ),
            started_at_utc=cls._parse_datetime(record["started_at_utc"]),
            confirmed_at_utc=cls._parse_datetime(record["confirmed_at_utc"]),
            resolved_at_utc=cls._parse_datetime(record["resolved_at_utc"]),
            duration_seconds=record["duration_seconds"],
            confidence=record["confidence"],
            bounding_box=tuple(bounding_box) if bounding_box is not None else None,
            evidence_path=record["evidence_path"],
            metadata=loads_json(record["metadata_json"]),
        )
