import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from alert_engine import EventStatus, ReviewEvent
from persistence import PersistenceError, SQLiteRepository
from serialization import dumps_json, loads_json
from session_service import (
    InvalidSessionTransitionError,
    SessionService,
    SessionStatus,
)


class FakeUtcClock:
    def __init__(self):
        self.now = datetime(2026, 8, 14, 12, 0, tzinfo=timezone.utc)

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += timedelta(seconds=seconds)


def confirmed_event(event_id="GAZE_DEVIATION-0001"):
    return ReviewEvent(
        event_id=event_id,
        event_type="GAZE_DEVIATION",
        status=EventStatus.CONFIRMED,
        detector_source="gaze",
        source_state="Looking Left",
        started_at=10.0,
        confirmed_at=13.0,
        resolved_at=None,
        confidence=None,
        bounding_box=(10, 20, 110, 120),
        metadata={
            "normalized": np.array([0.2, 0.4], dtype=np.float32),
            "count": np.int64(2),
        },
    )


class SessionPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "data" / "test.db"
        self.repository = SQLiteRepository(self.database_path)
        self.clock = FakeUtcClock()
        self.next_id = 0

        def id_factory():
            self.next_id += 1
            return f"session-{self.next_id}"

        self.service = SessionService(
            self.repository,
            clock=self.clock,
            id_factory=id_factory,
        )

    def tearDown(self):
        self.repository.close()
        self.temporary_directory.cleanup()

    def create_running_session(self):
        session = self.service.create_session("camera:0")
        self.clock.advance(1)
        self.service.start_calibration(session.session_id)
        self.clock.advance(5)
        return self.service.mark_running(
            session.session_id,
            {"samples": 20, "angles": np.zeros(3)},
        )

    def test_session_creation(self):
        session = self.service.create_session("camera:0")
        self.assertEqual(session.status, SessionStatus.CREATED)
        self.assertEqual(session.video_source, "camera:0")
        self.assertEqual(session.event_count, 0)
        self.assertIsNone(session.started_at_utc)
        self.assertTrue(self.database_path.is_file())

    def test_valid_lifecycle_transitions(self):
        session = self.service.create_session("camera:0")
        calibrating = self.service.start_calibration(session.session_id)
        self.assertEqual(calibrating.status, SessionStatus.CALIBRATING)
        running = self.service.mark_running(
            session.session_id,
            {"sample_count": 15},
        )
        self.assertEqual(running.status, SessionStatus.RUNNING)
        self.assertEqual(running.calibration_details["sample_count"], 15)
        stopped = self.service.stop_session(session.session_id, average_fps=4.2)
        self.assertEqual(stopped.status, SessionStatus.STOPPED)
        self.assertEqual(stopped.average_fps, 4.2)
        self.assertIsNotNone(stopped.ended_at_utc)

    def test_invalid_lifecycle_transition_is_rejected(self):
        session = self.service.create_session("camera:0")
        with self.assertRaises(InvalidSessionTransitionError):
            self.service.mark_running(session.session_id)
        self.service.fail_session(session.session_id, "model failed")
        with self.assertRaises(InvalidSessionTransitionError):
            self.service.start_calibration(session.session_id)

    def test_independent_sessions(self):
        first = self.service.create_session("camera:0")
        second = self.service.create_session("camera:1")
        self.service.start_calibration(first.session_id)
        self.assertNotEqual(first.session_id, second.session_id)
        self.assertEqual(
            self.service.get_session(first.session_id).status,
            SessionStatus.CALIBRATING,
        )
        self.assertEqual(
            self.service.get_session(second.session_id).status,
            SessionStatus.CREATED,
        )

    def test_confirmed_event_persistence(self):
        session = self.create_running_session()
        persisted = self.service.record_confirmed_event(
            session.session_id,
            confirmed_event(),
            f"data/evidence/{session.session_id}/event.jpg",
        )
        self.assertEqual(persisted.status, EventStatus.CONFIRMED)
        self.assertEqual(persisted.event_type, "GAZE_DEVIATION")
        self.assertEqual(persisted.duration_seconds, 3.0)
        self.assertEqual(
            self.service.get_session(session.session_id).event_count,
            1,
        )

    def test_event_is_associated_with_correct_session(self):
        first = self.create_running_session()
        second = self.service.create_session("camera:1")
        self.service.start_calibration(second.session_id)
        self.service.mark_running(second.session_id)
        event = confirmed_event()
        self.service.record_confirmed_event(
            first.session_id,
            event,
            f"data/evidence/{first.session_id}/{event.event_id}.jpg",
        )
        self.assertEqual(len(self.service.list_events(first.session_id)), 1)
        self.assertEqual(self.service.list_events(second.session_id), [])

        # The same per-run AlertEngine ID is valid in a different session.
        self.service.record_confirmed_event(
            second.session_id,
            event,
            f"data/evidence/{second.session_id}/{event.event_id}.jpg",
        )
        self.assertEqual(len(self.service.list_events(second.session_id)), 1)

    def test_resolution_updates_existing_event(self):
        session = self.create_running_session()
        event = confirmed_event()
        self.service.record_confirmed_event(
            session.session_id,
            event,
            "data/evidence/session/event.jpg",
        )
        self.clock.advance(7)
        resolved = replace(
            event,
            status=EventStatus.RESOLVED,
            resolved_at=20.0,
        )
        persisted = self.service.resolve_event(session.session_id, resolved)
        self.assertEqual(persisted.status, EventStatus.RESOLVED)
        self.assertEqual(persisted.duration_seconds, 10.0)
        self.assertIsNotNone(persisted.resolved_at_utc)
        self.assertEqual(len(self.service.list_events(session.session_id)), 1)
        self.assertEqual(
            self.service.get_session(session.session_id).event_count,
            1,
        )

    def test_evidence_path_persistence(self):
        session = self.create_running_session()
        evidence_path = f"data/evidence/{session.session_id}/event.jpg"
        event = self.service.record_confirmed_event(
            session.session_id,
            confirmed_event(),
            evidence_path,
        )
        self.assertEqual(event.evidence_path, evidence_path)

    def test_numpy_metadata_serialization(self):
        encoded = dumps_json(
            {
                "matrix": np.eye(2, dtype=np.float64),
                "integer": np.int32(7),
                "point": (np.float32(1.5), np.float64(2.5)),
            }
        )
        decoded = loads_json(encoded)
        self.assertEqual(decoded["matrix"], [[1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(decoded["integer"], 7)
        self.assertEqual(decoded["point"], [1.5, 2.5])

        session = self.create_running_session()
        persisted = self.service.record_confirmed_event(
            session.session_id,
            confirmed_event(),
            "data/evidence/session/event.jpg",
        )
        self.assertEqual(persisted.metadata["count"], 2)
        self.assertAlmostEqual(persisted.metadata["normalized"][0], 0.2, places=6)

    def test_foreign_key_enforcement(self):
        enabled = self.repository.connection.execute(
            "PRAGMA foreign_keys"
        ).fetchone()[0]
        self.assertEqual(enabled, 1)
        now = self.clock().isoformat()
        with self.assertRaises(PersistenceError):
            self.repository.insert_event(
                {
                    "event_id": "event-1",
                    "session_id": "missing-session",
                    "event_type": "NO_FACE",
                    "detector_source": "face_presence",
                    "source_state": "NO_FACES",
                    "status": "CONFIRMED",
                    "started_monotonic": 0.0,
                    "confirmed_monotonic": 3.0,
                    "started_at_utc": now,
                    "confirmed_at_utc": now,
                    "duration_seconds": 3.0,
                    "metadata_json": "{}",
                    "created_at_utc": now,
                    "updated_at_utc": now,
                }
            )

    def test_database_survives_reopen(self):
        session = self.create_running_session()
        event = confirmed_event()
        self.service.record_confirmed_event(
            session.session_id,
            event,
            "data/evidence/session/event.jpg",
        )
        self.repository.close()

        self.repository = SQLiteRepository(self.database_path)
        reloaded = SessionService(self.repository, clock=self.clock)
        self.assertEqual(
            reloaded.get_session(session.session_id).status,
            SessionStatus.RUNNING,
        )
        self.assertEqual(
            reloaded.get_event(session.session_id, event.event_id).event_type,
            "GAZE_DEVIATION",
        )

    def test_failed_session_is_recorded(self):
        session = self.service.create_session("camera:0")
        self.clock.advance(2)
        failed = self.service.fail_session(session.session_id, "camera unavailable")
        self.assertEqual(failed.status, SessionStatus.FAILED)
        self.assertEqual(failed.failure_reason, "camera unavailable")
        self.assertIsNotNone(failed.ended_at_utc)


if __name__ == "__main__":
    unittest.main()
