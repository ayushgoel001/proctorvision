import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from alert_engine import PHONE_DETECTED, AlertEngine, AlertRule
from detectors import NO_PHONE, PHONE_SOURCE, DetectionResult
from face_context import FaceLandmarkBatch
from persistence import SQLiteRepository
from session_service import SessionService, SessionStatus
from surveillance_engine import SurveillanceEngine


class ManualMonotonicClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


class EmptyFaceProvider:
    backend_name = "test_face_provider"

    def __init__(self):
        self.detect_calls = 0
        self.closed = False

    def detect(self, _clean_frame):
        self.detect_calls += 1
        return FaceLandmarkBatch((), (), (), ())

    def reset(self):
        pass

    def close(self):
        self.closed = True


class UnusedPhoneDetector:
    def reset(self):
        pass


class PipelineIntegrationTests(unittest.TestCase):
    def test_detection_alert_and_session_persist_as_one_event(self):
        monotonic_clock = ManualMonotonicClock()

        def wall_clock():
            return datetime(2026, 1, 1, tzinfo=timezone.utc)

        rule = AlertRule(
            PHONE_DETECTED,
            PHONE_SOURCE,
            minimum_duration_seconds=1.0,
            clear_grace_seconds=0.25,
            cooldown_seconds=2.0,
        )

        with TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "surveillance.db"
            evidence_path = Path(temporary_directory) / "evidence" / "phone.jpg"
            evidence_path.parent.mkdir(parents=True)
            evidence_path.write_bytes(b"controlled-test-evidence")

            repository = SQLiteRepository(database_path)
            service = SessionService(
                repository,
                clock=wall_clock,
                id_factory=lambda: "integration-session",
            )
            session = service.create_session("video:controlled.mp4")
            service.start_calibration(session.session_id)
            service.mark_running(session.session_id, {"sample_count": 20})

            alert_engine = AlertEngine((rule,), clock=monotonic_clock)
            phone_present = DetectionResult(
                detector=PHONE_SOURCE,
                state=PHONE_DETECTED,
                suspicious=True,
                confidence=0.84,
                bounding_box=(10, 20, 100, 180),
                metadata={"confidence_samples": np.array([0.81, 0.84])},
            )

            first_update = alert_engine.update((phone_present,))
            self.assertEqual(first_update.newly_confirmed_events, ())
            monotonic_clock.advance(1.0)
            confirmed_update = alert_engine.update((phone_present,))
            self.assertEqual(len(confirmed_update.newly_confirmed_events), 1)

            confirmed_event = confirmed_update.newly_confirmed_events[0]
            service.record_confirmed_event(
                session.session_id,
                confirmed_event,
                evidence_path.relative_to(temporary_directory),
            )

            phone_clear = DetectionResult(
                detector=PHONE_SOURCE,
                state=NO_PHONE,
                suspicious=False,
            )
            alert_engine.update((phone_clear,))
            monotonic_clock.advance(0.3)
            resolved_update = alert_engine.update((phone_clear,))
            self.assertEqual(len(resolved_update.resolved_events), 1)
            service.resolve_event(
                session.session_id,
                resolved_update.resolved_events[0],
            )
            service.stop_session(session.session_id, average_fps=14.5)

            persisted = service.get_event(session.session_id, confirmed_event.event_id)
            self.assertEqual(persisted.status, "RESOLVED")
            self.assertEqual(persisted.metadata["confidence_samples"], [0.81, 0.84])
            self.assertEqual(persisted.evidence_path, "evidence/phone.jpg")
            self.assertEqual(service.get_session(session.session_id).event_count, 1)
            repository.close()

            reopened_repository = SQLiteRepository(database_path)
            reopened_service = SessionService(reopened_repository)
            reopened_session = reopened_service.get_session(session.session_id)
            reopened_event = reopened_service.get_event(
                session.session_id,
                confirmed_event.event_id,
            )
            self.assertEqual(reopened_session.status, SessionStatus.STOPPED)
            self.assertEqual(reopened_event.status, "RESOLVED")
            reopened_repository.close()

    def test_surveillance_engine_calls_shared_face_provider_once_per_frame(self):
        provider = EmptyFaceProvider()
        engine = SurveillanceEngine(
            face_provider=provider,
            phone_detector=UnusedPhoneDetector(),
        )

        result = engine.process_frame(np.zeros((32, 32, 3), dtype=np.uint8))

        self.assertEqual(provider.detect_calls, 1)
        self.assertEqual(len(result.detections), 4)
        self.assertEqual(
            result.detection("face_presence").metadata["landmark_backend"],
            "test_face_provider",
        )
        engine.close()
        self.assertTrue(provider.closed)


if __name__ == "__main__":
    unittest.main()
