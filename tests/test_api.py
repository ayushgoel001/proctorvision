import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np

from api import create_app, get_session_service
from persistence import PersistenceError, SQLiteRepository
from session_service import SessionService

JPEG_BYTES = b"\xff\xd8phase-4b-evidence\xff\xd9"


class ApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.project_root = Path(self.temporary_directory.name)
        self.database_path = self.project_root / "data" / "test.db"
        self.evidence_root = self.project_root / "data" / "evidence"
        self.app = create_app(
            database_path=self.database_path,
            evidence_root=self.evidence_root,
            project_root=self.project_root,
        )
        self.lifespan_context = self.app.router.lifespan_context(self.app)
        await self.lifespan_context.__aenter__()
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(
                app=self.app,
                raise_app_exceptions=False,
            ),
            base_url="http://testserver",
        )

    async def asyncTearDown(self):
        self.app.dependency_overrides.clear()
        await self.client.aclose()
        await self.lifespan_context.__aexit__(None, None, None)
        self.temporary_directory.cleanup()

    @contextmanager
    def service(self):
        repository = SQLiteRepository(self.database_path)
        try:
            yield SessionService(repository)
        finally:
            repository.close()

    def seed_session(self, source="camera:0", running=True):
        with self.service() as service:
            session = service.create_session(source)
            if running:
                service.start_calibration(session.session_id)
                session = service.mark_running(
                    session.session_id,
                    {
                        "sample_count": np.int64(20),
                        "angles": np.zeros(3, dtype=np.float64),
                    },
                )
            return session

    def seed_event(
        self,
        session_id,
        event_id="GAZE_DEVIATION-0001",
        evidence_path=None,
    ):
        evidence_path = evidence_path or (
            f"data/evidence/{session_id}/{event_id}.jpg"
        )
        event = SimpleNamespace(
            event_id=event_id,
            event_type="GAZE_DEVIATION",
            status="CONFIRMED",
            detector_source="gaze",
            source_state="Looking Left",
            started_at=10.0,
            confirmed_at=13.0,
            resolved_at=None,
            confidence=np.float32(0.82),
            bounding_box=(10, 20, 110, 120),
            metadata={
                "normalized": np.array([0.2, 0.4], dtype=np.float32),
                "sample_count": np.int64(2),
            },
        )
        with self.service() as service:
            return service.record_confirmed_event(
                session_id,
                event,
                evidence_path,
            )

    def write_evidence(self, evidence_path):
        absolute_path = self.project_root / evidence_path
        absolute_path.parent.mkdir(parents=True, exist_ok=True)
        absolute_path.write_bytes(JPEG_BYTES)
        return absolute_path

    async def test_health_endpoint(self):
        response = await self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"status": "ok", "api": "ready", "database": "ready"},
        )

    async def test_empty_session_list(self):
        response = await self.client.get("/sessions")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [])

    async def test_session_list_after_insertion(self):
        session = self.seed_session()
        response = await self.client.get("/sessions")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()), 1)
        self.assertEqual(response.json()[0]["session_id"], session.session_id)
        self.assertEqual(response.json()[0]["status"], "RUNNING")

    async def test_get_existing_session(self):
        session = self.seed_session()
        response = await self.client.get(f"/sessions/{session.session_id}")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["source"], "camera:0")
        self.assertTrue(payload["calibration_completed"])
        self.assertEqual(payload["calibration_details"]["sample_count"], 20)

    async def test_missing_session_returns_404(self):
        response = await self.client.get("/sessions/missing-session")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json(), {"detail": "Session not found."})

    async def test_list_events_for_session(self):
        session = self.seed_session()
        event = self.seed_event(session.session_id)
        response = await self.client.get(
            f"/sessions/{session.session_id}/events"
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()), 1)
        self.assertEqual(response.json()[0]["event_id"], event.event_id)

    async def test_get_existing_event(self):
        session = self.seed_session()
        event = self.seed_event(session.session_id)
        response = await self.client.get(
            f"/events/{session.session_id}/{event.event_id}"
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["event_type"], "GAZE_DEVIATION")
        self.assertEqual(payload["bounding_box"], [10, 20, 110, 120])
        self.assertNotIn("started_monotonic", payload)
        self.assertNotIn("evidence_path", payload)

    async def test_missing_event_returns_404(self):
        session = self.seed_session()
        response = await self.client.get(
            f"/events/{session.session_id}/missing-event"
        )
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json(), {"detail": "Event not found."})

    async def test_event_session_mismatch_is_rejected(self):
        first = self.seed_session("camera:0")
        second = self.seed_session("camera:1")
        event = self.seed_event(first.session_id)
        response = await self.client.get(
            f"/events/{second.session_id}/{event.event_id}"
        )
        self.assertEqual(response.status_code, 404)

    async def test_evidence_is_returned(self):
        session = self.seed_session()
        event = self.seed_event(session.session_id)
        evidence_path = f"data/evidence/{session.session_id}/{event.event_id}.jpg"
        self.write_evidence(evidence_path)

        event_response = await self.client.get(
            f"/events/{session.session_id}/{event.event_id}"
        )
        self.assertTrue(event_response.json()["evidence_available"])
        self.assertEqual(
            event_response.json()["evidence_url"],
            f"/events/{session.session_id}/{event.event_id}/evidence",
        )

        response = await self.client.get(
            f"/events/{session.session_id}/{event.event_id}/evidence"
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/jpeg")
        self.assertEqual(response.content, JPEG_BYTES)

    async def test_missing_evidence_is_handled_cleanly(self):
        session = self.seed_session()
        event = self.seed_event(session.session_id)
        response = await self.client.get(
            f"/events/{session.session_id}/{event.event_id}/evidence"
        )
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json(), {"detail": "Evidence not found."})

    async def test_path_traversal_evidence_is_rejected(self):
        session = self.seed_session()
        traversal_path = (
            f"data/evidence/{session.session_id}/../../outside.jpg"
        )
        event = self.seed_event(
            session.session_id,
            evidence_path=traversal_path,
        )
        outside_path = (self.project_root / traversal_path).resolve()
        outside_path.parent.mkdir(parents=True, exist_ok=True)
        outside_path.write_bytes(JPEG_BYTES)

        event_response = await self.client.get(
            f"/events/{session.session_id}/{event.event_id}"
        )
        self.assertFalse(event_response.json()["evidence_available"])
        self.assertIsNone(event_response.json()["evidence_url"])

        response = await self.client.get(
            f"/events/{session.session_id}/{event.event_id}/evidence"
        )
        self.assertEqual(response.status_code, 404)
        self.assertNotIn(str(outside_path), response.text)

    async def test_numpy_metadata_is_json_safe(self):
        session = self.seed_session()
        event = self.seed_event(session.session_id)
        response = await self.client.get(
            f"/events/{session.session_id}/{event.event_id}"
        )
        self.assertEqual(response.status_code, 200)
        metadata = response.json()["metadata"]
        self.assertEqual(metadata["sample_count"], 2)
        self.assertAlmostEqual(metadata["normalized"][0], 0.2, places=6)

    async def test_pagination_and_limit_validation(self):
        sessions = [self.seed_session(f"camera:{index}") for index in range(3)]
        first_page = await self.client.get("/sessions?limit=2&offset=0")
        second_page = await self.client.get("/sessions?limit=2&offset=2")
        self.assertEqual(first_page.status_code, 200)
        self.assertEqual(second_page.status_code, 200)
        first_ids = {item["session_id"] for item in first_page.json()}
        second_ids = {item["session_id"] for item in second_page.json()}
        self.assertEqual(len(first_ids), 2)
        self.assertEqual(len(second_ids), 1)
        self.assertFalse(first_ids & second_ids)
        self.assertEqual(first_ids | second_ids, {item.session_id for item in sessions})
        self.assertEqual(
            (await self.client.get("/sessions?limit=0")).status_code,
            422,
        )
        self.assertEqual(
            (await self.client.get("/sessions?limit=101")).status_code,
            422,
        )

    async def test_database_failure_does_not_leak_details(self):
        secret_detail = r"C:\private\database\surveillance.db is locked"

        async def failing_service():
            raise PersistenceError(secret_detail)

        self.app.dependency_overrides[get_session_service] = failing_service
        response = await self.client.get("/sessions")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json(),
            {"detail": "Persistence service unavailable."},
        )
        self.assertNotIn(secret_detail, response.text)


if __name__ == "__main__":
    unittest.main()
