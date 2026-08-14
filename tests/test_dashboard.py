import re
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np

from api import create_app, get_optional_session_service
from persistence import SQLiteRepository
from session_service import SessionService

JPEG_BYTES = b"\xff\xd8phase-4c-dashboard\xff\xd9"


class DashboardTests(unittest.IsolatedAsyncioTestCase):
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
                        "valid_sample_seconds": np.float64(5.2),
                    },
                )
            return session

    def seed_event(
        self,
        session_id,
        event_id,
        event_type="GAZE_DEVIATION",
        detector_source="gaze",
        source_state="Looking Left",
        evidence_path=None,
        metadata=None,
    ):
        evidence_path = evidence_path or (
            f"data/evidence/{session_id}/{event_id}.jpg"
        )
        event = SimpleNamespace(
            event_id=event_id,
            event_type=event_type,
            status="CONFIRMED",
            detector_source=detector_source,
            source_state=source_state,
            started_at=10.0,
            confirmed_at=13.0,
            resolved_at=None,
            confidence=np.float32(0.82),
            bounding_box=(10, 20, 110, 120),
            metadata=metadata or {
                "normalized": np.array([0.2, 0.4], dtype=np.float32)
            },
        )
        with self.service() as service:
            return service.record_confirmed_event(
                session_id,
                event,
                evidence_path,
            )

    def write_evidence(self, relative_path):
        path = self.project_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(JPEG_BYTES)
        return path

    async def test_dashboard_loads_with_no_sessions(self):
        response = await self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Review dashboard", response.text)
        self.assertIn("No monitoring sessions yet", response.text)
        stylesheet = await self.client.get("/static/dashboard.css")
        self.assertEqual(stylesheet.status_code, 200)
        self.assertIn("text/css", stylesheet.headers["content-type"])

    async def test_recent_sessions_render(self):
        session = self.seed_session()
        response = await self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(session.session_id, response.text)
        self.assertIn("RUNNING", response.text)
        self.assertIn('data-auto-refresh="true"', response.text)

    async def test_session_detail_renders(self):
        session = self.seed_session()
        response = await self.client.get(
            f"/dashboard/sessions/{session.session_id}"
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(session.session_id, response.text)
        self.assertIn("Calibration", response.text)
        self.assertIn("20", response.text)
        self.assertIn("camera:0", response.text)

    async def test_event_counts_are_correct(self):
        session = self.seed_session()
        self.seed_event(session.session_id, "GAZE_DEVIATION-0001")
        self.seed_event(session.session_id, "GAZE_DEVIATION-0002")
        self.seed_event(
            session.session_id,
            "PHONE_DETECTED-0003",
            event_type="PHONE_DETECTED",
            detector_source="phone",
            source_state="PHONE_DETECTED",
        )
        response = await self.client.get(
            f"/dashboard/sessions/{session.session_id}"
        )
        self.assertRegex(
            response.text,
            re.compile(r"Gaze deviation\s*</span><strong>2</strong>", re.DOTALL),
        )
        self.assertRegex(
            response.text,
            re.compile(r"Phone detected\s*</span><strong>1</strong>", re.DOTALL),
        )
        self.assertRegex(
            response.text,
            re.compile(r"Multiple faces\s*</span><strong>0</strong>", re.DOTALL),
        )

    async def test_event_timeline_renders_chronologically(self):
        session = self.seed_session()
        first = self.seed_event(session.session_id, "GAZE_DEVIATION-0001")
        second = self.seed_event(
            session.session_id,
            "NO_FACE-0002",
            event_type="NO_FACE",
            detector_source="face_presence",
            source_state="NO_FACES",
        )
        response = await self.client.get(
            f"/dashboard/sessions/{session.session_id}"
        )
        self.assertIn("Review event timeline", response.text)
        self.assertLess(
            response.text.index(first.event_id),
            response.text.index(second.event_id),
        )

    async def test_evidence_thumbnail_and_link_are_safe(self):
        session = self.seed_session()
        event_id = "GAZE_DEVIATION-0001"
        relative_path = f"data/evidence/{session.session_id}/{event_id}.jpg"
        self.write_evidence(relative_path)
        event = self.seed_event(
            session.session_id,
            event_id,
            evidence_path=str((self.project_root / relative_path).resolve()),
        )
        response = await self.client.get(
            f"/dashboard/sessions/{session.session_id}"
        )
        evidence_url = f"/events/{session.session_id}/{event.event_id}/evidence"
        detail_url = f"/dashboard/events/{session.session_id}/{event.event_id}"
        self.assertIn(f'src="{evidence_url}"', response.text)
        self.assertIn(f'href="{detail_url}"', response.text)
        self.assertNotIn(str(self.project_root), response.text)

    async def test_event_detail_page_renders(self):
        session = self.seed_session()
        event = self.seed_event(session.session_id, "GAZE_DEVIATION-0001")
        response = await self.client.get(
            f"/dashboard/events/{session.session_id}/{event.event_id}"
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("Gaze deviation", response.text)
        self.assertIn("Looking Left", response.text)
        self.assertIn("Detector metadata", response.text)
        self.assertIn("normalized", response.text)

    async def test_missing_session_and_event_return_404(self):
        session_response = await self.client.get(
            "/dashboard/sessions/missing-session"
        )
        self.assertEqual(session_response.status_code, 404)
        self.assertIn("Session not found", session_response.text)

        session = self.seed_session()
        event_response = await self.client.get(
            f"/dashboard/events/{session.session_id}/missing-event"
        )
        self.assertEqual(event_response.status_code, 404)
        self.assertIn("Review event not found", event_response.text)

    async def test_missing_evidence_does_not_break_pages(self):
        session = self.seed_session()
        event = self.seed_event(session.session_id, "GAZE_DEVIATION-0001")
        session_response = await self.client.get(
            f"/dashboard/sessions/{session.session_id}"
        )
        self.assertEqual(session_response.status_code, 200)
        self.assertNotIn("Evidence thumbnail", session_response.text)

        event_response = await self.client.get(
            f"/dashboard/events/{session.session_id}/{event.event_id}"
        )
        self.assertEqual(event_response.status_code, 200)
        self.assertIn("No evidence image", event_response.text)

    async def test_html_never_exposes_absolute_evidence_paths(self):
        session = self.seed_session()
        event_id = "PHONE_DETECTED-0001"
        relative_path = f"data/evidence/{session.session_id}/{event_id}.jpg"
        absolute_path = self.write_evidence(relative_path).resolve()
        event = self.seed_event(
            session.session_id,
            event_id,
            event_type="PHONE_DETECTED",
            detector_source="phone",
            source_state="PHONE_DETECTED",
            evidence_path=str(absolute_path),
            metadata={
                "debug_path": str(absolute_path),
                "score": np.float32(0.82),
            },
        )
        for url in (
            f"/dashboard/sessions/{session.session_id}",
            f"/dashboard/events/{session.session_id}/{event.event_id}",
        ):
            response = await self.client.get(url)
            self.assertEqual(response.status_code, 200)
            self.assertNotIn(str(absolute_path), response.text)
            self.assertNotIn(str(self.project_root), response.text)
        self.assertIn("local path omitted", response.text)

    async def test_database_unavailable_renders_clean_error_page(self):
        async def unavailable_service():
            yield None

        self.app.dependency_overrides[
            get_optional_session_service
        ] = unavailable_service
        response = await self.client.get("/")
        self.assertEqual(response.status_code, 503)
        self.assertIn("Database unavailable", response.text)
        self.assertNotIn(str(self.database_path), response.text)


if __name__ == "__main__":
    unittest.main()
