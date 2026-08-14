"""Read-focused FastAPI boundary for persisted monitoring sessions and events."""

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict

from config import (
    DATABASE_PATH as DEFAULT_DATABASE_PATH,
)
from config import (
    EVIDENCE_DIRECTORY as DEFAULT_EVIDENCE_ROOT,
)
from config import (
    PROJECT_ROOT,
    STATIC_DIRECTORY,
    TEMPLATES_DIRECTORY,
)
from dashboard import create_dashboard_router
from evidence import resolve_event_evidence_path
from persistence import PersistenceError, SQLiteRepository
from session_service import (
    EventNotFoundError,
    MonitoringSession,
    PersistedEvent,
    SessionNotFoundError,
    SessionService,
    SessionStatus,
)

DEFAULT_PAGE_LIMIT = 50
MAX_PAGE_LIMIT = 100

LOGGER = logging.getLogger(__name__)

PageLimit = Annotated[int, Query(ge=1, le=MAX_PAGE_LIMIT)]
PageOffset = Annotated[int, Query(ge=0)]


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "unavailable"]
    api: Literal["ready"] = "ready"
    database: Literal["ready", "unavailable"]


class SessionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    status: SessionStatus
    created_at_utc: datetime
    started_at_utc: datetime | None
    ended_at_utc: datetime | None
    source: str
    calibration_completed: bool
    calibration_completed_at_utc: datetime | None
    calibration_details: dict[str, Any] | None
    event_count: int
    average_fps: float | None
    failure_reason: str | None


class EventResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str
    session_id: str
    event_type: str
    detector_source: str
    source_state: str
    status: Literal["CONFIRMED", "RESOLVED"]
    started_at_utc: datetime
    confirmed_at_utc: datetime
    resolved_at_utc: datetime | None
    duration_seconds: float | None
    confidence: float | None
    bounding_box: tuple[int, int, int, int] | None
    evidence_available: bool
    evidence_url: str | None
    metadata: dict[str, Any]


async def get_session_service(request: Request):
    """Provide an isolated SQLite connection for one API request."""

    repository = SQLiteRepository(
        request.app.state.database_path,
        initialize_schema=False,
    )
    try:
        yield SessionService(repository)
    finally:
        repository.close()


async def get_optional_session_service(request: Request):
    """Yield ``None`` so dashboard routes can render database error pages."""

    try:
        repository = SQLiteRepository(
            request.app.state.database_path,
            initialize_schema=False,
        )
    except PersistenceError:
        yield None
        return
    try:
        yield SessionService(repository)
    finally:
        repository.close()


def _session_response(session: MonitoringSession):
    return SessionResponse(
        session_id=session.session_id,
        status=session.status,
        created_at_utc=session.created_at_utc,
        started_at_utc=session.started_at_utc,
        ended_at_utc=session.ended_at_utc,
        source=session.video_source,
        calibration_completed=session.calibration_completed_at_utc is not None,
        calibration_completed_at_utc=session.calibration_completed_at_utc,
        calibration_details=session.calibration_details,
        event_count=session.event_count,
        average_fps=session.average_fps,
        failure_reason=session.failure_reason,
    )


def _resolve_evidence_path(app, event):
    return resolve_event_evidence_path(
        event,
        app.state.evidence_root,
        app.state.project_root,
    )


def _event_response(request, event: PersistedEvent):
    evidence_path = _resolve_evidence_path(request.app, event)
    evidence_url = None
    if evidence_path is not None:
        evidence_url = str(
            request.app.url_path_for(
                "get_event_evidence",
                session_id=event.session_id,
                event_id=event.event_id,
            )
        )
    return EventResponse(
        event_id=event.event_id,
        session_id=event.session_id,
        event_type=event.event_type,
        detector_source=event.detector_source,
        source_state=event.source_state,
        status=event.status,
        started_at_utc=event.started_at_utc,
        confirmed_at_utc=event.confirmed_at_utc,
        resolved_at_utc=event.resolved_at_utc,
        duration_seconds=event.duration_seconds,
        confidence=event.confidence,
        bounding_box=event.bounding_box,
        evidence_available=evidence_path is not None,
        evidence_url=evidence_url,
        metadata=event.metadata,
    )


def create_app(
    database_path=DEFAULT_DATABASE_PATH,
    evidence_root=DEFAULT_EVIDENCE_ROOT,
    project_root=PROJECT_ROOT,
):
    database_path = Path(database_path).resolve()
    evidence_root = Path(evidence_root).resolve()
    project_root = Path(project_root).resolve()

    @asynccontextmanager
    async def lifespan(application):
        repository = None
        try:
            repository = SQLiteRepository(database_path)
            if not repository.check_health():
                raise PersistenceError("SQLite schema readiness check returned false.")
        except PersistenceError:
            LOGGER.exception("API database initialization failed.")
        finally:
            if repository is not None:
                repository.close()
        yield

    application = FastAPI(
        title="ProctorVision Review API",
        version="1.0.0",
        description="Read-only access to persisted monitoring sessions and review events.",
        lifespan=lifespan,
    )
    application.state.database_path = database_path
    application.state.evidence_root = evidence_root
    application.state.project_root = project_root
    application.mount(
        "/static",
        StaticFiles(directory=str(STATIC_DIRECTORY)),
        name="static",
    )

    @application.exception_handler(PersistenceError)
    async def persistence_error_handler(_request, _exc):
        return JSONResponse(
            status_code=503,
            content={"detail": "Persistence service unavailable."},
        )

    @application.exception_handler(Exception)
    async def unexpected_error_handler(_request, exc):
        LOGGER.error(
            "Unexpected API failure: %s",
            type(exc).__name__,
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error."},
        )

    @application.get("/health", response_model=HealthResponse)
    async def get_health():
        repository = None
        try:
            repository = SQLiteRepository(
                database_path,
                initialize_schema=False,
            )
            if not repository.check_health():
                raise PersistenceError("SQLite readiness check returned false.")
            return HealthResponse(status="ok", database="ready")
        except PersistenceError:
            return JSONResponse(
                status_code=503,
                content=HealthResponse(
                    status="unavailable",
                    database="unavailable",
                ).model_dump(mode="json"),
            )
        finally:
            if repository is not None:
                repository.close()

    @application.get("/sessions", response_model=list[SessionResponse])
    async def list_sessions(
        limit: PageLimit = DEFAULT_PAGE_LIMIT,
        offset: PageOffset = 0,
        service: SessionService = Depends(get_session_service),
    ):
        return [
            _session_response(session)
            for session in service.list_sessions(limit=limit, offset=offset)
        ]

    @application.get("/sessions/{session_id}", response_model=SessionResponse)
    async def get_session(
        session_id: str,
        service: SessionService = Depends(get_session_service),
    ):
        try:
            return _session_response(service.get_session(session_id))
        except SessionNotFoundError:
            return JSONResponse(
                status_code=404,
                content={"detail": "Session not found."},
            )

    @application.get(
        "/sessions/{session_id}/events",
        response_model=list[EventResponse],
    )
    async def list_session_events(
        request: Request,
        session_id: str,
        limit: PageLimit = DEFAULT_PAGE_LIMIT,
        offset: PageOffset = 0,
        service: SessionService = Depends(get_session_service),
    ):
        try:
            events = service.list_events(session_id, limit=limit, offset=offset)
        except SessionNotFoundError:
            return JSONResponse(
                status_code=404,
                content={"detail": "Session not found."},
            )
        return [_event_response(request, event) for event in events]

    @application.get(
        "/events/{session_id}/{event_id}",
        response_model=EventResponse,
    )
    async def get_event(
        request: Request,
        session_id: str,
        event_id: str,
        service: SessionService = Depends(get_session_service),
    ):
        try:
            event = service.get_event(session_id, event_id)
        except EventNotFoundError:
            return JSONResponse(
                status_code=404,
                content={"detail": "Event not found."},
            )
        return _event_response(request, event)

    @application.get(
        "/events/{session_id}/{event_id}/evidence",
        response_class=FileResponse,
        name="get_event_evidence",
    )
    async def get_event_evidence(
        session_id: str,
        event_id: str,
        service: SessionService = Depends(get_session_service),
    ):
        try:
            event = service.get_event(session_id, event_id)
        except EventNotFoundError:
            return JSONResponse(
                status_code=404,
                content={"detail": "Event not found."},
            )
        evidence_path = _resolve_evidence_path(application, event)
        if evidence_path is None:
            return JSONResponse(
                status_code=404,
                content={"detail": "Evidence not found."},
            )
        return FileResponse(evidence_path, media_type="image/jpeg")

    application.include_router(
        create_dashboard_router(
            get_optional_session_service,
            TEMPLATES_DIRECTORY,
        )
    )

    return application


app = create_app()
