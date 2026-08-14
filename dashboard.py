"""Server-rendered dashboard routes and presentation view models."""

import re
from collections import Counter
from datetime import timezone
from pathlib import Path

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from evidence import resolve_event_evidence_path
from persistence import PersistenceError
from session_service import (
    EventNotFoundError,
    SessionNotFoundError,
    SessionService,
    SessionStatus,
)

RECENT_SESSION_LIMIT = 12
ACTIVE_SESSION_STATUSES = {SessionStatus.CALIBRATING, SessionStatus.RUNNING}
EVENT_TYPE_LABELS = (
    ("GAZE_DEVIATION", "Gaze deviation"),
    ("HEAD_DEVIATION", "Head deviation"),
    ("PHONE_DETECTED", "Phone detected"),
    ("NO_FACE", "No face"),
    ("MULTIPLE_FACES", "Multiple faces"),
)
WINDOWS_ABSOLUTE_PATH = re.compile(r"[A-Za-z]:[\\/][^\s\"'<>]+")
POSIX_ABSOLUTE_PATH = re.compile(r"(?<![:\w])/(?:[^\s\"'<>]+)")


def _redact_local_paths(value):
    if not isinstance(value, str):
        return value
    value = WINDOWS_ABSOLUTE_PATH.sub("[local path omitted]", value)
    return POSIX_ABSOLUTE_PATH.sub("[local path omitted]", value)


def _safe_display_value(value):
    if isinstance(value, str):
        return _redact_local_paths(value)
    if isinstance(value, dict):
        return {
            str(key): _safe_display_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_safe_display_value(item) for item in value]
    return value


def _format_datetime(value):
    if value is None:
        return "—"
    return value.astimezone(timezone.utc).strftime("%d %b %Y, %H:%M:%S UTC")


def _format_duration(value):
    return "—" if value is None else f"{value:.1f} s"


def _format_fps(value):
    return "—" if value is None else f"{value:.1f} FPS"


def _session_view(request, session):
    return {
        "session_id": session.session_id,
        "status": session.status.value,
        "status_class": session.status.value.lower(),
        "created_at": _format_datetime(session.created_at_utc),
        "started_at": _format_datetime(session.started_at_utc),
        "ended_at": _format_datetime(session.ended_at_utc),
        "source": _redact_local_paths(session.video_source),
        "calibration_completed": session.calibration_completed_at_utc is not None,
        "calibration_completed_at": _format_datetime(
            session.calibration_completed_at_utc
        ),
        "calibration_details": _safe_display_value(session.calibration_details),
        "event_count": session.event_count,
        "average_fps": _format_fps(session.average_fps),
        "failure_reason": _redact_local_paths(session.failure_reason),
        "detail_url": str(
            request.app.url_path_for(
                "dashboard_session",
                session_id=session.session_id,
            )
        ),
    }


def _event_view(request, event):
    evidence_path = resolve_event_evidence_path(
        event,
        request.app.state.evidence_root,
        request.app.state.project_root,
    )
    evidence_url = None
    if evidence_path is not None:
        evidence_url = str(
            request.app.url_path_for(
                "get_event_evidence",
                session_id=event.session_id,
                event_id=event.event_id,
            )
        )
    return {
        "event_id": event.event_id,
        "event_type": event.event_type,
        "event_type_label": dict(EVENT_TYPE_LABELS).get(
            event.event_type,
            event.event_type.replace("_", " ").title(),
        ),
        "detector_source": event.detector_source,
        "source_state": event.source_state,
        "status": event.status,
        "status_class": event.status.lower(),
        "started_at": _format_datetime(event.started_at_utc),
        "confirmed_at": _format_datetime(event.confirmed_at_utc),
        "resolved_at": _format_datetime(event.resolved_at_utc),
        "duration": _format_duration(event.duration_seconds),
        "confidence": (
            None if event.confidence is None else f"{event.confidence:.2f}"
        ),
        "bounding_box": event.bounding_box,
        "metadata": _safe_display_value(event.metadata),
        "evidence_available": evidence_path is not None,
        "evidence_url": evidence_url,
        "detail_url": str(
            request.app.url_path_for(
                "dashboard_event",
                session_id=event.session_id,
                event_id=event.event_id,
            )
        ),
    }


def _event_count_views(events):
    counts = Counter(event.event_type for event in events)
    return [
        {
            "event_type": event_type,
            "label": label,
            "count": counts[event_type],
        }
        for event_type, label in EVENT_TYPE_LABELS
    ]


def create_dashboard_router(optional_service_dependency, templates_directory):
    templates = Jinja2Templates(directory=str(Path(templates_directory).resolve()))
    router = APIRouter(include_in_schema=False)

    def render_error(request, status_code, title, message):
        return templates.TemplateResponse(
            request=request,
            name="error.html",
            context={
                "page_title": title,
                "error_title": title,
                "error_message": message,
                "auto_refresh": False,
            },
            status_code=status_code,
        )

    def database_unavailable(request):
        return render_error(
            request,
            503,
            "Database unavailable",
            "Session records cannot be read right now. Refresh after the monitoring "
            "database becomes available.",
        )

    @router.get("/", response_class=HTMLResponse, name="dashboard_home")
    async def dashboard_home(
        request: Request,
        service: SessionService | None = Depends(optional_service_dependency),
    ):
        if service is None:
            return database_unavailable(request)
        try:
            if not service.repository.check_health():
                return database_unavailable(request)
            sessions = service.list_sessions(limit=RECENT_SESSION_LIMIT, offset=0)
        except PersistenceError:
            return database_unavailable(request)

        session_views = [_session_view(request, session) for session in sessions]
        return templates.TemplateResponse(
            request=request,
            name="dashboard.html",
            context={
                "page_title": "Review dashboard",
                "database_ready": True,
                "sessions": session_views,
                "active_count": sum(
                    session.status in ACTIVE_SESSION_STATUSES for session in sessions
                ),
                "review_event_count": sum(
                    session.event_count for session in sessions
                ),
                "auto_refresh": any(
                    session.status in ACTIVE_SESSION_STATUSES for session in sessions
                ),
            },
        )

    @router.get(
        "/dashboard/sessions/{session_id}",
        response_class=HTMLResponse,
        name="dashboard_session",
    )
    async def dashboard_session(
        request: Request,
        session_id: str,
        service: SessionService | None = Depends(optional_service_dependency),
    ):
        if service is None:
            return database_unavailable(request)
        try:
            session = service.get_session(session_id)
            events = service.list_events(session_id)
        except SessionNotFoundError:
            return render_error(
                request,
                404,
                "Session not found",
                "The requested monitoring session does not exist.",
            )
        except PersistenceError:
            return database_unavailable(request)

        return templates.TemplateResponse(
            request=request,
            name="session_detail.html",
            context={
                "page_title": "Session details",
                "session": _session_view(request, session),
                "events": [_event_view(request, event) for event in events],
                "event_counts": _event_count_views(events),
                "auto_refresh": session.status in ACTIVE_SESSION_STATUSES,
            },
        )

    @router.get(
        "/dashboard/events/{session_id}/{event_id}",
        response_class=HTMLResponse,
        name="dashboard_event",
    )
    async def dashboard_event(
        request: Request,
        session_id: str,
        event_id: str,
        service: SessionService | None = Depends(optional_service_dependency),
    ):
        if service is None:
            return database_unavailable(request)
        try:
            session = service.get_session(session_id)
            event = service.get_event(session_id, event_id)
        except (SessionNotFoundError, EventNotFoundError):
            return render_error(
                request,
                404,
                "Review event not found",
                "The requested review event does not exist in this session.",
            )
        except PersistenceError:
            return database_unavailable(request)

        return templates.TemplateResponse(
            request=request,
            name="event_detail.html",
            context={
                "page_title": "Review event details",
                "session": _session_view(request, session),
                "event": _event_view(request, event),
                "auto_refresh": session.status in ACTIVE_SESSION_STATUSES,
            },
        )

    return router
