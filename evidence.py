"""Safe filesystem resolution shared by REST and dashboard evidence views."""

from pathlib import Path


def _path_is_within(path, directory):
    try:
        path.relative_to(directory)
        return True
    except ValueError:
        return False


def resolve_event_evidence_path(event, evidence_root, project_root):
    """Return an existing session-scoped evidence file, or ``None`` if unsafe."""

    if not event.evidence_path:
        return None

    try:
        evidence_root = Path(evidence_root).resolve()
        project_root = Path(project_root).resolve()
        session_root = (evidence_root / event.session_id).resolve()
        if not _path_is_within(session_root, evidence_root):
            return None

        recorded_path = Path(event.evidence_path)
        candidate = (
            recorded_path
            if recorded_path.is_absolute()
            else project_root / recorded_path
        ).resolve()
        if not _path_is_within(candidate, session_root) or not candidate.is_file():
            return None
        return candidate
    except (OSError, RuntimeError, ValueError):
        return None
