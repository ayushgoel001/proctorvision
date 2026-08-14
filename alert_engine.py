"""Detector-independent temporal rules that produce review events."""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from detectors import (
    CALIBRATING,
    FACE_PRESENCE_SOURCE,
    GAZE_SOURCE,
    HEAD_POSE_SOURCE,
    PHONE_SOURCE,
    DetectionResult,
)
from eye_movement import NO_FACE, UNKNOWN
from face_context import MULTIPLE_FACES, NO_FACES, PRIMARY_MISSING

GAZE_DEVIATION = "GAZE_DEVIATION"
HEAD_DEVIATION = "HEAD_DEVIATION"
PHONE_DETECTED = "PHONE_DETECTED"
NO_FACE_EVENT = "NO_FACE"
MULTIPLE_FACES_EVENT = "MULTIPLE_FACES"

DEFAULT_MINIMUM_DURATION_SECONDS = 3.0
DEFAULT_CLEAR_GRACE_SECONDS = 0.5
DEFAULT_COOLDOWN_SECONDS = 5.0


class EventStatus(str, Enum):
    IDLE = "IDLE"
    PENDING = "PENDING"
    CONFIRMED = "CONFIRMED"
    RESOLVED = "RESOLVED"


class ObservationKind(str, Enum):
    SUSPICIOUS = "SUSPICIOUS"
    CLEAR = "CLEAR"
    INDETERMINATE = "INDETERMINATE"


@dataclass(frozen=True, slots=True)
class AlertRule:
    event_type: str
    detector_source: str
    minimum_duration_seconds: float = DEFAULT_MINIMUM_DURATION_SECONDS
    clear_grace_seconds: float = DEFAULT_CLEAR_GRACE_SECONDS
    cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS

    def __post_init__(self):
        for field_name in (
            "minimum_duration_seconds",
            "clear_grace_seconds",
            "cooldown_seconds",
        ):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} cannot be negative.")


DEFAULT_ALERT_RULES = (
    AlertRule(GAZE_DEVIATION, GAZE_SOURCE),
    AlertRule(HEAD_DEVIATION, HEAD_POSE_SOURCE),
    AlertRule(PHONE_DETECTED, PHONE_SOURCE),
    AlertRule(NO_FACE_EVENT, FACE_PRESENCE_SOURCE),
    AlertRule(MULTIPLE_FACES_EVENT, FACE_PRESENCE_SOURCE),
)


@dataclass(frozen=True, slots=True)
class ReviewEvent:
    event_id: str
    event_type: str
    status: EventStatus
    detector_source: str
    source_state: str
    started_at: float
    confirmed_at: float
    resolved_at: float | None
    confidence: float | None
    bounding_box: tuple[int, int, int, int] | None
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RuleState:
    event_type: str
    status: EventStatus
    started_at: float | None
    confirmed_at: float | None
    resolved_at: float | None
    clear_started_at: float | None
    cooldown_until: float | None
    last_observation: ObservationKind
    source_state: str | None


@dataclass(frozen=True, slots=True)
class AlertUpdate:
    newly_confirmed_events: tuple[ReviewEvent, ...]
    resolved_events: tuple[ReviewEvent, ...]
    current_rule_states: dict[str, RuleState]


@dataclass(slots=True)
class _RuleRuntime:
    status: EventStatus = EventStatus.IDLE
    started_at: float | None = None
    confirmed_at: float | None = None
    resolved_at: float | None = None
    clear_started_at: float | None = None
    cooldown_until: float | None = None
    event_id: str | None = None
    last_observation: ObservationKind = ObservationKind.CLEAR
    last_detection: DetectionResult | None = None


class AlertEngine:
    """Maintain independent temporal state for each configured review-event rule."""

    def __init__(
        self,
        rules=DEFAULT_ALERT_RULES,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.rules = tuple(rules)
        event_types = [rule.event_type for rule in self.rules]
        if len(event_types) != len(set(event_types)):
            raise ValueError("Alert rule event types must be unique.")
        self.clock = clock
        self._runtimes = {
            rule.event_type: _RuleRuntime() for rule in self.rules
        }
        self._event_sequence = 0

    def reset(self):
        self._runtimes = {
            rule.event_type: _RuleRuntime() for rule in self.rules
        }
        self._event_sequence = 0

    def update(self, detections):
        now = float(self.clock())
        detections_by_source = {
            detection.detector: detection for detection in detections
        }
        newly_confirmed = []
        resolved = []

        for rule in self.rules:
            runtime = self._runtimes[rule.event_type]
            detection = detections_by_source.get(rule.detector_source)
            observation = self._classify_observation(rule, detection)
            runtime.last_observation = observation

            resolved_this_update = self._expire_clear_period_if_needed(
                rule,
                runtime,
                now,
                resolved,
            )

            if runtime.status == EventStatus.RESOLVED:
                if resolved_this_update or now < runtime.cooldown_until:
                    continue
                self._reset_runtime(runtime)

            if observation == ObservationKind.SUSPICIOUS:
                if runtime.status == EventStatus.IDLE:
                    runtime.status = EventStatus.PENDING
                    runtime.started_at = now
                    runtime.last_detection = detection
                else:
                    runtime.clear_started_at = None
                    runtime.last_detection = detection

                if (
                    runtime.status == EventStatus.PENDING
                    and now - runtime.started_at
                    >= rule.minimum_duration_seconds
                ):
                    runtime.status = EventStatus.CONFIRMED
                    runtime.confirmed_at = now
                    runtime.event_id = self._next_event_id(rule.event_type)
                    newly_confirmed.append(
                        self._event_snapshot(rule, runtime, EventStatus.CONFIRMED)
                    )
            elif runtime.status in {EventStatus.PENDING, EventStatus.CONFIRMED}:
                if runtime.clear_started_at is None:
                    runtime.clear_started_at = now

        return AlertUpdate(
            newly_confirmed_events=tuple(newly_confirmed),
            resolved_events=tuple(resolved),
            current_rule_states=self.current_rule_states(),
        )

    def current_rule_states(self):
        return {
            rule.event_type: RuleState(
                event_type=rule.event_type,
                status=runtime.status,
                started_at=runtime.started_at,
                confirmed_at=runtime.confirmed_at,
                resolved_at=runtime.resolved_at,
                clear_started_at=runtime.clear_started_at,
                cooldown_until=runtime.cooldown_until,
                last_observation=runtime.last_observation,
                source_state=(
                    runtime.last_detection.state
                    if runtime.last_detection is not None
                    else None
                ),
            )
            for rule in self.rules
            for runtime in (self._runtimes[rule.event_type],)
        }

    def _expire_clear_period_if_needed(
        self,
        rule,
        runtime,
        now,
        resolved_events,
    ):
        if (
            runtime.status not in {EventStatus.PENDING, EventStatus.CONFIRMED}
            or runtime.clear_started_at is None
            or now - runtime.clear_started_at < rule.clear_grace_seconds
        ):
            return False

        if runtime.status == EventStatus.PENDING:
            self._reset_runtime(runtime)
            return False

        runtime.status = EventStatus.RESOLVED
        runtime.resolved_at = runtime.clear_started_at + rule.clear_grace_seconds
        runtime.cooldown_until = runtime.resolved_at + rule.cooldown_seconds
        resolved_events.append(
            self._event_snapshot(rule, runtime, EventStatus.RESOLVED)
        )
        return True

    @staticmethod
    def _classify_observation(rule, detection):
        if detection is None:
            return ObservationKind.INDETERMINATE

        if rule.event_type in {GAZE_DEVIATION, HEAD_DEVIATION}:
            if detection.state in {UNKNOWN, NO_FACE, CALIBRATING}:
                return ObservationKind.INDETERMINATE
            return (
                ObservationKind.SUSPICIOUS
                if detection.suspicious
                else ObservationKind.CLEAR
            )

        if rule.event_type == PHONE_DETECTED:
            if detection.state in {UNKNOWN, CALIBRATING}:
                return ObservationKind.INDETERMINATE
            return (
                ObservationKind.SUSPICIOUS
                if detection.suspicious
                else ObservationKind.CLEAR
            )

        if rule.event_type == NO_FACE_EVENT:
            primary_missing = (
                detection.metadata.get("association_status") == PRIMARY_MISSING
            )
            return (
                ObservationKind.SUSPICIOUS
                if detection.state in {NO_FACES, PRIMARY_MISSING}
                or primary_missing
                else ObservationKind.CLEAR
            )

        if rule.event_type == MULTIPLE_FACES_EVENT:
            return (
                ObservationKind.SUSPICIOUS
                if detection.state == MULTIPLE_FACES
                else ObservationKind.CLEAR
            )

        return ObservationKind.INDETERMINATE

    def _event_snapshot(self, rule, runtime, status):
        detection = runtime.last_detection
        if detection is None or runtime.event_id is None:
            raise RuntimeError("Confirmed event state is missing its source detection.")
        return ReviewEvent(
            event_id=runtime.event_id,
            event_type=rule.event_type,
            status=status,
            detector_source=detection.detector,
            source_state=detection.state,
            started_at=runtime.started_at,
            confirmed_at=runtime.confirmed_at,
            resolved_at=runtime.resolved_at,
            confidence=detection.confidence,
            bounding_box=detection.bounding_box,
            metadata=dict(detection.metadata),
        )

    def _next_event_id(self, event_type):
        self._event_sequence += 1
        return f"{event_type}-{self._event_sequence:04d}"

    @staticmethod
    def _reset_runtime(runtime):
        runtime.status = EventStatus.IDLE
        runtime.started_at = None
        runtime.confirmed_at = None
        runtime.resolved_at = None
        runtime.clear_started_at = None
        runtime.cooldown_until = None
        runtime.event_id = None
        runtime.last_detection = None
