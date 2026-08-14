"""Common detector result model and thin wrappers around the working CV logic."""

import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from eye_movement import (
    GAZE_DOWN,
    GAZE_LEFT,
    GAZE_RIGHT,
    GAZE_UP,
    UNKNOWN,
    GazeState,
    analyze_eye_movement,
)
from face_context import FrameContext
from head_pose import (
    LOOKING_DOWN,
    LOOKING_LEFT,
    LOOKING_RIGHT,
    LOOKING_UP,
    TILTED,
    HeadPoseState,
    analyze_head_pose,
)
from mobile_detection import (
    PHONE_HIGH_RESOLUTION_TRIGGER_CONFIDENCE,
    PHONE_INFERENCE_EVERY_N_FRAMES,
    PHONE_RESULT_MAX_REUSE_AGE_SECONDS,
    YOLO_CONFIRMATION_INFERENCE_SIZE,
    YOLO_INFERENCE_SIZE,
    analyze_mobile_detection,
    load_mobile_model,
)

GAZE_SOURCE = "gaze"
HEAD_POSE_SOURCE = "head_pose"
PHONE_SOURCE = "phone"
FACE_PRESENCE_SOURCE = "face_presence"

PHONE_DETECTED = "PHONE_DETECTED"
NO_PHONE = "NO_PHONE"
CALIBRATING = "CALIBRATING"


@dataclass(slots=True)
class DetectionResult:
    """Small common representation returned by every detector."""

    detector: str
    state: str
    suspicious: bool
    confidence: float | None = None
    bounding_box: tuple[int, int, int, int] | None = None
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Detector(Protocol):
    """Minimal common API; detectors may ignore inputs they do not need."""

    detector: str

    def detect(
        self,
        clean_frame: np.ndarray,
        frame_context: FrameContext,
        timestamp: float,
    ) -> DetectionResult: ...


class GazeDetector:
    detector = GAZE_SOURCE

    def __init__(self):
        self.gaze_state = GazeState()

    @property
    def calibration_sample_count(self):
        return len(self.gaze_state.calibration_samples)

    @property
    def calibration_baseline(self):
        return self.gaze_state.baseline

    def finalize_calibration(self, minimum_samples):
        return self.gaze_state.finalize_calibration(minimum_samples)

    def reset(self):
        self.gaze_state.reset()

    def detect(
        self,
        clean_frame,
        frame_context,
        timestamp,
        *,
        calibrating=False,
        head_relative_angles=None,
        head_pose_reliable=True,
    ):
        del clean_frame
        state, metadata = analyze_eye_movement(
            frame_context,
            state=self.gaze_state,
            calibrating=calibrating,
            head_relative_angles=head_relative_angles,
            head_pose_reliable=head_pose_reliable,
        )
        return DetectionResult(
            detector=self.detector,
            state=state,
            suspicious=state in {GAZE_LEFT, GAZE_RIGHT, GAZE_UP, GAZE_DOWN},
            bounding_box=frame_context.primary_face_box,
            timestamp=timestamp,
            metadata=metadata,
        )


class HeadPoseDetector:
    detector = HEAD_POSE_SOURCE

    def __init__(self):
        self.pose_state = HeadPoseState()
        self.calibrated_rotation = None

    def set_calibration(self, calibrated_rotation):
        self.calibrated_rotation = np.asarray(
            calibrated_rotation,
            dtype=np.float64,
        ).copy()
        self.pose_state.set_reference(self.calibrated_rotation)

    def reset(self):
        self.pose_state.reset()
        self.calibrated_rotation = None

    def detect(self, clean_frame, frame_context, timestamp):
        del clean_frame
        state, metadata = analyze_head_pose(
            frame_context,
            calibrated_rotation=self.calibrated_rotation,
            state=self.pose_state,
        )
        return DetectionResult(
            detector=self.detector,
            state=state,
            suspicious=state
            in {LOOKING_LEFT, LOOKING_RIGHT, LOOKING_UP, LOOKING_DOWN, TILTED},
            bounding_box=frame_context.primary_face_box,
            timestamp=timestamp,
            metadata=metadata,
        )


class PhoneDetector:
    detector = PHONE_SOURCE

    def __init__(
        self,
        model=None,
        inference_size=YOLO_INFERENCE_SIZE,
        confirmation_size=YOLO_CONFIRMATION_INFERENCE_SIZE,
        confirmation_trigger=PHONE_HIGH_RESOLUTION_TRIGGER_CONFIDENCE,
        inference_every_n_frames=PHONE_INFERENCE_EVERY_N_FRAMES,
        max_reuse_age_seconds=PHONE_RESULT_MAX_REUSE_AGE_SECONDS,
        clock=time.monotonic,
    ):
        if not isinstance(inference_size, int) or inference_size <= 0:
            raise ValueError("YOLO inference size must be a positive integer.")
        if confirmation_size is not None and (
            not isinstance(confirmation_size, int) or confirmation_size <= 0
        ):
            raise ValueError(
                "YOLO confirmation size must be a positive integer or None."
            )
        if not 0.0 <= confirmation_trigger <= 1.0:
            raise ValueError("YOLO confirmation trigger must be between 0 and 1.")
        if (
            not isinstance(inference_every_n_frames, int)
            or inference_every_n_frames <= 0
        ):
            raise ValueError("Phone inference cadence must be a positive integer.")
        if max_reuse_age_seconds <= 0:
            raise ValueError("Phone result maximum reuse age must be positive.")
        self.model = model if model is not None else load_mobile_model()
        self.inference_size = inference_size
        self.confirmation_size = confirmation_size
        self.confirmation_trigger = float(confirmation_trigger)
        self.inference_every_n_frames = inference_every_n_frames
        self.max_reuse_age_seconds = float(max_reuse_age_seconds)
        self.clock = clock
        self.reset()

    def reset(self):
        self._last_fresh_result = None
        self._last_inference_at = None
        self._frames_since_inference = 0
        self._inference_sequence = 0

    def detect(self, clean_frame, frame_context, timestamp):
        del frame_context
        should_execute = (
            self._last_fresh_result is None
            or self._frames_since_inference
            >= self.inference_every_n_frames - 1
        )
        if not should_execute:
            self._frames_since_inference += 1
            return self._reused_result(timestamp)

        mobile_detected, metadata = analyze_mobile_detection(
            clean_frame,
            self.model,
            inference_size=self.inference_size,
            confirmation_size=self.confirmation_size,
            confirmation_trigger=self.confirmation_trigger,
        )
        accepted = metadata["accepted_detections"]
        best_detection = (
            max(accepted, key=lambda item: item["confidence"])
            if accepted
            else None
        )
        self._inference_sequence += 1
        metadata = {
            **metadata,
            "inference_executed": True,
            "fresh": True,
            "reused": False,
            "stale": False,
            "result_age_seconds": 0.0,
            "frames_since_inference": 0,
            "inference_every_n_frames": self.inference_every_n_frames,
            "inference_sequence": self._inference_sequence,
        }
        result = DetectionResult(
            detector=self.detector,
            state=PHONE_DETECTED if mobile_detected else NO_PHONE,
            suspicious=mobile_detected,
            confidence=(
                float(best_detection["confidence"])
                if best_detection is not None
                else None
            ),
            bounding_box=(
                best_detection["bounding_box"]
                if best_detection is not None
                else None
            ),
            timestamp=timestamp,
            metadata=metadata,
        )
        self._last_fresh_result = result
        self._last_inference_at = float(self.clock())
        self._frames_since_inference = 0
        return result

    def _reused_result(self, timestamp):
        cached = self._last_fresh_result
        age_seconds = max(0.0, float(self.clock()) - self._last_inference_at)
        stale = age_seconds > self.max_reuse_age_seconds
        metadata = {
            **cached.metadata,
            "inference_executed": False,
            "fresh": False,
            "reused": not stale,
            "stale": stale,
            "result_age_seconds": age_seconds,
            "frames_since_inference": self._frames_since_inference,
            "cached_state": cached.state,
            "cached_suspicious": cached.suspicious,
            "cached_confidence": cached.confidence,
            "cached_bounding_box": cached.bounding_box,
        }
        if stale:
            # Expired cached boxes are not rendered or exposed as current candidates.
            metadata["predictions"] = ()
            metadata["accepted_detections"] = ()

        # A skipped frame is intentionally indeterminate for temporal alert rules.
        # Cached state is retained only in explicit metadata for bounded UI reuse.
        return DetectionResult(
            detector=self.detector,
            state=UNKNOWN,
            suspicious=False,
            timestamp=timestamp,
            metadata=metadata,
        )
