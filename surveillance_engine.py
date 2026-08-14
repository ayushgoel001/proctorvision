"""One-frame orchestration for the modular surveillance pipeline."""

import time
from dataclasses import dataclass

import numpy as np

from detectors import (
    CALIBRATING,
    FACE_PRESENCE_SOURCE,
    PHONE_SOURCE,
    DetectionResult,
    GazeDetector,
    HeadPoseDetector,
    PhoneDetector,
)
from face_context import (
    MULTIPLE_FACES,
    NO_FACES,
    PRIMARY_MISSING,
    FrameContext,
    MediaPipeFaceLandmarkProvider,
    PrimaryFaceTracker,
    build_frame_context,
)


@dataclass(frozen=True, slots=True)
class ProcessingTiming:
    face_landmarks_ms: float
    gaze_ms: float
    head_pose_ms: float
    phone_ms: float
    phone_inference_executed: bool
    total_ms: float


@dataclass(frozen=True, slots=True)
class FrameProcessingResult:
    detections: tuple[DetectionResult, ...]
    face_context: FrameContext
    timing: ProcessingTiming

    def detection(self, source):
        for result in self.detections:
            if result.detector == source:
                return result
        raise KeyError(f"No detection result for source {source!r}.")


class SurveillanceEngine:
    """Build shared context and coordinate one frame of detector execution."""

    def __init__(
        self,
        gaze_detector=None,
        head_detector=None,
        phone_detector=None,
        face_provider=None,
    ):
        self.face_tracker = PrimaryFaceTracker()
        self.face_provider = face_provider or MediaPipeFaceLandmarkProvider()
        self.gaze_detector = gaze_detector or GazeDetector()
        self.head_detector = head_detector or HeadPoseDetector()
        self.phone_detector = phone_detector or PhoneDetector()

    @property
    def calibrated_rotation(self):
        return self.head_detector.calibrated_rotation

    def set_head_calibration(self, calibrated_rotation):
        self.head_detector.set_calibration(calibrated_rotation)

    @property
    def gaze_calibration_sample_count(self):
        return self.gaze_detector.calibration_sample_count

    @property
    def gaze_calibration_baseline(self):
        return self.gaze_detector.calibration_baseline

    def finalize_gaze_calibration(self, minimum_samples):
        return self.gaze_detector.finalize_calibration(minimum_samples)

    def reset_session(self):
        self.face_tracker.reset()
        self.face_provider.reset()
        self.gaze_detector.reset()
        self.head_detector.reset()
        reset_phone = getattr(self.phone_detector, "reset", None)
        if reset_phone is not None:
            reset_phone()

    def close(self):
        close_provider = getattr(self.face_provider, "close", None)
        if close_provider is not None:
            close_provider()

    def process_frame(self, clean_frame, calibrating=None):
        if clean_frame is None or not isinstance(clean_frame, np.ndarray):
            raise ValueError("A clean NumPy frame is required.")
        if clean_frame.size == 0:
            raise ValueError("A non-empty clean frame is required.")

        calibrating = (
            self.calibrated_rotation is None
            if calibrating is None
            else bool(calibrating)
        )
        timestamp = time.time()
        total_started_at = time.perf_counter()

        stage_started_at = time.perf_counter()
        frame_context = build_frame_context(
            clean_frame,
            self.face_tracker,
            self.face_provider,
        )
        face_landmarks_ms = (time.perf_counter() - stage_started_at) * 1000.0

        face_result = DetectionResult(
            detector=FACE_PRESENCE_SOURCE,
            state=frame_context.face_observation,
            suspicious=frame_context.face_observation
            in {MULTIPLE_FACES, PRIMARY_MISSING, NO_FACES},
            bounding_box=frame_context.primary_face_box,
            timestamp=timestamp,
            metadata={
                "face_status": frame_context.face_status,
                "face_count": frame_context.face_count,
                "face_boxes": frame_context.face_boxes,
                "association_status": frame_context.association_status,
                "additional_faces_present": frame_context.additional_faces_present,
                "primary_missing_seconds": frame_context.primary_missing_seconds,
                "landmark_backend": self.face_provider.backend_name,
            },
        )

        stage_started_at = time.perf_counter()
        head_result = self.head_detector.detect(
            clean_frame,
            frame_context,
            timestamp,
        )
        head_pose_ms = (time.perf_counter() - stage_started_at) * 1000.0

        head_relative_angles = head_result.metadata.get("relative_angles")
        head_pose_reliable = (
            head_result.metadata.get("current_rotation") is not None
            if calibrating
            else head_relative_angles is not None
        )
        stage_started_at = time.perf_counter()
        gaze_result = self.gaze_detector.detect(
            clean_frame,
            frame_context,
            timestamp,
            calibrating=calibrating,
            head_relative_angles=head_relative_angles,
            head_pose_reliable=head_pose_reliable,
        )
        gaze_ms = (time.perf_counter() - stage_started_at) * 1000.0

        if calibrating:
            phone_result = DetectionResult(
                detector=PHONE_SOURCE,
                state=CALIBRATING,
                suspicious=False,
                timestamp=timestamp,
                metadata={"skipped": "head_pose_calibration"},
            )
            phone_ms = 0.0
            phone_inference_executed = False
        else:
            stage_started_at = time.perf_counter()
            phone_result = self.phone_detector.detect(
                clean_frame,
                frame_context,
                timestamp,
            )
            phone_stage_ms = (time.perf_counter() - stage_started_at) * 1000.0
            phone_inference_executed = bool(
                phone_result.metadata.get("inference_executed", True)
            )
            # Keep this metric as actual YOLO latency. Skipped-frame wrapper cost is
            # already included in total_ms and should not dilute the inference mean.
            phone_ms = phone_stage_ms if phone_inference_executed else 0.0

        total_ms = (time.perf_counter() - total_started_at) * 1000.0
        return FrameProcessingResult(
            detections=(face_result, gaze_result, head_result, phone_result),
            face_context=frame_context,
            timing=ProcessingTiming(
                face_landmarks_ms=face_landmarks_ms,
                gaze_ms=gaze_ms,
                head_pose_ms=head_pose_ms,
                phone_ms=phone_ms,
                phone_inference_executed=phone_inference_executed,
                total_ms=total_ms,
            ),
        )
