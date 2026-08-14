"""Shared MediaPipe face landmarks and primary-candidate association."""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from config import FACE_LANDMARKER_MODEL_PATH

MIN_FACE_LANDMARKER_MODEL_BYTES = 1_000_000
MEDIAPIPE_MAX_FACES = 2
MEDIAPIPE_MIN_FACE_DETECTION_CONFIDENCE = 0.5
MEDIAPIPE_MIN_FACE_PRESENCE_CONFIDENCE = 0.5
MEDIAPIPE_MIN_TRACKING_CONFIDENCE = 0.5

FACE_DETECTED = "FACE_DETECTED"
NO_FACE = "NO_FACE"
UNKNOWN = "UNKNOWN"

PRIMARY_PRESENT = "PRIMARY_PRESENT"
PRIMARY_TEMPORARILY_MISSING = "PRIMARY_TEMPORARILY_MISSING"
PRIMARY_MISSING = "PRIMARY_MISSING"
MULTIPLE_FACES = "MULTIPLE_FACES"
NO_FACES = "NO_FACES"

PRIMARY_ACQUIRED = "PRIMARY_ACQUIRED"
PRIMARY_ASSOCIATED = "PRIMARY_ASSOCIATED"
TRACKING_UNINITIALIZED = "TRACKING_UNINITIALIZED"

MIN_ASSOCIATION_IOU = 0.20
MAX_CENTER_DISTANCE_RATIO = 0.75
MIN_AREA_SIMILARITY = 0.50
PRIMARY_MISSING_GRACE_SECONDS = 0.75
FACE_BOX_MARGIN_RATIO = 0.08

LOGGER = logging.getLogger(__name__)


def _load_mediapipe():
    try:
        import mediapipe
    except ImportError as exc:
        raise RuntimeError(
            "MediaPipe is required for live face-landmark inference. "
            "Install the runtime dependencies from requirements.txt."
        ) from exc
    return mediapipe


@dataclass(frozen=True, slots=True)
class FrameContext:
    """Primary-candidate data derived once from one untouched camera frame."""

    clean_frame: np.ndarray
    grayscale_frame: np.ndarray
    face_boxes: tuple[tuple[int, int, int, int], ...]
    primary_face_box: tuple[int, int, int, int] | None
    landmarks: np.ndarray | None
    normalized_landmarks: np.ndarray | None
    facial_transformation_matrix: np.ndarray | None
    face_count: int
    face_status: str
    face_observation: str
    association_status: str
    additional_faces_present: bool
    primary_missing_seconds: float


@dataclass(frozen=True, slots=True)
class FaceLandmarkBatch:
    face_boxes: tuple[tuple[int, int, int, int], ...]
    pixel_landmarks: tuple[np.ndarray, ...]
    normalized_landmarks: tuple[np.ndarray, ...]
    transformation_matrices: tuple[np.ndarray | None, ...]


def _box_area(box):
    left, top, right, bottom = box
    return max(0, right - left) * max(0, bottom - top)


def _box_iou(first, second):
    intersection_left = max(first[0], second[0])
    intersection_top = max(first[1], second[1])
    intersection_right = min(first[2], second[2])
    intersection_bottom = min(first[3], second[3])
    intersection = _box_area(
        (
            intersection_left,
            intersection_top,
            intersection_right,
            intersection_bottom,
        )
    )
    union = _box_area(first) + _box_area(second) - intersection
    return intersection / union if union > 0 else 0.0


def _center_distance_ratio(reference, candidate):
    reference_center = (
        (reference[0] + reference[2]) / 2.0,
        (reference[1] + reference[3]) / 2.0,
    )
    candidate_center = (
        (candidate[0] + candidate[2]) / 2.0,
        (candidate[1] + candidate[3]) / 2.0,
    )
    reference_width = max(1.0, float(reference[2] - reference[0]))
    reference_height = max(1.0, float(reference[3] - reference[1]))
    reference_diagonal = float(np.hypot(reference_width, reference_height))
    return float(
        np.hypot(
            candidate_center[0] - reference_center[0],
            candidate_center[1] - reference_center[1],
        )
        / reference_diagonal
    )


def _area_similarity(first, second):
    first_area = _box_area(first)
    second_area = _box_area(second)
    larger_area = max(first_area, second_area)
    return min(first_area, second_area) / larger_area if larger_area > 0 else 0.0


@dataclass(slots=True)
class PrimaryFaceTracker:
    """Associate one candidate across frames using only explainable box geometry."""

    primary_box: tuple[int, int, int, int] | None = None
    missed_frames: int = 0
    missing_grace_seconds: float = PRIMARY_MISSING_GRACE_SECONDS
    clock: Callable[[], float] = field(default=time.monotonic, repr=False)
    missing_since: float | None = field(default=None, init=False)

    def __post_init__(self):
        if self.missing_grace_seconds < 0:
            raise ValueError("Primary-face missing grace cannot be negative.")

    @property
    def missing_seconds(self):
        if self.missing_since is None:
            return 0.0
        return max(0.0, float(self.clock()) - self.missing_since)

    def reset(self):
        self.primary_box = None
        self.missed_frames = 0
        self.missing_since = None

    def _record_missing(self, visible_face_count):
        if self.missing_since is None:
            self.missing_since = float(self.clock())
            LOGGER.info(
                "Primary candidate temporarily missing; retaining box=%s",
                self.primary_box,
            )
        self.missed_frames += 1
        if self.missing_seconds < self.missing_grace_seconds:
            return PRIMARY_TEMPORARILY_MISSING
        LOGGER.debug(
            "Primary candidate missing beyond %.2fs grace; other_face_count=%d",
            self.missing_grace_seconds,
            visible_face_count,
        )
        return PRIMARY_MISSING

    def associate(self, face_boxes):
        """Return the associated box index and an explicit association status."""
        face_boxes = tuple(face_boxes)
        if not face_boxes:
            if self.primary_box is not None:
                return None, self._record_missing(0)
            return None, TRACKING_UNINITIALIZED

        if self.primary_box is None:
            primary_index = max(
                range(len(face_boxes)),
                key=lambda index: _box_area(face_boxes[index]),
            )
            self.primary_box = face_boxes[primary_index]
            self.missed_frames = 0
            self.missing_since = None
            LOGGER.info("Primary candidate acquired with box=%s", self.primary_box)
            return primary_index, PRIMARY_ACQUIRED

        candidates = []
        for index, face_box in enumerate(face_boxes):
            iou = _box_iou(self.primary_box, face_box)
            center_ratio = _center_distance_ratio(self.primary_box, face_box)
            area_similarity = _area_similarity(self.primary_box, face_box)
            accepted = iou >= MIN_ASSOCIATION_IOU or (
                center_ratio <= MAX_CENTER_DISTANCE_RATIO
                and area_similarity >= MIN_AREA_SIMILARITY
            )
            LOGGER.debug(
                "Primary association candidate index=%d box=%s iou=%.3f "
                "center_ratio=%.3f area_similarity=%.3f accepted=%s",
                index,
                face_box,
                iou,
                center_ratio,
                area_similarity,
                accepted,
            )
            if accepted:
                score = (
                    2.0 * iou
                    + max(0.0, 1.0 - center_ratio)
                    + 0.25 * area_similarity
                )
                candidates.append((score, -index, index))

        if not candidates:
            if self.missing_since is None:
                LOGGER.info(
                    "Primary candidate not associated; refusing to switch to %d other face(s)",
                    len(face_boxes),
                )
            return None, self._record_missing(len(face_boxes))

        _, _, primary_index = max(candidates)
        returning_after_loss = self.missed_frames > 0
        self.primary_box = face_boxes[primary_index]
        self.missed_frames = 0
        self.missing_since = None
        if returning_after_loss:
            LOGGER.info("Primary candidate reassociated with box=%s", self.primary_box)
        return primary_index, PRIMARY_ASSOCIATED


class MediaPipeFaceLandmarkProvider:
    """Synchronous VIDEO-mode MediaPipe Face Landmarker shared by one engine."""

    backend_name = "mediapipe_face_landmarker"

    def __init__(
        self,
        model_path=FACE_LANDMARKER_MODEL_PATH,
        max_faces=MEDIAPIPE_MAX_FACES,
        clock=time.monotonic,
    ):
        self.model_path = Path(model_path).resolve()
        self.max_faces = int(max_faces)
        self.clock = clock
        self._mediapipe = None
        self._landmarker = None
        self._last_timestamp_ms = -1
        self._validate_configuration()
        self._create_landmarker()

    def _validate_configuration(self):
        if self.max_faces < 2:
            raise ValueError(
                "Face Landmarker must allow at least two faces for MULTIPLE_FACES."
            )
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"MediaPipe Face Landmarker model not found: {self.model_path}. "
                "Place face_landmarker.task in the project's model directory."
            )
        if self.model_path.stat().st_size < MIN_FACE_LANDMARKER_MODEL_BYTES:
            raise RuntimeError(
                f"MediaPipe Face Landmarker model is too small or corrupt: "
                f"{self.model_path}."
            )

    def _create_landmarker(self):
        mp = _load_mediapipe()
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(self.model_path),
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=self.max_faces,
            min_face_detection_confidence=MEDIAPIPE_MIN_FACE_DETECTION_CONFIDENCE,
            min_face_presence_confidence=MEDIAPIPE_MIN_FACE_PRESENCE_CONFIDENCE,
            min_tracking_confidence=MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,
        )
        try:
            self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(
                options
            )
        except (RuntimeError, ValueError) as exc:
            raise RuntimeError(
                f"Failed to load MediaPipe Face Landmarker model at "
                f"{self.model_path}."
            ) from exc
        self._mediapipe = mp
        self._last_timestamp_ms = -1
        LOGGER.info(
            "Loaded MediaPipe Face Landmarker model=%s mode=VIDEO max_faces=%d "
            "transform_matrices=true",
            self.model_path,
            self.max_faces,
        )

    def _next_timestamp_ms(self):
        candidate = int(float(self.clock()) * 1000.0)
        self._last_timestamp_ms = max(candidate, self._last_timestamp_ms + 1)
        return self._last_timestamp_ms

    def detect(self, clean_frame):
        if self._landmarker is None or self._mediapipe is None:
            raise RuntimeError("MediaPipe Face Landmarker is closed.")
        mp = self._mediapipe
        rgb_frame = cv2.cvtColor(clean_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        try:
            result = self._landmarker.detect_for_video(
                mp_image,
                self._next_timestamp_ms(),
            )
        except (RuntimeError, ValueError) as exc:
            raise RuntimeError(
                "MediaPipe Face Landmarker inference failed for the current frame."
            ) from exc
        return _convert_landmarker_result(result, clean_frame.shape)

    def reset(self):
        self.close()
        self._create_landmarker()

    def close(self):
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
        self._mediapipe = None


def _landmarks_to_box(normalized_landmarks, frame_shape):
    height, width = frame_shape[:2]
    x_values = normalized_landmarks[:, 0]
    y_values = normalized_landmarks[:, 1]
    left = float(np.min(x_values))
    right = float(np.max(x_values))
    top = float(np.min(y_values))
    bottom = float(np.max(y_values))
    margin_x = (right - left) * FACE_BOX_MARGIN_RATIO
    margin_y = (bottom - top) * FACE_BOX_MARGIN_RATIO
    return (
        int(np.clip(round((left - margin_x) * width), 0, width - 1)),
        int(np.clip(round((top - margin_y) * height), 0, height - 1)),
        int(np.clip(round((right + margin_x) * width), 0, width - 1)),
        int(np.clip(round((bottom + margin_y) * height), 0, height - 1)),
    )


def _convert_landmarker_result(result, frame_shape):
    height, width = frame_shape[:2]
    normalized_faces = []
    pixel_faces = []
    face_boxes = []
    matrices = []
    result_matrices = tuple(result.facial_transformation_matrixes or ())

    for index, face_landmarks in enumerate(result.face_landmarks or ()):
        normalized = np.asarray(
            [(point.x, point.y, point.z) for point in face_landmarks],
            dtype=np.float64,
        )
        pixel = np.column_stack(
            (
                np.clip(np.rint(normalized[:, 0] * width), 0, width - 1),
                np.clip(np.rint(normalized[:, 1] * height), 0, height - 1),
            )
        ).astype(np.int32)
        normalized_faces.append(normalized)
        pixel_faces.append(pixel)
        face_boxes.append(_landmarks_to_box(normalized, frame_shape))
        matrix = (
            np.asarray(result_matrices[index], dtype=np.float64).copy()
            if index < len(result_matrices)
            else None
        )
        matrices.append(matrix)

    return FaceLandmarkBatch(
        face_boxes=tuple(face_boxes),
        pixel_landmarks=tuple(pixel_faces),
        normalized_landmarks=tuple(normalized_faces),
        transformation_matrices=tuple(matrices),
    )


_default_provider = None


def _get_default_provider():
    global _default_provider
    if _default_provider is None:
        _default_provider = MediaPipeFaceLandmarkProvider()
    return _default_provider


def _face_observation(face_count, primary_face_box, association_status):
    if face_count > 1:
        return MULTIPLE_FACES
    if association_status == PRIMARY_TEMPORARILY_MISSING:
        return PRIMARY_TEMPORARILY_MISSING
    if primary_face_box is None:
        if association_status == PRIMARY_MISSING:
            return NO_FACES if face_count == 0 else PRIMARY_MISSING
        return NO_FACES if face_count == 0 else PRIMARY_MISSING
    return PRIMARY_PRESENT


def build_frame_context(clean_frame, tracker=None, provider=None):
    """Build one tracked face/landmark context before gaze and head-pose logic."""
    if clean_frame is None or not isinstance(clean_frame, np.ndarray):
        raise ValueError("A clean NumPy frame is required.")
    if clean_frame.size == 0:
        raise ValueError("A non-empty clean frame is required.")

    tracker = tracker if tracker is not None else PrimaryFaceTracker()
    provider = provider if provider is not None else _get_default_provider()
    grayscale_frame = cv2.cvtColor(clean_frame, cv2.COLOR_BGR2GRAY)
    batch = provider.detect(clean_frame)
    face_boxes = batch.face_boxes
    primary_index, association_status = tracker.associate(face_boxes)
    primary_face_box = (
        face_boxes[primary_index] if primary_index is not None else None
    )
    observation = _face_observation(
        len(face_boxes),
        primary_face_box,
        association_status,
    )
    temporary_missing = association_status == PRIMARY_TEMPORARILY_MISSING
    face_status = (
        FACE_DETECTED
        if primary_index is not None
        else UNKNOWN
        if temporary_missing
        else NO_FACE
    )
    return FrameContext(
        clean_frame=clean_frame,
        grayscale_frame=grayscale_frame,
        face_boxes=face_boxes,
        primary_face_box=primary_face_box,
        landmarks=(
            batch.pixel_landmarks[primary_index]
            if primary_index is not None
            else None
        ),
        normalized_landmarks=(
            batch.normalized_landmarks[primary_index]
            if primary_index is not None
            else None
        ),
        facial_transformation_matrix=(
            batch.transformation_matrices[primary_index]
            if primary_index is not None
            else None
        ),
        face_count=len(face_boxes),
        face_status=face_status,
        face_observation=observation,
        association_status=association_status,
        additional_faces_present=(
            len(face_boxes) > 1 if primary_index is not None else len(face_boxes) > 0
        ),
        primary_missing_seconds=tracker.missing_seconds,
    )
