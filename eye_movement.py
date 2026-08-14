"""Calibrated gaze estimation from MediaPipe iris and eye landmarks."""

import logging
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from face_context import FACE_DETECTED, FrameContext
from face_context import NO_FACE as CONTEXT_NO_FACE

GAZE_CENTER = "Looking Center"
GAZE_LEFT = "Looking Left"
GAZE_RIGHT = "Looking Right"
GAZE_UP = "Looking Up"
GAZE_DOWN = "Looking Down"
NO_FACE = "NO_FACE"
UNKNOWN = "UNKNOWN"
CALIBRATING = "CALIBRATING"

MIN_GAZE_CALIBRATION_SAMPLES = 15
GAZE_HISTORY_SIZE = 5
GAZE_HORIZONTAL_THRESHOLD = 0.10
GAZE_VERTICAL_THRESHOLD = 0.12
GAZE_HYSTERESIS_RATIO = 0.75
MAX_CALIBRATION_BINOCULAR_DISAGREEMENT = 0.25
MAX_BINOCULAR_DISPLACEMENT_DISAGREEMENT = 0.12
MIN_NORMALIZED_EYE_WIDTH = 0.02
MIN_NORMALIZED_EYE_APERTURE = 0.008
GAZE_MAX_HEAD_PITCH_DEGREES = 28.0
GAZE_MAX_HEAD_YAW_DEGREES = 28.0
GAZE_MAX_HEAD_ROLL_DEGREES = 25.0

# MediaPipe Face Landmarker outputs 478 points. Names are anatomical from the
# candidate's perspective; normalization uses image min/max so both eyes share
# the same left-to-right coordinate convention.
LEFT_EYE = {
    "corners": (362, 263),
    "upper": (385, 386, 387),
    "lower": (373, 374, 380),
    "iris": (473, 474, 475, 476, 477),
}
RIGHT_EYE = {
    "corners": (33, 133),
    "upper": (158, 159, 160),
    "lower": (144, 145, 153),
    "iris": (468, 469, 470, 471, 472),
}

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class GazeCalibrationBaseline:
    left: tuple[float, float]
    right: tuple[float, float]
    combined: tuple[float, float]
    sample_count: int
    inlier_count: int

    def as_metadata(self):
        return {
            "left": self.left,
            "right": self.right,
            "combined": self.combined,
            "sample_count": self.sample_count,
            "inlier_count": self.inlier_count,
            "aggregation": "coordinate_median_with_mad_outlier_filter",
        }


@dataclass(slots=True)
class GazeState:
    """Per-session neutral calibration, smoothing, and hysteresis state."""

    calibration_samples: list = field(default_factory=list)
    baseline: GazeCalibrationBaseline | None = None
    displacement_history: deque = field(
        default_factory=lambda: deque(maxlen=GAZE_HISTORY_SIZE)
    )
    last_classification: str = GAZE_CENTER

    def reset(self):
        self.calibration_samples.clear()
        self.baseline = None
        self.displacement_history.clear()
        self.last_classification = GAZE_CENTER

    def observe_calibration(self, left, right):
        self.calibration_samples.append(
            (
                np.asarray(left, dtype=np.float64),
                np.asarray(right, dtype=np.float64),
            )
        )

    def finalize_calibration(self, minimum_samples=MIN_GAZE_CALIBRATION_SAMPLES):
        if len(self.calibration_samples) < minimum_samples:
            raise RuntimeError(
                "Gaze calibration failed: "
                f"received {len(self.calibration_samples)} valid binocular samples, "
                f"but {minimum_samples} are required."
            )
        left = np.asarray([sample[0] for sample in self.calibration_samples])
        right = np.asarray([sample[1] for sample in self.calibration_samples])
        combined = (left + right) / 2.0
        center = np.median(combined, axis=0)
        distances = np.linalg.norm(combined - center, axis=1)
        distance_median = float(np.median(distances))
        mad = float(np.median(np.abs(distances - distance_median)))
        inlier_limit = distance_median + 3.0 * max(mad, 0.005)
        inliers = distances <= inlier_limit
        minimum_inliers = max(5, minimum_samples // 2)
        if int(np.count_nonzero(inliers)) < minimum_inliers:
            raise RuntimeError(
                "Gaze calibration failed: neutral samples were too unstable after "
                "robust outlier filtering. Keep the head and eyes centered and retry."
            )
        left_median = np.median(left[inliers], axis=0)
        right_median = np.median(right[inliers], axis=0)
        baseline = GazeCalibrationBaseline(
            left=tuple(float(value) for value in left_median),
            right=tuple(float(value) for value in right_median),
            combined=tuple(float(value) for value in (left_median + right_median) / 2.0),
            sample_count=len(self.calibration_samples),
            inlier_count=int(np.count_nonzero(inliers)),
        )
        self.baseline = baseline
        self.displacement_history.clear()
        self.last_classification = GAZE_CENTER
        return baseline

    def smooth_displacement(self, displacement):
        self.displacement_history.append(
            np.asarray(displacement, dtype=np.float64)
        )
        smoothed = np.median(np.asarray(self.displacement_history), axis=0)
        return tuple(float(value) for value in smoothed)


def _eye_measurement(normalized_landmarks, pixel_landmarks, indices):
    points = normalized_landmarks
    corner_points = points[list(indices["corners"]), :2]
    upper_points = points[list(indices["upper"]), :2]
    lower_points = points[list(indices["lower"]), :2]
    iris_points = points[list(indices["iris"]), :2]

    x_min = float(np.min(corner_points[:, 0]))
    x_max = float(np.max(corner_points[:, 0]))
    upper_y = float(np.median(upper_points[:, 1]))
    lower_y = float(np.median(lower_points[:, 1]))
    width = x_max - x_min
    aperture = lower_y - upper_y
    debug = {
        "eye_width": width,
        "eye_aperture": aperture,
        "reason": None,
    }
    if width < MIN_NORMALIZED_EYE_WIDTH:
        debug["reason"] = "eye_width_too_small"
        return None, None, None, debug
    if aperture < MIN_NORMALIZED_EYE_APERTURE:
        debug["reason"] = "eye_closed_or_aperture_too_small"
        return None, None, None, debug

    iris_center = np.median(iris_points, axis=0)
    normalized = (
        float((iris_center[0] - x_min) / width),
        float((iris_center[1] - upper_y) / aperture),
    )
    if not np.isfinite(normalized).all() or not (
        -0.25 <= normalized[0] <= 1.25 and -0.25 <= normalized[1] <= 1.25
    ):
        debug["reason"] = "iris_outside_reliable_eye_region"
        return None, None, None, debug

    all_indices = tuple(
        dict.fromkeys(
            indices["corners"]
            + indices["upper"]
            + indices["lower"]
            + indices["iris"]
        )
    )
    pixels = pixel_landmarks[list(all_indices)]
    eye_box = (
        int(np.min(pixels[:, 0])),
        int(np.min(pixels[:, 1])),
        int(np.max(pixels[:, 0]) - np.min(pixels[:, 0]) + 1),
        int(np.max(pixels[:, 1]) - np.min(pixels[:, 1]) + 1),
    )
    iris_pixels = pixel_landmarks[list(indices["iris"])]
    iris_center_pixel = tuple(
        int(round(value)) for value in np.median(iris_pixels, axis=0)
    )
    return normalized, eye_box, iris_center_pixel, debug


def _raw_binocular_measurement(context):
    if (
        context.normalized_landmarks is None
        or context.landmarks is None
        or len(context.normalized_landmarks) < 478
    ):
        return None, None, (), (), {}, {}
    left, left_box, left_iris, left_debug = _eye_measurement(
        context.normalized_landmarks,
        context.landmarks,
        LEFT_EYE,
    )
    right, right_box, right_iris, right_debug = _eye_measurement(
        context.normalized_landmarks,
        context.landmarks,
        RIGHT_EYE,
    )
    eye_boxes = tuple(box for box in (left_box, right_box) if box is not None)
    iris_centers = tuple(
        center for center in (left_iris, right_iris) if center is not None
    )
    return left, right, eye_boxes, iris_centers, left_debug, right_debug


def _strongly_nonfrontal(relative_angles):
    if relative_angles is None:
        return False
    pitch, yaw, roll = relative_angles
    return (
        abs(pitch) > GAZE_MAX_HEAD_PITCH_DEGREES
        or abs(yaw) > GAZE_MAX_HEAD_YAW_DEGREES
        or abs(roll) > GAZE_MAX_HEAD_ROLL_DEGREES
    )


def _inside_gaze_hysteresis(previous_state, x_delta, y_delta):
    x_exit = GAZE_HORIZONTAL_THRESHOLD * GAZE_HYSTERESIS_RATIO
    y_exit = GAZE_VERTICAL_THRESHOLD * GAZE_HYSTERESIS_RATIO
    if previous_state == GAZE_LEFT:
        return x_delta < -x_exit
    if previous_state == GAZE_RIGHT:
        return x_delta > x_exit
    if previous_state == GAZE_UP:
        return y_delta < -y_exit
    if previous_state == GAZE_DOWN:
        return y_delta > y_exit
    return False


def classify_gaze_displacement(displacement, previous_state=GAZE_CENTER):
    x_delta, y_delta = displacement
    if _inside_gaze_hysteresis(previous_state, x_delta, y_delta):
        return previous_state
    normalized_deviations = {
        "horizontal": abs(x_delta) / GAZE_HORIZONTAL_THRESHOLD,
        "vertical": abs(y_delta) / GAZE_VERTICAL_THRESHOLD,
    }
    dominant = max(normalized_deviations, key=normalized_deviations.get)
    if normalized_deviations[dominant] <= 1.0:
        return GAZE_CENTER
    if dominant == "horizontal":
        return GAZE_LEFT if x_delta < 0 else GAZE_RIGHT
    return GAZE_UP if y_delta < 0 else GAZE_DOWN


def analyze_eye_movement(
    context: FrameContext,
    state=None,
    calibrating=False,
    head_relative_angles=None,
    head_pose_reliable=True,
):
    """Infer calibrated gaze from the tracked primary candidate's iris points."""
    state = state if state is not None else GazeState()
    metadata = {
        "eye_boxes": (),
        "pupil_centers": (),
        "left_normalized": None,
        "right_normalized": None,
        "combined_normalized": None,
        "binocular_disagreement": None,
        "gaze_displacement": None,
        "smoothed_displacement": None,
        "one_eye_fallback": False,
        "reliable": False,
        "unreliable_reason": None,
        "calibration_sample_valid": False,
        "calibration_sample_count": len(state.calibration_samples),
        "calibration_baseline": (
            state.baseline.as_metadata() if state.baseline is not None else None
        ),
        "primary_face_box": context.primary_face_box,
        "gaze_backend": "mediapipe_iris",
    }
    if context.face_status == CONTEXT_NO_FACE:
        return NO_FACE, metadata
    if context.face_status != FACE_DETECTED:
        metadata["unreliable_reason"] = "primary_temporarily_missing"
        return UNKNOWN, metadata

    left, right, eye_boxes, iris_centers, left_debug, right_debug = (
        _raw_binocular_measurement(context)
    )
    metadata["eye_boxes"] = eye_boxes
    metadata["pupil_centers"] = iris_centers
    metadata["left_normalized"] = left
    metadata["right_normalized"] = right
    metadata["left_detection"] = left_debug
    metadata["right_detection"] = right_debug
    if left is not None and right is not None:
        metadata["combined_normalized"] = tuple(
            float(value) for value in (np.asarray(left) + np.asarray(right)) / 2.0
        )

    if calibrating:
        if not head_pose_reliable:
            metadata["unreliable_reason"] = "head_pose_unreliable_during_calibration"
            return CALIBRATING, metadata
        if left is None or right is None:
            metadata["unreliable_reason"] = "binocular_iris_required_for_calibration"
            return CALIBRATING, metadata
        raw_disagreement = tuple(
            float(value) for value in np.abs(np.asarray(left) - np.asarray(right))
        )
        metadata["binocular_disagreement"] = raw_disagreement
        if max(raw_disagreement) > MAX_CALIBRATION_BINOCULAR_DISAGREEMENT:
            metadata["unreliable_reason"] = "binocular_calibration_disagreement"
            return CALIBRATING, metadata
        state.observe_calibration(left, right)
        metadata["calibration_sample_valid"] = True
        metadata["calibration_sample_count"] = len(state.calibration_samples)
        metadata["reliable"] = True
        return CALIBRATING, metadata

    if state.baseline is None:
        metadata["unreliable_reason"] = "gaze_not_calibrated"
        return UNKNOWN, metadata
    if _strongly_nonfrontal(head_relative_angles):
        metadata["unreliable_reason"] = "head_pose_strongly_nonfrontal"
        return UNKNOWN, metadata

    left_delta = (
        np.asarray(left) - np.asarray(state.baseline.left)
        if left is not None
        else None
    )
    right_delta = (
        np.asarray(right) - np.asarray(state.baseline.right)
        if right is not None
        else None
    )
    if left_delta is not None and right_delta is not None:
        disagreement = np.abs(left_delta - right_delta)
        metadata["binocular_disagreement"] = tuple(
            float(value) for value in disagreement
        )
        if np.any(disagreement > MAX_BINOCULAR_DISPLACEMENT_DISAGREEMENT):
            metadata["unreliable_reason"] = "strong_binocular_disagreement"
            return UNKNOWN, metadata
        displacement = (left_delta + right_delta) / 2.0
    elif left_delta is not None:
        displacement = left_delta
        metadata["one_eye_fallback"] = True
    elif right_delta is not None:
        displacement = right_delta
        metadata["one_eye_fallback"] = True
    else:
        metadata["unreliable_reason"] = "iris_unavailable_for_both_eyes"
        return UNKNOWN, metadata

    metadata["gaze_displacement"] = tuple(float(value) for value in displacement)
    smoothed = state.smooth_displacement(displacement)
    metadata["smoothed_displacement"] = smoothed
    gaze_state = classify_gaze_displacement(
        smoothed,
        previous_state=state.last_classification,
    )
    state.last_classification = gaze_state
    metadata["reliable"] = True
    LOGGER.debug(
        "MediaPipe gaze left=%s right=%s displacement=%s smoothed=%s "
        "head_relative=%s fallback=%s state=%s",
        left,
        right,
        metadata["gaze_displacement"],
        smoothed,
        head_relative_angles,
        metadata["one_eye_fallback"],
        gaze_state,
    )
    return gaze_state, metadata
