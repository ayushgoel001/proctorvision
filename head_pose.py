"""Head pose from MediaPipe facial transformation matrices."""

import logging
import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from face_context import FACE_DETECTED, FrameContext
from face_context import NO_FACE as CONTEXT_NO_FACE

LOOKING_AT_SCREEN = "Looking at Screen"
LOOKING_LEFT = "Looking Left"
LOOKING_RIGHT = "Looking Right"
LOOKING_UP = "Looking Up"
LOOKING_DOWN = "Looking Down"
TILTED = "Tilted"
NO_FACE = "NO_FACE"
UNKNOWN = "UNKNOWN"
CALIBRATING = "CALIBRATING"

ANGLE_HISTORY_SIZE = 5
PITCH_THRESHOLD = 18.0
YAW_THRESHOLD = 18.0
ROLL_THRESHOLD = 18.0
HEAD_POSE_HYSTERESIS_RATIO = 0.80
MAX_POSE_JUMP_DEGREES = 45.0
POSE_JUMP_CONFIRMATION_DEGREES = 10.0

LOGGER = logging.getLogger(__name__)


def rotation_matrix_to_euler_degrees(rotation_matrix):
    """Convert one rotation representation to pitch/yaw/roll in degrees."""
    rotation_matrix = np.asarray(rotation_matrix, dtype=np.float64)
    if rotation_matrix.shape != (3, 3):
        raise ValueError("A 3x3 rotation matrix is required.")

    sy = math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        pitch = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        yaw = math.atan2(-rotation_matrix[2, 0], sy)
        roll = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        pitch = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        yaw = math.atan2(-rotation_matrix[2, 0], sy)
        roll = 0.0
    return tuple(float(value) for value in np.degrees((pitch, yaw, roll)))


def mean_rotation_matrix(rotation_matrices):
    """Average calibration rotations and project the result back onto SO(3)."""
    matrices = np.asarray(rotation_matrices, dtype=np.float64)
    if matrices.ndim != 3 or matrices.shape[1:] != (3, 3) or len(matrices) == 0:
        raise ValueError("At least one 3x3 calibration rotation is required.")

    left, _, right_transposed = np.linalg.svd(np.mean(matrices, axis=0))
    mean_rotation = left @ right_transposed
    if np.linalg.det(mean_rotation) < 0:
        left[:, -1] *= -1
        mean_rotation = left @ right_transposed
    return mean_rotation


def relative_rotation_matrix(current_rotation, calibrated_rotation):
    """Return current head orientation relative to the calibrated camera pose."""
    current = np.asarray(current_rotation, dtype=np.float64)
    calibrated = np.asarray(calibrated_rotation, dtype=np.float64)
    if current.shape != (3, 3) or calibrated.shape != (3, 3):
        raise ValueError("Current and calibrated rotations must both be 3x3 matrices.")
    return current @ calibrated.T


def head_pose_deltas(current_rotation, calibrated_rotation):
    """Return relative pitch/yaw/roll; no absolute Euler subtraction is used."""
    return rotation_matrix_to_euler_degrees(
        relative_rotation_matrix(current_rotation, calibrated_rotation)
    )


def rotation_distance_degrees(first_rotation, second_rotation):
    """Return the geodesic angular distance between two rotation matrices."""
    relative = relative_rotation_matrix(second_rotation, first_rotation)
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def rotation_from_transformation_matrix(transformation_matrix):
    """Extract a proper 3D rotation from MediaPipe's 4x4 face transform."""
    matrix = np.asarray(transformation_matrix, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        return None
    raw_rotation = matrix[:3, :3]
    left, _, right_transposed = np.linalg.svd(raw_rotation)
    rotation = left @ right_transposed
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right_transposed
    return rotation


@dataclass
class HeadPoseState:
    """Per-run smoothing and one-frame jump-rejection state."""

    rotation_history: deque = field(
        default_factory=lambda: deque(maxlen=ANGLE_HISTORY_SIZE)
    )
    last_rotation: np.ndarray | None = field(default=None, repr=False)
    last_translation: np.ndarray | None = field(default=None, repr=False)
    pending_rotation: np.ndarray | None = field(default=None, repr=False)
    last_classification: str = LOOKING_AT_SCREEN

    def reset(self):
        self.rotation_history.clear()
        self.last_rotation = None
        self.last_translation = None
        self.pending_rotation = None
        self.last_classification = LOOKING_AT_SCREEN

    def set_reference(self, rotation, translation=None):
        self.reset()
        self.last_rotation = np.asarray(rotation, dtype=np.float64).copy()
        if translation is not None:
            self.last_translation = np.asarray(translation, dtype=np.float64).copy()

    def observe_pose(self, rotation, translation=None):
        """Reject an isolated jump; accept a new pose after one confirming frame."""
        rotation = np.asarray(rotation, dtype=np.float64)
        if self.last_rotation is None:
            self.last_rotation = rotation.copy()
            return True, 0.0

        jump = rotation_distance_degrees(self.last_rotation, rotation)
        if jump <= MAX_POSE_JUMP_DEGREES:
            self.last_rotation = rotation.copy()
            self.pending_rotation = None
            return True, jump

        if self.pending_rotation is not None:
            confirmation_distance = rotation_distance_degrees(
                self.pending_rotation,
                rotation,
            )
            if confirmation_distance <= POSE_JUMP_CONFIRMATION_DEGREES:
                self.last_rotation = rotation.copy()
                self.pending_rotation = None
                self.rotation_history.clear()
                return True, jump

        self.pending_rotation = rotation.copy()
        return False, jump

    def observe_rotation(self, rotation):
        return self.observe_pose(rotation)

    def smooth_rotation(self, rotation):
        self.rotation_history.append(np.asarray(rotation, dtype=np.float64).copy())
        return mean_rotation_matrix(self.rotation_history)


def _state_still_inside_hysteresis(previous_state, pitch, yaw, roll):
    exit_pitch = PITCH_THRESHOLD * HEAD_POSE_HYSTERESIS_RATIO
    exit_yaw = YAW_THRESHOLD * HEAD_POSE_HYSTERESIS_RATIO
    exit_roll = ROLL_THRESHOLD * HEAD_POSE_HYSTERESIS_RATIO
    if previous_state == LOOKING_LEFT:
        return yaw > exit_yaw
    if previous_state == LOOKING_RIGHT:
        return yaw < -exit_yaw
    if previous_state == LOOKING_UP:
        # MediaPipe's camera-view face transform produces negative relative
        # X-axis pitch for physical head-up motion and positive for head-down.
        return pitch < -exit_pitch
    if previous_state == LOOKING_DOWN:
        return pitch > exit_pitch
    if previous_state == TILTED:
        return abs(roll) > exit_roll
    return False


def classify_head_pose(relative_angles, previous_state=LOOKING_AT_SCREEN):
    """Classify the dominant calibrated rotation with small exit hysteresis."""
    pitch_delta, yaw_delta, roll_delta = relative_angles
    if _state_still_inside_hysteresis(
        previous_state,
        pitch_delta,
        yaw_delta,
        roll_delta,
    ):
        return previous_state

    deviations = {
        "yaw": abs(yaw_delta) / YAW_THRESHOLD,
        "pitch": abs(pitch_delta) / PITCH_THRESHOLD,
        "roll": abs(roll_delta) / ROLL_THRESHOLD,
    }
    dominant_axis = max(deviations, key=deviations.get)
    if deviations[dominant_axis] <= 1.0:
        return LOOKING_AT_SCREEN
    if dominant_axis == "yaw":
        return LOOKING_LEFT if yaw_delta > 0 else LOOKING_RIGHT
    if dominant_axis == "pitch":
        return LOOKING_UP if pitch_delta < 0 else LOOKING_DOWN
    return TILTED


def get_head_pose_rotation(transformation_matrix, frame_shape=None):
    del frame_shape
    return rotation_from_transformation_matrix(transformation_matrix)


def get_head_pose_angles(transformation_matrix, frame_shape=None):
    rotation = get_head_pose_rotation(transformation_matrix, frame_shape)
    return None if rotation is None else rotation_matrix_to_euler_degrees(rotation)


def analyze_head_pose(
    context: FrameContext,
    calibrated_rotation=None,
    state=None,
):
    """Estimate primary pose from its MediaPipe face transformation matrix."""
    state = state if state is not None else HeadPoseState()
    metadata = {
        "current_rotation": None,
        "current_translation": None,
        "current_absolute_angles": None,
        "calibrated_absolute_angles": None,
        "relative_angles": None,
        "jump_degrees": None,
        "primary_face_box": context.primary_face_box,
        "pose_backend": "mediapipe_transformation_matrix",
    }

    if context.face_status == CONTEXT_NO_FACE:
        return NO_FACE, metadata
    if context.face_status != FACE_DETECTED:
        return UNKNOWN, metadata

    current_rotation = rotation_from_transformation_matrix(
        context.facial_transformation_matrix
    )
    if current_rotation is None:
        return UNKNOWN, metadata
    metadata["current_rotation"] = current_rotation
    metadata["current_absolute_angles"] = rotation_matrix_to_euler_degrees(
        current_rotation
    )

    if calibrated_rotation is None:
        accepted, jump_degrees = state.observe_pose(current_rotation)
        metadata["jump_degrees"] = jump_degrees
        if not accepted:
            LOGGER.debug(
                "Rejected implausible calibration pose jump of %.2f degrees",
                jump_degrees,
            )
            metadata["current_rotation"] = None
        return CALIBRATING, metadata

    if state.last_rotation is None:
        state.set_reference(calibrated_rotation)
    metadata["calibrated_absolute_angles"] = rotation_matrix_to_euler_degrees(
        calibrated_rotation
    )
    accepted, jump_degrees = state.observe_pose(current_rotation)
    metadata["jump_degrees"] = jump_degrees
    if not accepted:
        LOGGER.debug(
            "Rejected implausible MediaPipe head-pose jump of %.2f degrees",
            jump_degrees,
        )
        return UNKNOWN, metadata

    smoothed_rotation = state.smooth_rotation(current_rotation)
    relative_angles = head_pose_deltas(smoothed_rotation, calibrated_rotation)
    metadata["relative_angles"] = relative_angles
    classification = classify_head_pose(
        relative_angles,
        previous_state=state.last_classification,
    )
    state.last_classification = classification
    return classification, metadata
