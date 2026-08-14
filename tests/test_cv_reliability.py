import math
import unittest

import numpy as np

from eye_movement import (
    GAZE_CENTER,
    GAZE_DOWN,
    GAZE_LEFT,
    GAZE_RIGHT,
    GAZE_UP,
    LEFT_EYE,
    RIGHT_EYE,
    GazeState,
    analyze_eye_movement,
    classify_gaze_displacement,
)
from eye_movement import (
    UNKNOWN as GAZE_UNKNOWN,
)
from face_context import (
    FACE_DETECTED,
    MULTIPLE_FACES,
    NO_FACE,
    NO_FACES,
    PRIMARY_PRESENT,
    PRIMARY_TEMPORARILY_MISSING,
    FaceLandmarkBatch,
    FrameContext,
    PrimaryFaceTracker,
    build_frame_context,
)
from head_pose import (
    LOOKING_AT_SCREEN,
    LOOKING_DOWN,
    LOOKING_LEFT,
    LOOKING_RIGHT,
    LOOKING_UP,
    TILTED,
    HeadPoseState,
    classify_head_pose,
    head_pose_deltas,
    rotation_from_transformation_matrix,
)


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class FakeFaceProvider:
    backend_name = "fake"

    def __init__(self, batches):
        self.batches = list(batches)

    def detect(self, clean_frame):
        del clean_frame
        return self.batches.pop(0)


def rotation_matrix(pitch=0.0, yaw=0.0, roll=0.0):
    pitch, yaw, roll = np.radians((pitch, yaw, roll))
    rotate_x = np.array(
        [[1, 0, 0], [0, math.cos(pitch), -math.sin(pitch)], [0, math.sin(pitch), math.cos(pitch)]]
    )
    rotate_y = np.array(
        [[math.cos(yaw), 0, math.sin(yaw)], [0, 1, 0], [-math.sin(yaw), 0, math.cos(yaw)]]
    )
    rotate_z = np.array(
        [[math.cos(roll), -math.sin(roll), 0], [math.sin(roll), math.cos(roll), 0], [0, 0, 1]]
    )
    return rotate_z @ rotate_y @ rotate_x


def landmark_batch(*boxes):
    normalized = tuple(np.zeros((478, 3), dtype=np.float64) for _ in boxes)
    pixels = tuple(np.zeros((478, 2), dtype=np.int32) for _ in boxes)
    matrices = tuple(np.eye(4, dtype=np.float64) for _ in boxes)
    return FaceLandmarkBatch(tuple(boxes), pixels, normalized, matrices)


def gaze_context(left_ratio=(0.5, 0.5), right_ratio=(0.5, 0.5), close_right=False):
    normalized = np.zeros((478, 3), dtype=np.float64)

    def place_eye(indices, x_min, x_max, y_top, y_bottom, ratio, close=False):
        for index, x_value in zip(indices["corners"], (x_min, x_max)):
            normalized[index, :2] = (x_value, (y_top + y_bottom) / 2.0)
        effective_bottom = y_top if close else y_bottom
        for index in indices["upper"]:
            normalized[index, :2] = ((x_min + x_max) / 2.0, y_top)
        for index in indices["lower"]:
            normalized[index, :2] = ((x_min + x_max) / 2.0, effective_bottom)
        iris_position = (
            x_min + ratio[0] * (x_max - x_min),
            y_top + ratio[1] * (effective_bottom - y_top),
        )
        for index in indices["iris"]:
            normalized[index, :2] = iris_position

    place_eye(LEFT_EYE, 0.55, 0.75, 0.40, 0.44, left_ratio)
    place_eye(RIGHT_EYE, 0.25, 0.45, 0.40, 0.44, right_ratio, close_right)
    pixels = np.column_stack(
        (
            np.rint(normalized[:, 0] * 640),
            np.rint(normalized[:, 1] * 480),
        )
    ).astype(np.int32)
    return FrameContext(
        clean_frame=np.zeros((480, 640, 3), dtype=np.uint8),
        grayscale_frame=np.zeros((480, 640), dtype=np.uint8),
        face_boxes=((150, 100, 480, 430),),
        primary_face_box=(150, 100, 480, 430),
        landmarks=pixels,
        normalized_landmarks=normalized,
        facial_transformation_matrix=np.eye(4),
        face_count=1,
        face_status=FACE_DETECTED,
        face_observation=PRIMARY_PRESENT,
        association_status="PRIMARY_ASSOCIATED",
        additional_faces_present=False,
        primary_missing_seconds=0.0,
    )


def calibrated_gaze_state():
    state = GazeState()
    for index in range(20):
        jitter = (index % 3 - 1) * 0.001
        state.observe_calibration(
            (0.5 + jitter, 0.5 - jitter),
            (0.5 - jitter, 0.5 + jitter),
        )
    state.finalize_calibration(15)
    return state


class FacePresenceTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.clock = FakeClock()
        self.tracker = PrimaryFaceTracker(
            missing_grace_seconds=0.75,
            clock=self.clock,
        )

    def test_temporary_loss_transitions_to_no_face_after_grace_and_reassociates(self):
        primary = (100, 100, 220, 260)
        provider = FakeFaceProvider(
            (
                landmark_batch(primary),
                landmark_batch(),
                landmark_batch(),
                landmark_batch((104, 102, 224, 262)),
            )
        )
        present = build_frame_context(self.frame, self.tracker, provider)
        self.clock.now = 0.1
        temporary = build_frame_context(self.frame, self.tracker, provider)
        self.clock.now = 0.9
        missing = build_frame_context(self.frame, self.tracker, provider)
        self.clock.now = 1.0
        returned = build_frame_context(self.frame, self.tracker, provider)

        self.assertEqual(present.face_observation, PRIMARY_PRESENT)
        self.assertEqual(temporary.face_observation, PRIMARY_TEMPORARILY_MISSING)
        self.assertNotEqual(temporary.face_status, NO_FACE)
        self.assertEqual(missing.face_observation, NO_FACES)
        self.assertEqual(missing.face_status, NO_FACE)
        self.assertEqual(returned.face_observation, PRIMARY_PRESENT)
        self.assertEqual(returned.primary_face_box, (104, 102, 224, 262))

    def test_initial_empty_frame_is_real_no_face(self):
        context = build_frame_context(
            self.frame,
            self.tracker,
            FakeFaceProvider((landmark_batch(),)),
        )
        self.assertEqual(context.face_observation, NO_FACES)
        self.assertEqual(context.face_status, NO_FACE)

    def test_second_larger_face_does_not_replace_primary(self):
        primary = (80, 100, 200, 260)
        associated = (84, 102, 204, 262)
        other = (390, 60, 620, 400)
        provider = FakeFaceProvider(
            (
                landmark_batch(primary),
                landmark_batch(associated, other),
                landmark_batch(other),
                landmark_batch((88, 104, 208, 264)),
            )
        )
        build_frame_context(self.frame, self.tracker, provider)
        multiple = build_frame_context(self.frame, self.tracker, provider)
        self.clock.now = 0.1
        refused_switch = build_frame_context(self.frame, self.tracker, provider)
        self.clock.now = 0.2
        returned = build_frame_context(self.frame, self.tracker, provider)

        self.assertEqual(multiple.face_observation, MULTIPLE_FACES)
        self.assertEqual(multiple.primary_face_box, associated)
        self.assertEqual(
            refused_switch.face_observation,
            PRIMARY_TEMPORARILY_MISSING,
        )
        self.assertIsNone(refused_switch.primary_face_box)
        self.assertEqual(returned.primary_face_box, (88, 104, 208, 264))


class GazeReliabilityTests(unittest.TestCase):
    def test_robust_calibration_uses_median_and_resets(self):
        state = calibrated_gaze_state()
        self.assertAlmostEqual(state.baseline.combined[0], 0.5, places=3)
        self.assertAlmostEqual(state.baseline.combined[1], 0.5, places=3)
        state.reset()
        self.assertIsNone(state.baseline)
        self.assertEqual(state.calibration_samples, [])

    def test_directional_gaze_with_frontal_head(self):
        cases = (
            ((0.35, 0.5), GAZE_LEFT),
            ((0.65, 0.5), GAZE_RIGHT),
            ((0.5, 0.30), GAZE_UP),
            ((0.5, 0.70), GAZE_DOWN),
            ((0.53, 0.53), GAZE_CENTER),
        )
        for ratio, expected in cases:
            with self.subTest(expected=expected):
                state = calibrated_gaze_state()
                observed, metadata = analyze_eye_movement(
                    gaze_context(ratio, ratio),
                    state=state,
                    head_relative_angles=(0.0, 0.0, 0.0),
                )
                self.assertEqual(observed, expected)
                self.assertTrue(metadata["reliable"])

    def test_one_eye_fallback_and_strong_disagreement(self):
        fallback_state = calibrated_gaze_state()
        fallback, metadata = analyze_eye_movement(
            gaze_context((0.35, 0.5), (0.5, 0.5), close_right=True),
            state=fallback_state,
            head_relative_angles=(0.0, 0.0, 0.0),
        )
        self.assertEqual(fallback, GAZE_LEFT)
        self.assertTrue(metadata["one_eye_fallback"])

        disagree_state = calibrated_gaze_state()
        disagree, metadata = analyze_eye_movement(
            gaze_context((0.30, 0.5), (0.70, 0.5)),
            state=disagree_state,
            head_relative_angles=(0.0, 0.0, 0.0),
        )
        self.assertEqual(disagree, GAZE_UNKNOWN)
        self.assertEqual(metadata["unreliable_reason"], "strong_binocular_disagreement")

    def test_strong_head_turn_makes_gaze_unknown_but_moderate_turn_does_not(self):
        moderate, _ = analyze_eye_movement(
            gaze_context((0.35, 0.5), (0.35, 0.5)),
            state=calibrated_gaze_state(),
            head_relative_angles=(0.0, 20.0, 0.0),
        )
        strong, metadata = analyze_eye_movement(
            gaze_context((0.35, 0.5), (0.35, 0.5)),
            state=calibrated_gaze_state(),
            head_relative_angles=(0.0, 30.0, 0.0),
        )
        self.assertEqual(moderate, GAZE_LEFT)
        self.assertEqual(strong, GAZE_UNKNOWN)
        self.assertEqual(metadata["unreliable_reason"], "head_pose_strongly_nonfrontal")

    def test_gaze_hysteresis_avoids_threshold_flicker(self):
        self.assertEqual(
            classify_gaze_displacement((-0.08, 0.0), previous_state=GAZE_LEFT),
            GAZE_LEFT,
        )
        self.assertEqual(
            classify_gaze_displacement((-0.05, 0.0), previous_state=GAZE_LEFT),
            GAZE_CENTER,
        )


class HeadPoseReliabilityTests(unittest.TestCase):
    def test_relative_rotation_is_derived_from_rotation_representations(self):
        calibrated = rotation_matrix(pitch=8.0, yaw=-6.0, roll=3.0)
        desired_relative = rotation_matrix(pitch=12.0, yaw=22.0, roll=-5.0)
        current = desired_relative @ calibrated
        pitch, yaw, roll = head_pose_deltas(current, calibrated)
        self.assertAlmostEqual(pitch, 12.0, places=5)
        self.assertAlmostEqual(yaw, 22.0, places=5)
        self.assertAlmostEqual(roll, -5.0, places=5)

    def test_head_direction_classifications(self):
        self.assertEqual(classify_head_pose((0.0, 0.0, 0.0)), LOOKING_AT_SCREEN)
        self.assertEqual(classify_head_pose((0.0, 25.0, 0.0)), LOOKING_LEFT)
        self.assertEqual(classify_head_pose((0.0, -25.0, 0.0)), LOOKING_RIGHT)
        # MediaPipe camera-view transforms produce negative relative pitch for
        # physical head-up motion and positive pitch for physical head-down.
        self.assertEqual(classify_head_pose((-25.0, 0.0, 0.0)), LOOKING_UP)
        self.assertEqual(classify_head_pose((25.0, 0.0, 0.0)), LOOKING_DOWN)
        self.assertEqual(classify_head_pose((0.0, 0.0, 25.0)), TILTED)

    def test_media_pipe_pitch_mapping_preserves_yaw_roll_and_hysteresis(self):
        self.assertEqual(
            classify_head_pose((-15.0, 0.0, 0.0), previous_state=LOOKING_UP),
            LOOKING_UP,
        )
        self.assertEqual(
            classify_head_pose((15.0, 0.0, 0.0), previous_state=LOOKING_DOWN),
            LOOKING_DOWN,
        )
        self.assertEqual(classify_head_pose((0.0, 25.0, 0.0)), LOOKING_LEFT)
        self.assertEqual(classify_head_pose((0.0, -25.0, 0.0)), LOOKING_RIGHT)
        self.assertEqual(classify_head_pose((0.0, 0.0, -25.0)), TILTED)

    def test_transformation_matrix_rotation_projection_and_jump_rejection(self):
        expected = rotation_matrix(yaw=20.0)
        transform = np.eye(4)
        transform[:3, :3] = expected * 1.25
        extracted = rotation_from_transformation_matrix(transform)
        self.assertTrue(np.allclose(extracted, expected, atol=1e-8))

        state = HeadPoseState()
        accepted, _ = state.observe_rotation(np.eye(3))
        rejected, _ = state.observe_rotation(rotation_matrix(yaw=80.0))
        confirmed, _ = state.observe_rotation(rotation_matrix(yaw=81.0))
        self.assertTrue(accepted)
        self.assertFalse(rejected)
        self.assertTrue(confirmed)


if __name__ == "__main__":
    unittest.main()
