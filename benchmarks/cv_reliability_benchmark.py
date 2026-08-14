"""Compare face/gaze/head reliability on the repository's saved frame sequence.

The screenshots are operational regression fixtures, not a controlled accuracy
dataset. Synthetic rule tests cover directional classifications separately.
"""

# ruff: noqa: E402 -- benchmark scripts add the repository root before imports.

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detectors import (  # noqa: E402
    NO_PHONE,
    PHONE_SOURCE,
    DetectionResult,
)
from eye_movement import NO_FACE as GAZE_NO_FACE  # noqa: E402
from eye_movement import UNKNOWN as GAZE_UNKNOWN
from head_pose import (  # noqa: E402
    CALIBRATING as HEAD_CALIBRATING,
)
from head_pose import (
    NO_FACE as HEAD_NO_FACE,
)
from head_pose import (
    UNKNOWN as HEAD_UNKNOWN,
)
from head_pose import (
    mean_rotation_matrix,
)
from surveillance_engine import SurveillanceEngine  # noqa: E402


class NoopPhoneDetector:
    detector = PHONE_SOURCE

    def detect(self, clean_frame, frame_context, timestamp):
        del clean_frame, frame_context
        return DetectionResult(
            detector=PHONE_SOURCE,
            state=NO_PHONE,
            suspicious=False,
            timestamp=timestamp,
            metadata={
                "predictions": (),
                "accepted_detections": (),
                "inference_executed": False,
                "fresh": False,
            },
        )

    def reset(self):
        return None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=PROJECT_ROOT / "data" / "evidence",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--calibration-frames", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_fixtures(directory):
    fixtures = []
    paths = sorted(
        path
        for pattern in ("*.png", "*.jpg", "*.jpeg")
        for path in directory.rglob(pattern)
    )
    for path in paths:
        frame = cv2.imread(str(path))
        if frame is None:
            raise RuntimeError(f"Could not read fixture: {path}")
        relative_name = path.relative_to(directory).as_posix()
        fixtures.append(
            {
                "name": relative_name,
                "frame": frame,
                # Every saved review-event screenshot currently contains a visible
                # primary candidate, confirmed by manual visual inspection.
                "expected_face_visible": directory.name == "evidence",
            }
        )
    if not fixtures:
        raise RuntimeError(f"No image fixtures found in {directory}")
    return fixtures


def calibrate(engine, fixtures, calibration_frames):
    rotation_samples = []
    reference_name = None
    for fixture in fixtures:
        engine.reset_session()
        for _ in range(calibration_frames):
            result = engine.process_frame(fixture["frame"], calibrating=True)
            rotation = result.detection("head_pose").metadata.get("current_rotation")
            if rotation is not None:
                rotation_samples.append(rotation)
        if len(rotation_samples) >= 15:
            reference_name = fixture["name"]
            break
        rotation_samples.clear()

    if len(rotation_samples) < 15:
        raise RuntimeError("No fixture produced enough valid head calibration samples.")

    engine.set_head_calibration(mean_rotation_matrix(rotation_samples))
    finalize_gaze = getattr(engine, "finalize_gaze_calibration", None)
    if finalize_gaze is not None:
        finalize_gaze(minimum_samples=15)
    return reference_name, len(rotation_samples)


def stable_fraction(values):
    if not values:
        return 0.0
    return Counter(values).most_common(1)[0][1] / len(values)


def main():
    args = parse_args()
    fixtures = load_fixtures(args.fixtures)
    engine = SurveillanceEngine(phone_detector=NoopPhoneDetector())
    reference_name, calibration_samples = calibrate(
        engine,
        fixtures,
        args.calibration_frames,
    )

    timings = defaultdict(list)
    observations = defaultdict(
        lambda: {
            "face": [],
            "gaze": [],
            "head": [],
            "hard_no_face": [],
            "head_relative": [],
            "gaze_displacement": [],
        }
    )
    for _ in range(args.repeats):
        engine.face_tracker.reset()
        previous_session = None
        for fixture in fixtures:
            fixture_session = fixture["name"].split("/", 1)[0]
            if previous_session is not None and fixture_session != previous_session:
                # Evidence directories are separate real monitoring sessions; the
                # primary-candidate tracker must not carry identity geometry across them.
                engine.face_tracker.reset()
            previous_session = fixture_session
            result = engine.process_frame(fixture["frame"], calibrating=False)
            context = result.face_context
            gaze = result.detection("gaze")
            head = result.detection("head_pose")
            face_valid = context.primary_face_box is not None and context.landmarks is not None
            observations[fixture["name"]]["face"].append(face_valid)
            observations[fixture["name"]]["hard_no_face"].append(
                context.face_status == "NO_FACE"
            )
            observations[fixture["name"]]["gaze"].append(gaze.state)
            observations[fixture["name"]]["head"].append(head.state)
            if head.metadata.get("relative_angles") is not None:
                observations[fixture["name"]]["head_relative"].append(
                    head.metadata["relative_angles"]
                )
            if gaze.metadata.get("smoothed_displacement") is not None:
                observations[fixture["name"]]["gaze_displacement"].append(
                    gaze.metadata["smoothed_displacement"]
                )
            timings["face_landmarks_ms"].append(result.timing.face_landmarks_ms)
            timings["gaze_ms"].append(result.timing.gaze_ms)
            timings["head_pose_ms"].append(result.timing.head_pose_ms)
            timings["total_without_yolo_ms"].append(result.timing.total_ms)

    fixture_results = []
    expected_face_observations = 0
    detected_expected_faces = 0
    valid_gaze_observations = 0
    valid_head_observations = 0
    hard_false_no_face_observations = 0
    unexpected_face_observations = 0
    for fixture in fixtures:
        result = observations[fixture["name"]]
        expected = fixture["expected_face_visible"]
        detected_count = sum(result["face"])
        if expected:
            expected_face_observations += len(result["face"])
            detected_expected_faces += detected_count
            hard_false_no_face_observations += sum(result["hard_no_face"])
            valid_gaze_observations += sum(
                state not in {GAZE_NO_FACE, GAZE_UNKNOWN, HEAD_CALIBRATING}
                for state in result["gaze"]
            )
            valid_head_observations += sum(
                state not in {HEAD_NO_FACE, HEAD_UNKNOWN, HEAD_CALIBRATING}
                for state in result["head"]
            )
        else:
            unexpected_face_observations += detected_count
        fixture_results.append(
            {
                "name": fixture["name"],
                "expected_face_visible": expected,
                "face_detection_rate": detected_count / len(result["face"]),
                "gaze_states": dict(Counter(result["gaze"])),
                "head_states": dict(Counter(result["head"])),
                "gaze_stable_fraction": stable_fraction(result["gaze"]),
                "head_stable_fraction": stable_fraction(result["head"]),
                "median_head_relative_angles": (
                    tuple(
                        float(value)
                        for value in np.median(
                            np.asarray(result["head_relative"]),
                            axis=0,
                        )
                    )
                    if result["head_relative"]
                    else None
                ),
                "median_gaze_displacement": (
                    tuple(
                        float(value)
                        for value in np.median(
                            np.asarray(result["gaze_displacement"]),
                            axis=0,
                        )
                    )
                    if result["gaze_displacement"]
                    else None
                ),
            }
        )

    mean_timings = {
        name: statistics.fmean(values) for name, values in timings.items()
    }
    report = {
        "fixture_directory": str(args.fixtures.resolve()),
        "fixture_count": len(fixtures),
        "repeats": args.repeats,
        "calibration_reference": reference_name,
        "calibration_samples": calibration_samples,
        "expected_visible_face_fixture_count": sum(
            item["expected_face_visible"] for item in fixtures
        ),
        "reliability": {
            "expected_face_detection_rate": detected_expected_faces
            / expected_face_observations,
            "raw_face_landmark_miss_count": expected_face_observations
            - detected_expected_faces,
            "false_no_face_count": hard_false_no_face_observations,
            "gaze_valid_rate_when_face_expected": valid_gaze_observations
            / expected_face_observations,
            "head_valid_rate_when_face_expected": valid_head_observations
            / expected_face_observations,
            "unexpected_face_detections_on_partial_or_empty_frames": (
                unexpected_face_observations
            ),
        },
        "mean_timing_ms": mean_timings,
        "fps_without_yolo": 1000.0 / mean_timings["total_without_yolo_ms"],
        "fixtures": fixture_results,
        "limitations": [
            "Saved screenshots are operational regressions, not controlled pose/gaze captures.",
            "Several screenshots contain prior UI annotations.",
            "Expected face visibility labels were assigned by visual inspection.",
        ],
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
