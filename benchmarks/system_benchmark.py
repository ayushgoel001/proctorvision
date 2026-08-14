"""Reproducible steady-state latency benchmark for the production pipeline.

This measures runtime stages; it is not an accuracy evaluation. Use raw,
consented fixtures when reporting results publicly.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detectors import PHONE_SOURCE  # noqa: E402
from mobile_detection import (  # noqa: E402
    PHONE_CONFIDENCE_THRESHOLD,
    PHONE_HIGH_RESOLUTION_TRIGGER_CONFIDENCE,
    PHONE_INFERENCE_EVERY_N_FRAMES,
    YOLO_CONFIRMATION_INFERENCE_SIZE,
    YOLO_INFERENCE_SIZE,
)
from surveillance_engine import SurveillanceEngine  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark the current ProctorVision inference pipeline."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "evidence",
        help="image directory, image file, or video file",
    )
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.warmup < 0 or args.frames < 1:
        parser.error("--warmup must be non-negative and --frames must be positive")
    return args


def _load_image(path):
    frame = cv2.imread(str(path))
    if frame is None:
        raise RuntimeError(f"OpenCV could not decode image: {path}")
    return frame


def load_frames(input_path):
    input_path = input_path.expanduser().resolve()
    if input_path.is_dir():
        paths = sorted(
            path
            for path in input_path.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        frames = [_load_image(path) for path in paths]
    elif input_path.is_file() and input_path.suffix.lower() in IMAGE_SUFFIXES:
        frames = [_load_image(input_path)]
    elif input_path.is_file():
        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            raise RuntimeError(f"OpenCV could not open video: {input_path}")
        frames = []
        try:
            while True:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                frames.append(frame)
        finally:
            capture.release()
    else:
        raise FileNotFoundError(f"Benchmark input not found: {input_path}")

    if not frames:
        raise RuntimeError(f"No decodable benchmark frames found in {input_path}")
    return input_path, frames


def percentile(values, percentage):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentage)
    return float(ordered[index])


def summarize(values):
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0}
    return {
        "mean": float(statistics.fmean(values)),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
    }


def benchmark(frames, warmup_count, measured_count):
    engine = SurveillanceEngine()
    engine.reset_session()
    try:
        for index in range(warmup_count):
            engine.process_frame(frames[index % len(frames)], calibrating=False)

        timings = []
        phone_confidences = []
        phone_positive_frames = 0
        started_at = time.perf_counter()
        for index in range(measured_count):
            result = engine.process_frame(
                frames[(warmup_count + index) % len(frames)],
                calibrating=False,
            )
            timings.append(asdict(result.timing))
            phone = result.detection(PHONE_SOURCE)
            if phone.suspicious:
                phone_positive_frames += 1
            confidence = phone.metadata.get("max_phone_confidence")
            if confidence is not None:
                phone_confidences.append(float(confidence))
        wall_seconds = time.perf_counter() - started_at
    finally:
        engine.close()

    executed_phone_ms = [
        row["phone_ms"] for row in timings if row["phone_inference_executed"]
    ]
    stage_values = {
        "face_landmarks_ms": [row["face_landmarks_ms"] for row in timings],
        "gaze_ms": [row["gaze_ms"] for row in timings],
        "head_pose_ms": [row["head_pose_ms"] for row in timings],
        "yolo_executed_ms": executed_phone_ms,
        "total_ms": [row["total_ms"] for row in timings],
    }
    total_mean = statistics.fmean(stage_values["total_ms"])
    phone_inference_count = len(executed_phone_ms)
    return {
        "warmup_frames": warmup_count,
        "measured_frames": measured_count,
        "stage_latency_ms": {
            stage: summarize(values) for stage, values in stage_values.items()
        },
        "effective_fps_from_mean_latency": 1000.0 / total_mean,
        "observed_processing_fps": measured_count / wall_seconds,
        "phone_inference_count": phone_inference_count,
        "phone_inference_hz": phone_inference_count / wall_seconds,
        "phone_positive_frame_count": phone_positive_frames,
        "highest_phone_confidence": max(phone_confidences, default=None),
    }


def main():
    args = parse_args()
    input_path, frames = load_frames(args.input)
    report = {
        "benchmark": "ProctorVision production pipeline",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor() or "not reported",
            "opencv": cv2.__version__,
        },
        "input": {
            "path": str(input_path),
            "unique_frame_count": len(frames),
            "first_frame_shape": list(frames[0].shape),
        },
        "production_phone_configuration": {
            "base_inference_size": YOLO_INFERENCE_SIZE,
            "confirmation_inference_size": YOLO_CONFIRMATION_INFERENCE_SIZE,
            "confirmation_trigger": PHONE_HIGH_RESOLUTION_TRIGGER_CONFIDENCE,
            "acceptance_confidence": PHONE_CONFIDENCE_THRESHOLD,
            "inference_every_n_frames": PHONE_INFERENCE_EVERY_N_FRAMES,
        },
        "results": benchmark(frames, args.warmup, args.frames),
        "limitations": [
            "This is a latency benchmark, not an accuracy or fairness evaluation.",
            "Results are hardware-, driver-, input-, and model-version-specific.",
            "Repeated still images do not reproduce camera motion or end-to-end UI cost.",
        ],
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
