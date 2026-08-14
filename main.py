import argparse
import logging
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from config import (
    CALIBRATION_SECONDS,
    CALIBRATION_TIMEOUT_SECONDS,
    DATABASE_PATH,
    DEFAULT_VIDEO_SOURCE,
    DIAGNOSTIC_LOG_INTERVAL_SECONDS,
    LATENCY_WINDOW_SIZE,
    MIN_CALIBRATION_SAMPLES,
    PROJECT_ROOT,
)
from config import (
    EVIDENCE_DIRECTORY as EVIDENCE_ROOT,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CaptureSource:
    capture_value: int | str
    session_label: str
    description: str
    is_camera: bool


def resolve_capture_source(source):
    """Resolve a camera index or project-relative video path for OpenCV."""

    if isinstance(source, int):
        if source < 0:
            raise ValueError("Camera index cannot be negative.")
        return CaptureSource(source, f"camera:{source}", f"camera index {source}", True)

    source_text = str(source).strip()
    if not source_text:
        raise ValueError("Video source cannot be empty.")
    if source_text.lstrip("-").isdecimal():
        camera_index = int(source_text)
        if camera_index < 0:
            raise ValueError("Camera index cannot be negative.")
        return CaptureSource(
            camera_index,
            f"camera:{camera_index}",
            f"camera index {camera_index}",
            True,
        )

    source_path = Path(source_text).expanduser()
    if not source_path.is_absolute():
        source_path = PROJECT_ROOT / source_path
    source_path = source_path.resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Video source not found: {source_path}")
    try:
        stored_path = source_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        stored_path = source_path.name
    return CaptureSource(
        str(source_path),
        f"video:{stored_path}",
        f"video file {source_path}",
        False,
    )


def _save_event_evidence(frame, session_id, event_id):
    evidence_directory = EVIDENCE_ROOT / session_id
    evidence_directory.mkdir(parents=True, exist_ok=True)
    absolute_path = evidence_directory / f"{event_id}.jpg"
    if not cv2.imwrite(str(absolute_path), frame):
        raise RuntimeError(f"Failed to write evidence screenshot: {absolute_path}")
    relative_path = absolute_path.relative_to(PROJECT_ROOT).as_posix()
    return absolute_path, relative_path


def _fail_session_safely(session_service, session_id, reason, average_fps=None):
    try:
        current = session_service.get_session(session_id)
        if current.status.value not in {"STOPPED", "FAILED"}:
            session_service.fail_session(
                session_id,
                reason,
                average_fps=average_fps,
            )
    except Exception as persistence_exc:
        LOGGER.error(
            "Failed to record terminal FAILED state for session %s: %s",
            session_id,
            persistence_exc,
        )


def run_application(source=DEFAULT_VIDEO_SOURCE, debug=False):
    try:
        capture_source = resolve_capture_source(source)
    except (FileNotFoundError, ValueError) as exc:
        LOGGER.error("Invalid monitoring source: %s", exc)
        return 1

    repository = None
    session_service = None
    session = None
    engine = None
    try:
        from persistence import SQLiteRepository
        from session_service import SessionService

        repository = SQLiteRepository(DATABASE_PATH)
        session_service = SessionService(repository)
        session = session_service.create_session(capture_source.session_label)
        LOGGER.info(
            "Monitoring session created id=%s source=%s",
            session.session_id,
            session.video_source,
        )
    except Exception as exc:
        if repository is not None:
            repository.close()
        LOGGER.error("Persistence/session initialization failed: %s", exc)
        return 1

    # Model imports happen after session creation so startup failures are persisted.
    try:
        from alert_engine import AlertEngine
        from detectors import (
            GAZE_SOURCE,
            HEAD_POSE_SOURCE,
            PHONE_DETECTED,
            PHONE_SOURCE,
        )
        from head_pose import (
            head_pose_deltas,
            mean_rotation_matrix,
            rotation_matrix_to_euler_degrees,
        )
        from renderer import render_frame
        from surveillance_engine import SurveillanceEngine

        engine = SurveillanceEngine()
        alert_engine = AlertEngine()
    except Exception as exc:
        _fail_session_safely(
            session_service,
            session.session_id,
            f"Model initialization failed: {exc}",
        )
        repository.close()
        LOGGER.error("Model initialization failed: %s", exc)
        return 1

    cap = None
    last_average_fps = None
    session_terminal = False
    try:
        cap = cv2.VideoCapture(capture_source.capture_value)
        if not cap.isOpened():
            raise RuntimeError(
                f"Unable to open {capture_source.description}. "
                "For a camera, check permissions and whether another application "
                "is using it. For a file, confirm that OpenCV supports its codec."
            )

        session = session_service.start_calibration(session.session_id)
        LOGGER.info("Session %s entered CALIBRATING", session.session_id)

        calibration_started_at = time.monotonic()
        calibration_first_sample_at = None
        calibration_latest_sample_at = None
        calibration_samples = []
        calibrated_rotation = None

        face_latencies = deque(maxlen=LATENCY_WINDOW_SIZE)
        gaze_latencies = deque(maxlen=LATENCY_WINDOW_SIZE)
        head_latencies = deque(maxlen=LATENCY_WINDOW_SIZE)
        yolo_latencies = deque(maxlen=LATENCY_WINDOW_SIZE)
        phone_inference_times = deque(maxlen=LATENCY_WINDOW_SIZE)
        total_latencies = deque(maxlen=LATENCY_WINDOW_SIZE)
        last_diagnostic_log_at = time.monotonic()
        last_head_pose_log_at = time.monotonic()
        frame_number = 0
        phone_test_visible_at = None
        phone_test_visible_frame = None
        phone_first_accepted_reported = False

        while True:
            ret, captured_frame = cap.read()
            if not ret or captured_frame is None:
                if capture_source.is_camera:
                    raise RuntimeError(
                        "Camera opened but failed to return a video frame."
                    )
                if frame_number == 0:
                    raise RuntimeError(
                        "Video source opened but contained no decodable frames."
                    )
                session = session_service.stop_session(
                    session.session_id,
                    average_fps=last_average_fps,
                )
                session_terminal = True
                LOGGER.info(
                    "Video source completed; session=%s events=%d average_fps=%s",
                    session.session_id,
                    session.event_count,
                    session.average_fps,
                )
                return 0

            # Inference consumes only this clean frame. Rendering happens after all
            # structured detector results have been produced.
            frame_number += 1
            clean_frame = captured_frame.copy()
            monitoring_frame = calibrated_rotation is not None
            processing_result = engine.process_frame(
                clean_frame,
                calibrating=not monitoring_frame,
            )
            gaze_result = processing_result.detection(GAZE_SOURCE)
            head_result = processing_result.detection(HEAD_POSE_SOURCE)
            phone_result = processing_result.detection(PHONE_SOURCE)
            head_direction = head_result.state
            current_rotation = head_result.metadata.get("current_rotation")

            now = time.monotonic()
            calibration_progress = None
            if calibrated_rotation is None:
                gaze_calibration_sample_valid = bool(
                    gaze_result.metadata.get("calibration_sample_valid", False)
                )
                if current_rotation is not None and gaze_calibration_sample_valid:
                    if calibration_first_sample_at is None:
                        calibration_first_sample_at = now
                    calibration_latest_sample_at = now
                    calibration_samples.append(current_rotation)

                # Duration advances only when a valid sample updates the latest timestamp.
                valid_sample_seconds = (
                    calibration_latest_sample_at - calibration_first_sample_at
                    if calibration_first_sample_at is not None
                    and calibration_latest_sample_at is not None
                    else 0.0
                )
                calibration_progress = {
                    "samples": len(calibration_samples),
                    "gaze_samples": engine.gaze_calibration_sample_count,
                    "minimum_samples": MIN_CALIBRATION_SAMPLES,
                    "valid_seconds": valid_sample_seconds,
                    "required_seconds": CALIBRATION_SECONDS,
                }

                calibration_ready = (
                    len(calibration_samples) >= MIN_CALIBRATION_SAMPLES
                    and engine.gaze_calibration_sample_count
                    >= MIN_CALIBRATION_SAMPLES
                    and valid_sample_seconds >= CALIBRATION_SECONDS
                )
                if calibration_ready:
                    calibrated_rotation = mean_rotation_matrix(calibration_samples)
                    engine.set_head_calibration(calibrated_rotation)
                    gaze_baseline = engine.finalize_gaze_calibration(
                        MIN_CALIBRATION_SAMPLES
                    )
                    calibrated_absolute_angles = rotation_matrix_to_euler_degrees(
                        calibrated_rotation
                    )
                    session = session_service.mark_running(
                        session.session_id,
                        {
                            "sample_count": len(calibration_samples),
                            "valid_sample_seconds": valid_sample_seconds,
                            "minimum_samples": MIN_CALIBRATION_SAMPLES,
                            "required_seconds": CALIBRATION_SECONDS,
                            "calibrated_absolute_angles": calibrated_absolute_angles,
                            "gaze_baseline": gaze_baseline.as_metadata(),
                        },
                    )
                    LOGGER.info(
                        "Session %s entered RUNNING; head-pose calibration completed "
                        "with %d samples; gaze_inliers=%d/%d absolute_angles=%s",
                        session.session_id,
                        len(calibration_samples),
                        gaze_baseline.inlier_count,
                        gaze_baseline.sample_count,
                        calibrated_absolute_angles,
                    )
                elif now - calibration_started_at >= CALIBRATION_TIMEOUT_SECONDS:
                    raise RuntimeError(
                        "Head/gaze calibration timed out: "
                        f"received {len(calibration_samples)} joint head/gaze samples "
                        f"({engine.gaze_calibration_sample_count} valid gaze samples) across "
                        f"{valid_sample_seconds:.1f} seconds. Keep one face visible, "
                        "look straight at the camera, and restart the application."
                    )

            if (
                calibrated_rotation is not None
                and current_rotation is not None
                and now - last_head_pose_log_at >= DIAGNOSTIC_LOG_INTERVAL_SECONDS
            ):
                pitch_delta, yaw_delta, roll_delta = head_pose_deltas(
                    current_rotation,
                    calibrated_rotation,
                )
                calibrated_absolute_angles = rotation_matrix_to_euler_degrees(
                    calibrated_rotation
                )
                current_absolute_angles = rotation_matrix_to_euler_degrees(
                    current_rotation
                )
                LOGGER.debug(
                    "Head pose calibration_absolute=(%.2f, %.2f, %.2f) "
                    "current_absolute=(%.2f, %.2f, %.2f) "
                    "relative=(raw_pitch=%.2f, yaw=%.2f, roll=%.2f) "
                    "pitch_convention=negative_is_physical_up state=%s",
                    *calibrated_absolute_angles,
                    *current_absolute_angles,
                    pitch_delta,
                    yaw_delta,
                    roll_delta,
                    head_direction,
                )
                last_head_pose_log_at = now

            if monitoring_frame and phone_result.metadata.get("fresh", False):
                max_phone_confidence = phone_result.metadata.get(
                    "max_phone_confidence"
                )
                base_max_phone_confidence = phone_result.metadata.get(
                    "base_max_phone_confidence",
                    max_phone_confidence,
                )
                acceptance_threshold = phone_result.metadata.get(
                    "acceptance_threshold"
                )
                accepted = phone_result.state == PHONE_DETECTED
                visible_elapsed = (
                    now - phone_test_visible_at
                    if phone_test_visible_at is not None
                    else None
                )
                LOGGER.debug(
                    "Fresh YOLO inference frame=%d sequence=%s sizes=%s "
                    "latency_ms=%.1f base_max_phone_confidence=%s "
                    "final_max_phone_confidence=%s threshold=%s "
                    "confirmation_executed=%s accepted=%s "
                    "test_visible_elapsed_seconds=%s",
                    frame_number,
                    phone_result.metadata.get("inference_sequence"),
                    phone_result.metadata.get("inference_sizes_executed"),
                    processing_result.timing.phone_ms,
                    (
                        f"{base_max_phone_confidence:.3f}"
                        if base_max_phone_confidence is not None
                        else "none"
                    ),
                    (
                        f"{max_phone_confidence:.3f}"
                        if max_phone_confidence is not None
                        else "none"
                    ),
                    (
                        f"{acceptance_threshold:.2f}"
                        if acceptance_threshold is not None
                        else "unknown"
                    ),
                    phone_result.metadata.get("confirmation_executed", False),
                    accepted,
                    (
                        f"{visible_elapsed:.3f}"
                        if visible_elapsed is not None
                        else "not_marked"
                    ),
                )
                if (
                    accepted
                    and phone_test_visible_at is not None
                    and not phone_first_accepted_reported
                ):
                    LOGGER.debug(
                        "Phone time-to-first-detection visible_frame=%d "
                        "accepted_frame=%d elapsed_seconds=%.3f "
                        "fresh_inference_sequence=%s confidence=%.3f",
                        phone_test_visible_frame,
                        frame_number,
                        visible_elapsed,
                        phone_result.metadata.get("inference_sequence"),
                        max_phone_confidence,
                    )
                    phone_first_accepted_reported = True

            newly_confirmed_events = ()
            if monitoring_frame:
                alert_update = alert_engine.update(processing_result.detections)
                newly_confirmed_events = alert_update.newly_confirmed_events
                for event in alert_update.resolved_events:
                    persisted_event = session_service.resolve_event(
                        session.session_id,
                        event,
                    )
                    LOGGER.info(
                        "Review event resolved session=%s id=%s type=%s "
                        "resolved_at_utc=%s",
                        session.session_id,
                        event.event_id,
                        event.event_type,
                        persisted_event.resolved_at_utc.isoformat(),
                    )

            rolling_performance = None
            if monitoring_frame:
                face_latencies.append(processing_result.timing.face_landmarks_ms)
                gaze_latencies.append(processing_result.timing.gaze_ms)
                head_latencies.append(processing_result.timing.head_pose_ms)
                if processing_result.timing.phone_inference_executed:
                    yolo_latencies.append(processing_result.timing.phone_ms)
                    phone_inference_times.append(now)
                total_latencies.append(processing_result.timing.total_ms)

                mean_face = float(np.mean(face_latencies))
                mean_gaze = float(np.mean(gaze_latencies))
                mean_head = float(np.mean(head_latencies))
                mean_yolo = (
                    float(np.mean(yolo_latencies)) if yolo_latencies else 0.0
                )
                mean_total = float(np.mean(total_latencies))
                rolling_fps = 1000.0 / mean_total if mean_total > 0 else 0.0
                phone_inference_hz = (
                    (len(phone_inference_times) - 1)
                    / (phone_inference_times[-1] - phone_inference_times[0])
                    if len(phone_inference_times) >= 2
                    and phone_inference_times[-1] > phone_inference_times[0]
                    else 0.0
                )
                phone_result_fresh = bool(phone_result.metadata.get("fresh", False))
                last_average_fps = rolling_fps
                rolling_performance = {
                    "face_ms": mean_face,
                    "gaze_ms": mean_gaze,
                    "head_ms": mean_head,
                    "phone_ms": mean_yolo,
                    "total_ms": mean_total,
                    "fps": rolling_fps,
                    "phone_hz": phone_inference_hz,
                    "phone_fresh": phone_result_fresh,
                }

                if now - last_diagnostic_log_at >= DIAGNOSTIC_LOG_INTERVAL_SECONDS:
                    LOGGER.info(
                        "Rolling performance face_landmarks_ms=%.1f gaze_logic_ms=%.1f "
                        "head_logic_ms=%.1f yolo_when_executed_ms=%.1f "
                        "total_ms=%.1f processing_fps=%.2f phone_inference_hz=%.2f "
                        "phone_result_fresh=%s samples=%d",
                        mean_face,
                        mean_gaze,
                        mean_head,
                        mean_yolo,
                        mean_total,
                        rolling_fps,
                        phone_inference_hz,
                        phone_result_fresh,
                        len(total_latencies),
                    )
                    last_diagnostic_log_at = now

            display_frame = render_frame(
                clean_frame,
                processing_result,
                rolling_performance=rolling_performance,
                calibration_progress=calibration_progress,
            )

            for event in newly_confirmed_events:
                absolute_evidence_path, evidence_path = _save_event_evidence(
                    display_frame,
                    session.session_id,
                    event.event_id,
                )
                try:
                    persisted_event = session_service.record_confirmed_event(
                        session.session_id,
                        event,
                        evidence_path,
                    )
                except Exception:
                    try:
                        absolute_evidence_path.unlink(missing_ok=True)
                    except OSError as cleanup_exc:
                        LOGGER.error(
                            "Failed to remove orphan evidence file %s: %s",
                            absolute_evidence_path,
                            cleanup_exc,
                        )
                    raise
                LOGGER.info(
                    "Review event confirmed session=%s id=%s type=%s source=%s "
                    "state=%s confirmed_at_utc=%s evidence=%s",
                    session.session_id,
                    event.event_id,
                    event.event_type,
                    event.detector_source,
                    event.source_state,
                    persisted_event.confirmed_at_utc.isoformat(),
                    evidence_path,
                )

            cv2.imshow("Combined Detection", display_frame)
            pressed_key = cv2.waitKey(1) & 0xFF
            if debug and pressed_key == ord("p"):
                phone_test_visible_at = time.monotonic()
                phone_test_visible_frame = frame_number
                phone_first_accepted_reported = False
                LOGGER.debug(
                    "Phone test-visible marker set at frame=%d. Fresh YOLO "
                    "inferences will report time-to-first-detection; press n "
                    "after removing the phone.",
                    frame_number,
                )
            elif debug and pressed_key == ord("n"):
                LOGGER.debug(
                    "Phone test-visible marker cleared at frame=%d.",
                    frame_number,
                )
                phone_test_visible_at = None
                phone_test_visible_frame = None
                phone_first_accepted_reported = False
            elif pressed_key == ord("q"):
                session = session_service.stop_session(
                    session.session_id,
                    average_fps=last_average_fps,
                )
                session_terminal = True
                LOGGER.info(
                    "Session %s stopped events=%d average_fps=%s",
                    session.session_id,
                    session.event_count,
                    session.average_fps,
                )
                return 0

    except KeyboardInterrupt:
        try:
            if not session_terminal:
                session = session_service.stop_session(
                    session.session_id,
                    average_fps=last_average_fps,
                )
                session_terminal = True
            LOGGER.info(
                "Monitoring stopped by user; session=%s status=%s events=%d",
                session.session_id,
                session.status.value,
                session.event_count,
            )
            return 0
        except Exception as exc:
            _fail_session_safely(
                session_service,
                session.session_id,
                f"Failed to stop session cleanly: {exc}",
                average_fps=last_average_fps,
            )
            LOGGER.error("Failed to stop monitoring session cleanly: %s", exc)
            return 1
    except Exception as exc:
        _fail_session_safely(
            session_service,
            session.session_id,
            str(exc),
            average_fps=last_average_fps,
        )
        LOGGER.error("Surveillance runtime failed: %s", exc)
        return 1
    finally:
        if cap is not None:
            cap.release()
        if engine is not None:
            try:
                engine.close()
            except Exception as close_exc:
                LOGGER.debug("Face Landmarker cleanup failed: %s", close_exc)
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            LOGGER.debug("OpenCV windows were not available for cleanup.")
        repository.close()


def main():
    parser = argparse.ArgumentParser(
        description="Run ProctorVision monitoring on a webcam or local video."
    )
    parser.add_argument(
        "--source",
        default=DEFAULT_VIDEO_SOURCE,
        help="camera index (default: 0) or a project-relative/absolute video path",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="show per-frame detector diagnostics for controlled validation",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    if args.debug:
        for logger_name in (
            __name__,
            "face_context",
            "eye_movement",
            "head_pose",
            "mobile_detection",
        ):
            logging.getLogger(logger_name).setLevel(logging.DEBUG)
    return run_application(source=args.source, debug=args.debug)


if __name__ == "__main__":
    raise SystemExit(main())
