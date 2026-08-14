"""OpenCV visualization for structured surveillance results."""

import cv2

from detectors import (
    CALIBRATING,
    FACE_PRESENCE_SOURCE,
    GAZE_SOURCE,
    HEAD_POSE_SOURCE,
    PHONE_DETECTED,
    PHONE_SOURCE,
)
from eye_movement import NO_FACE as EYE_NO_FACE
from eye_movement import UNKNOWN as EYE_UNKNOWN
from face_context import MULTIPLE_FACES, PRIMARY_TEMPORARILY_MISSING
from head_pose import NO_FACE as HEAD_NO_FACE
from head_pose import UNKNOWN as HEAD_UNKNOWN


def _draw_face_boxes(frame, face_context):
    for face_box in face_context.face_boxes:
        is_primary = face_box == face_context.primary_face_box
        color = (255, 200, 0) if is_primary else (0, 165, 255)
        thickness = 2 if is_primary else 1
        cv2.rectangle(
            frame,
            (face_box[0], face_box[1]),
            (face_box[2], face_box[3]),
            color,
            thickness,
        )


def _draw_gaze_markers(frame, gaze_result):
    for eye_rect in gaze_result.metadata.get("eye_boxes", ()):
        x, y, width, height = eye_rect
        cv2.rectangle(
            frame,
            (x, y),
            (x + width, y + height),
            (0, 255, 0),
            2,
        )
    for pupil_center in gaze_result.metadata.get("pupil_centers", ()):
        cv2.circle(frame, pupil_center, 5, (0, 0, 255), -1)


def _draw_phone_boxes(frame, phone_result):
    for detection in phone_result.metadata.get("accepted_detections", ()):
        x1, y1, x2, y2 = detection["bounding_box"]
        label = f"{detection['class_name']} ({detection['confidence']:.2f})"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )


def render_frame(
    clean_frame,
    processing_result,
    rolling_performance=None,
    calibration_progress=None,
):
    """Render one display frame without modifying inference inputs or results."""
    display_frame = clean_frame.copy()
    face_result = processing_result.detection(FACE_PRESENCE_SOURCE)
    gaze_result = processing_result.detection(GAZE_SOURCE)
    head_result = processing_result.detection(HEAD_POSE_SOURCE)
    phone_result = processing_result.detection(PHONE_SOURCE)

    _draw_face_boxes(display_frame, processing_result.face_context)
    _draw_gaze_markers(display_frame, gaze_result)
    _draw_phone_boxes(display_frame, phone_result)

    gaze_color = (0, 255, 0)
    if gaze_result.state == CALIBRATING:
        gaze_color = (0, 255, 255)
    elif gaze_result.state in {EYE_NO_FACE, EYE_UNKNOWN}:
        gaze_color = (
            (0, 0, 255) if gaze_result.state == EYE_NO_FACE else (0, 255, 255)
        )
    cv2.putText(
        display_frame,
        f"Gaze Direction: {gaze_result.state}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        gaze_color,
        2,
    )

    head_color = (0, 255, 0)
    if head_result.state == CALIBRATING:
        head_color = (0, 255, 255)
    elif head_result.state in {HEAD_NO_FACE, HEAD_UNKNOWN}:
        head_color = (
            (0, 0, 255) if head_result.state == HEAD_NO_FACE else (0, 255, 255)
        )
    cv2.putText(
        display_frame,
        f"Head Direction: {head_result.state}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        head_color,
        2,
    )

    phone_fresh = bool(phone_result.metadata.get("fresh", False))
    phone_reused = bool(phone_result.metadata.get("reused", False))
    phone_stale = bool(phone_result.metadata.get("stale", False))
    if phone_result.state == CALIBRATING:
        mobile_detected = False
        mobile_text = CALIBRATING
        mobile_color = (0, 255, 255)
    elif phone_fresh:
        mobile_detected = phone_result.state == PHONE_DETECTED
        mobile_text = f"{mobile_detected} (fresh)"
        mobile_color = (0, 0, 255) if mobile_detected else (0, 255, 0)
    elif phone_reused:
        mobile_detected = (
            phone_result.metadata.get("cached_state") == PHONE_DETECTED
        )
        mobile_text = (
            f"{mobile_detected} (cached "
            f"{phone_result.metadata.get('result_age_seconds', 0.0):.1f}s)"
        )
        mobile_color = (0, 165, 255)
    else:
        mobile_detected = False
        mobile_text = "UNKNOWN (stale)" if phone_stale else EYE_UNKNOWN
        mobile_color = (0, 255, 255)
    cv2.putText(
        display_frame,
        f"Mobile Detected: {mobile_text}",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        mobile_color,
        2,
    )

    face_color = (
        (0, 255, 255)
        if face_result.state in {MULTIPLE_FACES, PRIMARY_TEMPORARILY_MISSING}
        else (0, 255, 0)
        if processing_result.face_context.primary_face_box is not None
        else (0, 0, 255)
    )
    cv2.putText(
        display_frame,
        f"Face: {face_result.state} "
        f"(count={processing_result.face_context.face_count})",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        face_color,
        2,
    )

    if calibration_progress is not None:
        cv2.putText(
            display_frame,
            "Calibrating: keep head straight "
            f"({calibration_progress['samples']}/"
            f"{calibration_progress['minimum_samples']} joint samples, "
            f"{calibration_progress['valid_seconds']:.1f}/"
            f"{calibration_progress['required_seconds']:.0f}s)",
            (20, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )

    if rolling_performance is not None:
        cv2.putText(
            display_frame,
            f"Latency ms F:{rolling_performance['face_ms']:.0f} "
            f"G:{rolling_performance['gaze_ms']:.0f} "
            f"H:{rolling_performance['head_ms']:.0f} "
            f"Y:{rolling_performance['phone_ms']:.0f} "
            f"Total:{rolling_performance['total_ms']:.0f} "
            f"ProcFPS:{rolling_performance['fps']:.1f} "
            f"YHz:{rolling_performance['phone_hz']:.1f} "
            f"Fresh:{'Y' if rolling_performance['phone_fresh'] else 'N'}",
            (20, max(20, display_frame.shape[0] - 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1,
        )

    return display_frame
