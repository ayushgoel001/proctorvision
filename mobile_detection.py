import logging
from pathlib import Path
from zipfile import BadZipFile, ZipFile, is_zipfile

from config import YOLO_MODEL_PATH

MIN_CHECKPOINT_BYTES = 1_000_000
PHONE_CLASS_INDEX = 0
PHONE_CLASS_NAMES = {"phone", "mobile", "cellphone", "mobilephone", "smartphone"}
YOLO_CANDIDATE_CONFIDENCE = 0.05
# Provisional detector-only threshold. Controlled samples currently show overlap
# between weak true positives and background-person false positives, so this is
# intentionally not treated as a final operating threshold.
PHONE_CONFIDENCE_THRESHOLD = 0.65
# Use a modest base pass. A phone-class candidate at or
# above 0.42 is immediately re-evaluated on the same clean frame at 512 before
# acceptance. Archived paired-size results showed all 512-accepted hard phones
# first reached at least 0.424 at 384, while current no-phone captures topped out
# at 0.402. This avoids paying the 512 cost on every routine frame.
YOLO_INFERENCE_SIZE = 384
YOLO_CONFIRMATION_INFERENCE_SIZE = 512
PHONE_HIGH_RESOLUTION_TRIGGER_CONFIDENCE = 0.42
# Phone inference remains synchronous, but it is sampled at a conservative
# cadence because it is substantially more expensive than all other stages.
PHONE_INFERENCE_EVERY_N_FRAMES = 2
PHONE_RESULT_MAX_REUSE_AGE_SECONDS = 1.0

LOGGER = logging.getLogger(__name__)


def validate_checkpoint(model_path=YOLO_MODEL_PATH):
    """Validate the intended local PyTorch checkpoint without executing it."""
    path = Path(model_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"YOLO checkpoint not found: {path}. "
            "Place best_yolov12.pt in the project's model directory."
        )
    if path.stat().st_size < MIN_CHECKPOINT_BYTES:
        raise RuntimeError(
            f"YOLO checkpoint is too small to be valid: {path} "
            f"({path.stat().st_size} bytes)."
        )
    if not is_zipfile(path):
        raise RuntimeError(
            f"YOLO checkpoint is not a valid PyTorch ZIP archive: {path}."
        )

    try:
        with ZipFile(path) as archive:
            names = archive.namelist()
            if not any(name.endswith("/data.pkl") for name in names):
                raise RuntimeError(
                    f"YOLO checkpoint is missing PyTorch metadata: {path}."
                )
            corrupt_entry = archive.testzip()
            if corrupt_entry is not None:
                raise RuntimeError(
                    f"YOLO checkpoint contains a corrupt entry ({corrupt_entry}): {path}."
                )
    except BadZipFile as exc:
        raise RuntimeError(f"YOLO checkpoint is corrupt: {path}.") from exc

    return path


def load_mobile_model(model_path=YOLO_MODEL_PATH):
    """Explicitly load the validated local model; no download fallback is used."""
    path = validate_checkpoint(model_path)
    try:
        import torch
        from ultralytics import YOLO

        model = YOLO(str(path))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch and Ultralytics are required for phone inference. "
            "Install the runtime dependencies from requirements.txt."
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load YOLO checkpoint at {path}. "
            "Check that the checkpoint matches the installed Ultralytics version."
        ) from exc

    class_name = validate_phone_class(getattr(model, "names", None))
    LOGGER.info(
        "Loaded YOLO phone model class_map=%s expected_index=%d expected_name=%s "
        "candidate_confidence=%.2f acceptance_confidence=%.2f base_size=%d "
        "confirmation_size=%d confirmation_trigger=%.2f "
        "inference_every_n_frames=%d",
        model.names,
        PHONE_CLASS_INDEX,
        class_name,
        YOLO_CANDIDATE_CONFIDENCE,
        PHONE_CONFIDENCE_THRESHOLD,
        YOLO_INFERENCE_SIZE,
        YOLO_CONFIRMATION_INFERENCE_SIZE,
        PHONE_HIGH_RESOLUTION_TRIGGER_CONFIDENCE,
        PHONE_INFERENCE_EVERY_N_FRAMES,
    )
    return model


def validate_phone_class(names, expected_index=PHONE_CLASS_INDEX):
    """Verify that the configured YOLO class is semantically a phone class."""
    if isinstance(names, (list, tuple)):
        class_name = names[expected_index] if len(names) > expected_index else None
    elif isinstance(names, dict):
        class_name = names.get(expected_index, names.get(str(expected_index)))
    else:
        class_name = None

    if class_name is None:
        raise RuntimeError(
            f"YOLO checkpoint does not define the expected class index {expected_index}."
        )

    normalized_name = "".join(
        character
        for character in str(class_name).strip().lower()
        if character.isalnum()
    )
    if normalized_name not in PHONE_CLASS_NAMES:
        raise RuntimeError(
            f"YOLO class index {expected_index} is named {class_name!r}, not a recognized "
            "phone/mobile class. Refusing to start with an incompatible checkpoint."
        )

    return str(class_name)


def _class_name(names, class_index):
    if isinstance(names, dict):
        return str(names.get(class_index, names.get(str(class_index), f"class_{class_index}")))
    if isinstance(names, (list, tuple)) and 0 <= class_index < len(names):
        return str(names[class_index])
    return f"class_{class_index}"


def _analyze_mobile_detection_once(frame, model, inference_size):
    """Run one YOLO pass and return its thresholded phone observations."""
    if model is None:
        raise ValueError("A loaded YOLO model is required for mobile detection.")
    if not isinstance(inference_size, int) or inference_size <= 0:
        raise ValueError("YOLO inference size must be a positive integer.")

    try:
        # Ask YOLO to retain low-confidence candidates for diagnostics. The application
        # threshold below still decides whether a phone is accepted.
        results = model.predict(
            frame,
            conf=YOLO_CANDIDATE_CONFIDENCE,
            imgsz=inference_size,
            verbose=False,
        )
    except Exception as exc:
        raise RuntimeError("YOLO inference failed for the current frame.") from exc

    mobile_detected = False
    prediction_count = 0
    predictions = []
    accepted_detections = []
    max_phone_confidence = None

    for result in results:
        for box in result.boxes:
            prediction_count += 1
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            class_name = _class_name(result.names, cls)
            accepted = (
                cls == PHONE_CLASS_INDEX
                and conf >= PHONE_CONFIDENCE_THRESHOLD
            )
            LOGGER.debug(
                "YOLO prediction size=%d class_index=%d class_name=%s "
                "confidence=%.3f accepted=%s",
                inference_size,
                cls,
                class_name,
                conf,
                accepted,
            )

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            prediction = {
                "class_index": cls,
                "class_name": class_name,
                "confidence": float(conf),
                "bounding_box": (x1, y1, x2, y2),
                "accepted": accepted,
            }
            predictions.append(prediction)
            if cls == PHONE_CLASS_INDEX:
                max_phone_confidence = (
                    float(conf)
                    if max_phone_confidence is None
                    else max(max_phone_confidence, float(conf))
                )

            if not accepted:
                continue
            accepted_detections.append(prediction)
            mobile_detected = True

    if prediction_count == 0:
        LOGGER.debug(
            "YOLO produced no predictions above candidate confidence %.2f.",
            YOLO_CANDIDATE_CONFIDENCE,
        )

    return mobile_detected, {
        "predictions": tuple(predictions),
        "accepted_detections": tuple(accepted_detections),
        "inference_size": inference_size,
        "candidate_threshold": YOLO_CANDIDATE_CONFIDENCE,
        "acceptance_threshold": PHONE_CONFIDENCE_THRESHOLD,
        "max_phone_confidence": max_phone_confidence,
    }


def analyze_mobile_detection(
    frame,
    model,
    inference_size=YOLO_INFERENCE_SIZE,
    confirmation_size=YOLO_CONFIRMATION_INFERENCE_SIZE,
    confirmation_trigger=PHONE_HIGH_RESOLUTION_TRIGGER_CONFIDENCE,
):
    """Run the fast pass and conditionally confirm phone candidates at 512."""
    if confirmation_size is not None and (
        not isinstance(confirmation_size, int) or confirmation_size <= 0
    ):
        raise ValueError("YOLO confirmation size must be a positive integer or None.")
    if not 0.0 <= confirmation_trigger <= 1.0:
        raise ValueError("YOLO confirmation trigger must be between 0 and 1.")

    mobile_detected, base_metadata = _analyze_mobile_detection_once(
        frame,
        model,
        inference_size,
    )
    base_max_confidence = base_metadata["max_phone_confidence"]
    confirmation_executed = bool(
        confirmation_size is not None
        and confirmation_size != inference_size
        and base_max_confidence is not None
        and base_max_confidence >= confirmation_trigger
    )
    if confirmation_executed:
        mobile_detected, final_metadata = _analyze_mobile_detection_once(
            frame,
            model,
            confirmation_size,
        )
        LOGGER.debug(
            "YOLO high-resolution confirmation base_size=%d base_confidence=%.3f "
            "confirmation_size=%d confirmation_confidence=%s accepted=%s",
            inference_size,
            base_max_confidence,
            confirmation_size,
            (
                f"{final_metadata['max_phone_confidence']:.3f}"
                if final_metadata["max_phone_confidence"] is not None
                else "none"
            ),
            mobile_detected,
        )
    else:
        final_metadata = base_metadata

    return mobile_detected, {
        **final_metadata,
        "base_inference_size": inference_size,
        "base_max_phone_confidence": base_max_confidence,
        "confirmation_inference_size": confirmation_size,
        "confirmation_trigger": confirmation_trigger,
        "confirmation_executed": confirmation_executed,
        "confirmation_max_phone_confidence": (
            final_metadata["max_phone_confidence"]
            if confirmation_executed
            else None
        ),
        "inference_sizes_executed": (
            (inference_size, confirmation_size)
            if confirmation_executed
            else (inference_size,)
        ),
    }
