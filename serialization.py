"""Strict conversion of detector metadata to JSON-compatible values."""

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import numpy as np


def to_json_compatible(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("Non-finite floating-point values cannot be persisted as JSON.")
        return value
    if isinstance(value, np.generic):
        return to_json_compatible(value.item())
    if isinstance(value, np.ndarray):
        return to_json_compatible(value.tolist())
    if isinstance(value, Enum):
        return to_json_compatible(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError("Naive datetimes cannot be persisted; UTC is required.")
        return value.astimezone(timezone.utc).isoformat()
    if is_dataclass(value):
        return to_json_compatible(asdict(value))
    if isinstance(value, dict):
        return {
            str(key): to_json_compatible(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [to_json_compatible(item) for item in value]
    raise TypeError(
        f"Unsupported metadata value {type(value).__name__}; "
        "persist only JSON-compatible domain data."
    )


def dumps_json(value):
    return json.dumps(
        to_json_compatible(value),
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def loads_json(value):
    if value is None:
        return None
    return json.loads(value)
