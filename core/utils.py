"""
core/utils.py -- Shared utility functions used across all modules.

Responsibilities:
  * Image loading / validation
  * JSON I/O helpers
  * CSV session-log writer
  * Bounding-box geometry helpers
  * Timestamp generation
"""

from __future__ import annotations

import csv
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


# -- Timestamp -----------------------------------------------------------------

def now_str() -> str:
    """Return current local time as ISO-8601 string (second precision)."""
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


# -- Image I/O -----------------------------------------------------------------

def load_image(path: str) -> np.ndarray:
    """
    Load an image from *path* and return a BGR NumPy array.

    Raises:
        FileNotFoundError  -- if *path* does not exist.
        ValueError         -- if OpenCV cannot decode the file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found: {path}")

    img = cv2.imread(path)
    if img is None:
        raise ValueError(
            f"OpenCV could not read image at '{path}'. "
            "Ensure it is a valid JPEG/PNG file."
        )
    return img


def save_image(img: np.ndarray, path: str) -> None:
    """Save *img* to *path*, creating parent directories as needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)


# -- JSON I/O ------------------------------------------------------------------

def load_json(path: str) -> Any:
    """
    Load and return parsed JSON from *path*.

    Raises:
        FileNotFoundError -- if *path* does not exist.
        ValueError        -- if the file contains invalid JSON.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        try:
            return json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in '{path}': {exc}") from exc


def save_json(data: Any, path: str, indent: int = 2) -> None:
    """Write *data* as pretty JSON to *path*, creating directories as needed."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=indent)


# -- CSV Session Log -----------------------------------------------------------

_CSV_FIELDNAMES = [
    "timestamp", "frame_index", "slot_id", "slot_type",
    "is_occupied", "occupancy_confidence",
    "has_violation", "violation_type",
]


def append_to_session_log(
    log_path: str,
    frame_index: int,
    slot_dicts: List[Dict],
    violation_dicts: Optional[List[Dict]] = None,
) -> None:
    """
    Append per-slot records for one frame to the CSV session log.
    Creates the file (with header) if it does not exist.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    write_header = not os.path.exists(log_path)

    violation_map: Dict[int, str] = {}
    if violation_dicts:
        for v in violation_dicts:
            violation_map[v["slot_id"]] = v["violation_type"]

    with open(log_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDNAMES)
        if write_header:
            writer.writeheader()
        for slot in slot_dicts:
            sid = slot["slot_id"]
            writer.writerow(
                {
                    "timestamp": slot.get("timestamp", now_str()),
                    "frame_index": frame_index,
                    "slot_id": sid,
                    "slot_type": slot.get("slot_type", "regular"),
                    "is_occupied": slot.get("is_occupied", False),
                    "occupancy_confidence": slot.get("occupancy_confidence", 0.0),
                    "has_violation": sid in violation_map,
                    "violation_type": violation_map.get(sid, ""),
                }
            )


# -- Geometry Helpers ----------------------------------------------------------

def compute_iou(
    box_a: Tuple[int, int, int, int],
    box_b: Tuple[int, int, int, int],
) -> float:
    """
    Compute Intersection-over-Union (IoU) for two axis-aligned bounding boxes.

    Each box is (x1, y1, x2, y2).
    Returns a float in [0, 1].
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union_area = area_a + area_b - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area


def xywh_to_xyxy(x: int, y: int, w: int, h: int) -> Tuple[int, int, int, int]:
    """Convert (x, y, width, height) to (x1, y1, x2, y2)."""
    return (x, y, x + w, y + h)


# -- Input Validation ----------------------------------------------------------

def validate_image_path(path: str) -> None:
    """Raise ValueError with a helpful message if *path* is not a valid image path."""
    if not isinstance(path, str) or not path.strip():
        raise ValueError("Image path must be a non-empty string.")
    ext = os.path.splitext(path)[1].lower()
    if ext not in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}:
        raise ValueError(
            f"Unsupported image extension '{ext}'. "
            "Supported: .jpg, .jpeg, .png, .bmp, .tiff"
        )


def validate_slots_data(slots_data: Any) -> None:
    """
    Validate the top-level structure of a slots JSON object.

    Raises ValueError with a descriptive message on any structural problem.
    """
    if not isinstance(slots_data, dict):
        raise ValueError("slots.json must be a JSON object at the top level.")
    if "slots" not in slots_data:
        raise ValueError("slots.json must contain a 'slots' key.")
    if not isinstance(slots_data["slots"], list):
        raise ValueError("'slots' must be a JSON array.")
    required_keys = {"slot_id", "x", "y", "width", "height"}
    for i, slot in enumerate(slots_data["slots"]):
        missing = required_keys - set(slot.keys())
        if missing:
            raise ValueError(
                f"Slot at index {i} is missing required keys: {missing}"
            )
