"""
core/visualizer.py -- Drawing helpers for annotated output images.

All OpenCV drawing calls are centralised here so that module code
stays clean and presentation concerns are separated.
"""

from __future__ import annotations

from typing import List

import cv2
import numpy as np

from config import settings
from core.models import ParkingSlot, ViolationRecord


def draw_slots(
    frame: np.ndarray,
    slots: List[ParkingSlot],
    show_confidence: bool = True,
) -> np.ndarray:
    """
    Draw coloured bounding rectangles for every slot on *frame*.

    Green = free, Red = occupied.
    Returns a new annotated copy (the input frame is not mutated).
    """
    out = frame.copy()

    for slot in slots:
        colour = settings.COLOR_OCCUPIED if slot.is_occupied else settings.COLOR_FREE
        x, y, w, h = slot.rect
        cv2.rectangle(out, (x, y), (x + w, y + h), colour, 2)

        label_parts = [f"#{slot.slot_id}"]
        if slot.slot_type != "regular":
            label_parts.append(slot.slot_type.upper()[:3])
        if show_confidence and slot.is_occupied:
            label_parts.append(f"{slot.occupancy_confidence:.0%}")

        label = " ".join(label_parts)
        cv2.putText(
            out, label,
            (x + 3, y + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            settings.FONT_SCALE,
            settings.COLOR_TEXT,
            settings.FONT_THICKNESS,
            cv2.LINE_AA,
        )

    return out


def draw_violations(
    frame: np.ndarray,
    violations: List[ViolationRecord],
) -> np.ndarray:
    """
    Draw orange bounding boxes around vehicles with detected violations.

    Returns a new annotated copy.
    """
    out = frame.copy()

    for viol in violations:
        x1, y1, x2, y2 = viol.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), settings.COLOR_VIOLATION, 3)
        label = f"VIOLATION: {viol.violation_type.replace('_', ' ').title()}"
        cv2.putText(
            out, label,
            (x1 + 3, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            settings.FONT_SCALE,
            settings.COLOR_VIOLATION,
            settings.FONT_THICKNESS,
            cv2.LINE_AA,
        )

    return out


def draw_summary_overlay(
    frame: np.ndarray,
    total: int,
    occupied: int,
    violation_count: int,
) -> np.ndarray:
    """
    Draw a semi-transparent info bar at the top of *frame* showing summary stats.
    Returns a new annotated copy.
    """
    out = frame.copy()
    bar_height = 36

    # Semi-transparent dark rectangle
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (out.shape[1], bar_height), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, out, 0.4, 0, out)

    free = total - occupied
    text = (
        f"  Slots: {total}   "
        f"Free: {free}   "
        f"Occupied: {occupied}   "
        f"Violations: {violation_count}"
    )
    cv2.putText(
        out, text,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return out
