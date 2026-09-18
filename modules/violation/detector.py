"""
modules/violation/detector.py -- Module 2: Parking Violation Detection.

Three rule-based violation checks (no ML required):

  1. No-Parking Zone  -- slot is marked "no_parking" in slots.json AND is occupied.
  2. Double Parking   -- a vehicle contour overlaps TWO or more defined slots
                        with IoU >= OVERLAP_IOU_THRESHOLD.
  3. Oversized Vehicle -- vehicle contour area > slot_area x MAX_VEHICLE_AREA_RATIO.

Input  : BGR frame + list of ParkingSlot objects (already classified by Module 1)
Output : list of ViolationRecord objects
"""

from __future__ import annotations

import itertools
from typing import List, Tuple

import cv2
import numpy as np

from config import settings
from core.models import ParkingSlot, ViolationRecord
from core.utils import compute_iou, now_str, xywh_to_xyxy


class ViolationDetector:
    """
    Detects parking violations from an annotated frame and slot list.

    Parameters
    ----------
    iou_threshold : float
        Minimum IoU between a vehicle bbox and a slot to count as overlap.
    max_area_ratio : float
        Maximum allowed vehicle-area / slot-area ratio before "oversized" flag.
    min_contour_area : int
        Smallest contour (px sq) to be considered a vehicle (noise filter).
    """

    def __init__(
        self,
        iou_threshold: float | None = None,
        max_area_ratio: float | None = None,
        min_contour_area: int | None = None,
    ) -> None:
        self.iou_threshold = iou_threshold or settings.OVERLAP_IOU_THRESHOLD
        self.max_area_ratio = max_area_ratio or settings.MAX_VEHICLE_AREA_RATIO
        self.min_contour_area = min_contour_area or settings.MIN_VEHICLE_CONTOUR_AREA

    # -- Public API ------------------------------------------------------------

    def detect(
        self,
        frame: np.ndarray,
        slots: List[ParkingSlot],
    ) -> List[ViolationRecord]:
        """
        Run all three violation checks against *frame* and *slots*.

        Parameters
        ----------
        frame : np.ndarray
            Original BGR frame (same one passed to OccupancyDetector).
        slots : list of ParkingSlot
            Slots already classified by Module 1 (is_occupied set).

        Returns
        -------
        list of ViolationRecord
            May be empty if no violations are found.
        """
        violations: List[ViolationRecord] = []
        vehicle_bboxes = self._find_vehicle_bboxes(frame)
        ts = now_str()
        vid = itertools.count(1)

        # Check 1: No-Parking Zone
        for slot in slots:
            if slot.slot_type == "no_parking" and slot.is_occupied:
                # Find the vehicle blob closest to this slot to get a bbox
                bbox = self._closest_bbox_to_slot(slot, vehicle_bboxes) or slot.bbox
                violations.append(
                    ViolationRecord(
                        violation_id=next(vid),
                        slot_id=slot.slot_id,
                        violation_type="no_parking_zone",
                        timestamp=ts,
                        bbox=bbox,
                        description=f"Slot #{slot.slot_id} is a no-parking zone.",
                    )
                )

        # Check 2 & 3: Double Parking / Oversized Vehicle (vehicle-centric)
        for vbbox in vehicle_bboxes:
            overlapping_slots = self._overlapping_slots(vbbox, slots)

            # Check 2: double parking (overlaps ?2 regular/occupied slots)
            occupied_overlaps = [
                s for s in overlapping_slots
                if s.is_occupied and s.slot_type == "regular"
            ]
            if len(occupied_overlaps) >= 2:
                # Report against the first slot
                violations.append(
                    ViolationRecord(
                        violation_id=next(vid),
                        slot_id=occupied_overlaps[0].slot_id,
                        violation_type="double_parking",
                        timestamp=ts,
                        bbox=vbbox,
                        description=(
                            f"Vehicle spans slots "
                            f"{[s.slot_id for s in occupied_overlaps]}."
                        ),
                    )
                )

            # Check 3: oversized vehicle
            if overlapping_slots:
                primary = overlapping_slots[0]
                vx1, vy1, vx2, vy2 = vbbox
                vehicle_area = max(0, vx2 - vx1) * max(0, vy2 - vy1)
                if primary.area > 0 and vehicle_area > primary.area * self.max_area_ratio:
                    violations.append(
                        ViolationRecord(
                            violation_id=next(vid),
                            slot_id=primary.slot_id,
                            violation_type="oversized_vehicle",
                            timestamp=ts,
                            bbox=vbbox,
                            description=(
                                f"Vehicle area {vehicle_area}px sq exceeds "
                                f"{self.max_area_ratio}x slot area {primary.area}px sq."
                            ),
                        )
                    )

        return violations

    # -- Private helpers -------------------------------------------------------

    def _find_vehicle_bboxes(
        self, frame: np.ndarray
    ) -> List[Tuple[int, int, int, int]]:
        """
        Detect vehicle blobs in *frame* using edge + contour analysis.

        Returns a list of (x1, y1, x2, y2) bounding boxes for each blob
        whose area >= min_contour_area.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, settings.BLUR_KERNEL_SIZE, 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Dilate edges to connect nearby fragments of the same vehicle
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=2)

        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        bboxes: List[Tuple[int, int, int, int]] = []
        for cnt in contours:
            if cv2.contourArea(cnt) < self.min_contour_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            bboxes.append(xywh_to_xyxy(x, y, w, h))
        return bboxes

    def _overlapping_slots(
        self,
        vbbox: Tuple[int, int, int, int],
        slots: List[ParkingSlot],
    ) -> List[ParkingSlot]:
        """Return all slots whose IoU with *vbbox* >= iou_threshold."""
        result = []
        for slot in slots:
            iou = compute_iou(vbbox, slot.bbox)
            if iou >= self.iou_threshold:
                result.append(slot)
        return result

    @staticmethod
    def _closest_bbox_to_slot(
        slot: ParkingSlot,
        bboxes: List[Tuple[int, int, int, int]],
    ) -> Tuple[int, int, int, int] | None:
        """
        Return the vehicle bbox whose centre is closest to the slot centre.
        Returns None if *bboxes* is empty.
        """
        if not bboxes:
            return None
        sc_x = slot.x + slot.width // 2
        sc_y = slot.y + slot.height // 2

        def _dist(b: Tuple[int, int, int, int]) -> float:
            bx = (b[0] + b[2]) // 2
            by = (b[1] + b[3]) // 2
            return float((bx - sc_x) ** 2 + (by - sc_y) ** 2)

        return min(bboxes, key=_dist)
