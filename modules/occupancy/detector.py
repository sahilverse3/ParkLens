"""
modules/occupancy/detector.py -- Module 1: Parking Slot Occupancy Detection.

Algorithm
---------
For every defined slot, we crop its Region of Interest (ROI) from the
input frame, then classify it as OCCUPIED or FREE using the following steps:

  1. Convert ROI to grayscale.
  2. Apply Gaussian blur to reduce sensor noise.
  3. Apply adaptive thresholding to obtain a binary foreground mask.
  4. Compute the ratio  non_zero_pixels / total_pixels.
  5. If ratio > OCCUPIED_THRESHOLD  ->  OCCUPIED  (else FREE).

This threshold-based approach needs no pre-trained model and works
reliably on clear overhead/CCTV parking lot images.

Input  : BGR image (np.ndarray) + list of ParkingSlot definitions
Output : Same list of ParkingSlot objects with is_occupied & confidence filled
"""

from __future__ import annotations

from typing import List

import cv2
import numpy as np

from config import settings
from core.models import ParkingSlot
from core.utils import now_str


class OccupancyDetector:
    """
    Classifies each parking slot in a frame as FREE or OCCUPIED.

    Parameters
    ----------
    occupied_threshold : float
        Non-zero pixel ratio above which a slot is called OCCUPIED.
        Defaults to ``settings.OCCUPIED_THRESHOLD``.
    """

    def __init__(self, occupied_threshold: float | None = None) -> None:
        self.occupied_threshold = (
            occupied_threshold
            if occupied_threshold is not None
            else settings.OCCUPIED_THRESHOLD
        )

    # -- Public API ------------------------------------------------------------

    def detect(
        self,
        frame: np.ndarray,
        slots: List[ParkingSlot],
    ) -> List[ParkingSlot]:
        """
        Run occupancy detection on *frame* for each slot in *slots*.

        Mutates each slot's ``is_occupied``, ``occupancy_confidence``, and
        ``timestamp`` fields in-place, then returns the same list.

        Parameters
        ----------
        frame : np.ndarray
            BGR image from cv2.imread or a video capture frame.
        slots : list of ParkingSlot
            Slot definitions (loaded from slots.json via slot_loader).

        Returns
        -------
        list of ParkingSlot
            The same objects with occupancy fields populated.

        Raises
        ------
        ValueError
            If *frame* is None, empty, or not a NumPy array.
        """
        self._validate_frame(frame)

        ts = now_str()
        for slot in slots:
            roi = self._crop_roi(frame, slot)
            ratio = self._foreground_ratio(roi)
            slot.is_occupied = ratio >= self.occupied_threshold
            slot.occupancy_confidence = round(float(ratio), 4)
            slot.timestamp = ts

        return slots

    # -- Private helpers -------------------------------------------------------

    @staticmethod
    def _validate_frame(frame: np.ndarray) -> None:
        if frame is None:
            raise ValueError("Frame is None. Cannot perform occupancy detection.")
        if not isinstance(frame, np.ndarray):
            raise ValueError(
                f"Frame must be a NumPy ndarray, got {type(frame).__name__}."
            )
        if frame.size == 0:
            raise ValueError("Frame is empty (zero-size array).")

    @staticmethod
    def _crop_roi(frame: np.ndarray, slot: ParkingSlot) -> np.ndarray:
        """
        Crop the slot region from *frame*, clamping to image bounds
        to avoid index errors when a slot definition extends slightly
        beyond the image edge.
        """
        h, w = frame.shape[:2]
        x1 = max(0, slot.x)
        y1 = max(0, slot.y)
        x2 = min(w, slot.x + slot.width)
        y2 = min(h, slot.y + slot.height)

        if x2 <= x1 or y2 <= y1:
            # Slot entirely outside frame -- treat as free
            return np.zeros((1, 1, 3), dtype=np.uint8)

        return frame[y1:y2, x1:x2]

    @staticmethod
    def _foreground_ratio(roi: np.ndarray) -> float:
        """
        Return the fraction of ROI pixels classified as foreground
        (i.e. not background / not empty asphalt).

        Steps:
          grayscale -> Gaussian blur -> adaptive threshold -> count non-zero
        """
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, settings.BLUR_KERNEL_SIZE, 0)
        binary = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            settings.ADAPTIVE_BLOCK_SIZE,
            settings.ADAPTIVE_C,
        )
        total_pixels = binary.size
        if total_pixels == 0:
            return 0.0
        return float(cv2.countNonZero(binary)) / total_pixels
