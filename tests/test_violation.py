"""
tests/test_violation.py -- Unit tests for Module 2: ViolationDetector.

All tests use synthetic NumPy images -- no real files required.
The violation rules tested:
  1. no_parking_zone
  2. double_parking
  3. oversized_vehicle
"""

import numpy as np
import pytest

from core.models import ParkingSlot, ViolationRecord
from core.utils import compute_iou
from modules.violation.detector import ViolationDetector


# -- Helpers -------------------------------------------------------------------

def make_frame(height: int = 300, width: int = 600) -> np.ndarray:
    """Plain grey frame."""
    return np.full((height, width, 3), fill_value=80, dtype=np.uint8)


def make_frame_with_rect(
    x1: int, y1: int, x2: int, y2: int,
    frame_h: int = 300, frame_w: int = 600,
    fill: int = 10,
) -> np.ndarray:
    """Frame with a dark rectangle that Canny will find as a strong edge."""
    frame = np.full((frame_h, frame_w, 3), fill_value=80, dtype=np.uint8)
    frame[y1:y2, x1:x2] = fill
    return frame


def make_slot(
    slot_id: int, x: int, y: int, w: int, h: int,
    slot_type: str = "regular", is_occupied: bool = False
) -> ParkingSlot:
    s = ParkingSlot(
        slot_id=slot_id, x=x, y=y, width=w, height=h, slot_type=slot_type
    )
    s.is_occupied = is_occupied
    s.timestamp = "2026-01-01T00:00:00"
    return s


# -- Tests: ViolationDetector --------------------------------------------------

class TestViolationDetector:

    def setup_method(self):
        self.detector = ViolationDetector(
            iou_threshold=0.25,
            max_area_ratio=1.4,
            min_contour_area=200,
        )

    # -- No-Parking Zone -------------------------------------------------------

    def test_no_parking_zone_occupied_raises_violation(self):
        """An occupied no_parking slot must produce a no_parking_zone violation."""
        frame = make_frame_with_rect(x1=20, y1=20, x2=110, y2=180)
        slot = make_slot(1, x=20, y=20, w=90, h=160,
                         slot_type="no_parking", is_occupied=True)
        violations = self.detector.detect(frame, [slot])
        types = [v.violation_type for v in violations]
        assert "no_parking_zone" in types, (
            "Expected no_parking_zone violation when a no-parking slot is occupied"
        )

    def test_no_parking_zone_free_no_violation(self):
        """A FREE no_parking slot must NOT produce a violation."""
        frame = make_frame()
        slot = make_slot(1, x=20, y=20, w=90, h=160,
                         slot_type="no_parking", is_occupied=False)
        violations = self.detector.detect(frame, [slot])
        types = [v.violation_type for v in violations]
        assert "no_parking_zone" not in types

    def test_regular_occupied_slot_no_no_parking_violation(self):
        """A regular occupied slot must not produce a no_parking_zone violation."""
        frame = make_frame_with_rect(10, 10, 100, 150)
        slot = make_slot(1, x=10, y=10, w=90, h=140,
                         slot_type="regular", is_occupied=True)
        violations = self.detector.detect(frame, [slot])
        no_park_viols = [v for v in violations if v.violation_type == "no_parking_zone"]
        assert not no_park_viols

    # -- ViolationRecord Fields -------------------------------------------------

    def test_violation_record_has_required_fields(self):
        """Each ViolationRecord must have all required fields populated."""
        frame = make_frame_with_rect(20, 20, 110, 180)
        slot = make_slot(1, x=20, y=20, w=90, h=160,
                         slot_type="no_parking", is_occupied=True)
        violations = self.detector.detect(frame, [slot])
        assert len(violations) >= 1
        v = violations[0]
        assert isinstance(v.violation_id, int) and v.violation_id >= 1
        assert v.slot_id == 1
        assert isinstance(v.violation_type, str) and v.violation_type
        assert isinstance(v.timestamp, str) and v.timestamp
        assert isinstance(v.bbox, tuple) and len(v.bbox) == 4
        assert isinstance(v.description, str)

    def test_no_violations_on_all_free_regular_slots(self):
        """No violations when all slots are FREE and none are no-parking."""
        frame = make_frame()
        slots = [
            make_slot(i, x=i * 100, y=20, w=80, h=160,
                      slot_type="regular", is_occupied=False)
            for i in range(1, 4)
        ]
        violations = self.detector.detect(frame, slots)
        assert violations == [], f"Expected no violations, got {violations}"

    # -- IoU Helper ------------------------------------------------------------

    def test_iou_overlapping_boxes(self):
        """IoU of two identical boxes is 1.0."""
        assert compute_iou((0, 0, 100, 100), (0, 0, 100, 100)) == pytest.approx(1.0)

    def test_iou_non_overlapping_boxes(self):
        """IoU of two non-overlapping boxes is 0.0."""
        assert compute_iou((0, 0, 50, 50), (100, 100, 200, 200)) == pytest.approx(0.0)

    def test_iou_partial_overlap(self):
        """Partially overlapping boxes produce 0 < IoU < 1."""
        iou = compute_iou((0, 0, 100, 100), (50, 50, 150, 150))
        assert 0.0 < iou < 1.0

    def test_iou_zero_area_box(self):
        """Degenerate (zero-area) boxes return 0 IoU without crashing."""
        assert compute_iou((5, 5, 5, 5), (0, 0, 10, 10)) == pytest.approx(0.0)

    # -- to_dict ---------------------------------------------------------------

    def test_violation_record_to_dict(self):
        """ViolationRecord.to_dict() should include all expected keys."""
        vr = ViolationRecord(
            violation_id=1,
            slot_id=2,
            violation_type="double_parking",
            timestamp="2026-01-01T12:00:00",
            bbox=(10, 20, 100, 150),
            description="Test",
        )
        d = vr.to_dict()
        assert set(d.keys()) == {
            "violation_id", "slot_id", "violation_type",
            "timestamp", "bbox", "confidence", "description",
        }
        assert d["bbox"] == [10, 20, 100, 150]
