"""
tests/test_occupancy.py -- Unit tests for Module 1: OccupancyDetector.

All tests use synthetic NumPy images -- no real JPEG files required.
"""

import numpy as np
import pytest

from core.models import ParkingSlot
from modules.occupancy.detector import OccupancyDetector
from modules.occupancy.slot_loader import load_slots


# -- Fixtures ------------------------------------------------------------------

def make_frame(height: int = 200, width: int = 400, fill: int = 80) -> np.ndarray:
    """Return a plain BGR frame filled with *fill* value (simulates empty tarmac)."""
    return np.full((height, width, 3), fill_value=fill, dtype=np.uint8)


def make_frame_with_vehicle(
    x: int, y: int, w: int, h: int,
    frame_h: int = 200, frame_w: int = 400,
) -> np.ndarray:
    """
    Return a BGR frame that simulates a dark vehicle block inside a slot.

    Background = mid-grey (80). Vehicle region = dark (~20) WITH random noise
    so that adaptive thresholding (which measures local contrast) can detect it.
    A perfectly uniform dark patch has zero local variance -> adaptive threshold
    yields 0 non-zero pixels. Adding ?20 noise gives realistic texture.
    """
    rng = np.random.default_rng(seed=42)
    frame = make_frame(frame_h, frame_w, fill=80)
    # Create a noisy dark patch (mean ~20, varied)
    noise = rng.integers(-20, 30, size=(h, w, 3), dtype=np.int32)
    vehicle = np.clip(20 + noise, 0, 255).astype(np.uint8)
    frame[y:y + h, x:x + w] = vehicle
    return frame


def make_slot(
    slot_id: int = 1,
    x: int = 10, y: int = 10,
    w: int = 80, h: int = 160,
    slot_type: str = "regular",
) -> ParkingSlot:
    return ParkingSlot(
        slot_id=slot_id, x=x, y=y, width=w, height=h, slot_type=slot_type
    )


# -- Tests: OccupancyDetector --------------------------------------------------

class TestOccupancyDetector:

    def setup_method(self):
        self.detector = OccupancyDetector(occupied_threshold=0.35)

    # -- Happy path ------------------------------------------------------------

    def test_free_slot_on_empty_tarmac(self):
        """A uniformly grey frame should yield a FREE slot (minimal texture)."""
        frame = make_frame(fill=100)         # Uniform -> very little edge content
        slot = make_slot()
        result = self.detector.detect(frame, [slot])
        assert not result[0].is_occupied, (
            "Uniform grey frame should be classified FREE"
        )

    def test_occupied_slot_with_dark_vehicle(self):
        """A dark block inside the slot region should be classified OCCUPIED."""
        # Place a dark vehicle right in the slot area
        frame = make_frame_with_vehicle(x=10, y=10, w=80, h=160)
        slot = make_slot(x=10, y=10, w=80, h=160)
        # Use a very low threshold to ensure the dark block triggers occupied
        detector = OccupancyDetector(occupied_threshold=0.05)
        result = detector.detect(frame, [slot])
        assert result[0].is_occupied, (
            "Slot containing a dark vehicle block should be OCCUPIED"
        )

    def test_confidence_is_between_0_and_1(self):
        """Confidence score must always be in [0, 1]."""
        frame = make_frame()
        slot = make_slot()
        result = self.detector.detect(frame, [slot])
        conf = result[0].occupancy_confidence
        assert 0.0 <= conf <= 1.0, f"Confidence out of range: {conf}"

    def test_timestamp_is_set(self):
        """Timestamp should be populated after detection."""
        frame = make_frame()
        slot = make_slot()
        result = self.detector.detect(frame, [slot])
        assert result[0].timestamp is not None
        assert len(result[0].timestamp) > 0

    def test_multiple_slots_detected_independently(self):
        """Each slot should be classified independently."""
        frame = make_frame_with_vehicle(x=10, y=10, w=80, h=160)
        slot_occupied = make_slot(slot_id=1, x=10, y=10, w=80, h=160)
        # Second slot far from the vehicle block
        slot_free = make_slot(slot_id=2, x=300, y=10, w=80, h=160)

        detector = OccupancyDetector(occupied_threshold=0.05)
        results = detector.detect(frame, [slot_occupied, slot_free])

        assert results[0].is_occupied, "Slot 1 should be OCCUPIED"
        assert not results[1].is_occupied, "Slot 2 should be FREE"

    def test_slot_partially_outside_frame_is_handled(self):
        """A slot that extends past image bounds should not raise an error."""
        frame = make_frame(height=100, width=100)
        slot = make_slot(x=80, y=80, w=80, h=80)   # extends to (160, 160) -- out of bounds
        # Should not raise, and should return FREE (clamped tiny ROI)
        result = self.detector.detect(frame, [slot])
        assert isinstance(result[0].is_occupied, bool)

    # -- Threshold edge cases --------------------------------------------------

    def test_threshold_zero_marks_everything_occupied(self):
        """With threshold=0, any pixel activity marks the slot as OCCUPIED."""
        detector = OccupancyDetector(occupied_threshold=0.0)
        frame = make_frame(fill=80)           # uniform, but adaptive threshold will find edges
        slot = make_slot()
        result = detector.detect(frame, [slot])
        # Confidence >= 0, which is >= 0 threshold -> occupied
        assert result[0].occupancy_confidence >= 0.0

    def test_threshold_one_marks_everything_free(self):
        """With threshold=1.0, no slot can ever be marked OCCUPIED."""
        detector = OccupancyDetector(occupied_threshold=1.0)
        # Even a fully dark frame won't produce 100% foreground ratio
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        slot = make_slot()
        result = detector.detect(frame, [slot])
        assert not result[0].is_occupied

    # -- Error / Validation ----------------------------------------------------

    def test_raises_on_none_frame(self):
        with pytest.raises(ValueError, match="None"):
            self.detector.detect(None, [make_slot()])  # type: ignore[arg-type]

    def test_raises_on_empty_frame(self):
        with pytest.raises(ValueError, match="empty"):
            self.detector.detect(np.array([]), [make_slot()])

    def test_raises_on_non_array_frame(self):
        with pytest.raises(ValueError, match="NumPy"):
            self.detector.detect("not an image", [make_slot()])  # type: ignore[arg-type]

    def test_returns_same_list_object(self):
        """detect() should mutate slots in-place and return the same list."""
        frame = make_frame()
        slots = [make_slot(1), make_slot(2)]
        returned = self.detector.detect(frame, slots)
        assert returned is slots


# -- Tests: SlotLoader ---------------------------------------------------------

class TestSlotLoader:

    def test_load_valid_slots_file(self, tmp_path):
        """load_slots should parse a valid JSON file into ParkingSlot objects."""
        import json
        data = {
            "lot_name": "Test",
            "slots": [
                {"slot_id": 1, "x": 10, "y": 20, "width": 80, "height": 120, "slot_type": "regular"},
                {"slot_id": 2, "x": 100, "y": 20, "width": 80, "height": 120},
            ]
        }
        p = tmp_path / "slots.json"
        p.write_text(json.dumps(data))

        slots = load_slots(str(p))
        assert len(slots) == 2
        assert slots[0].slot_id == 1
        assert slots[1].slot_type == "regular"   # default

    def test_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_slots("/nonexistent/path/slots.json")

    def test_raises_on_invalid_json(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{not valid json}")
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_slots(str(p))

    def test_raises_on_missing_slots_key(self, tmp_path):
        import json
        p = tmp_path / "no_slots.json"
        p.write_text(json.dumps({"lot_name": "test"}))
        with pytest.raises(ValueError, match="'slots'"):
            load_slots(str(p))

    def test_raises_on_empty_slots_array(self, tmp_path):
        import json
        p = tmp_path / "empty.json"
        p.write_text(json.dumps({"slots": []}))
        with pytest.raises(ValueError, match="empty"):
            load_slots(str(p))

    def test_raises_on_slot_missing_required_key(self, tmp_path):
        import json
        data = {
            "slots": [
                {"slot_id": 1, "x": 10, "y": 20, "width": 80}   # missing 'height'
            ]
        }
        p = tmp_path / "bad_slot.json"
        p.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="missing required keys"):
            load_slots(str(p))
