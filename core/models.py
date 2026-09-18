"""
core/models.py -- Shared data-classes used across all three modules.

Keeping models in one place prevents circular imports and makes the
data contract between modules explicit.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import datetime


@dataclass
class ParkingSlot:
    """Represents one defined parking slot in the lot."""

    slot_id: int                          # Unique slot number (1-based)
    x: int                                # Top-left x of bounding rectangle
    y: int                                # Top-left y
    width: int
    height: int
    slot_type: str = "regular"            # "regular" | "no_parking" | "disabled"

    # Fields filled in by OccupancyDetector
    is_occupied: bool = False
    occupancy_confidence: float = 0.0     # 0.0 - 1.0
    timestamp: Optional[str] = None

    # -- Convenience helpers ---------------------------------------------------
    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def rect(self) -> Tuple[int, int, int, int]:
        """Return (x, y, w, h) tuple -- compatible with cv2 drawing calls."""
        return (self.x, self.y, self.width, self.height)

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        """Return (x1, y1, x2, y2) axis-aligned bbox."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)

    def to_dict(self) -> dict:
        return {
            "slot_id": self.slot_id,
            "x": self.x, "y": self.y,
            "width": self.width, "height": self.height,
            "slot_type": self.slot_type,
            "is_occupied": self.is_occupied,
            "occupancy_confidence": round(self.occupancy_confidence, 4),
            "timestamp": self.timestamp,
        }


@dataclass
class ViolationRecord:
    """Represents a single detected parking violation."""

    violation_id: int
    slot_id: int                          # Slot where violation occurred
    violation_type: str                   # "no_parking_zone" | "double_parking" | "oversized_vehicle"
    timestamp: str
    bbox: Tuple[int, int, int, int]       # (x1, y1, x2, y2) of offending vehicle
    confidence: float = 1.0              # Rule-based = always 1.0; extensible for ML

    # Human-readable description
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "violation_id": self.violation_id,
            "slot_id": self.slot_id,
            "violation_type": self.violation_type,
            "timestamp": self.timestamp,
            "bbox": list(self.bbox),
            "confidence": self.confidence,
            "description": self.description,
        }


@dataclass
class SessionSummary:
    """Aggregated statistics for one analysis session."""

    session_id: str
    image_source: str
    total_slots: int = 0
    occupied_count: int = 0
    free_count: int = 0
    violations: List[ViolationRecord] = field(default_factory=list)

    @property
    def occupancy_pct(self) -> float:
        if self.total_slots == 0:
            return 0.0
        return round(self.occupied_count / self.total_slots * 100, 2)

    @property
    def violation_count(self) -> int:
        return len(self.violations)
