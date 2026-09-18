"""
modules/occupancy/slot_loader.py -- Load and validate parking slot definitions.

Slot definitions are stored in data/slots.json.
This module is the only place that reads that file, keeping I/O concerns
separate from detection logic.
"""

from __future__ import annotations

from typing import List

from core.models import ParkingSlot
from core.utils import load_json, validate_slots_data


def load_slots(slots_file: str) -> List[ParkingSlot]:
    """
    Parse *slots_file* (JSON) and return a list of :class:`ParkingSlot` objects.

    The JSON schema expected::

        {
          "lot_name": "Main Lot",
          "slots": [
            {
              "slot_id": 1,
              "x": 10, "y": 20,
              "width": 80, "height": 120,
              "slot_type": "regular"      // optional; default "regular"
            },
            ...
          ]
        }

    Raises:
        FileNotFoundError -- if *slots_file* does not exist.
        ValueError        -- if the JSON is malformed or missing required keys.
    """
    raw = load_json(slots_file)
    validate_slots_data(raw)

    slots: List[ParkingSlot] = []
    for entry in raw["slots"]:
        slot = ParkingSlot(
            slot_id=int(entry["slot_id"]),
            x=int(entry["x"]),
            y=int(entry["y"]),
            width=int(entry["width"]),
            height=int(entry["height"]),
            slot_type=entry.get("slot_type", "regular"),
        )
        slots.append(slot)

    if not slots:
        raise ValueError("slots.json contains an empty 'slots' array.")

    return slots
