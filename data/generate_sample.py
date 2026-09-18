"""
data/generate_sample.py -- Generates a synthetic sample parking lot image.

Run this once to create data/sample_parking.jpg before running the main pipeline.
The image simulates a top-down view of 6 parking slots, with some slots occupied
(dark rectangles = vehicles) and one deliberately placed in a no-parking zone.

Usage:
    python data/generate_sample.py
"""

import os
import sys

import cv2
import numpy as np

# Ensure project root is on the path when run as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "sample_parking.jpg")


def generate() -> np.ndarray:
    """Create and return a synthetic parking lot image (BGR, 640x480)."""
    img = np.full((480, 640, 3), fill_value=80, dtype=np.uint8)   # Dark-grey tarmac

    # -- Draw lane markings ----------------------------------------------------
    cv2.line(img, (0, 240), (640, 240), (180, 180, 180), 2)        # Centre lane
    cv2.line(img, (0, 0),   (640, 0),   (180, 180, 180), 2)
    cv2.line(img, (0, 479), (640, 479), (180, 180, 180), 2)

    # -- Slot definitions (x, y, w, h, label, is_occupied) --------------------
    slot_defs = [
        #  x    y    w    h   slot   occupied
        ( 20,  20, 90, 200, "S1",   False),   # FREE
        (130,  20, 90, 200, "S2",   True),    # OCCUPIED
        (240,  20, 90, 200, "S3",   True),    # OCCUPIED
        (350,  20, 90, 200, "S4",   False),   # FREE (no-parking zone label added below)
        (460,  20, 90, 200, "S5",   True),    # OCCUPIED
        (550,  20, 90, 200, "S6",   False),   # FREE
    ]

    for x, y, w, h, label, occupied in slot_defs:
        # Slot outline
        cv2.rectangle(img, (x, y), (x + w, y + h), (200, 200, 200), 2)
        # Slot label
        cv2.putText(img, label, (x + 5, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
        if occupied:
            # Draw a dark "vehicle" block inside the slot
            pad = 8
            vehicle_color = (50 + np.random.randint(0, 40),
                             50 + np.random.randint(0, 40),
                             50 + np.random.randint(0, 40))
            cv2.rectangle(
                img,
                (x + pad, y + pad),
                (x + w - pad, y + h - pad),
                vehicle_color, -1
            )
            # Add a subtle highlight line to make it look more like a car
            cv2.line(img, (x + w // 2, y + pad), (x + w // 2, y + h - pad),
                     (100, 100, 100), 2)

    # -- No-parking zone hatch on S4 -------------------------------------------
    nx, ny, nw, nh = 350, 20, 90, 200
    for i in range(ny, ny + nh, 15):
        cv2.line(img, (nx, i), (nx + nw, i + 15), (0, 0, 200), 1)
    cv2.putText(img, "NO PARK", (nx + 2, ny + nh - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    # -- Second row (bottom half) -- a double-parking scenario -----------------
    # Single large vehicle blob that spans two slots
    cv2.rectangle(img, ( 20, 260), (640, 460), (70, 70, 70), -1)   # tarmac row
    cv2.rectangle(img, ( 30, 270), (310, 450), (180, 180, 180), 2) # slot A outline
    cv2.rectangle(img, (320, 270), (600, 450), (180, 180, 180), 2) # slot B outline
    # Big vehicle spanning both slots
    cv2.rectangle(img, ( 50, 285), (580, 435), (40, 40, 60), -1)   # vehicle fill
    cv2.line(img, (310, 285), (310, 435), (80, 80, 80), 2)         # centre line

    cv2.putText(img, "ParkLens -- Sample Parking Lot",
                (150, 478), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

    return img


if __name__ == "__main__":
    image = generate()
    cv2.imwrite(OUTPUT_PATH, image)
    print(f"[generate_sample] Saved synthetic parking image -> {OUTPUT_PATH}")
