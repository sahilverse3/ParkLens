"""
ParkLens -- Central Configuration
All tuneable constants live here. Adjust thresholds to match your parking lot.
"""

import os

# -- Paths ---------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data")
LOGS_DIR   = os.path.join(DATA_DIR, "logs")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

DEFAULT_SLOTS_FILE  = os.path.join(DATA_DIR, "slots.json")
DEFAULT_IMAGE_FILE  = os.path.join(DATA_DIR, "sample_parking.jpg")
SESSION_LOG_CSV     = os.path.join(LOGS_DIR, "session_log.csv")

# -- Occupancy Detection -------------------------------------------------------
# Fraction of ROI pixels that must be "foreground" to call a slot OCCUPIED.
OCCUPIED_THRESHOLD = 0.35

# GaussianBlur kernel size (must be odd)
BLUR_KERNEL_SIZE = (5, 5)

# Adaptive threshold block size (must be odd) and constant C
ADAPTIVE_BLOCK_SIZE = 11
ADAPTIVE_C          = 2

# -- Violation Detection -------------------------------------------------------
# A vehicle bounding box is considered to overlap a slot if IoU >= this value.
OVERLAP_IOU_THRESHOLD = 0.25

# A vehicle blob whose area > slot_area * this ratio is flagged "oversized".
MAX_VEHICLE_AREA_RATIO = 1.4

# Minimum contour area (px sq) to be considered a vehicle (filters noise).
MIN_VEHICLE_CONTOUR_AREA = 500

# -- Visualizer ----------------------------------------------------------------
COLOR_FREE     = (0, 255, 0)    # Green -- free slot
COLOR_OCCUPIED = (0, 0, 255)    # Red   -- occupied slot
COLOR_VIOLATION = (0, 165, 255) # Orange -- violation overlay
COLOR_TEXT     = (255, 255, 255)
FONT_SCALE     = 0.5
FONT_THICKNESS = 1

# -- Analytics -----------------------------------------------------------------
CHART_DPI    = 100
CHART_STYLE  = "seaborn-v0_8-whitegrid"   # fallback: "ggplot"
