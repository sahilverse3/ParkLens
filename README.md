# ParkLens — Parking Lot Occupancy & Violation Detector

> **Academic project** — A modular Python application that analyses a parking lot image,
> classifies each slot as FREE or OCCUPIED, detects three types of parking violations,
> and produces charts and a text analytics report — all locally, with no cloud or database.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Modules](#modules)
3. [Tech Stack](#tech-stack)
4. [Folder Structure](#folder-structure)
5. [Setup & Installation](#setup--installation)
6. [Running the Project](#running-the-project)
7. [Running Tests](#running-tests)
8. [Output Files](#output-files)
9. [Configuration](#configuration)
10. [Architecture](#architecture)

---

## Project Overview

ParkLens is a computer-vision pipeline built with Python and OpenCV that performs:

| # | Module | What it does |
|---|---|---|
| 1 | **Occupancy Detection** | Classifies each pre-defined parking slot as FREE / OCCUPIED using adaptive thresholding |
| 2 | **Violation Detection** | Flags three rule-based violations: *no-parking zone*, *double parking*, *oversized vehicle* |
| 3 | **Analytics Engine** | Reads the session log and produces occupancy pie chart, violation bar chart, and a text summary |

---

## Modules

### Module 1 — Parking Slot Occupancy Detection
**File:** `modules/occupancy/detector.py`

- Loads slot ROI coordinates from `data/slots.json`
- Crops each ROI from the input frame
- Converts to greyscale → Gaussian blur → adaptive threshold
- Counts foreground pixels; if ratio > `OCCUPIED_THRESHOLD` (default `0.35`) → **OCCUPIED**
- Returns `ParkingSlot` objects with `is_occupied`, `occupancy_confidence`, `timestamp`

### Module 2 — Parking Violation Detection
**File:** `modules/violation/detector.py`

| Check | Rule |
|---|---|
| No-Parking Zone | Slot `slot_type == "no_parking"` AND `is_occupied == True` |
| Double Parking | A vehicle contour overlaps ≥ 2 slots with IoU ≥ 0.25 |
| Oversized Vehicle | Vehicle contour area > slot area × 1.4 |

Returns `ViolationRecord` objects with `violation_type`, `bbox`, `description`.

### Module 3 — Parking Analytics
**File:** `modules/analytics/engine.py`

- Reads `data/logs/session_log.csv`
- Computes occupancy % and violation counts
- Writes `reports/occupancy_chart.png`, `reports/violation_chart.png`, `reports/session_report.txt`
- For multi-frame input: also writes `reports/occupancy_trend.png`

---

## Tech Stack

| Component | Library / Version |
|---|---|
| Language | Python 3.10+ |
| Image Processing | opencv-python ≥ 4.8 |
| Numerics | numpy ≥ 1.24 |
| Charts | matplotlib ≥ 3.7 |
| Testing | pytest ≥ 7.4 |

---

## Folder Structure

```
ParkLens/
├── main.py                     ← CLI entry point
├── requirements.txt
├── README.md
├── statement.md
│
├── config/
│   └── settings.py             ← All tuneable constants
│
├── core/
│   ├── models.py               ← ParkingSlot, ViolationRecord, SessionSummary
│   ├── utils.py                ← Image I/O, JSON, CSV, IoU geometry
│   └── visualizer.py          ← OpenCV drawing helpers
│
├── modules/
│   ├── occupancy/
│   │   ├── detector.py         ← OccupancyDetector (Module 1)
│   │   └── slot_loader.py      ← JSON → ParkingSlot parser
│   ├── violation/
│   │   └── detector.py         ← ViolationDetector (Module 2)
│   └── analytics/
│       └── engine.py           ← AnalyticsEngine (Module 3)
│
├── data/
│   ├── slots.json              ← Pre-defined slot coordinates
│   ├── generate_sample.py      ← Generates sample_parking.jpg
│   └── logs/                   ← Auto-created; session CSV logs
│
├── tests/
│   ├── test_occupancy.py
│   ├── test_violation.py
│   └── test_analytics.py
│
└── reports/                    ← Auto-created; charts + text report
```

---

## Setup & Installation

```bash
# 1. Clone / unzip the project
cd ParkLens

# 2. (Recommended) Create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate the sample parking image
python data/generate_sample.py
```

---

## Running the Project

### Analyse the sample image (default)
```bash
python main.py
```

### Specify a custom image and slot file
```bash
python main.py --image path/to/parking.jpg --slots path/to/slots.json
```

### Headless mode (no display window)
```bash
python main.py --no-display
```

### Run analytics on an existing log only
```bash
python main.py --analytics-only
```

### Adjust occupancy threshold
```bash
python main.py --threshold 0.4
```

### Full help
```bash
python main.py --help
```

---

## Running Tests

```bash
pytest tests/ -v
```

Expected output: **all tests pass** (no real images or files required — tests use synthetic NumPy arrays and temporary directories).

---

## Output Files

| File | Description |
|---|---|
| `reports/annotated_result.jpg` | Input image with coloured slot boxes and violation overlays |
| `reports/occupancy_chart.png` | Pie chart — free vs occupied |
| `reports/violation_chart.png` | Bar chart — violations by type |
| `reports/occupancy_trend.png` | Line chart — occupancy % over time (multi-frame only) |
| `reports/session_report.txt` | Plain-text analytics summary |
| `data/logs/session_log.csv` | Raw per-slot, per-frame log |

---

## Configuration

Edit `config/settings.py` to adjust any threshold without touching module code:

| Constant | Default | Meaning |
|---|---|---|
| `OCCUPIED_THRESHOLD` | `0.35` | Foreground pixel ratio to call a slot occupied |
| `OVERLAP_IOU_THRESHOLD` | `0.25` | Minimum IoU to count a vehicle as overlapping a slot |
| `MAX_VEHICLE_AREA_RATIO` | `1.4` | Oversized threshold (vehicle area / slot area) |
| `MIN_VEHICLE_CONTOUR_AREA` | `500` | Minimum blob area (px²) to be considered a vehicle |

---

## Architecture

```
Input Image / Video
        │
        ▼
┌───────────────────┐
│  Module 1         │  OccupancyDetector
│  Slot ROI crop    │  → Adaptive threshold → FREE / OCCUPIED
│  + confidence     │
└────────┬──────────┘
         │  ParkingSlot list (is_occupied set)
         ▼
┌───────────────────┐
│  Module 2         │  ViolationDetector
│  Contour detect   │  → Rule checks → ViolationRecord list
│  + IoU geometry   │
└────────┬──────────┘
         │  Writes session_log.csv
         ▼
┌───────────────────┐
│  Module 3         │  AnalyticsEngine
│  CSV aggregation  │  → Charts + session_report.txt
└───────────────────┘
```

---

