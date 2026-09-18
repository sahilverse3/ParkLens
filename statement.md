# ParkLens — Problem Statement

## Title
**ParkLens: Parking Lot Occupancy & Violation Detector**

## Domain
Computer Vision · Image Processing · Smart City Systems

---

## Background

Urban parking management is a significant operational challenge. Manual monitoring of parking lots is labour-intensive and error-prone. Real-time or near-real-time automated detection of parking slot status and rule violations can:

- Reduce traffic congestion caused by vehicles searching for parking.
- Enforce parking regulations without constant human supervision.
- Provide data-driven insights for parking lot management.

---

## Problem Definition

Given an overhead or CCTV image of a parking lot with pre-defined slot boundaries:

1. **Determine the occupancy state** (FREE or OCCUPIED) of every parking slot.
2. **Detect parking violations** — specifically:
   - Vehicles parked in no-parking / restricted zones.
   - Vehicles occupying more than one slot (double parking).
   - Oversized vehicles exceeding the designated slot dimensions.
3. **Produce analytics** — aggregate slot-level data into summary statistics and visualisations to assist parking administrators.

---

## Objectives

| # | Objective |
|---|---|
| O1 | Implement an image-processing pipeline that classifies each slot as FREE or OCCUPIED with a configurable confidence threshold. |
| O2 | Implement rule-based parking violation detection covering at least three distinct violation types. |
| O3 | Generate per-session analytics including occupancy percentage, violation counts, and chart outputs. |
| O4 | Provide a clean CLI interface that allows execution with a single command on any standard machine. |
| O5 | Write comprehensive unit tests for each module using synthetic data, achieving zero external test dependencies. |

---

## Scope

### In Scope
- Single static image analysis (primary use case).
- Pre-defined parking slot boundaries loaded from a JSON configuration file.
- Three violation types: no-parking zone, double parking, oversized vehicle.
- Designed for Python 3.10+ and verified on Windows with Python 3.13.5.
- Output: annotated image, CSV session log, PNG charts, text report.

### Out of Scope
- Live RTSP/webcam video stream processing.
- Deep-learning-based vehicle or licence-plate recognition.
- Cloud deployment, REST APIs, or web dashboards.
- Multi-camera or multi-lot management systems.

---

## Functional Requirements

| FR | Requirement |
|---|---|
| FR-1 | The system shall accept an input image path and a slot-definition JSON file via CLI arguments. |
| FR-2 | The system shall classify each defined parking slot as FREE or OCCUPIED with an associated confidence score. |
| FR-3 | The system shall detect and record no-parking zone violations, double-parking violations, and oversized-vehicle violations. |
| FR-4 | The system shall write an annotated output image with colour-coded slot boundaries and violation overlays. |
| FR-5 | The system shall write a per-session CSV log recording slot status and violation data. |
| FR-6 | The system shall generate an occupancy pie chart, a violation bar chart, and a plain-text analytics report. |

---

## Non-Functional Requirements

| NFR | Requirement |
|---|---|
| NFR-1 | **Modularity** — Each of the three processing modules (Occupancy, Violation, Analytics) shall be independently importable and testable without depending on the others. |
| NFR-2 | **Configurability** — All detection thresholds and paths shall be centralised in `config/settings.py`; no magic numbers shall appear in module code. |
| NFR-3 | **Robustness** — The system shall validate all inputs (image path, slot JSON schema, CSV existence) and raise descriptive exceptions rather than crashing silently. |
| NFR-4 | **Testability** — 40 automated unit tests covering the core modules and validation scenarios. |
| NFR-5 | **Portability** — Designed for Python 3.10+ and verified on Windows with Python 3.13.5. |
| NFR-6 | **Performance** — Single-image analysis (6 slots) shall complete in under 5 seconds on any modern consumer laptop. |

---

## Inputs and Outputs

### Inputs
| Input | Format | Example |
|---|---|---|
| Parking lot image | JPG / PNG | `data/sample_parking.jpg` |
| Slot definitions | JSON | `data/slots.json` |
| CLI flags | argparse | `--threshold 0.4 --no-display` |

### Outputs
| Output | Format | Location |
|---|---|---|
| Annotated image | JPG | `reports/annotated_result.jpg` |
| Session log | CSV | `data/logs/session_log.csv` |
| Occupancy chart | PNG | `reports/occupancy_chart.png` |
| Violation chart | PNG | `reports/violation_chart.png` |
| Text report | TXT | `reports/session_report.txt` |

---

## Technology Justification

| Technology | Justification |
|---|---|
| Python 3.10+ | Widely available, easy to execute on submission machines; strong CV ecosystem |
| OpenCV | Industry-standard for real-time image processing; requires no trained model download |
| NumPy | Numerical backbone for array operations; bundled with OpenCV |
| Matplotlib | Zero-configuration local chart rendering; no browser or server needed |
| pytest | Simple, widely-used test framework; works with `pytest tests/ -v` out of the box |
| JSON / CSV | Human-readable, no DB server required; inspectable with any text editor |

---

*ParkLens — Academic Submission | VITyarthi Project*
