"""
main.py -- ParkLens Entry Point

Orchestrates the three-module pipeline:
  Module 1: OccupancyDetector  -- classifies each slot as FREE / OCCUPIED
  Module 2: ViolationDetector  -- checks for rule-based violations
  Module 3: AnalyticsEngine    -- generates charts and a text report

Usage examples
--------------
  # Analyse a single image
  python main.py --image data/sample_parking.jpg --slots data/slots.json

  # Specify a custom output directory
  python main.py --image data/sample_parking.jpg --slots data/slots.json --out reports/

  # Run only analytics on an existing session log
  python main.py --analytics-only

  # Show help
  python main.py --help
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2

from config import settings
from core import utils, visualizer
from core.models import SessionSummary
from modules.occupancy.detector import OccupancyDetector
from modules.occupancy.slot_loader import load_slots
from modules.violation.detector import ViolationDetector
from modules.analytics.engine import AnalyticsEngine


# -- CLI -----------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="parklens",
        description="ParkLens -- Parking Lot Occupancy & Violation Detector",
    )
    p.add_argument(
        "--image", "-i",
        default=settings.DEFAULT_IMAGE_FILE,
        help="Path to input image (JPG/PNG). Default: data/sample_parking.jpg",
    )
    p.add_argument(
        "--slots", "-s",
        default=settings.DEFAULT_SLOTS_FILE,
        help="Path to slot definitions JSON. Default: data/slots.json",
    )
    p.add_argument(
        "--out", "-o",
        default=settings.REPORTS_DIR,
        help="Directory for output files. Default: reports/",
    )
    p.add_argument(
        "--no-display",
        action="store_true",
        help="Skip OpenCV display window (useful on headless systems).",
    )
    p.add_argument(
        "--analytics-only",
        action="store_true",
        help="Skip detection; run analytics on the existing session log CSV.",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            f"Occupancy threshold (0-1). "
            f"Default: {settings.OCCUPIED_THRESHOLD}"
        ),
    )
    return p


# -- Pipeline ------------------------------------------------------------------

def run_pipeline(args: argparse.Namespace) -> None:
    """Full detection + analytics pipeline for a single image."""

    # -- Validate inputs -------------------------------------------------------
    utils.validate_image_path(args.image)
    frame = utils.load_image(args.image)
    slots = load_slots(args.slots)

    print(f"\n[ParkLens] Image   : {args.image}")
    print(f"[ParkLens] Slots   : {len(slots)} defined")
    print(f"[ParkLens] Output  : {args.out}\n")

    # -- Module 1: Occupancy Detection -----------------------------------------
    occ_detector = OccupancyDetector(occupied_threshold=args.threshold)
    slots = occ_detector.detect(frame, slots)

    occupied = sum(1 for s in slots if s.is_occupied)
    free = len(slots) - occupied
    print(f"[Module 1 -- Occupancy]  Total={len(slots)}  Occupied={occupied}  Free={free}")
    for s in slots:
        status = "OCCUPIED" if s.is_occupied else "FREE    "
        print(f"  Slot #{s.slot_id:>2} [{s.slot_type:<11}]  {status}  "
              f"(confidence={s.occupancy_confidence:.2%})")

    # -- Module 2: Violation Detection -----------------------------------------
    viol_detector = ViolationDetector()
    violations = viol_detector.detect(frame, slots)

    print(f"\n[Module 2 -- Violations]  {len(violations)} violation(s) found")
    for v in violations:
        print(f"  Slot #{v.slot_id}  {v.violation_type:<25}  -- {v.description}")

    # -- Session log -----------------------------------------------------------
    os.makedirs(settings.LOGS_DIR, exist_ok=True)
    utils.append_to_session_log(
        settings.SESSION_LOG_CSV,
        frame_index=0,
        slot_dicts=[s.to_dict() for s in slots],
        violation_dicts=[v.to_dict() for v in violations],
    )

    # -- Annotated output image ------------------------------------------------
    annotated = visualizer.draw_slots(frame, slots)
    annotated = visualizer.draw_violations(annotated, violations)
    annotated = visualizer.draw_summary_overlay(
        annotated, len(slots), occupied, len(violations)
    )

    os.makedirs(args.out, exist_ok=True)
    out_image = os.path.join(args.out, "annotated_result.jpg")
    utils.save_image(annotated, out_image)
    print(f"\n[ParkLens] Annotated image saved -> {out_image}")
    if not args.no_display:
        cv2.imshow("ParkLens -- Result (press any key to close)", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # -- Module 3: Analytics --
    print("\n[Module 3 -- Analytics]")
    engine = AnalyticsEngine(reports_dir=args.out)
    engine.run()
    print(f"\n[ParkLens] Reports saved -> {args.out}")


def run_analytics_only(args: argparse.Namespace) -> None:
    """Analytics-only mode -- reads existing session log."""
    print("[ParkLens] Analytics-only mode")
    engine = AnalyticsEngine(reports_dir=args.out)
    engine.run()
    print(f"\n[ParkLens] Reports saved -> {args.out}")


# -- Main ----------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.analytics_only:
            run_analytics_only(args)
        else:
            run_pipeline(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[ParkLens] Interrupted by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()
