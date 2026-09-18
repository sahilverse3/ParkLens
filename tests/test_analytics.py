"""
tests/test_analytics.py -- Unit tests for Module 3: AnalyticsEngine.

Tests use a temporary CSV file (no real session log needed).
"""

import csv
import os

import pytest

from modules.analytics.engine import AnalyticsEngine


# -- CSV Fixtures --------------------------------------------------------------

_CSV_FIELDNAMES = [
    "timestamp", "frame_index", "slot_id", "slot_type",
    "is_occupied", "occupancy_confidence",
    "has_violation", "violation_type",
]


def write_csv(path: str, rows: list) -> None:
    """Write a list of row dicts to *path* as a session log CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def sample_rows(frame_index: int = 0) -> list:
    """Return 4 rows: 2 occupied (one with violation), 2 free."""
    ts = "2026-09-01T10:00:00"
    return [
        {"timestamp": ts, "frame_index": frame_index, "slot_id": 1,
         "slot_type": "regular", "is_occupied": True,
         "occupancy_confidence": 0.72, "has_violation": False, "violation_type": ""},
        {"timestamp": ts, "frame_index": frame_index, "slot_id": 2,
         "slot_type": "no_parking", "is_occupied": True,
         "occupancy_confidence": 0.85, "has_violation": True,
         "violation_type": "no_parking_zone"},
        {"timestamp": ts, "frame_index": frame_index, "slot_id": 3,
         "slot_type": "regular", "is_occupied": False,
         "occupancy_confidence": 0.10, "has_violation": False, "violation_type": ""},
        {"timestamp": ts, "frame_index": frame_index, "slot_id": 4,
         "slot_type": "regular", "is_occupied": False,
         "occupancy_confidence": 0.05, "has_violation": False, "violation_type": ""},
    ]


# -- Tests ---------------------------------------------------------------------

class TestAnalyticsEngine:

    # -- Stats correctness -----------------------------------------------------

    def test_occupancy_stats_correct(self, tmp_path):
        """Engine computes correct slot counts from CSV."""
        log = tmp_path / "logs" / "session_log.csv"
        write_csv(str(log), sample_rows())

        engine = AnalyticsEngine(log_path=str(log), reports_dir=str(tmp_path / "reports"))
        stats = engine.run()

        assert stats["total_slots"] == 4
        assert stats["occupied_readings"] == 2
        assert stats["free_readings"] == 2
        assert stats["avg_occupancy_pct"] == pytest.approx(50.0)

    def test_violation_count_correct(self, tmp_path):
        """Engine counts violations correctly."""
        log = tmp_path / "logs" / "session_log.csv"
        write_csv(str(log), sample_rows())

        engine = AnalyticsEngine(log_path=str(log), reports_dir=str(tmp_path / "reports"))
        stats = engine.run()

        assert stats["total_violations"] == 1
        assert stats["violation_breakdown"].get("no_parking_zone") == 1

    def test_no_violations_in_clean_session(self, tmp_path):
        """Engine handles zero violations gracefully."""
        rows = [
            {"timestamp": "2026-09-01T10:00:00", "frame_index": 0,
             "slot_id": i, "slot_type": "regular", "is_occupied": False,
             "occupancy_confidence": 0.0, "has_violation": False, "violation_type": ""}
            for i in range(1, 5)
        ]
        log = tmp_path / "logs" / "session_log.csv"
        write_csv(str(log), rows)

        engine = AnalyticsEngine(log_path=str(log), reports_dir=str(tmp_path / "reports"))
        stats = engine.run()

        assert stats["total_violations"] == 0
        assert stats["violation_breakdown"] == {}

    def test_multiple_frames_trend_computed(self, tmp_path):
        """Multi-frame CSV produces an occupancy trend list with len == frame count."""
        rows = sample_rows(frame_index=0) + sample_rows(frame_index=1)
        log = tmp_path / "logs" / "session_log.csv"
        write_csv(str(log), rows)

        engine = AnalyticsEngine(log_path=str(log), reports_dir=str(tmp_path / "reports"))
        stats = engine.run()

        # 2 frames -> trend list of length 2
        assert len(stats["occupancy_trend"]) == 2

    # -- Chart file creation ---------------------------------------------------

    def test_occupancy_chart_created(self, tmp_path):
        """Occupancy pie chart PNG is written to reports_dir."""
        log = tmp_path / "logs" / "session_log.csv"
        write_csv(str(log), sample_rows())
        reports = tmp_path / "reports"

        engine = AnalyticsEngine(log_path=str(log), reports_dir=str(reports))
        engine.run()

        assert (reports / "occupancy_chart.png").exists()

    def test_violation_chart_created(self, tmp_path):
        """Violation bar chart PNG is written when violations exist."""
        log = tmp_path / "logs" / "session_log.csv"
        write_csv(str(log), sample_rows())
        reports = tmp_path / "reports"

        engine = AnalyticsEngine(log_path=str(log), reports_dir=str(reports))
        engine.run()

        assert (reports / "violation_chart.png").exists()

    def test_text_report_created(self, tmp_path):
        """Text report is written to reports_dir."""
        log = tmp_path / "logs" / "session_log.csv"
        write_csv(str(log), sample_rows())
        reports = tmp_path / "reports"

        engine = AnalyticsEngine(log_path=str(log), reports_dir=str(reports))
        engine.run()

        report_path = reports / "session_report.txt"
        assert report_path.exists()
        content = report_path.read_text()
        assert "ParkLens" in content
        assert "OCCUPANCY SUMMARY" in content
        assert "VIOLATION SUMMARY" in content

    def test_trend_chart_created_for_multi_frame(self, tmp_path):
        """Trend line chart is written only when > 1 frame exists."""
        rows = sample_rows(0) + sample_rows(1)
        log = tmp_path / "logs" / "session_log.csv"
        write_csv(str(log), rows)
        reports = tmp_path / "reports"

        engine = AnalyticsEngine(log_path=str(log), reports_dir=str(reports))
        engine.run()

        assert (reports / "occupancy_trend.png").exists()

    def test_trend_chart_not_created_for_single_frame(self, tmp_path):
        """Trend line chart must NOT be written for a single-frame session."""
        log = tmp_path / "logs" / "session_log.csv"
        write_csv(str(log), sample_rows())
        reports = tmp_path / "reports"

        engine = AnalyticsEngine(log_path=str(log), reports_dir=str(reports))
        engine.run()

        assert not (reports / "occupancy_trend.png").exists()

    # -- Error handling --------------------------------------------------------

    def test_raises_when_log_missing(self, tmp_path):
        """FileNotFoundError when the log CSV does not exist."""
        engine = AnalyticsEngine(
            log_path=str(tmp_path / "no_such_file.csv"),
            reports_dir=str(tmp_path / "reports"),
        )
        with pytest.raises(FileNotFoundError):
            engine.run()

    def test_raises_when_log_empty(self, tmp_path):
        """ValueError when the CSV exists but has no data rows."""
        log = tmp_path / "logs" / "empty.csv"
        os.makedirs(str(tmp_path / "logs"), exist_ok=True)
        # Write header only -- no data rows
        with open(str(log), "w", newline="") as fh:
            fh.write(",".join(_CSV_FIELDNAMES) + "\n")

        engine = AnalyticsEngine(log_path=str(log), reports_dir=str(tmp_path / "reports"))
        with pytest.raises(ValueError, match="no data rows"):
            engine.run()

    # -- stats dict schema -----------------------------------------------------

    def test_stats_dict_has_all_expected_keys(self, tmp_path):
        """run() must return a dict with all documented keys."""
        log = tmp_path / "logs" / "session_log.csv"
        write_csv(str(log), sample_rows())

        engine = AnalyticsEngine(log_path=str(log), reports_dir=str(tmp_path / "reports"))
        stats = engine.run()

        expected_keys = {
            "total_slots", "total_readings", "occupied_readings", "free_readings",
            "avg_occupancy_pct", "total_violations", "violation_breakdown",
            "top_offending_slots", "occupancy_trend", "generated_at",
        }
        assert expected_keys.issubset(set(stats.keys()))
