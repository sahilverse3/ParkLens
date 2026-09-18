"""
modules/analytics/engine.py -- Module 3: Parking Occupancy & Violation Analytics.

Reads the CSV session log written by the pipeline and produces:
  * A printed / saved text summary  (reports/session_report.txt)
  * An occupancy pie chart          (reports/occupancy_chart.png)
  * A violation bar chart           (reports/violation_chart.png)

If the log contains multiple frames (video input), an occupancy-trend
line chart is also produced       (reports/occupancy_trend.png).
"""

from __future__ import annotations

import csv
import os
from collections import Counter, defaultdict
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")          # non-interactive backend -- safe for any OS
import matplotlib.pyplot as plt

from config import settings
from core.utils import now_str


class AnalyticsEngine:
    """
    Reads a session CSV log and generates summary statistics and charts.

    Parameters
    ----------
    log_path : str
        Path to the CSV file produced by the pipeline.
    reports_dir : str
        Directory where chart images and the text report are saved.
    """

    def __init__(
        self,
        log_path: str | None = None,
        reports_dir: str | None = None,
    ) -> None:
        self.log_path = log_path or settings.SESSION_LOG_CSV
        self.reports_dir = reports_dir or settings.REPORTS_DIR
        os.makedirs(self.reports_dir, exist_ok=True)

    # -- Public API ------------------------------------------------------------

    def run(self) -> Dict:
        """
        Full analytics pipeline:
          1. Load CSV log.
          2. Compute summary stats.
          3. Write charts.
          4. Write text report.

        Returns
        -------
        dict
            Summary statistics dictionary (also written to text report).

        Raises
        ------
        FileNotFoundError
            If the session log CSV does not exist.
        ValueError
            If the CSV is empty or has no data rows.
        """
        rows = self._load_csv()
        stats = self._compute_stats(rows)
        self._write_occupancy_chart(stats)
        self._write_violation_chart(stats)
        self._maybe_write_trend_chart(rows, stats)
        self._write_text_report(stats)
        return stats

    # -- Data Loading ----------------------------------------------------------

    def _load_csv(self) -> List[Dict]:
        if not os.path.exists(self.log_path):
            raise FileNotFoundError(
                f"Session log not found: {self.log_path}\n"
                "Run the pipeline at least once to generate it."
            )
        rows = []
        with open(self.log_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                rows.append(row)
        if not rows:
            raise ValueError("Session log CSV exists but contains no data rows.")
        return rows

    # -- Statistics ------------------------------------------------------------

    @staticmethod
    def _compute_stats(rows: List[Dict]) -> Dict:
        """Aggregate raw CSV rows into summary statistics."""
        total_slots_seen: set = set()
        occupied_readings = 0
        total_readings = 0
        violation_types: List[str] = []
        violation_slots: List[str] = []

        # Per-frame occupancy for trend chart
        frame_occupancy: Dict[str, List[bool]] = defaultdict(list)

        for row in rows:
            sid = row.get("slot_id", "")
            total_slots_seen.add(sid)
            is_occ = row.get("is_occupied", "False").strip().lower() == "true"
            frame_occupancy[row.get("frame_index", "0")].append(is_occ)
            total_readings += 1
            if is_occ:
                occupied_readings += 1

            vtype = row.get("violation_type", "").strip()
            if vtype:
                violation_types.append(vtype)
                violation_slots.append(sid)

        total_slots = len(total_slots_seen)
        # Average occupancy across all frames
        avg_occ_pct = (occupied_readings / total_readings * 100) if total_readings else 0

        # Per-frame occupancy % for trend
        trend: List[float] = []
        for fidx in sorted(frame_occupancy.keys(), key=lambda x: int(x)):
            occ_list = frame_occupancy[fidx]
            trend.append(sum(occ_list) / len(occ_list) * 100 if occ_list else 0)

        violation_counter = Counter(violation_types)
        offending_slots = Counter(violation_slots)

        return {
            "total_slots": total_slots,
            "total_readings": total_readings,
            "occupied_readings": occupied_readings,
            "free_readings": total_readings - occupied_readings,
            "avg_occupancy_pct": round(avg_occ_pct, 2),
            "total_violations": len(violation_types),
            "violation_breakdown": dict(violation_counter),
            "top_offending_slots": offending_slots.most_common(5),
            "occupancy_trend": trend,
            "generated_at": now_str(),
        }

    # -- Chart Writers ---------------------------------------------------------

    def _write_occupancy_chart(self, stats: Dict) -> None:
        """Pie chart: free vs occupied readings."""
        occupied = stats["occupied_readings"]
        free = stats["free_readings"]

        if occupied + free == 0:
            return

        fig, ax = plt.subplots(figsize=(5, 5), dpi=settings.CHART_DPI)
        ax.pie(
            [occupied, free],
            labels=["Occupied", "Free"],
            colors=["#e74c3c", "#2ecc71"],
            autopct="%1.1f%%",
            startangle=90,
        )
        ax.set_title("Parking Slot Occupancy", fontsize=14, fontweight="bold")
        path = os.path.join(self.reports_dir, "occupancy_chart.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)

    def _write_violation_chart(self, stats: Dict) -> None:
        """Bar chart: violation count by type."""
        breakdown = stats["violation_breakdown"]
        if not breakdown:
            return

        labels = [k.replace("_", " ").title() for k in breakdown.keys()]
        counts = list(breakdown.values())

        fig, ax = plt.subplots(figsize=(7, 4), dpi=settings.CHART_DPI)
        bars = ax.bar(labels, counts, color="#e67e22", edgecolor="white")
        ax.bar_label(bars, padding=3)
        ax.set_title("Violations by Type", fontsize=14, fontweight="bold")
        ax.set_ylabel("Count")
        ax.set_ylim(0, max(counts) + 2)
        fig.tight_layout()
        path = os.path.join(self.reports_dir, "violation_chart.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)

    def _maybe_write_trend_chart(self, rows: List[Dict], stats: Dict) -> None:
        """Line chart: occupancy % per frame (only if > 1 frame present)."""
        trend = stats["occupancy_trend"]
        if len(trend) <= 1:
            return

        fig, ax = plt.subplots(figsize=(9, 4), dpi=settings.CHART_DPI)
        ax.plot(range(len(trend)), trend, marker="o", color="#3498db", linewidth=2)
        ax.set_title("Occupancy % Over Time", fontsize=14, fontweight="bold")
        ax.set_xlabel("Frame Index")
        ax.set_ylabel("Occupancy (%)")
        ax.set_ylim(0, 105)
        ax.axhline(y=stats["avg_occupancy_pct"], color="#e74c3c",
                   linestyle="--", linewidth=1, label=f"Average {stats['avg_occupancy_pct']:.1f}%")
        ax.legend()
        fig.tight_layout()
        path = os.path.join(self.reports_dir, "occupancy_trend.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)

    # -- Text Report -----------------------------------------------------------

    def _write_text_report(self, stats: Dict) -> None:
        path = os.path.join(self.reports_dir, "session_report.txt")
        lines = [
            "=" * 60,
            "  ParkLens -- Session Analytics Report",
            f"  Generated: {stats['generated_at']}",
            "=" * 60,
            "",
            "  OCCUPANCY SUMMARY",
            "  -----------------",
            f"  Total parking slots  : {stats['total_slots']}",
            f"  Total slot readings  : {stats['total_readings']}",
            f"  Occupied readings    : {stats['occupied_readings']}",
            f"  Free readings        : {stats['free_readings']}",
            f"  Avg occupancy        : {stats['avg_occupancy_pct']:.2f}%",
            "",
            "  VIOLATION SUMMARY",
            "  -----------------",
            f"  Total violations     : {stats['total_violations']}",
        ]
        for vtype, count in stats["violation_breakdown"].items():
            lines.append(f"    * {vtype.replace('_', ' ').title():<25}: {count}")

        lines += [
            "",
            "  TOP OFFENDING SLOTS",
            "  -------------------",
        ]
        if stats["top_offending_slots"]:
            for slot_id, count in stats["top_offending_slots"]:
                lines.append(f"    * Slot #{slot_id:<5}: {count} violation(s)")
        else:
            lines.append("    (no violations recorded)")

        lines += ["", "=" * 60]

        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")

        # Also print to console
        print("\n".join(lines))
