"""Calibration, log numbering, diameter calculation, and uncertainty helpers."""

from __future__ import annotations

from math import hypot, sqrt
import re
from statistics import median
from typing import Iterable
from uuid import uuid4


RED_LIMIT_IN = 14.0
YELLOW_LIMIT_IN = 16.0


def diameter_group(diameter_in: float | None) -> str:
    """Return the requested colour group for an outside-bark diameter."""
    if diameter_in is None:
        return "Uncalibrated"
    if diameter_in < RED_LIMIT_IN:
        return "Red"
    if diameter_in <= YELLOW_LIMIT_IN:
        return "Yellow"
    return "Blue"


def ring_rgb(group: str) -> tuple[int, int, int]:
    return {
        "Red": (238, 47, 47),
        "Yellow": (255, 201, 20),
        "Blue": (25, 118, 255),
        "Uncalibrated": (0, 229, 255),
    }.get(group, (0, 229, 255))


def ensure_log_ids(circles: Iterable[dict]) -> list[dict]:
    """Keep existing IDs stable and assign IDs only to new/duplicate circles."""
    items = [dict(circle) for circle in circles]
    if not items:
        return []

    used: set[str] = set()
    needs_id: list[dict] = []
    for circle in items:
        circle["uid"] = str(circle.get("uid") or uuid4().hex)
        log_id = str(circle.get("id", ""))
        if re.fullmatch(r"L\d+", log_id) and log_id not in used:
            used.add(log_id)
        else:
            circle.pop("id", None)
            needs_id.append(circle)

    next_number = 1
    width = max(2, len(str(len(items))))
    for circle in needs_id:
        while f"L{next_number:0{width}d}" in used:
            next_number += 1
        circle["id"] = f"L{next_number:0{width}d}"
        used.add(circle["id"])
        next_number += 1
    return items


def assign_log_ids(circles: Iterable[dict]) -> list[dict]:
    """Reassign L01... IDs row by row, from upper-left to bottom-right."""
    items = ensure_log_ids(circles)
    if not items:
        return []

    typical_diameter = median(max(2.0, float(c["radius"]) * 2.0) for c in items)
    row_tolerance = max(12.0, typical_diameter * 0.55)
    rows: list[list[dict]] = []

    for circle in sorted(items, key=lambda c: (float(c["y"]), float(c["x"]))):
        best_row = None
        best_delta = float("inf")
        for row in rows:
            row_y = sum(float(c["y"]) for c in row) / len(row)
            delta = abs(float(circle["y"]) - row_y)
            if delta <= row_tolerance and delta < best_delta:
                best_row, best_delta = row, delta
        if best_row is None:
            rows.append([circle])
        else:
            best_row.append(circle)

    rows.sort(key=lambda row: sum(float(c["y"]) for c in row) / len(row))
    ordered = [circle for row in rows for circle in sorted(row, key=lambda c: float(c["x"]))]
    width = max(2, len(str(len(ordered))))
    for index, circle in enumerate(ordered, start=1):
        circle["id"] = f"L{index:0{width}d}"
    return ordered


def nearest_circle(circles: Iterable[dict], x: float, y: float) -> dict | None:
    items = list(circles)
    if not items:
        return None
    return min(items, key=lambda c: hypot(float(c["x"]) - x, float(c["y"]) - y))


def calibration_from_references(
    circles: Iterable[dict], references: dict[str, float]
) -> dict:
    """Calculate pixels-per-inch and a transparent estimated tolerance.

    The scale is the arithmetic mean of the reference scales.  Tolerance combines
    a 2 px/manual-boundary term with the disagreement between two references.
    One reference cannot reveal perspective variation, so the result explicitly
    reports that limitation.
    """
    by_id = {str(c.get("id")): c for c in circles}
    valid: list[dict] = []
    for log_id, actual_in in references.items():
        circle = by_id.get(str(log_id))
        actual = float(actual_in)
        if circle is not None and actual > 0:
            diameter_px = 2.0 * float(circle["radius"])
            valid.append(
                {
                    "id": str(log_id),
                    "actual_in": actual,
                    "diameter_px": diameter_px,
                    "pixels_per_inch": diameter_px / actual,
                }
            )
    if not valid:
        raise ValueError("Select at least one reference log and enter its actual diameter.")

    pixels_per_inch = sum(v["pixels_per_inch"] for v in valid) / len(valid)
    residuals = [v["diameter_px"] / pixels_per_inch - v["actual_in"] for v in valid]
    calibration_rmse = sqrt(sum(r * r for r in residuals) / len(residuals))
    boundary_uncertainty = 2.0 / pixels_per_inch
    tolerance = sqrt(boundary_uncertainty**2 + calibration_rmse**2)

    return {
        "pixels_per_inch": pixels_per_inch,
        "inches_per_pixel": 1.0 / pixels_per_inch,
        "estimated_tolerance_in": max(0.1, round(tolerance, 1)),
        "calibration_rmse_in": calibration_rmse,
        "reference_count": len(valid),
        "references": valid,
        "perspective_warning": len(valid) == 1,
    }


def measured_logs(circles: Iterable[dict], calibration: dict | None) -> list[dict]:
    """Return export-ready measurements rounded to one decimal inch."""
    pixels_per_inch = calibration.get("pixels_per_inch") if calibration else None
    tolerance = calibration.get("estimated_tolerance_in") if calibration else None
    result: list[dict] = []
    for circle in ensure_log_ids(circles):
        item = dict(circle)
        diameter_px = float(item["radius"]) * 2.0
        diameter_in = round(diameter_px / pixels_per_inch, 1) if pixels_per_inch else None
        item.update(
            {
                "diameter_px": round(diameter_px, 1),
                "diameter_in": diameter_in,
                "tolerance_in": tolerance,
                "group": diameter_group(diameter_in),
            }
        )
        result.append(item)
    return result
