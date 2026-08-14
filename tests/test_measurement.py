from measurement import (
    assign_log_ids,
    calibration_from_references,
    diameter_group,
    ensure_log_ids,
    measured_logs,
)


def test_colour_boundaries_are_exact():
    assert diameter_group(13.9) == "Red"
    assert diameter_group(14.0) == "Yellow"
    assert diameter_group(16.0) == "Yellow"
    assert diameter_group(16.1) == "Blue"


def test_calibration_and_one_decimal_measurement():
    circles = assign_log_ids(
        [
            {"x": 10, "y": 10, "radius": 65, "source": "Manual"},
            {"x": 100, "y": 10, "radius": 95, "source": "Manual"},
        ]
    )
    calibration = calibration_from_references(circles, {"L01": 13.0, "L02": 19.0})
    assert round(calibration["pixels_per_inch"], 3) == 10.0
    logs = measured_logs(circles, calibration)
    assert logs[0]["diameter_in"] == 13.0
    assert logs[1]["diameter_in"] == 19.0


def test_ids_are_unique_and_row_ordered():
    circles = assign_log_ids(
        [
            {"x": 90, "y": 100, "radius": 10},
            {"x": 10, "y": 10, "radius": 10},
            {"x": 20, "y": 100, "radius": 10},
        ]
    )
    assert [c["id"] for c in circles] == ["L01", "L02", "L03"]
    assert [(c["x"], c["y"]) for c in circles] == [(10, 10), (20, 100), (90, 100)]
    assert len({c["id"] for c in circles}) == 3


def test_manual_edits_keep_ids_until_explicit_reassignment():
    circles = assign_log_ids(
        [
            {"x": 10, "y": 10, "radius": 10},
            {"x": 100, "y": 10, "radius": 10},
        ]
    )
    moved = [dict(circles[0], x=150), dict(circles[1], x=5)]
    stable = ensure_log_ids(moved)
    assert [circle["id"] for circle in stable] == ["L01", "L02"]
    reassigned = assign_log_ids(stable)
    assert [(circle["id"], circle["x"]) for circle in reassigned] == [
        ("L01", 5),
        ("L02", 150),
    ]
