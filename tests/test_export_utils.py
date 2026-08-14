import json

import numpy as np

from export_utils import annotated_image, hover_id_overlay, measurements_csv, project_json


def _logs():
    return [
        {
            "id": "L01",
            "x": 20.0,
            "y": 20.0,
            "radius": 10.0,
            "diameter_px": 20.0,
            "diameter_in": 13.0,
            "tolerance_in": 0.4,
            "group": "Red",
            "source": "Manual",
            "confidence": 1.0,
        }
    ]


def test_csv_is_excel_friendly():
    data = measurements_csv(_logs(), {"L01": 13.0})
    assert data.startswith(b"\xef\xbb\xbf")
    assert b"outside_bark_diameter_in" in data


def test_annotated_png_is_valid():
    data = annotated_image(np.zeros((50, 50, 3), dtype=np.uint8), _logs(), "PNG")
    assert data.startswith(b"\x89PNG")


def test_hover_overlay_hides_labels_until_pointer_enters_ring():
    html = hover_id_overlay(np.zeros((50, 50, 3), dtype=np.uint8), _logs())
    assert 'data-log-id="L01"' in html
    assert "tip.style.display='block'" in html
    assert "tip.style.display='none'" in html


def test_project_round_trip_payload():
    data = project_json(
        "test.png", "abc", (50, 50, 3), _logs(), {"L01": 13.0}, {"pixels_per_inch": 1.0}, {}
    )
    payload = json.loads(data)
    assert payload["measurement_basis"] == "outside bark"
    assert payload["colour_rules_in"]["Yellow"] == "14.0-16.0 inclusive"
