from pathlib import Path

from PIL import Image

from image_processing import detect_log_ends, pil_to_rgb
from measurement import assign_log_ids, nearest_circle


def test_sample_detects_reference_regions():
    image_path = Path(__file__).parents[1] / "sample_data" / "opt_logs_reference.png"
    rgb = pil_to_rgb(Image.open(image_path))
    circles = assign_log_ids(detect_log_ends(rgb, 25, 70, 38, (18, 70)))
    assert 35 <= len(circles) <= 65
    ref1 = nearest_circle(circles, 947, 395)
    ref2 = nearest_circle(circles, 911, 483)
    assert ref1 is not None and ((ref1["x"] - 947) ** 2 + (ref1["y"] - 395) ** 2) ** 0.5 < 65
    assert ref2 is not None and ((ref2["x"] - 911) ** 2 + (ref2["y"] - 483) ** 2) ** 0.5 < 65
