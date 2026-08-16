import cv2
import numpy as np

from photo_quality import assess_photo_quality


def test_high_resolution_textured_landscape_is_ready():
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    image[:] = (145, 125, 95)
    for x in range(0, 1280, 20):
        cv2.line(image, (x, 0), (x, 719), (210, 180, 130), 3)
    result = assess_photo_quality(image)
    assert result["ready"]
    assert result["megapixels"] == 0.9


def test_small_dark_blurred_portrait_returns_guidance():
    image = np.full((640, 360, 3), 25, dtype=np.uint8)
    result = assess_photo_quality(image)
    assert not result["ready"]
    assert not result["checks"]["resolution"]
    assert not result["checks"]["brightness"]
    assert not result["checks"]["sharpness"]
    assert not result["checks"]["landscape"]
    assert len(result["guidance"]) == 4
