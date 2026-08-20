"""Practical, non-certified photo-quality guidance for log measurement images."""

from __future__ import annotations

import cv2
import numpy as np


def assess_photo_quality(rgb: np.ndarray) -> dict:
    """Return simple quality signals and actionable guidance.

    Thresholds are intentionally conservative workflow guidance rather than a
    guarantee of measurement accuracy. Users can continue with any valid image.
    """
    image = np.asarray(rgb, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Photo-quality assessment requires an RGB image.")

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    brightness = float(gray.mean())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    megapixels = float(width * height / 1_000_000)

    checks = {
        "resolution": width >= 1280 and height >= 720,
        "brightness": 65.0 <= brightness <= 210.0,
        "sharpness": sharpness >= 70.0,
        "landscape": width >= height,
    }
    guidance: list[str] = []
    if not checks["resolution"]:
        guidance.append("Use the original camera photo at a higher resolution; avoid screenshots.")
    if brightness < 65.0:
        guidance.append("The photo is dark. Retake it in brighter, even light if possible.")
    elif brightness > 210.0:
        guidance.append("The photo is very bright. Avoid glare and overexposed log faces.")
    if not checks["sharpness"]:
        guidance.append("The photo may be blurred. Clean the lens, hold steady, and refocus on the logs.")
    if not checks["landscape"]:
        guidance.append("Landscape orientation usually captures the full log stack more clearly.")

    return {
        "width": width,
        "height": height,
        "megapixels": round(megapixels, 1),
        "mean_brightness": round(brightness, 1),
        "sharpness_score": round(sharpness, 1),
        "checks": checks,
        "guidance": guidance,
        "ready": all(checks.values()),
    }

