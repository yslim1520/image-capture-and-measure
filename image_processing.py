"""OpenCV image enhancement and outside-bark log-end detection."""

from __future__ import annotations

from math import hypot
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def pil_to_rgb(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"))


def enhance_image(
    rgb: np.ndarray,
    brightness: int = 0,
    contrast: float = 1.0,
    sharpness: float = 1.0,
) -> np.ndarray:
    """Apply user-friendly brightness, contrast, and sharpness adjustments."""
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
    if brightness:
        image = ImageEnhance.Brightness(image).enhance(max(0.05, 1.0 + brightness / 100.0))
    if contrast != 1.0:
        image = ImageEnhance.Contrast(image).enhance(float(contrast))
    if sharpness != 1.0:
        image = ImageEnhance.Sharpness(image).enhance(float(sharpness))
    return np.asarray(image)


def edge_preview(rgb: np.ndarray, low: int = 55, high: int = 145) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, low, high)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)


def _radial_refine(gray: np.ndarray, x: float, y: float, radius: float) -> tuple[float, float]:
    """Refine a proposed ring toward the strongest nearby outside-bark edge."""
    if radius < 5:
        return radius, 0.2
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    radii = np.linspace(radius * 0.78, radius * 1.28, 35)
    angles = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    candidates: list[float] = []
    strengths: list[float] = []
    height, width = gray.shape
    for angle in angles:
        xs = np.rint(x + np.cos(angle) * radii).astype(int)
        ys = np.rint(y + np.sin(angle) * radii).astype(int)
        valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        if valid.sum() < 12:
            continue
        valid_radii = radii[valid]
        values = magnitude[ys[valid], xs[valid]]
        preference = 1.0 - 0.18 * np.abs(valid_radii - radius) / max(radius, 1.0)
        index = int(np.argmax(values * preference))
        candidates.append(float(valid_radii[index]))
        strengths.append(float(values[index]))
    if len(candidates) < 24:
        return radius, 0.2
    refined = float(np.median(candidates))
    mad = float(np.median(np.abs(np.asarray(candidates) - refined)))
    consistency = max(0.0, 1.0 - mad / max(refined * 0.22, 1.0))
    strength_score = min(1.0, float(np.median(strengths)) / 160.0)
    return refined, 0.45 * consistency + 0.55 * strength_score


def _deduplicate(circles: Iterable[dict], min_radius: float) -> list[dict]:
    accepted: list[dict] = []
    for candidate in sorted(circles, key=lambda c: float(c.get("confidence", 0)), reverse=True):
        duplicate = False
        for existing in accepted:
            distance = hypot(candidate["x"] - existing["x"], candidate["y"] - existing["y"])
            overlap_distance = 0.82 * min(candidate["radius"], existing["radius"])
            similar_radius = abs(candidate["radius"] - existing["radius"]) < 0.6 * max(
                candidate["radius"], existing["radius"]
            )
            if distance < max(min_radius * 0.55, overlap_distance) and similar_radius:
                duplicate = True
                break
        if not duplicate:
            accepted.append(candidate)
    return accepted


def _log_end_appearance(rgb: np.ndarray, circle: dict) -> tuple[bool, float]:
    """Reject foliage, blue equipment, and pale ground before ring refinement.

    Fresh and weathered log ends in the target workflow are warm-toned surfaces.
    The limits are intentionally loose, and the UI always retains manual add/edit.
    """
    x, y, radius = float(circle["x"]), float(circle["y"]), float(circle["radius"])
    height, width = rgb.shape[:2]
    if x - radius < 0 or y - radius < 0 or x + radius >= width or y + radius >= height:
        return False, 0.0
    margin = int(radius * 1.3) + 2
    left, right = max(0, int(x) - margin), min(width, int(x) + margin + 1)
    top, bottom = max(0, int(y) - margin), min(height, int(y) + margin + 1)
    if right - left < radius or bottom - top < radius:
        return False, 0.0
    patch = rgb[top:bottom, left:right]
    yy, xx = np.ogrid[top:bottom, left:right]
    distance_sq = (xx - x) ** 2 + (yy - y) ** 2
    inner = distance_sq <= (radius * 0.68) ** 2
    annulus = (distance_sq >= (radius * 1.03) ** 2) & (distance_sq <= (radius * 1.27) ** 2)
    if inner.sum() < 60 or annulus.sum() < 40:
        return False, 0.0
    inner_pixels = patch[inner]
    median_rgb = np.median(inner_pixels.astype(np.float32), axis=0)
    hsv_pixels = cv2.cvtColor(inner_pixels.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2HSV)
    median_saturation = float(np.median(hsv_pixels[:, 0, 1]))
    median_value = float(np.median(hsv_pixels[:, 0, 2]))
    red, green, blue = (float(v) for v in median_rgb)
    gray_patch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    contrast_to_surround = float(gray_patch[inner].mean() - gray_patch[annulus].mean())

    warm_surface = red >= green + 3.0 and green >= blue - 4.0
    coloured_enough = median_saturation >= 30.0
    usable_brightness = 52.0 <= median_value <= 245.0
    not_darker_than_surround = contrast_to_surround >= -7.0
    appearance_score = np.clip(
        0.35
        + (red - green) / 70.0
        + median_saturation / 420.0
        + max(-10.0, min(55.0, contrast_to_surround)) / 180.0,
        0.0,
        1.0,
    )
    accepted = (
        warm_surface and coloured_enough and usable_brightness and not_darker_than_surround
    ) or (appearance_score >= 0.84 and median_value < 210.0 and not_darker_than_surround)
    return bool(accepted), float(appearance_score)


def detect_log_ends(
    rgb: np.ndarray,
    min_radius: int,
    max_radius: int,
    hough_threshold: int = 30,
    search_y_percent: tuple[int, int] = (0, 100),
) -> list[dict]:
    """Detect log ends with Hough circles plus a contour/ellipse fallback.

    Returned rings are radially refined against the strongest nearby boundary,
    which biases the overlay toward outside bark rather than heartwood marks.
    """
    height, width = rgb.shape[:2]
    y0 = max(0, int(height * search_y_percent[0] / 100))
    y1 = min(height, int(height * search_y_percent[1] / 100))
    if y1 - y0 < max_radius * 2:
        raise ValueError("The selected vertical search region is too small.")

    roi = rgb[y0:y1]
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    blurred = cv2.GaussianBlur(equalized, (7, 7), 1.6)
    candidates: list[dict] = []

    hough = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.15,
        minDist=max(12, int(min_radius * 1.8)),
        param1=110,
        param2=int(hough_threshold),
        minRadius=int(min_radius),
        maxRadius=int(max_radius),
    )
    if hough is not None:
        for x, y, radius in np.round(hough[0]).astype(float):
            candidates.append(
                {
                    "x": float(x),
                    "y": float(y + y0),
                    "radius": float(radius),
                    "source": "Hough",
                    "confidence": 0.72,
                }
            )

    edges = cv2.Canny(blurred, 45, 135)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if len(contour) < 5 or perimeter <= 0:
            continue
        ellipse = cv2.fitEllipse(contour)
        (x, y), (axis_a, axis_b), _ = ellipse
        radius = (axis_a + axis_b) / 4.0
        axis_ratio = min(axis_a, axis_b) / max(axis_a, axis_b)
        circularity = 4.0 * np.pi * area / (perimeter * perimeter)
        if not (min_radius <= radius <= max_radius):
            continue
        if axis_ratio < 0.62 or circularity < 0.33:
            continue
        if x - radius < 0 or x + radius >= width or y - radius < 0 or y + radius >= roi.shape[0]:
            continue
        candidates.append(
            {
                "x": float(x),
                "y": float(y + y0),
                "radius": float(radius),
                "source": "Ellipse fallback",
                "confidence": float(min(0.68, 0.34 + 0.22 * axis_ratio + 0.18 * circularity)),
            }
        )

    appearance_filtered: list[dict] = []
    for circle in candidates:
        looks_like_log, appearance_confidence = _log_end_appearance(rgb, circle)
        if not looks_like_log:
            continue
        circle["confidence"] = 0.72 * float(circle["confidence"]) + 0.28 * appearance_confidence
        appearance_filtered.append(circle)

    deduped = _deduplicate(appearance_filtered, min_radius)
    full_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    for circle in deduped:
        radius, boundary_confidence = _radial_refine(
            full_gray, circle["x"], circle["y"], circle["radius"]
        )
        circle["radius"] = float(np.clip(radius, min_radius, max_radius))
        circle["confidence"] = round(
            0.55 * float(circle["confidence"]) + 0.45 * boundary_confidence, 3
        )
    return _deduplicate(deduped, min_radius)
