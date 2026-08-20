"""OpenCV image enhancement and outside-bark log-end detection."""

from __future__ import annotations

from math import hypot
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def pil_to_rgb(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"))


def resize_for_analysis(
    rgb: np.ndarray, max_dimension: int = 2400
) -> tuple[np.ndarray, float]:
    """Downscale very large phone photos for predictable cloud processing."""
    image = np.asarray(rgb, dtype=np.uint8)
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_dimension:
        return image, 1.0
    scale = float(max_dimension) / float(longest)
    resized = cv2.resize(
        image,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


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


def _gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(grad_x, grad_y)


def _radial_refine(
    gray: np.ndarray,
    x: float,
    y: float,
    radius: float,
    magnitude: np.ndarray | None = None,
) -> tuple[float, float]:
    """Refine a proposed ring toward the strongest nearby outside-bark edge."""
    if radius < 5:
        return radius, 0.2
    if magnitude is None:
        magnitude = _gradient_magnitude(gray)
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


def _boundary_roundness(
    magnitude: np.ndarray,
    x: float,
    y: float,
    radius: float,
    edge_threshold: float = 70.0,
) -> float:
    """Score how continuously a circular boundary is supported around a ring.

    Straight edges and partial arcs can fool HoughCircles. Sampling narrow radial
    bands around the full circumference rejects those shapes unless edge support
    is both broad and distributed across most angular sectors.
    """
    if radius < 5:
        return 0.0
    angles = np.linspace(0, 2 * np.pi, 144, endpoint=False)
    radius_factors = np.linspace(0.92, 1.08, 7)
    radii = radius * radius_factors[:, None]
    xs = np.rint(x + np.cos(angles)[None, :] * radii).astype(int)
    ys = np.rint(y + np.sin(angles)[None, :] * radii).astype(int)
    height, width = magnitude.shape
    valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    samples = np.zeros_like(xs, dtype=np.float32)
    samples[valid] = magnitude[ys[valid], xs[valid]]
    strongest = samples.max(axis=0)
    supported = strongest >= edge_threshold

    coverage = float(supported.mean())
    sector_support = supported.reshape(12, -1).mean(axis=1)
    balanced_sectors = float((sector_support >= 0.35).mean())
    return float(np.clip(0.65 * coverage + 0.35 * balanced_sectors, 0.0, 1.0))


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
    minimum_roundness: float = 0.70,
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
        if axis_ratio < 0.76 or circularity < 0.50:
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
    round_candidates: list[dict] = []
    magnitude = _gradient_magnitude(full_gray)
    for circle in deduped:
        radius, boundary_confidence = _radial_refine(
            full_gray, circle["x"], circle["y"], circle["radius"], magnitude
        )
        circle["radius"] = float(np.clip(radius, min_radius, max_radius))
        roundness = _boundary_roundness(
            magnitude, circle["x"], circle["y"], circle["radius"]
        )
        circle["roundness"] = round(roundness, 3)
        if roundness < float(minimum_roundness):
            continue
        circle["confidence"] = round(
            0.40 * float(circle["confidence"])
            + 0.30 * boundary_confidence
            + 0.30 * roundness,
            3,
        )
        round_candidates.append(circle)
    return _deduplicate(round_candidates, min_radius)


def _guided_circle_at_point(
    rgb: np.ndarray,
    point: tuple[float, float],
    min_radius: int,
    max_radius: int,
    hough_threshold: int,
    minimum_roundness: float,
) -> dict:
    """Fit one best-effort circular boundary around a user-marked log centre."""
    height, width = rgb.shape[:2]
    point_x = float(np.clip(point[0], 0, width - 1))
    point_y = float(np.clip(point[1], 0, height - 1))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    magnitude = _gradient_magnitude(gray)
    half_size = max(int(max_radius * 1.8), int(min_radius * 3.0))
    left = max(0, int(round(point_x)) - half_size)
    right = min(width, int(round(point_x)) + half_size + 1)
    top = max(0, int(round(point_y)) - half_size)
    bottom = min(height, int(round(point_y)) + half_size + 1)
    roi = gray[top:bottom, left:right]
    candidates: list[dict] = []
    if roi.size:
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        blurred = cv2.GaussianBlur(clahe.apply(roi), (7, 7), 1.6)
        hough = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.1,
            minDist=max(10, int(min_radius * 1.2)),
            param1=105,
            param2=max(14, int(hough_threshold * 0.72)),
            minRadius=int(min_radius),
            maxRadius=int(max_radius),
        )
        if hough is not None:
            for local_x, local_y, radius in hough[0]:
                x = float(local_x + left)
                y = float(local_y + top)
                distance = hypot(x - point_x, y - point_y)
                if distance > max(float(radius) * 1.1, float(min_radius) * 1.4):
                    continue
                refined_radius, boundary_confidence = _radial_refine(
                    gray, x, y, float(radius), magnitude
                )
                refined_radius = float(np.clip(refined_radius, min_radius, max_radius))
                roundness = _boundary_roundness(magnitude, x, y, refined_radius)
                proximity = max(0.0, 1.0 - distance / max(refined_radius * 1.1, 1.0))
                candidates.append(
                    {
                        "x": x,
                        "y": y,
                        "radius": refined_radius,
                        "source": "Guided local Hough",
                        "roundness": round(roundness, 3),
                        "confidence": round(
                            0.45 * roundness + 0.30 * boundary_confidence + 0.25 * proximity,
                            3,
                        ),
                    }
                )

    acceptable = [
        circle
        for circle in candidates
        if float(circle.get("roundness", 0.0)) >= max(0.35, minimum_roundness - 0.25)
    ]
    if acceptable:
        return max(acceptable, key=lambda circle: float(circle.get("confidence", 0.0)))

    max_inside_radius = max(
        5.0,
        min(point_x, point_y, width - 1 - point_x, height - 1 - point_y) - 1.0,
    )
    radius_ceiling = max(5.0, min(float(max_radius), max_inside_radius))
    radius_floor = min(float(min_radius), radius_ceiling)
    tested_radii = np.linspace(radius_floor, radius_ceiling, 30)
    scored = [
        (_boundary_roundness(magnitude, point_x, point_y, float(radius)), float(radius))
        for radius in tested_radii
    ]
    roundness, radius = max(scored, key=lambda item: item[0])
    radius, boundary_confidence = _radial_refine(gray, point_x, point_y, radius, magnitude)
    return {
        "x": point_x,
        "y": point_y,
        "radius": float(np.clip(radius, radius_floor, radius_ceiling)),
        "source": "Guided point fallback",
        "roundness": round(float(roundness), 3),
        "confidence": round(0.35 * float(roundness) + 0.25 * boundary_confidence, 3),
    }


def detect_log_ends_at_points(
    rgb: np.ndarray,
    points: Iterable[tuple[float, float]],
    min_radius: int,
    max_radius: int,
    hough_threshold: int = 30,
    search_y_percent: tuple[int, int] = (0, 100),
    minimum_roundness: float = 0.70,
) -> list[dict]:
    """Return exactly one fitted ring per marked point, or normal detection when empty."""
    marked_points = [(float(x), float(y)) for x, y in points]
    if not marked_points:
        return detect_log_ends(
            rgb,
            min_radius,
            max_radius,
            hough_threshold,
            search_y_percent,
            minimum_roundness,
        )

    automatic = detect_log_ends(
        rgb,
        min_radius,
        max_radius,
        hough_threshold,
        search_y_percent,
        minimum_roundness,
    )
    unused = set(range(len(automatic)))
    guided: list[dict] = []
    for point in marked_points:
        eligible: list[tuple[float, int]] = []
        for index in unused:
            candidate = automatic[index]
            distance = hypot(float(candidate["x"]) - point[0], float(candidate["y"]) - point[1])
            if distance <= max(float(candidate["radius"]) * 1.25, float(min_radius) * 1.5):
                score = distance / max(float(candidate["radius"]), 1.0) - 0.2 * float(
                    candidate.get("confidence", 0.0)
                )
                eligible.append((score, index))
        if eligible:
            _, best_index = min(eligible)
            unused.remove(best_index)
            circle = dict(automatic[best_index])
            circle["source"] = f"Guided match · {circle.get('source', 'automatic')}"
        else:
            circle = _guided_circle_at_point(
                rgb,
                point,
                min_radius,
                max_radius,
                hough_threshold,
                minimum_roundness,
            )
        circle["guide_x"], circle["guide_y"] = point
        guided.append(circle)
    return guided
