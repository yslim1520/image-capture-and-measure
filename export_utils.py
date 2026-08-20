"""CSV, JSON, and annotated-image export helpers."""

from __future__ import annotations

import base64
import csv
import html
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from measurement import ring_rgb


def hover_id_overlay(rgb: np.ndarray, circles: list[dict]) -> str:
    """Build a labelled-on-hover SVG overlay without changing the source image."""
    image = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise RuntimeError("OpenCV could not encode the hover preview image.")
    image_data = base64.b64encode(encoded.tobytes()).decode("ascii")
    height, width = image.shape[:2]
    ring_width = max(2, int(round(width / 700)))
    rings: list[str] = []
    for circle in circles:
        log_id = html.escape(str(circle.get("id", "Unnumbered")), quote=True)
        colour = "#%02x%02x%02x" % ring_rgb(circle.get("group", "Uncalibrated"))
        radius = max(2.0, float(circle["radius"]))
        diameter = radius * 2.0
        rings.append(
            f'<div class="log-ring" style="left:{float(circle["x"]) - radius:.2f}px;'
            f'top:{float(circle["y"]) - radius:.2f}px;width:{diameter:.2f}px;height:{diameter:.2f}px;'
            f'border:{ring_width}px solid {colour}" data-log-id="{log_id}" '
            f'aria-label="{log_id}" title="{log_id}"></div>'
        )
    return f"""<div class="opt-hover-map">
<style>
.opt-hover-map{{position:relative;max-width:100%;max-height:780px;overflow:auto;background:#111}}
.opt-hover-stage{{position:relative;width:{width}px;height:{height}px}}
.opt-hover-stage img{{position:absolute;inset:0;width:{width}px;height:{height}px;max-width:none}}
.opt-hover-map .log-ring{{position:absolute;box-sizing:border-box;border-radius:50%;cursor:help;
pointer-events:all;background:rgba(0,0,0,0.001)}}
.opt-hover-map .log-tip{{display:none;position:absolute;z-index:10;padding:5px 9px;border-radius:6px;
background:#111;color:#fff;font:600 14px system-ui,sans-serif;box-shadow:0 2px 8px #0008;
pointer-events:none}}
</style>
<div class="opt-hover-stage" role="img">
<img src="data:image/png;base64,{image_data}" width="{width}" height="{height}" alt="Log ID map">
{''.join(rings)}
<div class="log-tip"></div>
</div>
<script>
const root=document.currentScript.parentElement;
const tip=root.querySelector('.log-tip');
root.querySelectorAll('.log-ring').forEach((ring)=>{{
  ring.addEventListener('mouseenter',(event)=>{{
    const box=root.getBoundingClientRect();
    tip.textContent=ring.dataset.logId; tip.style.display='block';
    tip.style.left=(event.clientX-box.left+root.scrollLeft+12)+'px';
    tip.style.top=(event.clientY-box.top+root.scrollTop+12)+'px';
  }});
  ring.addEventListener('mousemove',(event)=>{{
    const box=root.getBoundingClientRect();
    tip.style.left=(event.clientX-box.left+root.scrollLeft+12)+'px';
    tip.style.top=(event.clientY-box.top+root.scrollTop+12)+'px';
  }});
  ring.addEventListener('mouseleave',()=>{{tip.style.display='none';}});
  ring.addEventListener('click',(event)=>{{
    event.stopPropagation();
    const box=root.getBoundingClientRect();
    tip.textContent=ring.dataset.logId; tip.style.display='block';
    tip.style.left=(event.clientX-box.left+root.scrollLeft+12)+'px';
    tip.style.top=(event.clientY-box.top+root.scrollTop+12)+'px';
  }});
}});
root.addEventListener('click',()=>{{tip.style.display='none';}});
</script></div>"""


def measurements_csv(logs: list[dict], references: dict[str, float]) -> bytes:
    output = io.StringIO(newline="")
    fields = [
        "log_id",
        "center_x_px",
        "center_y_px",
        "outside_bark_diameter_px",
        "outside_bark_diameter_in",
        "estimated_tolerance_in",
        "colour_group",
        "is_reference",
        "reference_actual_in",
        "detection_source",
        "fit_confidence",
    ]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for log in logs:
        writer.writerow(
            {
                "log_id": log["id"],
                "center_x_px": round(float(log["x"]), 1),
                "center_y_px": round(float(log["y"]), 1),
                "outside_bark_diameter_px": log["diameter_px"],
                "outside_bark_diameter_in": log["diameter_in"],
                "estimated_tolerance_in": log["tolerance_in"],
                "colour_group": log["group"],
                "is_reference": log["id"] in references,
                "reference_actual_in": references.get(log["id"], ""),
                "detection_source": log.get("source", "Manual"),
                "fit_confidence": log.get("confidence", ""),
            }
        )
    return output.getvalue().encode("utf-8-sig")


def project_json(
    image_name: str,
    image_sha256: str,
    image_shape: tuple[int, ...],
    circles: list[dict],
    references: dict[str, float],
    calibration: dict | None,
    enhancement: dict,
) -> bytes:
    payload = {
        "schema_version": 1,
        "app": "OPT Log Diameter Checker",
        "saved_utc": datetime.now(timezone.utc).isoformat(),
        "measurement_basis": "outside bark",
        "image": {
            "file_name": image_name,
            "sha256": image_sha256,
            "width": int(image_shape[1]),
            "height": int(image_shape[0]),
        },
        "enhancement": enhancement,
        "circles": circles,
        "reference_diameters_in": references,
        "calibration": calibration,
        "colour_rules_in": {
            "Red": "<14.0",
            "Yellow": "14.0-16.0 inclusive",
            "Blue": ">16.0",
        },
    }
    return json.dumps(payload, indent=2).encode("utf-8")


def load_project(data: bytes) -> dict:
    payload = json.loads(data.decode("utf-8-sig"))
    if payload.get("schema_version") != 1 or "circles" not in payload:
        raise ValueError("This is not a supported OPT Log Diameter Checker project file.")
    return payload


def annotated_image(rgb: np.ndarray, logs: list[dict], image_format: str = "PNG") -> bytes:
    canvas = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    height, width = canvas.shape[:2]
    font_scale = max(0.42, min(0.8, width / 2200.0))
    ring_width = max(2, int(round(width / 700)))
    for log in logs:
        center = (int(round(log["x"])), int(round(log["y"])))
        radius = max(2, int(round(log["radius"])))
        rgb_colour = ring_rgb(log["group"])
        bgr = (rgb_colour[2], rgb_colour[1], rgb_colour[0])
        cv2.circle(canvas, center, radius, bgr, ring_width, cv2.LINE_AA)
        diameter = log.get("diameter_in")
        label = f"{log['id']} {diameter:.1f} in" if diameter is not None else log["id"]
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        text_x = int(np.clip(center[0] - text_w / 2, 2, max(2, width - text_w - 2)))
        text_y = int(np.clip(center[1] + text_h / 2, text_h + 2, height - 3))
        cv2.putText(canvas, label, (text_x + 1, text_y + 1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(canvas, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, bgr, 2, cv2.LINE_AA)

    extension = ".jpg" if image_format.upper() in {"JPG", "JPEG"} else ".png"
    params = [cv2.IMWRITE_JPEG_QUALITY, 95] if extension == ".jpg" else [cv2.IMWRITE_PNG_COMPRESSION, 3]
    ok, encoded = cv2.imencode(extension, canvas, params)
    if not ok:
        raise RuntimeError("OpenCV could not encode the annotated image.")
    return encoded.tobytes()
