"""Streamlit user interface for OPT Log Diameter Checker."""

from __future__ import annotations

import hashlib
import json
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from export_utils import (
    annotated_image,
    hover_id_overlay,
    load_project,
    measurements_csv,
    project_json,
)
from image_processing import detect_log_ends, edge_preview, enhance_image, pil_to_rgb
from measurement import (
    assign_log_ids,
    calibration_from_references,
    ensure_log_ids,
    measured_logs,
    nearest_circle,
    ring_rgb,
)


APP_DIR = Path(__file__).resolve().parent
SAMPLE_IMAGE = APP_DIR / "sample_data" / "opt_logs_reference.png"
SAMPLE_REFERENCE_HINTS = [(947.0, 395.0, 19.0), (911.0, 483.0, 13.0)]


st.set_page_config(page_title="OPT Log Diameter Checker", page_icon="🪵", layout="wide")
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
      .step-card {background:#f7f9fb;border:1px solid #d9e1e8;border-radius:12px;padding:12px 16px;margin-bottom:12px;}
      .small-note {color:#536270;font-size:0.9rem;}
      div[data-testid="stMetric"] {background:#f7f9fb;border:1px solid #e3e8ee;border-radius:10px;padding:8px 12px;}
    </style>
    """,
    unsafe_allow_html=True,
)


def _read_source() -> tuple[bytes, str, bool] | None:
    st.sidebar.header("1 · Choose image")
    uploaded = st.sidebar.file_uploader("Upload a JPG or PNG", type=["jpg", "jpeg", "png"])
    use_sample = st.sidebar.checkbox("Use included OPT test photo", value=uploaded is None)
    if uploaded is not None:
        return uploaded.getvalue(), uploaded.name, False
    if use_sample and SAMPLE_IMAGE.exists():
        return SAMPLE_IMAGE.read_bytes(), SAMPLE_IMAGE.name, True
    return None


def _reset_for_image(image_sha: str) -> None:
    if st.session_state.get("image_sha") != image_sha:
        st.session_state.image_sha = image_sha
        st.session_state.circles = []
        st.session_state.references = {}
        st.session_state.canvas_revision = 0
        st.session_state.sample_refs_seeded = False
        st.session_state.show_reassigned_preview = False


def _fabric_objects(circles: list[dict], crop: tuple[int, int, int, int], scale: float) -> list[dict]:
    x0, y0, x1, y1 = crop
    objects = []
    for circle in circles:
        x, y, radius = float(circle["x"]), float(circle["y"]), float(circle["radius"])
        if not (x0 <= x <= x1 and y0 <= y <= y1):
            continue
        colour = ring_rgb(circle.get("group", "Uncalibrated"))
        stroke = "#%02x%02x%02x" % colour
        objects.append(
            {
                "type": "circle",
                "left": (x - radius - x0) * scale,
                "top": (y - radius - y0) * scale,
                "radius": radius * scale,
                "scaleX": 1.0,
                "scaleY": 1.0,
                "fill": "rgba(0,0,0,0.01)",
                "stroke": stroke,
                "strokeWidth": 3,
                "log_id": circle.get("id", ""),
                "uid": circle.get("uid", ""),
                "source": circle.get("source", "Manual"),
                "confidence": circle.get("confidence", 1.0),
            }
        )
    return objects


def _canvas_to_circles(
    json_data: dict | None,
    crop: tuple[int, int, int, int],
    scale: float,
    all_circles: list[dict],
) -> list[dict]:
    x0, y0, x1, y1 = crop
    outside = [
        dict(c)
        for c in all_circles
        if not (x0 <= float(c["x"]) <= x1 and y0 <= float(c["y"]) <= y1)
    ]
    inside = [
        dict(c)
        for c in all_circles
        if x0 <= float(c["x"]) <= x1 and y0 <= float(c["y"]) <= y1
    ]
    used_uids: set[str] = set()
    edited: list[dict] = []
    for obj in (json_data or {}).get("objects", []):
        if obj.get("type") != "circle":
            continue
        radius_x = float(obj.get("radius", 0)) * float(obj.get("scaleX", 1))
        radius_y = float(obj.get("radius", 0)) * float(obj.get("scaleY", 1))
        if radius_x < 3 or radius_y < 3:
            continue
        center_x = (float(obj.get("left", 0)) + radius_x) / scale + x0
        center_y = (float(obj.get("top", 0)) + radius_y) / scale + y0
        log_id = str(obj.get("log_id", ""))
        uid = str(obj.get("uid", ""))
        matched = next(
            (
                circle
                for circle in inside
                if (uid and str(circle.get("uid")) == uid)
                or (log_id and str(circle.get("id")) == log_id)
            ),
            None,
        )
        if matched is None and not (uid or log_id):
            available = [c for c in inside if str(c.get("uid")) not in used_uids]
            if available:
                nearest = min(
                    available,
                    key=lambda c: (float(c["x"]) - center_x) ** 2
                    + (float(c["y"]) - center_y) ** 2,
                )
                distance = (
                    (float(nearest["x"]) - center_x) ** 2
                    + (float(nearest["y"]) - center_y) ** 2
                ) ** 0.5
                edited_radius = (radius_x + radius_y) / (2.0 * scale)
                if distance <= max(12.0, 0.85 * max(float(nearest["radius"]), edited_radius)):
                    matched = nearest
        if matched is not None:
            used_uids.add(str(matched.get("uid")))
        edited.append(
            {
                **(matched or {}),
                "x": center_x,
                "y": center_y,
                "radius": (radius_x + radius_y) / (2.0 * scale),
                "id": log_id or (matched or {}).get("id", ""),
                "uid": uid or (matched or {}).get("uid", ""),
                "source": "Manual edit" if not obj.get("source") else obj.get("source"),
                "confidence": float(obj.get("confidence", 1.0)),
            }
        )
    return ensure_log_ids(outside + edited)


def _remap_references_after_reassignment(
    before: list[dict], after: list[dict], references: dict[str, float]
) -> dict[str, float]:
    reference_by_uid = {
        str(circle.get("uid")): float(references[circle["id"]])
        for circle in before
        if circle.get("id") in references and circle.get("uid")
    }
    return {
        circle["id"]: reference_by_uid[str(circle.get("uid"))]
        for circle in after
        if str(circle.get("uid")) in reference_by_uid
    }


def _seed_sample_references(circles: list[dict]) -> None:
    if st.session_state.get("sample_refs_seeded") or not circles:
        return
    references: dict[str, float] = {}
    for x, y, actual in SAMPLE_REFERENCE_HINTS:
        nearest = nearest_circle(circles, x, y)
        if nearest and ((nearest["x"] - x) ** 2 + (nearest["y"] - y) ** 2) ** 0.5 < 80:
            references[nearest["id"]] = actual
    st.session_state.references = references
    st.session_state.sample_refs_seeded = True


st.title("🪵 OPT Log Diameter Checker")
st.caption("Measure outside-bark log-end diameters, correct every ring, calibrate, and export.")

source = _read_source()
if source is None:
    st.info("Upload a JPG/PNG in the left panel, or enable the included OPT test photo.")
    st.stop()

image_bytes, image_name, is_sample = source
image_sha = hashlib.sha256(image_bytes).hexdigest()
_reset_for_image(image_sha)
try:
    original_pil = Image.open(BytesIO(image_bytes)).convert("RGB")
except Exception as exc:
    st.error(f"The image could not be opened: {exc}")
    st.stop()
original_rgb = pil_to_rgb(original_pil)
height, width = original_rgb.shape[:2]

st.sidebar.header("2 · Improve visibility")
brightness = st.sidebar.slider("Brightness", -50, 50, 0, 1)
contrast = st.sidebar.slider("Contrast", 0.5, 2.0, 1.05, 0.05)
sharpness = st.sidebar.slider("Sharpness", 0.5, 3.0, 1.2, 0.1)
show_edges = st.sidebar.checkbox("Show edge preview", value=False)
enhanced = enhance_image(original_rgb, brightness, contrast, sharpness)

st.sidebar.header("3 · Detection")
default_min = max(8, int(width * 0.014))
default_max = max(default_min + 5, int(width * 0.040))
min_radius = st.sidebar.slider("Smallest radius (px)", 5, max(10, width // 8), default_min)
max_radius = st.sidebar.slider(
    "Largest radius (px)", min_radius + 1, max(min_radius + 2, width // 5), default_max
)
hough_threshold = st.sidebar.slider(
    "Detection strictness", 18, 60, 38, help="Lower finds more rings; higher reduces false rings."
)
minimum_roundness = st.sidebar.slider(
    "Minimum roundness",
    0.40,
    0.90,
    0.70,
    0.05,
    help="Higher keeps only rings with more complete circular boundary support.",
)
search_y = st.sidebar.slider(
    "Vertical search area (%)", 0, 100, (18, 70) if is_sample else (0, 100)
)

top_actions = st.columns([1, 1, 3])
if top_actions[0].button("🔎 Auto-detect logs", type="primary", width="stretch"):
    with st.spinner("Fitting rings to visible outside-bark boundaries…"):
        try:
            st.session_state.circles = assign_log_ids(
                detect_log_ends(
                    enhanced,
                    min_radius,
                    max_radius,
                    hough_threshold,
                    search_y,
                    minimum_roundness,
                )
            )
            st.session_state.references = {}
            st.session_state.sample_refs_seeded = False
            st.session_state.show_reassigned_preview = False
            st.session_state.canvas_revision += 1
        except Exception as exc:
            st.error(f"Detection could not finish: {exc}")
if top_actions[1].button("Clear all rings", width="stretch"):
    st.session_state.circles = []
    st.session_state.references = {}
    st.session_state.show_reassigned_preview = False
    st.session_state.canvas_revision += 1

project_upload = st.sidebar.file_uploader("Reopen JSON project", type=["json"], key="project_upload")
if project_upload and st.sidebar.button("Load project measurements"):
    try:
        payload = load_project(project_upload.getvalue())
        if payload.get("image", {}).get("sha256") != image_sha:
            st.warning("The project was saved for a different image. Rings were not loaded.")
        else:
            st.session_state.circles = ensure_log_ids(payload["circles"])
            st.session_state.references = {
                str(k): float(v) for k, v in payload.get("reference_diameters_in", {}).items()
            }
            st.session_state.canvas_revision += 1
            st.success("Project measurements restored.")
    except Exception as exc:
        st.error(str(exc))

circles = ensure_log_ids(st.session_state.circles)
st.session_state.circles = circles
if is_sample:
    _seed_sample_references(circles)

tabs = st.tabs(["1 · Detect", "2 · Correct rings", "3 · Calibrate & review", "4 · Export"])

with tabs[0]:
    st.markdown('<div class="step-card"><b>Goal:</b> each outside-bark boundary should have exactly one ring. Auto-detection is only the starting point.</div>', unsafe_allow_html=True)
    left, right = st.columns([3, 2])
    left.image(enhanced, caption=f"{image_name} · {width} × {height} px", width="stretch")
    if show_edges:
        right.image(edge_preview(enhanced), caption="Edge preview", width="stretch")
    else:
        right.metric("Detected rings", len(circles))
        right.write("Continue to **Correct rings** to add, delete, move, or resize every result.")

with tabs[1]:
    st.markdown('<div class="step-card"><b>How to edit:</b> choose a focused region, then drag a ring to move/resize it. Select a ring and use the canvas trash icon to delete it. Switch to Add circle to draw a missing ring.</div>', unsafe_allow_html=True)
    sort_actions = st.columns([1, 2])
    if sort_actions[0].button("Reassign IDs row by row", type="secondary", width="stretch"):
        reassigned = assign_log_ids(circles)
        st.session_state.references = _remap_references_after_reassignment(
            circles, reassigned, st.session_state.references
        )
        st.session_state.circles = reassigned
        st.session_state.show_reassigned_preview = True
        st.session_state.canvas_revision += 1
        st.rerun()
    sort_actions[1].caption(
        "Apply canvas/table corrections first. Then this numbers from the upper-left row to the lower-right row."
    )
    if st.session_state.get("show_reassigned_preview") and circles:
        st.image(
            annotated_image(enhanced, measured_logs(circles, None), "PNG"),
            caption="Preview after row-by-row ID reassignment",
            width="stretch",
        )
    preset = st.radio("Focus region", ["Full image", "Left third", "Centre third", "Right third"], horizontal=True)
    preset_ranges = {
        "Full image": (0, width),
        "Left third": (0, width // 3 + width // 12),
        "Centre third": (width // 3 - width // 12, 2 * width // 3 + width // 12),
        "Right third": (2 * width // 3 - width // 12, width),
    }
    default_x = preset_ranges[preset]
    x_range = st.slider("Pan horizontally (pixels)", 0, width, default_x, key=f"xrange_{image_sha[:8]}_{preset}")
    y_range = st.slider("Focus vertically (pixels)", 0, height, (0, height), key=f"yrange_{image_sha[:8]}")
    zoom = st.slider("Canvas zoom", 0.7, 2.0, 1.0, 0.1)
    edit_mode = st.radio("Editing tool", ["Move / resize / delete", "Add circle"], horizontal=True)
    x0, x1 = x_range
    y0, y1 = y_range
    if x1 - x0 < 20 or y1 - y0 < 20:
        st.warning("Widen the focus region to at least 20 pixels.")
    else:
        base_scale = min(1350 / (x1 - x0), 760 / (y1 - y0))
        scale = max(0.15, base_scale * zoom)
        canvas_w = max(100, int((x1 - x0) * scale))
        canvas_h = max(100, int((y1 - y0) * scale))
        crop_image = Image.fromarray(enhanced[y0:y1, x0:x1]).resize((canvas_w, canvas_h))
        hover_circles = []
        for circle in circles:
            if x0 <= float(circle["x"]) <= x1 and y0 <= float(circle["y"]) <= y1:
                hover_circles.append(
                    {
                        **circle,
                        "x": (float(circle["x"]) - x0) * scale,
                        "y": (float(circle["y"]) - y0) * scale,
                        "radius": float(circle["radius"]) * scale,
                    }
                )
        if hover_circles:
            st.caption("Hover directly over a ring below to show its ID. Labels remain hidden otherwise.")
            st.html(
                hover_id_overlay(np.asarray(crop_image), hover_circles),
                unsafe_allow_javascript=True,
            )
            st.caption("Correction canvas")
        initial = {
            "version": "4.4.0",
            "objects": _fabric_objects(circles, (x0, y0, x1, y1), scale),
        }
        canvas_result = st_canvas(
            fill_color="rgba(0,0,0,0.01)",
            stroke_width=3,
            stroke_color="#00E5FF",
            background_image=crop_image,
            update_streamlit=True,
            height=canvas_h,
            width=canvas_w,
            drawing_mode="transform" if edit_mode.startswith("Move") else "circle",
            initial_drawing=initial,
            display_toolbar=True,
            key=f"canvas_{image_sha[:8]}_{preset}_{st.session_state.canvas_revision}_{edit_mode}",
        )
        if st.button("✓ Apply canvas corrections", type="primary"):
            updated_circles = _canvas_to_circles(
                canvas_result.json_data, (x0, y0, x1, y1), scale, circles
            )
            valid_ids = {circle["id"] for circle in updated_circles}
            st.session_state.circles = updated_circles
            st.session_state.references = {
                log_id: actual
                for log_id, actual in st.session_state.references.items()
                if log_id in valid_ids
            }
            st.session_state.show_reassigned_preview = False
            st.session_state.canvas_revision += 1
            st.rerun()

    with st.expander("Precise table corrections"):
        table = pd.DataFrame(
            [
                {
                    "ID": c["id"],
                    "Center X": round(c["x"], 1),
                    "Center Y": round(c["y"], 1),
                    "Diameter px": round(c["radius"] * 2, 1),
                    "Delete": False,
                }
                for c in circles
            ]
        )
        edited_table = st.data_editor(table, hide_index=True, width="stretch", disabled=["ID"])
        if st.button("Apply table corrections"):
            updated = []
            by_id = {c["id"]: c for c in circles}
            for row in edited_table.to_dict("records"):
                if row["Delete"]:
                    continue
                old = by_id[row["ID"]]
                updated.append(
                    {
                        **old,
                        "x": float(row["Center X"]),
                        "y": float(row["Center Y"]),
                        "radius": float(row["Diameter px"]) / 2.0,
                        "source": "Manual table edit",
                    }
                )
            valid_ids = {circle["id"] for circle in updated}
            st.session_state.circles = ensure_log_ids(updated)
            st.session_state.references = {
                log_id: actual
                for log_id, actual in st.session_state.references.items()
                if log_id in valid_ids
            }
            st.session_state.show_reassigned_preview = False
            st.session_state.canvas_revision += 1
            st.rerun()

with tabs[2]:
    st.markdown('<div class="step-card"><b>Calibration:</b> select one or two clearly visible reference logs, then enter their actual outside-bark diameter in inches.</div>', unsafe_allow_html=True)
    if circles:
        st.subheader("Annotated ID preview")
        st.image(
            annotated_image(enhanced, measured_logs(circles, None), "PNG"),
            caption="Use these printed IDs when selecting the reference log rows below.",
            width="stretch",
        )
    reference_rows = []
    for c in circles:
        actual = st.session_state.references.get(c["id"])
        reference_rows.append(
            {
                "Use as reference": actual is not None,
                "ID": c["id"],
                "Actual diameter (in)": actual if actual is not None else None,
                "Ring diameter (px)": round(c["radius"] * 2, 1),
            }
        )
    ref_df = pd.DataFrame(reference_rows)
    edited_refs = st.data_editor(
        ref_df,
        hide_index=True,
        width="stretch",
        disabled=["ID", "Ring diameter (px)"],
        column_config={
            "Use as reference": st.column_config.CheckboxColumn(),
            "Actual diameter (in)": st.column_config.NumberColumn(min_value=0.1, step=0.1, format="%.1f"),
        },
    )
    if st.button("Save reference selection", type="primary"):
        chosen = {}
        for row in edited_refs.to_dict("records"):
            if row["Use as reference"] and pd.notna(row["Actual diameter (in)"]):
                chosen[str(row["ID"])] = float(row["Actual diameter (in)"])
        if len(chosen) > 2:
            st.error("Choose no more than two reference logs.")
        elif not chosen:
            st.error("Choose at least one reference log.")
        else:
            st.session_state.references = chosen
            st.success("Reference selection saved.")

    calibration = None
    if st.session_state.references:
        try:
            calibration = calibration_from_references(circles, st.session_state.references)
        except ValueError as exc:
            st.warning(str(exc))
    logs = measured_logs(circles, calibration)
    if calibration:
        m1, m2, m3 = st.columns(3)
        m1.metric("Scale", f"{calibration['pixels_per_inch']:.2f} px/in")
        m2.metric("Estimated tolerance", f"±{calibration['estimated_tolerance_in']:.1f} in")
        m3.metric("Measured logs", len(logs))
        if calibration["perspective_warning"]:
            st.warning("⚠️ One reference cannot measure perspective variation. Add a second reference for a stronger tolerance estimate.")
        preview_bytes = annotated_image(enhanced, logs, "PNG")
        st.image(preview_bytes, caption="Review ring fit before exporting", width="stretch")
        result_df = pd.DataFrame(
            [
                {
                    "ID": log["id"],
                    "Outside-bark diameter (in)": log["diameter_in"],
                    "Tolerance (± in)": log["tolerance_in"],
                    "Group": log["group"],
                    "Diameter (px)": log["diameter_px"],
                    "Source": log.get("source", "Manual"),
                }
                for log in logs
            ]
        )
        st.dataframe(result_df, hide_index=True, width="stretch")
    else:
        st.info("Save at least one reference selection to calculate diameters.")

with tabs[3]:
    calibration = None
    if st.session_state.references:
        try:
            calibration = calibration_from_references(circles, st.session_state.references)
        except ValueError:
            pass
    logs = measured_logs(circles, calibration)
    if not calibration:
        st.warning("Calibrate the image before exporting measured diameters.")
    elif not logs:
        st.warning("There are no rings to export.")
    else:
        st.markdown('<div class="step-card"><b>Export package:</b> annotated image, Excel-friendly CSV, and a reopenable JSON project file.</div>', unsafe_allow_html=True)
        image_format = st.radio("Annotated image format", ["PNG", "JPG"], horizontal=True)
        annotated = annotated_image(enhanced, logs, image_format)
        csv_bytes = measurements_csv(logs, st.session_state.references)
        json_bytes = project_json(
            image_name,
            image_sha,
            original_rgb.shape,
            circles,
            st.session_state.references,
            calibration,
            {"brightness": brightness, "contrast": contrast, "sharpness": sharpness},
        )
        base_name = Path(image_name).stem + "_OPT_measured"
        c1, c2, c3 = st.columns(3)
        c1.download_button(
            f"Download annotated {image_format}",
            annotated,
            f"{base_name}.{image_format.lower()}",
            "image/jpeg" if image_format == "JPG" else "image/png",
            width="stretch",
        )
        c2.download_button(
            "Download measurements CSV",
            csv_bytes,
            f"{base_name}.csv",
            "text/csv",
            width="stretch",
        )
        c3.download_button(
            "Download project JSON",
            json_bytes,
            f"{base_name}.json",
            "application/json",
            width="stretch",
        )

st.sidebar.divider()
st.sidebar.caption("Colour rules: 🔴 <14.0 in · 🟡 14.0–16.0 in inclusive · 🔵 >16.0 in")
st.sidebar.caption("Always review and correct every outside-bark ring before using the measurements.")
