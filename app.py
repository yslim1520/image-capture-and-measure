"""Streamlit user interface for OPT Log Diameter Checker."""

from __future__ import annotations

import hashlib
import json
from io import BytesIO
from pathlib import Path

import altair as alt
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
from image_processing import (
    detect_log_ends,
    edge_preview,
    enhance_image,
    pil_to_rgb,
    resize_for_analysis,
)
from measurement import (
    assign_log_ids,
    calibration_from_references,
    ensure_log_ids,
    measurement_summary,
    measured_logs,
    nearest_circle,
    ring_rgb,
)
from photo_quality import assess_photo_quality


APP_DIR = Path(__file__).resolve().parent
SAMPLE_IMAGE = APP_DIR / "sample_data" / "opt_logs_reference.png"
SAMPLE_REFERENCE_HINTS = [(947.0, 395.0, 19.0), (911.0, 483.0, 13.0)]


st.set_page_config(
    page_title="OPT Log Diameter Checker",
    page_icon="🪵",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
      .step-card {background:#f7f9fb;border:1px solid #d9e1e8;border-radius:12px;padding:12px 16px;margin-bottom:12px;}
      .small-note {color:#536270;font-size:0.9rem;}
      div[data-testid="stMetric"] {background:#f7f9fb;border:1px solid #e3e8ee;border-radius:10px;padding:8px 12px;}
      div[data-testid="stFileUploader"] section {min-height:96px;border:2px dashed #1B6B50;border-radius:14px;}
      .mobile-guide {background:#eef7f2;border-left:5px solid #1B6B50;border-radius:10px;padding:12px 14px;margin:8px 0 14px;}
      @media (max-width: 700px) {
        .block-container {padding:0.7rem 0.75rem 2rem;}
        h1 {font-size:1.65rem !important;line-height:1.2 !important;}
        h2 {font-size:1.3rem !important;}
        .stButton > button, .stDownloadButton > button {min-height:48px;font-size:1rem;}
        div[data-testid="stTabs"] button {min-height:48px;padding-left:10px;padding-right:10px;}
        .step-card {font-size:0.95rem;padding:10px 12px;}
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def _read_source() -> tuple[bytes, str, bool] | None:
    st.subheader("Choose an existing log photo")
    st.markdown(
        """
        <div class="mobile-guide">
        <b>Before selecting the photo</b><br>
        • Use the original, high-resolution landscape photo—not a screenshot or messaging-app copy.<br>
        • Keep the phone steady, the lens clean, and the log ends bright without strong glare or deep shadow.<br>
        • Stand as square as practical to the log ends and include the complete stack.<br>
        • Before taking the photo, choose one or two clear reference logs, measure their <b>outside-bark</b>
          diameter, and mark them so you can identify them later.<br>
        • Reference logs positioned apart from each other help reveal scale changes across the image.
        </div>
        """,
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        "Select photo from phone or computer",
        type=["jpg", "jpeg", "png"],
        help="On Android, tap Browse files and choose the original photo from Gallery or Files.",
    )
    use_sample = st.checkbox("Try the included OPT example photo", value=False)
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
        st.session_state.reference_widget_revision = 0
        st.session_state.sample_refs_seeded = False
        st.session_state.show_reassigned_preview = False


def _clear_reference_widgets(image_sha: str) -> None:
    st.session_state.reference_widget_revision = (
        int(st.session_state.get("reference_widget_revision", 0)) + 1
    )


def _reference_widget_suffix(image_sha: str) -> str:
    return f"{image_sha[:10]}_{int(st.session_state.get('reference_widget_revision', 0))}"


def _sync_reference_widgets(image_sha: str, references: dict[str, float]) -> None:
    _clear_reference_widgets(image_sha)
    items = list(references.items())[:2]
    suffix = _reference_widget_suffix(image_sha)
    if items:
        st.session_state[f"reference_1_id_{suffix}"] = items[0][0]
        st.session_state[f"reference_1_diameter_{suffix}"] = float(items[0][1])
    if len(items) > 1:
        st.session_state[f"reference_2_id_{suffix}"] = items[1][0]
        st.session_state[f"reference_2_diameter_{suffix}"] = float(items[1][1])


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
    _sync_reference_widgets(st.session_state.image_sha, references)


st.title("🪵 OPT Log Diameter Checker")
st.caption("Android-friendly photo measurement: select, detect, correct, calibrate, and export.")

source = _read_source()
if source is None:
    st.info("Select a JPG or PNG above to begin. The app processes it in the current session and does not intentionally save it to persistent storage.")
    st.stop()

image_bytes, image_name, is_sample = source
image_sha = hashlib.sha256(image_bytes).hexdigest()
_reset_for_image(image_sha)
try:
    original_pil = Image.open(BytesIO(image_bytes)).convert("RGB")
except Exception as exc:
    st.error(f"The image could not be opened: {exc}")
    st.stop()
source_rgb = pil_to_rgb(original_pil)
original_rgb, analysis_scale = resize_for_analysis(source_rgb)
height, width = original_rgb.shape[:2]

quality = assess_photo_quality(source_rgb)
with st.expander("Photo quality check", expanded=not quality["ready"]):
    if quality["ready"]:
        st.success("This photo passes the app's basic resolution, lighting, sharpness, and orientation checks.")
    else:
        st.warning("This photo may still work, but retaking it could improve ring fitting.")
    st.caption(
        f"Resolution: {quality['width']} × {quality['height']} px ({quality['megapixels']:.1f} MP) · "
        f"Brightness signal: {quality['mean_brightness']:.1f} · "
        f"Sharpness signal: {quality['sharpness_score']:.1f}"
    )
    for guidance in quality["guidance"]:
        st.write(f"• {guidance}")
    if analysis_scale < 1.0:
        st.write(
            f"• The app created a {width} × {height} px working copy for faster cloud processing; "
            "the original upload was not overwritten."
        )
    st.caption("These checks are best-guess photo guidance, not a guarantee of measurement accuracy.")
del source_rgb

default_min = max(8, int(width * 0.014))
default_max = max(default_min + 5, int(width * 0.040))
with st.expander("Optional image and detection settings"):
    st.caption("The defaults are a precision-first starting point. Change these only when rings are missed or false rings remain.")
    brightness = st.slider("Brightness", -50, 50, 0, 1)
    contrast = st.slider("Contrast", 0.5, 2.0, 1.05, 0.05)
    sharpness = st.slider("Sharpness", 0.5, 3.0, 1.2, 0.1)
    show_edges = st.checkbox("Show edge preview", value=False)
    min_radius = st.slider("Smallest radius (px)", 5, max(10, width // 8), default_min)
    max_radius = st.slider(
        "Largest radius (px)", min_radius + 1, max(min_radius + 2, width // 5), default_max
    )
    hough_threshold = st.slider(
        "Detection strictness", 18, 60, 38, help="Lower finds more rings; higher reduces false rings."
    )
    minimum_roundness = st.slider(
        "Minimum roundness",
        0.40,
        0.90,
        0.70,
        0.05,
        help="Higher keeps only rings with more complete circular boundary support.",
    )
    search_y = st.slider(
        "Vertical search area (%)", 0, 100, (18, 70) if is_sample else (0, 100)
    )

enhanced = enhance_image(original_rgb, brightness, contrast, sharpness)

if st.button("Auto-detect log ends", type="primary", width="stretch"):
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
            _clear_reference_widgets(image_sha)
            st.session_state.sample_refs_seeded = False
            st.session_state.show_reassigned_preview = False
            st.session_state.canvas_revision += 1
        except Exception as exc:
            st.error(f"Detection could not finish: {exc}")
if st.button("Clear all rings", width="stretch"):
    st.session_state.circles = []
    st.session_state.references = {}
    st.session_state.show_reassigned_preview = False
    st.session_state.canvas_revision += 1

with st.expander("Open a previously saved project"):
    project_upload = st.file_uploader("Select project JSON", type=["json"], key="project_upload")
    if project_upload and st.button("Load project measurements", width="stretch"):
        try:
            payload = load_project(project_upload.getvalue())
            if payload.get("image", {}).get("sha256") != image_sha:
                st.warning("The project was saved for a different image. Rings were not loaded.")
            else:
                st.session_state.circles = ensure_log_ids(payload["circles"])
                st.session_state.references = {
                    str(k): float(v) for k, v in payload.get("reference_diameters_in", {}).items()
                }
                _sync_reference_widgets(image_sha, st.session_state.references)
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
    st.image(enhanced, caption=f"{image_name} · {width} × {height} px", width="stretch")
    if show_edges:
        st.image(edge_preview(enhanced), caption="Edge preview", width="stretch")
    else:
        st.metric("Detected rings", len(circles))
        st.write("Continue to **Correct rings** to add, delete, move, or resize every result.")

with tabs[1]:
    st.markdown('<div class="step-card"><b>How to edit:</b> choose a focused region. On a phone, tap a ring to select it, drag its centre to move it, and drag a corner handle to resize it. Use the trash icon to delete it, or switch to Add circle for a missing log.</div>', unsafe_allow_html=True)
    if st.button("Reassign IDs row by row", type="secondary", width="stretch"):
        reassigned = assign_log_ids(circles)
        st.session_state.references = _remap_references_after_reassignment(
            circles, reassigned, st.session_state.references
        )
        _sync_reference_widgets(image_sha, st.session_state.references)
        st.session_state.circles = reassigned
        st.session_state.show_reassigned_preview = True
        st.session_state.canvas_revision += 1
        st.rerun()
    st.caption(
        "Apply canvas/table corrections first. Then use this button to number logs from the upper-left row to the lower-right row."
    )
    editor_profile = st.radio(
        "Editing screen size",
        ["Phone", "Tablet", "Desktop"],
        horizontal=True,
        help="Phone keeps the touch canvas narrow enough for an Android screen.",
    )
    preset = st.radio(
        "Focus region",
        ["Full image", "Left third", "Centre third", "Right third"],
        index=1 if editor_profile == "Phone" else 0,
        horizontal=True,
    )
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
        target_width, target_height = {
            "Phone": (360, 620),
            "Tablet": (720, 720),
            "Desktop": (1200, 760),
        }[editor_profile]
        base_scale = min(target_width / (x1 - x0), target_height / (y1 - y0))
        scale = max(0.15, base_scale * zoom)
        canvas_w = max(100, int((x1 - x0) * scale))
        canvas_h = max(100, int((y1 - y0) * scale))
        st.markdown(
            f'<style>iframe[title="streamlit_drawable_canvas.st_canvas"] '
            f'{{height:{canvas_h + 12}px !important;min-height:{canvas_h + 12}px !important;}}</style>',
            unsafe_allow_html=True,
        )
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
            st.caption("Tap a ring on a phone, or hover over it on a computer, to show its ID.")
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

    if circles:
        st.subheader("Annotated ID preview")
        if st.session_state.get("show_reassigned_preview"):
            st.success("IDs are reassigned row by row, from upper-left to lower-right.")
        st.image(
            annotated_image(enhanced, measured_logs(circles, None), "PNG"),
            caption="Final on-the-spot check after corrections and ID reassignment.",
            width="stretch",
        )
        st.caption("Confirm that every outside-bark ring and printed ID matches the intended log before calibration.")

with tabs[2]:
    st.markdown('<div class="step-card"><b>Calibration:</b> select one or two clearly visible reference logs, then enter their actual outside-bark diameter in inches.</div>', unsafe_allow_html=True)
    log_ids = [str(circle["id"]) for circle in circles]
    saved_references = [
        (log_id, float(actual))
        for log_id, actual in st.session_state.references.items()
        if log_id in log_ids
    ]
    if saved_references:
        saved_summary = " · ".join(
            f"{log_id} = {actual:.1f} in" for log_id, actual in saved_references
        )
        st.success(f"Active reference calibration: {saved_summary}")
        edit_references = st.checkbox(
            "Change saved reference logs",
            value=False,
            key=f"edit_references_{image_sha[:10]}_{st.session_state.get('reference_widget_revision', 0)}",
        )
    else:
        edit_references = True

    if edit_references:
        first_saved = saved_references[0] if saved_references else (None, 0.0)
        second_saved = saved_references[1] if len(saved_references) > 1 else (None, 0.0)
        first_options = ["Choose a log ID"] + log_ids
        second_options = ["No second reference"] + log_ids
        reference_suffix = _reference_widget_suffix(image_sha)
        first_id_key = f"reference_1_id_{reference_suffix}"
        first_diameter_key = f"reference_1_diameter_{reference_suffix}"
        second_id_key = f"reference_2_id_{reference_suffix}"
        second_diameter_key = f"reference_2_diameter_{reference_suffix}"
        st.session_state.setdefault(first_id_key, first_saved[0] or first_options[0])
        st.session_state.setdefault(first_diameter_key, float(first_saved[1]))
        st.session_state.setdefault(second_id_key, second_saved[0] or second_options[0])
        st.session_state.setdefault(second_diameter_key, float(second_saved[1]))

        with st.container(border=True):
            st.markdown("**Reference log 1**")
            first_id = st.selectbox("Log ID", first_options, key=first_id_key)
            first_actual = st.number_input(
                "Actual outside-bark diameter (in)",
                min_value=0.0,
                step=0.1,
                format="%.1f",
                key=first_diameter_key,
            )

        with st.container(border=True):
            st.markdown("**Reference log 2 - optional but recommended**")
            second_id = st.selectbox("Second log ID", second_options, key=second_id_key)
            second_actual = st.number_input(
                "Second actual outside-bark diameter (in)",
                min_value=0.0,
                step=0.1,
                format="%.1f",
                key=second_diameter_key,
            )

        if st.button("Save reference logs", type="primary", width="stretch"):
            chosen: dict[str, float] = {}
            if first_id != first_options[0] and first_actual > 0:
                chosen[first_id] = float(first_actual)
            if second_id != second_options[0] and second_actual > 0:
                chosen[second_id] = float(second_actual)
            if not chosen:
                st.error("Choose at least one log ID and enter its measured diameter.")
            elif first_id == second_id and second_id != second_options[0]:
                st.error("Choose two different logs, or remove the second reference.")
            else:
                st.session_state.references = chosen
                st.success("Reference logs saved.")

    calibration = None
    if st.session_state.references:
        try:
            calibration = calibration_from_references(circles, st.session_state.references)
        except ValueError as exc:
            st.warning(str(exc))
    logs = measured_logs(circles, calibration)
    if calibration:
        st.metric("Scale", f"{calibration['pixels_per_inch']:.2f} px/in")
        st.metric("Estimated tolerance", f"±{calibration['estimated_tolerance_in']:.1f} in")
        st.metric("Measured logs", len(logs))
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

        st.subheader("Measurement summary")
        summary_rows = measurement_summary(logs)
        summary_df = pd.DataFrame(summary_rows)
        st.dataframe(summary_df, hide_index=True, width="stretch")

        st.subheader("Measurement graph")
        graph_choice = st.selectbox(
            "Graph view",
            ["Diameter by log ID", "Diameter distribution", "Log count by colour range"],
        )
        colour_scale = alt.Scale(
            domain=["Blue", "Yellow", "Red"],
            range=["#1976FF", "#E0A800", "#EE2F2F"],
        )
        chart_df = pd.DataFrame(
            [
                {
                    "ID": log["id"],
                    "Diameter (in)": log["diameter_in"],
                    "Range": log["group"],
                }
                for log in logs
            ]
        )
        if graph_choice == "Diameter by log ID":
            chart = (
                alt.Chart(chart_df)
                .mark_bar()
                .encode(
                    x=alt.X("ID:N", sort=None, title="Log ID", axis=alt.Axis(labelAngle=-45)),
                    y=alt.Y("Diameter (in):Q", title="Diameter (in)"),
                    color=alt.Color("Range:N", scale=colour_scale, title="Range"),
                    tooltip=["ID:N", "Diameter (in):Q", "Range:N"],
                )
            )
        elif graph_choice == "Diameter distribution":
            chart = (
                alt.Chart(chart_df)
                .mark_bar()
                .encode(
                    x=alt.X("Diameter (in):Q", bin=alt.Bin(maxbins=12), title="Diameter (in)"),
                    y=alt.Y("count():Q", title="Logs"),
                    color=alt.Color("Range:N", scale=colour_scale, title="Range"),
                    tooltip=[alt.Tooltip("count():Q", title="Logs"), "Range:N"],
                )
            )
        else:
            count_df = summary_df.iloc[1:][["Range", "Logs"]].copy()
            count_df["Colour"] = ["Blue", "Yellow", "Red"]
            chart = (
                alt.Chart(count_df)
                .mark_bar()
                .encode(
                    x=alt.X("Logs:Q", title="Logs"),
                    y=alt.Y(
                        "Range:N",
                        sort=["Blue >16", "Yellow 14–16", "Red <14"],
                        title=None,
                    ),
                    color=alt.Color("Colour:N", scale=colour_scale, legend=None),
                    tooltip=["Range:N", "Logs:Q"],
                )
            )
        st.altair_chart(chart, width="stretch")
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
        st.download_button(
            f"Download annotated {image_format}",
            annotated,
            f"{base_name}.{image_format.lower()}",
            "image/jpeg" if image_format == "JPG" else "image/png",
            width="stretch",
        )
        st.download_button(
            "Download measurements CSV",
            csv_bytes,
            f"{base_name}.csv",
            "text/csv",
            width="stretch",
        )
        st.download_button(
            "Download project JSON",
            json_bytes,
            f"{base_name}.json",
            "application/json",
            width="stretch",
        )

st.divider()
st.caption("Colour rules: 🔴 <14.0 in · 🟡 14.0–16.0 in inclusive · 🔵 >16.0 in")
st.caption("Always review and correct every outside-bark ring before using the measurements.")
