# OPT Log Diameter Checker

A Windows-friendly local app for detecting and measuring **outside-bark** log-end diameters from JPG/PNG photographs. Auto-detection is followed by a required visual correction workflow; the app is not a substitute for a calibrated field measurement.

## What the MVP does

- Uploads JPG, JPEG, and PNG images.
- Adjusts brightness, contrast, and sharpness; optionally previews edges.
- Proposes log-end rings with OpenCV Hough circles and a contour/ellipse fallback.
- Refines proposed radii toward visible outside-bark boundaries.
- Adds, deletes, moves, and resizes rings on an interactive canvas.
- Supports full-image and left/centre/right focused editing, horizontal/vertical panning, and canvas zoom.
- Calibrates from one or two reference logs in inches.
- Reports diameter to one decimal place and an estimated tolerance.
- Assigns unique IDs (`L01`, `L02`, ...).
- Colours rings: Red `<14.0 in`, Yellow `14.0–16.0 in` inclusive, Blue `>16.0 in`.
- Exports annotated PNG/JPG, Excel-friendly CSV, and a reopenable JSON project file.

## Easiest Windows setup

1. Install [Python for Windows](https://www.python.org/downloads/windows/) (Python 3.11 or newer). During setup, tick **Add python.exe to PATH**.
2. Double-click `install_and_run.bat` once. It creates a private environment, installs the required packages, and starts the app.
3. Later, double-click `run_app.bat`.
4. A browser window opens at `http://localhost:8501`.

The app runs locally. Uploaded photos are not sent anywhere by the application.

## Publish the prepared local repository to GitHub

The project is already initialized on the `main` branch with a local commit. GitHub repository names use a URL-safe slug, so the requested title **image capture and measure** becomes `image-capture-and-measure`.

After signing in with the official GitHub CLI, run this command from the project folder:

```powershell
gh repo create image-capture-and-measure --private --source . --remote origin --push --description "OPT Log Diameter Checker"
```

This creates a private repository, adds it as `origin`, and pushes `main`.

## Recommended measurement workflow

1. Upload the original, highest-resolution photo. Keep the camera as square to the log ends as practical.
2. Adjust visibility, then choose **Auto-detect logs**.
3. In **Correct rings**, inspect every log. The ring must follow the **outside bark**, not the pale wood or an internal mark.
4. Delete false rings, add missing rings, and move/resize imperfect rings. Use focused regions for close work.
5. In **Calibrate & review**, select one or two reference logs and enter actual outside-bark diameters.
6. Review the annotated preview and tolerance. A second reference helps expose scale/perspective disagreement.
7. Export the annotated image, CSV, and JSON project.

The included OPT photograph seeds the two user-identified references near the blue marks as **19.0 in** and **13.0 in**. Confirm both rings visually before using the calibration.

## Technical run and tests

```powershell
py -3 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pytest -q
.venv\Scripts\python -m streamlit run app.py
```

## Project structure

- `app.py` — Streamlit workflow and interactive correction canvas.
- `image_processing.py` — enhancement, Hough detection, ellipse fallback, boundary refinement.
- `measurement.py` — IDs, calibration, categories, and tolerance.
- `export_utils.py` — annotated image, CSV, and JSON exports.
- `sample_data/` — supplied OPT calibration photograph.
- `tests/` — calculation, export, and sample-image checks.

## Technical foundations

- [Streamlit application documentation](https://docs.streamlit.io/)
- [OpenCV Hough circle transform](https://docs.opencv.org/4.x/d4/d70/tutorial_hough_circle.html)
- [Compatible drawable canvas package](https://pypi.org/project/streamlit-drawable-canvas-fix/)

## Accuracy and tolerance

The displayed tolerance combines a two-pixel boundary-selection allowance with the disagreement between reference scales. It is an estimate, not a certified confidence interval. With one reference, perspective variation cannot be estimated; with two references close together, variation elsewhere in the photograph can still be larger. Photograph geometry and careful manual ring correction remain the main accuracy limits.

See `TEST_REPORT.md` for the validated MVP environment and sample-image checks.
