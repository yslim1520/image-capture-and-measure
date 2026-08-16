# MVP test report

Tested on Windows on 17 August 2026 with Python 3.12.13.

## Automated checks

- Thirteen calculation, category-boundary, CSV, JSON, image-export, stable-ID, tap/hover-label, photo-quality, large-image resizing, roundness, and sample-detection tests pass.
- The Streamlit application test runner completes with no application exceptions.
- Streamlit 1.61.1, OpenCV 5.0.0, and the interactive drawable-canvas component import successfully.

## Android-sized browser check

- The responsive workflow was checked at a 390 × 844 pixel viewport.
- Gallery-first upload guidance and the mobile photo-quality panel render without horizontal page overflow.
- The phone editing profile uses a 360 pixel target canvas and focused image regions.
- The drawable correction frame remains visible with the current Streamlit version.
- Tapping a detected ring displays its `Lxx` ID, while tapping away hides it.
- The calibration screen shows the active 19.0 in and 13.0 in reference values.

## Supplied OPT image

- Image size: 1841 × 772 pixels.
- Default precision-first sample settings propose 50 editable candidate rings.
- The roundness check rejects the known background/road circle while keeping both marked references.
- A candidate is found near the user-marked No. 1 log and seeded with 19.0 in.
- A candidate is found near the user-marked No. 2 log and seeded with 13.0 in.
- The seeded two-reference calibration reports 5.90 px/in and an estimated ±0.6 in tolerance before further manual correction.

The 50 candidates are not claimed to be 50 confirmed logs. The photograph was used to verify that detection reaches the marked references and provides a practical correction starting point. A user must add missed logs, delete false rings, and fit every ring to the outside bark before treating exported measurements as final.
