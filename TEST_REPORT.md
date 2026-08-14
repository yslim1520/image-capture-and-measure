# MVP test report

Tested on Windows on 15 August 2026 with Python 3.12.13.

## Automated checks

- Ten calculation, category-boundary, CSV, JSON, image-export, stable-ID, hover-label, roundness, and sample-detection tests pass.
- The Streamlit application test runner completes with no application exceptions.
- Streamlit 1.61.1, OpenCV 5.0.0, and the interactive drawable-canvas component import successfully.

## Supplied OPT image

- Image size: 1840 × 772 pixels.
- Default precision-first sample settings propose 50 editable candidate rings.
- The roundness check rejects the known background/road circle while keeping both marked references.
- A candidate is found near the user-marked No. 1 log and seeded with 19.0 in.
- A candidate is found near the user-marked No. 2 log and seeded with 13.0 in.
- The seeded two-reference calibration reports 5.90 px/in and an estimated ±0.6 in tolerance before further manual correction.

The 50 candidates are not claimed to be 50 confirmed logs. The photograph was used to verify that detection reaches the marked references and provides a practical correction starting point. A user must add missed logs, delete false rings, and fit every ring to the outside bark before treating exported measurements as final.
