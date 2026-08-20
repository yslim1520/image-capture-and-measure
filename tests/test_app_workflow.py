from pathlib import Path

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def _sample_app() -> AppTest:
    app = AppTest.from_file(APP_PATH, default_timeout=30).run()
    app.checkbox[0].check().run()
    assert not list(app.exception)
    return app


def test_preselect_mode_exposes_touch_controls():
    app = _sample_app()
    preselect = next(widget for widget in app.toggle if widget.label.startswith("Preselect mode"))

    preselect.set_value(True).run()

    assert not list(app.exception)
    assert any(metric.label == "Marked logs" for metric in app.metric)
    assert any(button.label == "Clear preselection points" for button in app.button)
    size_control = next(
        widget for widget in app.segmented_control if widget.label == "Preselect canvas size"
    )
    assert size_control.options == ["Phone", "Tablet", "Desktop"]


def test_correction_tools_are_separate_and_add_is_tap_sized():
    app = _sample_app()
    tool = next(
        widget for widget in app.segmented_control if widget.label == "Correction tool"
    )
    assert tool.options == ["Move", "Add", "Resize", "Delete"]

    tool.set_value("Add").run()

    assert not list(app.exception)
    assert any(slider.label == "New circle diameter (px)" for slider in app.slider)
    add_button = next(button for button in app.button if button.label == "Apply added circles")
    assert add_button.disabled
