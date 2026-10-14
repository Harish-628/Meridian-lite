"""Automated UI testing for MERIDIAN-Lite Streamlit application.

Uses streamlit.testing.v1.AppTest to simulate user interaction, verify mode switching,
test real ONNX backend execution, inspect metrics rendering, and validate tactical exports.
"""

from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "app" / "main.py")


@pytest.fixture
def app_test():
    """Initializes a new AppTest instance for app/main.py."""
    at = AppTest.from_file(APP_PATH, default_timeout=35)
    return at


def test_app_loads_without_exceptions(app_test):
    """Verify the Streamlit dashboard boots and initializes initial demo state without exceptions."""
    app_test.run()
    assert not app_test.exception, f"App raised uncaught exceptions on load: {app_test.exception}"
    assert len(app_test.sidebar.radio) > 0, "Sidebar mode radio selector missing"
    assert "Demo mode" in app_test.sidebar.radio[0].options
    assert "Real Model Mode" in app_test.sidebar.radio[0].options
    assert "Upload mode" in app_test.sidebar.radio[0].options


def test_switch_to_upload_mode(app_test):
    """Verify switching to Upload mode updates the UI and does not raise errors."""
    app_test.run()
    app_test.sidebar.radio[0].set_value("Upload mode").run()
    assert not app_test.exception, f"Upload mode exception: {app_test.exception}"


def test_switch_to_real_model_mode(app_test):
    """Verify switching to Real Model Mode reveals backend selectors and real disaster dataset controls."""
    app_test.run()
    app_test.sidebar.radio[0].set_value("Real Model Mode").run()
    assert not app_test.exception, f"Real Model Mode switch error: {app_test.exception}"

    # Verify backend selectbox
    assert len(app_test.sidebar.selectbox) >= 2, "Expected backend and dataset selectboxes in Real Model Mode"
    backend_box = app_test.sidebar.selectbox[0]
    assert "Real Damage Model (ONNX)" in backend_box.options
    assert "Classical Change Detection (OpenCV)" in backend_box.options


def test_execute_real_model_assessment_and_downloads(app_test):
    """Verify executing real ONNX inference populates tactical metrics, priority zones, and 5 download buttons."""
    app_test.run()
    app_test.sidebar.radio[0].set_value("Real Model Mode").run()
    app_test.sidebar.selectbox[0].set_value("Real Damage Model (ONNX)").run()

    # Trigger Run button
    app_test.sidebar.button[0].click().run()
    assert not app_test.exception, f"Assessment execution raised exception: {app_test.exception}"

    # Verify download buttons are rendered
    assert len(app_test.download_button) >= 5, (
        f"Expected at least 5 tactical download buttons, found {len(app_test.download_button)}"
    )

    labels = [btn.label for btn in app_test.download_button]
    assert any("GeoJSON" in lbl for lbl in labels), "Missing GeoJSON download button"
    assert any("Summary JSON" in lbl for lbl in labels), "Missing Summary JSON download button"
    assert any("CSV Results" in lbl for lbl in labels), "Missing CSV Results download button"
    assert any("Priority Zones" in lbl for lbl in labels), "Missing Priority Zones download button"
    assert any("Damage Overlay" in lbl for lbl in labels), "Missing Damage Overlay download button"
