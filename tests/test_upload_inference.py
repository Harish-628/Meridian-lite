"""Unit and integration tests for Upload mode real model inference."""

from pathlib import Path
import pytest
from models.real_predict import run_real_inference
from utils.metrics import calculate_summary_metrics, features_to_dataframe
from utils.priority import compute_priority_zones
from utils.image_utils import create_damage_overlay
from app.ui.map_view import build_folium_map


@pytest.fixture
def sample_upload_images(tmp_path):
    """Provide valid pre- and post-disaster images for upload testing."""
    import cv2
    import numpy as np

    pre = np.full((128, 128, 3), 100, dtype=np.uint8)
    # Add a roof rectangle
    cv2.rectangle(pre, (30, 30), (80, 80), (220, 220, 220), -1)

    post = pre.copy()
    # Destroy half of the roof rectangle
    cv2.rectangle(post, (30, 30), (80, 55), (40, 40, 40), -1)

    pre_path = tmp_path / "custom_pre.png"
    post_path = tmp_path / "custom_post.png"
    cv2.imwrite(str(pre_path), pre)
    cv2.imwrite(str(post_path), post)
    return pre_path, post_path


def test_upload_mode_real_onnx_inference(sample_upload_images):
    """Verify that real ONNX inference executes on custom uploaded image pairs."""
    pre_path, post_path = sample_upload_images
    fc = run_real_inference(
        pre_image_path=pre_path,
        post_image_path=post_path,
        center_lat=25.7617,
        center_lon=-80.1918,
        confidence_threshold=0.20,
        backend_name="real_damage_model",
        is_custom_upload=True,
    )
    assert fc["type"] == "FeatureCollection"
    assert "metadata" in fc
    assert fc["metadata"]["is_simulated"] is False
    assert "real_damage_model" in fc["metadata"]["backend_name"]


def test_upload_mode_classical_inference(sample_upload_images):
    """Verify classical CV backend also executes on custom uploaded image pairs."""
    pre_path, post_path = sample_upload_images
    fc = run_real_inference(
        pre_image_path=pre_path,
        post_image_path=post_path,
        center_lat=25.7617,
        center_lon=-80.1918,
        confidence_threshold=0.20,
        backend_name="classical_change_detection",
        is_custom_upload=True,
    )
    assert fc["type"] == "FeatureCollection"
    assert fc["metadata"]["is_simulated"] is False


def test_upload_mode_dataframe_and_metrics(sample_upload_images):
    """Verify that detected features convert cleanly to DataFrame with non-empty feature_id and real flags."""
    pre_path, post_path = sample_upload_images
    fc = run_real_inference(
        pre_image_path=pre_path,
        post_image_path=post_path,
        center_lat=25.7617,
        center_lon=-80.1918,
        confidence_threshold=0.20,
        backend_name="real_damage_model",
        is_custom_upload=True,
    )
    metrics = calculate_summary_metrics(fc)
    assert "total_buildings" in metrics
    assert metrics["is_simulated"] is False

    df = features_to_dataframe(fc)
    assert not df.empty
    assert "feature_id" in df.columns
    assert "is_simulated" in df.columns
    assert df["feature_id"].iloc[0] != ""
    assert df["is_simulated"].iloc[0] == False