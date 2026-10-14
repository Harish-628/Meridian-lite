"""
Unit tests for mock AI damage assessment engine (models/predict.py).
"""

from pathlib import Path
import pytest
from models.predict import EXACT_DAMAGE_CLASSES, run_inference
from utils.image_utils import create_demo_images


@pytest.fixture(scope="module")
def sample_images(tmp_path_factory):
    """Generate temporary sample images for inference testing."""
    test_dir = tmp_path_factory.mktemp("infer_data")
    pre_path, post_path = create_demo_images(test_dir)
    return pre_path, post_path


def test_inference_output_type(sample_images):
    """Verify inference returns a valid dict with FeatureCollection structure."""
    pre_path, post_path = sample_images
    result = run_inference(
        pre_image_path=pre_path,
        post_image_path=post_path,
        center_lat=13.0827,
        center_lon=80.2707,
        seed=42,
        building_count=18,
    )

    assert isinstance(result, dict)
    assert result.get("type") == "FeatureCollection"
    assert "metadata" in result
    assert "features" in result
    assert result["metadata"]["is_simulated"] is True
    assert result["metadata"]["stage"] == "Stage 1"
    assert result["metadata"]["model_type"] == "simulated_xview2_baseline"


def test_inference_building_count_clamping(sample_images):
    """Verify requested building count is clamped safely between 15 and 20."""
    pre_path, post_path = sample_images

    # Below minimum (10 -> 15)
    res_low = run_inference(pre_path, post_path, 13.0, 80.0, seed=1, building_count=10)
    assert len(res_low["features"]) == 15

    # Above maximum (25 -> 20)
    res_high = run_inference(pre_path, post_path, 13.0, 80.0, seed=1, building_count=25)
    assert len(res_high["features"]) == 20

    # Within range (17 -> 17)
    res_mid = run_inference(pre_path, post_path, 13.0, 80.0, seed=1, building_count=17)
    assert len(res_mid["features"]) == 17


def test_inference_reproducibility(sample_images):
    """Verify that runs with the same seed generate bit-for-bit identical GeoJSON features."""
    pre_path, post_path = sample_images
    run1 = run_inference(pre_path, post_path, 13.0827, 80.2707, seed=99, building_count=18)
    run2 = run_inference(pre_path, post_path, 13.0827, 80.2707, seed=99, building_count=18)

    assert run1 == run2


def test_inference_different_seeds(sample_images):
    """Verify different seeds yield varied results while maintaining valid structure."""
    pre_path, post_path = sample_images
    runA = run_inference(pre_path, post_path, 13.0827, 80.2707, seed=10, building_count=18)
    runB = run_inference(pre_path, post_path, 13.0827, 80.2707, seed=20, building_count=18)

    classesA = [f["properties"]["damage_class"] for f in runA["features"]]
    classesB = [f["properties"]["damage_class"] for f in runB["features"]]
    assert classesA != classesB


def test_damage_classes_and_confidences(sample_images):
    """Verify all features have valid damage classes and confidences within [0.75, 0.99]."""
    pre_path, post_path = sample_images
    result = run_inference(pre_path, post_path, 13.0827, 80.2707, seed=42, building_count=20)

    for feature in result["features"]:
        props = feature["properties"]
        assert props["damage_class"] in EXACT_DAMAGE_CLASSES
        conf = props["confidence"]
        assert 0.75 <= conf <= 0.99
        assert props["is_simulated"] is True
        assert props["source"] == "simulated_xview2_baseline"


def test_polygon_closure_and_coordinate_order(sample_images):
    """Verify polygon coordinates are [lon, lat], valid ranges, and strictly closed."""
    pre_path, post_path = sample_images
    result = run_inference(pre_path, post_path, 13.0827, 80.2707, seed=42, building_count=18)

    for feature in result["features"]:
        geom = feature["geometry"]
        assert geom["type"] == "Polygon"
        coordinates = geom["coordinates"]
        assert len(coordinates) >= 1

        ring = coordinates[0]
        assert len(ring) == 5  # 4 vertices + closing point
        assert ring[0] == ring[-1], "Exterior polygon ring must be closed"

        for pt in ring:
            lon, lat = pt[0], pt[1]
            assert -180.0 <= lon <= 180.0
            assert -90.0 <= lat <= 90.0
            # Around expected center (13.0827, 80.2707)
            assert abs(lat - 13.0827) < 0.1
            assert abs(lon - 80.2707) < 0.1
