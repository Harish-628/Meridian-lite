"""
Unit tests for application module imports and end-to-end demo pipeline execution.
"""

from pathlib import Path
import pytest


def test_module_imports():
    """Verify all project modules import cleanly without circular dependencies."""
    import utils.paths
    import utils.logging_utils
    import utils.image_utils
    import utils.geojson_utils
    import utils.metrics
    import models.predict

    assert utils.paths.PROJECT_ROOT.exists()
    assert utils.logging_utils.get_logger("test") is not None


def test_end_to_end_demo_pipeline(tmp_path: Path):
    """Verify full end-to-end execution of the demo assessment pipeline."""
    from utils.image_utils import create_demo_images, create_damage_overlay
    from models.predict import run_inference
    from utils.geojson_utils import validate_feature_collection, save_geojson, load_geojson
    from utils.metrics import calculate_summary_metrics, features_to_dataframe

    # 1. Generate demo imagery
    pre_path, post_path = create_demo_images(tmp_path)
    assert pre_path.exists()
    assert post_path.exists()

    # 2. Run mock inference
    geojson_result = run_inference(
        pre_image_path=pre_path,
        post_image_path=post_path,
        center_lat=13.0827,
        center_lon=80.2707,
        seed=42,
        building_count=18,
    )
    assert len(geojson_result["features"]) == 18

    # 3. Validate GeoJSON
    validate_feature_collection(geojson_result)

    # 4. Generate visual difference overlay
    overlay_out = tmp_path / "damage_overlay.png"
    overlay, out_p = create_damage_overlay(pre_path, post_path, overlay_out)
    assert out_p.exists()
    assert overlay.shape[0] > 0

    # 5. Calculate metrics
    metrics = calculate_summary_metrics(geojson_result)
    assert metrics["total_buildings"] == 18
    assert metrics["damaged_count"] >= 0
    assert 0.0 <= metrics["damaged_percentage"] <= 100.0

    # 6. Convert to DataFrame
    df = features_to_dataframe(geojson_result)
    assert len(df) == 18

    # 7. Save and reload
    save_path = tmp_path / "output.geojson"
    save_geojson(geojson_result, save_path)
    reloaded = load_geojson(save_path)
    assert len(reloaded["features"]) == 18
