"""End-to-end integration tests for Stage 2 real dataset and inference pipeline."""

import json
from pathlib import Path
import pytest

from models.predict import run_inference as run_stage1_inference
from models.real_predict import run_real_inference
from utils.geojson_utils import validate_feature_collection
from utils.paths import PROJECT_ROOT
from utils.priority import compute_priority_zones

DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
HURRICANE_PRE = DATA_PROCESSED / "hurricane_michael_pre.png"
HURRICANE_POST = DATA_PROCESSED / "hurricane_michael_post.png"
METADATA_FILE = DATA_PROCESSED / "dataset_metadata.json"


def test_real_dataset_integrity():
    """Verify downloaded Hurricane Michael tiles and provenance metadata exist."""
    assert HURRICANE_PRE.exists(), f"Missing {HURRICANE_PRE}"
    assert HURRICANE_POST.exists(), f"Missing {HURRICANE_POST}"
    assert METADATA_FILE.exists(), f"Missing {METADATA_FILE}"

    assert HURRICANE_PRE.stat().st_size > 100000
    assert HURRICANE_POST.stat().st_size > 100000

    with open(METADATA_FILE, "r") as f:
        meta = json.load(f)

    assert meta["disaster_name"] == "Hurricane Michael"
    assert "DamageTriage-Bench" in meta["source"]
    assert "bounding_box" in meta
    bbox = meta["bounding_box"]
    assert bbox["min_lon"] < bbox["max_lon"]
    assert bbox["min_lat"] < bbox["max_lat"]


def test_real_inference_classical_pipeline():
    """Verify real inference pipeline with classical OpenCV change detection."""
    fc = run_real_inference(
        pre_image_path=HURRICANE_PRE,
        post_image_path=HURRICANE_POST,
        confidence_threshold=0.35,
        max_features=100,
        backend_name="classical_change_detection",
    )

    validate_feature_collection(fc)
    feats = fc.get("features", [])
    assert len(feats) > 0, "Expected building detections on real disaster imagery"

    # Verify feature properties
    f0 = feats[0]["properties"]
    assert "feature_id" in f0
    assert "damage_class" in f0
    assert f0["damage_class"] in ["no-damage", "minor-damage", "major-damage", "destroyed"]
    assert "confidence" in f0
    assert f0["confidence"] >= 0.35
    assert f0["source_model"] == "Classical Change Detection (OpenCV)"
    assert f0["coordinate_source"] == "georeferenced_bounds"

    # Compute priority zones
    zones = compute_priority_zones(fc, grid_rows=3, grid_cols=3)
    assert len(zones) == 9
    assert zones[0]["priority_rank"] == 1
    assert zones[0]["priority_score"] >= zones[-1]["priority_score"]


def test_real_inference_onnx_pipeline():
    """Verify real inference pipeline with ONNX neural model backend."""
    fc = run_real_inference(
        pre_image_path=HURRICANE_PRE,
        post_image_path=HURRICANE_POST,
        confidence_threshold=0.30,
        max_features=100,
        backend_name="real_damage_model",
    )

    validate_feature_collection(fc)
    feats = fc.get("features", [])
    assert len(feats) > 0

    f0 = feats[0]["properties"]
    assert f0["source_model"] == "Real Damage Model (ONNX)"
    assert "probabilities" in f0
    assert f0["damage_class"] in ["no-damage", "minor-damage", "major-damage", "destroyed"]

    # Verify top-level metadata
    assert "metadata" in fc
    meta = fc["metadata"]
    assert meta["backend"] == "Real Damage Model (ONNX)"
    assert meta["device"] == "cpu"
    assert meta["inference_time_seconds"] > 0.0


def test_stage1_simulation_regression():
    """Verify that Stage 1 simulated inference engine continues to operate without regression."""
    from utils.paths import DEMO_PRE_IMAGE, DEMO_POST_IMAGE

    fc_sim = run_stage1_inference(
        pre_image_path=DEMO_PRE_IMAGE,
        post_image_path=DEMO_POST_IMAGE,
        center_lat=13.0827,
        center_lon=80.2707,
        seed=42,
        building_count=18,
    )
    validate_feature_collection(fc_sim)
    assert len(fc_sim["features"]) == 18
