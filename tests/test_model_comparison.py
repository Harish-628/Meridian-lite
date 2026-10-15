"""
Automated Model Comparison and Inference Contract Validation Tests.
Validates:
1. Candidate evaluation report outputs/reports/model_comparison.json.
2. Inference output GeoJSON contracts between ONNX model and classical baseline.
3. Damage classifications adhere strictly to the 4 xBD/EMS-98 tiers.
4. Priority scores and confidence values are strictly bounded in [0.0, 1.0].
"""

import json
from pathlib import Path
import pytest
import numpy as np

from models.real_predict import run_real_inference


COMPARISON_REPORT_PATH = Path("outputs/reports/model_comparison.json")
DECISION_DOC_PATH = Path("docs/model_selection_decision.md")
PRE_IMAGE_PATH = Path("data/processed/hurricane_michael_pre.png")
POST_IMAGE_PATH = Path("data/processed/hurricane_michael_post.png")

VALID_DAMAGE_CLASSES = {"no-damage", "minor-damage", "major-damage", "destroyed"}


def test_comparison_report_exists_and_recommends_local_model():
    """Verify that model_comparison.json exists and documents current model retention."""
    assert COMPARISON_REPORT_PATH.exists(), f"Missing {COMPARISON_REPORT_PATH}"
    with open(COMPARISON_REPORT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "current_model" in data
    assert "candidate_models" in data
    assert len(data["candidate_models"]) >= 2

    # Check candidates
    candidate_names = [c["name"] for c in data["candidate_models"]]
    assert any("xView2" in name for name in candidate_names)

    # Check decision
    decision = data.get("decision_summary", {})
    assert "damage_classifier.onnx" in decision.get("selected_model", "")


def test_decision_document_exists():
    """Verify that ADR-004 decision record exists and contains the required declaration."""
    assert DECISION_DOC_PATH.exists()
    content = DECISION_DOC_PATH.read_text(encoding="utf-8")
    assert "Model provenance not independently verified" in content
    assert "We have a verified local ONNX model" in content


@pytest.mark.skipif(not PRE_IMAGE_PATH.exists(), reason="Hurricane Michael test imagery not present")
def test_real_model_inference_output_contract():
    """Verify that real ONNX inference produces a valid GeoJSON FeatureCollection with bounded scores."""
    geojson_result = run_real_inference(
        pre_image_path=PRE_IMAGE_PATH,
        post_image_path=POST_IMAGE_PATH,
        confidence_threshold=0.20,
        max_features=25,
        grid_rows=3,
        grid_cols=3,
        model_name="onnx",
    )

    assert geojson_result.get("type") == "FeatureCollection"
    features = geojson_result.get("features", [])
    assert len(features) > 0, "Expected non-zero features from real test tile"

    for feat in features:
        props = feat.get("properties", {})
        assert "damage_class" in props
        assert props["damage_class"] in VALID_DAMAGE_CLASSES
        assert 0.0 <= props.get("confidence", 0.0) <= 1.0
        assert "feature_id" in props

    from utils.priority import compute_priority_zones
    zones = compute_priority_zones(geojson_result, grid_rows=3, grid_cols=3)
    assert len(zones) == 9
    for z in zones:
        assert 0.0 <= z["priority_score"] <= 1.0
        assert "priority_rank" in z


@pytest.mark.skipif(not PRE_IMAGE_PATH.exists(), reason="Hurricane Michael test imagery not present")
def test_classical_vs_onnx_comparison_contract():
    """Verify that both ONNX and classical backends conform to the identical GeoJSON contract."""
    onnx_res = run_real_inference(
        pre_image_path=PRE_IMAGE_PATH,
        post_image_path=POST_IMAGE_PATH,
        confidence_threshold=0.25,
        max_features=20,
        model_name="onnx",
    )

    classical_res = run_real_inference(
        pre_image_path=PRE_IMAGE_PATH,
        post_image_path=POST_IMAGE_PATH,
        confidence_threshold=0.25,
        max_features=20,
        model_name="classical",
    )

    assert onnx_res.get("type") == "FeatureCollection"
    assert classical_res.get("type") == "FeatureCollection"
    assert len(onnx_res["features"]) > 0
    assert len(classical_res["features"]) > 0

    # Ensure both models populate required tactical properties
    for feat in onnx_res["features"] + classical_res["features"]:
        p = feat["properties"]
        assert p["damage_class"] in VALID_DAMAGE_CLASSES
        assert 0.0 <= p.get("confidence", 0.0) <= 1.0
        assert "source_model" in p
