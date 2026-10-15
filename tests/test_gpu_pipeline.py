"""
Tests for End-to-End GPU Pipeline Execution, Fallback, Serialization, and Edge Safeguards.
"""

import json
from pathlib import Path
import pytest

from models.gpu_predict import (
    GPU_GEOJSON_OUTPUT,
    GPU_METRICS_OUTPUT,
    GPU_ZONES_OUTPUT,
    run_gpu_inference,
)
from utils.geojson_utils import validate_feature_collection

PRE_IMG = Path("data/processed/hurricane_michael_pre.png")
POST_IMG = Path("data/processed/hurricane_michael_post.png")


def test_invalid_image_path_handling():
    """Verify that run_gpu_inference rejects invalid or non-existent file paths."""
    with pytest.raises((FileNotFoundError, ValueError)):
        run_gpu_inference(
            pre_image_path="data/does_not_exist_pre.png",
            post_image_path=POST_IMG if POST_IMG.exists() else "data/does_not_exist_post.png",
        )


@pytest.mark.skipif(not PRE_IMG.exists(), reason="Real disaster test imagery not present")
def test_gpu_pipeline_end_to_end_execution():
    """Verify that run_gpu_inference executes cleanly, performs CPU fallback, and serializes all artifacts."""
    fc = run_gpu_inference(
        pre_image_path=PRE_IMG,
        post_image_path=POST_IMG,
        confidence_threshold=0.30,
        max_features=25,
        grid_rows=3,
        grid_cols=3,
    )

    # 1. GeoJSON Structure
    validate_feature_collection(fc)
    assert fc["type"] == "FeatureCollection"
    features = fc.get("features", [])
    assert len(features) > 0
    assert len(features) <= 25

    # 2. Required Feature Properties
    for feat in features:
        props = feat["properties"]
        assert "damage_class" in props
        assert props["damage_class"] in {"no-damage", "minor-damage", "major-damage", "destroyed"}
        assert 0.0 <= props["confidence"] <= 1.0
        assert props["backend_name"] == "Advanced Damage Model Backend (GPU-Ready / CPU Fallback)"
        assert "device" in props
        assert "source_image_identifiers" in props
        assert "pre" in props["source_image_identifiers"]
        assert "post" in props["source_image_identifiers"]
        assert "coordinate_source" in props
        assert props["crs"] == "urn:ogc:def:crs:OGC:1.3:CRS84"

    # 3. File Artifacts Exist and Are Valid JSON
    assert GPU_GEOJSON_OUTPUT.exists()
    assert GPU_ZONES_OUTPUT.exists()
    assert GPU_METRICS_OUTPUT.exists()

    with open(GPU_ZONES_OUTPUT, "r", encoding="utf-8") as f:
        zones = json.load(f)
        assert len(zones) == 9
        for z in zones:
            assert 0.0 <= z["priority_score"] <= 1.0
            assert "priority_rank" in z

    with open(GPU_METRICS_OUTPUT, "r", encoding="utf-8") as f:
        metrics = json.load(f)
        assert metrics["backend_name"] == "Advanced Damage Model Backend (GPU-Ready / CPU Fallback)"
        assert metrics["total_features_detected"] == len(features)
        assert "gpu_readiness" in metrics


@pytest.mark.skipif(not PRE_IMG.exists(), reason="Real disaster test imagery not present")
def test_confidence_filtering_and_feature_clamping():
    """Verify that confidence thresholds and max_features constraints are strictly respected."""
    # Strict confidence threshold
    fc_high = run_gpu_inference(
        pre_image_path=PRE_IMG,
        post_image_path=POST_IMG,
        confidence_threshold=0.45,
        max_features=10,
    )
    for feat in fc_high.get("features", []):
        assert feat["properties"]["confidence"] >= 0.45

    assert len(fc_high.get("features", [])) <= 10


@pytest.mark.gpu
def test_cuda_hardware_execution_conditional():
    """Verify CUDA execution if and only if CUDAExecutionProvider is present in onnxruntime."""
    import onnxruntime as ort
    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" not in providers:
        pytest.skip("CUDAExecutionProvider not compiled into current ONNX Runtime wheel")

    # If CUDA is available, verify that backend uses cuda device
    from models.model_registry import AdvancedDamageModelGPUBackend
    import numpy as np
    from utils.georeferencing import AffineGeoreferencer

    backend = AdvancedDamageModelGPUBackend()
    dummy = np.zeros((64, 64, 3), dtype=np.uint8)
    geo = AffineGeoreferencer(64, 64, (-85.55, 30.15, -85.54, 30.16))
    _, meta = backend.predict(dummy, dummy, geo)
    assert meta["device"] == "cuda"
