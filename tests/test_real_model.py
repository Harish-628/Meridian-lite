"""Unit tests for model registry and real model backend adapters."""

from pathlib import Path
import numpy as np
import pytest

from models.model_registry import (
    ClassicalChangeDetectionBackend,
    RealDamageModelBackend,
    SimulatedXView2Backend,
    check_model_health,
    get_backend,
    list_available_backends,
)
from utils.georeferencing import AffineGeoreferencer


@pytest.fixture
def sample_image_pair():
    """Create synthetic pre and post image pair for testing."""
    h, w = 128, 128
    pre = np.full((h, w, 3), 100, dtype=np.uint8)
    # Add a mock building in pre
    pre[40:80, 40:80] = 200

    # Post has damage: building area altered
    post = pre.copy()
    post[40:80, 40:80] = 30  # structural change / debris

    return pre, post


@pytest.fixture
def sample_georeferencer():
    return AffineGeoreferencer(128, 128, (-85.555, 30.170, -85.545, 30.180))


def test_model_registry_backends():
    """Verify registry backend lookup and availability."""
    backends = list_available_backends()
    assert "real_damage_model" in backends
    assert "classical_change_detection" in backends
    assert "simulated_xview2" in backends

    b_real = get_backend("real_damage_model")
    assert isinstance(b_real, RealDamageModelBackend)

    b_cv = get_backend("classical_change_detection")
    assert isinstance(b_cv, ClassicalChangeDetectionBackend)

    b_sim = get_backend("simulated_xview2")
    assert isinstance(b_sim, SimulatedXView2Backend)


def test_unknown_backend_raises():
    """Verify unknown backend key raises KeyError."""
    with pytest.raises(KeyError, match="Unknown backend"):
        get_backend("nonexistent_cloud_model")


def test_model_health_check():
    """Verify system health audit report."""
    health = check_model_health()
    assert "model_available" in health
    assert "device_available" in health
    assert health["device_available"] == "cpu"
    assert "expected_output_format" in health
    assert "available_backends" in health
    assert len(health["available_backends"]) >= 2


def test_classical_change_detection_predict(sample_image_pair, sample_georeferencer):
    """Verify classical backend returns valid detections with damage classes."""
    pre, post = sample_image_pair
    backend = ClassicalChangeDetectionBackend()
    assert backend.is_available()

    fc, meta = backend.predict(
        pre_image=pre,
        post_image=post,
        georeferencer=sample_georeferencer,
        confidence_threshold=0.30,
        max_features=50,
    )

    assert fc["type"] == "FeatureCollection"
    assert "inference_time_seconds" in meta
    assert meta["device"] == "cpu"
    for feat in fc["features"]:
        assert feat["type"] == "Feature"
        props = feat["properties"]
        assert props["damage_class"] in ["no-damage", "minor-damage", "major-damage", "destroyed"]
        assert 0.0 <= props["confidence"] <= 1.0


def test_real_onnx_backend_predict(sample_image_pair, sample_georeferencer):
    """Verify ONNX backend executes on CPU and returns predictions."""
    pre, post = sample_image_pair
    backend = RealDamageModelBackend()
    if not backend.is_available():
        pytest.skip("ONNX weights not found")

    fc, meta = backend.predict(
        pre_image=pre,
        post_image=post,
        georeferencer=sample_georeferencer,
        confidence_threshold=0.20,
        max_features=50,
    )

    assert fc["type"] == "FeatureCollection"
    assert meta["device"] == "cpu"
    assert "inference_time_seconds" in meta


def test_feature_limit_enforced(sample_image_pair, sample_georeferencer):
    """Verify max_features constraint is strictly respected."""
    pre, post = sample_image_pair
    backend = ClassicalChangeDetectionBackend()
    fc, _ = backend.predict(
        pre_image=pre,
        post_image=post,
        georeferencer=sample_georeferencer,
        confidence_threshold=0.10,
        max_features=1,
    )
    assert len(fc["features"]) <= 1
