"""
Tests for GPU Readiness Detection, Hardware Auditing, and AdvancedDamageModelGPUBackend Registry.
"""

from pathlib import Path
import pytest

from models.model_registry import (
    AdvancedDamageModelGPUBackend,
    PretrainedXBDGPUBackend,
    detect_gpu_hardware,
    get_backend,
    get_gpu_readiness_status,
)


def test_gpu_hardware_detection():
    """Verify that detect_gpu_hardware parses nvidia-smi correctly."""
    hw = detect_gpu_hardware()
    assert isinstance(hw, dict)
    # The host environment has an active NVIDIA RTX 3050 GPU
    if hw.get("gpu_detected"):
        assert "RTX 3050" in hw.get("name", "")
        assert hw.get("vram_mb") == 6144
        assert hw.get("driver_version") is not None
        assert hw.get("cuda_driver_version") == "13.2"


def test_gpu_readiness_status_distinguishes_hardware_from_runtime():
    """Verify that get_gpu_readiness_status accurately reports runtime provider availability."""
    readiness = get_gpu_readiness_status()
    assert isinstance(readiness, dict)
    assert "gpu_detected" in readiness
    assert "onnx_cuda_available" in readiness
    assert "gpu_inference_runtime_available" in readiness

    # In our audited Python 3.14 environment, CUDAExecutionProvider is not available in the ONNX wheel
    assert readiness["onnx_cuda_available"] is False
    assert readiness["gpu_inference_runtime_available"] is False
    assert readiness["active_inference_device"] == "cpu"
    assert readiness["recommended_backend"] == "Real Damage Model (ONNX)"

    # Check that the required notice is present
    expected_notice = (
        "The NVIDIA GPU is detected, but the installed ONNX Runtime does not include "
        "CUDAExecutionProvider. Therefore, the system performs real inference on CPU "
        "using the verified lightweight ONNX backend. No pretrained xBD/xView2 checkpoint "
        "is currently integrated."
    )
    assert expected_notice in readiness["status_notice"]


def test_backend_registry_gpu_lookup():
    """Verify that get_backend resolves the optional GPU backend with aliases."""
    backend = get_backend("advanced_damage_model_gpu")
    assert isinstance(backend, AdvancedDamageModelGPUBackend)
    assert backend.name == "Advanced Damage Model Backend (GPU-Ready / CPU Fallback)"
    assert backend.is_available() is True

    # Test alias lookups
    assert isinstance(get_backend("Advanced Damage Model Backend (GPU-Ready / CPU Fallback)"), AdvancedDamageModelGPUBackend)
    assert isinstance(get_backend("pretrained_xbd_gpu"), AdvancedDamageModelGPUBackend)
    assert isinstance(get_backend("xbd_gpu"), AdvancedDamageModelGPUBackend)


def test_gpu_backend_cpu_fallback_execution():
    """Verify that AdvancedDamageModelGPUBackend falls back to CPU cleanly when called directly."""
    from utils.georeferencing import AffineGeoreferencer
    import numpy as np

    backend = AdvancedDamageModelGPUBackend()
    pre_dummy = np.zeros((128, 128, 3), dtype=np.uint8)
    post_dummy = np.zeros((128, 128, 3), dtype=np.uint8)
    geo = AffineGeoreferencer(128, 128, (-85.55, 30.15, -85.54, 30.16))

    fc, meta = backend.predict(
        pre_image=pre_dummy,
        post_image=post_dummy,
        georeferencer=geo,
        confidence_threshold=0.30,
        max_features=10,
    )

    assert fc.get("type") == "FeatureCollection"
    assert meta["backend"] == "Advanced Damage Model Backend (GPU-Ready / CPU Fallback)"
    assert "cpu" in meta["device"]
    assert meta["cuda_provider_available"] is False
