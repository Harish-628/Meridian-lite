"""Tests for ONNX Real Damage Model inference engine.

Validates the ONNX neural model at models/weights/damage_classifier.onnx:
- Input/output specifications (batch_size, 6, 64, 64) -> (batch_size, 4)
- Softmax probability distributions (summing to 1.0, non-negative)
- Multi-batch processing (batch size 1, 4, 16)
- Edge cases (all zeros, all ones, out-of-range dimensions)
- Inference latency performance guarantees (< 500ms on CPU)
"""

import os
import time
import pytest
import numpy as np
import onnxruntime as ort

MODEL_PATH = "models/weights/damage_classifier.onnx"


@pytest.fixture(scope="module")
def onnx_session():
    """Initializes ONNX Runtime session."""
    assert os.path.exists(MODEL_PATH), f"Model weight missing at {MODEL_PATH}"
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    return session


def test_model_file_exists_and_metadata(onnx_session):
    """Verify model file metadata and input/output contracts."""
    inputs = onnx_session.get_inputs()
    outputs = onnx_session.get_outputs()

    assert len(inputs) == 1, "Expected single input tensor"
    assert inputs[0].name == "input"
    assert inputs[0].type == "tensor(float)"
    assert inputs[0].shape[1:] == [6, 64, 64]

    assert len(outputs) == 1, "Expected single output tensor"
    assert outputs[0].name == "probabilities"
    assert outputs[0].type == "tensor(float)"
    assert outputs[0].shape[1:] == [4]


def test_single_item_inference(onnx_session):
    """Verify inference on a single sample (batch size 1)."""
    np.random.seed(42)
    sample_input = np.random.uniform(0.0, 1.0, size=(1, 6, 64, 64)).astype(np.float32)

    outputs = onnx_session.run(None, {"input": sample_input})
    probs = outputs[0]

    assert isinstance(probs, np.ndarray)
    assert probs.shape == (1, 4)
    assert probs.dtype == np.float32

    # Verify probability properties
    assert np.all(probs >= 0.0), "Probabilities must be non-negative"
    assert np.all(probs <= 1.0), "Probabilities must be <= 1.0"
    assert np.isclose(np.sum(probs), 1.0, atol=1e-5), "Probabilities must sum to 1.0"


def test_batched_inference(onnx_session):
    """Verify inference on batch sizes > 1."""
    for batch_size in [2, 4, 16]:
        batch_input = np.random.uniform(0.0, 1.0, size=(batch_size, 6, 64, 64)).astype(np.float32)
        outputs = onnx_session.run(None, {"input": batch_input})
        probs = outputs[0]

        assert probs.shape == (batch_size, 4)
        sums = np.sum(probs, axis=1)
        assert np.allclose(sums, 1.0, atol=1e-5), f"Batch {batch_size} sums do not equal 1.0"


def test_edge_case_all_zeros(onnx_session):
    """Verify robustness on all-zero input tensor."""
    zero_input = np.zeros((1, 6, 64, 64), dtype=np.float32)
    outputs = onnx_session.run(None, {"input": zero_input})
    probs = outputs[0]

    assert probs.shape == (1, 4)
    assert np.isclose(np.sum(probs), 1.0, atol=1e-5)
    assert not np.isnan(probs).any(), "Found NaN in output"
    assert not np.isinf(probs).any(), "Found Inf in output"


def test_edge_case_all_ones(onnx_session):
    """Verify robustness on all-ones input tensor."""
    ones_input = np.ones((1, 6, 64, 64), dtype=np.float32)
    outputs = onnx_session.run(None, {"input": ones_input})
    probs = outputs[0]

    assert probs.shape == (1, 4)
    assert np.isclose(np.sum(probs), 1.0, atol=1e-5)
    assert not np.isnan(probs).any(), "Found NaN in output"
    assert not np.isinf(probs).any(), "Found Inf in output"


def test_invalid_input_dimensions_raises_error(onnx_session):
    """Verify that malformed input dimensions raise appropriate ONNX Runtime exceptions."""
    # Wrong number of channels (3 instead of 6)
    invalid_channels = np.random.randn(1, 3, 64, 64).astype(np.float32)
    with pytest.raises(Exception):
        onnx_session.run(None, {"input": invalid_channels})

    # Wrong spatial dimensions (32x32 instead of 64x64)
    invalid_spatial = np.random.randn(1, 6, 32, 32).astype(np.float32)
    with pytest.raises(Exception):
        onnx_session.run(None, {"input": invalid_spatial})

    # Wrong rank (3D instead of 4D)
    invalid_rank = np.random.randn(6, 64, 64).astype(np.float32)
    with pytest.raises(Exception):
        onnx_session.run(None, {"input": invalid_rank})


def test_inference_latency_benchmark(onnx_session):
    """Verify inference latency is well within hackathon decision-support thresholds (< 500ms on CPU)."""
    input_tensor = np.random.randn(1, 6, 64, 64).astype(np.float32)

    # Warmup
    for _ in range(5):
        onnx_session.run(None, {"input": input_tensor})

    # Benchmark 25 iterations
    start_time = time.perf_counter()
    for _ in range(25):
        onnx_session.run(None, {"input": input_tensor})
    total_time_ms = (time.perf_counter() - start_time) * 1000.0
    avg_latency_ms = total_time_ms / 25.0

    # Ensure single patch latency is ultra-fast (< 100ms on CPU, typically < 1ms)
    assert avg_latency_ms < 100.0, f"Average patch latency too high: {avg_latency_ms:.2f} ms"

    # Also benchmark full grid batch of 16 patches
    grid_batch = np.random.randn(16, 6, 64, 64).astype(np.float32)
    t0 = time.perf_counter()
    onnx_session.run(None, {"input": grid_batch})
    batch_latency_ms = (time.perf_counter() - t0) * 1000.0

    assert batch_latency_ms < 500.0, f"Batch latency exceeded 500ms threshold: {batch_latency_ms:.2f} ms"
