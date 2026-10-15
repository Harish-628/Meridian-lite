"""
Automated Model Provenance and Technical Integrity Tests for MERIDIAN-Lite.
Verifies that:
1. models/weights/damage_classifier.onnx exists and matches the exact audited SHA-256 checksum.
2. The ONNX graph passes onnx.checker validation.
3. Input and output tensor shapes, names, and data types match the operational contract.
4. Execution runs reliably on CPU fallback without requiring external CUDA libraries.
5. The model is formally classified as 'Model provenance not independently verified'
   and never falsely claimed as 'pretrained' without documentary evidence.
"""

import hashlib
from pathlib import Path
import json
import numpy as np
import onnx
import onnxruntime as ort
import pytest


MODEL_PATH = Path("models/weights/damage_classifier.onnx")
AUDIT_REPORT_PATH = Path("outputs/reports/model_provenance_audit.json")
EXPECTED_SHA256 = "bda9515c3a3c431abf46a8792e20ff131c09aa6d23a1e54c387d344c9a05f753"


def test_model_file_exists_and_sha256_matches():
    """Verify that the model file exists, is not empty, and matches the exact SHA-256 hash."""
    assert MODEL_PATH.exists(), f"Model file missing at {MODEL_PATH}"
    assert MODEL_PATH.stat().st_size > 0, "Model file is empty"
    assert MODEL_PATH.stat().st_size == 23568, f"Unexpected model size: {MODEL_PATH.stat().st_size}"

    hasher = hashlib.sha256()
    with open(MODEL_PATH, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    actual_sha256 = hasher.hexdigest()
    assert actual_sha256 == EXPECTED_SHA256, (
        f"Model SHA-256 mismatch! Expected {EXPECTED_SHA256}, got {actual_sha256}"
    )


def test_onnx_model_graph_integrity():
    """Verify that the ONNX protobuf is structurally valid and passes checker."""
    model_proto = onnx.load(str(MODEL_PATH))
    onnx.checker.check_model(model_proto)

    assert model_proto.ir_version == 10
    opset_imports = {op.domain: op.version for op in model_proto.opset_import}
    assert opset_imports.get("", 0) == 14, "Expected Opset 14"

    graph = model_proto.graph
    assert graph.name == "meridian_damage_net"

    # Verify input tensor
    assert len(graph.input) == 1
    inp = graph.input[0]
    assert inp.name == "input"
    assert inp.type.tensor_type.elem_type == onnx.TensorProto.FLOAT

    # Verify output tensor
    assert len(graph.output) == 1
    out = graph.output[0]
    assert out.name == "probabilities"
    assert out.type.tensor_type.elem_type == onnx.TensorProto.FLOAT


def test_onnx_cpu_runtime_inference():
    """Verify that the model can be instantiated and executed on CPU fallback."""
    session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    assert "CPUExecutionProvider" in session.get_providers()

    # Create dummy bi-temporal batch of 2 patches: [B=2, C=6, H=64, W=64]
    dummy_input = np.random.uniform(0.0, 1.0, size=(2, 6, 64, 64)).astype(np.float32)
    outputs = session.run(["probabilities"], {"input": dummy_input})

    assert len(outputs) == 1
    result = outputs[0]
    assert result.shape == (2, 4), f"Expected shape (2, 4), got {result.shape}"
    assert np.all(result >= 0.0) and np.all(result <= 1.0), "Outputs must be probabilities in [0, 1]"
    assert np.allclose(np.sum(result, axis=1), [1.0, 1.0], atol=1e-4), "Softmax probabilities must sum to 1"


def test_audit_report_provenance_verdict():
    """Verify that outputs/reports/model_provenance_audit.json records the exact required verdict."""
    assert AUDIT_REPORT_PATH.exists(), f"Audit report missing at {AUDIT_REPORT_PATH}"
    with open(AUDIT_REPORT_PATH, "r", encoding="utf-8") as f:
        audit_data = json.load(f)

    meta = audit_data.get("audit_metadata", {})
    assert meta.get("formal_provenance_verdict") == "Model provenance not independently verified"
    assert meta.get("sha256_checksum") == EXPECTED_SHA256

    answers = audit_data.get("provenance_inquiry_answers", {})
    assert answers.get("3_was_xbd_used") is False
    assert answers.get("4_was_xview2_used") is False
    assert answers.get("7_was_the_model_pretrained") is False
    assert answers.get("10_is_it_only_a_demonstration_model") is True
    assert answers.get("11_is_there_training_test_data_leakage") is False


def test_no_false_pretrained_claims_in_ui_or_code():
    """Verify that app/ui code does not label the model as 'pretrained xView2' or claim verified training."""
    ui_sidebar_content = Path("app/ui/sidebar.py").read_text(encoding="utf-8")
    assert "pretrained xView2" not in ui_sidebar_content.lower()
    assert "Pretrained status not independently verified" in ui_sidebar_content
