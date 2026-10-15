"""
MERIDIAN-Lite Models Package.
Provides unified model backends, registry abstractions, and inference pipelines.
"""

from models.model_registry import (
    AdvancedDamageModelGPUBackend,
    ClassicalChangeDetectionBackend,
    ModelBackend,
    PretrainedXBDGPUBackend,
    RealDamageModelBackend,
    SimulatedXView2Backend,
    check_model_health,
    detect_gpu_hardware,
    get_backend,
    get_gpu_readiness_status,
    list_available_backends,
)
from models.predict import run_inference
from models.real_predict import run_real_inference
from models.gpu_predict import run_gpu_inference

__all__ = [
    "ModelBackend",
    "RealDamageModelBackend",
    "ClassicalChangeDetectionBackend",
    "SimulatedXView2Backend",
    "AdvancedDamageModelGPUBackend",
    "PretrainedXBDGPUBackend",
    "get_backend",
    "list_available_backends",
    "check_model_health",
    "detect_gpu_hardware",
    "get_gpu_readiness_status",
    "run_inference",
    "run_real_inference",
    "run_gpu_inference",
]
