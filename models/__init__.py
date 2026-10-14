"""
MERIDIAN-Lite Models Package.
Provides unified model backends, registry abstractions, and inference pipelines.
"""

from models.model_registry import (
    ClassicalChangeDetectionBackend,
    ModelBackend,
    RealDamageModelBackend,
    SimulatedXView2Backend,
    check_model_health,
    detect_gpu_hardware,
    get_backend,
    list_available_backends,
)
from models.predict import run_inference
from models.real_predict import run_real_inference

__all__ = [
    "ModelBackend",
    "RealDamageModelBackend",
    "ClassicalChangeDetectionBackend",
    "SimulatedXView2Backend",
    "get_backend",
    "list_available_backends",
    "check_model_health",
    "detect_gpu_hardware",
    "run_inference",
    "run_real_inference",
]
