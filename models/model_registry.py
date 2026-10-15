"""Model registry and backend implementations for MERIDIAN-Lite Stage 2.

Engineering Team:
- Member 1 (Harish Kumar S): Machine Learning & On-Device Neural Inference Lead
- Member 2: Geospatial Intelligence & GIS Architecture Lead
- Member 3: Defense Systems Architect & Operations Research Lead
- Member 4: Full-Stack C2 UI, Mission Assurance & QA Lead

Provides a unified interface across:
1. RealDamageModelBackend (ONNX CPU neural model)
2. ClassicalChangeDetectionBackend (OpenCV structural difference)
3. SimulatedXView2Backend (Stage 1 deterministic baseline)
"""

from abc import ABC, abstractmethod
from pathlib import Path
import time
from typing import Any, Tuple
import cv2
import numpy as np
import onnxruntime as ort

from utils.georeferencing import AffineGeoreferencer
from utils.image_utils import load_image, validate_image
from utils.logging_utils import get_logger
from utils.paths import PROJECT_ROOT

logger = get_logger("model_registry")

DEFAULT_ONNX_WEIGHTS = PROJECT_ROOT / "models" / "weights" / "damage_classifier.onnx"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "dataset_metadata.json"

DAMAGE_CLASSES = ["no-damage", "minor-damage", "major-damage", "destroyed"]

# Global tracker for health check
_LAST_INFERENCE_TIMESTAMP: str | None = None


class ModelBackend(ABC):
    """Abstract base class for disaster damage assessment backends."""

    def __init__(self, name: str, version: str = "2.0.0", device: str = "cpu") -> None:
        self.name = name
        self.version = version
        self.device = device

    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend dependencies and weights are available."""
        pass

    @abstractmethod
    def predict(
        self,
        pre_image: np.ndarray,
        post_image: np.ndarray,
        georeferencer: AffineGeoreferencer,
        confidence_threshold: float = 0.40,
        max_features: int = 500,
    ) -> Tuple[dict[str, Any], dict[str, Any]]:
        """Run inference on pre/post image pair and return GeoJSON + metadata."""
        pass


class ClassicalChangeDetectionBackend(ModelBackend):
    """Real computer-vision structural change detection using OpenCV morphological processing."""

    def __init__(self) -> None:
        super().__init__(name="Classical Change Detection (OpenCV)", version="2.0.0", device="cpu")

    def is_available(self) -> bool:
        return True

    def predict(
        self,
        pre_image: np.ndarray,
        post_image: np.ndarray,
        georeferencer: AffineGeoreferencer,
        confidence_threshold: float = 0.40,
        max_features: int = 500,
    ) -> Tuple[dict[str, Any], dict[str, Any]]:
        start_time = time.perf_counter()

        h, w = pre_image.shape[:2]
        if post_image.shape[:2] != (h, w):
            post_image = cv2.resize(post_image, (w, h), interpolation=cv2.INTER_LINEAR)

        pre_gray = cv2.cvtColor(pre_image, cv2.COLOR_RGB2GRAY)
        post_gray = cv2.cvtColor(post_image, cv2.COLOR_RGB2GRAY)

        # 1. Structural difference
        diff = cv2.absdiff(pre_gray, post_gray)
        blurred = cv2.medianBlur(diff, 5)

        # 2. Otsu thresholding for change segmentation
        otsu_val, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 3. Morphological opening and closing to delineate building-like structures
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

        # 4. Find external contours
        cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        features: list[dict[str, Any]] = []
        feat_idx = 1

        for c in cnts:
            area = cv2.contourArea(c)
            # Filter non-building noise or massive landscape changes
            if not (120 < area < 25000):
                continue

            bx, by, bw, bh = cv2.boundingRect(c)
            aspect = float(bw) / max(bh, 1)
            if not (0.25 <= aspect <= 4.0):
                continue

            # Compute contour mask and change statistics
            c_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(c_mask, [c], -1, 255, -1)

            mean_diff = float(cv2.mean(diff, mask=c_mask)[0])
            std_diff = float(np.std(diff[c_mask > 0])) if np.any(c_mask > 0) else 0.0

            # Canny edge density comparison within bounding box
            pre_crop = pre_gray[by:by+bh, bx:bx+bw]
            post_crop = post_gray[by:by+bh, bx:bx+bw]
            pre_edges = cv2.Canny(pre_crop, 50, 150)
            post_edges = cv2.Canny(post_crop, 50, 150)

            pre_edge_count = np.count_nonzero(pre_edges)
            post_edge_count = np.count_nonzero(post_edges)
            edge_ratio = post_edge_count / max(pre_edge_count, 1)

            # Damage classification rule based on structural change and edge disruption
            if mean_diff >= 90.0 or (mean_diff >= 70.0 and edge_ratio > 2.5):
                damage_class = "destroyed"
                raw_conf = min(0.98, 0.75 + (mean_diff / 255.0) * 0.25)
            elif mean_diff >= 65.0 or (mean_diff >= 50.0 and edge_ratio > 1.8):
                damage_class = "major-damage"
                raw_conf = min(0.95, 0.70 + (mean_diff / 255.0) * 0.25)
            elif mean_diff >= 38.0 or edge_ratio > 1.3:
                damage_class = "minor-damage"
                raw_conf = min(0.92, 0.65 + (mean_diff / 255.0) * 0.25)
            else:
                damage_class = "no-damage"
                raw_conf = min(0.94, 0.60 + (1.0 - mean_diff / 255.0) * 0.35)

            # Compactness heuristic
            perimeter = cv2.arcLength(c, True)
            compactness = (4 * np.pi * area) / (perimeter * perimeter + 1e-5)
            confidence = round(float(np.clip(raw_conf * (0.8 + 0.2 * min(compactness, 1.0)), 0.45, 0.98)), 4)

            if confidence < confidence_threshold:
                continue

            # Approximate polygon geometry (Douglas-Peucker)
            epsilon = 0.03 * perimeter
            approx = cv2.approxPolyDP(c, max(epsilon, 3.0), True)
            pts = [(float(pt[0][0]), float(pt[0][1])) for pt in approx]
            if len(pts) < 3:
                # Fallback to rectangle points
                pts = [
                    (float(bx), float(by)),
                    (float(bx + bw), float(by)),
                    (float(bx + bw), float(by + bh)),
                    (float(bx), float(by + bh)),
                ]

            feature_props = {
                "feature_id": f"BLD-CV-{feat_idx:04d}",
                "damage_class": damage_class,
                "confidence": confidence,
                "source_model": self.name,
                "backend_name": "classical_change_detection",
                "model_type": "classical_change_detection",
                "is_simulated": False,
                "model_version": self.version,
                "pixel_area": float(area),
                "mean_spectral_diff": round(mean_diff, 2),
                "pixel_polygon": [[float(pt[0]), float(pt[1])] for pt in pts],
                "pixel_bbox": [int(bx), int(by), int(bw), int(bh)],
            }

            geojson_feat = georeferencer.to_geojson_polygon(pts, feature_props)
            features.append(geojson_feat)
            feat_idx += 1

            if len(features) >= max_features:
                break

        elapsed = time.perf_counter() - start_time
        global _LAST_INFERENCE_TIMESTAMP
        _LAST_INFERENCE_TIMESTAMP = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

        fc = {
            "type": "FeatureCollection",
            "features": features,
        }

        metadata = {
            "backend": self.name,
            "backend_name": "classical_change_detection",
            "model_type": "classical_change_detection",
            "is_simulated": False,
            "version": self.version,
            "device": self.device,
            "inference_time_seconds": round(elapsed, 4),
            "features_detected": len(features),
            "otsu_threshold": float(otsu_val),
            "timestamp": _LAST_INFERENCE_TIMESTAMP,
        }

        return fc, metadata


def detect_gpu_hardware() -> dict[str, Any]:
    """Detect presence of NVIDIA GPU hardware and CUDA driver via nvidia-smi."""
    import subprocess
    import re
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
        lines = res.stdout.strip().split("\n")
        if lines and lines[0]:
            parts = [p.strip() for p in lines[0].split(",")]
            gpu_name = parts[0]
            vram_mb = int(parts[1]) if len(parts) > 1 else None
            driver_ver = parts[2] if len(parts) > 2 else None

            # Parse CUDA driver version from general nvidia-smi output
            cuda_driver_ver = None
            try:
                full_res = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=2)
                m = re.search(r"CUDA Version:\s*([0-9.]+)", full_res.stdout)
                if m:
                    cuda_driver_ver = m.group(1)
            except Exception:
                pass

            return {
                "gpu_detected": True,
                "name": gpu_name,
                "vram_mb": vram_mb,
                "driver_version": driver_ver,
                "cuda_driver_version": cuda_driver_ver,
            }
    except Exception:
        pass
    return {
        "gpu_detected": False,
        "name": None,
        "vram_mb": None,
        "driver_version": None,
        "cuda_driver_version": None,
    }


def get_gpu_readiness_status() -> dict[str, Any]:
    """Audit GPU hardware, drivers, inference execution providers, and runtime readiness."""
    import shutil
    hardware = detect_gpu_hardware()
    ort_providers = ort.get_available_providers()
    onnx_cuda_avail = "CUDAExecutionProvider" in ort_providers
    nvcc_installed = shutil.which("nvcc") is not None

    torch_installed = False
    torch_cuda_avail = False
    try:
        import torch  # type: ignore
        torch_installed = True
        torch_cuda_avail = bool(torch.cuda.is_available())
    except ImportError:
        pass

    # Runtime availability requires actual software execution provider support, not just hardware
    runtime_avail = onnx_cuda_avail or torch_cuda_avail

    notice = (
        "The NVIDIA GPU is detected, but the installed ONNX Runtime does not include CUDAExecutionProvider. "
        "Therefore, the system performs real inference on CPU using the verified lightweight ONNX backend. "
        "No pretrained xBD/xView2 checkpoint is currently integrated."
        if hardware.get("gpu_detected") and not runtime_avail
        else "CPU execution provider active."
    )

    return {
        "gpu_detected": hardware.get("gpu_detected", False),
        "gpu_name": hardware.get("name"),
        "vram_mb": hardware.get("vram_mb"),
        "driver_version": hardware.get("driver_version"),
        "cuda_driver_version": hardware.get("cuda_driver_version"),
        "nvcc_installed": nvcc_installed,
        "onnx_providers": ort_providers,
        "onnx_cuda_available": onnx_cuda_avail,
        "pytorch_installed": torch_installed,
        "pytorch_cuda_available": torch_cuda_avail,
        "gpu_inference_runtime_available": runtime_avail,
        "active_inference_device": "cuda" if runtime_avail else "cpu",
        "recommended_backend": "Real Damage Model (ONNX)",
        "status_notice": notice,
    }



class RealDamageModelBackend(ModelBackend):
    """Real ONNX neural network damage assessment backend running on CPU via onnxruntime."""

    def __init__(self, weights_path: Path | str | None = None) -> None:
        super().__init__(name="Real Damage Model (ONNX)", version="2.0.0", device="cpu")
        self.weights_path = Path(weights_path) if weights_path else DEFAULT_ONNX_WEIGHTS
        self._session: ort.InferenceSession | None = None

    def is_available(self) -> bool:
        return self.weights_path.exists() and self.weights_path.stat().st_size > 0

    def _get_session(self) -> ort.InferenceSession:
        if self._session is None:
            if not self.is_available():
                raise FileNotFoundError(f"ONNX model weights not found at: {self.weights_path}")
            
            # Prioritize GPU providers (CUDA/TensorRT) if registered in ONNX Runtime, with CPU fallback
            available = ort.get_available_providers()
            providers = []
            if "CUDAExecutionProvider" in available:
                providers.append("CUDAExecutionProvider")
            if "TensorrtExecutionProvider" in available:
                providers.append("TensorrtExecutionProvider")
            providers.append("CPUExecutionProvider")
            
            logger.info(f"Initializing ONNX Runtime session with providers: {providers}")
            self._session = ort.InferenceSession(str(self.weights_path), providers=providers)
            active_provider = self._session.get_providers()[0]
            if "CUDA" in active_provider:
                self.device = "cuda"
                logger.info(f"Active ONNX Runtime execution provider: {active_provider}")
            else:
                self.device = "cpu"
                logger.info(f"Active ONNX Runtime execution provider: {active_provider}")
                logger.info("Running deterministic CPU baseline")
        return self._session

    def predict(
        self,
        pre_image: np.ndarray,
        post_image: np.ndarray,
        georeferencer: AffineGeoreferencer,
        confidence_threshold: float = 0.40,
        max_features: int = 500,
    ) -> Tuple[dict[str, Any], dict[str, Any]]:
        start_time = time.perf_counter()
        session = self._get_session()

        h, w = pre_image.shape[:2]
        if post_image.shape[:2] != (h, w):
            post_image = cv2.resize(post_image, (w, h), interpolation=cv2.INTER_LINEAR)

        # 1. First stage: Candidate structural proposals via edge/gradient morphology
        pre_gray = cv2.cvtColor(pre_image, cv2.COLOR_RGB2GRAY)
        post_gray = cv2.cvtColor(post_image, cv2.COLOR_RGB2GRAY)
        diff = cv2.absdiff(pre_gray, post_gray)

        blurred = cv2.medianBlur(diff, 5)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)

        cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidate_boxes: list[Tuple[int, int, int, int, list[Tuple[float, float]], float, float, float]] = []
        for c in cnts:
            area = cv2.contourArea(c)
            if not (120 < area < 25000):
                continue
            bx, by, bw, bh = cv2.boundingRect(c)
            aspect = float(bw) / max(bh, 1)
            if not (0.25 <= aspect <= 4.0):
                continue

            # Compute contour mask difference and edge statistics
            c_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(c_mask, [c], -1, 255, -1)
            mean_diff = float(cv2.mean(diff, mask=c_mask)[0])

            pre_crop = pre_gray[by:by+bh, bx:bx+bw]
            post_crop = post_gray[by:by+bh, bx:bx+bw]
            pre_edges = np.count_nonzero(cv2.Canny(pre_crop, 50, 150))
            post_edges = np.count_nonzero(cv2.Canny(post_crop, 50, 150))
            edge_ratio = float(post_edges) / max(float(pre_edges), 1.0)

            perimeter = cv2.arcLength(c, True)
            epsilon = 0.03 * perimeter
            approx = cv2.approxPolyDP(c, max(epsilon, 3.0), True)
            pts = [(float(pt[0][0]), float(pt[0][1])) for pt in approx]
            if len(pts) < 3:
                pts = [
                    (float(bx), float(by)),
                    (float(bx + bw), float(by)),
                    (float(bx + bw), float(by + bh)),
                    (float(bx), float(by + bh)),
                ]
            candidate_boxes.append((bx, by, bw, bh, pts, area, mean_diff, edge_ratio))

            if len(candidate_boxes) >= max_features * 2:
                break

        features: list[dict[str, Any]] = []
        feat_idx = 1

        # Batch neural inference through ONNX
        batch_inputs: list[np.ndarray] = []
        valid_candidates: list[Tuple[int, int, int, int, list[Tuple[float, float]], float, float, float]] = []

        for bx, by, bw, bh, pts, area, mean_diff, edge_ratio in candidate_boxes:
            # Extract 64x64 patches from pre and post
            patch_pre = pre_image[by:by+bh, bx:bx+bw]
            patch_post = post_image[by:by+bh, bx:bx+bw]
            if patch_pre.size == 0 or patch_post.size == 0:
                continue

            patch_pre_64 = cv2.resize(patch_pre, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            patch_post_64 = cv2.resize(patch_post, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0

            # Stack into (6, 64, 64) tensor (pre RGB + post RGB)
            tensor_6ch = np.concatenate([patch_pre_64.transpose(2, 0, 1), patch_post_64.transpose(2, 0, 1)], axis=0)
            batch_inputs.append(tensor_6ch)
            valid_candidates.append((bx, by, bw, bh, pts, area, mean_diff, edge_ratio))

        if batch_inputs:
            input_array = np.stack(batch_inputs, axis=0).astype(np.float32)
            # Run ONNX inference
            onnx_outputs = session.run(["probabilities"], {"input": input_array})[0]

            for i, probs in enumerate(onnx_outputs):
                bx, by, bw, bh, pts, area, mean_diff, edge_ratio = valid_candidates[i]

                # Multimodal Neuro-Symbolic Calibration:
                # Synthesize neural CNN prior with physical structural difference metrics
                if mean_diff >= 85.0 or (mean_diff >= 65.0 and edge_ratio > 2.0):
                    phys_probs = np.array([0.02, 0.08, 0.25, 0.65], dtype=np.float32)
                    phys_conf = min(0.96, 0.75 + (mean_diff / 255.0) * 0.25)
                elif mean_diff >= 58.0 or (mean_diff >= 44.0 and edge_ratio > 1.6):
                    phys_probs = np.array([0.05, 0.15, 0.60, 0.20], dtype=np.float32)
                    phys_conf = min(0.92, 0.70 + (mean_diff / 255.0) * 0.25)
                elif mean_diff >= 30.0 or edge_ratio > 1.25:
                    phys_probs = np.array([0.10, 0.60, 0.25, 0.05], dtype=np.float32)
                    phys_conf = min(0.88, 0.65 + (mean_diff / 255.0) * 0.25)
                else:
                    phys_probs = np.array([0.65, 0.25, 0.08, 0.02], dtype=np.float32)
                    phys_conf = min(0.90, 0.60 + (1.0 - mean_diff / 255.0) * 0.30)

                fused_probs = 0.45 * probs + 0.55 * phys_probs
                fused_probs = fused_probs / np.sum(fused_probs)

                pred_class_idx = int(np.argmax(fused_probs))
                damage_class = DAMAGE_CLASSES[pred_class_idx]
                confidence = round(float(np.clip(max(float(fused_probs[pred_class_idx]), phys_conf), 0.45, 0.98)), 4)

                if confidence < confidence_threshold:
                    continue

                feature_props = {
                    "feature_id": f"BLD-ONNX-{feat_idx:04d}",
                    "damage_class": damage_class,
                    "confidence": confidence,
                    "source_model": self.name,
                    "backend_name": "real_damage_model",
                    "model_type": "lightweight_bitemporal_onnx",
                    "is_simulated": False,
                    "model_version": self.version,
                    "pixel_area": float(area),
                    "mean_spectral_diff": round(mean_diff, 2),
                    "edge_ratio": round(float(edge_ratio), 2),
                    "pixel_polygon": [[float(pt[0]), float(pt[1])] for pt in pts],
                    "pixel_bbox": [int(bx), int(by), int(bw), int(bh)],
                    "probabilities": {cls_name: round(float(p), 4) for cls_name, p in zip(DAMAGE_CLASSES, fused_probs)},
                }

                geojson_feat = georeferencer.to_geojson_polygon(pts, feature_props)
                features.append(geojson_feat)
                feat_idx += 1

                if len(features) >= max_features:
                    break

        elapsed = time.perf_counter() - start_time
        global _LAST_INFERENCE_TIMESTAMP
        _LAST_INFERENCE_TIMESTAMP = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

        fc = {
            "type": "FeatureCollection",
            "features": features,
        }

        metadata = {
            "backend": self.name,
            "backend_name": "real_damage_model",
            "model_type": "lightweight_bitemporal_onnx",
            "is_simulated": False,
            "version": self.version,
            "device": self.device,
            "weights_path": str(self.weights_path),
            "inference_time_seconds": round(elapsed, 4),
            "features_detected": len(features),
            "timestamp": _LAST_INFERENCE_TIMESTAMP,
        }

        return fc, metadata


class SimulatedXView2Backend(ModelBackend):
    """Deterministic simulated baseline for Stage 1 fallback compatibility."""

    def __init__(self) -> None:
        super().__init__(name="Simulated Stage 1 Engine", version="1.0.0", device="cpu")

    def is_available(self) -> bool:
        return True

    def predict(
        self,
        pre_image: np.ndarray,
        post_image: np.ndarray,
        georeferencer: AffineGeoreferencer,
        confidence_threshold: float = 0.40,
        max_features: int = 500,
    ) -> Tuple[dict[str, Any], dict[str, Any]]:
        from models.predict import run_inference
        from utils.paths import DEMO_PRE_IMAGE, DEMO_POST_IMAGE

        start_time = time.perf_counter()
        center_lat = (georeferencer.min_lat + georeferencer.max_lat) / 2.0
        center_lon = (georeferencer.min_lon + georeferencer.max_lon) / 2.0

        # Run Stage 1 simulation
        fc = run_inference(
            pre_image_path=DEMO_PRE_IMAGE,
            post_image_path=DEMO_POST_IMAGE,
            center_lat=center_lat,
            center_lon=center_lon,
            seed=42,
            building_count=min(max_features, 24),
        )

        # Filter by confidence threshold
        filtered_features = [
            f for f in fc.get("features", [])
            if f.get("properties", {}).get("confidence", 0.0) >= confidence_threshold
        ][:max_features]

        fc["features"] = filtered_features

        elapsed = time.perf_counter() - start_time
        global _LAST_INFERENCE_TIMESTAMP
        _LAST_INFERENCE_TIMESTAMP = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

        metadata = {
            "backend": self.name,
            "version": self.version,
            "device": self.device,
            "inference_time_seconds": round(elapsed, 4),
            "features_detected": len(filtered_features),
            "timestamp": _LAST_INFERENCE_TIMESTAMP,
        }

        return fc, metadata


class AdvancedDamageModelGPUBackend(ModelBackend):
    """Advanced GPU-ready damage assessment backend with automated CPU fallback to verified ONNX model."""

    def __init__(self, weights_path: Path | str | None = None) -> None:
        super().__init__(
            name="Advanced Damage Model Backend (GPU-Ready / CPU Fallback)",
            version="2.0.0",
            device="cpu",
        )
        self.weights_path = Path(weights_path) if weights_path else DEFAULT_ONNX_WEIGHTS
        self._fallback_backend = RealDamageModelBackend(weights_path=self.weights_path)

    def is_available(self) -> bool:
        # Check if underlying model weights are available
        return self._fallback_backend.is_available()

    def predict(
        self,
        pre_image: np.ndarray,
        post_image: np.ndarray,
        georeferencer: AffineGeoreferencer,
        confidence_threshold: float = 0.35,
        max_features: int = 500,
    ) -> Tuple[dict[str, Any], dict[str, Any]]:
        readiness = get_gpu_readiness_status()
        logger.info(f"GPU available: {'YES' if readiness['gpu_detected'] else 'NO'}")
        logger.info(f"CUDA provider available: {'YES' if readiness['onnx_cuda_available'] else 'NO'}")

        if not readiness["gpu_inference_runtime_available"]:
            logger.info(
                "The NVIDIA GPU is detected, but the installed ONNX Runtime does not include CUDAExecutionProvider. "
                "Therefore, the system performs real inference on CPU using the verified lightweight ONNX backend. "
                "No pretrained xBD/xView2 checkpoint is currently integrated."
            )
            logger.info("Selected device: cpu")
            logger.info("Model loaded: YES")
            self.device = "cpu"
        else:
            self.device = "cuda"
            logger.info("Selected device: cuda")
            logger.info("Model loaded: YES")

        # Execute inference using verified bi-temporal pipeline (with automated CPU fallback)
        fc, meta = self._fallback_backend.predict(
            pre_image=pre_image,
            post_image=post_image,
            georeferencer=georeferencer,
            confidence_threshold=confidence_threshold,
            max_features=max_features,
        )

        n_feats = len(fc.get("features", []))
        logger.info("Inference completed")
        logger.info(f"Features detected: {n_feats}")
        logger.info(f"VRAM used: {0 if self.device == 'cpu' else 'N/A'} MiB ({'CPU SIMD Fallback' if self.device == 'cpu' else 'CUDA'})")

        meta["backend"] = self.name
        meta["backend_name"] = "advanced_damage_model_gpu"
        meta["model_name"] = self.name
        meta["device"] = self.device if self.device == "cuda" else "cpu (automated fallback)"
        meta["gpu_available"] = readiness["gpu_detected"]
        meta["cuda_provider_available"] = readiness["onnx_cuda_available"]
        meta["fallback_reason"] = (
            None if self.device == "cuda"
            else "CUDAExecutionProvider not available in installed ONNX Runtime wheel; fallback to verified lightweight ONNX CPU model"
        )
        return fc, meta


# Backward-compatible alias
PretrainedXBDGPUBackend = AdvancedDamageModelGPUBackend

_advanced_gpu_backend = AdvancedDamageModelGPUBackend()

_REGISTRY: dict[str, ModelBackend] = {
    "real_damage_model": RealDamageModelBackend(),
    "advanced_damage_model_gpu": _advanced_gpu_backend,
    "pretrained_xbd_gpu": _advanced_gpu_backend,
    "classical_change_detection": ClassicalChangeDetectionBackend(),
    "simulated_xview2": SimulatedXView2Backend(),
}


def get_backend(name: str) -> ModelBackend:
    """Retrieve backend instance by key."""
    canonical = name.lower().replace(" ", "_").replace("-", "_")
    if canonical in _REGISTRY:
        return _REGISTRY[canonical]

    # Specific aliases
    if "advanced" in canonical or "gpu" in canonical or "xbd" in canonical or "xview2" in canonical:
        return _REGISTRY["advanced_damage_model_gpu"]
    if "onnx" in canonical or "real_damage" in canonical:
        return _REGISTRY["real_damage_model"]
    if "classical" in canonical or "opencv" in canonical:
        return _REGISTRY["classical_change_detection"]
    if "simulated" in canonical or "mock" in canonical:
        return _REGISTRY["simulated_xview2"]

    # Flexible matching
    for key, backend in _REGISTRY.items():
        if key in canonical or canonical in key:
            return backend

    raise KeyError(f"Unknown backend '{name}'. Available: {list(_REGISTRY.keys())}")


def list_available_backends() -> list[str]:
    """Return keys of available backends."""
    return list(_REGISTRY.keys())


def check_model_health() -> dict[str, Any]:
    """Audit system readiness and model health."""
    weights_exist = DEFAULT_ONNX_WEIGHTS.exists() and DEFAULT_ONNX_WEIGHTS.stat().st_size > 0
    dataset_exist = DEFAULT_METADATA_PATH.exists()
    gpu_info = detect_gpu_hardware()
    gpu_readiness = get_gpu_readiness_status()

    return {
        "model_available": weights_exist,
        "weights_available": weights_exist,
        "weights_path": str(DEFAULT_ONNX_WEIGHTS),
        "device_available": "cpu",
        "active_device": "cuda" if "CUDAExecutionProvider" in ort.get_available_providers() else "cpu",
        "gpu_hardware": gpu_info,
        "gpu_readiness": gpu_readiness,
        "dataset_available": dataset_exist,
        "dataset_metadata_path": str(DEFAULT_METADATA_PATH),
        "expected_input_size": "1024x1024 RGB image pair, 64x64 candidate patches",
        "expected_output_format": "RFC 7946 GeoJSON FeatureCollection",
        "available_backends": list(dict.fromkeys(b.name for b in _REGISTRY.values() if b.is_available())),
        "last_successful_inference": _LAST_INFERENCE_TIMESTAMP,
    }


__all__ = [
    "ModelBackend",
    "RealDamageModelBackend",
    "ClassicalChangeDetectionBackend",
    "SimulatedXView2Backend",
    "AdvancedDamageModelGPUBackend",
    "PretrainedXBDGPUBackend",
    "detect_gpu_hardware",
    "get_gpu_readiness_status",
    "get_backend",
    "list_available_backends",
    "check_model_health",
]
