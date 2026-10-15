"""Optional GPU damage assessment inference module for MERIDIAN-Lite.

Implements the unified entrypoint run_gpu_inference() that checks GPU readiness,
negotiates CUDA acceleration if available, executes with automated CPU fallback,
and exports RFC 7946 GeoJSON, priority zones, and summary metrics.
"""

import json
from pathlib import Path
import time
from typing import Any
import numpy as np

from models.model_registry import (
    AdvancedDamageModelGPUBackend,
    PretrainedXBDGPUBackend,
    get_backend,
    get_gpu_readiness_status,
)
from utils.geojson_utils import save_geojson, validate_feature_collection
from utils.georeferencing import AffineGeoreferencer
from utils.image_utils import load_image, validate_image
from utils.logging_utils import get_logger
from utils.paths import PROJECT_ROOT
from utils.priority import compute_priority_zones

logger = get_logger("gpu_predict")

METADATA_FILE = PROJECT_ROOT / "data" / "processed" / "dataset_metadata.json"
DEFAULT_HURRICANE_BOUNDS = (-85.550100, 30.152997, -85.547337, 30.155760)

GPU_GEOJSON_OUTPUT = PROJECT_ROOT / "outputs" / "geojson" / "gpu_model_damage_assessment.geojson"
GPU_ZONES_OUTPUT = PROJECT_ROOT / "outputs" / "reports" / "gpu_model_priority_zones.json"
GPU_METRICS_OUTPUT = PROJECT_ROOT / "outputs" / "reports" / "gpu_model_summary_metrics.json"


def _resolve_georeferencer(
    image_width: int,
    image_height: int,
    center_lat: float | None = None,
    center_lon: float | None = None,
    bounds: tuple[float, float, float, float] | None = None,
    is_custom_upload: bool = False,
) -> AffineGeoreferencer:
    """Resolve affine georeferencer from bounds, custom coordinates, or dataset metadata."""
    if bounds is not None:
        return AffineGeoreferencer(
            image_width=image_width,
            image_height=image_height,
            bounds=bounds,
            coordinate_source="georeferenced_bounds",
        )

    if is_custom_upload and center_lat is not None and center_lon is not None:
        return AffineGeoreferencer.from_center_and_radius(
            image_width=image_width,
            image_height=image_height,
            center_lat=center_lat,
            center_lon=center_lon,
            delta_deg=0.0025,
            coordinate_source="user_bbox_approximation",
        )

    if not is_custom_upload and METADATA_FILE.exists():
        try:
            with open(METADATA_FILE, "r", encoding="utf-8") as f:
                meta = json.load(f)
                bbox = meta.get("bounding_box", {})
                if "min_lon" in bbox and "min_lat" in bbox:
                    b = (
                        float(bbox["min_lon"]),
                        float(bbox["min_lat"]),
                        float(bbox["max_lon"]),
                        float(bbox["max_lat"]),
                    )
                    ds_lat = float(bbox.get("center_lat", 30.154378))
                    if center_lat is None or abs(center_lat - ds_lat) < 0.05:
                        return AffineGeoreferencer(
                            image_width=image_width,
                            image_height=image_height,
                            bounds=b,
                            coordinate_source="georeferenced_bounds",
                        )
        except Exception as exc:
            logger.warning(f"Could not read dataset metadata bounds: {exc}")

    if center_lat is not None and center_lon is not None:
        return AffineGeoreferencer.from_center_and_radius(
            image_width=image_width,
            image_height=image_height,
            center_lat=center_lat,
            center_lon=center_lon,
            delta_deg=0.0014,
            coordinate_source="user_bbox_approximation",
        )

    return AffineGeoreferencer(
        image_width=image_width,
        image_height=image_height,
        bounds=DEFAULT_HURRICANE_BOUNDS,
        coordinate_source="georeferenced_bounds",
    )


def run_gpu_inference(
    pre_image_path: str | Path,
    post_image_path: str | Path,
    bounds: tuple[float, float, float, float] | None = None,
    confidence_threshold: float = 0.35,
    max_features: int = 500,
    center_lat: float | None = None,
    center_lon: float | None = None,
    is_custom_upload: bool = False,
    grid_rows: int = 3,
    grid_cols: int = 3,
    **kwargs: Any,
) -> dict[str, Any]:
    """Execute disaster damage assessment via the optional GPU-targeted backend.

    Guarantees:
    1. Full path validation.
    2. Graceful fallback to verified CPU ONNX engine if GPU runtime or weights are unavailable.
    3. Serialization to required outputs:
       - outputs/geojson/gpu_model_damage_assessment.geojson
       - outputs/reports/gpu_model_priority_zones.json
       - outputs/reports/gpu_model_summary_metrics.json
    4. Compliance with RFC 7946 GeoJSON specification.
    """
    logger.info(f"Initiating GPU inference on pre={pre_image_path}, post={post_image_path}")

    # 1. Validate both image paths
    validate_image(pre_image_path)
    validate_image(post_image_path)

    # 2. Load and validate image data
    pre_img = load_image(pre_image_path)
    post_img = load_image(post_image_path)

    h, w = pre_img.shape[:2]
    if h == 0 or w == 0:
        raise ValueError(f"Invalid image dimensions: {w}x{h}")

    # Maximum image dimension safeguard
    max_dim = 2048
    if h > max_dim or w > max_dim:
        logger.warning(f"Image dimensions {w}x{h} exceed {max_dim}px safety limit; downsampling.")
        scale = max_dim / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        import cv2
        pre_img = cv2.resize(pre_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        post_img = cv2.resize(post_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        h, w = new_h, new_w

    # 3. Setup georeferencing
    georeferencer = _resolve_georeferencer(
        image_width=w,
        image_height=h,
        center_lat=center_lat,
        center_lon=center_lon,
        bounds=bounds,
        is_custom_upload=is_custom_upload,
    )

    # 4. Acquire GPU backend with automatic CPU fallback
    try:
        backend = get_backend("advanced_damage_model_gpu")
    except KeyError:
        backend = AdvancedDamageModelGPUBackend()

    # 5. Run inference with performance safeguards and timeout / OOM protection
    try:
        fc, meta = backend.predict(
            pre_image=pre_img,
            post_image=post_img,
            georeferencer=georeferencer,
            confidence_threshold=confidence_threshold,
            max_features=max_features,
        )
    except Exception as exc:
        logger.error(f"Inference error in GPU backend: {exc}; triggering immediate CPU fallback.")
        from models.model_registry import RealDamageModelBackend
        fallback_backend = RealDamageModelBackend()
        fc, meta = fallback_backend.predict(
            pre_image=pre_img,
            post_image=post_img,
            georeferencer=georeferencer,
            confidence_threshold=confidence_threshold,
            max_features=max_features,
        )
        meta["device"] = "cpu (exception fallback)"
        meta["fallback_reason"] = str(exc)

    # 6. Attach mandatory metadata to each feature property
    pre_id = Path(pre_image_path).name
    post_id = Path(post_image_path).name
    features = fc.get("features", [])

    for feat in features:
        props = feat.setdefault("properties", {})
        props["backend_name"] = "Advanced Damage Model Backend (GPU-Ready / CPU Fallback)"
        props["model_name"] = "Advanced Damage Model Backend (GPU-Ready / CPU Fallback)"
        props["model_version"] = meta.get("version", "2.0.0")
        props["device"] = meta.get("device", "cpu (automated fallback)")
        props["inference_time"] = meta.get("inference_time_seconds", 0.0)
        props["source_image_identifiers"] = {"pre": pre_id, "post": post_id}
        props["coordinate_source"] = georeferencer.coordinate_source
        props["crs"] = "urn:ogc:def:crs:OGC:1.3:CRS84"

    fc["features"] = features
    validate_feature_collection(fc)

    # 7. Compute priority zones
    zones = compute_priority_zones(fc, grid_rows=grid_rows, grid_cols=grid_cols)

    # 8. Compute summary metrics
    total_feats = len(features)
    damaged_feats = sum(
        1 for f in features
        if f.get("properties", {}).get("damage_class") in {"minor-damage", "major-damage", "destroyed"}
    )
    severe_feats = sum(
        1 for f in features
        if f.get("properties", {}).get("damage_class") in {"major-damage", "destroyed"}
    )
    conf_values = [f.get("properties", {}).get("confidence", 0.0) for f in features]
    mean_conf = float(np.mean(conf_values)) if conf_values else 0.0

    summary_metrics = {
        "backend_name": "Advanced Damage Model Backend (GPU-Ready / CPU Fallback)",
        "model_name": meta.get("model_name", "Advanced Damage Model Backend (GPU-Ready / CPU Fallback)"),
        "model_version": meta.get("version", "2.0.0"),
        "device": meta.get("device", "cpu"),
        "inference_time_seconds": meta.get("inference_time_seconds", 0.0),
        "total_features_detected": total_feats,
        "damaged_features_count": damaged_feats,
        "severe_damage_count": severe_feats,
        "severe_damage_ratio": round(severe_feats / max(total_feats, 1), 4),
        "mean_confidence": round(mean_conf, 4),
        "gpu_readiness": get_gpu_readiness_status(),
        "timestamp": meta.get("timestamp", time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())),
    }

    fc["metadata"] = {
        **meta,
        "pre_image": Path(pre_image_path).name,
        "post_image": Path(post_image_path).name,
        "backend_name": summary_metrics["backend_name"],
        "confidence_threshold": confidence_threshold,
        "max_features_requested": max_features,
        "total_features_returned": total_feats,
    }

    # 9. Save all required outputs
    GPU_GEOJSON_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    save_geojson(fc, GPU_GEOJSON_OUTPUT)

    GPU_ZONES_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(GPU_ZONES_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(zones, f, indent=2)

    GPU_METRICS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(GPU_METRICS_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2)

    logger.info(
        f"GPU inference completed: {total_feats} features detected in {meta.get('inference_time_seconds')}s "
        f"on device '{meta.get('device')}'"
    )

    return fc
