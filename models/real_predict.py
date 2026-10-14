"""Real damage assessment inference entrypoint for MERIDIAN-Lite Stage 2.

Dispatches inference to the active ModelBackend (ONNX CPU model or OpenCV classical change detection),
georeferences extracted building footprints, and returns an RFC 7946 compliant GeoJSON FeatureCollection.
"""

import json
from pathlib import Path
from typing import Any
import numpy as np

from models.model_registry import (
    ClassicalChangeDetectionBackend,
    RealDamageModelBackend,
    get_backend,
)
from utils.georeferencing import AffineGeoreferencer
from utils.image_utils import load_image, validate_image
from utils.logging_utils import get_logger
from utils.paths import PROJECT_ROOT

logger = get_logger("real_predict")

METADATA_FILE = PROJECT_ROOT / "data" / "processed" / "dataset_metadata.json"
DEFAULT_HURRICANE_BOUNDS = (-85.550100, 30.152997, -85.547337, 30.155760)


def _resolve_bounds(
    image_width: int,
    image_height: int,
    center_lat: float | None,
    center_lon: float | None,
    bounds: tuple[float, float, float, float] | None,
) -> AffineGeoreferencer:
    """Determine the appropriate georeferencer for the inference pair."""
    if bounds is not None:
        return AffineGeoreferencer(
            image_width=image_width,
            image_height=image_height,
            bounds=bounds,
            coordinate_source="georeferenced_bounds",
        )

    if center_lat is not None and center_lon is not None:
        return AffineGeoreferencer.from_center_and_radius(
            image_width=image_width,
            image_height=image_height,
            center_lat=center_lat,
            center_lon=center_lon,
            delta_deg=0.005,
            coordinate_source="user_bbox_approximation",
        )

    # Check dataset metadata for known dataset bounds
    if METADATA_FILE.exists():
        try:
            with open(METADATA_FILE, "r") as f:
                meta = json.load(f)
                bbox = meta.get("bounding_box", {})
                if "min_lon" in bbox and "min_lat" in bbox:
                    b = (
                        float(bbox["min_lon"]),
                        float(bbox["min_lat"]),
                        float(bbox["max_lon"]),
                        float(bbox["max_lat"]),
                    )
                    return AffineGeoreferencer(
                        image_width=image_width,
                        image_height=image_height,
                        bounds=b,
                        coordinate_source="georeferenced_bounds",
                    )
        except Exception as exc:
            logger.warning(f"Could not read dataset metadata bounds: {exc}")

    # Fallback default
    return AffineGeoreferencer(
        image_width=image_width,
        image_height=image_height,
        bounds=DEFAULT_HURRICANE_BOUNDS,
        coordinate_source="georeferenced_bounds",
    )


def run_real_inference(
    pre_image_path: str | Path,
    post_image_path: str | Path,
    center_lat: float | None = None,
    center_lon: float | None = None,
    confidence_threshold: float = 0.40,
    max_features: int = 500,
    backend_name: str = "real_damage_model",
    bounds: tuple[float, float, float, float] | None = None,
) -> dict[str, Any]:
    """Execute real disaster damage assessment inference on an aerial imagery pair.

    Args:
        pre_image_path: Path to pre-disaster RGB image.
        post_image_path: Path to post-disaster RGB image.
        center_lat: Optional latitude center for user-provided bounding box approximation.
        center_lon: Optional longitude center for user-provided bounding box approximation.
        confidence_threshold: Minimum confidence score [0.0 - 1.0] to include a feature.
        max_features: Maximum number of building features to return.
        backend_name: Name of backend to use ('real_damage_model', 'classical_change_detection', 'simulated_xview2').
        bounds: Optional explicit (min_lon, min_lat, max_lon, max_lat).

    Returns:
        RFC 7946 GeoJSON FeatureCollection dict.
    """
    logger.info(f"Running real inference with backend '{backend_name}' on {pre_image_path} and {post_image_path}")

    # 1. Validate image paths
    validate_image(pre_image_path)
    validate_image(post_image_path)

    # 2. Load images as RGB numpy arrays
    pre_img = load_image(pre_image_path)
    post_img = load_image(post_image_path)

    h, w = pre_img.shape[:2]
    if h == 0 or w == 0:
        raise ValueError(f"Invalid image dimensions: {w}x{h}")

    # 3. Setup georeferencing
    georeferencer = _resolve_bounds(
        image_width=w,
        image_height=h,
        center_lat=center_lat,
        center_lon=center_lon,
        bounds=bounds,
    )

    # 4. Select backend with graceful fallback
    try:
        backend = get_backend(backend_name)
        if not backend.is_available():
            logger.warning(f"Backend '{backend_name}' not fully available; falling back to classical change detection.")
            backend = ClassicalChangeDetectionBackend()
    except Exception as exc:
        logger.warning(f"Failed to load backend '{backend_name}': {exc}; falling back to classical change detection.")
        backend = ClassicalChangeDetectionBackend()

    # 5. Run inference
    fc, meta = backend.predict(
        pre_image=pre_img,
        post_image=post_img,
        georeferencer=georeferencer,
        confidence_threshold=confidence_threshold,
        max_features=max_features,
    )

    # 6. Attach image identifier and run metadata to feature properties
    pre_id = Path(pre_image_path).name
    post_id = Path(post_image_path).name

    for feat in fc.get("features", []):
        props = feat.setdefault("properties", {})
        props.setdefault("image_identifiers", {"pre": pre_id, "post": post_id})
        props.setdefault("inference_time_seconds", meta.get("inference_time_seconds"))
        props.setdefault("backend_name", meta.get("backend_name", backend_name))
        props.setdefault("model_type", meta.get("model_type", "lightweight_bitemporal_onnx" if backend_name == "real_damage_model" else backend_name))
        props.setdefault("is_simulated", meta.get("is_simulated", False))
        props.setdefault("preprocessing_metadata", {
            "image_width": w,
            "image_height": h,
            "channels": 3,
            "crop_patch_size": 64,
        })

    # Record overall run metadata in FeatureCollection top-level
    fc["metadata"] = {
        **meta,
        "pre_image": pre_id,
        "post_image": post_id,
        "backend_name": meta.get("backend_name", backend_name),
        "model_type": meta.get("model_type", "lightweight_bitemporal_onnx" if backend_name == "real_damage_model" else backend_name),
        "is_simulated": meta.get("is_simulated", False),
        "confidence_threshold": confidence_threshold,
        "max_features_requested": max_features,
        "total_features_returned": len(fc.get("features", [])),
    }

    return fc
