"""
Simulated AI Inference Engine for MERIDIAN-Lite (Stage 1 Baseline).

NOTE AND DISCLAIMER:
This module provides a deterministic, simulated building-damage assessment
pipeline designed for Stage 1 integration testing, geospatial verification,
and UI readiness. It does NOT run an actual trained deep learning model
(such as a production xView2/xBD dual-backbone segmentation network).
The function signatures and return schemas are deliberately constructed to
mirror standard xView2 polygon localization and classification interfaces,
ensuring seamless drop-in replacement by a trained model in future iterations.
"""

from pathlib import Path
from typing import Any, Dict, List
import numpy as np

from utils.image_utils import load_image, validate_image
from utils.logging_utils import get_logger
from utils.geojson_utils import validate_feature_collection

logger = get_logger("predict")

EXACT_DAMAGE_CLASSES = [
    "no-damage",
    "minor-damage",
    "major-damage",
    "destroyed",
]


def run_inference(
    pre_image_path: str | Path,
    post_image_path: str | Path,
    center_lat: float,
    center_lon: float,
    seed: int = 42,
    building_count: int = 18,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Run simulated damage assessment across an aerial image pair.

    Validates pre/post images, deterministically generates clustered building
    polygons around (center_lat, center_lon), assigns standardized xView2
    damage classes with confidences, and returns a verified GeoJSON
    FeatureCollection.

    Args:
        pre_image_path: Path to pre-disaster aerial imagery file.
        post_image_path: Path to post-disaster aerial imagery file.
        center_lat: Center latitude in decimal degrees.
        center_lon: Center longitude in decimal degrees.
        seed: Random seed for deterministic simulation.
        building_count: Requested building polygon count (clamped to 15-20).

    Returns:
        GeoJSON FeatureCollection as a Python dictionary.
    """
    logger.info("Initializing Stage 1 simulated inference engine...")
    logger.info(
        f"Inputs: pre='{pre_image_path}', post='{post_image_path}', "
        f"center=({center_lat}, {center_lon}), seed={seed}, requested_buildings={building_count}"
    )

    # 1-4. Validate and load images
    validate_image(pre_image_path)
    validate_image(post_image_path)

    pre_img = load_image(pre_image_path)
    post_img = load_image(post_image_path)

    if pre_img.size == 0 or post_img.size == 0:
        raise ValueError("Loaded image array has size 0 (empty image)")

    # 5. Isolated deterministic random number generator (avoids global state pollution)
    rng = np.random.RandomState(seed)

    # 6-7. Clamp building count safely to [15, 20]
    count = max(15, min(20, int(building_count)))
    logger.info(f"Generating {count} simulated building damage features...")

    # 8-10. Generate clustered building footprints around (center_lat, center_lon)
    features: List[Dict[str, Any]] = []

    # Approximate 1 degree lat ~ 111,000 meters.
    # We scatter buildings within a ~600m bounding radius (approx 0.0055 degrees).
    # Typical building footprint size: ~20m to 45m (~0.00018 to 0.00040 degrees).
    radius_deg = 0.0040

    # Create grid offsets with jitter to ensure realistic non-overlapping distribution
    grid_cols = int(np.ceil(np.sqrt(count)))
    grid_rows = int(np.ceil(count / grid_cols))

    lat_steps = np.linspace(-radius_deg, radius_deg, grid_rows)
    lon_steps = np.linspace(-radius_deg, radius_deg, grid_cols)

    building_idx = 0
    for r in range(grid_rows):
        for c in range(grid_cols):
            if building_idx >= count:
                break

            base_d_lat = lat_steps[r]
            base_d_lon = lon_steps[c]

            # Deterministic jitter (up to 30% of step distance)
            jitter_lat = rng.uniform(-radius_deg * 0.15, radius_deg * 0.15)
            jitter_lon = rng.uniform(-radius_deg * 0.15, radius_deg * 0.15)

            # Dimensions for building rectangle (in degrees)
            b_width_deg = rng.uniform(0.00025, 0.00050)
            b_height_deg = rng.uniform(0.00020, 0.00045)

            # South-West corner of the building
            sw_lat = float(center_lat + base_d_lat + jitter_lat)
            sw_lon = float(center_lon + base_d_lon + jitter_lon)

            # Construct rectangular polygon with GeoJSON [longitude, latitude] ordering
            # Order: SW -> SE -> NE -> NW -> SW (closed ring)
            p_sw = [round(sw_lon, 7), round(sw_lat, 7)]
            p_se = [round(sw_lon + b_width_deg, 7), round(sw_lat, 7)]
            p_ne = [round(sw_lon + b_width_deg, 7), round(sw_lat + b_height_deg, 7)]
            p_nw = [round(sw_lon, 7), round(sw_lat + b_height_deg, 7)]
            p_close = [round(sw_lon, 7), round(sw_lat, 7)]

            ring = [p_sw, p_se, p_ne, p_nw, p_close]

            # 11. Assign damage class with realistic disaster distribution
            # 35% no-damage, 25% minor-damage, 25% major-damage, 15% destroyed
            damage_class = rng.choice(
                EXACT_DAMAGE_CLASSES,
                p=[0.35, 0.25, 0.25, 0.15],
            )

            # 12. Assign confidence between 0.75 and 0.99
            confidence = round(float(rng.uniform(0.75, 0.99)), 4)

            # 13. Stable feature ID
            feat_id = f"building-{building_idx + 1:03d}"

            # 14. Feature dictionary
            feature = {
                "type": "Feature",
                "id": feat_id,
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [ring],
                },
                "properties": {
                    "damage_class": str(damage_class),
                    "confidence": confidence,
                    "source": "simulated_xview2_baseline",
                    "is_simulated": True,
                },
            }
            features.append(feature)
            building_idx += 1

    geojson_collection: Dict[str, Any] = {
        "type": "FeatureCollection",
        "metadata": {
            "system": "MERIDIAN-Lite",
            "stage": "Stage 1",
            "model_type": "simulated_xview2_baseline",
            "is_simulated": True,
            "seed": seed,
            "center_lat": center_lat,
            "center_lon": center_lon,
            "building_count": count,
        },
        "features": features,
    }

    # Strict output verification before returning
    validate_feature_collection(geojson_collection)
    logger.info(
        f"Inference successfully simulated: {len(features)} building damage features generated."
    )

    return geojson_collection
