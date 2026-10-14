"""
GeoJSON parsing, validation, serialization, bounds extraction, and color styling.
Enforces RFC 7946 GeoJSON compliance, polygon ring closure, coordinate bounds,
and required building damage schema.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
from utils.logging_utils import get_logger

logger = get_logger("geojson_utils")

ALLOWED_DAMAGE_CLASSES = {
    "no-damage",
    "minor-damage",
    "major-damage",
    "destroyed",
}

DAMAGE_COLORS = {
    "no-damage": "green",
    "minor-damage": "yellow",
    "major-damage": "orange",
    "destroyed": "red",
}

DAMAGE_HEX_COLORS = {
    "no-damage": "#28a745",
    "minor-damage": "#ffc107",
    "major-damage": "#fd7e14",
    "destroyed": "#dc3545",
}


def damage_color(damage_class: str, as_hex: bool = False) -> str:
    """
    Retrieve mapped color for damage classification.
    Returns standard color name ('green', 'yellow', 'orange', 'red')
    or hex code ('#28a745', '#ffc107', '#fd7e14', '#dc3545') if as_hex=True.
    """
    cleaned_class = damage_class.strip().lower() if isinstance(damage_class, str) else ""
    if as_hex:
        return DAMAGE_HEX_COLORS.get(cleaned_class, "#6c757d")
    return DAMAGE_COLORS.get(cleaned_class, "gray")


def validate_feature_collection(data: Dict[str, Any]) -> None:
    """
    Strictly validate a GeoJSON FeatureCollection dictionary.
    Raises ValueError with actionable explanations if invalid.
    """
    if not isinstance(data, dict):
        raise ValueError(f"GeoJSON data must be a dictionary, got {type(data).__name__}")

    if data.get("type") != "FeatureCollection":
        raise ValueError(f"Expected GeoJSON type 'FeatureCollection', got '{data.get('type')}'")

    features = data.get("features")
    if not isinstance(features, list):
        raise ValueError("GeoJSON 'features' field must be a list")

    for idx, feature in enumerate(features):
        if not isinstance(feature, dict):
            raise ValueError(f"Feature at index {idx} must be a dictionary")

        if feature.get("type") != "Feature":
            raise ValueError(f"Feature {idx} type must be 'Feature', got '{feature.get('type')}'")

        geometry = feature.get("geometry")
        if not isinstance(geometry, dict):
            raise ValueError(f"Feature {idx} geometry must be a dictionary")

        geom_type = geometry.get("type")
        if geom_type != "Polygon":
            raise ValueError(f"Feature {idx} geometry type must be 'Polygon', got '{geom_type}'")

        coordinates = geometry.get("coordinates")
        if not isinstance(coordinates, list) or len(coordinates) == 0:
            raise ValueError(f"Feature {idx} geometry coordinates must be a non-empty list of rings")

        for ring_idx, ring in enumerate(coordinates):
            if not isinstance(ring, list) or len(ring) < 4:
                raise ValueError(
                    f"Feature {idx} ring {ring_idx} must contain at least 4 positions for a closed polygon"
                )

            # Check polygon closure: ring[0] == ring[-1]
            first_pt, last_pt = ring[0], ring[-1]
            if len(first_pt) < 2 or len(last_pt) < 2:
                raise ValueError(f"Feature {idx} ring positions must contain [lon, lat]")

            if not (abs(first_pt[0] - last_pt[0]) < 1e-7 and abs(first_pt[1] - last_pt[1]) < 1e-7):
                raise ValueError(
                    f"Feature {idx} polygon ring {ring_idx} is not closed: "
                    f"start={first_pt} != end={last_pt}"
                )

            # Verify coordinate ranges: lon in [-180, 180], lat in [-90, 90]
            for pt_idx, pt in enumerate(ring):
                if not (isinstance(pt, (list, tuple)) and len(pt) >= 2):
                    raise ValueError(f"Feature {idx} ring point {pt_idx} must be a [lon, lat] sequence")
                lon, lat = pt[0], pt[1]
                if not (-180.0 <= lon <= 180.0):
                    raise ValueError(
                        f"Feature {idx} longitude {lon} out of valid range [-180, 180]"
                    )
                if not (-90.0 <= lat <= 90.0):
                    raise ValueError(
                        f"Feature {idx} latitude {lat} out of valid range [-90, 90]"
                    )

        properties = feature.get("properties")
        if not isinstance(properties, dict):
            raise ValueError(f"Feature {idx} must contain a 'properties' dictionary")

        damage_class = properties.get("damage_class")
        if damage_class not in ALLOWED_DAMAGE_CLASSES:
            raise ValueError(
                f"Feature {idx} invalid damage_class '{damage_class}'. "
                f"Must be one of: {sorted(ALLOWED_DAMAGE_CLASSES)}"
            )

        confidence = properties.get("confidence")
        if confidence is None or not isinstance(confidence, (int, float)):
            raise ValueError(f"Feature {idx} missing or invalid numeric 'confidence'")
        if not (0.0 <= confidence <= 1.0):
            raise ValueError(f"Feature {idx} confidence {confidence} out of range [0.0, 1.0]")


def is_valid_feature_collection(data: Any) -> bool:
    """Check whether data is a valid GeoJSON FeatureCollection without raising exceptions."""
    try:
        validate_feature_collection(data)
        return True
    except Exception as exc:
        logger.debug(f"GeoJSON validation failed: {exc}")
        return False


def feature_collection_to_json(data: Dict[str, Any], indent: int = 2) -> str:
    """Serialize a validated FeatureCollection to a formatted JSON string."""
    validate_feature_collection(data)
    return json.dumps(data, indent=indent, ensure_ascii=False)


def save_geojson(data: Dict[str, Any], output_path: str | Path) -> Path:
    """Save a validated FeatureCollection to disk as readable GeoJSON."""
    validate_feature_collection(data)
    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    json_str = json.dumps(data, indent=2, ensure_ascii=False)
    target_path.write_text(json_str, encoding="utf-8")
    logger.info(f"GeoJSON saved successfully: {target_path} ({len(data.get('features', []))} features)")
    return target_path


def load_geojson(path: str | Path) -> Dict[str, Any]:
    """Load and validate GeoJSON FeatureCollection from a file path."""
    target_path = Path(path)
    if not target_path.exists():
        err_msg = f"GeoJSON file not found: {target_path}"
        logger.error(err_msg)
        raise FileNotFoundError(err_msg)

    content = target_path.read_text(encoding="utf-8")
    data = json.loads(content)
    validate_feature_collection(data)
    return data


def get_bounds(data: Dict[str, Any]) -> List[List[float]]:
    """
    Extract spatial bounding box formatted for Folium map.fit_bounds:
    [[min_lat, min_lon], [max_lat, max_lon]]
    """
    validate_feature_collection(data)
    features = data.get("features", [])
    if not features:
        return [[0.0, 0.0], [0.0, 0.0]]

    lons: List[float] = []
    lats: List[float] = []

    for feature in features:
        for ring in feature["geometry"]["coordinates"]:
            for pt in ring:
                lons.append(pt[0])
                lats.append(pt[1])

    if not lons or not lats:
        return [[0.0, 0.0], [0.0, 0.0]]

    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)

    return [[min_lat, min_lon], [max_lat, max_lon]]
