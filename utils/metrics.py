"""
Metrics calculation for MERIDIAN-Lite damage assessment.
Computes aggregate counts, damage percentages, confidence statistics,
and formats data for dashboard display and reporting.
"""

from typing import Any, Dict, List
import pandas as pd
from utils.logging_utils import get_logger

logger = get_logger("metrics")


def calculate_summary_metrics(feature_collection: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate summary metrics from a GeoJSON FeatureCollection dictionary.
    Handles zero features safely without division-by-zero errors.
    """
    features: List[Dict[str, Any]] = feature_collection.get("features", [])
    metadata: Dict[str, Any] = feature_collection.get("metadata", {})

    total_buildings = len(features)

    no_damage_count = 0
    minor_damage_count = 0
    major_damage_count = 0
    destroyed_count = 0

    confidences: List[float] = []

    for feature in features:
        props = feature.get("properties", {})
        damage_class = props.get("damage_class", "")
        confidence = props.get("confidence", 0.0)

        if isinstance(confidence, (int, float)):
            confidences.append(float(confidence))

        if damage_class == "no-damage":
            no_damage_count += 1
        elif damage_class == "minor-damage":
            minor_damage_count += 1
        elif damage_class == "major-damage":
            major_damage_count += 1
        elif damage_class == "destroyed":
            destroyed_count += 1

    damaged_count = minor_damage_count + major_damage_count + destroyed_count

    if total_buildings > 0:
        damaged_percentage = round((damaged_count / total_buildings) * 100.0, 2)
        avg_confidence = round(sum(confidences) / len(confidences), 4) if confidences else 0.0
        min_confidence = round(min(confidences), 4) if confidences else 0.0
        max_confidence = round(max(confidences), 4) if confidences else 0.0
    else:
        damaged_percentage = 0.0
        avg_confidence = 0.0
        min_confidence = 0.0
        max_confidence = 0.0

    model_type = metadata.get("model_type", "simulated_xview2_baseline")
    is_simulated = metadata.get("is_simulated", True)

    metrics = {
        "total_buildings": total_buildings,
        "no_damage_count": no_damage_count,
        "minor_damage_count": minor_damage_count,
        "major_damage_count": major_damage_count,
        "destroyed_count": destroyed_count,
        "damaged_count": damaged_count,
        "damaged_percentage": damaged_percentage,
        "average_confidence": avg_confidence,
        "minimum_confidence": min_confidence,
        "maximum_confidence": max_confidence,
        "model_type": model_type,
        "is_simulated": is_simulated,
    }

    logger.debug(f"Computed summary metrics: {metrics}")
    return metrics


def features_to_dataframe(feature_collection: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert GeoJSON features into a tabular Pandas DataFrame for table display and CSV export.
    """
    records = []
    features = feature_collection.get("features", [])

    for feature in features:
        props = feature.get("properties", {})
        feat_id = feature.get("id") or props.get("feature_id", "")
        geometry = feature.get("geometry", {})
        coords = geometry.get("coordinates", [[]])

        # Calculate approximate center from polygon coordinates
        ring = coords[0] if coords else []
        if ring:
            lons = [p[0] for p in ring]
            lats = [p[1] for p in ring]
            center_lon = round(sum(lons) / len(lons), 6)
            center_lat = round(sum(lats) / len(lats), 6)
        else:
            center_lon, center_lat = 0.0, 0.0

        records.append({
            "feature_id": feat_id,
            "damage_class": props.get("damage_class", "unknown"),
            "confidence": props.get("confidence", 0.0),
            "source": props.get("source_model") or props.get("backend_name") or props.get("source", "simulated_xview2_baseline"),
            "latitude": center_lat,
            "longitude": center_lon,
            "is_simulated": props.get("is_simulated", False),
        })

    return pd.DataFrame(records)
