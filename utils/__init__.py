"""
MERIDIAN-Lite Utility Package.
Provides helpers for paths, logging, image processing, GeoJSON parsing, georeferencing,
and metrics / priority-zone calculations.
"""

from utils.geojson_utils import (
    ALLOWED_DAMAGE_CLASSES,
    DAMAGE_COLORS,
    DAMAGE_HEX_COLORS,
    damage_color,
    get_bounds,
    is_valid_feature_collection,
    load_geojson,
    save_geojson,
    validate_feature_collection,
)
from utils.georeferencing import AffineGeoreferencer
from utils.image_utils import (
    create_damage_overlay,
    create_demo_images,
    create_tactical_detection_overlay,
    image_dimensions,
    load_image,
    validate_image,
)
get_image_dimensions = image_dimensions
from utils.logging_utils import get_logger, setup_logging
from utils.metrics import calculate_summary_metrics, features_to_dataframe
from utils.paths import (
    DEFAULT_GEOJSON_OUTPUT,
    DEFAULT_METRICS_CSV_OUTPUT,
    DEFAULT_METRICS_JSON_OUTPUT,
    DEFAULT_OVERLAY_OUTPUT,
    DEMO_BUILDINGS_GEOJSON,
    DEMO_DATA_DIR,
    DEMO_POST_IMAGE,
    DEMO_PRE_IMAGE,
    PROJECT_ROOT,
    ensure_directories,
)
from utils.priority import compute_priority_zones, priority_zones_to_geojson

__all__ = [
    "ALLOWED_DAMAGE_CLASSES",
    "DAMAGE_COLORS",
    "DAMAGE_HEX_COLORS",
    "damage_color",
    "get_bounds",
    "is_valid_feature_collection",
    "load_geojson",
    "save_geojson",
    "validate_feature_collection",
    "AffineGeoreferencer",
    "create_damage_overlay",
    "create_demo_images",
    "create_tactical_detection_overlay",
    "image_dimensions",
    "get_image_dimensions",
    "load_image",
    "validate_image",
    "get_logger",
    "setup_logging",
    "calculate_summary_metrics",
    "features_to_dataframe",
    "compute_priority_zones",
    "priority_zones_to_geojson",
    "PROJECT_ROOT",
    "DEMO_DATA_DIR",
    "DEMO_PRE_IMAGE",
    "DEMO_POST_IMAGE",
    "DEMO_BUILDINGS_GEOJSON",
    "DEFAULT_GEOJSON_OUTPUT",
    "DEFAULT_METRICS_JSON_OUTPUT",
    "DEFAULT_METRICS_CSV_OUTPUT",
    "DEFAULT_OVERLAY_OUTPUT",
    "ensure_directories",
]
