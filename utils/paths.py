"""
Path management for MERIDIAN-Lite.
Provides consistent cross-platform paths to core directories and files.
"""

from pathlib import Path

# Base project directory (resolved dynamically based on this file's location)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Application and model directories
APP_DIR = PROJECT_ROOT / "app"
MODELS_DIR = PROJECT_ROOT / "models"
UTILS_DIR = PROJECT_ROOT / "utils"
TESTS_DIR = PROJECT_ROOT / "tests"
DOCS_DIR = PROJECT_ROOT / "docs"

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DEMO_DATA_DIR = DATA_DIR / "demo"

# Demo assets
DEMO_PRE_IMAGE = DEMO_DATA_DIR / "pre_disaster.png"
DEMO_POST_IMAGE = DEMO_DATA_DIR / "post_disaster.png"
DEMO_BUILDINGS_GEOJSON = DEMO_DATA_DIR / "buildings.geojson"

# Output directories
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
GEOJSON_OUTPUT_DIR = OUTPUTS_DIR / "geojson"
OVERLAYS_OUTPUT_DIR = OUTPUTS_DIR / "overlays"
REPORTS_OUTPUT_DIR = OUTPUTS_DIR / "reports"

# Standard report and log files
DEFAULT_LOG_FILE = REPORTS_OUTPUT_DIR / "meridian_stage1.log"
DEFAULT_GEOJSON_OUTPUT = GEOJSON_OUTPUT_DIR / "damage_assessment.geojson"
DEFAULT_METRICS_JSON_OUTPUT = REPORTS_OUTPUT_DIR / "summary_metrics.json"
DEFAULT_METRICS_CSV_OUTPUT = REPORTS_OUTPUT_DIR / "building_assessment.csv"
DEFAULT_OVERLAY_OUTPUT = OVERLAYS_OUTPUT_DIR / "damage_overlay.png"


def ensure_directories() -> None:
    """Ensure all required project data, demo, and output directories exist."""
    directories = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        DEMO_DATA_DIR,
        GEOJSON_OUTPUT_DIR,
        OVERLAYS_OUTPUT_DIR,
        REPORTS_OUTPUT_DIR,
        DOCS_DIR,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


# Initialize directories on import
ensure_directories()
