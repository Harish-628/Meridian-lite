"""
Unit tests for GeoJSON utilities (utils/geojson_utils.py).
"""

from pathlib import Path
import pytest
from utils.geojson_utils import (
    damage_color,
    feature_collection_to_json,
    get_bounds,
    is_valid_feature_collection,
    load_geojson,
    save_geojson,
    validate_feature_collection,
)


@pytest.fixture
def valid_geojson_fixture():
    """Returns a valid minimal GeoJSON FeatureCollection."""
    return {
        "type": "FeatureCollection",
        "metadata": {"system": "MERIDIAN-Lite"},
        "features": [
            {
                "type": "Feature",
                "id": "b-01",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [80.270, 13.080],
                            [80.271, 13.080],
                            [80.271, 13.081],
                            [80.270, 13.081],
                            [80.270, 13.080],
                        ]
                    ],
                },
                "properties": {
                    "damage_class": "destroyed",
                    "confidence": 0.95,
                },
            }
        ],
    }


def test_valid_feature_collection(valid_geojson_fixture):
    """Verify valid GeoJSON passes validation without error."""
    assert is_valid_feature_collection(valid_geojson_fixture) is True
    validate_feature_collection(valid_geojson_fixture)


def test_reject_unclosed_polygon(valid_geojson_fixture):
    """Verify polygon with unclosed coordinates is rejected."""
    # Alter the last point so it doesn't match the first point
    valid_geojson_fixture["features"][0]["geometry"]["coordinates"][0][-1] = [80.275, 13.085]
    assert is_valid_feature_collection(valid_geojson_fixture) is False
    with pytest.raises(ValueError, match="not closed"):
        validate_feature_collection(valid_geojson_fixture)


def test_reject_invalid_coordinate_ranges(valid_geojson_fixture):
    """Verify coordinates outside [-180, 180] or [-90, 90] are rejected."""
    # Set latitude to 120.0
    valid_geojson_fixture["features"][0]["geometry"]["coordinates"][0][0] = [80.270, 120.0]
    valid_geojson_fixture["features"][0]["geometry"]["coordinates"][0][-1] = [80.270, 120.0]
    assert is_valid_feature_collection(valid_geojson_fixture) is False
    with pytest.raises(ValueError, match="latitude"):
        validate_feature_collection(valid_geojson_fixture)


def test_reject_invalid_damage_class(valid_geojson_fixture):
    """Verify arbitrary damage classes are rejected."""
    valid_geojson_fixture["features"][0]["properties"]["damage_class"] = "super-damaged"
    assert is_valid_feature_collection(valid_geojson_fixture) is False
    with pytest.raises(ValueError, match="invalid damage_class"):
        validate_feature_collection(valid_geojson_fixture)


def test_reject_invalid_confidence(valid_geojson_fixture):
    """Verify confidence values > 1.0 or < 0.0 are rejected."""
    valid_geojson_fixture["features"][0]["properties"]["confidence"] = 1.45
    assert is_valid_feature_collection(valid_geojson_fixture) is False
    with pytest.raises(ValueError, match="confidence"):
        validate_feature_collection(valid_geojson_fixture)


def test_save_and_load_geojson(tmp_path: Path, valid_geojson_fixture):
    """Verify save_geojson and load_geojson roundtrip without alteration."""
    out_file = tmp_path / "test.geojson"
    saved_path = save_geojson(valid_geojson_fixture, out_file)
    assert saved_path.exists()

    loaded = load_geojson(saved_path)
    assert loaded == valid_geojson_fixture


def test_get_bounds(valid_geojson_fixture):
    """Verify bounding box calculation returns [[min_lat, min_lon], [max_lat, max_lon]]."""
    bounds = get_bounds(valid_geojson_fixture)
    assert len(bounds) == 2
    min_lat, min_lon = bounds[0]
    max_lat, max_lon = bounds[1]
    assert min_lat == 13.080
    assert max_lat == 13.081
    assert min_lon == 80.270
    assert max_lon == 80.271


def test_damage_color_mapping():
    """Verify damage color mappings for all required categories."""
    assert damage_color("no-damage") == "green"
    assert damage_color("minor-damage") == "yellow"
    assert damage_color("major-damage") == "orange"
    assert damage_color("destroyed") == "red"

    assert damage_color("no-damage", as_hex=True) == "#28a745"
    assert damage_color("minor-damage", as_hex=True) == "#ffc107"
    assert damage_color("major-damage", as_hex=True) == "#fd7e14"
    assert damage_color("destroyed", as_hex=True) == "#dc3545"


def test_feature_collection_to_json(valid_geojson_fixture):
    """Verify serialization to formatted JSON string."""
    json_str = feature_collection_to_json(valid_geojson_fixture)
    assert isinstance(json_str, str)
    assert '"type": "FeatureCollection"' in json_str
