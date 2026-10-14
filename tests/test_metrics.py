"""
Unit tests for metrics calculation (utils/metrics.py).
"""

import pandas as pd
import pytest
from utils.metrics import calculate_summary_metrics, features_to_dataframe


@pytest.fixture
def sample_feature_collection():
    """Constructs a sample feature collection with known damage distributions."""
    return {
        "type": "FeatureCollection",
        "metadata": {
            "model_type": "simulated_xview2_baseline",
            "is_simulated": True,
        },
        "features": [
            {
                "type": "Feature",
                "id": "b1",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[80.0, 13.0], [80.1, 13.0], [80.1, 13.1], [80.0, 13.1], [80.0, 13.0]]],
                },
                "properties": {"damage_class": "no-damage", "confidence": 0.80},
            },
            {
                "type": "Feature",
                "id": "b2",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[80.0, 13.0], [80.1, 13.0], [80.1, 13.1], [80.0, 13.1], [80.0, 13.0]]],
                },
                "properties": {"damage_class": "minor-damage", "confidence": 0.85},
            },
            {
                "type": "Feature",
                "id": "b3",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[80.0, 13.0], [80.1, 13.0], [80.1, 13.1], [80.0, 13.1], [80.0, 13.0]]],
                },
                "properties": {"damage_class": "major-damage", "confidence": 0.90},
            },
            {
                "type": "Feature",
                "id": "b4",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[80.0, 13.0], [80.1, 13.0], [80.1, 13.1], [80.0, 13.1], [80.0, 13.0]]],
                },
                "properties": {"damage_class": "destroyed", "confidence": 0.95},
            },
        ],
    }


def test_calculate_summary_metrics(sample_feature_collection):
    """Verify metrics calculation correctly aggregates counts and percentages."""
    metrics = calculate_summary_metrics(sample_feature_collection)

    assert metrics["total_buildings"] == 4
    assert metrics["no_damage_count"] == 1
    assert metrics["minor_damage_count"] == 1
    assert metrics["major_damage_count"] == 1
    assert metrics["destroyed_count"] == 1

    # damaged_count = minor + major + destroyed = 1 + 1 + 1 = 3
    assert metrics["damaged_count"] == 3
    # damaged_percentage = 3 / 4 * 100 = 75.0%
    assert metrics["damaged_percentage"] == 75.0

    # confidence stats: [0.80, 0.85, 0.90, 0.95]
    assert metrics["minimum_confidence"] == 0.80
    assert metrics["maximum_confidence"] == 0.95
    assert metrics["average_confidence"] == pytest.approx(0.875, 0.001)

    assert metrics["model_type"] == "simulated_xview2_baseline"
    assert metrics["is_simulated"] is True


def test_zero_feature_metrics():
    """Verify zero features are handled gracefully without division-by-zero errors."""
    empty_collection = {
        "type": "FeatureCollection",
        "metadata": {"model_type": "simulated_xview2_baseline", "is_simulated": True},
        "features": [],
    }
    metrics = calculate_summary_metrics(empty_collection)

    assert metrics["total_buildings"] == 0
    assert metrics["damaged_count"] == 0
    assert metrics["damaged_percentage"] == 0.0
    assert metrics["average_confidence"] == 0.0
    assert metrics["minimum_confidence"] == 0.0
    assert metrics["maximum_confidence"] == 0.0


def test_features_to_dataframe(sample_feature_collection):
    """Verify conversion of features into a pandas DataFrame."""
    df = features_to_dataframe(sample_feature_collection)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4
    assert "feature_id" in df.columns
    assert "damage_class" in df.columns
    assert "confidence" in df.columns
    assert "latitude" in df.columns
    assert "longitude" in df.columns
