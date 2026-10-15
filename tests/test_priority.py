"""Unit tests for Stage 2 priority zone scoring and ranking."""

import pytest
from utils.priority import compute_priority_zones, priority_zones_to_geojson


def _make_mock_feature(lon: float, lat: float, damage: str, conf: float) -> dict:
    """Helper to create a small building polygon feature."""
    d = 0.0005
    return {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [lon - d, lat - d],
                    [lon + d, lat - d],
                    [lon + d, lat + d],
                    [lon - d, lat + d],
                    [lon - d, lat - d],
                ]
            ],
        },
        "properties": {
            "feature_id": "TEST-F",
            "damage_class": damage,
            "confidence": conf,
        },
    }


def test_empty_feature_collection():
    """Verify empty feature collection returns safe default zero-score zones."""
    empty_fc = {"type": "FeatureCollection", "features": []}
    zones = compute_priority_zones(empty_fc, grid_rows=2, grid_cols=2)

    assert len(zones) == 4
    for z in zones:
        assert z["total_buildings"] == 0
        assert z["damaged_buildings"] == 0
        assert z["priority_score"] == 0.0
        assert "No detected structures" in z["reason"]


def test_priority_formula_weighting():
    """Verify that zones with destroyed structures achieve higher priority scores."""
    # Place structures in distinct sectors:
    # Sector A (North-East, high destroyed damage)
    feats_a = [
        _make_mock_feature(-85.546, 30.179, "destroyed", 0.95),
        _make_mock_feature(-85.547, 30.178, "destroyed", 0.90),
        _make_mock_feature(-85.546, 30.178, "major-damage", 0.92),
    ]
    # Sector B (South-West, undamaged)
    feats_b = [
        _make_mock_feature(-85.554, 30.171, "no-damage", 0.85),
        _make_mock_feature(-85.553, 30.172, "no-damage", 0.88),
    ]

    fc = {"type": "FeatureCollection", "features": feats_a + feats_b}
    zones = compute_priority_zones(fc, grid_rows=2, grid_cols=2)

    assert len(zones) == 4
    top_zone = zones[0]
    assert top_zone["priority_rank"] == 1
    assert top_zone["destroyed_buildings"] == 2
    assert top_zone["major_damage_buildings"] == 1
    assert top_zone["priority_score"] > 0.70
    assert "Critical severe structural destruction" in top_zone["reason"]

    # Lowest rank should be the undamaged or empty zones
    assert zones[-1]["priority_score"] <= top_zone["priority_score"]


def test_priority_ranking_sorting():
    """Verify zones are strictly sorted descending by priority_score."""
    feats = [
        _make_mock_feature(-85.550, 30.175, "minor-damage", 0.75),
        _make_mock_feature(-85.546, 30.179, "destroyed", 0.95),
        _make_mock_feature(-85.554, 30.171, "no-damage", 0.80),
    ]
    fc = {"type": "FeatureCollection", "features": feats}
    zones = compute_priority_zones(fc, grid_rows=3, grid_cols=3)

    assert len(zones) == 9
    for i in range(len(zones) - 1):
        assert zones[i]["priority_score"] >= zones[i + 1]["priority_score"]
        assert zones[i]["priority_rank"] == i + 1


def test_priority_zones_to_geojson():
    """Verify conversion of priority zone list to RFC 7946 GeoJSON FeatureCollection."""
    feats = [_make_mock_feature(-85.550, 30.175, "destroyed", 0.90)]
    fc = {"type": "FeatureCollection", "features": feats}
    zones = compute_priority_zones(fc, grid_rows=2, grid_cols=2)

    pz_geojson = priority_zones_to_geojson(zones)
    assert pz_geojson["type"] == "FeatureCollection"
    assert len(pz_geojson["features"]) == 4

    f0 = pz_geojson["features"][0]
    assert f0["type"] == "Feature"
    assert f0["geometry"]["type"] == "Polygon"
    assert "zone_id" in f0["properties"]
    assert "priority_rank" in f0["properties"]
    assert "priority_score" in f0["properties"]


def test_priority_multi_criteria_and_resource_allocation():
    """Verify that compute_priority_zones calculates multi-source parameters (Ez, Iz, Dz, Sz)
    and greedy USAR/ambulance/debris resource allocations.
    """
    feats = [
        _make_mock_feature(-85.546, 30.179, "destroyed", 0.95),
        _make_mock_feature(-85.547, 30.178, "destroyed", 0.90),
        _make_mock_feature(-85.546, 30.178, "major-damage", 0.92),
        _make_mock_feature(-85.554, 30.171, "minor-damage", 0.85),
    ]
    fc = {"type": "FeatureCollection", "features": feats}
    zones = compute_priority_zones(fc, grid_rows=2, grid_cols=2)

    top_z = zones[0]
    assert "damage_density" in top_z
    assert "severe_damage_ratio" in top_z
    assert "exposed_population_factor" in top_z
    assert "isolation_factor" in top_z
    assert "allocated_usar_squads" in top_z
    assert "allocated_ambulances" in top_z
    assert "allocated_debris_units" in top_z

    assert top_z["allocated_usar_squads"] >= 1
    assert top_z["allocated_ambulances"] >= 1
    assert top_z["allocated_debris_units"] >= 1
