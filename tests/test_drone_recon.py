"""Unit tests for Stage 2 & Stage 3 Live Drone Reconnaissance, Flight Routing,
Resource Allocation, and Predictive Risk Modeling.
"""

import numpy as np
import pytest
from pathlib import Path

from utils.drone_stream import (
    MISSION_WAYPOINTS,
    draw_drone_hud,
    get_search_flight_path,
    process_drone_sector_stream,
)
from utils.predictive_risk import (
    calculate_predictive_hazard_analysis,
    METEOROLOGICAL_BASELINE,
)
from utils.resource_allocation import (
    calculate_tactical_resource_allocation,
    summarize_total_fleet_requirements,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_drone_waypoints_structure():
    """Verify drone mission waypoints and telemetry sensor data."""
    assert len(MISSION_WAYPOINTS) == 5
    for wp in MISSION_WAYPOINTS:
        assert wp.waypoint_id.startswith("WP-")
        assert 30.150 <= wp.lat <= 30.160
        assert -85.555 <= wp.lon <= -85.545
        assert 100.0 <= wp.altitude_m <= 150.0
        assert wp.speed_mps > 10.0
        assert 0 <= wp.battery_pct <= 100
        assert wp.wind_speed_kt > 20.0
        assert "Sony Exmor" in wp.sensor_model
        assert "FIXED" in wp.rtk_status


def test_search_flight_path_calculation():
    """Verify autonomous drone search route calculation (Stage 2)."""
    flight_plan = get_search_flight_path()
    assert flight_plan["waypoints_count"] == 5
    assert flight_plan["total_distance_meters"] > 200.0
    assert flight_plan["estimated_duration_seconds"] > 10.0
    assert flight_plan["survey_altitude_agl_m"] == 122.0
    assert len(flight_plan["coordinates"]) == 5


def test_draw_drone_hud():
    """Verify HUD drawing on synthetic image frame."""
    dummy_img = np.zeros((512, 512, 3), dtype=np.uint8)
    telem = MISSION_WAYPOINTS[0]
    dummy_detections = [
        {
            "feature_id": "BLD-001",
            "damage_class": "destroyed",
            "confidence": 0.92,
            "pixel_bbox": (100, 100, 200, 200),
        }
    ]

    hud_img = draw_drone_hud(
        dummy_img,
        telemetry=telem,
        detections=dummy_detections,
        active_sector_y_range=(50, 150),
        scan_line_y=100,
    )

    assert hud_img.shape == (512, 512, 3)
    # Ensure drawing modified pixels (not all black)
    assert np.count_nonzero(hud_img) > 1000


def test_process_drone_sector_stream():
    """Verify live drone stream sector processing on real disaster images."""
    pre_p = PROJECT_ROOT / "data" / "processed" / "hurricane_michael_pre.png"
    post_p = PROJECT_ROOT / "data" / "processed" / "hurricane_michael_post.png"

    if pre_p.exists() and post_p.exists():
        result = process_drone_sector_stream(
            sector_idx=2,
            pre_image_path=pre_p,
            post_image_path=post_p,
            confidence_threshold=0.35,
        )

        assert "hud_post_image" in result
        assert "hud_pre_image" in result
        assert "telemetry" in result
        assert "cumulative_detections" in result
        assert result["hud_post_image"].shape == (1024, 1024, 3)
        assert result["telemetry"].waypoint_id == "WP-03"


def test_tactical_resource_allocation():
    """Verify Stage 2 automated emergency resource dispatch calculation."""
    mock_zones = [
        {
            "zone_id": "Z-001",
            "priority_rank": 1,
            "priority_score": 0.85,
            "total_buildings": 8,
            "destroyed_buildings": 4,
            "major_damage_buildings": 2,
            "bounds": [-85.550, 30.154, -85.548, 30.156],
        },
        {
            "zone_id": "Z-002",
            "priority_rank": 2,
            "priority_score": 0.45,
            "total_buildings": 3,
            "destroyed_buildings": 0,
            "major_damage_buildings": 1,
            "bounds": [-85.548, 30.154, -85.546, 30.156],
        },
    ]

    deployments = calculate_tactical_resource_allocation(mock_zones)
    assert len(deployments) == 2
    d1 = deployments[0]
    assert d1["zone_id"] == "Z-001"
    assert d1["usar_teams_needed"] >= 2  # 4 destroyed -> 2 teams
    assert d1["paramedic_units_needed"] >= 2
    assert d1["debris_clearance_units"] == 1
    assert "CRITICAL" in d1["dispatch_urgency"]

    fleet = summarize_total_fleet_requirements(deployments)
    assert fleet["total_usar_teams"] >= 2
    assert fleet["total_paramedic_units"] >= 3


def test_predictive_hazard_analysis():
    """Verify Stage 3 multi-hazard predictive risk modeling."""
    mock_zones = [
        {
            "zone_id": "Z-001",
            "priority_rank": 1,
            "priority_score": 0.70,
            "total_buildings": 5,
            "destroyed_buildings": 2,
            "major_damage_buildings": 2,
            "minor_damage_buildings": 1,
        }
    ]

    risks = calculate_predictive_hazard_analysis(
        priority_zones=mock_zones,
        wind_gusts_kt=38.0,
        surge_height_m=2.5,
    )

    assert len(risks) == 1
    r = risks[0]
    assert r["zone_id"] == "Z-001"
    assert 0.0 <= r["compound_vulnerability_index"] <= 1.0
    assert r["secondary_collapse_risk"] > 0.3
    assert r["risk_tier"] in ["CRITICAL RISK", "ELEVATED RISK"]
    assert len(r["preventative_countermeasure"]) > 10
