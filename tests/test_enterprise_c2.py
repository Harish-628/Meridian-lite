"""Comprehensive Enterprise C2 Situational Awareness & Triage Verification Suite.

Verifies:
1. Real ONNX inference pipeline generates valid RFC 7946 GeoJSON.
2. Storm surge slider recalculates priority zone scores deterministically.
3. Dynamic routing severing logic successfully diverts around blocked zones.
4. Streamlit C2 components import without circular dependencies or syntax errors.
"""

import math
from pathlib import Path
import pytest

from models.real_predict import run_real_inference
from utils.geojson_utils import is_valid_feature_collection
from utils.prediction import (
    calculate_inundation_probability,
    generate_forecast_inundation_polygons,
    get_elevation_and_hand,
    recalculate_priority_zones_with_inundation,
)
from utils.priority import compute_priority_zones
from utils.routing import (
    apply_dynamic_hazard_impedance,
    build_mexico_beach_road_network,
    compute_evacuation_routing,
    compute_usar_ambulance_dispatch_matrix,
)
from utils.paths import PROJECT_ROOT

PRE_TILE = PROJECT_ROOT / "data" / "processed" / "hurricane_michael_pre.png"
POST_TILE = PROJECT_ROOT / "data" / "processed" / "hurricane_michael_post.png"


def test_real_onnx_inference_generates_rfc7946_geojson():
    """1. Verify real ONNX neural model produces valid RFC 7946 GeoJSON on real disaster tile."""
    assert PRE_TILE.exists(), f"Missing pre-disaster imagery at {PRE_TILE}"
    assert POST_TILE.exists(), f"Missing post-disaster imagery at {POST_TILE}"

    geojson_result = run_real_inference(
        pre_image_path=PRE_TILE,
        post_image_path=POST_TILE,
        backend_name="real_damage_model",
        confidence_threshold=0.30,
        max_features=250,
    )

    assert geojson_result.get("type") == "FeatureCollection"
    assert is_valid_feature_collection(geojson_result), "GeoJSON failed RFC 7946 specification validation"
    features = geojson_result.get("features", [])
    assert len(features) > 0, "Expected non-zero detected building footprints"

    # Verify building feature structure
    sample = features[0]
    assert sample["type"] == "Feature"
    assert sample["geometry"]["type"] in ("Polygon", "MultiPolygon")
    props = sample["properties"]
    assert "damage_class" in props
    assert props["damage_class"] in ("no-damage", "minor-damage", "major-damage", "destroyed")
    assert 0.0 <= props["confidence"] <= 1.0
    assert props["backend_name"] == "real_damage_model"


def test_storm_surge_recalculates_priority_zones_deterministically():
    """2. Verify storm surge slider dynamically recalculates priority zone scores and forecast hazard polygons."""
    geojson_result = run_real_inference(
        pre_image_path=PRE_TILE,
        post_image_path=POST_TILE,
        backend_name="real_damage_model",
        confidence_threshold=0.30,
        max_features=150,
    )
    base_zones = compute_priority_zones(geojson_result, grid_rows=3, grid_cols=3)
    assert len(base_zones) == 9

    # Recalculate at low surge (0.5m) vs severe surge (4.5m)
    zones_low = recalculate_priority_zones_with_inundation(base_zones, surge_height_m=0.5, rain_mm=20.0)
    zones_high = recalculate_priority_zones_with_inundation(base_zones, surge_height_m=4.5, rain_mm=120.0)

    assert len(zones_low) == 9
    assert len(zones_high) == 9

    # High surge must yield higher or equal flood risk and priority scores across coastal sectors
    mean_score_low = sum(z["priority_score"] for z in zones_low) / 9.0
    mean_score_high = sum(z["priority_score"] for z in zones_high) / 9.0
    assert mean_score_high > mean_score_low, "Severe surge must elevate mean composite priority score"

    # Verify mathematical formula on top zone: 0.35*Damage_Density + 0.25*Severe_Ratio + 0.25*Flood_Risk + 0.15*Isolation_Factor
    top_z = zones_high[0]
    expected_formula_score = (
        0.35 * (top_z["damaged_buildings"] / max(top_z["total_buildings"], 1) if top_z["total_buildings"] > 0 else 0.0)
        + 0.25 * ((top_z["destroyed_buildings"] + top_z["major_damage_buildings"]) / max(top_z["damaged_buildings"], 1) if top_z["damaged_buildings"] > 0 else 0.0)
        + 0.25 * top_z["flood_risk"]
        + 0.15 * top_z["isolation_factor"]
    )
    assert pytest.approx(top_z["priority_score"], abs=1e-3) == round(expected_formula_score, 4)

    # Verify forecast multi-horizon polygons
    forecast_geojson = generate_forecast_inundation_polygons(surge_height_m=3.5, rain_mm=80.0)
    assert forecast_geojson["type"] == "FeatureCollection"
    assert len(forecast_geojson["features"]) > 0
    horizons = [f["properties"]["horizon"] for f in forecast_geojson["features"]]
    assert "2h" in horizons
    assert "4h" in horizons
    assert "6h" in horizons


def test_dynamic_routing_severs_blocked_zones_and_diverts():
    """3. Verify dynamic spatial routing graph severs damaged/flooded roads and calculates safe detour corridors."""
    geojson_result = run_real_inference(
        pre_image_path=PRE_TILE,
        post_image_path=POST_TILE,
        backend_name="real_damage_model",
        confidence_threshold=0.30,
        max_features=150,
    )
    features = geojson_result.get("features", [])

    # Build road network
    G = build_mexico_beach_road_network(grid_cols=7, grid_rows=7)
    assert G.number_of_nodes() == 49
    assert G.number_of_edges() > 0

    # Flood overlay
    flood_geojson = generate_forecast_inundation_polygons(surge_height_m=3.8)

    # Apply dynamic impedance
    G_impeded, stats = apply_dynamic_hazard_impedance(
        G=G,
        building_features=features,
        flood_geojson=flood_geojson,
        rubble_buffer_m=12.0,
    )
    assert stats["severed_edges"] > 0, "Rubble and storm surge must sever at least one road edge"
    assert stats["passable_edges"] > 0

    # Verify impassable edges have math.inf
    impassable_edges = [d for _, _, d in G_impeded.edges(data=True) if not d["is_passable"]]
    assert len(impassable_edges) == stats["severed_edges"]
    assert all(math.isinf(d["impedance"]) for d in impassable_edges)

    # Compute baseline vs evasive routing
    staging_base = (-85.5540, 30.1715)  # West US-98
    target_zone = (-85.5470, 30.1780)   # Inland northeast sector
    routing_result = compute_evacuation_routing(
        G=G_impeded,
        start_coords=staging_base,
        target_coords=target_zone,
    )

    r_geojson = routing_result["geojson"]
    assert r_geojson["type"] == "FeatureCollection"
    assert len(r_geojson["features"]) == 2  # Baseline + Hazard-Evasive
    r_metrics = routing_result["metrics"]
    assert r_metrics["safe_route_distance_km"] > 0.0
    assert r_metrics["safe_route_duration_mins"] > 0.0
    assert r_metrics["clearance_status"] in ("CLEARED_HAZARD_DETOUR", "ARTERIAL_SEVERED_PARTIAL_CORRIDOR")

    # Verify greedy dispatch allocation matrix
    base_zones = compute_priority_zones(geojson_result, grid_rows=3, grid_cols=3)
    dispatch_plan = compute_usar_ambulance_dispatch_matrix(base_zones)
    assert len(dispatch_plan) == 3  # Top 3 critical zones
    assert all(d["usar_squads_dispatched"] >= 1 for d in dispatch_plan)
    assert all(d["paramedic_ambulances_dispatched"] >= 1 for d in dispatch_plan)
    assert all(d["estimated_transit_eta_mins"] > 0.0 for d in dispatch_plan)


def test_streamlit_c2_components_import_and_integrate():
    """4. Verify Streamlit components and C2 split terminal import cleanly with zero circular dependencies."""
    from app.ui import (
        build_folium_map,
        render_c2_split_terminal,
        render_folium_map,
        render_header_banners,
        render_sidebar,
        render_tactical_incident_action_plan,
        render_tactical_summary,
    )
    from streamlit.testing.v1 import AppTest

    APP_PATH = str(PROJECT_ROOT / "app" / "main.py")
    at = AppTest.from_file(APP_PATH, default_timeout=35)
    at.run()
    assert not at.exception, f"App raised uncaught exceptions on boot: {at.exception}"

    # Switch to Real Model Mode to inspect live C2 Split Terminal widgets
    at.sidebar.radio[0].set_value("Real Model Mode").run()
    assert not at.exception, f"C2 terminal switch raised exception: {at.exception}"

    # Verify two-column split layout components
    assert len(at.slider) > 0, "Expected dynamic storm surge slider in C2 layout"
    assert any("Storm Surge" in s.label for s in at.slider), "Storm surge slider label missing"

    # Verify button "Compute Safe Infiltration Routes" is rendered
    labels = [b.label for b in at.button]
    assert any("Compute Safe Infiltration Routes" in lbl for lbl in labels), (
        f"Missing 'Compute Safe Infiltration Routes' button. Found: {labels}"
    )
