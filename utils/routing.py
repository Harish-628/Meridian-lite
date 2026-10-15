"""Stage 2 & Stage 3: Dynamic Impedance Routing & Tactical USAR Allocation Engine for MERIDIAN-Lite.

Engineering Team:
- Member 1 (Harish Kumar S): Machine Learning & On-Device Neural Inference Lead
- Member 2: Geospatial Intelligence & GIS Architecture Lead
- Member 3: Defense Systems Architect & Operations Research Lead
- Member 4: Full-Stack C2 UI, Mission Assurance & QA Lead

Synthesizes a realistic, deterministic road network graph for Mexico Beach, FL using NetworkX,
intersects road edges with building rubble buffers and storm-surge flood hazard layers,
computes dynamic Dijkstra/A* baseline vs hazard-evasive routes, and produces an automated
greedy cost-minimization USAR and ambulance dispatch matrix.
"""

from dataclasses import dataclass, asdict
import math
from typing import Any, Tuple, Optional
import networkx as nx
import numpy as np
from shapely.geometry import Point, LineString, Polygon, MultiPolygon, box
from shapely.ops import unary_union

from utils.logging_utils import get_logger

logger = get_logger("routing")

# Tactical disaster survey bounding box for Mexico Beach coastal corridor
DEFAULT_COASTAL_BOUNDS = (-85.550100, 30.152997, -85.547337, 30.155760)


def haversine_distance_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Calculate great-circle distance between two GPS coordinates in meters."""
    r = 6371000.0  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return r * c


def build_mexico_beach_road_network(
    bounds: tuple[float, float, float, float] = DEFAULT_COASTAL_BOUNDS,
    grid_cols: int = 7,
    grid_rows: int = 7,
) -> nx.Graph:
    """Synthesize a deterministic road network graph for Mexico Beach coastal strip.

    Graph Topology:
    - Row 0 (South): US Highway 98 (Primary arterial corridor, 65 km/h).
    - Row grid_rows-1 (North): Inland Ridge Road / Highway 386 connector (Secondary, 50 km/h).
    - Columns 0, 2, 4, 6: Key North-South Access Avenues (7th, 15th, 22nd, 30th St, 45 km/h).
    - Grid interior: Residential access lanes and coastal cross-streets (25 km/h).
    """
    min_lon, min_lat, max_lon, max_lat = bounds
    lon_step = (max_lon - min_lon) / (grid_cols - 1)
    lat_step = (max_lat - min_lat) / (grid_rows - 1)

    G = nx.Graph()

    # 1. Add intersection nodes
    node_id = 0
    node_coords: dict[tuple[int, int], str] = {}
    for r in range(grid_rows):
        for c in range(grid_cols):
            lon = round(min_lon + c * lon_step, 6)
            lat = round(min_lat + r * lat_step, 6)
            n_name = f"N_{r}_{c}"
            G.add_node(
                n_name,
                lon=lon,
                lat=lat,
                pos=(lon, lat),
                row=r,
                col=c,
            )
            node_coords[(r, c)] = n_name

    # 2. Add edges with realistic classifications
    for r in range(grid_rows):
        for c in range(grid_cols):
            curr_node = node_coords[(r, c)]
            curr_pos = G.nodes[curr_node]["pos"]

            # Horizontal edge (East-West)
            if c + 1 < grid_cols:
                east_node = node_coords[(r, c + 1)]
                east_pos = G.nodes[east_node]["pos"]
                length_m = haversine_distance_m(curr_pos[0], curr_pos[1], east_pos[0], east_pos[1])

                if r == 0:
                    road_type = "primary"
                    speed_kmh = 65.0
                    name = "US Highway 98 (Coastal Arterial)"
                elif r == grid_rows - 1:
                    road_type = "secondary"
                    speed_kmh = 50.0
                    name = "Inland Ridge Bypass"
                else:
                    road_type = "residential"
                    speed_kmh = 25.0
                    name = f"Access Lane R-{r}"

                base_time_s = length_m / ((speed_kmh * 1000.0) / 3600.0)
                geom = LineString([curr_pos, east_pos])

                G.add_edge(
                    curr_node,
                    east_node,
                    length_m=round(length_m, 2),
                    speed_kmh=speed_kmh,
                    base_weight=round(base_time_s, 2),
                    impedance=round(base_time_s, 2),
                    road_type=road_type,
                    name=name,
                    geometry=geom,
                    is_passable=True,
                    blockade_reason=None,
                )

            # Vertical edge (North-South)
            if r + 1 < grid_rows:
                north_node = node_coords[(r + 1, c)]
                north_pos = G.nodes[north_node]["pos"]
                length_m = haversine_distance_m(curr_pos[0], curr_pos[1], north_pos[0], north_pos[1])

                if c in (0, 2, 4, 6):
                    road_type = "secondary"
                    speed_kmh = 45.0
                    name = f"Arterial Avenue C-{c}"
                else:
                    road_type = "residential"
                    speed_kmh = 25.0
                    name = f"Local Cross-Street C-{c}"

                base_time_s = length_m / ((speed_kmh * 1000.0) / 3600.0)
                geom = LineString([curr_pos, north_pos])

                G.add_edge(
                    curr_node,
                    north_node,
                    length_m=round(length_m, 2),
                    speed_kmh=speed_kmh,
                    base_weight=round(base_time_s, 2),
                    impedance=round(base_time_s, 2),
                    road_type=road_type,
                    name=name,
                    geometry=geom,
                    is_passable=True,
                    blockade_reason=None,
                )

    return G


def find_nearest_node(G: nx.Graph, lon: float, lat: float) -> str:
    """Find the nearest graph intersection node to a target GPS coordinate."""
    best_node = None
    min_dist = float("inf")
    for n, data in G.nodes(data=True):
        d = haversine_distance_m(lon, lat, data["lon"], data["lat"])
        if d < min_dist:
            min_dist = d
            best_node = n
    return best_node or list(G.nodes)[0]


def apply_dynamic_hazard_impedance(
    G: nx.Graph,
    building_features: list[dict[str, Any]],
    flood_geojson: Optional[dict[str, Any]] = None,
    rubble_buffer_m: float = 15.0,
) -> Tuple[nx.Graph, dict[str, Any]]:
    """Intersect road edges with severe building damage buffers and flood layers.

    Sets impassable edges to math.inf impedance.
    """
    hazard_polygons = []

    # 1. Rubble buffers for destroyed and major-damaged structures
    # 1 deg lat is ~111,000 meters
    deg_per_meter = 1.0 / 111000.0
    buffer_deg = rubble_buffer_m * deg_per_meter

    for feat in building_features:
        props = feat.get("properties", {})
        dmg = str(props.get("damage_class", props.get("damage", ""))).lower()
        if dmg in ("destroyed", "major-damage", "major"):
            geom_data = feat.get("geometry", {})
            coords = geom_data.get("coordinates", [])
            if geom_data.get("type") == "Polygon" and coords and coords[0]:
                poly = Polygon(coords[0])
                if poly.is_valid:
                    hazard_polygons.append(poly.buffer(buffer_deg))

    # 2. Flood hazard polygons (current horizon or surge inundation)
    if flood_geojson:
        for feat in flood_geojson.get("features", []):
            geom_data = feat.get("geometry", {})
            coords = geom_data.get("coordinates", [])
            if geom_data.get("type") == "Polygon" and coords and coords[0]:
                poly = Polygon(coords[0])
                if poly.is_valid:
                    hazard_polygons.append(poly)
            elif geom_data.get("type") == "MultiPolygon" and coords:
                for p_coords in coords:
                    if p_coords and p_coords[0]:
                        poly = Polygon(p_coords[0])
                        if poly.is_valid:
                            hazard_polygons.append(poly)

    hazard_union = unary_union(hazard_polygons) if hazard_polygons else None

    # 3. Intersect every road edge
    total_edges = G.number_of_edges()
    severed_edges = 0

    for u, v, data in G.edges(data=True):
        edge_geom: LineString = data["geometry"]
        if hazard_union and not hazard_union.is_empty and edge_geom.intersects(hazard_union):
            data["impedance"] = math.inf
            data["is_passable"] = False
            data["blockade_reason"] = "Rubble collapse or storm surge inundation"
            severed_edges += 1
        else:
            data["impedance"] = data["base_weight"]
            data["is_passable"] = True
            data["blockade_reason"] = None

    stats = {
        "total_edges": total_edges,
        "severed_edges": severed_edges,
        "passable_edges": total_edges - severed_edges,
        "severed_percentage": round((severed_edges / max(total_edges, 1)) * 100.0, 1),
    }

    return G, stats


def compute_evacuation_routing(
    G: nx.Graph,
    start_coords: tuple[float, float],  # (lon, lat)
    target_coords: tuple[float, float], # (lon, lat)
    staging_name: str = "Staging Base Alpha (West)",
    target_name: str = "Priority Sector 1 Target",
) -> dict[str, Any]:
    """Compute baseline vs. dynamic hazard-evasive vehicle routing.

    Returns GeoJSON polylines, segment clearance status, duration (mins), and detour metrics.
    """
    start_node = find_nearest_node(G, start_coords[0], start_coords[1])
    target_node = find_nearest_node(G, target_coords[0], target_coords[1])

    # 1. Baseline shortest path (ignoring hazards)
    try:
        baseline_path = nx.dijkstra_path(G, start_node, target_node, weight="base_weight")
        baseline_dist_m = 0.0
        baseline_time_s = 0.0
        baseline_coords = []
        for i in range(len(baseline_path) - 1):
            u, v = baseline_path[i], baseline_path[i + 1]
            edata = G[u][v]
            baseline_dist_m += edata["length_m"]
            baseline_time_s += edata["base_weight"]
            baseline_coords.append(G.nodes[u]["pos"])
        baseline_coords.append(G.nodes[baseline_path[-1]]["pos"])
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        baseline_path = []
        baseline_coords = [start_coords, target_coords]
        baseline_dist_m = haversine_distance_m(start_coords[0], start_coords[1], target_coords[0], target_coords[1])
        baseline_time_s = baseline_dist_m / 10.0

    # 2. Hazard-evasive path (strict avoidance of impedance == inf)
    H = G.copy()
    for u, v, d in list(H.edges(data=True)):
        if not d.get("is_passable", True) or not math.isfinite(d.get("impedance", 0)):
            H.remove_edge(u, v)

    evasive_coords = []
    evasive_dist_m = 0.0
    evasive_time_s = 0.0
    clearance_status = "CLEARED_HAZARD_DETOUR"

    connected_nodes = [n for n in H.nodes() if H.degree(n) > 0]
    if not connected_nodes:
        evasive_coords = baseline_coords
        evasive_dist_m = baseline_dist_m * 1.5
        evasive_time_s = baseline_time_s * 2.0
        clearance_status = "ALL_CORRIDORS_SEVERED"
    else:
        effective_start = start_node
        if H.degree(start_node) == 0:
            effective_start = min(
                connected_nodes,
                key=lambda n: haversine_distance_m(H.nodes[n]["pos"][0], H.nodes[n]["pos"][1], start_coords[0], start_coords[1])
            )
        effective_target = target_node
        if H.degree(target_node) == 0:
            effective_target = min(
                connected_nodes,
                key=lambda n: haversine_distance_m(H.nodes[n]["pos"][0], H.nodes[n]["pos"][1], target_coords[0], target_coords[1])
            )

        try:
            if nx.has_path(H, effective_start, effective_target):
                evasive_path = nx.dijkstra_path(H, effective_start, effective_target, weight="impedance")
                for i in range(len(evasive_path) - 1):
                    u, v = evasive_path[i], evasive_path[i + 1]
                    edata = H[u][v]
                    evasive_dist_m += edata["length_m"]
                    evasive_time_s += edata["impedance"]
                    evasive_coords.append(H.nodes[u]["pos"])
                evasive_coords.append(H.nodes[evasive_path[-1]]["pos"])
            else:
                clearance_status = "ARTERIAL_SEVERED_PARTIAL_CORRIDOR"
                reachable = nx.descendants(H, effective_start)
                reachable.add(effective_start)
                best_reach = min(
                    reachable,
                    key=lambda n: haversine_distance_m(H.nodes[n]["pos"][0], H.nodes[n]["pos"][1], target_coords[0], target_coords[1])
                )
                evasive_path = nx.dijkstra_path(H, effective_start, best_reach, weight="impedance")
                for i in range(len(evasive_path) - 1):
                    u, v = evasive_path[i], evasive_path[i + 1]
                    edata = H[u][v]
                    evasive_dist_m += edata["length_m"]
                    evasive_time_s += edata["impedance"]
                    evasive_coords.append(H.nodes[u]["pos"])
                evasive_coords.append(H.nodes[evasive_path[-1]]["pos"])
        except Exception:
            evasive_coords = baseline_coords
            evasive_dist_m = baseline_dist_m * 1.5
            evasive_time_s = baseline_time_s * 2.0
            clearance_status = "ARTERIAL_SEVERED_PARTIAL_CORRIDOR"

    # Metrics
    base_mins = round(baseline_time_s / 60.0, 1)
    evasive_mins = round(evasive_time_s / 60.0, 1)
    detour_mins = max(0.0, round(evasive_mins - base_mins, 1))
    detour_km = max(0.0, round((evasive_dist_m - baseline_dist_m) / 1000.0, 2))

    # FeatureCollection representation
    features = [
        {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": baseline_coords,
            },
            "properties": {
                "route_type": "baseline_shortest_direct",
                "label": "Direct Unconstrained Baseline (Passes through Hazard)",
                "distance_km": round(baseline_dist_m / 1000.0, 2),
                "duration_mins": base_mins,
                "color": "#00f0ff",
                "stroke_style": "dash",
            },
        },
        {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": evasive_coords,
            },
            "properties": {
                "route_type": "dynamic_hazard_evasive",
                "label": "Safe Infiltration Corridor (Dynamic Debris & Flood Avoidance)",
                "distance_km": round(evasive_dist_m / 1000.0, 2),
                "duration_mins": evasive_mins,
                "detour_penalty_mins": detour_mins,
                "detour_penalty_km": detour_km,
                "clearance_status": clearance_status,
                "color": "#00ff88" if clearance_status == "CLEARED_HAZARD_DETOUR" else "#ffb703",
                "stroke_style": "solid",
            },
        },
    ]

    return {
        "geojson": {
            "type": "FeatureCollection",
            "features": features,
        },
        "metrics": {
            "staging_origin": staging_name,
            "target_destination": target_name,
            "baseline_distance_km": round(baseline_dist_m / 1000.0, 2),
            "baseline_duration_mins": base_mins,
            "safe_route_distance_km": round(evasive_dist_m / 1000.0, 2),
            "safe_route_duration_mins": evasive_mins,
            "detour_delay_mins": detour_mins,
            "detour_delay_km": detour_km,
            "clearance_status": clearance_status,
        },
    }


def compute_usar_ambulance_dispatch_matrix(
    priority_zones: list[dict[str, Any]],
    staging_bases: Optional[list[dict[str, Any]]] = None,
    total_usar_squads: int = 6,
    total_ambulances: int = 10,
    bounds: Optional[tuple[float, float, float, float]] = None,
) -> list[dict[str, Any]]:
    """Greedy cost-minimization allocator mapping nearest available staging units to critical priority zones.

    Allocates units to top 3 critical priority zones based on distance and severity.
    """
    survey_b = bounds or (priority_zones[0].get("bounds", DEFAULT_COASTAL_BOUNDS) if priority_zones else DEFAULT_COASTAL_BOUNDS)

    if staging_bases is None:
        # Position staging bases realistically at the tactical survey perimeter (60-80m offset)
        alpha_lon = survey_b[0] - 0.0006
        alpha_lat = survey_b[1] + (survey_b[3] - survey_b[1]) * 0.15
        bravo_lon = survey_b[2] + 0.0005
        bravo_lat = survey_b[3] + 0.0003

        staging_bases = [
            {"base_id": "BASE-ALPHA", "name": "Forward Staging Base Alpha (West Ingress)", "coords": (alpha_lon, alpha_lat), "usar_pool": 8, "amb_pool": 14},
            {"base_id": "BASE-BRAVO", "name": "Tactical Staging Base Bravo (North Ridge)", "coords": (bravo_lon, bravo_lat), "usar_pool": 6, "amb_pool": 10},
        ]

    critical_zones = sorted(priority_zones, key=lambda z: z.get("priority_rank", 999))[:3]
    dispatch_plan = []

    for zone in critical_zones:
        z_id = zone.get("zone_id", "Z-001")
        rank = zone.get("priority_rank", 1)
        z_bounds = zone.get("bounds", DEFAULT_COASTAL_BOUNDS)
        z_c_lon = (z_bounds[0] + z_bounds[2]) / 2.0
        z_c_lat = (z_bounds[1] + z_bounds[3]) / 2.0

        dest_cnt = zone.get("destroyed_buildings", 0)
        major_cnt = zone.get("major_damage_buildings", 0)

        # Greedy match to nearest staging base with available capacity, or nearest overall
        available_bases = [b for b in staging_bases if b["usar_pool"] > 0 or b["amb_pool"] > 0]
        candidate_bases = available_bases if available_bases else staging_bases
        best_base = min(
            candidate_bases,
            key=lambda b: haversine_distance_m(b["coords"][0], b["coords"][1], z_c_lon, z_c_lat)
        )

        dist_km = round(haversine_distance_m(best_base["coords"][0], best_base["coords"][1], z_c_lon, z_c_lat) / 1000.0, 2)
        eta_mins = round(dist_km * 2.2 + 3.0, 1)

        # Allocate resources (guarantee at least 1 unit to every critical sector)
        req_usar = max(1, math.ceil(dest_cnt / 2.0) if dest_cnt > 0 else 1)
        allocated_usar = max(1, min(best_base["usar_pool"], req_usar))
        best_base["usar_pool"] = max(0, best_base["usar_pool"] - allocated_usar)

        req_amb = max(1, major_cnt + dest_cnt)
        allocated_amb = max(1, min(best_base["amb_pool"], req_amb))
        best_base["amb_pool"] = max(0, best_base["amb_pool"] - allocated_amb)

        dispatch_plan.append({
            "target_zone_id": z_id,
            "priority_rank": rank,
            "assigned_staging_base": best_base["name"],
            "transit_distance_km": dist_km,
            "estimated_transit_eta_mins": eta_mins,
            "usar_squads_dispatched": allocated_usar,
            "paramedic_ambulances_dispatched": allocated_amb,
            "mission_directive": (
                f"Deploy {allocated_usar} USAR squad(s) and {allocated_amb} ambulance(s) from {best_base['name']} "
                f"via cleared corridor to Sector {z_id} (ETA: {eta_mins} mins)."
            ),
        })

    return dispatch_plan
