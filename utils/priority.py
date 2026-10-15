"""Priority-Zone Scoring & Tactical Resource Allocation Engine for MERIDIAN-Lite.

Engineering Team:
- Member 1 (Harish Kumar S): Machine Learning & On-Device Neural Inference Lead
- Member 2: Geospatial Intelligence & GIS Architecture Lead
- Member 3: Defense Systems Architect & Operations Research Lead
- Member 4: Full-Stack C2 UI, Mission Assurance & QA Lead

Implements the multi-criteria prioritization formula:
    Priority Score = 0.35 * D_z + 0.25 * S_z + 0.20 * E_z + 0.20 * I_z
Where:
    - D_z: Damage density (ratio of damaged buildings to total buildings in zone)
    - S_z: Severe damage ratio (ratio of destroyed + major damage to damaged buildings)
    - E_z: Exposed population / critical infrastructure landmarks factor
    - I_z: Isolation factor (accessibility penalty based on road corridor severance)
"""

from typing import Any
import math
import numpy as np


# Critical landmark coordinates in the Mexico Beach disaster theater
CRITICAL_LANDMARKS = [
    {"name": "Emergency Aid Station / Clinic", "lon": -85.5525, "lat": 30.1740, "critical_weight": 1.5},
    {"name": "Temporary Evacuation Shelter", "lon": -85.5475, "lat": 30.1780, "critical_weight": 2.0},
    {"name": "Municipal Water / Power Substation", "lon": -85.5490, "lat": 30.1710, "critical_weight": 1.2},
]


def allocate_tactical_resources(
    priority_zones: list[dict[str, Any]],
    total_usar_squads: int = 8,
    total_ambulances: int = 14,
    total_debris_units: int = 6,
) -> list[dict[str, Any]]:
    """Greedy matching allocator dispatching tactical resources to critical zones.

    - Primary USAR Heavy Rescue Squads: Allocated to Rank #1 and structural collapse clusters.
    - Paramedic Ambulances: Scaled by casualty exposure and severe structural damage.
    - Debris Clearance Dozers: Routed to high-isolation and rubble-blocked access corridors.
    """
    avail_usar = total_usar_squads
    avail_amb = total_ambulances
    avail_debris = total_debris_units

    updated_zones = []
    for z in priority_zones:
        zone = dict(z)
        dest = zone.get("destroyed_buildings", 0)
        major = zone.get("major_damage_buildings", 0)
        dmg = zone.get("damaged_buildings", 0)
        tot = zone.get("total_buildings", 0)
        rank = zone.get("priority_rank", 999)
        isolation = zone.get("isolation_factor", 0.0)

        # 1. USAR Heavy Rescue Squads
        req_usar = 0
        if dest > 0:
            req_usar = max(1, math.ceil(dest / 2.0))
        elif major > 0 and rank <= 3:
            req_usar = 1
        allocated_usar = min(avail_usar, req_usar)
        avail_usar = max(0, avail_usar - allocated_usar)

        # 2. Paramedic Ambulances based on casualty exposure
        req_amb = 0
        if dest > 0 or major > 0:
            req_amb = max(1, dest + major)
        elif dmg > 0:
            req_amb = 1
        allocated_amb = min(avail_amb, req_amb)
        avail_amb = max(0, avail_amb - allocated_amb)

        # 3. Debris Clearance Dozers
        req_debris = 0
        if isolation > 0.4 or dest > 0:
            req_debris = 1 if dest <= 2 else 2
        allocated_debris = min(avail_debris, req_debris)
        avail_debris = max(0, avail_debris - allocated_debris)

        zone["allocated_usar_squads"] = allocated_usar
        zone["allocated_ambulances"] = allocated_amb
        zone["allocated_debris_units"] = allocated_debris

        updated_zones.append(zone)

    return updated_zones


def compute_priority_zones(
    feature_collection: dict[str, Any],
    grid_rows: int = 3,
    grid_cols: int = 3,
    default_bounds: tuple[float, float, float, float] | None = None,
) -> list[dict[str, Any]]:
    """Partition the disaster footprint into a grid and rank priority zones.

    Formula:
        Priority Score = 0.35 * D_z + 0.25 * S_z + 0.20 * E_z + 0.20 * I_z

    Where:
        - D_z: Damage density (ratio of damaged buildings to total buildings in zone)
        - S_z: Severe damage ratio (ratio of destroyed + major damage to damaged buildings)
        - E_z: Exposed population and critical asset proximity factor
        - I_z: Isolation and access impediment factor

    Returns:
        List of zone dicts sorted descending by priority_score.
    """
    features = feature_collection.get("features", [])

    # Determine spatial bounding box
    if features:
        all_lons: list[float] = []
        all_lats: list[float] = []
        for feat in features:
            geom = feat.get("geometry", {})
            coords = geom.get("coordinates", [])
            if geom.get("type") == "Polygon" and coords:
                for pt in coords[0]:
                    all_lons.append(pt[0])
                    all_lats.append(pt[1])
            elif geom.get("type") == "MultiPolygon" and coords:
                for poly in coords:
                    if poly:
                        for pt in poly[0]:
                            all_lons.append(pt[0])
                            all_lats.append(pt[1])

        if all_lons and all_lats:
            min_lon, max_lon = min(all_lons), max(all_lons)
            min_lat, max_lat = min(all_lats), max(all_lats)
            if min_lon == max_lon:
                min_lon -= 0.001
                max_lon += 0.001
            if min_lat == max_lat:
                min_lat -= 0.001
                max_lat += 0.001
        else:
            min_lon, min_lat, max_lon, max_lat = default_bounds or (-85.5550, 30.1700, -85.5450, 30.1800)
    else:
        min_lon, min_lat, max_lon, max_lat = default_bounds or (-85.5550, 30.1700, -85.5450, 30.1800)

    lon_step = (max_lon - min_lon) / max(grid_cols, 1)
    lat_step = (max_lat - min_lat) / max(grid_rows, 1)

    # Initialize raw zones
    raw_zones: list[dict[str, Any]] = []
    zone_idx = 1
    for r in range(grid_rows):
        for c in range(grid_cols):
            # Row 0 is northernmost (highest latitude)
            z_max_lat = max_lat - r * lat_step
            z_min_lat = z_max_lat - lat_step
            z_min_lon = min_lon + c * lon_step
            z_max_lon = z_min_lon + lon_step

            raw_zones.append({
                "zone_id": f"Z-{zone_idx:03d}",
                "bounds": [round(z_min_lon, 7), round(z_min_lat, 7), round(z_max_lon, 7), round(z_max_lat, 7)],
                "buildings": [],
            })
            zone_idx += 1

    # Assign buildings to zones by centroid
    for feat in features:
        geom = feat.get("geometry", {})
        coords = geom.get("coordinates", [])
        centroid_lon, centroid_lat = None, None

        if geom.get("type") == "Polygon" and coords and coords[0]:
            pts = coords[0]
            centroid_lon = sum(p[0] for p in pts) / len(pts)
            centroid_lat = sum(p[1] for p in pts) / len(pts)
        elif geom.get("type") == "MultiPolygon" and coords and coords[0] and coords[0][0]:
            pts = coords[0][0]
            centroid_lon = sum(p[0] for p in pts) / len(pts)
            centroid_lat = sum(p[1] for p in pts) / len(pts)

        if centroid_lon is None or centroid_lat is None:
            continue

        for zone in raw_zones:
            b = zone["bounds"]
            if (b[0] <= centroid_lon <= b[2] or np.isclose(centroid_lon, b[2])) and \
               (b[1] <= centroid_lat <= b[3] or np.isclose(centroid_lat, b[3])):
                zone["buildings"].append(feat)
                break

    # Calculate zone base metrics
    zone_stats: list[dict[str, Any]] = []
    for zone in raw_zones:
        b_list = zone["buildings"]
        total_b = len(b_list)

        no_dmg = 0
        minor_dmg = 0
        major_dmg = 0
        destroyed = 0
        confidences: list[float] = []

        for b in b_list:
            props = b.get("properties", {})
            dmg = props.get("damage_class", props.get("damage", "no-damage")).lower()
            conf = float(props.get("confidence", 0.85))
            confidences.append(conf)

            if dmg in ("destroyed",):
                destroyed += 1
            elif dmg in ("major-damage", "major"):
                major_dmg += 1
            elif dmg in ("minor-damage", "minor"):
                minor_dmg += 1
            elif dmg in ("damaged",):
                major_dmg += 1
            else:
                no_dmg += 1

        damaged_b = minor_dmg + major_dmg + destroyed
        mean_conf = float(np.mean(confidences)) if confidences else 0.0

        # Exposed population: 2.4 residents per residential parcel
        resident_pop = int(round(total_b * 2.4))

        # Check for critical landmark presence
        z_bounds = zone["bounds"]
        critical_landmarks_found = []
        critical_weight_sum = 0.0
        for lm in CRITICAL_LANDMARKS:
            if z_bounds[0] <= lm["lon"] <= z_bounds[2] and z_bounds[1] <= lm["lat"] <= z_bounds[3]:
                critical_landmarks_found.append(lm["name"])
                critical_weight_sum += lm["critical_weight"]

        zone_stats.append({
            "zone_id": zone["zone_id"],
            "bounds": zone["bounds"],
            "total_buildings": total_b,
            "damaged_buildings": damaged_b,
            "destroyed_buildings": destroyed,
            "major_damage_buildings": major_dmg,
            "minor_damage_buildings": minor_dmg,
            "undamaged_buildings": no_dmg,
            "mean_confidence": round(mean_conf, 4),
            "estimated_population": resident_pop,
            "critical_landmarks": critical_landmarks_found,
            "critical_weight": critical_weight_sum,
        })

    # Normalization parameter across zones
    max_raw_exposure = max([z["estimated_population"] + (z["critical_weight"] * 10) for z in zone_stats] + [1])

    scored_zones: list[dict[str, Any]] = []
    for z in zone_stats:
        tot = z["total_buildings"]
        dmg = z["damaged_buildings"]
        dest = z["destroyed_buildings"]
        major = z["major_damage_buildings"]
        conf = z["mean_confidence"]

        if tot == 0:
            # Empty zone
            scored_zones.append({
                "zone_id": z["zone_id"],
                "priority_score": 0.0,
                "damage_density": 0.0,
                "severe_damage_ratio": 0.0,
                "exposed_population_factor": 0.0,
                "isolation_factor": 0.0,
                "total_buildings": 0,
                "damaged_buildings": 0,
                "destroyed_buildings": 0,
                "major_damage_buildings": 0,
                "estimated_population": 0,
                "critical_landmarks": [],
                "mean_confidence": 0.0,
                "bounds": z["bounds"],
                "reason": "No detected structures in zone footprint.",
                "allocated_usar_squads": 0,
                "allocated_ambulances": 0,
                "allocated_debris_units": 0,
            })
            continue

        # 1. Damage Density (D_z): ratio of damaged to total buildings
        damage_density = dmg / tot

        # 2. Severe Damage Ratio (S_z): ratio of (destroyed + major) to damaged buildings
        severe_ratio = ((dest + major) / dmg) if dmg > 0 else 0.0

        # 3. Exposed Population & Critical Assets (E_z)
        raw_exposure = z["estimated_population"] + (z["critical_weight"] * 10)
        exposed_factor = min(1.0, max(0.2, raw_exposure / max_raw_exposure))

        # 4. Isolation Factor (I_z): road blockade and severe rubble impact
        # High structural collapse generates road blockades that isolate survivors
        isolation_factor = min(1.0, 0.4 * damage_density + 0.6 * severe_ratio)
        if dest > 0:
            isolation_factor = max(isolation_factor, 0.70)

        # Multi-source formula: 0.35 * D_z + 0.25 * S_z + 0.20 * E_z + 0.20 * I_z
        raw_score = (
            0.35 * damage_density
            + 0.25 * severe_ratio
            + 0.20 * exposed_factor
            + 0.20 * isolation_factor
        )
        score = round(float(raw_score), 4)

        # Build human-readable operational reason
        if dest > 0 and major > 0:
            reason = f"Critical severe structural destruction: {dest} destroyed, {major} major damage."
        elif dest > 0:
            reason = f"High structural collapse risk: {dest} destroyed buildings detected."
        elif major > 0:
            reason = f"Significant structural compromise: {major} major damaged structures."
        elif dmg > 0:
            reason = f"Moderate impact: {dmg} damaged buildings ({damage_density:.0%} damage density)."
        else:
            reason = f"All {tot} structures identified as undamaged."

        scored_zones.append({
            "zone_id": z["zone_id"],
            "priority_score": score,
            "damage_density": round(damage_density, 4),
            "severe_damage_ratio": round(severe_ratio, 4),
            "exposed_population_factor": round(exposed_factor, 4),
            "isolation_factor": round(isolation_factor, 4),
            "total_buildings": tot,
            "damaged_buildings": dmg,
            "destroyed_buildings": dest,
            "major_damage_buildings": major,
            "estimated_population": z["estimated_population"],
            "critical_landmarks": z["critical_landmarks"],
            "mean_confidence": round(conf, 4),
            "bounds": z["bounds"],
            "reason": reason,
        })

    # Sort descending by priority_score, then damaged_buildings, then destroyed_buildings
    scored_zones.sort(
        key=lambda x: (x["priority_score"], x["damaged_buildings"], x["destroyed_buildings"]),
        reverse=True,
    )

    # Assign 1-indexed priority_rank
    for idx, z in enumerate(scored_zones, start=1):
        z["priority_rank"] = idx

    # Compute greedy resource allocation across the ranked zones
    scored_zones = allocate_tactical_resources(scored_zones)

    return scored_zones

    return scored_zones


def priority_zones_to_geojson(zones: list[dict[str, Any]]) -> dict[str, Any]:
    """Convert ranked priority zones into a GeoJSON FeatureCollection with polygon boundaries."""
    features = []
    for z in zones:
        min_lon, min_lat, max_lon, max_lat = z["bounds"]
        poly_coords = [
            [
                [min_lon, max_lat],
                [max_lon, max_lat],
                [max_lon, min_lat],
                [min_lon, min_lat],
                [min_lon, max_lat],
            ]
        ]
        feat = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": poly_coords,
            },
            "properties": {
                "zone_id": z["zone_id"],
                "priority_rank": z["priority_rank"],
                "priority_score": z["priority_score"],
                "total_buildings": z["total_buildings"],
                "damaged_buildings": z["damaged_buildings"],
                "destroyed_buildings": z["destroyed_buildings"],
                "major_damage_buildings": z["major_damage_buildings"],
                "mean_confidence": z["mean_confidence"],
                "reason": z["reason"],
                "damage_class": "destroyed" if z["destroyed_buildings"] > 0 else ("major-damage" if z["major_damage_buildings"] > 0 else ("minor-damage" if z["damaged_buildings"] > 0 else "no-damage")),
                "confidence": float(z["mean_confidence"]),
            },
        }
        features.append(feat)

    return {
        "type": "FeatureCollection",
        "features": features,
    }
