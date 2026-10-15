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


# Comprehensive Critical Emergency Infrastructure in Mexico Beach disaster theater
CRITICAL_FACILITIES = [
    {
        "id": "FAC-MED-01",
        "name": "Bay County Emergency Field Hospital & Trauma Center",
        "facility_type": "hospital",
        "lon": -85.5525,
        "lat": 30.1740,
        "capacity": "45 Acute Trauma Beds / 4 Surgical Bays",
        "status": "Operational (Designated Triage Hub Alpha)",
        "icon": "plus",
        "marker_color": "red",
        "critical_weight": 2.5,
    },
    {
        "id": "FAC-SHL-01",
        "name": "FEMA / Red Cross Evacuation Shelter & Relief Depot",
        "facility_type": "shelter",
        "lon": -85.5475,
        "lat": 30.1780,
        "capacity": "350 Evacuees / Food, Water & Generator Power",
        "status": "Active Intake & Humanitarian Hub",
        "icon": "home",
        "marker_color": "blue",
        "critical_weight": 2.0,
    },
    {
        "id": "FAC-FIRE-01",
        "name": "Mexico Beach Fire & USAR Station #1",
        "facility_type": "fire_station",
        "lon": -85.5450,
        "lat": 30.1650,
        "capacity": "4 Heavy Rescue Trucks / Structural Shoring Kits",
        "status": "Active Rescue Dispatch",
        "icon": "fire",
        "marker_color": "orange",
        "critical_weight": 1.8,
    },
    {
        "id": "FAC-UTIL-01",
        "name": "Municipal Emergency Power & Water Substation",
        "facility_type": "utility",
        "lon": -85.5490,
        "lat": 30.1710,
        "capacity": "Industrial Generator Backup / Potable Water Reserves",
        "status": "Operational (Lifeline Infrastructure)",
        "icon": "bolt",
        "marker_color": "purple",
        "critical_weight": 1.5,
    },
    {
        "id": "FAC-FOB-01",
        "name": "Forward Operating Base (FOB) Alpha Helipad",
        "facility_type": "helipad",
        "lon": -85.5510,
        "lat": 30.1530,
        "capacity": "2 Air Ambulance Helipads / Forward Staging Base",
        "status": "Active Medevac Perimeter",
        "icon": "plane",
        "marker_color": "darkblue",
        "critical_weight": 2.2,
    },
]

# Backward compatibility alias
CRITICAL_LANDMARKS = CRITICAL_FACILITIES


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Calculate the great-circle distance between two GPS coordinates in kilometers."""
    r = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * (math.sin(delta_lambda / 2.0) ** 2)
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return round(r * c, 3)


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
        minor_dmg = z.get("minor_damage_buildings", 0)
        conf = z["mean_confidence"]

        # Determine zone center and calculate distance to critical facilities
        z_b = z["bounds"]
        z_c_lon = (z_b[0] + z_b[2]) / 2.0
        z_c_lat = (z_b[1] + z_b[3]) / 2.0

        nearest_fac = "Bay County Emergency Field Hospital & Trauma Center"
        min_fac_dist = 999.0
        nearest_hosp = "Bay County Emergency Field Hospital & Trauma Center"
        min_hosp_dist = 999.0
        facs_in_zone = []

        for fac in CRITICAL_FACILITIES:
            dist = haversine_km(z_c_lon, z_c_lat, fac["lon"], fac["lat"])
            if dist < min_fac_dist:
                min_fac_dist = dist
                nearest_fac = fac["name"]
            if fac.get("facility_type") == "hospital" and dist < min_hosp_dist:
                min_hosp_dist = dist
                nearest_hosp = fac["name"]
            if z_b[0] <= fac["lon"] <= z_b[2] and z_b[1] <= fac["lat"] <= z_b[3]:
                facs_in_zone.append(fac["name"])

        resident_pop = z["estimated_population"]
        vulnerable_pop = int(round(resident_pop * 0.22))
        est_casualties = int(round(dest * 1.2 + major * 0.5 + minor_dmg * 0.1))

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
                "vulnerable_population": 0,
                "estimated_casualties": 0,
                "critical_landmarks": [],
                "facilities_in_zone": facs_in_zone,
                "nearest_facility": nearest_fac,
                "facility_distance_km": round(min_fac_dist, 2),
                "nearest_hospital": nearest_hosp,
                "hospital_distance_km": round(min_hosp_dist, 2),
                "transit_time_mins": 2.0,
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

        # Realistic transit time in minutes: 35 km/h transit + rubble isolation penalty
        transit_time_mins = round((min_hosp_dist / 35.0 * 60.0) + (isolation_factor * 8.5), 1)

        # Build human-readable operational reason with demographic and routing context
        if dest > 0 and major > 0:
            reason = f"Critical severe structural destruction: {dest} destroyed, {major} major damage. {est_casualties} est. casualties among {resident_pop} residents ({vulnerable_pop} vulnerable). Route to {nearest_hosp}: {min_hosp_dist:.2f} km ({transit_time_mins} mins)."
        elif dest > 0:
            reason = f"High structural collapse risk: {dest} destroyed buildings. {est_casualties} est. casualties. Transit to {nearest_hosp}: {transit_time_mins} mins."
        elif major > 0:
            reason = f"Significant structural compromise: {major} major damaged structures. {resident_pop} residents exposed."
        elif dmg > 0:
            reason = f"Moderate impact: {dmg} damaged buildings ({damage_density:.0%} damage density). {resident_pop} residents."
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
            "estimated_population": resident_pop,
            "vulnerable_population": vulnerable_pop,
            "estimated_casualties": est_casualties,
            "critical_landmarks": z["critical_landmarks"],
            "facilities_in_zone": facs_in_zone,
            "nearest_facility": nearest_fac,
            "facility_distance_km": round(min_fac_dist, 2),
            "nearest_hospital": nearest_hosp,
            "hospital_distance_km": round(min_hosp_dist, 2),
            "transit_time_mins": transit_time_mins,
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
