"""Priority-zone engine for MERIDIAN-Lite Stage 2.

Divides the affected disaster area into spatial grid zones and computes
prioritized response rankings using transparent, multi-factor decision formulas.
"""

from typing import Any
import numpy as np


def compute_priority_zones(
    feature_collection: dict[str, Any],
    grid_rows: int = 3,
    grid_cols: int = 3,
    default_bounds: tuple[float, float, float, float] | None = None,
) -> list[dict[str, Any]]:
    """Partition the disaster footprint into a grid and rank priority zones.

    Formula:
        priority_score = (
            0.40 * damage_density
            + 0.25 * severe_damage_ratio
            + 0.20 * total_affected_buildings
            + 0.15 * confidence_weight
        )

    Args:
        feature_collection: GeoJSON FeatureCollection with building polygon features.
        grid_rows: Number of grid rows (default: 3).
        grid_cols: Number of grid columns (default: 3).
        default_bounds: Optional (min_lon, min_lat, max_lon, max_lat) if feature collection is empty.

    Returns:
        List of zone dicts sorted by priority_rank (descending priority score).
    """
    features = feature_collection.get("features", [])

    # Determine spatial bounding box
    if features:
        all_lons: list[float] = []
        all_lats: list[float] = []
        for feat in features:
            geom = feat.get("geometry", {})
            coords = geom.get("coordinates", [])
            # Support Polygon and MultiPolygon
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
            # Add small padding if bounds are degenerate
            if min_lon == max_lon:
                min_lon -= 0.001
                max_lon += 0.001
            if min_lat == max_lat:
                min_lat -= 0.001
                max_lat += 0.001
        else:
            min_lon, min_lat, max_lon, max_lat = default_bounds or (-85.555, 30.170, -85.545, 30.180)
    else:
        min_lon, min_lat, max_lon, max_lat = default_bounds or (-85.555, 30.170, -85.545, 30.180)

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
            # Inclusive check for grid boundary
            if (b[0] <= centroid_lon <= b[2] or np.isclose(centroid_lon, b[2])) and \
               (b[1] <= centroid_lat <= b[3] or np.isclose(centroid_lat, b[3])):
                zone["buildings"].append(feat)
                break

    # Calculate zone metrics
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
                # Binary damage fallback - count as major damage
                major_dmg += 1
            else:
                no_dmg += 1

        damaged_b = minor_dmg + major_dmg + destroyed
        mean_conf = float(np.mean(confidences)) if confidences else 0.0

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
        })

    # Normalization parameters across all zones
    max_affected = max([z["damaged_buildings"] for z in zone_stats] + [1])

    scored_zones: list[dict[str, Any]] = []
    for z in zone_stats:
        tot = z["total_buildings"]
        dmg = z["damaged_buildings"]
        dest = z["destroyed_buildings"]
        major = z["major_damage_buildings"]
        conf = z["mean_confidence"]

        # 1. Damage density: ratio of damaged to total buildings in zone
        damage_density = (dmg / tot) if tot > 0 else 0.0

        # 2. Severe damage ratio: ratio of severe (destroyed + major) to damaged buildings
        severe_ratio = ((dest + major) / dmg) if dmg > 0 else 0.0

        # 3. Normalized total affected buildings across the disaster footprint
        total_affected_norm = dmg / max_affected

        # 4. Confidence weight
        confidence_weight = conf

        # Transparent formula
        raw_score = (
            0.40 * damage_density
            + 0.25 * severe_ratio
            + 0.20 * total_affected_norm
            + 0.15 * confidence_weight
        )
        score = round(float(raw_score), 4)

        # Build human-readable reason
        if tot == 0:
            reason = "No detected structures in zone footprint."
        elif dest > 0 and major > 0:
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
            "total_buildings": tot,
            "damaged_buildings": dmg,
            "destroyed_buildings": dest,
            "major_damage_buildings": major,
            "mean_confidence": round(conf, 4),
            "bounds": z["bounds"],
            "reason": reason,
        })

    # Sort descending by priority_score, then damaged_buildings
    scored_zones.sort(key=lambda x: (x["priority_score"], x["damaged_buildings"], x["destroyed_buildings"]), reverse=True)

    # Assign 1-indexed priority_rank
    for idx, z in enumerate(scored_zones, start=1):
        z["priority_rank"] = idx

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
