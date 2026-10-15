"""Stage 3: Predictive Multi-Hazard Risk & Intelligence Engine for MERIDIAN-Lite.

Combines structural damage assessments with historical Hurricane Michael weather data,
environmental topography, elevation, and storm surge models to forecast high-risk
secondary hazard zones (secondary structural collapse, storm surge inundation, debris access blockades).
"""

from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class PredictiveRiskAssessment:
    """Predictive compound risk forecast for a disaster priority sector."""
    zone_id: str
    compound_vulnerability_index: float  # [0.0 - 1.0]
    risk_tier: str                       # CRITICAL, SEVERE, ELEVATED, LOW
    secondary_collapse_risk: float       # [0.0 - 1.0]
    surge_inundation_risk: float         # [0.0 - 1.0]
    access_blockade_risk: float          # [0.0 - 1.0]
    primary_threat_description: str
    preventative_countermeasure: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Historical NOAA Hurricane Michael meteorological & environmental baseline for Bay County, FL
METEOROLOGICAL_BASELINE = {
    "storm_name": "Hurricane Michael",
    "landfall_category": 5,
    "sustained_wind_speed_kt": 140.0,
    "peak_gusts_kt": 162.0,
    "post_storm_residual_gusts_kt": 38.0,
    "storm_surge_peak_m": 3.8,
    "mean_ground_elevation_msl_m": 4.5,
    "rainfall_accumulation_mm": 145.0,
}


def calculate_predictive_hazard_analysis(
    priority_zones: list[dict[str, Any]],
    wind_gusts_kt: float = 38.0,
    surge_height_m: float = 2.4,
    ground_elevation_m: float = 4.2,
) -> list[dict[str, Any]]:
    """Forecast compound multi-hazard risks across priority zones.

    Formulation:
    1. Secondary Collapse Risk:
       Structures with minor/major damage that risk total structural collapse
       due to compromised roof integrity and persistent post-event gale gusts (>30 kt).
    2. Surge Inundation Risk:
       Topographic deficit = max(0.0, surge_height_m - (ground_elevation_m * elevation_factor)).
    3. Access Blockade Risk:
       High tree-fall probability in western forest fringe combining with road proximity.
    4. Compound Vulnerability Index (CVI):
       CVI = 0.45 * Damage_Score + 0.30 * Collapse_Risk + 0.15 * Surge_Risk + 0.10 * Blockade_Risk.

    Args:
        priority_zones: List of priority zones from compute_priority_zones().
        wind_gusts_kt: Active residual wind gusts in knots.
        surge_height_m: Predicted storm surge water level above mean sea level.
        ground_elevation_m: Estimated terrain elevation in meters.

    Returns:
        List of PredictiveRiskAssessment dictionaries.
    """
    assessments: list[dict[str, Any]] = []

    for zone in sorted(priority_zones, key=lambda z: z.get("priority_rank", 999)):
        z_id = zone.get("zone_id", "Z-000")
        score = zone.get("priority_score", 0.0)
        tot_bld = zone.get("total_buildings", 0)
        dest_cnt = zone.get("destroyed_buildings", 0)
        maj_cnt = zone.get("major_damage_buildings", 0)
        min_cnt = zone.get("minor_damage_buildings", 0)

        # 1. Secondary Collapse Probability
        if tot_bld > 0:
            fragility = (maj_cnt * 0.8 + min_cnt * 0.5) / float(tot_bld)
            wind_stress = min(1.0, max(0.0, (wind_gusts_kt - 20.0) / 40.0))
            collapse_risk = round(min(1.0, fragility * 0.7 + wind_stress * 0.3), 3)
        else:
            collapse_risk = 0.0

        # 2. Surge Inundation Risk (relative to local elevation)
        # Western sectors have slightly lower elevation runoff into drainage canals
        z_num = int(z_id.split("-")[-1]) if "-" in z_id else 1
        local_elev = ground_elevation_m - (0.3 if z_num in [1, 4, 7] else 0.0)
        topographic_head = surge_height_m - local_elev
        if topographic_head > 0:
            surge_risk = round(min(1.0, 0.5 + topographic_head * 0.4), 3)
        else:
            surge_risk = round(max(0.05, min(0.35, (surge_height_m / 4.0))), 3)

        # 3. Access Blockade Risk (woodland tree blowdown)
        # Sectors 1, 4, 7 borders the dense pine stand on the west
        is_woodland_border = z_num in [1, 4, 7]
        blockade_risk = 0.78 if is_woodland_border and wind_gusts_kt >= 30.0 else 0.28

        # 4. Compound Vulnerability Index (CVI)
        cvi = round(
            0.45 * score +
            0.30 * collapse_risk +
            0.15 * surge_risk +
            0.10 * blockade_risk,
            3
        )

        # Risk Tier Classification
        if cvi >= 0.55 or (dest_cnt >= 2 and collapse_risk >= 0.5):
            tier = "CRITICAL RISK"
            threat = "Severe structural failure & immediate secondary roof collapse under residual gales."
            action = "Establish 50-meter perimeter cordon; restrict entry until urban shoring teams arrive."
        elif cvi >= 0.35:
            tier = "ELEVATED RISK"
            threat = "Partial wall instability and localized road obstruction from downed timber."
            action = "Pre-position heavy clearing wheel loaders; issue helmet and PPE mandate for ground crews."
        elif tot_bld > 0:
            tier = "MODERATE RISK"
            threat = "Superficial cladding debris and localized standing water pooling."
            action = "Conduct systematic visual utility inspections (gas, overhead power)."
        else:
            tier = "LOW RISK"
            threat = "Open terrain with minor canopy defoliation; no structural hazards."
            action = "Designate as temporary tactical staging zone or emergency helicopter landing zone (HLZ)."

        assessment = PredictiveRiskAssessment(
            zone_id=z_id,
            compound_vulnerability_index=cvi,
            risk_tier=tier,
            secondary_collapse_risk=collapse_risk,
            surge_inundation_risk=surge_risk,
            access_blockade_risk=blockade_risk,
            primary_threat_description=threat,
            preventative_countermeasure=action,
        )
        assessments.append(assessment.to_dict())

    return assessments
