"""Tactical emergency response resource allocation engine for MERIDIAN-Lite Stage 2.

Translates building damage triage and priority zone scores into actionable
tactical resource deployment recommendations (USAR teams, ambulances, debris clearance squads).
"""

from dataclasses import dataclass, asdict
import math
from typing import Any


@dataclass
class ZoneResourceDeployment:
    """Tactical emergency resource dispatch recommendation for a single priority zone."""
    zone_id: str
    priority_rank: int
    priority_score: float
    total_buildings: int
    destroyed_count: int
    major_damage_count: int
    usar_teams_needed: int
    paramedic_units_needed: int
    debris_clearance_units: int
    uav_secondary_recon: int
    staging_coordinates: tuple[float, float]
    dispatch_urgency: str
    action_directive: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def calculate_tactical_resource_allocation(
    priority_zones: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compute automated emergency resource dispatch for all priority zones.

    Allocation Rules:
    - USAR (Urban Search & Rescue) Squads: 1 team per 2 destroyed structures (min 1 if destroyed > 0).
    - Paramedic / Triage Units: 1 unit per major damaged structure.
    - Heavy Debris Clearance Squads: 1 squad if severe structures >= 2 or priority score >= 0.50.
    - Secondary UAV Re-reconnaissance: 1 drone for zones with rank <= 3.

    Args:
        priority_zones: List of priority zone dictionaries from compute_priority_zones().

    Returns:
        List of ZoneResourceDeployment dictionaries sorted by priority rank.
    """
    deployments: list[dict[str, Any]] = []

    for zone in sorted(priority_zones, key=lambda z: z.get("priority_rank", 999)):
        z_id = zone.get("zone_id", "Z-000")
        rank = zone.get("priority_rank", 999)
        score = zone.get("priority_score", 0.0)
        tot_bld = zone.get("total_buildings", 0)
        dest_cnt = zone.get("destroyed_buildings", 0)
        maj_cnt = zone.get("major_damage_buildings", 0)
        bounds = zone.get("bounds", [-85.548, 30.154, -85.548, 30.154])

        c_lon = round((bounds[0] + bounds[2]) / 2.0, 6)
        c_lat = round((bounds[1] + bounds[3]) / 2.0, 6)

        # USAR calculation
        if dest_cnt > 0:
            usar = max(1, math.ceil(dest_cnt / 2.0))
        elif maj_cnt > 1:
            usar = 1
        else:
            usar = 0

        # Medical calculation
        med = max(0, maj_cnt + (1 if dest_cnt > 0 else 0))

        # Heavy debris clearance
        if dest_cnt + maj_cnt >= 2 or score >= 0.50:
            debris = 1
        else:
            debris = 0

        # UAV Secondary
        uav = 1 if rank <= 3 and tot_bld > 0 else 0

        # Urgency & directive
        if rank == 1 and score > 0.4:
            urgency = "CRITICAL (IMMEDIATE DISPATCH)"
            directive = "Deploy primary USAR heavy rescue squad; initiate structural breach & survivor extraction."
        elif rank <= 3 and score > 0.2:
            urgency = "HIGH (FIRST WAVE)"
            directive = "Dispatch paramedic triage unit and secondary reconnaissance drone; clear access corridors."
        elif tot_bld > 0:
            urgency = "MODERATE (SECOND WAVE)"
            directive = "Conduct secondary building stability assessments and structural utility isolation."
        else:
            urgency = "LOW / STANDBY"
            directive = "Maintain passive drone perimeter monitoring; zero active structural threats detected."

        deployment = ZoneResourceDeployment(
            zone_id=z_id,
            priority_rank=rank,
            priority_score=score,
            total_buildings=tot_bld,
            destroyed_count=dest_cnt,
            major_damage_count=maj_cnt,
            usar_teams_needed=usar,
            paramedic_units_needed=med,
            debris_clearance_units=debris,
            uav_secondary_recon=uav,
            staging_coordinates=(c_lat, c_lon),
            dispatch_urgency=urgency,
            action_directive=directive,
        )
        deployments.append(deployment.to_dict())

    return deployments


def summarize_total_fleet_requirements(deployments: list[dict[str, Any]]) -> dict[str, int]:
    """Aggregate total emergency fleet assets required across the entire disaster area."""
    return {
        "total_usar_teams": sum(d["usar_teams_needed"] for d in deployments),
        "total_paramedic_units": sum(d["paramedic_units_needed"] for d in deployments),
        "total_debris_clearance_units": sum(d["debris_clearance_units"] for d in deployments),
        "total_uav_secondary_missions": sum(d["uav_secondary_recon"] for d in deployments),
    }
