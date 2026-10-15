"""
Tactical Metrics & Priority-Zone Intelligence Views for MERIDIAN-Lite.
Provides metric cards, damage class breakdowns, ranked priority zone tables, and inventory dataframes.
"""

from typing import Any
import pandas as pd
import streamlit as st

from utils.resource_allocation import calculate_tactical_resource_allocation


def render_tactical_summary(metrics: dict[str, Any], processing_time_ms: float | None = None) -> None:
    """Render top-level tactical assessment summary cards and breakdown counts."""
    st.markdown("### 📊 Tactical Assessment Summary")

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric(
            label="Total Buildings Detected",
            value=metrics["total_buildings"],
            help="Total building polygons extracted in target reconnaissance sector by the real AI model.",
        )
    with m_col2:
        severe_count = metrics["major_damage_count"] + metrics["destroyed_count"]
        severe_pct = (
            round((severe_count / metrics["total_buildings"]) * 100, 1)
            if metrics["total_buildings"] > 0
            else 0.0
        )
        st.metric(
            label="Destroyed / Major Damage",
            value=f"{severe_count} ({severe_pct}%)",
            delta=f"-{severe_pct}% severe" if severe_pct > 0 else "None",
            delta_color="inverse",
            help="Count and percentage of structures suffering major damage or total destruction.",
        )
    with m_col3:
        st.metric(
            label="Total Damaged Percentage",
            value=f"{metrics['damaged_percentage']}%",
            help="Combined percentage: minor-damage + major-damage + destroyed.",
        )
    with m_col4:
        conf_pct = round(metrics["average_confidence"] * 100, 1)
        st.metric(
            label="Mean Assessment Confidence",
            value=f"{conf_pct}%",
            help="Average model classification confidence across detected building features.",
        )

    # Detailed Breakdown Counts
    breakdown_col1, breakdown_col2, breakdown_col3, breakdown_col4, breakdown_col5 = st.columns(5)
    with breakdown_col1:
        st.markdown(f"**🟩 No Damage:** {metrics['no_damage_count']}")
    with breakdown_col2:
        st.markdown(f"**🟨 Minor Damage:** {metrics['minor_damage_count']}")
    with breakdown_col3:
        st.markdown(f"**🟧 Major Damage:** {metrics['major_damage_count']}")
    with breakdown_col4:
        st.markdown(f"**🟥 Destroyed:** {metrics['destroyed_count']}")
    with breakdown_col5:
        proc_time = processing_time_ms if processing_time_ms is not None else 180.0
        st.markdown(f"**⏱️ Latency:** {proc_time} ms")

    st.markdown("---")


def render_priority_zones_section(priority_zones: list[dict[str, Any]]) -> None:
    """Render Stage 2 priority-zone scoring, first-rescue callout, and tactical deployment table."""
    st.markdown("### 🎯 Stage 2: Priority-Zone Scoring & Emergency Rescue Allocation")
    st.caption(
        "Multi-factor decision-support ranking derived from: "
        "0.40 × Damage Density + 0.25 × Severe Damage Ratio + 0.20 × Total Affected Structures + 0.15 × Confidence Weight."
    )

    if not priority_zones:
        st.info("No priority zones available for current view.")
        st.markdown("---")
        return

    top_zone = priority_zones[0]
    severe_count = top_zone.get("destroyed_buildings", 0) + top_zone.get("major_damage_buildings", 0)

    # Calculate tactical resource dispatch
    deployments = calculate_tactical_resource_allocation(priority_zones)
    top_deploy = deployments[0] if deployments else None

    usar_squads = top_deploy.get("usar_teams_needed", 1) if top_deploy else 1
    ambulances = top_deploy.get("paramedic_units_needed", 1) if top_deploy else 1
    debris_units = top_deploy.get("debris_clearance_units", 1) if top_deploy else 1
    action_directive = top_deploy.get("action_directive", "Execute immediate structural search and rescue sweep.") if top_deploy else "Immediate SAR deployment."

    # Highlight the Area to be Rescued First
    st.markdown(
        f"""
        <div style="background: linear-gradient(90deg, #7f1d1d 0%, #991b1b 100%);
                    border: 2px solid #ef4444; border-radius: 8px; padding: 16px 20px;
                    margin-bottom: 20px; color: white;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
                <div>
                    <span style="background: #ef4444; color: white; padding: 4px 10px; border-radius: 4px; font-weight: 800; font-size: 0.82rem; letter-spacing: 1px;">
                        🚨 IMMEDIATE RESCUE PRIORITY · DISPATCH FIRST
                    </span>
                    <div style="font-size: 1.45rem; font-weight: 800; margin-top: 6px; letter-spacing: 0.5px;">
                        SECTOR: PRIORITY ZONE {top_zone['zone_id']} · Score: {top_zone['priority_score']:.3f} (Rank #1)
                    </div>
                    <div style="font-size: 0.92rem; color: #fecaca; margin-top: 4px;">
                        <b>Triage Rationale:</b> {top_zone.get('reason', 'High concentration of damaged structures requiring urgent search and rescue.')}
                    </div>
                    <div style="font-size: 0.88rem; color: #fef08a; margin-top: 4px; font-weight: 600;">
                        <b>Directive:</b> {action_directive}
                    </div>
                </div>
                <div style="text-align: right; background: rgba(0,0,0,0.35); padding: 12px 18px; border-radius: 6px; border: 1px solid rgba(239,68,68,0.5);">
                    <div style="font-size: 0.78rem; color: #fca5a5; text-transform: uppercase; font-weight: 700;">First-Wave Resource Deployment</div>
                    <div style="font-size: 1.15rem; font-weight: 700; color: white; margin-top: 4px;">
                        🪖 {usar_squads} USAR Squad(s) &nbsp;|&nbsp; 🚑 {ambulances} Ambulance(s)
                    </div>
                    <div style="font-size: 0.82rem; color: #fecaca; margin-top: 3px;">
                        🚜 {debris_units} Heavy Debris Clearance Unit(s)
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    pz_c1, pz_c2, pz_c3, pz_c4, pz_c5 = st.columns(5)
    with pz_c1:
        st.metric("Top Priority Sector", top_zone["zone_id"], help="Highest-need triage sector.")
    with pz_c2:
        st.metric(
            "Priority Score",
            f"{top_zone['priority_score']:.3f}",
            delta="Rank #1",
            help="Multi-criteria triage score [0.0 - 1.0].",
        )
    with pz_c3:
        st.metric(
            "Exposed Population",
            f"{top_zone.get('estimated_population', 0)} occupants",
            delta=f"{top_zone.get('vulnerable_population', 0)} vulnerable",
            help="Estimated resident population and high-vulnerability citizens.",
        )
    with pz_c4:
        st.metric(
            "Estimated Casualties",
            f"{top_zone.get('estimated_casualties', 0)}",
            delta=f"{severe_count} severe collapses",
            delta_color="inverse",
            help="Estimated trapped or injured persons based on structural collapse model.",
        )
    with pz_c5:
        st.metric(
            "Transit to Hospital",
            f"{top_zone.get('transit_time_mins', 3.0)} mins",
            delta=f"{top_zone.get('hospital_distance_km', 1.0):.2f} km",
            help="Estimated ambulance transit time along cleared safe corridors.",
        )

    # Format zones table with Mission 6 Core: Population, Facilities, and Routing
    pz_rows = []
    for z in priority_zones:
        # Find matching deployment
        d = next((item for item in deployments if item["zone_id"] == z["zone_id"]), {})
        pz_rows.append({
            "Priority Rank": f"Rank #{z['priority_rank']}",
            "Sector ID": z["zone_id"],
            "Priority Score": f"{z['priority_score']:.3f}",
            "Structures": z["total_buildings"],
            "Severe Damage": z["destroyed_buildings"] + z["major_damage_buildings"],
            "Exposed Pop": f"{z.get('estimated_population', 0)} ({z.get('vulnerable_population', 0)} vuln)",
            "Est. Casualties": f"⚠️ {z.get('estimated_casualties', 0)}",
            "Nearest Facility": z.get("nearest_facility", "Field Hospital"),
            "Facility Dist": f"{z.get('facility_distance_km', 1.0):.2f} km",
            "Transit Time": f"{z.get('transit_time_mins', 3.0)} min",
            "USAR Teams": f"🪖 {z.get('allocated_usar_squads', d.get('usar_teams_needed', 0))}",
            "Ambulances": f"🚑 {z.get('allocated_ambulances', d.get('paramedic_units_needed', 0))}",
            "Tactical Action Directive": z.get("reason", d.get("action_directive", "Standard sweep")),
        })
    df_zones = pd.DataFrame(pz_rows)
    st.dataframe(df_zones, width="stretch", hide_index=True)

    st.markdown("---")


def render_tactical_incident_action_plan(
    priority_zones: list[dict[str, Any]],
    metrics: dict[str, Any],
    processing_time_ms: float | None = None,
) -> None:
    """Render high-priority tactical rescue directives answering Where to Focus and What to Do."""
    if not priority_zones:
        return

    top_zone = priority_zones[0]
    deployments = calculate_tactical_resource_allocation(priority_zones)
    top_deploy = deployments[0] if deployments else {}

    total_usar = sum(d.get("usar_teams_needed", 0) for d in deployments)
    total_med = sum(d.get("paramedic_units_needed", 0) for d in deployments)
    total_debris = sum(d.get("debris_clearance_units", 0) for d in deployments)
    total_uav = sum(d.get("uav_secondary_recon", 0) for d in deployments)

    st.markdown("## 🚨 Tactical Incident Action Plan (Where to Focus & What to Do)")
    st.caption("AI Decision-Support System: Translating drone reconnaissance imagery into immediate search & rescue priorities.")

    # 1. High-Impact Target Alert Card (WHERE TO FOCUS FIRST)
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, #1e1b4b 0%, #311042 50%, #7f1d1d 100%);
                    border: 2px solid #ef4444; border-radius: 10px; padding: 22px 26px;
                    margin-bottom: 22px; color: white; box-shadow: 0 10px 25px -5px rgba(239, 68, 68, 0.25);">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 16px;">
                <div style="flex: 1; min-width: 280px;">
                    <div style="display: inline-block; background: #ef4444; color: white; padding: 4px 12px; border-radius: 4px; font-weight: 800; font-size: 0.84rem; letter-spacing: 1.2px; text-transform: uppercase;">
                        🚨 PRIMARY RESCUE FOCUS: PRIORITY RANK #1
                    </div>
                    <div style="font-size: 1.85rem; font-weight: 900; margin-top: 8px; letter-spacing: 0.5px;">
                        SECTOR: PRIORITY ZONE {top_zone['zone_id']}
                    </div>
                    <div style="font-size: 1.05rem; color: #fecaca; margin-top: 6px; font-weight: 500;">
                        <b>Urgency Level:</b> <span style="color: #fef08a; font-weight: 800;">CRITICAL (IMMEDIATE LIFE EXTRACTION)</span> · Priority Score: <b>{top_zone['priority_score']:.3f}</b>
                    </div>
                    <div style="font-size: 0.95rem; color: #e2e8f0; margin-top: 6px;">
                        <b>Severe Impact:</b> {top_zone['destroyed_buildings']} Destroyed Structures · {top_zone['major_damage_buildings']} Major Damage Structures
                    </div>
                    <div style="font-size: 0.92rem; color: #cbd5e1; margin-top: 8px; border-left: 3px solid #f59e0b; padding-left: 10px;">
                        <b>Tactical Rationale:</b> {top_zone.get('reason', 'Highest density of catastrophic building collapses. Severe occupant entrapment hazard.')}
                    </div>
                </div>
                <div style="background: rgba(15, 23, 42, 0.75); border: 1.5px solid rgba(239, 68, 68, 0.6); border-radius: 8px; padding: 16px 20px; min-width: 260px;">
                    <div style="font-size: 0.80rem; color: #f87171; text-transform: uppercase; font-weight: 800; letter-spacing: 1px;">
                        Immediate Resource Dispatch
                    </div>
                    <div style="font-size: 1.25rem; font-weight: 800; color: #ffffff; margin-top: 6px;">
                        🪖 {top_deploy.get('usar_teams_needed', 1)} USAR Squad(s)
                    </div>
                    <div style="font-size: 1.15rem; font-weight: 700; color: #fecaca; margin-top: 4px;">
                        🚑 {top_deploy.get('paramedic_units_needed', 1)} Paramedic Ambulance(s)
                    </div>
                    <div style="font-size: 0.90rem; color: #cbd5e1; margin-top: 4px;">
                        🚜 {top_deploy.get('debris_clearance_units', 1)} Heavy Clearance Unit(s)
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Sequential Operational Command Directives (WHAT TO DO)
    st.markdown("#### 📋 Operational Command Directives for Field Incident Commanders")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            f"""
            <div style="background: #ffffff; border-left: 5px solid #ef4444; border-radius: 6px; padding: 14px 18px; margin-bottom: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); border-top: 1px solid #e2e8f0; border-right: 1px solid #e2e8f0; border-bottom: 1px solid #e2e8f0;">
                <span style="background: #fee2e2; color: #991b1b; padding: 2px 8px; border-radius: 4px; font-weight: 800; font-size: 0.75rem;">
                    PHASE 1 · 0-2 HOURS
                </span>
                <div style="font-weight: 700; font-size: 1.05rem; color: #0f172a; margin-top: 6px;">
                    1. Primary Life Extraction & Search Sweep
                </div>
                <div style="font-size: 0.88rem; color: #334155; margin-top: 4px;">
                    Deploy <b>{top_deploy.get('usar_teams_needed', 1)} Urban Search & Rescue Squad(s)</b> with acoustic listening probes and search canines directly to <b>Sector {top_zone['zone_id']}</b>. Prioritize confirmed collapsed voids for live survivor recovery.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div style="background: #ffffff; border-left: 5px solid #f59e0b; border-radius: 6px; padding: 14px 18px; margin-bottom: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); border-top: 1px solid #e2e8f0; border-right: 1px solid #e2e8f0; border-bottom: 1px solid #e2e8f0;">
                <span style="background: #fef3c7; color: #92400e; padding: 2px 8px; border-radius: 4px; font-weight: 800; font-size: 0.75rem;">
                    PHASE 2 · 2-4 HOURS
                </span>
                <div style="font-weight: 700; font-size: 1.05rem; color: #0f172a; margin-top: 6px;">
                    2. Trauma Triage & Emergency Medical Evacuation
                </div>
                <div style="font-size: 0.88rem; color: #334155; margin-top: 4px;">
                    Establish Field Casualty Collection Point (CCP) at adjacent perimeter. Stage <b>{top_deploy.get('paramedic_units_needed', 1)} Ambulance Unit(s)</b> to handle immediate trauma stabilization and regional hospital transport.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div style="background: #ffffff; border-left: 5px solid #0284c7; border-radius: 6px; padding: 14px 18px; margin-bottom: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); border-top: 1px solid #e2e8f0; border-right: 1px solid #e2e8f0; border-bottom: 1px solid #e2e8f0;">
                <span style="background: #e0f2fe; color: #075985; padding: 2px 8px; border-radius: 4px; font-weight: 800; font-size: 0.75rem;">
                    PHASE 3 · 4-8 HOURS
                </span>
                <div style="font-weight: 700; font-size: 1.05rem; color: #0f172a; margin-top: 6px;">
                    3. Arterial Roadway & Heavy Rubble Clearance
                </div>
                <div style="font-size: 0.88rem; color: #334155; margin-top: 4px;">
                    Dispatch <b>{top_deploy.get('debris_clearance_units', 1)} Heavy Debris Clearance Squad(s)</b> to remove collapsed roof trusses and roadway blockages to maintain an open relief corridor into Sector {top_zone['zone_id']}.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div style="background: #ffffff; border-left: 5px solid #10b981; border-radius: 6px; padding: 14px 18px; margin-bottom: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); border-top: 1px solid #e2e8f0; border-right: 1px solid #e2e8f0; border-bottom: 1px solid #e2e8f0;">
                <span style="background: #d1fae5; color: #065f46; padding: 2px 8px; border-radius: 4px; font-weight: 800; font-size: 0.75rem;">
                    PHASE 4 · 8-12 HOURS
                </span>
                <div style="font-weight: 700; font-size: 1.05rem; color: #0f172a; margin-top: 6px;">
                    4. Continuous Drone Overwatch & Hazard Tracking
                </div>
                <div style="font-size: 0.88rem; color: #334155; margin-top: 4px;">
                    Maintain tactical drone reconnaissance over secondary priority sectors. Monitor for ruptured gas lines, live electrical lines, and compromised walls threatening first responders.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 3. Overall Sector Resource Footprint
    res_c1, res_c2, res_c3, res_c4 = st.columns(4)
    with res_c1:
        st.metric("Total USAR Squads", f"🪖 {total_usar} Teams", help="Total Urban Search & Rescue teams needed across all sectors.")
    with res_c2:
        st.metric("Paramedic Ambulances", f"🚑 {total_med} Units", help="Emergency medical ambulances required for casualties.")
    with res_c3:
        st.metric("Debris Clearance", f"🚜 {total_debris} Squads", help="Heavy mechanical clearing units needed for roads.")
    with res_c4:
        st.metric("Secondary UAV Sweeps", f"🚁 {total_uav} Drones", help="Targeted drone flights required for secondary validation.")

    # 4. Copyable Formal Incident Action Plan (IAP)
    with st.expander("📄 View Formal Incident Action Plan (IAP) Radio Dispatch Briefing", expanded=False):
        iap_text = (
            "================================================================================\n"
            "INCIDENT ACTION PLAN (IAP) · TACTICAL SEARCH & RESCUE DISPATCH\n"
            "ORGANIZATION: Disaster Emergency Operations Center (EOC)\n"
            f"PRIMARY RESCUE SECTOR: Sector {top_zone['zone_id']} (Priority Rank #1)\n"
            f"PRIORITY SCORE: {top_zone['priority_score']:.4f} (Severity: CRITICAL)\n"
            f"ESTIMATED COLLAPSED STRUCTURES: {top_zone['destroyed_buildings']} Destroyed, {top_zone['major_damage_buildings']} Major Damage\n"
            "--------------------------------------------------------------------------------\n"
            "OPERATIONAL DIRECTIVES:\n"
            f"1. USAR SQUADS: Dispatch {top_deploy.get('usar_teams_needed', 1)} Heavy Rescue Squad(s) to Sector {top_zone['zone_id']} for immediate survivor search.\n"
            f"2. MEDICAL EVAC: Mobilize {top_deploy.get('paramedic_units_needed', 1)} Ambulance(s) to perimeter Casualty Collection Point.\n"
            f"3. INGRESS ACCESS: Clear main access road to Sector {top_zone['zone_id']} using debris squad.\n"
            "4. SAFETY PERIMETER: Establish 100m perimeter around structurally compromised walls.\n"
            "================================================================================"
        )
        st.code(iap_text, language="text")

    st.markdown("---")


def render_building_inventory_table(df_results: pd.DataFrame) -> None:
    """Render tabular building inventory with sorting, search, and class details."""
    st.markdown("### 📋 Detected Building Inventory & Damage Classifications")
    st.caption("Tabular building footprint inventory with model confidence scores and bounding coordinates.")

    if not df_results.empty:
        st.dataframe(
            df_results,
            width="stretch",
            hide_index=True,
        )
    else:
        st.warning("No building features to display.")

    st.markdown("---")
