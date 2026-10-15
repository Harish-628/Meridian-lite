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

    pz_c1, pz_c2, pz_c3, pz_c4 = st.columns(4)
    with pz_c1:
        st.metric("Top Priority Zone", top_zone["zone_id"], help="Highest-need triage sector.")
    with pz_c2:
        st.metric(
            "Priority Score",
            f"{top_zone['priority_score']:.3f}",
            delta="Rank #1",
            help="Normalized urgency score [0.0 - 1.0].",
        )
    with pz_c3:
        st.metric(
            "Severe Structures",
            f"{severe_count}",
            help="Destroyed + major damage count in top zone.",
        )
    with pz_c4:
        st.metric(
            "Total Affected",
            f"{top_zone['damaged_buildings']} / {top_zone['total_buildings']}",
            help="Damaged buildings vs total in top zone.",
        )

    # Format zones table with rescue allocations
    pz_rows = []
    for z in priority_zones:
        # Find matching deployment
        d = next((item for item in deployments if item["zone_id"] == z["zone_id"]), {})
        pz_rows.append({
            "Priority Rank": f"Rank #{z['priority_rank']}",
            "Zone ID": z["zone_id"],
            "Priority Score": f"{z['priority_score']:.4f}",
            "Structures": z["total_buildings"],
            "Damaged": z["damaged_buildings"],
            "Severe": z["destroyed_buildings"] + z["major_damage_buildings"],
            "USAR Teams": f"🪖 {d.get('usar_teams_needed', 0)}",
            "Ambulances": f"🚑 {d.get('paramedic_units_needed', 0)}",
            "Urgency": d.get("dispatch_urgency", "Normal"),
            "Tactical Action": d.get("action_directive", z.get("reason", "Standard survey")),
        })
    df_zones = pd.DataFrame(pz_rows)
    st.dataframe(df_zones, width="stretch", hide_index=True)

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
