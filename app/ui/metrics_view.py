"""
Tactical Metrics & Priority-Zone Intelligence Views for MERIDIAN-Lite.
Provides metric cards, damage class breakdowns, ranked priority zone tables, and inventory dataframes.
"""

from typing import Any
import pandas as pd
import streamlit as st


def render_tactical_summary(metrics: dict[str, Any], processing_time_ms: float | None = None) -> None:
    """Render top-level tactical assessment summary cards and breakdown counts."""
    st.markdown("### 📊 Tactical Assessment Summary")

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric(
            label="Total Buildings Detected",
            value=metrics["total_buildings"],
            help="Total building polygons extracted in target reconnaissance sector.",
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
        proc_time = processing_time_ms if processing_time_ms is not None else 120.0
        st.markdown(f"**⏱️ Processing:** {proc_time} ms")

    st.markdown("---")


def render_priority_zones_section(priority_zones: list[dict[str, Any]]) -> None:
    """Render Stage 2 priority-zone scoring metric cards and ranked allocation table."""
    st.markdown("### 🎯 Stage 2: Priority-Zone Scoring & Resource Allocation")
    st.caption(
        "Multi-factor decision-support ranking derived from: "
        "0.40 × Damage Density + 0.25 × Severe Damage Ratio + 0.20 × Total Affected Structures + 0.15 × Confidence Weight."
    )

    if not priority_zones:
        st.info("No priority zones available for current view.")
        st.markdown("---")
        return

    top_zone = priority_zones[0]
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
            f"{top_zone['destroyed_buildings'] + top_zone['major_damage_buildings']}",
            help="Destroyed + major damage count in top zone.",
        )
    with pz_c4:
        st.metric(
            "Total Affected",
            f"{top_zone['damaged_buildings']} / {top_zone['total_buildings']}",
            help="Damaged buildings vs total in top zone.",
        )

    # Format zones table
    pz_rows = []
    for z in priority_zones:
        pz_rows.append({
            "Priority Rank": f"Rank #{z['priority_rank']}",
            "Zone ID": z["zone_id"],
            "Priority Score": f"{z['priority_score']:.4f}",
            "Total Structures": z["total_buildings"],
            "Damaged Structures": z["damaged_buildings"],
            "Destroyed": z["destroyed_buildings"],
            "Major Damage": z["major_damage_buildings"],
            "Mean Confidence": f"{z['mean_confidence']:.1%}",
            "Triage Rationale": z["reason"],
        })
    df_zones = pd.DataFrame(pz_rows)
    st.dataframe(df_zones, use_container_width=True, hide_index=True)

    st.markdown("---")


def render_building_inventory_table(df_results: pd.DataFrame) -> None:
    """Render tabular building inventory with sorting, search, and class details."""
    st.markdown("### 📋 Detected Building Inventory & Damage Classifications")
    st.caption("Tabular building footprint inventory with model confidence scores and bounding coordinates.")

    if not df_results.empty:
        st.dataframe(
            df_results,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning("No building features to display.")

    st.markdown("---")
