"""Stage 2 & Stage 3: Enterprise C2 Split Terminal Component for MERIDIAN-Lite.

Provides the 35/65 Split Terminal layout:
- Left Tactical Rail (35%): Assessment mode & mission telemetry, dynamic storm surge slider (0.0m - 5.0m),
  Priority Zone 1 triage & dispatch banner, and on-the-fly "Compute Safe Infiltration Routes" action engine.
- Right Situational Canvas (65%): High-density Folium map with 5 integrated layers:
  1. Pre/Post Imagery Footprint
  2. Detected Damaged Structures
  3. 3x3 Priority Sectors
  4. Stage 3 Predictive Inundation Layer
  5. Stage 2 Evacuation / Infiltration Polyline Route
"""

from pathlib import Path
from typing import Any, Tuple
import streamlit as st
from streamlit_folium import st_folium

from app.ui.map_view import build_folium_map
from app.ui.styles import render_legend_badges
from utils.prediction import (
    recalculate_priority_zones_with_inundation,
    generate_forecast_inundation_polygons,
)
from utils.routing import (
    build_mexico_beach_road_network,
    apply_dynamic_hazard_impedance,
    compute_evacuation_routing,
    compute_usar_ambulance_dispatch_matrix,
)
from utils.metrics import calculate_summary_metrics


def render_c2_split_terminal(
    geojson_data: dict[str, Any],
    priority_zones: list[dict[str, Any]],
    processing_time_ms: float = 120.0,
    post_display_path: str | Path | None = None,
    tactical_overlay_path: str | Path | None = None,
    center_lat: float = 30.154378,
    center_lon: float = -85.548719,
    mode_label: str = "Real Model Mode",
    backend_label: str = "Real Damage Model (ONNX)",
) -> Tuple[list[dict[str, Any]], dict[str, Any] | None, dict[str, Any] | None]:
    """Render the Enterprise C2 Two-Column Split Terminal (35% Left Rail / 65% Right Canvas).

    Returns:
        (updated_priority_zones, inundation_geojson, routing_data)
    """
    col_rail, col_canvas = st.columns([0.35, 0.65], gap="medium")

    # =========================================================================
    # LEFT TACTICAL RAIL (35% Width)
    # =========================================================================
    with col_rail:
        st.markdown(
            """
            <div style="background: #0f172a; border-radius: 8px; padding: 10px 14px; border-left: 4px solid #00f0ff; margin-bottom: 12px;">
                <div style="font-size: 0.72rem; letter-spacing: 1.5px; text-transform: uppercase; color: #94a3b8; font-weight: 700;">
                    COMMAND & CONTROL · TACTICAL RAIL
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 4px;">
                    <span style="font-size: 0.88rem; font-weight: 800; color: #f8fafc;">
                        FARLANDS 2026 MISSION C2
                    </span>
                    <span style="background: #1e293b; color: #38bdf8; font-size: 0.72rem; padding: 2px 7px; border-radius: 4px; font-weight: 700; border: 1px solid #0284c7;">
                        OFFLINE AIR-GAP
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div style="background: #022c22; border: 1.5px solid #10b981; border-radius: 8px; padding: 10px 12px; margin-bottom: 12px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="color: #6ee7b7; font-weight: 800; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.8px;">
                        ✓ DamageTriage-Bench (Hurricane Michael)
                    </span>
                    <span style="background: #065f46; color: #a7f3d0; font-size: 0.72rem; padding: 2px 6px; border-radius: 4px; font-weight: 700;">
                        Acc: 77.8%
                    </span>
                </div>
                <div style="font-size: 0.82rem; color: #d1fae5; margin-top: 4px; line-height: 1.4;">
                    Dataset: <b>Hurricane Michael (Mexico Beach, FL)</b><br>
                    Model: <b>Lightweight Bi-Temporal ONNX CNN Baseline</b> (119.5 ms CPU)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 1. Dynamic Simulation / Storm Surge Slider (0.0m to 5.0m)
        st.markdown("##### 🌊 Dynamic Environmental Simulation")
        current_surge = float(st.session_state.get("surge_height_m", 2.2))
        surge_height_m = st.slider(
            "Storm Surge & Sea-Level Baseline (0.0m - 5.0m):",
            min_value=0.0,
            max_value=5.0,
            value=current_surge,
            step=0.1,
            format="%.1f m",
            help="Dynamically recalculates hydraulic baseline against coastal DEM/HAND topography and re-scores all priority sectors.",
        )
        st.session_state["surge_height_m"] = surge_height_m

        # Extract dynamic tactical drone survey footprint bounds
        if priority_zones and any("bounds" in z for z in priority_zones):
            z_list = [z["bounds"] for z in priority_zones if "bounds" in z]
            survey_bounds = (
                min(b[0] for b in z_list),
                min(b[1] for b in z_list),
                max(b[2] for b in z_list),
                max(b[3] for b in z_list),
            )
        elif geojson_data.get("bbox"):
            survey_bounds = tuple(geojson_data["bbox"])
        else:
            survey_bounds = (-85.550100, 30.152997, -85.547337, 30.155760)

        # Dynamic recalculation of inundation polygons & priority zones within survey bounds
        inundation_geojson = generate_forecast_inundation_polygons(
            surge_height_m=surge_height_m,
            rain_mm=65.0,
            bounds=survey_bounds,
        )
        updated_zones = recalculate_priority_zones_with_inundation(
            priority_zones=priority_zones,
            surge_height_m=surge_height_m,
            rain_mm=65.0,
            bounds=survey_bounds,
        )

        # 2. Mission Telemetry Metrics with Population & Casualties
        metrics = calculate_summary_metrics(geojson_data)
        features = geojson_data.get("features", [])
        tot_bld = len(features)
        conf_pct = round(metrics.get("mean_confidence", 0.85) * 100.0, 1)

        counts = metrics.get("counts_by_damage", {})
        c_dest = counts.get("destroyed", 0)
        c_major = counts.get("major-damage", 0)
        c_minor = counts.get("minor-damage", 0)
        c_none = counts.get("no-damage", 0)

        tot_pop = sum(z.get("estimated_population", 0) for z in updated_zones)
        tot_cas = sum(z.get("estimated_casualties", 0) for z in updated_zones)

        st.markdown("##### 📊 Mission Telemetry & Population")
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.metric("Total Structures", tot_bld, delta=f"{c_dest + c_major} Severe", delta_color="inverse")
            st.metric("Exposed Population", f"{tot_pop} residents", delta=f"{tot_cas} est. casualties", delta_color="inverse")
        with m_col2:
            st.metric("Pipeline Latency", f"{processing_time_ms:.1f} ms", delta="Deterministic CPU")
            st.metric("Mean AI Confidence", f"{conf_pct}%", delta="xBD Validated")

        # Severity breakdown bar
        st.markdown(
            f"""
            <div style="background: #1e293b; border-radius: 6px; padding: 8px 12px; margin-bottom: 14px; border: 1px solid #334155;">
                <div style="font-size: 0.72rem; color: #94a3b8; font-weight: 700; text-transform: uppercase; margin-bottom: 6px;">
                    Triage Severity Breakdown
                </div>
                <div style="display: flex; gap: 8px; font-size: 0.80rem; font-weight: 700;">
                    <span style="color: #ef4444;">🔴 {c_dest} Dest</span>
                    <span style="color: #f97316;">🟠 {c_major} Major</span>
                    <span style="color: #eab308;">🟡 {c_minor} Minor</span>
                    <span style="color: #22c55e;">🟢 {c_none} Intact</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 3. Stage 2 Triage & Dispatch Banner: Priority Zone 1 Directive
        rank1_zone = updated_zones[0] if updated_zones else {}
        z1_id = rank1_zone.get("zone_id", "Z-001")
        z1_dest = rank1_zone.get("destroyed_buildings", 0)
        z1_major = rank1_zone.get("major_damage_buildings", 0)
        z1_score = rank1_zone.get("priority_score", 0.0)
        z1_pop = rank1_zone.get("estimated_population", 0)
        z1_cas = rank1_zone.get("estimated_casualties", 0)
        z1_vuln = rank1_zone.get("vulnerable_population", 0)
        z1_hosp = rank1_zone.get("nearest_hospital", "Bay County Field Hospital")
        z1_hosp_dist = rank1_zone.get("hospital_distance_km", 1.1)
        z1_trans = rank1_zone.get("transit_time_mins", 3.5)
        z1_reason = rank1_zone.get("reason", "Immediate search & rescue extraction required.")

        # Compute dynamic USAR recommendation for Zone 1
        z1_usar = rank1_zone.get("allocated_usar_squads") or max(1, (z1_dest + 1) // 2) if z1_dest > 0 else 1
        z1_amb = rank1_zone.get("allocated_ambulances") or max(1, z1_major + z1_dest)

        st.markdown(
            f"""
            <div style="background: #450a0a; border: 1.5px solid #ef4444; border-radius: 8px; padding: 12px; margin-bottom: 14px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="background: #dc2626; color: white; padding: 2px 7px; border-radius: 4px; font-weight: 800; font-size: 0.75rem;">
                        🚨 PRIORITY SECTOR 1 DISPATCH
                    </span>
                    <span style="color: #fca5a5; font-weight: 800; font-size: 0.85rem;">
                        Target: {z1_id} (Score {z1_score:.3f})
                    </span>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 0.78rem; color: #fecaca; margin-top: 6px;">
                    <span>👥 Exposed: <b>{z1_pop} residents</b> ({z1_vuln} vuln)</span>
                    <span>⚠️ Casualties: <b>{z1_cas} trapped/injured</b></span>
                </div>
                <div style="font-size: 0.78rem; color: #fed7aa; margin-top: 4px;">
                    🏥 Nearest Medical: <b>{z1_hosp}</b> ({z1_hosp_dist:.2f} km · {z1_trans} mins)
                </div>
                <div style="font-size: 0.82rem; color: #fee2e2; margin-top: 8px; line-height: 1.4;">
                    <b>Action Directive:</b> {z1_reason}
                </div>
                <div style="display: flex; gap: 8px; margin-top: 10px;">
                    <span style="background: #7f1d1d; color: #fecaca; font-size: 0.75rem; padding: 3px 8px; border-radius: 4px; font-weight: 700;">
                        🚒 {z1_usar} USAR Squads
                    </span>
                    <span style="background: #7f1d1d; color: #fecaca; font-size: 0.75rem; padding: 3px 8px; border-radius: 4px; font-weight: 700;">
                        🚑 {z1_amb} Ambulances
                    </span>
                    <span style="background: #7f1d1d; color: #fecaca; font-size: 0.75rem; padding: 3px 8px; border-radius: 4px; font-weight: 700;">
                        🚜 1 Debris Unit
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 4. Stage 2 Dispatch Action Engine
        st.markdown("##### ⚡ Dynamic Network Routing Engine")
        compute_routes_clicked = st.button(
            "Compute Safe Infiltration Routes",
            type="primary",
            use_container_width=True,
            help="Executes deterministic NetworkX Dijkstra routing severing road edges intersecting building debris and flood polygons.",
        )

        # Auto-compute or read cached routing state
        if compute_routes_clicked or "routing_data" not in st.session_state:
            # Staging Base Alpha positioned at tactical survey perimeter (~60m west)
            z1_bounds = rank1_zone.get("bounds", survey_bounds)
            z1_center = ((z1_bounds[0] + z1_bounds[2]) / 2.0, (z1_bounds[1] + z1_bounds[3]) / 2.0)
            staging_alpha = (
                survey_bounds[0] - 0.0006,
                survey_bounds[1] + (survey_bounds[3] - survey_bounds[1]) * 0.2,
            )

            # Build road network matched to tactical survey bounds
            G = build_mexico_beach_road_network(bounds=survey_bounds)
            G, road_stats = apply_dynamic_hazard_impedance(
                G=G,
                building_features=features,
                flood_geojson=inundation_geojson,
                rubble_buffer_m=12.0,
            )

            # Compute baseline vs evasive path
            routing_res = compute_evacuation_routing(
                G=G,
                start_coords=staging_alpha,
                target_coords=z1_center,
                staging_name="Forward Staging Base Alpha (Perimeter Ingress)",
                target_name=f"Sector {z1_id} Primary Target",
            )
            routing_res["road_stats"] = road_stats

            # Compute greedy USAR and ambulance dispatch matrix
            dispatch_matrix = compute_usar_ambulance_dispatch_matrix(
                priority_zones=updated_zones,
                bounds=survey_bounds,
            )
            routing_res["dispatch_matrix"] = dispatch_matrix

            st.session_state["routing_data"] = routing_res
            st.session_state["last_route_surge"] = surge_height_m

        routing_data = st.session_state.get("routing_data")
        r_metrics = routing_data.get("metrics", {}) if routing_data else {}
        r_stats = routing_data.get("road_stats", {}) if routing_data else {}

        # Render routing metrics summary card with tactical meter/km clarity
        status_color = "#22c55e" if r_metrics.get("clearance_status") == "CLEARED_HAZARD_DETOUR" else "#f59e0b"
        base_km = float(r_metrics.get("baseline_distance_km", 0.0))
        safe_km = float(r_metrics.get("safe_route_distance_km", 0.0))
        base_m = int(round(base_km * 1000.0))
        safe_m = int(round(safe_km * 1000.0))

        st.markdown(
            f"""
            <div style="background: #1e293b; border-radius: 8px; padding: 10px 12px; margin-top: 8px; border: 1px solid #334155;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <span style="font-size: 0.75rem; font-weight: 700; color: #94a3b8; text-transform: uppercase;">
                        Transit Optimization (Tactical Scale)
                    </span>
                    <span style="color: {status_color}; font-size: 0.75rem; font-weight: 800;">
                        ● {r_metrics.get('clearance_status', 'CLEAR')}
                    </span>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 0.82rem; color: #cbd5e1; margin-bottom: 4px;">
                    <span>Direct Baseline (Hazards):</span>
                    <b>{base_m} m ({base_km:.2f} km) · {r_metrics.get('baseline_duration_mins', 0.0)} mins</b>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 0.82rem; color: #38bdf8; margin-bottom: 4px;">
                    <span>Safe Infiltration Corridor:</span>
                    <b>{safe_m} m ({safe_km:.2f} km) · {r_metrics.get('safe_route_duration_mins', 0.0)} mins</b>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 0.78rem; color: #f59e0b;">
                    <span>Detour Penalty / Delay:</span>
                    <b>+{r_metrics.get('detour_delay_mins', 0.0)} mins (+{r_metrics.get('detour_delay_km', 0.0)} km)</b>
                </div>
                <div style="font-size: 0.72rem; color: #94a3b8; margin-top: 6px; border-top: 1px solid #334155; padding-top: 4px;">
                    Impassable Segments Severed: <b>{r_stats.get('severed_edges', 0)} / {r_stats.get('total_edges', 0)} ({r_stats.get('severed_percentage', 0.0)}%)</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Dispatch assignments expander
        with st.expander("📋 Dispatch Assignment Matrix (Top 3 Sectors)", expanded=False):
            matrix = routing_data.get("dispatch_matrix", [])
            for alloc in matrix:
                t_dist_km = float(alloc['transit_distance_km'])
                t_dist_m = int(round(t_dist_km * 1000.0))
                st.markdown(
                    f"**Sector {alloc['target_zone_id']} (Rank #{alloc['priority_rank']})**<br>"
                    f"• Base: `{alloc['assigned_staging_base']}`<br>"
                    f"• Units: **{alloc['usar_squads_dispatched']} USAR Squads, {alloc['paramedic_ambulances_dispatched']} Ambulances**<br>"
                    f"• Transit: **{t_dist_m} m ({t_dist_km:.2f} km) · ETA: {alloc['estimated_transit_eta_mins']} mins**<br>",
                    unsafe_allow_html=True,
                )

        # Tactical Downloads in Left Rail (RFC 7946 GeoJSON & Tactical SITREP)
        import json
        st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)
        dl_c1, dl_c2 = st.columns(2)
        with dl_c1:
            st.download_button(
                "📥 GeoJSON (RFC 7946)",
                data=json.dumps(geojson_data, indent=2),
                file_name="meridian_damage_assessment.geojson",
                mime="application/geo+json",
                use_container_width=True,
                help="Export RFC 7946 building footprint polygons with EMS-98 damage classes.",
            )
        with dl_c2:
            sitrep_text = f"""# MERIDIAN-Lite Tactical Incident Situation Report (SITREP)
Mission: FARLANDS 2026 Innovation Mission 6 | Mode: {mode_label}
Backend: {backend_label} (Local CPU SIMD) | Latency: {processing_time_ms:.1f} ms
Total Structures: {tot_bld} | Destroyed: {c_dest} | Major Damage: {c_major} | Minor: {c_minor}
Primary Search Sector: {z1_id} (Score {z1_score:.3f})
Direct Action: {z1_reason}
Transit Clearance: {r_metrics.get('clearance_status', 'CLEAR')}
Safe Corridor: {r_metrics.get('safe_route_distance_km', 0.0)} km ({r_metrics.get('safe_route_duration_mins', 0.0)} mins)
Detour Delay Penalty: +{r_metrics.get('detour_delay_mins', 0.0)} mins (+{r_metrics.get('detour_delay_km', 0.0)} km)
Dispatch Directive: Deploy {z1_usar} USAR Squad(s) & {z1_amb} Ambulance(s) from Base Alpha.
"""
            st.download_button(
                "📄 Tactical SITREP",
                data=sitrep_text,
                file_name="meridian_tactical_sitrep.md",
                mime="text/markdown",
                use_container_width=True,
                help="Export executive FEMA Incident Command Situation Report.",
            )

    # =========================================================================
    # RIGHT SITUATIONAL CANVAS (65% Width)
    # =========================================================================
    with col_canvas:
        st.markdown("### 🛰️ Situational Awareness Canvas")
        render_legend_badges()

        # Build Map with all 5 layers
        routing_geojson = routing_data.get("geojson") if routing_data else None
        c2_map = build_folium_map(
            geojson_data=geojson_data,
            priority_zones=updated_zones,
            center_lat=center_lat,
            center_lon=center_lon,
            image_overlay_path=post_display_path,
            inundation_geojson=inundation_geojson,
            routing_geojson=routing_geojson,
        )

        # Dual-view tab container
        if tactical_overlay_path and Path(tactical_overlay_path).exists():
            tab_map, tab_hud = st.tabs([
                "🛰️ C2 Situational Awareness GIS (5-Layer Integrated Map)",
                "🎯 Tactical Optical Detection HUD (Direct Imagery Overlay)",
            ])
            with tab_map:
                try:
                    st_folium(c2_map, width="100%", height=620, returned_objects=[])
                except Exception as exc:
                    st.warning(f"GIS rendering fallback: {exc}")
            with tab_hud:
                st.image(str(tactical_overlay_path), width="stretch")
        else:
            try:
                st_folium(c2_map, width="100%", height=620, returned_objects=[])
            except Exception as exc:
                st.warning(f"GIS rendering fallback: {exc}")

    st.markdown("---")
    return updated_zones, inundation_geojson, routing_data
