"""
Interactive Geospatial Map View for MERIDIAN-Lite.
Builds and renders Folium GIS maps with building footprints, damage styling,
priority zone overlays, satellite basemaps, and tactical optical overlays.
"""

from pathlib import Path
from typing import Any
import folium
from streamlit_folium import st_folium
import streamlit as st

from app.ui.styles import render_legend_badges
from utils.geojson_utils import DAMAGE_HEX_COLORS, get_bounds


def build_folium_map(
    geojson_data: dict[str, Any],
    priority_zones: list[dict[str, Any]] | None = None,
    center_lat: float = 30.1750,
    center_lon: float = -85.5500,
    zoom_start: int = 15,
    is_drone_stream: bool = False,
    image_overlay_path: str | Path | None = None,
    inundation_geojson: dict[str, Any] | None = None,
    routing_geojson: dict[str, Any] | None = None,
) -> folium.Map:
    """Construct an interactive Folium map populated with building polygons, priority zones,
    predictive inundation layers, dynamic evacuation/infiltration routes, and aerial imagery overlay.
    """
    features = geojson_data.get("features", []) if isinstance(geojson_data, dict) else []
    all_lats: list[float] = []
    all_lons: list[float] = []

    for f in features:
        geom = f.get("geometry", {})
        coords = geom.get("coordinates", [])
        if geom.get("type") == "Polygon" and coords:
            for pt in coords[0]:
                all_lons.append(float(pt[0]))
                all_lats.append(float(pt[1]))
        elif geom.get("type") == "MultiPolygon" and coords:
            for poly in coords:
                if poly:
                    for pt in poly[0]:
                        all_lons.append(float(pt[0]))
                        all_lats.append(float(pt[1]))

    if all_lats and all_lons:
        c_lat = float(sum(all_lats) / len(all_lats))
        c_lon = float(sum(all_lons) / len(all_lons))
    else:
        c_lat, c_lon = center_lat, center_lon

    # Initialize map with Esri Satellite as primary basemap centered on Mexico Beach
    m = folium.Map(
        location=[c_lat, c_lon],
        zoom_start=zoom_start,
        tiles=None,
        control_scale=True,
    )

    # 1. Add Satellite Basemap (Esri World Imagery)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="🛰️ Satellite Earth Imagery (Esri)",
        overlay=False,
        control=True,
    ).add_to(m)

    # 2. Add OpenStreetMap Basemap (Streets & Infrastructure)
    folium.TileLayer(
        tiles="OpenStreetMap",
        name="🗺️ OpenStreetMap (Streets)",
        overlay=False,
        control=True,
    ).add_to(m)

    # 3. Layer: 🏥 Critical Facilities & Emergency Hubs
    from utils.priority import CRITICAL_FACILITIES
    fac_fg = folium.FeatureGroup(name="🏥 Critical Facilities & Emergency Hubs", show=True)
    for fac in CRITICAL_FACILITIES:
        folium.Marker(
            location=[fac["lat"], fac["lon"]],
            icon=folium.Icon(color=fac.get("marker_color", "red"), icon=fac.get("icon", "plus"), prefix="fa"),
            tooltip=f"🏥 {fac['name']} ({fac['status']})",
            popup=(
                f"<div style='min-width: 220px; font-family: sans-serif;'>"
                f"<b style='color: #0f172a; font-size: 1.05rem;'>{fac['name']}</b><br>"
                f"<span style='background: #e0f2fe; color: #0369a1; padding: 2px 6px; border-radius: 4px; font-weight: 700; font-size: 0.75rem; text-transform: uppercase;'>"
                f"{fac['facility_type']}</span><br><br>"
                f"<b>Capacity:</b> {fac['capacity']}<br>"
                f"<b>Status:</b> {fac['status']}<br>"
                f"<b>Critical Weight:</b> {fac['critical_weight']}<br>"
                f"<b>Coordinates:</b> {fac['lat']:.5f}° N, {fac['lon']:.5f}° W"
                f"</div>"
            ),
        ).add_to(fac_fg)
    fac_fg.add_to(m)

    # 4. Layer 1: Pre/Post Imagery Footprint Overlay
    if image_overlay_path and Path(image_overlay_path).exists():
        bounds = get_bounds(geojson_data)
        if bounds != [[0.0, 0.0], [0.0, 0.0]]:
            try:
                folium.raster_layers.ImageOverlay(
                    name="📸 1. Pre/Post Drone Imagery Footprint",
                    image=str(image_overlay_path),
                    bounds=bounds,
                    opacity=0.85,
                    interactive=True,
                    zindex=2,
                ).add_to(m)
            except Exception:
                pass

    # 5. Layer 4: Stage 3 Predictive Inundation Layer
    if inundation_geojson and inundation_geojson.get("features"):
        inund_fg = folium.FeatureGroup(name="🌊 4. Stage 3 Predictive Inundation Layer", show=True)
        for feat in inundation_geojson["features"]:
            geom = feat.get("geometry", {})
            props = feat.get("properties", {})
            color = props.get("fill_color", "#00b4d8")
            h_label = props.get("horizon", "Forecast")
            eff_s = props.get("effective_surge_m", 0.0)
            desc = props.get("description", "Inundation Zone")

            folium.GeoJson(
                feat,
                style_function=lambda x, c=color: {
                    "fillColor": c,
                    "color": c,
                    "weight": 1.8,
                    "fillOpacity": 0.38,
                },
                tooltip=f"🌊 {h_label}: {desc} ({eff_s:.1f}m Hydraulic Head)",
                popup=f"<b>Hydraulic Threat Horizon:</b> {h_label}<br><b>Surge Level:</b> {eff_s:.1f}m<br><b>Status:</b> Impassable Submersion",
            ).add_to(inund_fg)
        inund_fg.add_to(m)

    # 5. Layer 2: Detected Damaged Structures
    def style_function(feature: dict[str, Any]) -> dict[str, Any]:
        props = feature.get("properties", {})
        dmg_class = props.get("damage_class", props.get("damage", "no-damage"))
        hex_color = DAMAGE_HEX_COLORS.get(dmg_class, "#6c757d")
        return {
            "fillColor": hex_color,
            "color": hex_color,
            "weight": 2.5,
            "fillOpacity": 0.65,
        }

    def highlight_function(feature: dict[str, Any]) -> dict[str, Any]:
        return {
            "weight": 4.0,
            "fillOpacity": 0.88,
        }

    available_fields = ["damage_class", "confidence"]
    available_aliases = ["Damage Status:", "Confidence:"]
    sample_props = (
        geojson_data.get("features", [{}])[0].get("properties", {})
        if geojson_data.get("features")
        else {}
    )
    if "feature_id" in sample_props:
        available_fields.insert(0, "feature_id")
        available_aliases.insert(0, "Building ID:")
    if "is_simulated" in sample_props:
        available_fields.append("is_simulated")
        available_aliases.append("Simulation Mode:")

    folium.GeoJson(
        geojson_data,
        name="🏢 2. Detected Damaged Structures",
        style_function=style_function,
        highlight_function=highlight_function,
        tooltip=folium.GeoJsonTooltip(
            fields=available_fields,
            aliases=available_aliases,
            localize=True,
            sticky=True,
        ),
        popup=folium.GeoJsonPopup(
            fields=available_fields,
            aliases=available_aliases,
            localize=True,
        ),
    ).add_to(m)

    # 6. Layer 3: 3x3 Hex/Grid Priority Sectors
    if priority_zones:
        zone_fg = folium.FeatureGroup(name="🎯 3. 3x3 Priority Response Sectors", show=True)
        for z in priority_zones:
            min_lon, min_lat, max_lon, max_lat = z["bounds"]
            rank = z.get("priority_rank", 999)
            score = z.get("priority_score", 0.0)
            is_rank1 = (rank == 1)
            zone_color = "#dc2626" if is_rank1 else ("#ea580c" if rank <= 3 else "#0284c7")

            # Semi-transparent alpha fill matching triage severity
            folium.Rectangle(
                bounds=[[min_lat, min_lon], [max_lat, max_lon]],
                color=zone_color,
                weight=3.5 if is_rank1 else 2.0,
                dash_array="6, 6",
                fill=True,
                fill_opacity=0.28 if is_rank1 else 0.14,
                tooltip=f"{'🚨 RESCUE FIRST: ' if is_rank1 else ''}{z['zone_id']} | Rank #{rank} (Score: {score:.3f})",
                popup=(
                    f"<b>{'🚨 PRIMARY RESCUE SECTOR<br>' if is_rank1 else ''}Zone ID:</b> {z['zone_id']}<br>"
                    f"<b>Priority Rank:</b> #{rank}<br>"
                    f"<b>Priority Score:</b> {score:.4f}<br>"
                    f"<b>Damaged Structures:</b> {z.get('damaged_buildings', 0)} / {z.get('total_buildings', 0)}<br>"
                    f"<b>Destroyed:</b> {z.get('destroyed_buildings', 0)}<br>"
                    f"<b>Major Damage:</b> {z.get('major_damage_buildings', 0)}<br>"
                    f"<b>Flood Risk:</b> {z.get('flood_risk', 0.0):.1%}<br>"
                    f"<b>Mean Confidence:</b> {z.get('mean_confidence', 0.0):.1%}<br>"
                    f"<b>Directive:</b> {z.get('reason', '')}"
                ),
            ).add_to(zone_fg)

            # Center label marker
            z_c_lat = (min_lat + max_lat) / 2.0
            z_c_lon = (min_lon + max_lon) / 2.0
            badge_text = f"🚨 RESCUE FIRST: {z['zone_id']}" if is_rank1 else f"Rank #{rank} ({z['zone_id']})"
            badge_bg = "#dc2626" if is_rank1 else "rgba(255, 255, 255, 0.92)"
            badge_fg = "#ffffff" if is_rank1 else "#0f172a"

            label_html = (
                f'<div style="font-size: 9pt; font-weight: 800; color: {badge_fg}; '
                f'background: {badge_bg}; padding: 3px 6px; border-radius: 4px; '
                f'border: 1.5px solid {zone_color}; white-space: nowrap; box-shadow: 1px 2px 4px rgba(0,0,0,0.3);">'
                f'{badge_text}'
                f'</div>'
            )
            folium.Marker(
                location=[z_c_lat, z_c_lon],
                icon=folium.DivIcon(html=label_html),
            ).add_to(zone_fg)

        zone_fg.add_to(m)

    # Layer: 👥 Population Density & Casualty Triage Heatmap
    if priority_zones:
        pop_fg = folium.FeatureGroup(name="👥 Population & Casualty Exposure Heatmap", show=True)
        for z in priority_zones:
            min_lon, min_lat, max_lon, max_lat = z["bounds"]
            pop = z.get("estimated_population", 0)
            vuln = z.get("vulnerable_population", 0)
            cas = z.get("estimated_casualties", 0)
            hosp = z.get("nearest_hospital", "Bay County Field Hospital")
            hosp_dist = z.get("hospital_distance_km", 1.0)
            transit_m = z.get("transit_time_mins", 3.0)

            # Color by casualty / vulnerability severity
            fill_c = "#ef4444" if cas >= 3 else ("#f97316" if cas >= 1 else ("#3b82f6" if pop > 10 else "#64748b"))
            folium.Rectangle(
                bounds=[[min_lat, min_lon], [max_lat, max_lon]],
                color=fill_c,
                weight=1.5,
                fill=True,
                fill_opacity=0.32 if cas > 0 else 0.15,
                tooltip=f"👥 {z['zone_id']}: {pop} Residents | {vuln} Vulnerable | ⚠️ {cas} Casualties",
                popup=(
                    f"<b>Sector {z['zone_id']} Demographic & Triage Data:</b><br>"
                    f"<b>Exposed Residents:</b> {pop} occupants<br>"
                    f"<b>Vulnerable (Elderly/Children):</b> {vuln}<br>"
                    f"<b>Estimated Casualties/Trapped:</b> <span style='color: red; font-weight: bold;'>{cas}</span><br>"
                    f"<b>Nearest Hospital:</b> {hosp} ({hosp_dist:.2f} km)<br>"
                    f"<b>Ambulance Transit Time:</b> {transit_m} mins"
                ),
            ).add_to(pop_fg)
        pop_fg.add_to(m)

    # 7. Layer 5: Stage 2 Evacuation / Infiltration Polyline Route
    if routing_geojson and routing_geojson.get("features"):
        route_fg = folium.FeatureGroup(name="🚒 5. Stage 2 Safe Infiltration Route", show=True)
        for rf in routing_geojson["features"]:
            coords = rf.get("geometry", {}).get("coordinates", [])
            props = rf.get("properties", {})
            r_type = props.get("route_type", "route")
            color = props.get("color", "#00ff88")
            dist_km = props.get("distance_km", 0.0)
            time_mins = props.get("duration_mins", 0.0)
            label = props.get("label", "Route")
            is_baseline = ("baseline" in r_type)

            # Flip coordinates to [lat, lon] for Folium
            latlon_coords = [[pt[1], pt[0]] for pt in coords]
            folium.PolyLine(
                locations=latlon_coords,
                color=color,
                weight=3.5 if is_baseline else 5.0,
                dash_array="8, 8" if is_baseline else None,
                opacity=0.75 if is_baseline else 0.95,
                tooltip=f"{label} ({dist_km} km, {time_mins} mins)",
                popup=(
                    f"<b>{label}</b><br>"
                    f"Distance: {dist_km} km<br>"
                    f"Transit Time: {time_mins} mins<br>"
                    f"Clearance Status: {props.get('clearance_status', 'N/A')}<br>"
                    f"Detour Penalty: +{props.get('detour_penalty_mins', 0.0)} mins"
                ),
            ).add_to(route_fg)

        # Add Staging Base Markers
        if len(latlon_coords) > 0:
            start_pt = latlon_coords[0]
            end_pt = latlon_coords[-1]
            folium.Marker(
                location=start_pt,
                icon=folium.Icon(color="blue", icon="shield", prefix="fa"),
                tooltip="Staging Base Alpha (West US-98 Insertion Point)",
            ).add_to(route_fg)
            folium.Marker(
                location=end_pt,
                icon=folium.Icon(color="red", icon="ambulance", prefix="fa"),
                tooltip="Priority Zone 1 Triage Target",
            ).add_to(route_fg)

        route_fg.add_to(m)

    # Drone flight route in Live Drone Recon Stream mode
    if is_drone_stream:
        from utils.drone_stream import MISSION_WAYPOINTS
        drone_fg = folium.FeatureGroup(name="✈️ Stage 2 Drone Recon Route", show=True)
        route_coords = [[wp.lat, wp.lon] for wp in MISSION_WAYPOINTS]
        folium.PolyLine(
            locations=route_coords,
            color="#0284c7",
            weight=3.5,
            dash_array="8, 8",
            tooltip="Autonomous UAV-SAR Search Route",
        ).add_to(drone_fg)
        drone_fg.add_to(m)

    # Keep LayerControl collapsed so it does not obstruct the map and priority sectors
    folium.LayerControl(position="topright", collapsed=True).add_to(m)

    # Fit map bounds to encompass all detected building footprints
    bounds = get_bounds(geojson_data)
    if bounds != [[0.0, 0.0], [0.0, 0.0]]:
        m.fit_bounds(bounds, padding=[25, 25])

    return m


def render_folium_map(
    m: folium.Map,
    tactical_overlay_path: str | Path | None = None,
) -> None:
    """Render Folium map in Streamlit with tabbed Tactical Optical View option."""
    st.markdown("### 🗺️ Geospatial Damage Assessment & Tactical Mapping")
    st.caption(
        "Dual-View Intelligence: Compare the direct pixel-accurate optical reconnaissance overlay "
        "with the interactive satellite GIS map."
    )

    render_legend_badges()

    if tactical_overlay_path and Path(tactical_overlay_path).exists():
        tab_opt, tab_gis = st.tabs([
            "🎯 Tactical Optical Detection Overlay (Direct Image Analysis)",
            "🛰️ Interactive Satellite GIS Map (Folium & Priority Sectors)",
        ])
        with tab_opt:
            st.caption(
                "Direct pixel-registered detection overlay on your uploaded post-disaster imagery. "
                "Every building footprint is color-coded by EMS-98 damage severity, and the primary rescue area is targeted."
            )
            st.image(str(tactical_overlay_path), width="stretch")

        with tab_gis:
            try:
                st_folium(m, width="100%", height=530, returned_objects=[])
            except Exception as exc:
                st.warning(f"Map rendering in fallback mode: {exc}")
    else:
        try:
            st_folium(m, width="100%", height=530, returned_objects=[])
        except Exception as exc:
            st.warning(f"Map rendering in fallback mode: {exc}")

    st.markdown("---")
