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
    center_lat: float = 30.154378,
    center_lon: float = -85.548719,
    is_drone_stream: bool = False,
    image_overlay_path: str | Path | None = None,
) -> folium.Map:
    """Construct an interactive Folium map populated with building polygons, priority zones, and optional aerial imagery overlay."""
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

    # Initialize map with Esri Satellite as primary basemap
    m = folium.Map(
        location=[c_lat, c_lon],
        zoom_start=17,
        tiles=None,  # We add custom tile layers with clean names below
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

    # 3. Optional: Overlay the Actual Uploaded Aerial/Drone Photo directly onto map surface
    if image_overlay_path and Path(image_overlay_path).exists():
        bounds = get_bounds(geojson_data)
        if bounds != [[0.0, 0.0], [0.0, 0.0]]:
            try:
                folium.raster_layers.ImageOverlay(
                    name="📸 Uploaded Aerial Image Overlay",
                    image=str(image_overlay_path),
                    bounds=bounds,
                    opacity=0.85,
                    interactive=True,
                    zindex=2,
                ).add_to(m)
            except Exception as e:
                pass

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

    # Add building GeoJSON layer
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

    folium_geojson = folium.GeoJson(
        geojson_data,
        name="🏢 Building Footprints & Damage Classification",
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
    )
    folium_geojson.add_to(m)

    # Add Priority Zones Layer
    if priority_zones:
        zone_fg = folium.FeatureGroup(name="🎯 Priority Response Zones", show=True)
        for z in priority_zones:
            min_lon, min_lat, max_lon, max_lat = z["bounds"]
            rank = z["priority_rank"]
            score = z["priority_score"]
            is_rank1 = (rank == 1)
            zone_color = "#dc2626" if is_rank1 else ("#ea580c" if rank <= 3 else "#0284c7")

            # Rectangle boundary
            folium.Rectangle(
                bounds=[[min_lat, min_lon], [max_lat, max_lon]],
                color=zone_color,
                weight=3.5 if is_rank1 else 2.0,
                dash_array="6, 6",
                fill=True,
                fill_opacity=0.22 if is_rank1 else 0.10,
                tooltip=f"{'🚨 RESCUE FIRST: ' if is_rank1 else ''}{z['zone_id']} | Rank #{rank} (Score: {score:.3f})",
                popup=(
                    f"<b>{'🚨 PRIMARY RESCUE SECTOR<br>' if is_rank1 else ''}Zone ID:</b> {z['zone_id']}<br>"
                    f"<b>Priority Rank:</b> #{rank}<br>"
                    f"<b>Priority Score:</b> {score:.4f}<br>"
                    f"<b>Damaged Structures:</b> {z['damaged_buildings']} / {z['total_buildings']}<br>"
                    f"<b>Destroyed:</b> {z['destroyed_buildings']}<br>"
                    f"<b>Major Damage:</b> {z['major_damage_buildings']}<br>"
                    f"<b>Mean Confidence:</b> {z['mean_confidence']:.1%}<br>"
                    f"<b>Directive:</b> {z['reason']}"
                ),
            ).add_to(zone_fg)

            # Center label marker
            c_lat = (min_lat + max_lat) / 2.0
            c_lon = (min_lon + max_lon) / 2.0
            badge_text = f"🚨 RESCUE FIRST: {z['zone_id']}" if is_rank1 else f"Rank #{rank} ({z['zone_id']})"
            badge_bg = "#dc2626" if is_rank1 else "rgba(255, 255, 255, 0.92)"
            badge_fg = "#ffffff" if is_rank1 else "#0f172a"

            label_html = (
                f'<div style="font-size: 9.5pt; font-weight: 800; color: {badge_fg}; '
                f'background: {badge_bg}; padding: 3px 7px; border-radius: 4px; '
                f'border: 1.5px solid {zone_color}; white-space: nowrap; box-shadow: 1px 2px 4px rgba(0,0,0,0.3);">'
                f'{badge_text}'
                f'</div>'
            )
            folium.Marker(
                location=[c_lat, c_lon],
                icon=folium.DivIcon(html=label_html),
            ).add_to(zone_fg)

        zone_fg.add_to(m)

    # ONLY add drone flight route in Live Drone Recon Stream mode
    if is_drone_stream:
        from utils.drone_stream import MISSION_WAYPOINTS
        route_fg = folium.FeatureGroup(name="Stage 2 Drone Search Route", show=True)
        route_coords = [[wp.lat, wp.lon] for wp in MISSION_WAYPOINTS]
        folium.PolyLine(
            locations=route_coords,
            color="#0284c7",
            weight=3.5,
            dash_array="8, 8",
            tooltip="Autonomous UAV-SAR Search Route",
        ).add_to(route_fg)

        for wp in MISSION_WAYPOINTS:
            wp_html = (
                f'<div style="font-size: 8.5pt; font-weight: bold; color: #0284c7; '
                f'background: rgba(255, 255, 255, 0.92); padding: 1px 5px; border-radius: 3px; '
                f'border: 1px solid #0284c7; white-space: nowrap; box-shadow: 1px 1px 2px rgba(0,0,0,0.2);">'
                f'✈️ {wp.waypoint_id}'
                f'</div>'
            )
            folium.Marker(
                location=[wp.lat, wp.lon],
                icon=folium.DivIcon(html=wp_html),
                tooltip=f"{wp.waypoint_id}: {wp.sector_name}",
            ).add_to(route_fg)

        route_fg.add_to(m)

    folium.LayerControl(position="topright", collapsed=False).add_to(m)

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
