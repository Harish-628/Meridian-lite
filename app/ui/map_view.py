"""
Interactive Geospatial Map View for MERIDIAN-Lite.
Builds and renders Folium GIS maps with building footprints, damage styling, and priority zone overlays.
"""

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
) -> folium.Map:
    """Construct an interactive Folium map populated with building polygons and priority zones."""
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=16,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    def style_function(feature: dict[str, Any]) -> dict[str, Any]:
        props = feature.get("properties", {})
        dmg_class = props.get("damage_class", props.get("damage", "no-damage"))
        hex_color = DAMAGE_HEX_COLORS.get(dmg_class, "#6c757d")
        return {
            "fillColor": hex_color,
            "color": hex_color,
            "weight": 2.5,
            "fillOpacity": 0.60,
        }

    def highlight_function(feature: dict[str, Any]) -> dict[str, Any]:
        return {
            "weight": 4.0,
            "fillOpacity": 0.85,
        }

    # Add building GeoJSON layer
    available_fields = ["damage_class", "confidence"]
    available_aliases = ["Damage Status:", "Confidence:"]
    sample_props = (
        geojson_data.get("features", [{}])[0].get("properties", {})
        if geojson_data.get("features")
        else {}
    )
    if "is_simulated" in sample_props:
        available_fields.append("is_simulated")
        available_aliases.append("Simulation Mode:")
    if "source_model" in sample_props:
        available_fields.append("source_model")
        available_aliases.append("Source Model:")

    folium_geojson = folium.GeoJson(
        geojson_data,
        name="Building Damage Assessment",
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
        zone_fg = folium.FeatureGroup(name="Stage 2 Priority Zones", show=True)
        for z in priority_zones:
            min_lon, min_lat, max_lon, max_lat = z["bounds"]
            rank = z["priority_rank"]
            score = z["priority_score"]
            zone_color = "#dc3545" if rank == 1 else ("#fd7e14" if rank <= 3 else "#0d6efd")

            # Rectangle boundary
            folium.Rectangle(
                bounds=[[min_lat, min_lon], [max_lat, max_lon]],
                color=zone_color,
                weight=2.5,
                dash_array="6, 6",
                fill=True,
                fill_opacity=0.15,
                tooltip=f"{z['zone_id']} | Rank #{rank} (Score: {score:.3f})",
                popup=(
                    f"<b>Zone ID:</b> {z['zone_id']}<br>"
                    f"<b>Priority Rank:</b> #{rank}<br>"
                    f"<b>Priority Score:</b> {score:.4f}<br>"
                    f"<b>Damaged Structures:</b> {z['damaged_buildings']} / {z['total_buildings']}<br>"
                    f"<b>Destroyed:</b> {z['destroyed_buildings']}<br>"
                    f"<b>Major Damage:</b> {z['major_damage_buildings']}<br>"
                    f"<b>Mean Confidence:</b> {z['mean_confidence']:.1%}<br>"
                    f"<b>Rationale:</b> {z['reason']}"
                ),
            ).add_to(zone_fg)

            # Center label marker
            c_lat = (min_lat + max_lat) / 2.0
            c_lon = (min_lon + max_lon) / 2.0
            label_html = (
                f'<div style="font-size: 10pt; font-weight: bold; color: #0f172a; '
                f'background: rgba(255, 255, 255, 0.88); padding: 2px 6px; border-radius: 4px; '
                f'border: 1.5px solid {zone_color}; white-space: nowrap; box-shadow: 1px 1px 3px rgba(0,0,0,0.2);">'
                f'Rank #{rank} ({z["zone_id"]})'
                f'</div>'
            )
            folium.Marker(
                location=[c_lat, c_lon],
                icon=folium.DivIcon(html=label_html),
            ).add_to(zone_fg)

        zone_fg.add_to(m)

    folium.LayerControl().add_to(m)

    # Fit map bounds to encompass all detected building footprints
    bounds = get_bounds(geojson_data)
    if bounds != [[0.0, 0.0], [0.0, 0.0]]:
        m.fit_bounds(bounds, padding=[20, 20])

    return m


def render_folium_map(m: folium.Map) -> None:
    """Render Folium map in Streamlit with legend badges and error fallback."""
    st.markdown("### 🗺️ Geospatial Damage Assessment Map")
    st.caption(
        "Interactive Folium GIS visualization with building polygons and Stage 2 priority response zones. "
        "Offline capable: GeoJSON and priority overlays operate completely without internet connectivity."
    )

    render_legend_badges()

    try:
        st_folium(m, width="100%", height=530, returned_objects=[])
    except Exception as exc:
        st.warning(f"Map rendering in fallback mode: {exc}")

    st.markdown("---")
