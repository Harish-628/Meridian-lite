"""
MERIDIAN-Lite UI Subsystem Package.
Contains modular presentation components for the Streamlit decision-support dashboard.
"""

from app.ui.drone_view import render_live_drone_recon_view
from app.ui.export_view import render_export_section
from app.ui.imagery_view import render_imagery_section
from app.ui.map_view import build_folium_map, render_folium_map
from app.ui.metrics_view import (
    render_building_inventory_table,
    render_priority_zones_section,
    render_tactical_incident_action_plan,
    render_tactical_summary,
)
from app.ui.styles import inject_custom_styles, render_legend_badges, render_header_banners
from app.ui.sidebar import render_sidebar, SidebarConfig
from app.ui.c2_terminal import render_c2_split_terminal

__all__ = [
    "inject_custom_styles",
    "render_legend_badges",
    "render_header_banners",
    "render_sidebar",
    "SidebarConfig",
    "render_tactical_incident_action_plan",
    "render_tactical_summary",
    "render_priority_zones_section",
    "render_imagery_section",
    "render_live_drone_recon_view",
    "build_folium_map",
    "render_folium_map",
    "render_c2_split_terminal",
    "render_building_inventory_table",
    "render_export_section",
]
