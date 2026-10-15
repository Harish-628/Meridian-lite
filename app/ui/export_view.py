"""
Export & Tactical Serialization Views for MERIDIAN-Lite.
Provides standardized download buttons for GeoJSON, Priority Zones JSON, CSV, and Overlays.
"""

import io
import json
from pathlib import Path
from typing import Any
import pandas as pd
import streamlit as st

from utils.paths import (
    DEFAULT_METRICS_CSV_OUTPUT,
    DEFAULT_METRICS_JSON_OUTPUT,
    PROJECT_ROOT,
)


def render_export_section(
    geojson_data: dict[str, Any],
    metrics: dict[str, Any],
    df_results: pd.DataFrame,
    priority_zones: list[dict[str, Any]] | None = None,
    overlay_path: str | Path | None = None,
) -> None:
    """Render 5-column download buttons and synchronize tactical reports to disk."""
    st.markdown("### 💾 Export Tactical Outputs")
    st.caption("Download verified geospatial layers, tabular summaries, and priority zones for emergency responder workflows.")

    # Precompute export artifacts
    geojson_str = json.dumps(geojson_data, indent=2, ensure_ascii=False)
    metrics_str = json.dumps(metrics, indent=2)

    csv_buffer = io.StringIO()
    df_results.to_csv(csv_buffer, index=False)
    csv_str = csv_buffer.getvalue()

    priority_zones_str = json.dumps(priority_zones or [], indent=2)

    overlay_bytes = b""
    if overlay_path and Path(overlay_path).exists():
        overlay_bytes = Path(overlay_path).read_bytes()

    # Build Complete Mission Package (.ZIP) in memory
    import zipfile
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("meridian_damage_assessment.geojson", geojson_str)
        zf.writestr("meridian_summary_metrics.json", metrics_str)
        zf.writestr("meridian_building_assessment.csv", csv_str)
        zf.writestr("meridian_priority_zones.json", priority_zones_str)
        sitrep_content = (
            f"# MERIDIAN-Lite Tactical Incident Situation Report\n\n"
            f"Total Structures Evaluated: {metrics.get('total_buildings', len(df_results))}\n"
            f"Destroyed: {metrics.get('destroyed_count', 0)}\n"
            f"Major Damage: {metrics.get('major_damage_count', 0)}\n"
            f"Minor Damage: {metrics.get('minor_damage_count', 0)}\n"
            f"Intact (No Damage): {metrics.get('no_damage_count', 0)}\n"
            f"Mean Model Confidence: {metrics.get('average_confidence', 0.85):.3f}\n"
        )
        zf.writestr("meridian_tactical_sitrep.md", sitrep_content)
        if overlay_bytes:
            zf.writestr("meridian_damage_overlay.png", overlay_bytes)
    zip_bytes = zip_buffer.getvalue()

    # Primary 1-Click Complete Mission Package Download
    st.download_button(
        label="📦 Download Complete Mission Package (.ZIP)",
        data=zip_bytes,
        file_name="meridian_complete_mission_package.zip",
        mime="application/zip",
        type="primary",
        width="stretch",
        help="One-click tactical bundle containing GeoJSON, CSV inventory, summary metrics, priority zones, and tactical SITREP.",
    )

    # Advanced Individual Format Downloads Expander
    with st.expander("📂 Advanced / Individual File Formats (GeoJSON, CSV, JSON, Overlay)", expanded=False):
        st.caption("Select individual standardized data formats for specialized GIS or tabular ingest:")
        export_col1, export_col2, export_col3, export_col4, export_col5 = st.columns(5)

        # 1. GeoJSON export
        with export_col1:
            st.download_button(
                label="📥 Download GeoJSON",
                data=geojson_str,
                file_name="meridian_damage_assessment.geojson",
                mime="application/geo+json",
                width="stretch",
            )

        # 2. Summary JSON export
        with export_col2:
            st.download_button(
                label="📥 Download Summary JSON",
                data=metrics_str,
                file_name="meridian_summary_metrics.json",
                mime="application/json",
                width="stretch",
            )

        # 3. CSV export
        with export_col3:
            st.download_button(
                label="📥 Download CSV Results",
                data=csv_str,
                file_name="meridian_building_assessment.csv",
                mime="text/csv",
                width="stretch",
            )

        # 4. Priority Zones JSON export
        with export_col4:
            st.download_button(
                label="📥 Download Priority Zones JSON",
                data=priority_zones_str,
                file_name="meridian_priority_zones.json",
                mime="application/json",
                width="stretch",
            )

        # 5. Overlay Image export
        with export_col5:
            st.download_button(
                label="📥 Download Damage Overlay",
                data=overlay_bytes,
                file_name="meridian_damage_overlay.png",
                mime="image/png",
                width="stretch",
                disabled=(len(overlay_bytes) == 0),
            )

    # Save reports to outputs directory
    try:
        DEFAULT_METRICS_JSON_OUTPUT.write_text(metrics_str, encoding="utf-8")
        DEFAULT_METRICS_CSV_OUTPUT.write_text(csv_str, encoding="utf-8")
        pz_target = PROJECT_ROOT / "outputs" / "reports" / "priority_zones.json"
        pz_target.parent.mkdir(parents=True, exist_ok=True)
        pz_target.write_text(priority_zones_str, encoding="utf-8")
    except Exception:
        pass

    st.markdown("---")
    st.markdown(
        "<small style='color: #94A3B8;'>"
        "MERIDIAN-Lite Stage 2 · College Hackathon Disaster Response DSS · Offline Ready · Real Model & Classical Backends"
        "</small>",
        unsafe_allow_html=True,
    )
