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

    export_col1, export_col2, export_col3, export_col4, export_col5 = st.columns(5)

    # 1. GeoJSON export
    geojson_str = json.dumps(geojson_data, indent=2, ensure_ascii=False)
    with export_col1:
        st.download_button(
            label="📥 Download GeoJSON",
            data=geojson_str,
            file_name="meridian_damage_assessment.geojson",
            mime="application/geo+json",
            use_container_width=True,
        )

    # 2. Summary JSON export
    metrics_str = json.dumps(metrics, indent=2)
    with export_col2:
        st.download_button(
            label="📥 Download Summary JSON",
            data=metrics_str,
            file_name="meridian_summary_metrics.json",
            mime="application/json",
            use_container_width=True,
        )

    # 3. CSV export
    csv_buffer = io.StringIO()
    df_results.to_csv(csv_buffer, index=False)
    csv_str = csv_buffer.getvalue()
    with export_col3:
        st.download_button(
            label="📥 Download CSV Results",
            data=csv_str,
            file_name="meridian_building_assessment.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # 4. Priority Zones JSON export
    priority_zones_str = json.dumps(priority_zones or [], indent=2)
    with export_col4:
        st.download_button(
            label="📥 Download Priority Zones JSON",
            data=priority_zones_str,
            file_name="meridian_priority_zones.json",
            mime="application/json",
            use_container_width=True,
        )

    # 5. Overlay Image export
    overlay_bytes = b""
    if overlay_path and Path(overlay_path).exists():
        overlay_bytes = Path(overlay_path).read_bytes()

    with export_col5:
        st.download_button(
            label="📥 Download Damage Overlay",
            data=overlay_bytes,
            file_name="meridian_damage_overlay.png",
            mime="image/png",
            use_container_width=True,
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
