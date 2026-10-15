"""
UI Styling, Theme Configuration, and Badge Components for MERIDIAN-Lite.
Provides centralized CSS injection, color definitions, and alert banners.
"""

import streamlit as st

INLINE_CSS = """
<style>
.main-title {
    font-size: 2.2rem;
    font-weight: 700;
    color: #1E293B;
    margin-bottom: 0.2rem;
}
.sub-title {
    font-size: 1.05rem;
    color: #64748B;
    margin-bottom: 1.2rem;
}
.status-card {
    background-color: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 15px;
}
.legend-badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 4px;
    color: white;
    font-weight: 600;
    font-size: 0.85rem;
    margin-right: 8px;
}
.badge-no-damage { background-color: #28a745; }
.badge-minor-damage { background-color: #ffc107; color: #212529 !important; }
.badge-major-damage { background-color: #fd7e14; }
.badge-destroyed { background-color: #dc3545; }
.badge-real-mode { background-color: #0284c7; color: white; padding: 3px 8px; border-radius: 4px; font-weight: bold; }
.badge-sim-mode { background-color: #d97706; color: white; padding: 3px 8px; border-radius: 4px; font-weight: bold; }
</style>
"""


def inject_custom_styles() -> None:
    """Inject global CSS rules into Streamlit application."""
    st.markdown(INLINE_CSS, unsafe_allow_html=True)


def render_legend_badges() -> None:
    """Render color-coded damage classification and priority zone legend badges."""
    st.markdown(
        """
        <div style="margin-bottom: 12px;">
            <span class="legend-badge badge-no-damage">🟩 No Damage</span>
            <span class="legend-badge badge-minor-damage">🟨 Minor Damage</span>
            <span class="legend-badge badge-major-damage">🟧 Major Damage</span>
            <span class="legend-badge badge-destroyed">🟥 Destroyed</span>
            <span class="legend-badge" style="background-color: #3b82f6;">🟦 Priority Zone Boundary</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_header_banners(mode: str, dataset_choice: str = "", backend_choice: str = "") -> None:
    """Render prominent visual distinction banners separating real model output from simulation."""
    if mode == "Real Model Mode":
        st.markdown(
            f"""
            <div style="background-color: #EFF6FF; border: 1.5px solid #3B82F6; border-radius: 8px; padding: 12px 16px; margin-bottom: 16px;">
                <span class="badge-real-mode">Real Model Inference: local ONNX execution</span>
                <span style="color: #1E3A8A; font-weight: 600; margin-left: 10px;">
                    Active Backend: {backend_choice} &nbsp;|&nbsp; Sector: {dataset_choice}
                </span>
                <div style="color: #3B82F6; font-size: 0.88rem; margin-top: 4px;">
                    Verified Local ONNX Model (bda9515c) · CPU SIMD Execution. Model Provenance Not Independently Verified (Demonstration CNN).
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif mode == "Demo mode":
        st.warning(
            "⚠️ **Demo Simulation: deterministic synthetic output.** "
            "To evaluate genuine disaster drone imagery with local neural network inference, switch to **Real Model Mode** or **Upload mode** in the sidebar."
        )
    elif mode == "Live Drone Recon Stream":
        st.markdown(
            """
            <div style="background-color: #0F172A; border: 1.5px solid #00F0D2; border-radius: 8px; padding: 12px 16px; margin-bottom: 16px;">
                <span style="background-color: #EF4444; color: white; padding: 3px 8px; border-radius: 4px; font-weight: bold;">● LIVE DRONE RECON STREAM</span>
                <span style="color: #00F0D2; font-weight: 600; margin-left: 10px;">
                    UAV-SAR-01 Optical Datalink Active &nbsp;|&nbsp; Local ONNX Real-Time Neural Inference
                </span>
                <div style="color: #94A3B8; font-size: 0.88rem; margin-top: 4px;">
                    Real Model Inference: local ONNX execution. Progressive drone waypoint scanning, live sensor telemetry, HUD bounding boxes, and tactical resource allocation.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif mode == "Upload mode":
        st.markdown(
            f"""
            <div style="background-color: #F0FDF4; border: 1.5px solid #22C55E; border-radius: 8px; padding: 12px 16px; margin-bottom: 16px;">
                <span style="background-color: #15803D; color: white; padding: 3px 8px; border-radius: 4px; font-weight: bold; font-size: 0.8rem;">Real Model Inference: local ONNX execution</span>
                <span style="color: #166534; font-weight: 600; margin-left: 10px;">
                    Active Backend: {backend_choice} &nbsp;|&nbsp; Custom User Imagery
                </span>
                <div style="color: #15803D; font-size: 0.88rem; margin-top: 4px;">
                    Upload custom pre- and post-disaster aerial imagery pairs. MERIDIAN-Lite executes genuine local neural network inference (bda9515c), calculates EMS-98 building damage severity, and georeferences spatial polygons.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
