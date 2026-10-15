"""
Imagery & Visual Comparison Views for MERIDIAN-Lite.
Provides 3-column pre-disaster, post-disaster, and visual difference overlay rendering.
"""

from pathlib import Path
import streamlit as st


def render_imagery_section(
    pre_path: str | Path | None,
    post_path: str | Path | None,
    overlay_path: str | Path | None,
    tactical_overlay_path: str | Path | None = None,
) -> None:
    """Render 3-column imagery comparison section."""
    st.markdown("### 📷 Aerial Reconnaissance Imagery & Comparison")

    img_col1, img_col2, img_col3 = st.columns(3)

    with img_col1:
        st.caption("**1. Pre-Disaster Reference Imagery**")
        if pre_path and Path(pre_path).exists():
            st.image(str(pre_path), width="stretch")
        else:
            st.info("Pre-disaster image not loaded.")

    with img_col2:
        st.caption("**2. Post-Disaster Drone Imagery**")
        if post_path and Path(post_path).exists():
            st.image(str(post_path), width="stretch")
        else:
            st.info("Post-disaster image not loaded.")

    # Prioritize high-fidelity tactical optical detection overlay in column 3
    display_overlay = tactical_overlay_path if (tactical_overlay_path and Path(tactical_overlay_path).exists()) else overlay_path

    with img_col3:
        if tactical_overlay_path and Path(tactical_overlay_path).exists():
            st.caption("**3. AI Optical Damage Detection & Triage HUD**")
            st.caption("*(EMS-98 Color Coded: Red=Dest, Orange=Major, Yellow=Minor)*")
            st.image(str(tactical_overlay_path), width="stretch")
        elif overlay_path and Path(overlay_path).exists():
            st.caption("**3. Visual Comparison (Pixel Difference Overlay)**")
            st.caption("*(Visual difference comparison)*")
            st.image(str(overlay_path), width="stretch")
        else:
            st.caption("**3. Optical Triage Overlay**")
            st.info("Overlay not yet rendered.")

    if tactical_overlay_path and overlay_path and Path(overlay_path).exists() and str(tactical_overlay_path) != str(overlay_path):
        with st.expander("🔬 View Raw Spectral Difference Map (Change Heatmap)", expanded=False):
            st.image(str(overlay_path), width="stretch", caption="Raw pixel-subtraction change map")

    st.markdown("---")
