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
) -> None:
    """Render 3-column imagery comparison section."""
    st.markdown("### 📷 Aerial Reconnaissance Imagery & Comparison")

    img_col1, img_col2, img_col3 = st.columns(3)

    with img_col1:
        st.caption("**Pre-Disaster Imagery**")
        if pre_path and Path(pre_path).exists():
            st.image(str(pre_path), use_container_width=True)
        else:
            st.info("Pre-disaster image not loaded.")

    with img_col2:
        st.caption("**Post-Disaster Imagery**")
        if post_path and Path(post_path).exists():
            st.image(str(post_path), use_container_width=True)
        else:
            st.info("Post-disaster image not loaded.")

    with img_col3:
        st.caption("**Visual Comparison (Pixel Difference Overlay)**")
        st.caption("*(Visual comparison only — not model segmentation)*")
        if overlay_path and Path(overlay_path).exists():
            st.image(str(overlay_path), use_container_width=True)
        else:
            st.info("Overlay not yet rendered.")

    st.markdown("---")
