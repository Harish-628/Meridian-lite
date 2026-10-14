"""
MERIDIAN-Lite: Disaster Response Decision-Support System.
Tactical GIS Dashboard for Drone Reconnaissance, Situational Awareness, and Priority-Zone Intelligence.

Main Application Coordinator.
"""

from pathlib import Path
import time
import streamlit as st

from app.ui import (
    build_folium_map,
    inject_custom_styles,
    render_building_inventory_table,
    render_export_section,
    render_folium_map,
    render_header_banners,
    render_imagery_section,
    render_priority_zones_section,
    render_sidebar,
    render_tactical_summary,
)
from models.predict import run_inference
from models.real_predict import run_real_inference
from utils.geojson_utils import load_geojson, save_geojson
from utils.image_utils import create_damage_overlay, create_demo_images, validate_image
from utils.logging_utils import get_logger, setup_logging
from utils.metrics import calculate_summary_metrics, features_to_dataframe
from utils.paths import (
    DEFAULT_GEOJSON_OUTPUT,
    DEFAULT_OVERLAY_OUTPUT,
    DEMO_BUILDINGS_GEOJSON,
    DEMO_DATA_DIR,
    DEMO_POST_IMAGE,
    DEMO_PRE_IMAGE,
    PROJECT_ROOT,
    ensure_directories,
)
from utils.priority import compute_priority_zones

# Setup structured logging and ensure workspace directories
setup_logging()
logger = get_logger("app.main")
ensure_directories()

# Streamlit Page Configuration
st.set_page_config(
    page_title="MERIDIAN-Lite | Disaster Response DSS",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom styles & typography
inject_custom_styles()


def _ensure_demo_assets() -> None:
    """Ensure local synthetic baseline imagery and GeoJSON are present."""
    if not DEMO_PRE_IMAGE.exists() or not DEMO_POST_IMAGE.exists():
        logger.info("Demo imagery missing; generating synthetic baseline...")
        create_demo_images(DEMO_DATA_DIR)

    if not DEMO_BUILDINGS_GEOJSON.exists():
        logger.info("Demo GeoJSON missing; generating locally...")
        demo_geojson = run_inference(
            pre_image_path=DEMO_PRE_IMAGE,
            post_image_path=DEMO_POST_IMAGE,
            center_lat=13.0827,
            center_lon=80.2707,
            seed=42,
            building_count=18,
        )
        demo_geojson.setdefault("metadata", {})["is_demo"] = True
        for f in demo_geojson.get("features", []):
            f.setdefault("properties", {})["is_demo"] = True
        save_geojson(demo_geojson, DEMO_BUILDINGS_GEOJSON)


_ensure_demo_assets()

# Initialize session state cache
for key, default in [
    ("assessment_results", None),
    ("priority_zones", None),
    ("overlay_path", None),
    ("pre_display_path", None),
    ("post_display_path", None),
    ("processing_time_ms", None),
    ("active_mode", "Demo mode"),
    ("active_backend", "Simulated Stage 1 Engine"),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# Header Title
st.markdown("<div class='main-title'>🛰️ MERIDIAN-Lite: Disaster Response Decision-Support System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Tactical Drone Reconnaissance, Multi-Factor Triage & Situational Awareness · Innovation Mission 6</div>", unsafe_allow_html=True)

# Render Sidebar Control Panel
cfg = render_sidebar()

# Render Visual Distinction Banners
render_header_banners(mode=cfg.mode, dataset_choice=cfg.dataset_choice, backend_choice=cfg.backend_choice)

# ==========================================
# INFERENCE EXECUTION HANDLER
# ==========================================
if cfg.run_button:
    st.session_state["active_mode"] = cfg.mode
    st.session_state["active_backend"] = cfg.backend_choice

    if cfg.mode == "Real Model Mode":
        logger.info(f"Executing assessment in Real Model Mode with {cfg.backend_choice}...")
        start_t = time.time()
        with st.spinner("Executing real disaster model inference and priority zone scoring..."):
            if cfg.dataset_choice == "Hurricane Michael (Mexico Beach, FL - Oct 2018)":
                pre_path = PROJECT_ROOT / "data" / "processed" / "hurricane_michael_pre.png"
                post_path = PROJECT_ROOT / "data" / "processed" / "hurricane_michael_post.png"
            else:
                if not cfg.uploaded_pre_file or not cfg.uploaded_post_file:
                    st.error("❌ Both pre-disaster and post-disaster images are required.")
                    st.stop()
                temp_upload_dir = PROJECT_ROOT / "outputs" / "temp_uploads"
                temp_upload_dir.mkdir(parents=True, exist_ok=True)
                pre_path = temp_upload_dir / f"real_pre_{cfg.uploaded_pre_file.name}"
                post_path = temp_upload_dir / f"real_post_{cfg.uploaded_post_file.name}"
                pre_path.write_bytes(cfg.uploaded_pre_file.getvalue())
                post_path.write_bytes(cfg.uploaded_post_file.getvalue())

            backend_key = "real_damage_model"
            if "Classical" in cfg.backend_choice:
                backend_key = "classical_change_detection"
            elif "Simulated" in cfg.backend_choice:
                backend_key = "simulated_xview2"

            geojson_result = run_real_inference(
                pre_image_path=pre_path,
                post_image_path=post_path,
                center_lat=cfg.center_lat,
                center_lon=cfg.center_lon,
                confidence_threshold=cfg.confidence_threshold,
                max_features=cfg.max_features,
                backend_name=backend_key,
            )

            _, overlay_p = create_damage_overlay(
                pre_image_path=pre_path,
                post_image_path=post_path,
                output_path=DEFAULT_OVERLAY_OUTPUT,
            )

            priority_zones = compute_priority_zones(
                feature_collection=geojson_result,
                grid_rows=cfg.grid_rows,
                grid_cols=cfg.grid_cols,
                default_bounds=(-85.555, 30.170, -85.545, 30.180),
            )

            save_geojson(geojson_result, DEFAULT_GEOJSON_OUTPUT)
            elapsed_ms = round((time.time() - start_t) * 1000, 1)

            st.session_state["assessment_results"] = geojson_result
            st.session_state["priority_zones"] = priority_zones
            st.session_state["overlay_path"] = str(overlay_p)
            st.session_state["pre_display_path"] = str(pre_path)
            st.session_state["post_display_path"] = str(post_path)
            st.session_state["processing_time_ms"] = elapsed_ms

        st.toast(f"Real model assessment completed in {elapsed_ms} ms!", icon="✅")

    elif cfg.mode == "Demo mode":
        logger.info("Executing assessment in Demo mode...")
        start_t = time.time()
        with st.spinner("Processing aerial imagery and executing building damage assessment..."):
            pre_path = DEMO_PRE_IMAGE
            post_path = DEMO_POST_IMAGE

            geojson_result = run_inference(
                pre_image_path=pre_path,
                post_image_path=post_path,
                center_lat=cfg.center_lat,
                center_lon=cfg.center_lon,
                seed=cfg.seed,
                building_count=cfg.building_count,
            )

            _, overlay_p = create_damage_overlay(
                pre_image_path=pre_path,
                post_image_path=post_path,
                output_path=DEFAULT_OVERLAY_OUTPUT,
            )

            priority_zones = compute_priority_zones(
                feature_collection=geojson_result,
                grid_rows=cfg.grid_rows,
                grid_cols=cfg.grid_cols,
                default_bounds=(
                    cfg.center_lon - 0.005,
                    cfg.center_lat - 0.005,
                    cfg.center_lon + 0.005,
                    cfg.center_lat + 0.005,
                ),
            )

            save_geojson(geojson_result, DEFAULT_GEOJSON_OUTPUT)
            elapsed_ms = round((time.time() - start_t) * 1000, 1)

            st.session_state["assessment_results"] = geojson_result
            st.session_state["priority_zones"] = priority_zones
            st.session_state["overlay_path"] = str(overlay_p)
            st.session_state["pre_display_path"] = str(pre_path)
            st.session_state["post_display_path"] = str(post_path)
            st.session_state["processing_time_ms"] = elapsed_ms

        st.toast("Assessment completed successfully!", icon="✅")

    elif cfg.mode == "Upload mode":
        if not cfg.uploaded_pre_file or not cfg.uploaded_post_file:
            st.error("❌ Both pre-disaster and post-disaster images are required to execute assessment.")
        else:
            logger.info("Executing assessment in Upload mode...")
            start_t = time.time()
            with st.spinner("Validating uploads and executing damage assessment..."):
                temp_upload_dir = PROJECT_ROOT / "outputs" / "temp_uploads"
                temp_upload_dir.mkdir(parents=True, exist_ok=True)

                pre_temp_path = temp_upload_dir / f"upload_pre_{cfg.uploaded_pre_file.name}"
                post_temp_path = temp_upload_dir / f"upload_post_{cfg.uploaded_post_file.name}"

                pre_temp_path.write_bytes(cfg.uploaded_pre_file.getvalue())
                post_temp_path.write_bytes(cfg.uploaded_post_file.getvalue())

                validate_image(pre_temp_path)
                validate_image(post_temp_path)

                geojson_result = run_inference(
                    pre_image_path=pre_temp_path,
                    post_image_path=post_temp_path,
                    center_lat=cfg.center_lat,
                    center_lon=cfg.center_lon,
                    seed=cfg.seed,
                    building_count=cfg.building_count,
                )

                _, overlay_p = create_damage_overlay(
                    pre_image_path=pre_temp_path,
                    post_image_path=post_temp_path,
                    output_path=DEFAULT_OVERLAY_OUTPUT,
                )

                priority_zones = compute_priority_zones(
                    feature_collection=geojson_result,
                    grid_rows=cfg.grid_rows,
                    grid_cols=cfg.grid_cols,
                )

                save_geojson(geojson_result, DEFAULT_GEOJSON_OUTPUT)
                elapsed_ms = round((time.time() - start_t) * 1000, 1)

                st.session_state["assessment_results"] = geojson_result
                st.session_state["priority_zones"] = priority_zones
                st.session_state["overlay_path"] = str(overlay_p)
                st.session_state["pre_display_path"] = str(pre_temp_path)
                st.session_state["post_display_path"] = str(post_temp_path)
                st.session_state["processing_time_ms"] = elapsed_ms

            st.toast("Upload assessment completed successfully!", icon="✅")

# Initialize default view from baseline if empty
if st.session_state["assessment_results"] is None:
    logger.info("Loading baseline demo assessment into initial dashboard view...")
    initial_geojson = load_geojson(DEMO_BUILDINGS_GEOJSON)
    st.session_state["assessment_results"] = initial_geojson
    st.session_state["priority_zones"] = compute_priority_zones(initial_geojson, 3, 3)
    st.session_state["pre_display_path"] = str(DEMO_PRE_IMAGE)
    st.session_state["post_display_path"] = str(DEMO_POST_IMAGE)
    if not DEFAULT_OVERLAY_OUTPUT.exists():
        _, overlay_p = create_damage_overlay(DEMO_PRE_IMAGE, DEMO_POST_IMAGE, DEFAULT_OVERLAY_OUTPUT)
        st.session_state["overlay_path"] = str(overlay_p)
    else:
        st.session_state["overlay_path"] = str(DEFAULT_OVERLAY_OUTPUT)
    st.session_state["processing_time_ms"] = 112.5

geojson_data = st.session_state["assessment_results"]
priority_zones = st.session_state.get("priority_zones") or compute_priority_zones(geojson_data, 3, 3)
metrics = calculate_summary_metrics(geojson_data)
df_results = features_to_dataframe(geojson_data)

# ==========================================
# PRESENTATION VIEWS
# ==========================================
render_tactical_summary(metrics, st.session_state.get("processing_time_ms"))
render_priority_zones_section(priority_zones)
render_imagery_section(
    st.session_state.get("pre_display_path"),
    st.session_state.get("post_display_path"),
    st.session_state.get("overlay_path"),
)
map_instance = build_folium_map(geojson_data, priority_zones, cfg.center_lat, cfg.center_lon)
render_folium_map(map_instance)
render_building_inventory_table(df_results)
render_export_section(
    geojson_data=geojson_data,
    metrics=metrics,
    df_results=df_results,
    priority_zones=priority_zones,
    overlay_path=st.session_state.get("overlay_path"),
)
