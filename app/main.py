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
    render_live_drone_recon_view,
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
should_run = cfg.run_button

# Auto-run Demo mode immediately (no button needed)
if cfg.mode == "Demo mode":
    if st.session_state.get("active_mode") != "Demo mode" or cfg.run_button:
        should_run = True

# Auto-run Real Model Mode when dataset changes
if cfg.mode == "Real Model Mode":
    current_dataset_key = f"real_mode_{cfg.dataset_choice}_{cfg.backend_choice}"
    if st.session_state.get("last_real_dataset_key") != current_dataset_key or cfg.run_button:
        if cfg.dataset_choice != "Upload Custom Real Imagery" or (cfg.uploaded_pre_file and cfg.uploaded_post_file):
            should_run = True
            st.session_state["last_real_dataset_key"] = current_dataset_key

# Upload mode: auto-run when new images detected
if cfg.mode == "Upload mode":
    if cfg.uploaded_pre_file and cfg.uploaded_post_file:
        current_pair = (cfg.uploaded_pre_file.name, cfg.uploaded_post_file.name)
        up_col1, up_col2 = st.columns([3, 1])
        with up_col1:
            st.success(f"📁 Imagery Ready: **{cfg.uploaded_pre_file.name}** and **{cfg.uploaded_post_file.name}**")
        with up_col2:
            if st.button("🚀 Run AI Assessment", type="primary", key="quick_upload_run"):
                should_run = True
        if st.session_state.get("last_uploaded_pair") != current_pair:
            should_run = True
    else:
        st.info("ℹ️ Upload pre-disaster and post-disaster images in the sidebar to execute real damage detection.")

if should_run:
    st.session_state["active_mode"] = cfg.mode
    st.session_state["active_backend"] = cfg.backend_choice

    if cfg.mode == "Real Model Mode":
        logger.info(f"Executing assessment in Real Model Mode with {cfg.backend_choice}...")
        start_t = time.time()
        with st.spinner("Executing real disaster model inference and priority zone scoring..."):
            if cfg.dataset_choice == "Hurricane Michael (Mexico Beach, FL - Oct 2018)":
                pre_path = PROJECT_ROOT / "data" / "processed" / "hurricane_michael_pre.png"
                post_path = PROJECT_ROOT / "data" / "processed" / "hurricane_michael_post.png"
                dataset_bounds = (-85.550100, 30.152997, -85.547337, 30.155760)
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
                dataset_bounds = None

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
                bounds=dataset_bounds,
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
                default_bounds=dataset_bounds or (-85.550100, 30.152997, -85.547337, 30.155760),
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
            logger.info("Executing assessment in Upload mode with real model inference...")
            start_t = time.time()
            with st.spinner("Analyzing uploaded imagery with real AI neural model and CV pipeline..."):
                temp_upload_dir = PROJECT_ROOT / "outputs" / "temp_uploads"
                temp_upload_dir.mkdir(parents=True, exist_ok=True)

                pre_temp_path = temp_upload_dir / f"upload_pre_{cfg.uploaded_pre_file.name}"
                post_temp_path = temp_upload_dir / f"upload_post_{cfg.uploaded_post_file.name}"

                pre_temp_path.write_bytes(cfg.uploaded_pre_file.getvalue())
                post_temp_path.write_bytes(cfg.uploaded_post_file.getvalue())

                validate_image(pre_temp_path)
                validate_image(post_temp_path)

                backend_key = "real_damage_model"
                if "Classical" in cfg.backend_choice:
                    backend_key = "classical_change_detection"
                elif "Simulated" in cfg.backend_choice:
                    backend_key = "simulated_xview2"

                if backend_key == "simulated_xview2":
                    geojson_result = run_inference(
                        pre_image_path=pre_temp_path,
                        post_image_path=post_temp_path,
                        center_lat=cfg.center_lat,
                        center_lon=cfg.center_lon,
                        seed=cfg.seed,
                        building_count=cfg.building_count,
                    )
                else:
                    geojson_result = run_real_inference(
                        pre_image_path=pre_temp_path,
                        post_image_path=post_temp_path,
                        center_lat=cfg.center_lat,
                        center_lon=cfg.center_lon,
                        confidence_threshold=cfg.confidence_threshold,
                        max_features=cfg.max_features,
                        backend_name=backend_key,
                        bounds=None,
                        is_custom_upload=True,
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
                st.session_state["last_uploaded_pair"] = (cfg.uploaded_pre_file.name, cfg.uploaded_post_file.name)

            st.toast(f"Upload assessment completed in {elapsed_ms} ms ({len(geojson_result.get('features', []))} buildings detected)!", icon="✅")

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
if cfg.mode == "Live Drone Recon Stream":
    drone_pre = PROJECT_ROOT / "data" / "processed" / "hurricane_michael_pre.png"
    drone_post = PROJECT_ROOT / "data" / "processed" / "hurricane_michael_post.png"
    render_live_drone_recon_view(
        pre_image_path=drone_pre,
        post_image_path=drone_post,
        priority_zones=priority_zones,
        confidence_threshold=cfg.confidence_threshold,
        video_source_type=cfg.video_source_type,
        video_file=cfg.video_file,
        rtsp_url=cfg.rtsp_url,
        use_webcam=cfg.use_webcam,
        skip_frames=cfg.skip_frames,
        max_features=cfg.max_features,
        center_lat=cfg.center_lat,
        center_lon=cfg.center_lon,
        backend_choice=cfg.backend_choice,
        drone_pre_file=cfg.drone_pre_file,
        drone_post_file=cfg.drone_post_file,
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
elif cfg.mode == "Upload mode" and not (cfg.uploaded_pre_file and cfg.uploaded_post_file) and st.session_state.get("last_uploaded_pair") is None:
    st.markdown(
        """
        <div style="border: 2px dashed #94A3B8; border-radius: 12px; padding: 40px; text-align: center; margin: 24px 0; background: #F8FAFC;">
            <div style="font-size: 2.5rem; margin-bottom: 12px;">📤</div>
            <div style="font-size: 1.25rem; font-weight: 600; color: #1E293B; margin-bottom: 8px;">Upload Pre- and Post-Disaster Imagery</div>
            <div style="font-size: 0.95rem; color: #64748B; max-width: 600px; margin: 0 auto 16px auto;">
                Use the sidebar file uploaders on the left to upload pre- and post-disaster aerial or drone images. MERIDIAN-Lite will automatically detect structural changes, classify building damage according to EMS-98, and calculate spatial priority zones.
            </div>
            <div style="display: inline-block; background: #E2E8F0; padding: 6px 14px; border-radius: 6px; font-size: 0.85rem; color: #334155; font-weight: 500;">
                Supported Formats: PNG, JPG, JPEG · Local Real ONNX AI Inference
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
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
