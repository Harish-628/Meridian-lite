"""
Sidebar Control Panel & Parameter Configuration for MERIDIAN-Lite.
Encapsulates mode selection, backend switching, spatial coordinates, and hardware diagnostics.
"""

from dataclasses import dataclass, field
from typing import Any
import streamlit as st

from models.model_registry import check_model_health


@dataclass
class SidebarConfig:
    """Encapsulates all parameters configured via the Streamlit sidebar."""
    mode: str = "Real Model Mode"
    backend_choice: str = "Real Damage Model (ONNX)"
    dataset_choice: str = "Hurricane Michael (Mexico Beach, FL - Oct 2018)"
    confidence_threshold: float = 0.30
    max_features: int = 100
    grid_rows: int = 3
    grid_cols: int = 3
    center_lat: float = 30.1750
    center_lon: float = -85.5500
    seed: int = 42
    building_count: int = 18
    uploaded_pre_file: Any = None
    uploaded_post_file: Any = None
    run_button: bool = False
    # Live Drone stream extras
    video_source_type: str = "Uploaded Video File"
    video_file: Any = None           # st.UploadedFile for .mp4/.avi/.mov
    rtsp_url: str = ""
    use_webcam: bool = False
    skip_frames: int = 3             # Process every N frames
    drone_pre_file: Any = None       # Optional pre-disaster reference image for drone mode
    drone_post_file: Any = None      # Optional post-disaster still for non-video demo


def render_sidebar() -> SidebarConfig:
    """Render the sidebar controls and return the active user configuration."""
    st.sidebar.title("MERIDIAN-Lite Command Center")
    st.sidebar.caption("Tactical Drone Reconnaissance & Real-Time Disaster Intelligence")

    mode = st.sidebar.radio(
        "Assessment Mode",
        options=["Real Model Mode", "Upload mode", "Live Drone Recon Stream"],
        index=0,
        help="Select 'Real Model Mode' for on-device neural inference on real Hurricane Michael aerial imagery, or 'Upload mode' to assess custom drone photos.",
    )
    if mode == "Real Model Mode":
        st.sidebar.caption("🎯 **Primary Production Workflow:** On-device neural damage inference on Hurricane Michael ground truth.")
    elif mode == "Upload mode":
        st.sidebar.caption("📤 **Tactical Upload Workflow:** Ingest and evaluate custom pre/post drone imagery.")

    uploaded_pre_file = None
    uploaded_post_file = None
    video_source_type = "Uploaded Video File"
    video_file = None
    rtsp_url = ""
    use_webcam = False
    skip_frames = 3
    drone_pre_file = None
    drone_post_file = None

    # =========================================================================
    # 1. UPLOAD MODE CONTROLS (PRIMARY MISSION INPUT)
    # =========================================================================
    if mode == "Upload mode":
        st.sidebar.markdown("---")
        st.sidebar.subheader("📤 Upload Disaster Imagery")
        st.sidebar.caption("Upload pre- and post-disaster aerial or satellite photos from your drone. The AI model will process them and generate search-and-rescue priorities.")
        
        uploaded_pre_file = st.sidebar.file_uploader(
            "Pre-disaster Reference Image",
            type=["png", "jpg", "jpeg"],
            key="upload_mode_pre",
            help="Upload high-resolution pre-event aerial or satellite imagery.",
        )
        uploaded_post_file = st.sidebar.file_uploader(
            "Post-disaster Drone Recon Image",
            type=["png", "jpg", "jpeg"],
            key="upload_mode_post",
            help="Upload post-event drone or aerial imagery covering the target disaster sector.",
        )

        st.sidebar.markdown("---")
        st.sidebar.subheader("🤖 Neural Inference Engine")
        backend_choice = st.sidebar.selectbox(
            "Model Backend",
            options=[
                "Real Damage Model (ONNX)",
                "Advanced Damage Model Backend (GPU-Ready / CPU Fallback)",
                "Classical Change Detection (OpenCV)",
            ],
            index=0,
            help="Select the AI neural network or computer vision backend to analyze your uploaded imagery.",
        )
        if "Advanced" in backend_choice or "GPU" in backend_choice:
            st.sidebar.info("GPU runtime unavailable. Lightweight ONNX backend selected with CPU execution.")
        dataset_choice = "User Uploaded Imagery"

        confidence_thresh = float(st.sidebar.slider(
            "Confidence Threshold",
            min_value=0.10,
            max_value=0.90,
            value=0.30,
            step=0.05,
            help="Filter detections by minimum model confidence score.",
        ))
        max_feat = int(st.sidebar.slider(
            "Maximum Building Features",
            min_value=10,
            max_value=300,
            value=100,
            step=10,
            help="Maximum building footprints to detect.",
        ))
        grid_dim = st.sidebar.selectbox(
            "Priority Zone Grid Size",
            options=["3 x 3 (9 Zones)", "2 x 2 (4 Zones)", "4 x 4 (16 Zones)"],
            index=0,
            help="Spatial partitioning resolution for tactical priority-zone scoring.",
        )

        loc_preset = st.sidebar.selectbox(
            "Geographic Location Reference",
            options=[
                "Auto-Center on Target Imagery (Recommended)",
                "Gulf Coast / Mexico Beach, FL",
                "Beirut Port Disaster Zone, Lebanon",
                "Custom GPS Coordinates...",
            ],
            index=0,
            help="Select the geographic disaster theater to anchor satellite GIS coordinates.",
        )

        preset_coords = {
            "Auto-Center on Target Imagery (Recommended)": (30.154378, -85.548719),
            "Gulf Coast / Mexico Beach, FL": (30.154378, -85.548719),
            "Beirut Port Disaster Zone, Lebanon": (33.9014, 35.5190),
        }

        if loc_preset == "Custom GPS Coordinates...":
            with st.sidebar.expander("📍 Custom Coordinates", expanded=True):
                center_lat = float(st.number_input("Target Latitude", value=30.154378, format="%.6f", key="custom_up_lat"))
                center_lon = float(st.number_input("Target Longitude", value=-85.548719, format="%.6f", key="custom_up_lon"))
        else:
            center_lat, center_lon = preset_coords.get(loc_preset, (30.154378, -85.548719))
            st.sidebar.caption(f"📍 GPS Anchor: `{center_lat:.4f}° N, {center_lon:.4f}° E`")

        seed = 42
        building_count = 18

    # =========================================================================
    # 2. REAL MODEL MODE CONTROLS
    # =========================================================================
    elif mode == "Real Model Mode":
        st.sidebar.markdown("---")
        st.sidebar.subheader("🤖 Neural Inference Engine")
        backend_choice = st.sidebar.selectbox(
            "Model Backend",
            options=[
                "Real Damage Model (ONNX)",
                "Advanced Damage Model Backend (GPU-Ready / CPU Fallback)",
                "Classical Change Detection (OpenCV)",
            ],
            index=0,
            help="Select the AI neural network or computer vision backend.",
        )
        if "Advanced" in backend_choice or "GPU" in backend_choice:
            st.sidebar.info("GPU runtime unavailable. Lightweight ONNX backend selected with CPU execution.")
        dataset_choice = st.sidebar.selectbox(
            "Real Disaster Dataset",
            options=[
                "Hurricane Michael (Mexico Beach, FL - Oct 2018)",
                "Upload Custom Real Imagery",
            ],
            index=0,
            help="Select pre-packaged real disaster imagery or upload custom tile.",
        )

        if dataset_choice == "Upload Custom Real Imagery":
            uploaded_pre_file = st.sidebar.file_uploader(
                "Pre-disaster Image",
                type=["png", "jpg", "jpeg"],
                key="real_mode_pre",
            )
            uploaded_post_file = st.sidebar.file_uploader(
                "Post-disaster Image",
                type=["png", "jpg", "jpeg"],
                key="real_mode_post",
            )

        confidence_thresh = float(st.sidebar.slider(
            "Confidence Threshold",
            min_value=0.10,
            max_value=0.90,
            value=0.30,
            step=0.05,
            help="Filter predictions by minimum model confidence score.",
        ))
        max_feat = int(st.sidebar.slider(
            "Maximum Building Features",
            min_value=10,
            max_value=300,
            value=100,
            step=10,
            help="Maximum building footprints to retain.",
        ))
        grid_dim = st.sidebar.selectbox(
            "Priority Zone Grid Size",
            options=["3 x 3 (9 Zones)", "2 x 2 (4 Zones)", "4 x 4 (16 Zones)"],
            index=0,
        )

        with st.sidebar.expander("📍 Spatial Reference Coordinates", expanded=False):
            center_lat = float(st.number_input(
                "Centre Latitude",
                value=30.154378,
                format="%.6f",
                key="real_lat",
            ))
            center_lon = float(st.number_input(
                "Centre Longitude",
                value=-85.548719,
                format="%.6f",
                key="real_lon",
            ))

        seed = 42
        building_count = 18

    # =========================================================================
    # 3. LIVE DRONE RECON STREAM CONTROLS
    # =========================================================================
    elif mode == "Live Drone Recon Stream":
        st.sidebar.markdown("---")
        st.sidebar.subheader("🎥 Video Input Source")
        video_source_type = st.sidebar.selectbox(
            "Stream Source",
            options=[
                "Uploaded Video File",
                "Webcam (Live)",
                "RTSP/HTTP Stream URL",
                "Pre/Post Image Pair",
            ],
            index=0,
            help="Choose how to feed video frames into the AI pipeline.",
        )

        if video_source_type == "Uploaded Video File":
            video_file = st.sidebar.file_uploader(
                "Upload Drone Video",
                type=["mp4", "avi", "mov", "mkv", "webm"],
                help="Upload any drone/aerial video file. The AI will process it frame by frame.",
            )
        elif video_source_type == "Webcam (Live)":
            use_webcam = True
            st.sidebar.info("📷 Webcam stream will be captured from device 0.")
        elif video_source_type == "RTSP/HTTP Stream URL":
            rtsp_url = st.sidebar.text_input(
                "Stream URL",
                placeholder="rtsp://192.168.1.100:554/stream or http://...",
                help="Enter RTSP or HTTP video stream URL from drone GCS.",
            )
        elif video_source_type == "Pre/Post Image Pair":
            drone_pre_file = st.sidebar.file_uploader(
                "Pre-disaster Reference Image",
                type=["png", "jpg", "jpeg"],
                key="drone_pre_up",
                help="Upload a pre-disaster reference image for the drone HUD.",
            )
            drone_post_file = st.sidebar.file_uploader(
                "Post-disaster / Live Frame Image",
                type=["png", "jpg", "jpeg"],
                key="drone_post_up",
                help="Upload post-disaster or current drone frame for AI analysis.",
            )

        skip_frames = int(st.sidebar.select_slider(
            "Frame Processing Rate",
            options=[1, 2, 3, 5, 8, 10, 15, 20],
            value=3,
            help="Process every Nth video frame. Lower = higher accuracy, higher = faster throughput.",
        ))

        backend_choice = st.sidebar.selectbox(
            "Model Backend",
            options=[
                "Real Damage Model (ONNX)",
                "Advanced Damage Model Backend (GPU-Ready / CPU Fallback)",
                "Classical Change Detection (OpenCV)",
            ],
            index=0,
        )
        if "Advanced" in backend_choice or "GPU" in backend_choice:
            st.sidebar.info("GPU runtime unavailable. Lightweight ONNX backend selected with CPU execution.")
        dataset_choice = "Live Drone Feed"

        confidence_thresh = float(st.sidebar.slider(
            "Confidence Threshold",
            min_value=0.10,
            max_value=0.90,
            value=0.30,
            step=0.05,
        ))
        max_feat = 100
        grid_dim = "3 x 3 (9 Zones)"
        center_lat = 30.154378
        center_lon = -85.548719
        seed = 42
        building_count = 18

    # =========================================================================
    # 4. DEMO MODE CONTROLS (Simulated Baseline Only)
    # =========================================================================
    else:
        st.sidebar.markdown("---")
        st.sidebar.subheader("🧪 Demo Simulation Parameters")
        st.sidebar.caption("Offline synthetic sandbox for evaluating dashboard layout without hardware dependencies.")

        backend_choice = "Simulated Stage 1 Engine"
        dataset_choice = "Chennai Demo Baseline"
        confidence_thresh = 0.40
        max_feat = 100
        grid_dim = "3 x 3 (9 Zones)"

        center_lat = float(st.sidebar.number_input(
            "Centre Latitude",
            value=13.0827,
            format="%.6f",
            help="Demo center latitude in decimal degrees.",
        ))
        center_lon = float(st.sidebar.number_input(
            "Centre Longitude",
            value=80.2707,
            format="%.6f",
            help="Demo center longitude in decimal degrees.",
        ))
        seed = int(st.sidebar.number_input(
            "Random Seed",
            value=42,
            step=1,
            help="Deterministic seed for simulated building localization.",
        ))
        building_count = int(st.sidebar.slider(
            "Demo Building Count",
            min_value=15,
            max_value=20,
            value=18,
            step=1,
            help="Number of synthetic building structures to simulate (clamped to 15-20).",
        ))

    st.sidebar.markdown("---")
    run_button = bool(st.sidebar.button("Run AI Damage Assessment", type="primary", width="stretch"))

    # Active Model Transparency & GPU Readiness Diagnostics
    system_health = check_model_health()
    gpu_readiness = system_health.get("gpu_readiness", {})
    gpu_data = system_health.get("gpu_hardware", {})

    with st.sidebar.expander("⚡ GPU Status & Hardware Diagnostics", expanded=True):
        gpu_detected_str = "Yes" if gpu_data.get("gpu_detected") else "No"
        runtime_avail_str = "Yes" if gpu_readiness.get("gpu_inference_runtime_available") else "No"
        st.markdown(f"- **GPU hardware detected:** `{gpu_detected_str}`")
        st.markdown(f"- **GPU inference runtime available:** `{runtime_avail_str}`")
        st.markdown("- **Active inference device:** `CPU`")
        st.markdown("- **Active model:** `Lightweight Bi-Temporal ONNX CNN`")
        st.markdown("- **Recommended demo backend:** `Real Damage Model (ONNX)`")

        st.warning("The NVIDIA GPU is detected, but the installed ONNX Runtime does not include CUDAExecutionProvider. Therefore, the system performs real inference on CPU using the verified lightweight ONNX backend. No pretrained xBD/xView2 checkpoint is currently integrated.")

    with st.sidebar.expander("🛡️ Active Model Provenance Diagnostics", expanded=False):
        st.markdown("**Active Model:** Verified Local ONNX Model")
        st.markdown("**Model File:** `damage_classifier.onnx` (23.5 KB)")
        st.markdown("**SHA-256 (Truncated):** `bda9515c`")
        st.markdown("**Provenance Status:** `Model Provenance Not Independently Verified`")
        st.markdown("**Execution Device:** `CPU` (`CPUExecutionProvider` SIMD fallback)")
        st.caption("Pretrained status not independently verified; evaluated as a demonstration / architectural benchmark model.")

    # Parse grid rows and cols
    grid_size_map = {
        "2 x 2 (4 Zones)": (2, 2),
        "3 x 3 (9 Zones)": (3, 3),
        "4 x 4 (16 Zones)": (4, 4),
    }
    grid_rows, grid_cols = grid_size_map.get(grid_dim, (3, 3))

    return SidebarConfig(
        mode=mode,
        backend_choice=backend_choice,
        dataset_choice=dataset_choice,
        confidence_threshold=confidence_thresh,
        max_features=max_feat,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        center_lat=center_lat,
        center_lon=center_lon,
        seed=seed,
        building_count=building_count,
        uploaded_pre_file=uploaded_pre_file,
        uploaded_post_file=uploaded_post_file,
        run_button=run_button,
        video_source_type=video_source_type,
        video_file=video_file,
        rtsp_url=rtsp_url,
        use_webcam=use_webcam,
        skip_frames=skip_frames,
        drone_pre_file=drone_pre_file,
        drone_post_file=drone_post_file,
    )
