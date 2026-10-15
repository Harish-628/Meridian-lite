"""Interactive Live Drone Reconnaissance Streaming & Telemetry View for MERIDIAN-Lite.

Renders real-time drone optical camera stream with tactical HUD, live bounding boxes,
telemetry sensor instrumentation, autonomous search route, and resource allocation.

Supports four input modes:
1. Uploaded video file (.mp4/.avi/.mov) — processed frame by frame with AI inference
2. Webcam (device 0) — live camera stream with real-time damage detection
3. RTSP/HTTP URL — network stream from a drone ground control station
4. Pre/Post image pair — static bi-temporal comparison with AI analysis
"""

from pathlib import Path
from typing import Any
import time
import cv2
import numpy as np
import streamlit as st

from utils.drone_stream import (
    MISSION_WAYPOINTS,
    get_search_flight_path,
    process_drone_sector_stream,
)
from utils.predictive_risk import calculate_predictive_hazard_analysis
from utils.resource_allocation import calculate_tactical_resource_allocation
from utils.video_stream import (
    VideoStreamSource,
    FrameDetectionResult,
    process_frame_pair,
    yield_video_frames,
    save_temp_video,
    draw_detections_on_frame,
)
from utils.image_utils import load_image
from utils.logging_utils import get_logger

logger = get_logger("drone_view")


def _mission_status_banner() -> None:
    """Render top mission communications banner."""
    st.markdown(
        """
        <div style="background: linear-gradient(90deg, #091e3a 0%, #1e3a8a 100%);
                    border: 1.5px solid #3b82f6; border-radius: 8px; padding: 12px 18px; margin-bottom: 16px;
                    display: flex; justify-content: space-between; align-items: center; color: white;">
            <div>
                <span style="font-weight: 700; font-size: 1.05rem; letter-spacing: 0.5px;">
                    🛰️ FLIGHT STATUS: ACTIVE AIR RECONNAISSANCE · MISSION ID: SAR-MERIDIAN-01
                </span><br>
                <span style="font-size: 0.85rem; color: #93c5fd;">
                    Datalink: AES-256 Encrypted (5.8 GHz) · Telemetry: Active · RTK GNSS: Fixed (15 Satellites) · Neural Engine: ONNX CPU SIMD
                </span>
            </div>
            <div style="text-align: right;">
                <span style="background: #ef4444; color: white; padding: 4px 10px; border-radius: 12px; font-weight: 700; font-size: 0.8rem; letter-spacing: 1px;">
                    ● LIVE STREAM
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_telemetry_panel(telem: Any) -> None:
    """Render 4-column telemetry sensor instrumentation panel."""
    t_col1, t_col2, t_col3, t_col4 = st.columns(4)

    with t_col1:
        st.markdown(
            f"""
            <div style="background: #0f172a; border: 1px solid #334155; border-radius: 6px; padding: 12px; color: white;">
                <div style="font-size: 0.78rem; color: #94a3b8; text-transform: uppercase; font-weight: 600;">✈️ Flight Telemetry</div>
                <div style="font-size: 1.35rem; font-weight: 700; color: #38bdf8; margin: 4px 0;">{telem.altitude_m:.1f} m AGL</div>
                <div style="font-size: 0.82rem; color: #cbd5e1;">Speed: <b>{telem.speed_mps:.1f} m/s</b> ({telem.speed_mps*3.6:.0f} km/h)</div>
                <div style="font-size: 0.82rem; color: #cbd5e1;">Heading: <b>{telem.heading_deg:.0f}° South</b></div>
                <div style="font-size: 0.78rem; color: #4ade80; margin-top: 4px;">● GNSS RTK {telem.rtk_status}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with t_col2:
        st.markdown(
            f"""
            <div style="background: #0f172a; border: 1px solid #334155; border-radius: 6px; padding: 12px; color: white;">
                <div style="font-size: 0.78rem; color: #94a3b8; text-transform: uppercase; font-weight: 600;">🌡️ Environmental Sensors</div>
                <div style="font-size: 1.35rem; font-weight: 700; color: #fbbf24; margin: 4px 0;">{telem.wind_speed_kt:.1f} kt {telem.wind_dir}</div>
                <div style="font-size: 0.82rem; color: #cbd5e1;">Barometer: <b>{telem.baro_pressure_hpa:.1f} hPa</b></div>
                <div style="font-size: 0.82rem; color: #cbd5e1;">Storm Surge: <b>{telem.surge_level_m:.2f} m MSL</b></div>
                <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 4px;">Rain Rate: <b>{telem.rainfall_rate_mmh:.1f} mm/h</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with t_col3:
        st.markdown(
            f"""
            <div style="background: #0f172a; border: 1px solid #334155; border-radius: 6px; padding: 12px; color: white;">
                <div style="font-size: 0.78rem; color: #94a3b8; text-transform: uppercase; font-weight: 600;">📷 Optical Sensor Payload</div>
                <div style="font-size: 1.15rem; font-weight: 700; color: #a78bfa; margin: 4px 0;">4K RGB Optical</div>
                <div style="font-size: 0.82rem; color: #cbd5e1;">Model: <b>Sony Exmor 24mm</b></div>
                <div style="font-size: 0.82rem; color: #cbd5e1;">Resolution: <b>{telem.gsd_m:.2f} m/px GSD</b></div>
                <div style="font-size: 0.78rem; color: #a78bfa; margin-top: 4px;">Framerate: <b>15 FPS Low-Latency</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with t_col4:
        st.markdown(
            f"""
            <div style="background: #0f172a; border: 1px solid #334155; border-radius: 6px; padding: 12px; color: white;">
                <div style="font-size: 0.78rem; color: #94a3b8; text-transform: uppercase; font-weight: 600;">⚡ Avionics & Datalink</div>
                <div style="font-size: 1.35rem; font-weight: 700; color: #34d399; margin: 4px 0;">{telem.battery_pct}% Battery</div>
                <div style="font-size: 0.82rem; color: #cbd5e1;">RF Link RSSI: <b>{telem.link_rssi_dbm} dBm</b></div>
                <div style="font-size: 0.82rem; color: #cbd5e1;">Endurance: <b>~24 min remaining</b></div>
                <div style="font-size: 0.78rem; color: #34d399; margin-top: 4px;">Failsafe: <b>RTH Armed (Auto)</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_video_stream(
    video_source: str | int,
    center_lat: float,
    center_lon: float,
    confidence_threshold: float,
    max_features: int,
    skip_frames: int,
    use_classical: bool = False,
) -> None:
    """Process and display a live video stream with AI inference overlaid per frame."""
    st.markdown("### 🎥 Live AI Video Processing Feed")
    st.caption(
        "Each video frame is analyzed by the ONNX damage detection model. "
        "Bounding boxes show structural damage assessments in real time."
    )

    # Placeholders for live update
    frame_placeholder = st.empty()
    stats_placeholder = st.empty()
    progress_bar = st.progress(0)

    # Try to get video metadata first
    cap_probe = cv2.VideoCapture(video_source)
    total_frames = int(cap_probe.get(cv2.CAP_PROP_FRAME_COUNT)) if cap_probe.isOpened() else 0
    fps = cap_probe.get(cv2.CAP_PROP_FPS) or 30.0
    cap_probe.release()

    max_frames_to_show = 200  # Cap to avoid infinite loops
    if total_frames > 0:
        st.caption(f"📹 Video: {total_frames} frames @ {fps:.1f} FPS · Processing every {skip_frames}th frame")

    baseline_frame: np.ndarray | None = None
    cumulative_total = 0
    cumulative_breakdown = {"destroyed": 0, "major-damage": 0, "minor-damage": 0, "no-damage": 0}
    processed_count = 0
    all_frame_results: list[FrameDetectionResult] = []

    stop_btn_col, _ = st.columns([1, 4])
    stop_flag = stop_btn_col.button("⏹ Stop Processing", key="stop_video_stream")

    if stop_flag:
        st.info("Stream stopped by user.")
        return

    try:
        for frame_rgb in yield_video_frames(video_source, skip_n=skip_frames, max_frames=max_frames_to_show):
            if baseline_frame is None:
                # Use first frame as the "pre-disaster" baseline
                baseline_frame = frame_rgb.copy()
                frame_placeholder.image(
                    baseline_frame,
                    caption="📡 Baseline frame captured — starting damage detection...",
                    width="stretch",
                )
                time.sleep(0.3)
                continue

            # Run AI inference on (baseline, current) frame pair
            result = process_frame_pair(
                pre_frame_rgb=baseline_frame,
                post_frame_rgb=frame_rgb,
                frame_index=processed_count,
                center_lat=center_lat,
                center_lon=center_lon,
                confidence_threshold=confidence_threshold,
                max_features=max_features,
                use_classical=use_classical,
            )

            all_frame_results.append(result)
            processed_count += 1
            cumulative_total += result.total_detected
            for k, v in result.damage_breakdown.items():
                cumulative_breakdown[k] = cumulative_breakdown.get(k, 0) + v

            # Update live display
            frame_placeholder.image(
                result.frame_rgb,
                caption=f"Frame {result.frame_index:04d} · {result.total_detected} structures detected · {result.inference_time_ms:.0f}ms inference",
                width="stretch",
            )

            stats_placeholder.markdown(
                f"**🔴 Cumulative Detections:** {cumulative_total} | "
                f"🔴 Destroyed: **{cumulative_breakdown['destroyed']}** | "
                f"🟠 Major: **{cumulative_breakdown['major-damage']}** | "
                f"🟡 Minor: **{cumulative_breakdown['minor-damage']}** | "
                f"🟢 OK: **{cumulative_breakdown['no-damage']}** | "
                f"⏱ Inference: **{result.inference_time_ms:.1f}ms**"
            )

            if total_frames > 0:
                pct = min(1.0, processed_count * skip_frames / max(1, total_frames))
                progress_bar.progress(pct)

            # Refresh baseline every 30 processed frames to account for camera drift
            if processed_count % 30 == 0:
                baseline_frame = frame_rgb.copy()

    except Exception as e:
        st.error(f"❌ Video processing error: {e}")
        logger.error(f"Video stream processing failed: {e}", exc_info=True)

    progress_bar.progress(1.0)
    if processed_count > 0:
        st.success(
            f"✅ Processing complete · {processed_count} frames analyzed · "
            f"{cumulative_total} cumulative structure detections"
        )


def _render_image_pair_stream(
    pre_image_path: Path,
    post_image_path: Path,
    center_lat: float,
    center_lon: float,
    confidence_threshold: float,
    max_features: int,
    selected_sector_idx: int,
) -> dict[str, Any]:
    """Process static image pair through the drone HUD pipeline and return stream data."""
    return process_drone_sector_stream(
        sector_idx=selected_sector_idx,
        pre_image_path=pre_image_path,
        post_image_path=post_image_path,
        confidence_threshold=confidence_threshold,
    )


def render_live_drone_recon_view(
    pre_image_path: Path,
    post_image_path: Path,
    priority_zones: list[dict[str, Any]],
    confidence_threshold: float = 0.35,
    # Video stream params from sidebar
    video_source_type: str = "Pre/Post Image Pair",
    video_file: Any = None,
    rtsp_url: str = "",
    use_webcam: bool = False,
    skip_frames: int = 3,
    max_features: int = 100,
    center_lat: float = 30.154378,
    center_lon: float = -85.548719,
    backend_choice: str = "Real Damage Model (ONNX)",
    drone_pre_file: Any = None,
    drone_post_file: Any = None,
) -> None:
    """Render the full interactive Live Drone Reconnaissance streaming interface."""
    st.markdown("## 🔴 Live Tactical Drone Reconnaissance Stream (UAV-SAR-01)")
    st.caption(
        "Real-time aerial reconnaissance feed processing: Ingests optical sensor video frames, "
        "flight telemetry, and environmental sensors to detect structural damage bounding boxes on the fly."
    )

    _mission_status_banner()

    use_classical = "Classical" in backend_choice

    # =====================================================================
    # Determine actual video source and render accordingly
    # =====================================================================
    is_video_mode = False

    if video_source_type == "Uploaded Video File" and video_file is not None:
        is_video_mode = True
        st.markdown("### 📷 Dual-Sensor Live Optical Feeds")
        st.info(
            f"📁 **Video loaded:** `{video_file.name}` · "
            "Click **▶ Process Video** below to start AI frame analysis."
        )

        col_go, col_info = st.columns([1, 3])
        with col_go:
            start_video = st.button("▶ Process Video", type="primary", key="start_video_btn")
        with col_info:
            st.caption(
                f"Processing every {skip_frames}th frame with {'ONNX' if not use_classical else 'OpenCV Classical'} backend · "
                "Bounding boxes drawn in real time with EMS-98 damage classifications."
            )

        if start_video:
            # Save video to temp file (needed by OpenCV)
            vid_suffix = Path(video_file.name).suffix or ".mp4"
            temp_video_path = save_temp_video(video_file.getvalue(), suffix=vid_suffix)
            _render_video_stream(
                video_source=str(temp_video_path),
                center_lat=center_lat,
                center_lon=center_lon,
                confidence_threshold=confidence_threshold,
                max_features=max_features,
                skip_frames=skip_frames,
                use_classical=use_classical,
            )

    elif video_source_type == "Webcam (Live)" and use_webcam:
        is_video_mode = True
        st.markdown("### 📷 Webcam Live Feed — Real-Time AI Processing")

        col_go, col_stop = st.columns([1, 1])
        start_cam = col_go.button("📷 Start Webcam Stream", type="primary", key="start_webcam_btn")

        if start_cam:
            _render_video_stream(
                video_source=0,
                center_lat=center_lat,
                center_lon=center_lon,
                confidence_threshold=confidence_threshold,
                max_features=max_features,
                skip_frames=skip_frames,
                use_classical=use_classical,
            )

    elif video_source_type == "RTSP/HTTP Stream URL" and rtsp_url.strip():
        is_video_mode = True
        st.markdown(f"### 📡 RTSP Stream: `{rtsp_url}`")

        col_go, _ = st.columns([1, 3])
        start_rtsp = col_go.button("📡 Connect & Process Stream", type="primary", key="start_rtsp_btn")

        if start_rtsp:
            _render_video_stream(
                video_source=rtsp_url.strip(),
                center_lat=center_lat,
                center_lon=center_lon,
                confidence_threshold=confidence_threshold,
                max_features=max_features,
                skip_frames=skip_frames,
                use_classical=use_classical,
            )

    elif video_source_type == "Pre/Post Image Pair" and drone_pre_file and drone_post_file:
        # Use custom uploaded image pair
        is_video_mode = False
        from utils.paths import PROJECT_ROOT
        temp_dir = PROJECT_ROOT / "outputs" / "temp_uploads"
        temp_dir.mkdir(parents=True, exist_ok=True)
        _pre_path = temp_dir / f"drone_custom_pre_{drone_pre_file.name}"
        _post_path = temp_dir / f"drone_custom_post_{drone_post_file.name}"
        _pre_path.write_bytes(drone_pre_file.getvalue())
        _post_path.write_bytes(drone_post_file.getvalue())
        pre_image_path = _pre_path
        post_image_path = _post_path
        st.success(f"✅ Custom image pair loaded: `{drone_pre_file.name}` vs `{drone_post_file.name}`")

    # =====================================================================
    # Standard HUD sector view (image pair mode, including built-in dataset)
    # =====================================================================
    if not is_video_mode or video_source_type == "Pre/Post Image Pair":
        # Sector waypoint selector
        st.markdown("### 📷 Dual-Sensor Live Optical Feeds")
        col_wp, col_btn, col_stats = st.columns([2.5, 1.2, 1.3])

        with col_wp:
            options = [f"{wp.waypoint_id}: {wp.sector_name}" for wp in MISSION_WAYPOINTS]
            selected_idx = st.selectbox(
                "Active Survey Waypoint & Sector",
                options=range(len(options)),
                format_func=lambda i: options[i],
                index=st.session_state.get("drone_sector_idx", 2),
                help="Select the tactical flight sector to stream and analyze in real time.",
            )
            st.session_state["drone_sector_idx"] = selected_idx

        with col_btn:
            st.write("")
            st.write("")
            if st.button("▶ Next Sector", width="stretch", help="Advance drone flight transect to the next sector."):
                next_idx = (selected_idx + 1) % len(MISSION_WAYPOINTS)
                st.session_state["drone_sector_idx"] = next_idx
                st.rerun()

        with col_stats:
            stream_data = _render_image_pair_stream(
                pre_image_path=pre_image_path,
                post_image_path=post_image_path,
                center_lat=center_lat,
                center_lon=center_lon,
                confidence_threshold=confidence_threshold,
                max_features=max_features,
                selected_sector_idx=selected_idx,
            )
            telem = stream_data["telemetry"]
            st.metric(
                label="Inference Latency",
                value=f"{stream_data.get('inference_ms', 3.2):.1f} ms",
                delta="Real-Time ONNX",
                help="CPU SIMD neural inference latency per 64x64 building patch.",
            )

        # Dual viewfinder display
        col_live, col_ref = st.columns(2)
        with col_live:
            st.markdown(
                f"**🔴 Live Post-Disaster Drone Camera Feed (HUD & Bounding Boxes)** — *{telem.waypoint_id}*"
            )
            st.image(
                stream_data["hud_post_image"],
                caption=f"UAV-SAR-01 Optical Stream · {telem.sector_name} · Real-Time Bounding Boxes (EMS-98)",
                width="stretch",
            )
        with col_ref:
            st.markdown("**🔵 Pre-Disaster Baseline Optical Reference Feed** — *Matched Coordinate Viewport*")
            st.image(
                stream_data["hud_pre_image"],
                caption=f"Pre-Disaster Reference Baseline · Flight Corridor Aligned ({telem.lat:.5f}°N, {telem.lon:.5f}°W)",
                width="stretch",
            )

        # Telemetry panel
        st.markdown("### 🛰️ Real-Time Sensor Telemetry & Environmental Stream")
        _render_telemetry_panel(telem)
        st.write("")

        # Sector detection stats
        b_down = stream_data["damage_breakdown"]
        st.markdown(
            f"**Real-Time Detections in Active Sector:** "
            f"🔴 **Destroyed:** {b_down['destroyed']} | "
            f"🟠 **Major Damage:** {b_down['major-damage']} | "
            f"🟡 **Minor Damage:** {b_down['minor-damage']} | "
            f"🟢 **Undamaged:** {b_down['no-damage']} | "
            f"📦 **Cumulative Recon Total:** {stream_data['total_cumulative_detected']} structures identified."
        )
    elif is_video_mode and not (video_file or use_webcam or rtsp_url.strip()):
        # Show upload instructions
        st.info(
            "📤 **Select a video source above** to start live AI drone processing.\n\n"
            "• **Uploaded Video File** — Upload any drone/aerial video (.mp4, .avi, .mov)\n"
            "• **Webcam** — Stream from your connected camera with live frame analysis\n"
            "• **RTSP URL** — Connect to a drone GCS network stream\n"
            "• **Image Pair** — Upload pre/post images for bi-temporal analysis"
        )
    else:
        # Video mode but not yet started — show placeholder telemetry
        # Use mission waypoint 0 for telemetry display
        telem = MISSION_WAYPOINTS[0]
        st.markdown("### 🛰️ Real-Time Sensor Telemetry & Environmental Stream")
        _render_telemetry_panel(telem)

    st.markdown("---")

    # =====================================================================
    # Stage 2: Autonomous Drone Search Route & Mission Plan
    # =====================================================================
    st.markdown("### ✈️ Stage 2: Autonomous Drone Search Route & Flight Plan")
    st.caption("Linear Transect Reconnaissance route computed to maximize building coverage while conserving battery.")

    flight_plan = get_search_flight_path()
    fp_col1, fp_col2, fp_col3, fp_col4 = st.columns(4)
    fp_col1.metric("Flight Distance", f"{flight_plan['total_distance_meters']} m")
    fp_col2.metric("Estimated Flight Duration", f"{flight_plan['estimated_duration_seconds']:.0f} sec")
    fp_col3.metric("Survey Altitude", f"{flight_plan['survey_altitude_agl_m']} m AGL")
    fp_col4.metric("Battery Consumption", f"{flight_plan['battery_budget_pct']}% of Capacity")

    # =====================================================================
    # Stage 2: Tactical Emergency Resource Allocation Matrix
    # =====================================================================
    st.markdown("### 🚑 Stage 2: Tactical Resource Allocation Matrix")
    st.caption(
        "Automated deployment calculator assigning Urban Search & Rescue (USAR) squads, "
        "medical paramedic units, and heavy debris clearing equipment to each priority zone."
    )

    deployments = calculate_tactical_resource_allocation(priority_zones)
    table_rows = []
    for d in deployments:
        table_rows.append({
            "Zone ID": d["zone_id"],
            "Rank": f"#{d['priority_rank']}",
            "Score": f"{d['priority_score']:.3f}",
            "Structures": d["total_buildings"],
            "Destroyed": d["destroyed_count"],
            "USAR Teams": f"🪖 {d['usar_teams_needed']} Squad(s)",
            "Ambulances": f"🚑 {d['paramedic_units_needed']} Unit(s)",
            "Heavy Debris Units": f"🚜 {d['debris_clearance_units']} Unit(s)",
            "Urgency": d["dispatch_urgency"],
            "Tactical Action": d["action_directive"],
        })
    st.dataframe(table_rows, width="stretch", hide_index=True)

    st.markdown("---")

    # =====================================================================
    # Stage 3: Predictive Multi-Hazard Intelligence & Early Warning
    # =====================================================================
    st.markdown("### 🔮 Stage 3: Predictive Multi-Hazard Intelligence & Risk Modeling")
    st.caption(
        "Combines structural damage with historical Hurricane Michael meteorological models, "
        "ground elevation, and storm surge dynamics to forecast compound secondary risks."
    )

    active_telem = MISSION_WAYPOINTS[st.session_state.get("drone_sector_idx", 0)]
    risk_assessments = calculate_predictive_hazard_analysis(
        priority_zones=priority_zones,
        wind_gusts_kt=active_telem.wind_speed_kt,
        surge_height_m=active_telem.surge_level_m,
    )

    risk_rows = []
    for r in risk_assessments:
        risk_rows.append({
            "Zone ID": r["zone_id"],
            "Compound Vulnerability (CVI)": f"{r['compound_vulnerability_index']:.3f}",
            "Risk Tier": r["risk_tier"],
            "Secondary Collapse Hazard": f"{r['secondary_collapse_risk']:.1%}",
            "Surge Inundation Hazard": f"{r['surge_inundation_risk']:.1%}",
            "Access Blockade Hazard": f"{r['access_blockade_risk']:.1%}",
            "Forecasted Threat": r["primary_threat_description"],
            "Preventative Directive": r["preventative_countermeasure"],
        })

    st.dataframe(risk_rows, width="stretch", hide_index=True)
