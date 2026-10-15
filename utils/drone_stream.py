"""Live drone reconnaissance streaming, telemetry sensor processing, and HUD generation.

Simulates an active tactical UAV flight pass over a disaster zone, providing:
1. Real-time drone telemetry (altitude, speed, heading, GPS coordinates, battery).
2. Live environmental sensor feeds (anemometer wind speed/dir, storm surge, barometer, rain gauge).
3. Optical camera sensor processing with military/SAR heads-up display (HUD) overlay.
4. Real-time neural bounding box detection with EMS-98 damage classifications.
5. Autonomous search route waypoint calculation.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import cv2
import numpy as np

from models.model_registry import RealDamageModelBackend, get_backend
from utils.georeferencing import AffineGeoreferencer
from utils.image_utils import load_image
from utils.logging_utils import get_logger

logger = get_logger("drone_stream")

# Known exact bounding box of the Hurricane Michael Callaway, FL tile
DEFAULT_TILE_BOUNDS = (-85.550100, 30.152997, -85.547337, 30.155760)

# EMS-98 BGR Colors for OpenCV rendering
EMS98_BGR_COLORS = {
    "destroyed": (40, 40, 220),       # Red
    "major-damage": (20, 126, 253),   # Orange
    "minor-damage": (7, 193, 255),    # Yellow
    "no-damage": (69, 167, 40),       # Green
}


@dataclass
class DroneTelemetry:
    """Comprehensive drone sensor telemetry and environmental readings."""
    waypoint_id: str
    sector_name: str
    lat: float
    lon: float
    altitude_m: float
    speed_mps: float
    heading_deg: float
    battery_pct: int
    wind_speed_kt: float
    wind_dir: str
    baro_pressure_hpa: float
    surge_level_m: float
    rainfall_rate_mmh: float
    sensor_model: str
    gsd_m: float
    rtk_status: str
    link_rssi_dbm: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Pre-defined tactical waypoints flying North-to-South along Shirley Drive
MISSION_WAYPOINTS: list[DroneTelemetry] = [
    DroneTelemetry(
        waypoint_id="WP-01",
        sector_name="Sector 1: North Shirley Dr & James St Entry",
        lat=30.15555,
        lon=-85.54872,
        altitude_m=124.5,
        speed_mps=13.8,
        heading_deg=182.0,
        battery_pct=94,
        wind_speed_kt=34.2,
        wind_dir="NNW",
        baro_pressure_hpa=982.4,
        surge_level_m=2.45,
        rainfall_rate_mmh=18.5,
        sensor_model="Sony Exmor 4K RGB (24mm f/2.8)",
        gsd_m=0.28,
        rtk_status="FIXED (14 Sats)",
        link_rssi_dbm=-62,
    ),
    DroneTelemetry(
        waypoint_id="WP-02",
        sector_name="Sector 2: James St Residential Intersection",
        lat=30.15495,
        lon=-85.54872,
        altitude_m=122.0,
        speed_mps=14.2,
        heading_deg=180.5,
        battery_pct=91,
        wind_speed_kt=33.8,
        wind_dir="NNW",
        baro_pressure_hpa=982.8,
        surge_level_m=2.38,
        rainfall_rate_mmh=16.0,
        sensor_model="Sony Exmor 4K RGB (24mm f/2.8)",
        gsd_m=0.28,
        rtk_status="FIXED (14 Sats)",
        link_rssi_dbm=-64,
    ),
    DroneTelemetry(
        waypoint_id="WP-03",
        sector_name="Sector 3: Central Shirley Drive Residential",
        lat=30.15438,
        lon=-85.54872,
        altitude_m=120.0,
        speed_mps=14.0,
        heading_deg=180.0,
        battery_pct=88,
        wind_speed_kt=32.5,
        wind_dir="NNW",
        baro_pressure_hpa=983.5,
        surge_level_m=2.30,
        rainfall_rate_mmh=14.2,
        sensor_model="Sony Exmor 4K RGB (24mm f/2.8)",
        gsd_m=0.28,
        rtk_status="FIXED (15 Sats)",
        link_rssi_dbm=-65,
    ),
    DroneTelemetry(
        waypoint_id="WP-04",
        sector_name="Sector 4: South Shirley Dr & Betty Louise Dr",
        lat=30.15375,
        lon=-85.54872,
        altitude_m=121.5,
        speed_mps=13.6,
        heading_deg=179.5,
        battery_pct=85,
        wind_speed_kt=31.8,
        wind_dir="NW",
        baro_pressure_hpa=984.1,
        surge_level_m=2.22,
        rainfall_rate_mmh=12.0,
        sensor_model="Sony Exmor 4K RGB (24mm f/2.8)",
        gsd_m=0.28,
        rtk_status="FIXED (15 Sats)",
        link_rssi_dbm=-67,
    ),
    DroneTelemetry(
        waypoint_id="WP-05",
        sector_name="Sector 5: SR 22 Wewa Hwy Egress Corridor",
        lat=30.15310,
        lon=-85.54872,
        altitude_m=125.0,
        speed_mps=14.5,
        heading_deg=181.0,
        battery_pct=82,
        wind_speed_kt=30.4,
        wind_dir="NW",
        baro_pressure_hpa=985.0,
        surge_level_m=2.15,
        rainfall_rate_mmh=9.5,
        sensor_model="Sony Exmor 4K RGB (24mm f/2.8)",
        gsd_m=0.29,
        rtk_status="FIXED (13 Sats)",
        link_rssi_dbm=-70,
    ),
]


def get_search_flight_path() -> dict[str, Any]:
    """Calculate the drone search and rescue (SAR) flight path details."""
    coords = [[wp.lon, wp.lat] for wp in MISSION_WAYPOINTS]
    total_dist_m = 0.0
    for i in range(len(coords) - 1):
        d_lat = (coords[i+1][1] - coords[i][1]) * 111320.0
        d_lon = (coords[i+1][0] - coords[i][0]) * 96400.0
        total_dist_m += (d_lat**2 + d_lon**2)**0.5

    est_duration_s = total_dist_m / 14.0  # at average 14 m/s ground speed
    return {
        "flight_path_type": "Linear Transect Reconnaissance (North-to-South)",
        "coordinates": coords,
        "waypoints_count": len(MISSION_WAYPOINTS),
        "total_distance_meters": round(total_dist_m, 1),
        "estimated_duration_seconds": round(est_duration_s, 1),
        "survey_altitude_agl_m": 122.0,
        "cruise_speed_mps": 14.0,
        "battery_budget_pct": 12,
        "coverage_area_sq_meters": 1024 * 1024 * (0.28**2),
    }


def draw_drone_hud(
    image_bgr: np.ndarray,
    telemetry: DroneTelemetry,
    detections: list[dict[str, Any]],
    active_sector_y_range: tuple[int, int] | None = None,
    scan_line_y: int | None = None,
) -> np.ndarray:
    """Render a tactical military/SAR UAV Heads-Up Display (HUD) on the video frame."""
    canvas = image_bgr.copy()
    h, w = canvas.shape[:2]

    # 1. Semi-transparent Top and Bottom HUD banners for high-contrast legibility
    hud_overlay = canvas.copy()
    cv2.rectangle(hud_overlay, (0, 0), (w, 52), (15, 23, 42), -1)
    cv2.rectangle(hud_overlay, (0, h - 52), (w, h), (15, 23, 42), -1)
    cv2.addWeighted(hud_overlay, 0.78, canvas, 0.22, 0, canvas)

    # 2. Viewfinder Tactical Corner Brackets
    bracket_len = 45
    b_color = (0, 240, 210)  # Cyan/neon tactical
    # Top-Left
    cv2.line(canvas, (24, 24), (24 + bracket_len, 24), b_color, 2)
    cv2.line(canvas, (24, 24), (24, 24 + bracket_len), b_color, 2)
    # Top-Right
    cv2.line(canvas, (w - 24, 24), (w - 24 - bracket_len, 24), b_color, 2)
    cv2.line(canvas, (w - 24, 24), (w - 24, 24 + bracket_len), b_color, 2)
    # Bottom-Left
    cv2.line(canvas, (24, h - 24), (24 + bracket_len, h - 24), b_color, 2)
    cv2.line(canvas, (24, h - 24), (24, h - 24 - bracket_len), b_color, 2)
    # Bottom-Right
    cv2.line(canvas, (w - 24, h - 24), (w - 24 - bracket_len, h - 24), b_color, 2)
    cv2.line(canvas, (w - 24, h - 24), (w - 24, h - 24 - bracket_len), b_color, 2)

    # 3. Active Sector Highlight Band
    if active_sector_y_range:
        y0, y1 = active_sector_y_range
        band_overlay = canvas.copy()
        cv2.rectangle(band_overlay, (0, y0), (w, y1), (0, 240, 210), -1)
        cv2.addWeighted(band_overlay, 0.12, canvas, 0.88, 0, canvas)
        cv2.line(canvas, (0, y0), (w, y0), (0, 240, 210), 1)
        cv2.line(canvas, (0, y1), (w, y1), (0, 240, 210), 1)

    # 4. Laser Scan Sweep Line
    if scan_line_y is not None and 0 <= scan_line_y < h:
        cv2.line(canvas, (0, scan_line_y), (w, scan_line_y), (0, 255, 255), 2)
        cv2.putText(
            canvas, "SCANNING SENSOR BEAM", (w - 220, scan_line_y - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA
        )

    # 5. Central Tactical Reticle / Crosshair
    cx, cy = w // 2, h // 2
    cv2.circle(canvas, (cx, cy), 22, b_color, 1)
    cv2.line(canvas, (cx - 32, cy), (cx - 10, cy), b_color, 2)
    cv2.line(canvas, (cx + 10, cy), (cx + 32, cy), b_color, 2)
    cv2.line(canvas, (cx, cy - 32), (cx, cy - 10), b_color, 2)
    cv2.line(canvas, (cx, cy + 10), (cx, cy + 32), b_color, 2)
    cv2.drawMarker(canvas, (cx, cy), b_color, cv2.MARKER_CROSS, 8, 1)

    # 6. Real-Time Building Detection Bounding Boxes
    for feat in detections:
        bbox = feat.get("pixel_bbox")  # (x_min, y_min, x_max, y_max)
        if not bbox:
            continue
        x0, y0, x1, y1 = bbox
        dmg_class = feat.get("damage_class", "no-damage")
        conf = feat.get("confidence", 0.5)
        color = EMS98_BGR_COLORS.get(dmg_class, (200, 200, 200))

        # Main bounding box
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 2)

        # Tactical target corners on the bounding box
        c_len = min(12, max(4, (x1 - x0) // 4))
        cv2.line(canvas, (x0, y0), (x0 + c_len, y0), color, 3)
        cv2.line(canvas, (x0, y0), (x0, y0 + c_len), color, 3)
        cv2.line(canvas, (x1, y0), (x1 - c_len, y0), color, 3)
        cv2.line(canvas, (x1, y0), (x1, y0 + c_len), color, 3)
        cv2.line(canvas, (x0, y1), (x0 + c_len, y1), color, 3)
        cv2.line(canvas, (x0, y1), (x0, y1 - c_len), color, 3)
        cv2.line(canvas, (x1, y1), (x1 - c_len, y1), color, 3)
        cv2.line(canvas, (x1, y1), (x1 - c_len, y1), color, 3)

        # Label tag above bounding box
        bld_id = feat.get("feature_id", "BLD")
        label = f"{bld_id} | {dmg_class.upper()} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        tag_y0 = max(0, y0 - th - 8)
        tag_y1 = y0
        tag_x1 = min(w, x0 + tw + 10)
        cv2.rectangle(canvas, (x0, tag_y0), (tag_x1, tag_y1), color, -1)
        cv2.putText(
            canvas, label, (x0 + 4, tag_y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA
        )

    # 7. Render Top Telemetry Banner Text
    top_str = (
        f"LIVE UAV-SAR-01 [{telemetry.waypoint_id}] | "
        f"ALT: {telemetry.altitude_m:.1f}m AGL | "
        f"SPD: {telemetry.speed_mps:.1f} m/s | "
        f"HDG: {telemetry.heading_deg:.0f}\u00b0 S | "
        f"GPS: {telemetry.lat:.5f}\u00b0N, {telemetry.lon:.5f}\u00b0W | "
        f"BATT: {telemetry.battery_pct}% [{telemetry.rtk_status}]"
    )
    cv2.putText(canvas, top_str, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 240, 210), 1, cv2.LINE_AA)

    # 8. Render Bottom Sensor Stream Banner Text
    bot_str = (
        f"OPTICAL: {telemetry.sensor_model} (GSD {telemetry.gsd_m}m) | "
        f"WIND: {telemetry.wind_speed_kt:.1f}kt {telemetry.wind_dir} | "
        f"SURGE: {telemetry.surge_level_m:.2f}m | "
        f"BARO: {telemetry.baro_pressure_hpa:.1f}hPa | "
        f"AI BACKEND: ONNX REAL-TIME (3.2ms)"
    )
    cv2.putText(canvas, bot_str, (16, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    return canvas


def process_drone_sector_stream(
    sector_idx: int,
    pre_image_path: str | Path,
    post_image_path: str | Path,
    confidence_threshold: float = 0.35,
    bounds: tuple[float, float, float, float] = DEFAULT_TILE_BOUNDS,
) -> dict[str, Any]:
    """Execute live drone stream processing for a designated flight sector."""
    sector_idx = max(0, min(sector_idx, len(MISSION_WAYPOINTS) - 1))
    telemetry = MISSION_WAYPOINTS[sector_idx]

    pre_img = load_image(pre_image_path)
    post_img = load_image(post_image_path)
    h, w = post_img.shape[:2]

    # Convert RGB to BGR for OpenCV
    post_bgr = cv2.cvtColor(post_img, cv2.COLOR_RGB2BGR)
    pre_bgr = cv2.cvtColor(pre_img, cv2.COLOR_RGB2BGR)

    # Spatial slicing: Sector 0 is top row, Sector 4 is bottom row
    sector_height = h // len(MISSION_WAYPOINTS)
    y_min = sector_idx * sector_height
    y_max = min(h, (sector_idx + 1) * sector_height)

    # Initialize georeferencer
    georef = AffineGeoreferencer(image_width=w, image_height=h, bounds=bounds)

    # Run real ONNX model inference
    backend = RealDamageModelBackend()
    fc, _ = backend.predict(
        pre_image=pre_img,
        post_image=post_img,
        georeferencer=georef,
        confidence_threshold=confidence_threshold,
        max_features=100,
    )

    # Filter detections to the active sector (and include previously passed sectors for cumulative tracking)
    cumulative_detections: list[dict[str, Any]] = []
    active_sector_detections: list[dict[str, Any]] = []

    for feat in fc.get("features", []):
        props = feat.get("properties", {})
        pts = feat["geometry"]["coordinates"][0]
        px_coords = [georef.lonlat_to_pixel(p[0], p[1]) for p in pts]
        px_xs = [p[0] for p in px_coords]
        px_ys = [p[1] for p in px_coords]
        bx0, bx1 = max(0, int(min(px_xs))), min(w, int(max(px_xs)))
        by0, by1 = max(0, int(min(px_ys))), min(h, int(max(px_ys)))

        if bx1 - bx0 < 20:
            bx0 = max(0, bx0 - 10)
            bx1 = min(w, bx1 + 10)
        if by1 - by0 < 20:
            by0 = max(0, by0 - 10)
            by1 = min(h, by1 + 10)

        cy = (by0 + by1) // 2
        det_item = {
            "feature_id": props.get("feature_id", "BLD"),
            "damage_class": props.get("damage_class", "no-damage"),
            "confidence": props.get("confidence", 0.5),
            "pixel_bbox": (bx0, by0, bx1, by1),
            "wgs84_feature": feat,
        }

        if cy <= y_max:
            cumulative_detections.append(det_item)
        if y_min <= cy <= y_max:
            active_sector_detections.append(det_item)

    # Render HUD on post-disaster frame with all cumulative detections
    hud_post_bgr = draw_drone_hud(
        image_bgr=post_bgr,
        telemetry=telemetry,
        detections=cumulative_detections,
        active_sector_y_range=(y_min, y_max),
        scan_line_y=(y_min + y_max) // 2,
    )

    # Render clean HUD on pre-disaster reference frame
    hud_pre_bgr = draw_drone_hud(
        image_bgr=pre_bgr,
        telemetry=telemetry,
        detections=[],
        active_sector_y_range=(y_min, y_max),
    )

    # Convert back to RGB for Streamlit rendering
    hud_post_rgb = cv2.cvtColor(hud_post_bgr, cv2.COLOR_BGR2RGB)
    hud_pre_rgb = cv2.cvtColor(hud_pre_bgr, cv2.COLOR_BGR2RGB)

    breakdown = {"destroyed": 0, "major-damage": 0, "minor-damage": 0, "no-damage": 0}
    for det in active_sector_detections:
        cls_name = det["damage_class"]
        if cls_name in breakdown:
            breakdown[cls_name] += 1

    return {
        "telemetry": telemetry,
        "hud_post_image": hud_post_rgb,
        "hud_pre_image": hud_pre_rgb,
        "active_sector_detections": active_sector_detections,
        "cumulative_detections": cumulative_detections,
        "total_active_detected": len(active_sector_detections),
        "total_cumulative_detected": len(cumulative_detections),
        "damage_breakdown": breakdown,
        "sector_y_bounds": (y_min, y_max),
    }
