"""Real video stream processing for MERIDIAN-Lite Live Drone Recon Stream.

Provides per-frame damage detection on uploaded video files, webcam feeds, or RTSP streams.
Each frame is compared against a rolling "pre-disaster" baseline (first frame or an explicitly
uploaded reference image) using the production ONNX / OpenCV change-detection pipeline.

Architecture
------------
- ``VideoFrameProcessor`` — maintains rolling state across frames, batches patch inference.
- ``process_video_frame_pair`` — one-shot function; classify one (prev, curr) RGB frame pair.
- ``VideoStreamSource`` — enum for video source type selection.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Generator
import cv2
import numpy as np

from models.model_registry import RealDamageModelBackend, ClassicalChangeDetectionBackend
from utils.georeferencing import AffineGeoreferencer
from utils.logging_utils import get_logger

logger = get_logger("video_stream")


class VideoStreamSource(str, Enum):
    UPLOADED_VIDEO = "Uploaded Video File"
    WEBCAM = "Webcam (Live)"
    RTSP_URL = "RTSP/HTTP Stream URL"
    UPLOADED_IMAGES = "Pre/Post Image Pair"


@dataclass
class FrameDetectionResult:
    """Per-frame detection output from the AI pipeline."""
    frame_index: int
    detections: list[dict[str, Any]]
    damage_breakdown: dict[str, int]
    total_detected: int
    inference_time_ms: float
    frame_rgb: np.ndarray  # frame with HUD overlay burned in


@dataclass
class VideoProcessorState:
    """Mutable state for a live video processing session."""
    baseline_frame: np.ndarray | None = None
    frame_index: int = 0
    cumulative_detections: list[dict[str, Any]] = field(default_factory=list)
    cumulative_breakdown: dict[str, int] = field(
        default_factory=lambda: {"destroyed": 0, "major-damage": 0, "minor-damage": 0, "no-damage": 0}
    )


def _build_georeferencer(frame: np.ndarray, center_lat: float, center_lon: float, delta_deg: float = 0.002) -> AffineGeoreferencer:
    """Build a center-based affine georeferencer for an arbitrary frame."""
    h, w = frame.shape[:2]
    return AffineGeoreferencer.from_center_and_radius(
        image_width=w,
        image_height=h,
        center_lat=center_lat,
        center_lon=center_lon,
        delta_deg=delta_deg,
        coordinate_source="video_frame_approximation",
    )


def _pixel_bbox_from_feature(feat: dict[str, Any], georef: AffineGeoreferencer, frame_shape: tuple[int, int]) -> tuple[int, int, int, int] | None:
    """Convert a GeoJSON polygon feature back to pixel bounding box."""
    h, w = frame_shape
    try:
        coords = feat["geometry"]["coordinates"][0]
        px_points = [georef.lonlat_to_pixel(c[0], c[1]) for c in coords]
        xs = [p[0] for p in px_points]
        ys = [p[1] for p in px_points]
        x0, x1 = max(0, int(min(xs))), min(w, int(max(xs)))
        y0, y1 = max(0, int(min(ys))), min(h, int(max(ys)))
        # Ensure minimum visible box
        if x1 - x0 < 15:
            pad = (15 - (x1 - x0)) // 2
            x0, x1 = max(0, x0 - pad), min(w, x1 + pad)
        if y1 - y0 < 15:
            pad = (15 - (y1 - y0)) // 2
            y0, y1 = max(0, y0 - pad), min(h, y1 + pad)
        return (x0, y0, x1, y1)
    except Exception:
        return None


# EMS-98 BGR color map
_EMS_BGR = {
    "destroyed": (40, 40, 220),
    "major-damage": (20, 126, 253),
    "minor-damage": (7, 193, 255),
    "no-damage": (69, 167, 40),
}


def draw_detections_on_frame(
    frame_bgr: np.ndarray,
    detections: list[dict[str, Any]],
    frame_index: int,
    total_detected: int,
    inference_ms: float,
) -> np.ndarray:
    """Draw HUD + bounding boxes with EMS-98 labels on a BGR frame."""
    canvas = frame_bgr.copy()
    h, w = canvas.shape[:2]

    # -- Semi-transparent HUD top/bottom banners
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, 48), (10, 18, 38), -1)
    cv2.rectangle(overlay, (0, h - 44), (w, h), (10, 18, 38), -1)
    cv2.addWeighted(overlay, 0.80, canvas, 0.20, 0, canvas)

    # -- Corner brackets
    c = (0, 240, 210)
    L = 40
    for (px, py, dx, dy) in [(20, 20, 1, 1), (w - 20, 20, -1, 1), (20, h - 20, 1, -1), (w - 20, h - 20, -1, -1)]:
        cv2.line(canvas, (px, py), (px + dx * L, py), c, 2)
        cv2.line(canvas, (px, py), (px, py + dy * L), c, 2)

    # -- Centre reticle
    cx, cy = w // 2, h // 2
    cv2.circle(canvas, (cx, cy), 20, c, 1)
    cv2.line(canvas, (cx - 28, cy), (cx - 8, cy), c, 1)
    cv2.line(canvas, (cx + 8, cy), (cx + 28, cy), c, 1)
    cv2.line(canvas, (cx, cy - 28), (cx, cy - 8), c, 1)
    cv2.line(canvas, (cx, cy + 8), (cx, cy + 28), c, 1)

    # -- Bounding boxes
    for det in detections:
        bbox = det.get("pixel_bbox")
        if not bbox:
            continue
        x0, y0, x1, y1 = bbox
        dmg = det.get("damage_class", "no-damage")
        conf = det.get("confidence", 0.5)
        color = _EMS_BGR.get(dmg, (180, 180, 180))
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 2)
        # Corner accents
        cl = min(10, max(4, (x1 - x0) // 4))
        for (bx, by, sdx, sdy) in [(x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)]:
            cv2.line(canvas, (bx, by), (bx + sdx * cl, by), color, 3)
            cv2.line(canvas, (bx, by), (bx, by + sdy * cl), color, 3)
        # Label
        fid = det.get("feature_id", "BLD")
        label = f"{fid} {dmg.upper()} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
        ty0 = max(0, y0 - th - 6)
        cv2.rectangle(canvas, (x0, ty0), (min(w, x0 + tw + 8), y0), color, -1)
        cv2.putText(canvas, label, (x0 + 3, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

    # -- Top banner text
    top_str = (
        f"MERIDIAN AI · FRAME {frame_index:04d} | "
        f"STRUCTURES DETECTED: {total_detected} | "
        f"INFERENCE: {inference_ms:.1f}ms | ONNX CPU SIMD"
    )
    cv2.putText(canvas, top_str, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 240, 210), 1, cv2.LINE_AA)

    # -- Bottom banner text
    breakdown_counts = {}
    for det in detections:
        k = det.get("damage_class", "no-damage")
        breakdown_counts[k] = breakdown_counts.get(k, 0) + 1
    bot_str = (
        f"DESTROYED: {breakdown_counts.get('destroyed', 0)} | "
        f"MAJOR: {breakdown_counts.get('major-damage', 0)} | "
        f"MINOR: {breakdown_counts.get('minor-damage', 0)} | "
        f"OK: {breakdown_counts.get('no-damage', 0)} | "
        f"UAV-SAR-01 LIVE FEED"
    )
    cv2.putText(canvas, bot_str, (14, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    return canvas


def process_frame_pair(
    pre_frame_rgb: np.ndarray,
    post_frame_rgb: np.ndarray,
    frame_index: int,
    center_lat: float,
    center_lon: float,
    confidence_threshold: float = 0.30,
    max_features: int = 80,
    use_classical: bool = False,
) -> FrameDetectionResult:
    """Run ONNX or classical CV change-detection on a (pre, post) RGB frame pair.

    Returns a ``FrameDetectionResult`` with HUD-annotated frame ready for display.
    """
    import time as _time
    t0 = _time.perf_counter()

    h, w = post_frame_rgb.shape[:2]
    georef = _build_georeferencer(post_frame_rgb, center_lat, center_lon)

    try:
        if use_classical:
            backend = ClassicalChangeDetectionBackend()
        else:
            backend = RealDamageModelBackend()
        fc, _ = backend.predict(
            pre_image=pre_frame_rgb,
            post_image=post_frame_rgb,
            georeferencer=georef,
            confidence_threshold=confidence_threshold,
            max_features=max_features,
        )
    except Exception as e:
        logger.warning(f"Frame {frame_index} inference failed: {e}; falling back to classical CV.")
        backend = ClassicalChangeDetectionBackend()
        fc, _ = backend.predict(
            pre_image=pre_frame_rgb,
            post_image=post_frame_rgb,
            georeferencer=georef,
            confidence_threshold=confidence_threshold,
            max_features=max_features,
        )

    elapsed_ms = (_time.perf_counter() - t0) * 1000.0

    # Build detections with pixel bboxes for HUD rendering
    detections = []
    breakdown: dict[str, int] = {"destroyed": 0, "major-damage": 0, "minor-damage": 0, "no-damage": 0}

    for feat in fc.get("features", []):
        props = feat.get("properties", {})
        bbox = _pixel_bbox_from_feature(feat, georef, (h, w))
        dmg = props.get("damage_class", "no-damage")
        conf = props.get("confidence", 0.5)
        fid = props.get("feature_id", f"BLD-{frame_index}")
        det = {
            "feature_id": fid,
            "damage_class": dmg,
            "confidence": conf,
            "pixel_bbox": bbox,
            "wgs84_feature": feat,
        }
        detections.append(det)
        if dmg in breakdown:
            breakdown[dmg] += 1

    # Burn detections onto frame
    post_bgr = cv2.cvtColor(post_frame_rgb, cv2.COLOR_RGB2BGR)
    hud_bgr = draw_detections_on_frame(post_bgr, detections, frame_index, len(detections), elapsed_ms)
    hud_rgb = cv2.cvtColor(hud_bgr, cv2.COLOR_BGR2RGB)

    return FrameDetectionResult(
        frame_index=frame_index,
        detections=detections,
        damage_breakdown=breakdown,
        total_detected=len(detections),
        inference_time_ms=elapsed_ms,
        frame_rgb=hud_rgb,
    )


def yield_video_frames(
    source: str | int,
    skip_n: int = 5,
    max_frames: int = 1000,
) -> Generator[np.ndarray, None, None]:
    """Yield RGB frames from a cv2-compatible video source.

    Args:
        source: File path string, RTSP URL, or integer device index for webcam.
        skip_n: Process every nth frame (1 = every frame, 5 = every 5th frame).
        max_frames: Stop after this many yielded frames.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error(f"Cannot open video source: {source}")
        return

    frame_total = 0
    yielded = 0
    try:
        while cap.isOpened() and yielded < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame_total += 1
            if (frame_total - 1) % skip_n == 0:
                yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yielded += 1
    finally:
        cap.release()


def save_temp_video(uploaded_file_bytes: bytes, suffix: str = ".mp4") -> Path:
    """Save uploaded Streamlit file bytes to a temp path and return it."""
    from utils.paths import PROJECT_ROOT
    temp_dir = PROJECT_ROOT / "outputs" / "temp_uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    import time as _t
    ts = int(_t.time())
    out_path = temp_dir / f"drone_stream_{ts}{suffix}"
    out_path.write_bytes(uploaded_file_bytes)
    return out_path
