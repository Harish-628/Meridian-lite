"""Tests for utils/video_stream.py — real video frame processing pipeline.

Verifies per-frame AI inference, HUD rendering, and video frame extraction
using synthetic numpy frames (no actual video file needed).
"""

from pathlib import Path
import numpy as np
import pytest

from utils.video_stream import (
    VideoStreamSource,
    FrameDetectionResult,
    draw_detections_on_frame,
    process_frame_pair,
    save_temp_video,
)


def _make_synthetic_frame(h: int = 256, w: int = 256) -> np.ndarray:
    """Create a deterministic synthetic RGB frame for testing."""
    rng = np.random.default_rng(seed=42)
    return (rng.integers(0, 200, size=(h, w, 3), dtype=np.uint8) + 55).astype(np.uint8)


def test_video_stream_source_enum():
    """VideoStreamSource enum covers all four expected source types."""
    values = {e.value for e in VideoStreamSource}
    assert "Uploaded Video File" in values
    assert "Webcam (Live)" in values
    assert "RTSP/HTTP Stream URL" in values
    assert "Pre/Post Image Pair" in values


def test_draw_detections_on_frame_no_detections():
    """draw_detections_on_frame works with zero detections and returns correct shape."""
    frame = _make_synthetic_frame(320, 480)
    import cv2
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    result = draw_detections_on_frame(
        frame_bgr=frame_bgr,
        detections=[],
        frame_index=0,
        total_detected=0,
        inference_ms=5.2,
    )
    assert result.shape == frame_bgr.shape
    assert result.dtype == np.uint8


def test_draw_detections_on_frame_with_bbox():
    """draw_detections_on_frame correctly renders a detection bounding box."""
    frame = _make_synthetic_frame(256, 256)
    import cv2
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    detections = [
        {
            "feature_id": "BLD-001",
            "damage_class": "major-damage",
            "confidence": 0.85,
            "pixel_bbox": (40, 40, 100, 100),
        }
    ]
    result = draw_detections_on_frame(frame_bgr, detections, 1, 1, 12.4)
    assert result.shape == (256, 256, 3)
    # The result should differ from original due to HUD overlay
    assert not np.array_equal(result, frame_bgr)


def test_process_frame_pair_returns_result():
    """process_frame_pair returns a FrameDetectionResult with correct structure."""
    pre = _make_synthetic_frame(128, 128)
    post = _make_synthetic_frame(128, 128)
    # Introduce some difference so change detection can find something
    post[30:60, 30:60] = 0  # darken a region to simulate damage

    result = process_frame_pair(
        pre_frame_rgb=pre,
        post_frame_rgb=post,
        frame_index=7,
        center_lat=30.154378,
        center_lon=-85.548719,
        confidence_threshold=0.20,
        max_features=50,
        use_classical=True,  # Use classical CV for deterministic test
    )

    assert isinstance(result, FrameDetectionResult)
    assert result.frame_index == 7
    assert isinstance(result.detections, list)
    assert isinstance(result.damage_breakdown, dict)
    assert "destroyed" in result.damage_breakdown
    assert "major-damage" in result.damage_breakdown
    assert isinstance(result.frame_rgb, np.ndarray)
    assert result.frame_rgb.shape[2] == 3  # RGB


def test_process_frame_pair_frame_shape_preserved():
    """Output frame has same spatial dimensions as input."""
    pre = _make_synthetic_frame(240, 320)
    post = _make_synthetic_frame(240, 320)

    result = process_frame_pair(
        pre_frame_rgb=pre,
        post_frame_rgb=post,
        frame_index=0,
        center_lat=30.154378,
        center_lon=-85.548719,
        confidence_threshold=0.40,
        max_features=20,
        use_classical=True,
    )
    assert result.frame_rgb.shape == (240, 320, 3)


def test_process_frame_pair_onnx_backend():
    """process_frame_pair works with ONNX backend (use_classical=False)."""
    pre = _make_synthetic_frame(128, 128)
    post = _make_synthetic_frame(128, 128)
    post[50:80, 50:80] = 10  # simulate structural change

    result = process_frame_pair(
        pre_frame_rgb=pre,
        post_frame_rgb=post,
        frame_index=3,
        center_lat=30.154378,
        center_lon=-85.548719,
        confidence_threshold=0.25,
        max_features=30,
        use_classical=False,  # ONNX
    )
    assert isinstance(result, FrameDetectionResult)
    assert result.inference_time_ms >= 0.0
    assert result.total_detected == len(result.detections)


def test_save_temp_video(tmp_path):
    """save_temp_video saves bytes to a file in temp_uploads directory."""
    dummy_bytes = b"fake video content for test"
    saved_path = save_temp_video(dummy_bytes, suffix=".mp4")
    assert saved_path.exists()
    assert saved_path.suffix == ".mp4"
    assert saved_path.read_bytes() == dummy_bytes
    # Cleanup
    saved_path.unlink(missing_ok=True)
