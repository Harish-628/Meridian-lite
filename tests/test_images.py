"""
Unit tests for image loading, validation, overlay generation, and synthetic demo asset creation.
"""

from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from utils.image_utils import (
    create_damage_overlay,
    create_demo_images,
    image_dimensions,
    load_image,
    validate_image,
)


def test_create_demo_images(tmp_path: Path):
    """Verify demo images can be generated locally and have non-empty valid content."""
    pre_path, post_path = create_demo_images(tmp_path)
    assert pre_path.exists()
    assert post_path.exists()
    assert pre_path.stat().st_size > 0
    assert post_path.stat().st_size > 0

    assert validate_image(pre_path) is True
    assert validate_image(post_path) is True

    pre_img = load_image(pre_path)
    post_img = load_image(post_path)

    assert isinstance(pre_img, np.ndarray)
    assert isinstance(post_img, np.ndarray)
    assert pre_img.shape == (800, 800, 3)
    assert post_img.shape == (800, 800, 3)


def test_validate_image_missing_file(tmp_path: Path):
    """Verify missing file raises FileNotFoundError."""
    missing = tmp_path / "non_existent.png"
    with pytest.raises(FileNotFoundError):
        validate_image(missing)


def test_validate_image_empty_file(tmp_path: Path):
    """Verify 0-byte image file raises ValueError."""
    empty_file = tmp_path / "empty.png"
    empty_file.touch()
    with pytest.raises(ValueError, match="empty"):
        validate_image(empty_file)


def test_validate_image_corrupt_file(tmp_path: Path):
    """Verify non-image content with .png extension raises ValueError."""
    corrupt_file = tmp_path / "corrupt.png"
    corrupt_file.write_text("NOT_AN_IMAGE_DATA")
    with pytest.raises(ValueError):
        validate_image(corrupt_file)


def test_validate_unsupported_extension(tmp_path: Path):
    """Verify unsupported file extension raises ValueError."""
    txt_file = tmp_path / "sample.txt"
    txt_file.write_text("Hello")
    with pytest.raises(ValueError, match="Unsupported image extension"):
        validate_image(txt_file)


def test_image_dimensions(tmp_path: Path):
    """Verify image dimensions accurately report (width, height)."""
    img = Image.new("RGB", (320, 240), color="blue")
    img_path = tmp_path / "test_dim.png"
    img.save(img_path)

    dims = image_dimensions(img_path)
    assert dims == (320, 240)


def test_damage_overlay_different_dimensions(tmp_path: Path):
    """Verify overlay computation succeeds and resizes post image when dimensions differ."""
    img_pre = Image.new("RGB", (600, 400), color="green")
    pre_path = tmp_path / "pre.png"
    img_pre.save(pre_path)

    img_post = Image.new("RGB", (300, 200), color="red")
    post_path = tmp_path / "post.png"
    img_post.save(post_path)

    out_overlay = tmp_path / "overlay.png"
    overlay, out_path = create_damage_overlay(pre_path, post_path, out_overlay)

    assert out_path.exists()
    assert overlay.shape == (400, 600, 3)
    assert out_path.stat().st_size > 0
