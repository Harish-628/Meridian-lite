"""
Image validation, loading, overlay generation, and synthetic demo asset creation.
Supports PNG and JPEG formats, validates dimensions and readability,
and generates visual difference overlays.
"""

from pathlib import Path
from typing import Tuple
import cv2
import numpy as np
from PIL import Image, ImageDraw

from utils.logging_utils import get_logger
from utils.paths import DEFAULT_OVERLAY_OUTPUT, DEMO_DATA_DIR

logger = get_logger("image_utils")

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}


def validate_image(path: str | Path) -> bool:
    """
    Validate that an image exists, has a supported format, is readable, and non-empty.
    Raises FileNotFoundError or ValueError if invalid.
    """
    img_path = Path(path)
    if not img_path.exists():
        err_msg = f"Image file does not exist: {img_path}"
        logger.error(err_msg)
        raise FileNotFoundError(err_msg)

    if not img_path.is_file():
        err_msg = f"Path is not a regular file: {img_path}"
        logger.error(err_msg)
        raise ValueError(err_msg)

    if img_path.suffix not in SUPPORTED_EXTENSIONS:
        err_msg = (
            f"Unsupported image extension '{img_path.suffix}' for {img_path.name}. "
            f"Supported extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
        logger.error(err_msg)
        raise ValueError(err_msg)

    if img_path.stat().st_size == 0:
        err_msg = f"Image file is empty (0 bytes): {img_path}"
        logger.error(err_msg)
        raise ValueError(err_msg)

    try:
        with Image.open(img_path) as img:
            img.verify()
    except Exception as exc:
        err_msg = f"Corrupt or unreadable image file {img_path}: {exc}"
        logger.error(err_msg)
        raise ValueError(err_msg) from exc

    return True


def load_image(path: str | Path) -> np.ndarray:
    """
    Load an image from disk as an RGB NumPy ndarray.
    Validates file integrity and ensures non-empty dimensions.
    """
    validate_image(path)
    img_path = Path(path)

    # Load via OpenCV (handling BGR -> RGB)
    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if bgr is None or bgr.size == 0:
        err_msg = f"Failed to decode image content: {img_path}"
        logger.error(err_msg)
        raise ValueError(err_msg)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def image_dimensions(path: str | Path) -> Tuple[int, int]:
    """
    Retrieve image dimensions as (width, height) without decoding full array if possible.
    """
    validate_image(path)
    with Image.open(path) as img:
        return img.size  # (width, height)


def create_damage_overlay(
    pre_image_path: str | Path,
    post_image_path: str | Path,
    output_path: str | Path | None = None,
) -> Tuple[np.ndarray, Path]:
    """
    Compute a visible pre/post difference overlay and save to disk.
    If image dimensions differ, resizes the post-disaster image to match the pre-disaster image.
    Note: Clearly labeled as a visual comparison, not model segmentation.
    """
    logger.info("Computing pre/post damage difference overlay...")
    pre_img = load_image(pre_image_path)
    post_img = load_image(post_image_path)

    pre_h, pre_w = pre_img.shape[:2]
    post_h, post_w = post_img.shape[:2]

    # Handle dimension mismatches safely
    if (pre_w, pre_h) != (post_w, post_h):
        logger.warning(
            f"Image dimensions differ: pre=({pre_w}x{pre_h}), post=({post_w}x{post_h}). "
            f"Safely resizing post-disaster image to match pre-disaster dimensions."
        )
        post_img = cv2.resize(post_img, (pre_w, pre_h), interpolation=cv2.INTER_LINEAR)

    # Calculate absolute difference
    abs_diff = cv2.absdiff(pre_img, post_img)
    gray_diff = cv2.cvtColor(abs_diff, cv2.COLOR_RGB2GRAY)

    # Apply Gaussian blur and threshold to isolate prominent structural shifts
    blurred = cv2.GaussianBlur(gray_diff, (7, 7), 0)
    _, mask = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)

    # Color map for diff (Red/Orange highlighting for damage zones)
    heat_color = np.zeros_like(post_img)
    heat_color[:, :, 0] = 230  # Red channel
    heat_color[:, :, 1] = 50   # Green channel
    heat_color[:, :, 2] = 40   # Blue channel

    # Blend post image with heat color where mask is active
    mask_3d = np.repeat(mask[:, :, np.newaxis] / 255.0, 3, axis=2)
    overlay = (post_img * (1 - 0.6 * mask_3d) + heat_color * (0.6 * mask_3d)).astype(np.uint8)

    # Target save path
    out_target = Path(output_path) if output_path else DEFAULT_OVERLAY_OUTPUT
    out_target.parent.mkdir(parents=True, exist_ok=True)

    # Save to disk as RGB -> BGR
    bgr_overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_target), bgr_overlay)
    logger.info(f"Visual difference overlay saved to: {out_target}")

    return overlay, out_target


def create_demo_images(
    output_directory: str | Path | None = None,
) -> Tuple[Path, Path]:
    """
    Generate synthetic pre-disaster and post-disaster aerial-style images locally.
    Contains terrain background, roads, and distinct building polygons with realistic roof colors.
    The post-disaster image includes collapsed roofs, debris textures, and rubble.
    """
    out_dir = Path(output_directory) if output_directory else DEMO_DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    pre_path = out_dir / "pre_disaster.png"
    post_path = out_dir / "post_disaster.png"

    width, height = 800, 800

    # Base terrain (aerial green/tan grass landscape)
    np.random.seed(42)
    noise = np.random.randint(-15, 15, (height, width, 3), dtype=np.int16)
    base_color = np.array([120, 155, 110], dtype=np.int16)  # Aerial vegetation green
    terrain = np.clip(base_color + noise, 0, 255).astype(np.uint8)

    pre_img = Image.fromarray(terrain, "RGB")
    draw_pre = ImageDraw.Draw(pre_img)

    # Road network (gray asphalt roads)
    road_color = (80, 85, 90)
    road_border = (60, 65, 70)
    # Horizontal avenue
    draw_pre.rectangle([(0, 380), (800, 420)], fill=road_color, outline=road_border, width=2)
    # Vertical boulevard
    draw_pre.rectangle([(380, 0), (420, 800)], fill=road_color, outline=road_border, width=2)
    # Diagonal residential street
    draw_pre.line([(0, 150), (800, 650)], fill=road_color, width=22)

    # Synthetic buildings layout (18 structured residential and commercial building blocks)
    building_specs = [
        # Sector 1: Top-Left
        (100, 80, 70, 60, (190, 175, 160)),
        (220, 90, 80, 70, (200, 95, 80)),
        (110, 200, 65, 85, (160, 180, 195)),
        (230, 210, 90, 65, (215, 190, 150)),
        (80, 300, 80, 50, (170, 110, 90)),
        (240, 300, 60, 60, (180, 185, 175)),
        # Sector 2: Top-Right
        (480, 80, 85, 75, (210, 130, 90)),
        (620, 90, 90, 65, (155, 170, 185)),
        (490, 220, 70, 90, (195, 185, 170)),
        (630, 210, 80, 75, (180, 100, 85)),
        # Sector 3: Bottom-Left
        (100, 470, 75, 75, (175, 180, 190)),
        (230, 480, 85, 60, (205, 120, 95)),
        (110, 610, 90, 70, (190, 170, 150)),
        (240, 600, 70, 80, (160, 175, 185)),
        # Sector 4: Bottom-Right
        (490, 480, 80, 70, (200, 115, 85)),
        (620, 490, 70, 85, (185, 190, 175)),
        (500, 620, 90, 75, (165, 180, 195)),
        (630, 610, 85, 70, (210, 140, 100)),
    ]

    for x, y, w, h, color in building_specs:
        # Shadow
        draw_pre.rectangle([(x + 5, y + 5), (x + w + 5, y + h + 5)], fill=(40, 50, 40))
        # Building roof
        draw_pre.rectangle([(x, y), (x + w, y + h)], fill=color, outline=(30, 30, 30), width=2)
        # Rooftop ventilation unit or detail
        draw_pre.rectangle([(x + w // 3, y + h // 3), (x + 2 * w // 3, y + 2 * h // 3)], fill=(230, 230, 230))

    pre_img.save(pre_path)
    logger.info(f"Synthetic pre-disaster image saved to: {pre_path}")

    # Now create post-disaster image with visible structural impacts
    post_array = np.array(pre_img).copy()

    # Apply disaster effects to specific buildings:
    # Building 1: Destroyed (Sector 1: 100, 80) -> rubble / crater / dark debris
    post_array[75:150, 95:180] = [80, 75, 70]
    # Add random rubble debris
    np.random.seed(101)
    debris_noise = np.random.randint(-40, 40, (75, 85, 3), dtype=np.int16)
    post_array[75:150, 95:180] = np.clip(post_array[75:150, 95:180].astype(np.int16) + debris_noise, 0, 255)

    # Building 8: Destroyed (Sector 2: 620, 90) -> collapsed / scorch
    post_array[85:160, 615:715] = [70, 65, 60]
    debris_noise2 = np.random.randint(-35, 35, (75, 100, 3), dtype=np.int16)
    post_array[85:160, 615:715] = np.clip(post_array[85:160, 615:715].astype(np.int16) + debris_noise2, 0, 255)

    # Building 11: Major damage (Sector 3: 230, 480) -> partial roof collapse / tear
    post_array[480:525, 230:315] = [95, 85, 80]

    # Building 15: Major damage (Sector 4: 490, 480) -> fracture and debris
    post_array[495:550, 520:570] = [105, 95, 85]

    # Minor damage on building 4 and 17 (tarp/holes)
    post_array[215:240, 240:270] = [50, 100, 180]  # Blue emergency tarp
    post_array[630:650, 520:550] = [90, 80, 75]

    # Add localized flood / scorch stain near crossroad
    post_img = Image.fromarray(post_array, "RGB")
    draw_post = ImageDraw.Draw(post_img)
    draw_post.ellipse([(360, 360), (440, 440)], fill=(70, 60, 50, 180))

    post_img.save(post_path)
    logger.info(f"Synthetic post-disaster image saved to: {post_path}")

    return pre_path, post_path


def create_tactical_detection_overlay(
    post_image_path: str | Path,
    feature_collection: dict[str, Any],
    priority_zones: list[dict[str, Any]],
    output_path: str | Path | None = None,
) -> Tuple[np.ndarray, Path]:
    """Render a tactical detection map overlay directly onto the post-disaster image pixels.

    Draws:
    1. Every detected building footprint in EMS-98 colors (Red/Orange/Yellow/Green) with ID tags.
    2. Spatial priority zone grid cells with zone IDs and ranks.
    3. Prominent '🚨 RESCUE FIRST' red tactical targeting highlight over the Rank #1 priority zone.
    4. Tactical military HUD framing, corner brackets, and mission status banners.
    """
    logger.info("Generating tactical optical detection map overlay...")
    post_img = load_image(post_image_path)
    h, w = post_img.shape[:2]
    canvas = cv2.cvtColor(post_img, cv2.COLOR_RGB2BGR)

    # EMS-98 BGR colors
    ems_colors = {
        "destroyed": (40, 40, 220),       # Red
        "major-damage": (20, 126, 253),   # Orange
        "minor-damage": (7, 193, 255),    # Yellow
        "no-damage": (69, 167, 40),       # Green
    }

    # Extract all feature points to determine georeferencing scale as fallback
    features = feature_collection.get("features", [])
    all_lons: list[float] = []
    all_lats: list[float] = []
    for feat in features:
        for ring in feat.get("geometry", {}).get("coordinates", []):
            for pt in ring:
                all_lons.append(float(pt[0]))
                all_lats.append(float(pt[1]))

    if all_lons and all_lats:
        min_lon, max_lon = min(all_lons), max(all_lons)
        min_lat, max_lat = min(all_lats), max(all_lats)
        # Add slight margin to avoid zero-division
        d_lon = max_lon - min_lon if max_lon != min_lon else 0.001
        d_lat = max_lat - min_lat if max_lat != min_lat else 0.001

        def to_px(lon: float, lat: float) -> tuple[int, int]:
            px = int(((lon - min_lon) / d_lon) * (w - 40) + 20)
            py = int(((max_lat - lat) / d_lat) * (h - 40) + 20)
            return max(0, min(w - 1, px)), max(0, min(h - 1, py))
    else:
        def to_px(lon: float, lat: float) -> tuple[int, int]:
            return w // 2, h // 2

    # 1. Draw Priority Zone Grid Overlays
    if priority_zones:
        top_zone = priority_zones[0]
        top_zone_id = top_zone.get("zone_id", "Z-001")

        # Determine grid dimensions from count
        grid_count = len(priority_zones)
        grid_rows = int(grid_count**0.5) if int(grid_count**0.5)**2 == grid_count else 3
        grid_cols = grid_count // grid_rows if grid_rows > 0 else 3

        cell_w = w // max(1, grid_cols)
        cell_h = h // max(1, grid_rows)

        # Draw grid lines
        grid_overlay = canvas.copy()
        for r in range(grid_rows + 1):
            y = min(h - 1, r * cell_h)
            cv2.line(grid_overlay, (0, y), (w, y), (180, 220, 240), 1)
        for c in range(grid_cols + 1):
            x = min(w - 1, c * cell_w)
            cv2.line(grid_overlay, (x, 0), (x, h), (180, 220, 240), 1)
        cv2.addWeighted(grid_overlay, 0.35, canvas, 0.65, 0, canvas)

        # Highlight Rank #1 Zone (RESCUE FIRST)
        for idx, z in enumerate(priority_zones):
            rank = z.get("priority_rank", idx + 1)
            zid = z.get("zone_id", f"Z-{idx+1:03d}")
            r_idx = idx // grid_cols
            c_idx = idx % grid_cols
            zx0 = c_idx * cell_w
            zy0 = r_idx * cell_h
            zx1 = min(w, (c_idx + 1) * cell_w)
            zy1 = min(h, (r_idx + 1) * cell_h)

            if rank == 1 or zid == top_zone_id:
                # Strong red semi-transparent overlay for first rescue area
                rescue_overlay = canvas.copy()
                cv2.rectangle(rescue_overlay, (zx0, zy0), (zx1, zy1), (30, 30, 220), -1)
                cv2.addWeighted(rescue_overlay, 0.22, canvas, 0.78, 0, canvas)

                # Bright tactical red boundary + target corners
                cv2.rectangle(canvas, (zx0, zy0), (zx1, zy1), (40, 40, 240), 3)
                t_len = min(30, cell_w // 4)
                for (cx, cy, cdx, cdy) in [
                    (zx0, zy0, 1, 1), (zx1, zy0, -1, 1),
                    (zx0, zy1, 1, -1), (zx1, zy1, -1, -1)
                ]:
                    cv2.line(canvas, (cx, cy), (cx + cdx * t_len, cy), (0, 255, 255), 4)
                    cv2.line(canvas, (cx, cy), (cx, cy + cdy * t_len), (0, 255, 255), 4)

                # Center target reticle
                rcx, rcy = (zx0 + zx1) // 2, (zy0 + zy1) // 2
                cv2.circle(canvas, (rcx, rcy), 18, (0, 255, 255), 2)
                cv2.drawMarker(canvas, (rcx, rcy), (0, 255, 255), cv2.MARKER_CROSS, 28, 2)

                # Label tag
                tag = f"RESCUE FIRST: {zid} (Rank #1 Focus)"
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 2)
                cv2.rectangle(canvas, (zx0 + 6, zy0 + 6), (zx0 + tw + 18, zy0 + th + 16), (40, 40, 240), -1)
                cv2.putText(canvas, tag, (zx0 + 12, zy0 + th + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                # Normal zone tag
                z_tag = f"{zid} (Rank #{rank})"
                cv2.putText(canvas, z_tag, (zx0 + 8, zy0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 230, 255), 1, cv2.LINE_AA)

    # 2. Draw Detected Building Footprints & EMS-98 Labels (Semi-transparent Fill + Crisp Borders)
    fill_layer = canvas.copy()
    bld_polys_to_stroke: list[tuple[np.ndarray, tuple[int, int, int], str, str, float, tuple[int, int, int, int]]] = []

    for feat in features:
        props = feat.get("properties", {})
        dmg = props.get("damage_class", props.get("damage", "no-damage"))
        conf = float(props.get("confidence", 0.5))
        bld_id = props.get("feature_id", "BLD")
        color = ems_colors.get(dmg, (200, 200, 200))

        # Check for direct native pixel polygon first
        if "pixel_polygon" in props and len(props["pixel_polygon"]) >= 3:
            pts = np.array(props["pixel_polygon"], dtype=np.int32)
        else:
            ring = feat.get("geometry", {}).get("coordinates", [[]])[0]
            if len(ring) >= 3:
                pts = np.array([to_px(pt[0], pt[1]) for pt in ring], dtype=np.int32)
            else:
                continue

        # Fill building polygon with semi-transparent tint
        cv2.fillPoly(fill_layer, [pts], color)
        bx0, by0, bw, bh = cv2.boundingRect(pts)
        bld_polys_to_stroke.append((pts, color, bld_id, dmg, conf, (bx0, by0, bw, bh)))

    # Blend polygon fills into canvas
    cv2.addWeighted(fill_layer, 0.28, canvas, 0.72, 0, canvas)

    # Stroke building outlines and draw tactical tags
    for pts, color, bld_id, dmg, conf, (bx0, by0, bw, bh) in bld_polys_to_stroke:
        cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=2)

        # Corner accents around bounding box
        al = min(8, max(3, bw // 3))
        cv2.line(canvas, (bx0, by0), (bx0 + al, by0), color, 2)
        cv2.line(canvas, (bx0, by0), (bx0, by0 + al), color, 2)
        cv2.line(canvas, (bx0 + bw, by0), (bx0 + bw - al, by0), color, 2)
        cv2.line(canvas, (bx0 + bw, by0), (bx0 + bw, by0 + al), color, 2)

        # Render tactical label pill on damaged structures
        if dmg in ["destroyed", "major-damage", "minor-damage"]:
            short_lbl = f"{dmg[:4].upper()} {int(conf*100)}%"
            (lw, lh), _ = cv2.getTextSize(short_lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.34, 1)
            lx = max(2, bx0)
            ly = max(lh + 4, by0 - 2)
            cv2.rectangle(canvas, (lx - 1, ly - lh - 3), (lx + lw + 3, ly + 2), (15, 23, 42), -1)
            cv2.rectangle(canvas, (lx - 1, ly - lh - 3), (lx + lw + 3, ly + 2), color, 1)
            cv2.putText(canvas, short_lbl, (lx + 1, ly - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 255, 255), 1, cv2.LINE_AA)

    # 3. Tactical Military Corner Brackets
    L = 35
    tac_color = (0, 240, 210)  # Cyan
    for (px, py, dx, dy) in [(15, 15, 1, 1), (w - 15, 15, -1, 1), (15, h - 15, 1, -1), (w - 15, h - 15, -1, -1)]:
        cv2.line(canvas, (px, py), (px + dx * L, py), tac_color, 2)
        cv2.line(canvas, (px, py), (px, py + dy * L), tac_color, 2)

    # 4. Top/Bottom HUD Banners
    hud_bg = canvas.copy()
    cv2.rectangle(hud_bg, (0, 0), (w, 38), (15, 23, 42), -1)
    cv2.rectangle(hud_bg, (0, h - 34), (w, h), (15, 23, 42), -1)
    cv2.addWeighted(hud_bg, 0.75, canvas, 0.25, 0, canvas)

    cv2.putText(
        canvas,
        f"MERIDIAN TACTICAL OPTICAL MAPPING · {len(features)} STRUCTURES DETECTED · NATIVE RESOLUTION",
        (14, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 240, 210), 1, cv2.LINE_AA
    )
    cv2.putText(
        canvas,
        "RED: DESTROYED  |  ORANGE: MAJOR  |  YELLOW: MINOR  |  GREEN: UNDAMAGED  |  TARGET: FIRST RESCUE SECTOR",
        (14, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA
    )

    out_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    out_target = Path(output_path) if output_path else DEFAULT_OVERLAY_OUTPUT.parent / "tactical_detection_map.png"
    out_target.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_target), canvas)
    logger.info(f"Tactical detection map overlay saved to: {out_target}")

    return out_rgb, out_target
