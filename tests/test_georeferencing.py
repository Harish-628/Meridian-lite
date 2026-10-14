"""Unit tests for Stage 2 georeferencing and coordinate transformation."""

import pytest
from utils.georeferencing import AffineGeoreferencer


def test_georeferencer_init_valid():
    """Verify initialization with valid dimensions and bounds."""
    bounds = (-85.555, 30.170, -85.545, 30.180)
    geo = AffineGeoreferencer(1024, 1024, bounds)
    assert geo.image_width == 1024
    assert geo.image_height == 1024
    assert geo.min_lon == -85.555
    assert geo.max_lat == 30.180


def test_georeferencer_invalid_dimensions():
    """Verify error raised on invalid image dimensions."""
    bounds = (-85.555, 30.170, -85.545, 30.180)
    with pytest.raises(ValueError, match="Image dimensions must be positive"):
        AffineGeoreferencer(0, 1024, bounds)
    with pytest.raises(ValueError, match="Image dimensions must be positive"):
        AffineGeoreferencer(1024, -10, bounds)


def test_georeferencer_invalid_bounds():
    """Verify validation for coordinate bounds."""
    with pytest.raises(ValueError, match="min_lon .* must be less than max_lon"):
        AffineGeoreferencer(100, 100, (10.0, 20.0, 5.0, 30.0))
    with pytest.raises(ValueError, match="min_lat .* must be less than max_lat"):
        AffineGeoreferencer(100, 100, (10.0, 30.0, 20.0, 20.0))
    with pytest.raises(ValueError, match="Longitudes must be in"):
        AffineGeoreferencer(100, 100, (-190.0, 20.0, 20.0, 30.0))


def test_pixel_to_lonlat_corners():
    """Verify pixel corner coordinate mappings match bounding box corners."""
    bounds = (-85.555, 30.170, -85.545, 30.180)
    geo = AffineGeoreferencer(1000, 1000, bounds)

    # Top-left (0, 0) -> (min_lon, max_lat)
    lon_tl, lat_tl = geo.pixel_to_lonlat(0, 0)
    assert pytest.approx(lon_tl, rel=1e-5) == -85.555
    assert pytest.approx(lat_tl, rel=1e-5) == 30.180

    # Bottom-right (1000, 1000) -> (max_lon, min_lat)
    lon_br, lat_br = geo.pixel_to_lonlat(1000, 1000)
    assert pytest.approx(lon_br, rel=1e-5) == -85.545
    assert pytest.approx(lat_br, rel=1e-5) == 30.170

    # Center (500, 500) -> (-85.550, 30.175)
    lon_c, lat_c = geo.pixel_to_lonlat(500, 500)
    assert pytest.approx(lon_c, rel=1e-5) == -85.550
    assert pytest.approx(lat_c, rel=1e-5) == 30.175


def test_pixel_polygon_to_lonlat():
    """Verify polygon points are converted and closed into a valid ring."""
    bounds = (-85.555, 30.170, -85.545, 30.180)
    geo = AffineGeoreferencer(1000, 1000, bounds)

    pts = [(100, 100), (200, 100), (200, 200), (100, 200)]
    ring = geo.pixel_polygon_to_lonlat(pts)
    assert len(ring) == 5
    assert ring[0] == ring[-1]  # Closed ring check


def test_to_geojson_polygon():
    """Verify creation of GeoJSON feature with CRS and metadata."""
    bounds = (-85.555, 30.170, -85.545, 30.180)
    geo = AffineGeoreferencer(1000, 1000, bounds, coordinate_source="test_source")

    pts = [(100, 100), (200, 100), (200, 200)]
    feat = geo.to_geojson_polygon(pts, {"feature_id": "TEST-1", "damage_class": "destroyed"})

    assert feat["type"] == "Feature"
    assert feat["geometry"]["type"] == "Polygon"
    assert feat["properties"]["feature_id"] == "TEST-1"
    assert feat["properties"]["coordinate_source"] == "test_source"
    assert feat["properties"]["crs"] == "EPSG:4326"


def test_from_center_and_radius():
    """Verify factory creation from center and radius."""
    geo = AffineGeoreferencer.from_center_and_radius(
        image_width=512,
        image_height=512,
        center_lat=30.0,
        center_lon=-85.0,
        delta_deg=0.01,
    )
    assert geo.min_lon == pytest.approx(-85.01)
    assert geo.max_lon == pytest.approx(-84.99)
    assert geo.coordinate_source == "user_bbox_approximation"
