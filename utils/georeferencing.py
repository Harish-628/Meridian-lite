"""Georeferencing utility for MERIDIAN-Lite Stage 2.

Handles coordinate transformation between 2D image pixel space (x, y)
and geographic coordinates (longitude, latitude in EPSG:4326 WGS84).
"""

from typing import Any


class AffineGeoreferencer:
    """Transforms pixel coordinates to geographic coordinates using an affine bounding box."""

    def __init__(
        self,
        image_width: int,
        image_height: int,
        bounds: tuple[float, float, float, float],
        coordinate_source: str = "georeferenced_bounds",
        crs: str = "EPSG:4326",
    ) -> None:
        """Initialize georeferencer.

        Args:
            image_width: Width of image in pixels (must be > 0).
            image_height: Height of image in pixels (must be > 0).
            bounds: Tuple of (min_lon, min_lat, max_lon, max_lat).
            coordinate_source: Description of coordinate source (e.g. 'georeferenced_bounds', 'user_bbox_approximation').
            crs: Coordinate Reference System identifier (default: EPSG:4326).
        """
        if image_width <= 0 or image_height <= 0:
            raise ValueError(f"Image dimensions must be positive, got {image_width}x{image_height}")

        min_lon, min_lat, max_lon, max_lat = bounds
        if not (-180.0 <= min_lon <= 180.0 and -180.0 <= max_lon <= 180.0):
            raise ValueError(f"Longitudes must be in [-180, 180], got ({min_lon}, {max_lon})")
        if not (-90.0 <= min_lat <= 90.0 and -90.0 <= max_lat <= 90.0):
            raise ValueError(f"Latitudes must be in [-90, 90], got ({min_lat}, {max_lat})")
        if min_lon >= max_lon:
            raise ValueError(f"min_lon ({min_lon}) must be less than max_lon ({max_lon})")
        if min_lat >= max_lat:
            raise ValueError(f"min_lat ({min_lat}) must be less than max_lat ({max_lat})")

        self.image_width = image_width
        self.image_height = image_height
        self.min_lon = min_lon
        self.min_lat = min_lat
        self.max_lon = max_lon
        self.max_lat = max_lat
        self.coordinate_source = coordinate_source
        self.crs = crs

        self.dlon = (max_lon - min_lon) / float(image_width)
        self.dlat = (max_lat - min_lat) / float(image_height)

    def pixel_to_lonlat(self, x: float, y: float) -> tuple[float, float]:
        """Convert pixel (x, y) to geographic (lon, lat).

        Note: Image y=0 corresponds to the top (max_lat), y=image_height to bottom (min_lat).
        """
        clamped_x = max(0.0, min(float(x), float(self.image_width)))
        clamped_y = max(0.0, min(float(y), float(self.image_height)))

        lon = self.min_lon + clamped_x * self.dlon
        lat = self.max_lat - clamped_y * self.dlat
        return round(lon, 7), round(lat, 7)

    def lonlat_to_pixel(self, lon: float, lat: float) -> tuple[float, float]:
        """Convert geographic (lon, lat) to pixel (x, y)."""
        x = (lon - self.min_lon) / self.dlon
        y = (self.max_lat - lat) / self.dlat
        return round(x, 2), round(y, 2)

    def pixel_polygon_to_lonlat(
        self, pixel_points: list[tuple[float, float] | list[float]]
    ) -> list[list[float]]:
        """Convert a list of pixel points to a closed geographic polygon ring [[lon, lat], ...]."""
        if len(pixel_points) < 3:
            raise ValueError(f"Polygon must have at least 3 points, got {len(pixel_points)}")

        ring: list[list[float]] = []
        for pt in pixel_points:
            x, y = pt[0], pt[1]
            lon, lat = self.pixel_to_lonlat(x, y)
            ring.append([lon, lat])

        # Ensure ring is closed
        if ring[0] != ring[-1]:
            ring.append(ring[0])

        return ring

    def to_geojson_polygon(
        self,
        pixel_points: list[tuple[float, float] | list[float]],
        properties: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a GeoJSON Feature from pixel polygon coordinates."""
        ring = self.pixel_polygon_to_lonlat(pixel_points)
        try:
            from shapely.geometry import Polygon
            from shapely.validation import make_valid

            poly = Polygon(ring)
            if not poly.is_valid:
                repaired = make_valid(poly)
                if repaired.geom_type == "MultiPolygon":
                    repaired = max(repaired.geoms, key=lambda g: g.area)
                elif repaired.geom_type == "GeometryCollection":
                    polys = [g for g in repaired.geoms if g.geom_type == "Polygon"]
                    if polys:
                        repaired = max(polys, key=lambda g: g.area)
                    else:
                        repaired = poly.convex_hull
                elif repaired.geom_type != "Polygon":
                    repaired = poly.convex_hull
                ring = [list(c) for c in repaired.exterior.coords]
        except Exception:
            pass

        props = dict(properties or {})
        props.setdefault("pixel_polygon", [[float(pt[0]), float(pt[1])] for pt in pixel_points])
        props.setdefault("coordinate_source", self.coordinate_source)
        props.setdefault("crs", self.crs)

        feat_dict: dict[str, Any] = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [ring],
            },
            "properties": props,
        }
        if "feature_id" in props and props["feature_id"]:
            feat_dict["id"] = props["feature_id"]
        return feat_dict

    @classmethod
    def from_center_and_radius(
        cls,
        image_width: int,
        image_height: int,
        center_lat: float,
        center_lon: float,
        delta_deg: float = 0.005,
        coordinate_source: str = "user_bbox_approximation",
    ) -> "AffineGeoreferencer":
        """Factory method to create a georeferencer around a center coordinate."""
        bounds = (
            center_lon - delta_deg,
            center_lat - delta_deg,
            center_lon + delta_deg,
            center_lat + delta_deg,
        )
        return cls(image_width, image_height, bounds, coordinate_source=coordinate_source)
