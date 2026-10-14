# MERIDIAN-Lite Demo Datasets

This directory contains synthetic aerial imagery and mock building polygon GeoJSON data for offline demonstration and testing of the MERIDIAN-Lite Stage 1 system.

## Notice and Limitations

> [!WARNING]
> **Synthetic Testing Data Only**
> - All files in this directory (`pre_disaster.png`, `post_disaster.png`, `buildings.geojson`) are **completely synthetic and computer-generated**.
> - The coordinates and building boundaries are fictional and clustered around nominal sample coordinates for demonstration purposes.
> - **These files DO NOT represent real disaster observations, actual casualties, or real building damages.**
> - Never use or represent these demonstration files as actual post-disaster reconnaissance or operational ground-truth data.

## File Manifest

- `pre_disaster.png`: Synthetically generated 800x800 RGB image showing pre-disaster building rooftops, roads, and landscape features.
- `post_disaster.png`: Synthetically generated 800x800 RGB image with induced structural damage features (collapsed rooftops, debris patterns, scorch marks).
- `buildings.geojson`: RFC 7946 compliant GeoJSON FeatureCollection containing 18 sample building polygons labeled with simulated damage classifications (`no-damage`, `minor-damage`, `major-damage`, `destroyed`). Marked with `"is_demo": true`.
