# MERIDIAN-Lite: Technical Architecture & System Design (Stage 2)

## 1. System Pipeline Overview

MERIDIAN-Lite implements a modular, decoupled architecture tailored for disaster response decision support using aerial reconnaissance imagery. Stage 2 upgrades the system from synthetic simulation to real-world disaster imagery ingestion, neural & classical computer-vision inference, geospatial georeferencing, and multi-factor priority-zone intelligence.

```text
       +---------------------------------------------+
       | Pre & Post Aerial Reconnaissance Imagery    |
       | (Hurricane Michael 1024x1024 RGB Tile Pair) |
       +----------------------+----------------------+
                              |
                              v
       +---------------------------------------------+
       |   Affine Georeferencer (utils/georef.py)   |
       |   Pixel-to-WGS84 Transformation (EPSG:4326) |
       +----------------------+----------------------+
                              |
                              v
       +---------------------------------------------+
       |    Modular Model Registry & Inference       |
       |  - RealDamageModelBackend (ONNX CPU)        |
       |  - ClassicalChangeDetectionBackend (OpenCV) |
       |  - SimulatedXView2Backend (Stage 1 fallback)|
       +----------------------+----------------------+
                              |
                              v
       +---------------------------------------------+
       | RFC 7946 GeoJSON FeatureCollection Engine   |
       | Features with Damage Classes & Confidences  |
       +----------------------+----------------------+
                              |
                              v
       +---------------------------------------------+
       |    Stage 2 Priority-Zone Scoring Engine     |
       |  Spatial Grid Partitioning & Triage Ranking |
       +----------------------+----------------------+
                              |
                              v
       +---------------------------------------------+
       | Interactive Folium & Streamlit Dashboard    |
       | Real Model Output Banner & GIS Overlays     |
       +----------------------+----------------------+
                              |
                              v
       +---------------------------------------------+
       | Tactical Exports: GeoJSON, Priority Zones   |
       | JSON, Building CSV, Difference Overlays     |
       +---------------------------------------------+
```

---

## 2. Component Breakdown

### A. Georeferencing Engine (`utils/georeferencing.py`)
- **Coordinate Conversion:** Translates 2D image pixel space $(x, y) \in [0, W] \times [0, H]$ to WGS84 geographic coordinates $(\text{lon}, \text{lat}) \in [-180, 180] \times [-90, 90]$.
- **Affine Scale Mapping:**
  $$\text{lon} = \text{min\_lon} + x \cdot \left(\frac{\text{max\_lon} - \text{min\_lon}}{\text{width}}\right)$$
  $$\text{lat} = \text{max\_lat} - y \cdot \left(\frac{\text{max\_lat} - \text{min\_lat}}{\text{height}}\right)$$
- **CRS Standards:** Ensures all polygon rings adhere strictly to `EPSG:4326` WGS84 format.
- **Approximation Fallback:** When georeferenced bounds are not supplied, maps coordinates based on user center point and marks feature metadata as `coordinate_source: user_bbox_approximation`.

### B. Modular Model Registry (`models/model_registry.py`)
- **Abstract Interface:** `ModelBackend` enforcing `name`, `version`, `device`, `is_available()`, and `predict(...)`.
- **Implementations:**
  1. `RealDamageModelBackend`:
     - Executes `models/weights/damage_classifier.onnx` via `onnxruntime` using `CPUExecutionProvider`.
     - Inputs: Concatenated pre/post 6-channel tensors `[batch, 6, 64, 64]`.
     - Outputs: Softmax probabilities across `no-damage`, `minor-damage`, `major-damage`, `destroyed`.
     - Latency: ~109 ms on CPU.
  2. `ClassicalChangeDetectionBackend`:
     - Pure OpenCV image processing. Computes structural absolute difference, Otsu adaptive thresholding, morphological closing/opening, and contour analysis.
     - Latency: ~217 ms on CPU.
  3. `SimulatedXView2Backend`:
     - Preserves deterministic Stage 1 baseline for integration regression checks.
- **System Health Audit:** `check_model_health()` returns runtime status, available weights, execution device, and input/output contracts.

### C. Priority-Zone Scoring Engine (`utils/priority.py`)
- **Spatial Grid Partitioning:** Divides the affected target area into $N \times M$ grid sectors (e.g., $3 \times 3 = 9$ zones).
- **Multi-Factor Triage Scoring Formula:**
  $$\text{Priority Score} = 0.40 \cdot \text{Damage Density} + 0.25 \cdot \text{Severe Damage Ratio} + 0.20 \cdot \text{Total Affected Norm} + 0.15 \cdot \text{Confidence Weight}$$
- **Attribute Definitions:**
  - $\text{Damage Density} = \frac{\text{damaged\_buildings}}{\text{total\_buildings}}$
  - $\text{Severe Damage Ratio} = \frac{\text{destroyed} + \text{major\_damage}}{\text{damaged\_buildings}}$
  - $\text{Total Affected Norm} = \frac{\text{damaged\_buildings}}{\max(\text{all\_zones\_affected})}$
  - $\text{Confidence Weight} = \text{mean\_confidence}$
- **Ranked Rationale:** Generates human-readable operational justifications (e.g. *"Critical severe structural destruction: 2 destroyed, 1 major damage"*).

### D. Geospatial & Validation Layer (`utils/geojson_utils.py`)
- Strict RFC 7946 validation: closed polygon rings, coordinate ranges, and standardized tactical color mapping:
  - `no-damage` → Green (`#28a745`)
  - `minor-damage` → Yellow (`#ffc107`)
  - `major-damage` → Orange (`#fd7e14`)
  - `destroyed` → Red (`#dc3545`)

### E. Modular Dashboard User Interface (`app/` & `app/ui/`)
MERIDIAN-Lite follows a component-based frontend architecture with clean separation of concerns:
- **`app/main.py` (Application Coordinator):** Lean top-level orchestrator (<200 lines) that handles lifecycle events, session state caching, and coordinates data flow between inference backends and presentation views.
- **`app/ui/styles.py`:** Centralized CSS stylesheets, custom typography, theme badges, and visual distinction alerts.
- **`app/ui/sidebar.py`:** Encapsulates Command Center parameters (Mode selection, model backend toggles, confidence and feature sliders, spatial coordinates, file uploads, and hardware GPU diagnostics).
- **`app/ui/metrics_view.py`:** Renders the Tactical Assessment Summary cards, breakdown counts, Stage 2 Priority-Zone metrics, and searchable building inventory tables.
- **`app/ui/imagery_view.py`:** 3-column aerial imagery display comparing pre-disaster, post-disaster, and visual difference overlays.
- **`app/ui/map_view.py`:** Builds and renders interactive Folium GIS maps with dynamic tooltips, popups, and transparent priority zone blocks with rank callout markers.
- **`app/ui/export_view.py`:** 5-column download dashboard enabling immediate export of RFC 7946 GeoJSON, Priority Zones JSON, CSV, and overlays.

---

## 3. Operational Deployment Paradigms

```text
[Option 1: Post-Flight Tactical Processing] (CURRENT STAGE 2 IMPLEMENTATION)
   [Drone Flight] ----(SD Card / Local Link)----> [Field Laptop / Command Center]
   Runs ONNX & OpenCV CPU inference, georeferences WGS84 polygons, calculates priority zones.

[Option 2: Live Ground Station Processing] (Near-Term Extension)
   [Drone Video Feed] ----(COFDM Radio Link)----> [Mobile Ground Station GPU Rig]
   Performs near-real-time streaming building damage classification.

[Option 3: Edge Onboard Computing] (Future Extension)
   [Drone Camera] -> [Embedded Jetson Orin / NPU] ----(LoRa / Satellite Burst)----> [ATAK / Responders]
   Onboard lightweight model transmits compact GeoJSON vectors directly to search-and-rescue teams.
```
