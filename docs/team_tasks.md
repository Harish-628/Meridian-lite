# MERIDIAN Hackathon Engineering Team Task Distribution (Mission 6)

This document formally defines the technical roles, completed deliverables, operational responsibilities, and future roadmaps for each of our four registered team members across the MERIDIAN-Lite Innovation Mission 6 lifecycle.

---

## Member 1: Harish Kumar S — Machine Learning & On-Device Neural Inference Lead

### Core Role
Directly responsible for computer vision damage assessment pipelines, ONNX bi-temporal neural architectures, lightweight CPU SIMD execution optimization, and real disaster ground-truth validation.

### Completed Deliverables (Innovation Mission 6)
- [x] **Disaster Benchmark Ingestion:** Sourced real hurricane disaster imagery and ground truth annotations from DamageTriage-Bench (Hurricane Michael, Mexico Beach, FL).
- [x] **Modular Model Registry (`models/model_registry.py`):** Architected runtime-agnostic abstraction (`ModelBackend`, `check_model_health`) supporting ONNX neural, classical morphological OpenCV, and synthetic fallbacks.
- [x] **Bi-Temporal Neural Classifier (`models/weights/damage_classifier.onnx`):** Calibrated and validated lightweight dual-feed CNN (Opset 14, IR v10) achieving 72–151 ms CPU inference latency with zero GPU requirements.
- [x] **Classical Computer Vision Engine (`models/model_registry.py`):** Implemented multi-channel Otsu adaptive thresholding and Canny edge analysis for structural difference detection.
- [x] **Real Neural Pipeline (`models/real_predict.py`):** Authored deterministic inference runner outputting compliant RFC 7946 GeoJSON with bounding boxes and per-feature damage classifications.
- [x] **Verification Suites:** Created and verified unit test suites in `tests/test_real_model.py` and `tests/test_real_pipeline.py`.

### Mission Roadmap
1. **INT8 Quantization:** Apply dynamic INT8 weight quantization via ONNX Runtime to drive CPU inference down to < 50 ms.
2. **Edge Hardware Target:** Benchmark inference container on NVIDIA Jetson Orin Nano with TensorRT backend.

---

## Member 2: Geospatial Intelligence & GIS Architecture Lead

### Core Role
Responsible for geospatial standards, RFC 7946 GeoJSON schema enforcement, affine spatial transforms, WGS84 coordinate projections, and priority-zone triage algorithms.

### Completed Deliverables (Innovation Mission 6)
- [x] **Affine Georeferencing (`utils/georeferencing.py`):** Designed `AffineGeoreferencer` bridging pixel raster coordinates to WGS84 geographic coordinates (`EPSG:4326`) for Mexico Beach coastal bounds (`[-85.550100, 30.152997, -85.547337, 30.155760]`).
- [x] **Spatial Priority-Zone Scoring Engine (`utils/priority.py`):** Architected spatial sector grid triage engine using multi-criteria weighted formulation:
  $$\text{Priority Score} = 0.35 \cdot D_z + 0.25 \cdot S_z + 0.20 \cdot E_z + 0.20 \cdot I_z$$
  incorporating damage density ($D_z$), severe damage ratio ($S_z$), exposed civilian population ($E_z$), and infrastructure isolation ($I_z$).
- [x] **Critical Infrastructure Proximity:** Implemented automated distance metrics to Emergency Aid Stations, Evacuation Shelters, and Water/Power Substations.
- [x] **GIS Schema Compliance (`utils/geojson_utils.py`):** Formally validated all FeatureCollections against RFC 7946 specifications with strict polygon coordinate topology.
- [x] **Verification Suites:** Authored and verified `tests/test_georeferencing.py` and `tests/test_priority.py`.

### Mission Roadmap
1. **Satellite Constellation Integration:** Implement automated STAC (SpatioTemporal Asset Catalog) API ingestion for European Space Agency Sentinel-2 imagery.
2. **LiDAR Elevation Fusion:** Ingest post-disaster 3D point clouds to compute volumetric flood depth contours.

---

## Member 3: Defense Systems Architect & Operations Research Lead

### Core Role
Responsible for tactical Command & Control (C2) operational architecture, emergency safe-routing graph algorithms, dynamic road obstruction impedance, and greedy search-and-rescue resource allocation.

### Completed Deliverables (Innovation Mission 6)
- [x] **Dynamic Road Impedance Routing (`utils/routing.py`):** Constructed NetworkX graph routing topology connecting coastal arterial corridors (Hwy 98) with dynamic edge penalties.
- [x] **Structural Debris Occlusion Modeling:** Applied $15.0\text{ m}$ spatial buffer around collapsed/destroyed buildings, automatically setting obstructed road edges to $50\times$ impedance or marking complete impassability.
- [x] **Greedy Tactical Resource Allocator (`utils/priority.py`):** Implemented deterministic allocation dispatching Heavy Urban Search & Rescue (USAR) Squads, Advanced Life Support Ambulances, and Heavy Engineering Debris Units based on sector triage scores.
- [x] **Standardized Tactical Reports:** Engineered ICS-201 Incident Action Plan generator and SitRep markdown summaries with operational urgency grading.
- [x] **Verification Suites:** Authored automated routing validation in `tests/test_enterprise_c2.py`.

### Mission Roadmap
1. **Multi-UAV Mesh Tasking:** Implement decentralized task-allocation algorithms for multi-drone swarm coverage.
2. **Tactical Data Link (Link 16 / CoT):** Export situational awareness tracks via Cursor-on-Target (CoT) XML for ATAK (Android Tactical Assault Kit) field clients.

---

## Member 4: Full-Stack C2 UI, Mission Assurance & QA Lead

### Core Role
Responsible for the mission-critical Streamlit C2 Tactical Split Terminal, synchronized GIS mapping, live drone feed telemetry, end-to-end headless browser testing, and automated release verification.

### Completed Deliverables (Innovation Mission 6)
- [x] **Enterprise C2 Split Terminal (`app/ui/c2_terminal.py`):** Engineered 35/65 Split Terminal layout featuring Left Tactical Rail with telemetry metrics, resource dispatching, SitRep generation, and direct downloads.
- [x] **Interactive Tactical Map View (`app/ui/map_view.py`):** Fixed layer occlusion bugs by collapsing Folium controls into clean top-right icons, with custom damage layer styling, SAR route overlays, and building markers.
- [x] **Live Drone Recon Stream View (`app/ui/drone_view.py`):** Synchronized aerial video frames with live UAV position marker and dynamic sensor field-of-view (FOV) bounding on the GIS map.
- [x] **Automated Browser & Regression Suites:** Authored and maintained Playwright browser tests (`tests/test_browser_live.py`, `tests/test_browser_stage2_live.py`) and UI unit tests (`tests/test_ui_app.py`).
- [x] **System Verification Scripts:** Created and maintained end-to-end audit pipelines `test_pipeline.py` and `test_stage2_pipeline.py`.

### Mission Roadmap
1. **WebGL / Deck.gl Migration:** Transition Folium Leaflet components to high-framerate WebGL/Deck.gl for rendering 50,000+ vector geometries at 60 FPS.
2. **Offline PWA Packaging:** Package the terminal with local service workers and IndexedDB caching for ruggedized field tablets.
