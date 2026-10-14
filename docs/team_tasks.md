# MERIDIAN Hackathon Team Task Distribution (Stage 2)

This document defines the roles, completed deliverables, and next milestones for each of our three team members across Stage 1 and Stage 2.

---

## Member 1: ML Engineer & Inference Systems Lead

### Core Role
Responsible for the damage assessment inference backends, computer vision pipelines, ONNX model weights, and performance optimization on CPU.

### Stage 1 & Stage 2 Completed Deliverables
- [x] Sourced real disaster imagery and ground-truth annotations from DamageTriage-Bench (Hurricane Michael, Mexico Beach, FL).
- [x] Built `models/model_registry.py` with modular backend abstractions (`ModelBackend`, `check_model_health`).
- [x] Engineered and calibrated bi-temporal convolutional neural network in ONNX format (`models/weights/damage_classifier.onnx`, opset 14, IR v10) for CPU execution.
- [x] Implemented classical OpenCV change detection backend (`ClassicalChangeDetectionBackend`) using morphological Otsu filtering and Canny structural edge analysis.
- [x] Implemented `models/real_predict.py` with standardized `run_real_inference` returning RFC 7946 GeoJSON.
- [x] Authored automated test suites in `tests/test_real_model.py` and `tests/test_real_pipeline.py`.

### Next Milestones (Stage 3 Roadmap)
1. **Model Quantization:** Quantize ONNX model to INT8 via ONNX Runtime quantization tool to reduce latency to < 50 ms.
2. **Edge Hardware Target:** Benchmark model on NVIDIA Jetson Orin Nano with TensorRT backend.

---

## Member 2: Geospatial Engineer & GIS Intelligence Lead

### Core Role
Responsible for spatial data standards, RFC 7946 GeoJSON compliance, Folium GIS map configuration, affine georeferencing, and the priority-zone triage scoring engine.

### Stage 1 & Stage 2 Completed Deliverables
- [x] Implemented `utils/georeferencing.py` (`AffineGeoreferencer`) mapping 2D pixel coordinates to real WGS84 geographic degrees (`EPSG:4326`).
- [x] Implemented `utils/priority.py` dividing affected disaster footprints into spatial grid sectors and calculating priority scores:
  $$\text{Priority Score} = 0.40 \cdot \text{Damage Density} + 0.25 \cdot \text{Severe Damage Ratio} + 0.20 \cdot \text{Total Affected Norm} + 0.15 \cdot \text{Confidence Weight}$$
- [x] Configured transparent priority zone overlays with rank callout markers (`Rank #1`, `Rank #2`) and detailed triage rationale.
- [x] Enforced strict RFC 7946 validation in `utils/geojson_utils.py`.
- [x] Authored automated test suites in `tests/test_georeferencing.py` and `tests/test_priority.py`.

### Next Milestones (Stage 3 Roadmap)
1. **Tactical Road Ingestion:** Ingest OpenStreetMap road vectors and perform spatial intersection analysis with rubble debris footprints to identify blocked access routes.
2. **Search-and-Rescue Route Planning:** Implement Dijkstra/A* routing for drone search trajectories avoiding hazardous sectors.

---

## Member 3: Full-Stack UI, QA & Technical Documentation Lead

### Core Role
Responsible for the Streamlit dashboard, Playwright browser test automation, data serialization, technical documentation, and the judge presentation script.

### Stage 1 & Stage 2 Completed Deliverables
- [x] Upgraded `app/main.py` with Real Model Mode, dataset selector, model backend selector, prominent output distinction banners, and top priority zones table.
- [x] Implemented live downloads for GeoJSON, Priority Zones JSON, building CSV, and difference overlays.
- [x] Authored dual Playwright browser automation tests (`tests/test_browser_live.py` and `tests/test_browser_stage2_live.py`), verifying full UI functionality in headless Chromium and capturing screenshots.
- [x] Created standalone verification scripts `test_pipeline.py` (Stage 1) and `test_stage2_pipeline.py` (Stage 2).
- [x] Authored comprehensive documentation: `docs/real_model_setup.md`, `docs/judge_demo_script.md`, `docs/judge_questions.md`, `stage1_build_log.md`, and `stage2_build_log.md`.

### Next Milestones (Stage 3 Roadmap)
1. **Interactive Routing View:** Add interactive routing waypoint controls for SAR drone flight planning.
2. **Video Pitch & Rehearsal:** Execute live demonstration following `docs/judge_demo_script.md`.
