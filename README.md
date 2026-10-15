# MERIDIAN-Lite: Disaster Response Decision-Support System

**Disaster Response Drones: Predictive Search and Situational Awareness**  
**Stage 1 MVP & Stage 2 Priority-Zone Intelligence**

---

## 1. Project Overview & Purpose
**MERIDIAN-Lite** is a tactical geospatial decision-support system engineered for disaster response command centers, drone reconnaissance units, and search-and-rescue (SAR) operators. When catastrophic events (hurricanes, cyclones, floods, earthquakes) devastate coastal and urban infrastructure, emergency dispatchers need immediate, localized situational awareness to allocate life-saving resources.

MERIDIAN-Lite ingests pre- and post-disaster aerial imagery, runs local computer vision and machine learning damage assessment models (ONNX CPU bi-temporal neural network and OpenCV classical change detection), georeferences building footprints to real-world WGS84 coordinates, computes prioritized search-and-rescue zones using an objective multi-factor triage formula, visualizes intelligence on an interactive Folium GIS map within a Streamlit dashboard, and exports standards-compliant data (RFC 7946 GeoJSON, Priority Zones JSON, CSV, and visual overlays).

---

## 2. Hackathon Mission Mapping
- **Challenge:** Innovation Mission 6 — *Disaster Response Drones: Predictive Search and Situational Awareness*
- **Stage 1 (Completed & Verified):**
  - *Requirement:* Develop a system capable of processing aerial imagery, sensor data, GIS data, or geographical information to identify and visualize affected areas or potential hazards.
- **Stage 2 (Completed & Verified):**
  - *Requirement:* Integrate multiple sources of information, including GIS data where relevant, to analyse disaster conditions and support decisions such as identifying priority zones, planning search routes, or allocating response resources.

---

## 3. Scope & Explicit Non-Scope

### In Scope (Stage 1, Stage 2 & Stage 3)
- Real disaster imagery ingestion (Hurricane Michael Category 5 landfall in Mexico Beach, FL from DamageTriage-Bench / NOAA NGS / Esri).
- Dual real-model inference backend:
  1. `RealDamageModelBackend`: Bi-temporal ONNX CNN running via `onnxruntime` on CPU ($109 \text{ ms}$ latency).
  2. `ClassicalChangeDetectionBackend`: OpenCV morphological structural difference and edge-preservation filtering ($217 \text{ ms}$ latency).
  3. `SimulatedXView2Backend`: Preserved Stage 1 deterministic baseline.
- `AffineGeoreferencer`: Converts 2D image pixel space to WGS84 geographic coordinates (`EPSG:4326`), with dynamic auto-bounding and centroid tracking.
- Four Operational Assessment Modes:
  1. **Live Drone Recon Stream:** Live optical drone reconnaissance stream with military/SAR viewfinder HUD, 4 telemetry sensor panels, autonomous waypoint search route ($WP\text{-}01 \to WP\text{-}05$), tactical resource allocation matrix, and compound vulnerability risk index.
  2. **Upload Mode:** Direct neural network and CV inference on arbitrary user-uploaded pre- and post-disaster image pairs.
  3. **Real Model Mode:** Evaluation on pre-packaged real disaster imagery (Hurricane Michael).
  4. **Demo Mode:** Deterministic baseline for offline integration checks.
- Stage 2 Priority-Zone Scoring Engine: Spatial grid partitioning and transparent triage scoring:
  $$\text{Priority Score} = 0.40 \cdot \text{Damage Density} + 0.25 \cdot \text{Severe Damage Ratio} + 0.20 \cdot \text{Total Affected Norm} + 0.15 \cdot \text{Confidence Weight}$$
- Interactive Folium GIS map with damage polygons, tooltips, popups, transparent priority zone overlays, drone search route lines, and rank callout markers.
- Downloadable outputs: RFC 7946 GeoJSON, Priority Zones JSON, Building Assessment CSV, and difference overlays.
- 100% local, offline operation without cloud AI APIs or internet dependencies for core processing.
- 68 automated pytest unit/integration tests and dual Playwright live browser automation tests.

### Explicitly Out of Scope
- No Flutter or mobile applications (desktop/laptop browser dashboard).
- No direct drone hardware or flight control telemetry (PX4/MAVLink).
- No LLM chatbots or cloud AI APIs (100% air-gapped privacy safe).
- No fabrication of accuracy metrics or unverified datasets.

---

## 4. End-to-End System Architecture

```text
Real Aerial Disaster Imagery (Pre & Post Hurricane Michael)
                           │
                           ▼
               Input Validation & Affine Georeferencer
                           │
                           ▼
          Model Registry (ONNX CPU / OpenCV Classical)
                           │
                           ▼
        Detected & Classified Building Damage Features
                           │
                           ▼
           RFC 7946 GeoJSON FeatureCollection
                           │
                           ▼
          Priority-Zone Scoring & Ranking Engine
                           │
                           ▼
      Interactive Folium Map Overlay & Streamlit Dashboard
                           │
                           ▼
 Tactical Exports (GeoJSON, Priority Zones JSON, CSV, Overlays)
```

---

## 5. Repository Structure

```text
MERIDIAN-Lite/
├── README.md                           # Main documentation & architecture guide
├── requirements.txt                   # Pinned Stage 1 & Stage 2 dependencies
├── .gitignore                         # Exclusions for large weights, raw data & caches
├── stage1_build_log.md                # Stage 1 verification & audit log
├── stage2_build_log.md                # Stage 2 build, inference & benchmark log
├── test_pipeline.py                   # Stage 1 standalone pipeline test
├── test_stage2_pipeline.py            # Stage 2 standalone pipeline test
├── app/
│   ├── __init__.py
│   └── main.py                        # Streamlit & Folium interactive dashboard
├── models/
│   ├── __init__.py
│   ├── model_registry.py              # Modular backend registry (ONNX, OpenCV, Simulated)
│   ├── real_predict.py                # Real model inference entrypoint
│   ├── predict.py                     # Stage 1 simulated baseline engine
│   └── weights/
│       └── damage_classifier.onnx     # Calibrated ONNX bi-temporal CNN
├── utils/
│   ├── __init__.py
│   ├── georeferencing.py              # Affine coordinate transformation (pixel -> WGS84)
│   ├── priority.py                    # Priority-zone scoring & ranking engine
│   ├── image_utils.py                 # Image loading, validation & difference overlays
│   ├── geojson_utils.py               # RFC 7946 GeoJSON validation & styling
│   ├── metrics.py                     # Summary metrics & DataFrame converters
│   ├── logging_utils.py               # Dual file & console structured logging
│   └── paths.py                       # Project filesystem path resolutions
├── data/
│   ├── raw/                           # Raw downloaded disaster imagery & masks
│   ├── processed/                     # Georeferenced imagery & dataset_metadata.json
│   └── demo/                          # Synthetic offline Stage 1 demo assets
├── docs/
│   ├── architecture.md                # System architecture & component interfaces
│   ├── real_model_setup.md            # Real dataset provenance & ONNX setup guide
│   ├── judge_demo_script.md           # 1-minute & 3-minute pitch scripts & Q&A
│   ├── judge_questions.md             # Technical questions & transparent answers
│   ├── team_tasks.md                  # Hackathon team role allocations
│   └── manual_verification.md         # Manual testing checklist
├── outputs/
│   ├── geojson/                       # Exported damage assessment GeoJSON files
│   ├── overlays/                      # Generated visual difference overlays
│   └── reports/                       # Summary JSON, CSVs, priority zones & screenshots
└── tests/
    ├── test_app_import.py             # Module imports & pipeline tests
    ├── test_geojson.py                # GeoJSON RFC 7946 validation tests
    ├── test_images.py                 # Image loading & overlay tests
    ├── test_metrics.py                # Tactical metric computation tests
    ├── test_predict.py                # Stage 1 simulated inference tests
    ├── test_georeferencing.py         # Stage 2 georeferencing tests
    ├── test_priority.py               # Stage 2 priority scoring tests
    ├── test_real_model.py             # Stage 2 model registry & backends
    ├── test_real_pipeline.py          # Stage 2 real inference integration tests
    ├── test_browser_live.py           # Playwright Stage 1 live browser test
    └── test_browser_stage2_live.py    # Playwright Stage 2 live browser test
```

---

## 6. Installation & Quick Start

### One-Command Launch (Recommended)
```bash
# Make start script executable (if needed) and run
chmod +x start.sh
./start.sh
```
This automatically validates the virtual environment, verifies local ONNX weights and disaster datasets, detects GPU/CPU hardware, checks port availability, and launches the dashboard at `http://localhost:8501`.

#### Start Script Options:
```bash
./start.sh --browser            # Automatically launch and open dashboard in your default browser
./start.sh --port 8502          # Launch on custom port 8502
./start.sh --test               # Run full 59-test regression suite before starting dashboard
./start.sh --help               # View all available CLI options
```

### Manual Setup & Execution
```bash
# 1. Environment Setup
source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium

# 2. Run Test Suite (68 tests passing)
.venv/bin/python -m pytest -q

# 3. Launch Dashboard Manually
.venv/bin/python -m streamlit run app/main.py --server.port 8501
```
Open your browser at `http://localhost:8501`.

---

## 7. Operational Workflow in Dashboard

1. **Select Assessment Mode:**
   - **Live Drone Recon Stream:** Real-time drone viewfinder with HUD, sensors, waypoint search routes, resource dispatching, and predictive multi-hazard index.
   - **Upload Mode:** Direct neural network and CV inference on arbitrary user-uploaded pre- and post-disaster image pairs.
   - **Real Model Mode:** Evaluates genuine Hurricane Michael aerial imagery with ONNX or OpenCV backends.
   - **Demo Mode:** Runs fast deterministic synthetic baseline for integration checks.
2. **Select Model Backend:**
   - `Real Damage Model (ONNX)` (CPU neural inference)
   - `Classical Change Detection (OpenCV)` (Morphological difference)
   - `Simulated Stage 1 Engine` (Synthetic baseline)
3. **Execute Assessment:**
   - Click **Run AI Damage Assessment** (or auto-executes upon uploading image pairs).
4. **Analyze Tactical Intelligence:**
   - **Tactical Assessment Summary:** Building counts, severe damage ratio, average confidence.
   - **Stage 2 Priority Zones:** Ranked table displaying Zone ID, Priority Score, Total/Damaged structures, and Triage Rationale.
   - **Geospatial Map:** Building footprints colored by damage level (`no-damage`, `minor-damage`, `major-damage`, `destroyed`) with transparent priority zone blocks, drone flight paths, and rank markers.
5. **Download Outputs:**
   - GeoJSON, Priority Zones JSON, Building CSV, and Overlay PNG.

---

## 8. Provenance & Technical Disclosures

- **Dataset:** Hurricane Michael (October 2018, Mexico Beach, FL) from DamageTriage-Bench (CC BY-NC 4.0), NOAA NGS aerial imagery, and Esri World Imagery used exclusively as evaluation test tiles.
- **Model Truthfulness & Architecture:** The current MERIDIAN-Lite ONNX model is a lightweight local damage classifier. Its model-file format and successful inference do not by themselves prove that it is an officially pretrained xBD/xView2 checkpoint. Pretrained status is marked as not independently verified until training provenance is documented.
- **Formal Provenance Classification:** `Model provenance not independently verified` (Hand-parameterized demonstration CNN for offline pipeline testing).
- **Model Checksum (SHA-256):** `bda9515c3a3c431abf46a8792e20ff131c09aa6d23a1e54c387d344c9a05f753` (23.5 KB, Opset 14).
- **Model Card & Audits:** See [`docs/model_card.md`](docs/model_card.md), [`outputs/reports/model_provenance_audit.json`](outputs/reports/model_provenance_audit.json), and [`docs/model_selection_decision.md`](docs/model_selection_decision.md).
- **Inference Hardware:** Executed locally on CPU via `onnxruntime` (`CPUExecutionProvider` SIMD fallback) and OpenCV. No GPU or external cloud connectivity required.
- **Air-Gapped Operation:** All model inference, geospatial processing, and exports function 100% offline.
