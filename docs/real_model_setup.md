# MERIDIAN-Lite: Real Model & Dataset Setup Guide (Stage 2)

This document provides complete technical documentation of the real-world dataset, computer vision models, geospatial conversion, and inference setup powering **MERIDIAN-Lite Stage 2**.

---

## 1. Real Disaster Dataset

### 1.1 Dataset Identification & Provenance
- **Dataset Name:** DamageTriage-Bench (Hurricane Michael Subset)
- **Primary Source:** [DamageTriage-Bench on Hugging Face](https://huggingface.co/datasets/Ymx1025/DamageTriage-Bench)
- **Sensor Data Providers:**
  - **Post-Disaster Imagery:** NOAA National Geodetic Survey (NGS) Remote Sensing Division, Emergency Response Airborne Imagery (October 11, 2018).
  - **Pre-Disaster Baseline Imagery:** Esri High-Resolution World Imagery REST Service (WGS84 export).
  - **Ground-Truth Building Footprints & Damage Annotations:** DamageTriage-Bench manual expert triage annotations.
- **Disaster Event:** Hurricane Michael (Category 5 landfall, October 10, 2018).
- **Target Geographic Sector:** Mexico Beach, Bay County, Florida, USA.
- **Center Coordinates:** $30.1750^\circ \text{N}, -85.5500^\circ \text{W}$ (WGS84).
- **Bounding Box:**
  - Minimum Longitude: $-85.5550^\circ$
  - Minimum Latitude: $30.1700^\circ$
  - Maximum Longitude: $-85.5450^\circ$
  - Maximum Latitude: $30.1800^\circ$
  - CRS: `EPSG:4326 (WGS84)`

### 1.2 License & Citation
- **License:** Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) for DamageTriage-Bench annotations; NOAA public domain imagery.
- **Academic Citation:**
  > *DamageTriage-Bench: A Benchmark for Aerial Disaster Building Damage Assessment* (2024); NOAA National Geodetic Survey Remote Sensing Division Hurricane Michael Emergency Response Imagery (2018).

### 1.3 Download Method & Local Cache
- **Download Automation:** Automatable via HTTPS REST requests through Python standard library `urllib.request`.
- **Downloaded File Inventory:**
  | File Path | Dimensions | Channels | Size | Purpose |
  | :--- | :--- | :--- | :--- | :--- |
  | `data/raw/hurricane_michael_pre.png` | $1024 \times 1024$ | 3 (RGB) | 799 KB | High-resolution pre-event satellite baseline |
  | `data/raw/hurricane_michael_post.png` | $1024 \times 1024$ | 3 (RGB) | 2,199 KB | Post-disaster NOAA aerial reconnaissance |
  | `data/raw/hurricane_michael_ground_truth_mask.png` | $1024 \times 1024$ | 3 (RGB) | 7.8 KB | Ground-truth triage damage masks |
  | `data/processed/dataset_metadata.json` | JSON | — | 1.4 KB | Full provenance & bounding box metadata |

---

## 2. Computer Vision & ML Models

MERIDIAN-Lite incorporates a dual real-model architecture alongside the Stage 1 simulation fallback:

### 2.1 Model Architecture 1: Bi-Temporal ONNX CNN (`RealDamageModelBackend`)
> **Important Architecture Disclosure:** The ONNX backend performs real local neural-network inference, but it is a lightweight locally implemented bi-temporal model. It is not a large pretrained xView2/xBD checkpoint. Therefore, do not describe it as a pretrained xView2 model.

- **Framework:** ONNX Runtime (`onnxruntime` v1.31.0) with `CPUExecutionProvider`.
- **Weights File:** `models/weights/damage_classifier.onnx` (23,568 bytes).
- **OPSET Version:** 14 (IR Version 10).
- **Input Specification:**
  - Tensor Name: `input`
  - Shape: `[batch_size, 6, 64, 64]` (float32)
  - Layout: Channels 0..2 = Pre-disaster RGB (normalized $[0.0, 1.0]$); Channels 3..5 = Post-disaster RGB (normalized $[0.0, 1.0]$).
- **Network Pipeline:**
  1. `Conv1`: 16 filters ($3 \times 3$, padding 1) computing bi-temporal differential gradients ($\text{Post} - \text{Pre}$, $\text{Pre} - \text{Post}$), edge disturbance (Laplacian), and baseline structural presence.
  2. `Relu` activation + `MaxPool` ($2 \times 2$, stride 2).
  3. `Conv2`: 32 filters ($3 \times 3$, padding 1) learning higher-level rubble and structural collapse representations.
  4. `Relu` activation + `MaxPool` ($2 \times 2$, stride 2).
  5. `GlobalAveragePool`: spatial aggregation to 32 feature descriptors.
  6. `Flatten` + `Gemm` (Dense linear projection: $32 \rightarrow 4$).
  7. `Softmax`: Normalizes logits into 4 class probabilities.
- **Output Classes:**
  - Index 0: `no-damage`
  - Index 1: `minor-damage`
  - Index 2: `major-damage`
  - Index 3: `destroyed`
- **Output Specification:**
  - Tensor Name: `probabilities`
  - Shape: `[batch_size, 4]` (float32)

### 2.2 Model Architecture 2: OpenCV Classical Change Detection (`ClassicalChangeDetectionBackend`)
- **Framework:** OpenCV Headless (`opencv-python-headless` v5.0.0).
- **Methodology:**
  1. Dual-channel structural luminance difference: $|\text{Gray}_{\text{post}} - \text{Gray}_{\text{pre}}|$.
  2. 5x5 Median noise reduction filtering.
  3. Otsu global adaptive thresholding to isolate significant structural shifts.
  4. Morphological opening and closing (5x5 rectangular structuring element) to disconnect foliage noise and seal contiguous roof footprints.
  5. External contour extraction with Douglas-Peucker polygon approximation.
  6. Multi-attribute structural triage:
     - Mean spectral deviation within polygon mask.
     - Pre- vs post-event Canny edge ratio (structural integrity loss).
     - Contour compactness: $\frac{4\pi \cdot \text{Area}}{\text{Perimeter}^2}$.

### 2.3 Model Architecture 3: Simulated xView2 Baseline (`SimulatedXView2Backend`)
- Deterministic Stage 1 baseline for integration regression testing and air-gapped synthetic demonstration.

---

## 3. Hardware Requirements & Performance

- **Inference Hardware:** CPU execution (Intel/AMD x86_64, 16 cores utilized).
- **RAM Footprint:** ~180 MB peak memory during full-resolution batch inference.
- **Inference Latency Benchmarks:**
  | Backend | Image Resolution | Detected Features | CPU Execution Time |
  | :--- | :--- | :--- | :--- |
  | **ONNX Bi-temporal CNN** | $1024 \times 1024$ (64x64 patches) | 51 features | **~109 ms** |
  | **Classical OpenCV** | $1024 \times 1024$ | 92 features | **~217 ms** |
  | **Stage 1 Simulated** | $800 \times 800$ synthetic | 18 features | **~112 ms** |

---

## 4. Geospatial Conversion & Georeferencing

Implemented in `utils/georeferencing.py` via `AffineGeoreferencer`:
- Maps pixel space coordinates $(x, y) \in [0, \text{width}] \times [0, \text{height}]$ to WGS84 geographic coordinates $(\lambda, \phi) \in [-180, 180] \times [-90, 90]$:
  $$\lambda = \lambda_{\min} + x \cdot \left(\frac{\lambda_{\max} - \lambda_{\min}}{\text{width}}\right)$$
  $$\phi = \phi_{\max} - y \cdot \left(\frac{\phi_{\max} - \phi_{\min}}{\text{height}}\right)$$
- Output polygons strictly follow **RFC 7946 GeoJSON** standards (first position identical to last position, coordinates ordered as `[longitude, latitude]`).
- Metadata attributes stored in each feature:
  - `coordinate_source`: `"georeferenced_bounds"`
  - `crs`: `"EPSG:4326"`
  - `source_model`: Specific inference backend used

---

## 5. Priority-Zone Intelligence Engine

Implemented in `utils/priority.py` via `compute_priority_zones`:
- Spatial grid partitioning ($2 \times 2$, $3 \times 3$, or $4 \times 4$).
- Computes metrics per zone: total structures, damaged count, destroyed count, major-damage count, damage density, mean confidence.
- Multi-factor tactical ranking formula:
  $$\text{Priority Score} = 0.40 \cdot \text{Damage Density} + 0.25 \cdot \text{Severe Damage Ratio} + 0.20 \cdot \text{Total Affected Norm} + 0.15 \cdot \text{Confidence Weight}$$
- Ranked in descending order from highest triage priority (Rank #1) to lowest.

---

## 6. How to Reproduce & Execute

### Installation
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
```

### Standalone Stage 2 Verification
```bash
.venv/bin/python test_stage2_pipeline.py
```

### Full Automated Test Suite
```bash
.venv/bin/python -m pytest -v
```

### Launch Interactive Dashboard
```bash
.venv/bin/python -m streamlit run app/main.py --server.port 8501
```

---

## 7. Known Technical Limitations
1. **Single Geographic Demonstration Sector:** The packaged offline dataset includes the catastrophic Mexico Beach, FL coastal tile ($1024 \times 1024$). Processing other geographic sectors requires providing corresponding bounding box coordinates in Upload mode.
2. **CPU-Only Execution:** Models execute via CPU onnxruntime. While latency is under 250 ms, GPU TensorRT acceleration would be required for ultra-high-throughput real-time video stream processing (>30 FPS).
3. **Planar Elevation Assumption:** Coordinates are mapped via 2D affine bounding box transformation; severe digital elevation models (DEM) or terrain relief are not orthorectified on the fly.
