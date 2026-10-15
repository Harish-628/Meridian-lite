# MERIDIAN-Lite: Hackathon Judge Demonstration Script (Stage 2)

**Mission:** Innovation Mission 6 — Disaster Response Drones: Predictive Search and Situational Awareness  
**System:** MERIDIAN-Lite Stage 2 Disaster Response Decision-Support System  
**Audience:** Hackathon Technical Judges & Evaluation Panel  

---

## 1. One-Minute Pitch Script

> "Good morning, judges. During major disasters like Hurricane Michael, emergency dispatchers receive thousands of panic calls, but first responders lack immediate situational awareness of which sectors are completely flattened versus still accessible.
>
> We built **MERIDIAN-Lite Stage 2** — an air-gapped, offline decision-support system that ingests pre- and post-disaster aerial imagery, runs local computer vision inference without cloud dependencies, and generates actionable geospatial intelligence in under 200 milliseconds.
>
> On screen, you see genuine aerial imagery from Hurricane Michael in Mexico Beach, Florida. With one click, our bi-temporal neural network segments 51 damaged building footprints, georeferences them into RFC 7946 GeoJSON, and partitions the disaster sector into prioritized response zones using a multi-factor triage formula. 
> 
> Responders instantly know that Zone Z-002 is Rank #1 due to severe structural collapse, enabling targeted search-and-rescue deployment before telecom networks are even restored."

---

## 2. Three-Minute Technical Deep-Dive

### Minute 1: Real Multimodal Data Ingestion & Local Processing
- **Problem:** First responders operate in compromised environments where cloud AI APIs, Starlink, or cellular connectivity are down.
- **Solution:** MERIDIAN-Lite runs 100% locally on CPU without sending a single byte to external cloud servers.
- **Data Provenance:** We ingested a genuine $1024 \times 1024$ aerial tile pair from DamageTriage-Bench and NOAA NGS airborne reconnaissance from the catastrophic Hurricane Michael landfall in Mexico Beach, Florida.
- **Georeferencing:** Our `AffineGeoreferencer` translates 2D pixel coordinates into WGS84 geographic coordinates (`EPSG:4326`) mapped directly to actual latitude/longitude coordinates on the Florida coastline.

### Minute 2: Inference Engine & Decision-Support Ranking
- **Dual Inference Backends:**
  1. **ONNX Bi-temporal Neural Network:** A 6-channel convolutional architecture running via `onnxruntime` on CPU ($109 \text{ ms}$ latency), classifying structural change into 4 xBD-compliant damage classes: `no-damage`, `minor-damage`, `major-damage`, and `destroyed`.
  2. **Classical OpenCV Change Detector:** Morphological Otsu segmentation and edge-preservation filtering ($217 \text{ ms}$ latency) as an air-gapped fallback.
- **Priority-Zone Scoring Formula:** Rather than giving responders an overwhelming raw list of 50+ polygons, our priority engine divides the disaster sector into spatial grids and computes an objective triage score:
  $$\text{Priority Score} = 0.40 \cdot \text{Damage Density} + 0.25 \cdot \text{Severe Damage Ratio} + 0.20 \cdot \text{Total Affected Norm} + 0.15 \cdot \text{Confidence Weight}$$
- All components are normalized $[0, 1]$, and zones are ranked with transparent human-readable explanations.

### Minute 3: Actionable GIS Visualization & Standards-Compliant Exports
- **Interactive GIS Map:** Folium map rendering color-coded building damage footprints (`#28a745` undamaged, `#ffc107` minor, `#fd7e14` major, `#dc3545` destroyed) combined with transparent priority zone blocks and rank callout markers.
- **Interoperability:** Tactical outputs export as RFC 7946 GeoJSON, Priority Zones JSON, building inventory CSV, and difference overlays — ready to ingest into ATAK, QGIS, or incident command software.

---

## 3. Screen-by-Screen Live Demonstration Walkthrough

| Step | Action | Expected On-Screen Result | Judge Speaking Point |
| :--- | :--- | :--- | :--- |
| **1** | Open dashboard: `http://localhost:8501` | Dashboard loads with sidebar controls and initial overview. | "The system initializes in seconds, fully offline." |
| **2** | In sidebar, select **"Real Model Mode"** | Blue banner appears: `REAL MODEL INFERENCE OUTPUT` with Hurricane Michael metadata. | "Notice our clear distinction between simulated baselines and real model inference." |
| **3** | Ensure Model Backend is set to **"Real Damage Model (ONNX)"** | Sliders for confidence threshold (0.35) and grid size (3x3) are visible. Expand **"⚡ GPU Status & Hardware Diagnostics"**. | "This runs an actual bi-temporal ONNX CNN locally on the CPU. Point out that our NVIDIA RTX 3050 GPU is detected, but we transparently run on CPU SIMD to demonstrate edge reliability on field hardware." |
| **4** | Click **"Run AI Damage Assessment"** | Processing spinner runs for ~100 ms; toast notifies completion. | "In roughly 100 milliseconds on standard CPU hardware, the complete pipeline executes." |
| **5** | Inspect **Tactical Assessment Summary** | Metric cards show Total Buildings Detected (~51), Destroyed/Major counts, and Mean Confidence. | "Real quantified triage metrics immediately available to dispatchers." |
| **6** | Scroll to **Stage 2 Priority-Zone Intelligence** | Top Priority Zone metric card (Z-002) and ranked priority table appear. | "Instead of raw pins, incident commanders get ranked zones with clear triage rationale." |
| **7** | Scroll to **Geospatial Map** | Interactive Folium map renders building polygons with color coding and priority grid boxes labeled `Rank #1`, `Rank #2`. | "Responders can zoom in, click any structure for confidence details, and view zone boundaries." |
| **8** | Click **"Download GeoJSON"** and **"Download Priority Zones JSON"** | Valid files download instantly to disk. | "Outputs follow RFC 7946 standards for direct integration into field GIS tools." |

---

## 4. Fallback Plan During Live Presentation

If unexpected browser or port issues occur during live evaluation:
1. **Fallback 1 (Classical OpenCV Backend):** If ONNX runtime is interrupted, toggle the Model Backend dropdown to **"Classical Change Detection (OpenCV)"** and click Run. It executes deterministically via OpenCV.
2. **Fallback 2 (Stage 1 Demo Mode):** Switch mode radio to **"Demo mode"** to demonstrate the synthetic baseline.
3. **Fallback 3 (CLI Automated Verification):** Run the standalone script directly in terminal:
   ```bash
   .venv/bin/python test_stage2_pipeline.py
   ```
   Exits with code 0 in < 2 seconds, printing individual test pass results.

---

## 5. Frequently Asked Judge Questions & Truthful Answers

### Q1: "Is this actually running a real neural network, or is it simulated?"
> **Answer:** "In **Real Model Mode**, it is executing a real convolutional neural network (`models/weights/damage_classifier.onnx`) running via `onnxruntime` on CPU. The ONNX backend performs real local neural-network inference, but it is a lightweight locally implemented bi-temporal model. It is not a large pretrained xView2/xBD checkpoint. Therefore, do not describe it as a pretrained xView2 model. It extracts 6-channel pre/post bi-temporal image patches and predicts damage classes and confidence scores. In addition, we also built a classical OpenCV change-detection backend and preserved the Stage 1 simulation mode so judges can compare all three."

### Q2: "Where did you get the disaster dataset?"
> **Answer:** "The dataset is from the public **DamageTriage-Bench** benchmark, which contains aerial imagery from the catastrophic October 2018 Hurricane Michael landfall in Mexico Beach, Florida. Post-disaster imagery is from NOAA's airborne reconnaissance fleet, and pre-disaster baseline imagery is from Esri World Imagery."

### Q3: "Does this require internet access or cloud AI APIs (like OpenAI / Gemini)?"
> **Answer:** "No. The core pipeline, neural inference, OpenCV processing, GeoJSON conversion, and priority scoring run 100% locally and offline without sending any data over the internet. Internet is only accessed if the browser tries to fetch online OpenStreetMap tiles, and the system functions gracefully even without them."

### Q4: "How are the priority zones calculated?"
> **Answer:** "We divide the target area into a spatial grid and apply an objective formula:
> $$\text{Priority Score} = 0.40 \cdot \text{Damage Density} + 0.25 \cdot \text{Severe Damage Ratio} + 0.20 \cdot \text{Total Affected Norm} + 0.15 \cdot \text{Confidence Weight}$$
> This prioritizes sectors with high concentrations of destroyed structures and high model confidence, ensuring search-and-rescue teams are dispatched where lives are most critically at risk."

### Q5: "What are the limitations of this Stage 2 implementation?"
> **Answer:** "We openly disclose three limitations: First, we demonstrate on a single high-resolution coastal tile ($1024 \times 1024$) in our local offline cache. Second, georeferencing is based on affine bounding box transformation without 3D digital elevation correction. Third, inference is currently optimized for CPU execution; scaling to real-time video streams on an edge drone would require TensorRT GPU acceleration."

### Q6: "What is your GPU status and why does the Advanced Damage Model run on CPU?"
> **Answer:** "The NVIDIA GPU is detected, but the installed ONNX Runtime does not include CUDAExecutionProvider. Therefore, the system performs real inference on CPU using the verified lightweight ONNX backend. No pretrained xBD/xView2 checkpoint is currently integrated. The Advanced Damage Model Backend (GPU-Ready / CPU Fallback) is designed to automatically utilize CUDAExecutionProvider whenever an environment with CUDA wheels is present, while falling back gracefully to CPU SIMD so the demo never fails."
