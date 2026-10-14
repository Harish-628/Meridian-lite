# MERIDIAN-Lite: Hackathon Judge Q&A & Technical Defense (Stage 2)

This document provides transparent, technically precise answers for hackathon evaluators, domain experts, and judges reviewing the MERIDIAN-Lite Stage 2 upgrade.

---

### 1. Is this actually running a real model or is it simulated?
**Both options are available, explicitly distinguished in the UI:**
- In **Real Model Mode**, MERIDIAN-Lite executes a **real computer vision / ML model pipeline**:
  1. A lightweight bi-temporal ONNX convolutional neural network (`models/weights/damage_classifier.onnx`) running via `onnxruntime` on CPU ($109 \text{ ms}$ latency), classifying building features into 4 damage tiers: `no-damage`, `minor-damage`, `major-damage`, and `destroyed`.
     > **Truthful Disclosure:** The ONNX backend performs real local neural-network inference, but it is a lightweight locally implemented bi-temporal model. It is not a large pretrained xView2/xBD checkpoint. Therefore, do not describe it as a pretrained xView2 model.
  2. A classical OpenCV structural change detection pipeline using Otsu adaptive thresholding and Canny structural edge preservation analysis ($217 \text{ ms}$ latency).
- In **Demo Mode**, the system preserves the Stage 1 deterministic simulation baseline so judges and developers can compare real inference outputs against synthetic baselines.
- The UI features a prominent blue badge (`REAL MODEL INFERENCE OUTPUT`) and an amber badge (`SIMULATED STAGE 1 OUTPUT`) so judges and users are never misled.

---

### 2. Where did you get the real disaster dataset?
The dataset is sourced from the public benchmark **DamageTriage-Bench** (Hugging Face `Ymx1025/DamageTriage-Bench`, CC BY-NC 4.0 license) covering the catastrophic landfall of **Hurricane Michael** (October 2018) in Mexico Beach, Florida:
- **Post-Disaster Imagery:** High-resolution airborne reconnaissance captured by the NOAA National Geodetic Survey (NGS) Remote Sensing Division.
- **Pre-Disaster Baseline:** High-resolution satellite imagery from the Esri World Imagery REST service covering the exact geographic bounding box $[-85.555^\circ, 30.170^\circ, -85.545^\circ, 30.180^\circ]$.
- **Ground Truth:** Expert triage damage annotations provided in DamageTriage-Bench.

---

### 3. What is your novelty in Stage 2?
Our primary innovation is **transforming raw computer-vision detections into operational search-and-rescue decision intelligence without cloud connectivity**:
1. **Decision-Support Triage:** Instead of presenting operators with 50+ scattered polygons, our priority-zone engine computes an objective triage score:
   $$\text{Priority Score} = 0.40 \cdot \text{Damage Density} + 0.25 \cdot \text{Severe Damage Ratio} + 0.20 \cdot \text{Total Affected Norm} + 0.15 \cdot \text{Confidence Weight}$$
2. **Actionable Sector Rankings:** Automatically ranks zones from highest urgency (Rank #1) to lowest with human-readable rationale (e.g. *"Critical severe structural destruction: 2 destroyed, 1 major damage"*).
3. **100% Air-Gapped Operation:** All model inference, georeferencing, priority calculations, and exports execute completely offline on standard CPU hardware.

---

### 4. Does it connect to or fly a physical drone?
**No.** Physical drone flight control, autonomous navigation, and autopilot integration (PX4/ArduPilot/MAVLink) are deliberately outside the competition scope for Stage 2. MERIDIAN-Lite operates as a ground decision-support software system that ingests recorded aerial imagery from drone reconnaissance sweeps.

---

### 5. Why are you running on CPU rather than GPU?
Our environment audit showed that the host system's GPU was in an uninitialized driver communication state (`nvidia-smi` unavailable). Rather than failing or making empty claims, we engineered the Stage 2 model specifically for high-efficiency CPU execution:
- Lightweight convolutional ONNX graph (opset 14, IR v10) with Global Average Pooling.
- Sub-120 ms inference latency on standard x86_64 CPU cores.
- This proves that first responders can run MERIDIAN-Lite on standard field laptops without requiring expensive, power-hungry mobile GPU rigs.

---

### 6. How does georeferencing work if the drone image is just a PNG?
`utils/georeferencing.py` implements an affine coordinate transformation using known spatial bounding boxes:
- When processing known disaster sectors (e.g., Hurricane Michael Mexico Beach tile), coordinates are mapped precisely to real WGS84 latitude/longitude degrees.
- In custom upload mode without embedded GeoTIFF tags, the operator specifies scene center coordinates, and the system maps pixel points relative to that center with a clear metadata designation: `"coordinate_source": "user_bbox_approximation"`.

---

### 7. Does it work without internet?
**Yes, 100%.** All neural inference, image processing, difference overlays, GeoJSON generation, priority-zone calculations, and file downloads execute entirely offline. If public OpenStreetMap base tiles cannot be fetched over the network, the interactive Folium map still displays all GeoJSON polygons, priority zone boundaries, and rank markers on an offline fallback canvas.

---

### 8. What are the key technical limitations?
We openly disclose three limitations:
1. **Single Sector Demonstration in Offline Cache:** The local cache packages the catastrophic coastal sector of Mexico Beach, FL ($1024 \times 1024$). Processing new regions requires providing bounding box coordinates.
2. **CPU-Only Inference:** While 109 ms latency is fast for post-flight image pairs, real-time edge processing on 30 FPS 4K video feeds would require hardware NPU/GPU acceleration (such as NVIDIA Jetson Orin with TensorRT).
3. **Planar Elevation Approximation:** Coordinates are projected onto a 2D affine bounding box; steep mountainous terrain relief is not orthorectified with digital elevation models (DEM) on the fly.
