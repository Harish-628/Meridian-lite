# MERIDIAN-Lite: Hackathon Judge Q&A & Technical Defense (Stage 2)

This document provides transparent, technically precise answers for hackathon evaluators, domain experts, and judges reviewing the MERIDIAN-Lite Stage 2 upgrade.

---

### 1. Is this model pretrained?
**No. Pretrained status is marked as not independently verified.**
The current MERIDIAN-Lite ONNX model is a lightweight local damage classifier. Its model-file format and successful inference do not by themselves prove that it is an officially pretrained xBD/xView2 checkpoint. Pretrained status is marked as not independently verified until training provenance is documented.

It is an internally synthesized, hand-parameterized 2-layer convolutional neural network (23.5 KB) designed for offline pipeline testing and bi-temporal change classification. It executes real local neural network inference on CPU via ONNX Runtime, but it is not an official checkpoint trained via stochastic gradient descent on xBD.

---

### 1b. What data was it trained on? Can you prove it?
**None.** Technical inspection of the ONNX weight initializers reveals that the convolutional filters are analytical spatial difference kernels ($-1/27$ on pre-disaster channels, $+1/27$ on post-disaster channels) paired with Laplacian edge detectors. No training dataset was used, which also guarantees **zero training/test data leakage** when evaluating test tiles. All details are formally documented in [`outputs/reports/model_provenance_audit.json`](../outputs/reports/model_provenance_audit.json) and [`docs/model_card.md`](model_card.md).

---

### 1c. Why did you choose this model instead of a larger pretrained checkpoint?
We conducted a formal candidate evaluation against the xView2 Baseline (U-Net Siamese ResNet-50, 195 MB, 51.2M parameters) and RescueNet (44.8 MB) — documented in [`outputs/reports/model_comparison.json`](../outputs/reports/model_comparison.json) and [`docs/model_selection_decision.md`](model_selection_decision.md):
1. **Zero External Dependency & Python 3.14 Compatibility:** PyTorch currently lacks pre-built wheels for Python 3.14, whereas our ONNX model executes natively with zero external C++ build issues.
2. **Deterministic Offline Execution (<120 ms):** The lightweight model processes 58 candidate building patches in 113.4 ms on CPU SIMD with <35 MB RAM, compared to ~4,800 ms and 1.45 GB RAM for a 50M parameter U-Net.
3. **Transparency over False Claims:** Rather than downloading unverified third-party checkpoints and claiming they are official xBD models, we maintain full transparency with clear labeling: *"Verified Local ONNX Model (Lightweight Demonstration CNN)"*.

---

### 1d. What happens if the network is disconnected?
**100% of the system continues to work.** MERIDIAN-Lite is fully air-gapped. The ONNX inference runtime, classical OpenCV fallbacks, geospatial affine coordinate projection, priority-zone calculations, and data exporters (GeoJSON/CSV/JSON) execute locally on CPU without making any network calls. If internet tiles cannot load, Folium renders all geospatial polygons and tactical markers on an offline fallback canvas.

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
