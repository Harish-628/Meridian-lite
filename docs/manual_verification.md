# MERIDIAN-Lite: Manual & Automated Browser Verification Guide (Stage 2)

This document provides both the manual testing checklist and the automated Playwright verification procedures for MERIDIAN-Lite Stage 1 and Stage 2.

---

## 1. Automated Browser Verification (Playwright)

Automated headless browser testing is available and pre-verified using Playwright and Chromium:

```bash
cd /home/harish/Music/Meridian-lite
source .venv/bin/activate

# 1. Run Stage 1 automated browser verification (Demo mode)
.venv/bin/python tests/test_browser_live.py

# 2. Run Stage 2 automated browser verification (Real Model Mode & Priority Zones)
.venv/bin/python tests/test_browser_stage2_live.py

# 3. Run Stage 2 ONNX dedicated browser verification & export check
.venv/bin/python tests/test_browser_onnx_live.py
```

All three tests automatically start a background Streamlit server, launch headless Chromium, click through all controls, verify all metrics and priority zone elements, execute downloads, and save visual screenshots to:
- `outputs/reports/browser_dashboard_screenshot.png` (Stage 1)
- `outputs/reports/browser_real_model_screenshot.png` (Stage 2)
- `outputs/reports/onnx_browser_verification.png` (Stage 2 ONNX Live Verification)

> **Architecture Disclosure:** The ONNX backend performs real local neural-network inference, but it is a lightweight locally implemented bi-temporal model. It is not a large pretrained xView2/xBD checkpoint. Therefore, do not describe it as a pretrained xView2 model.

---

## 2. Manual Browser Testing Protocol

### Starting the Application
```bash
.venv/bin/python -m streamlit run app/main.py --server.port 8501
```
Open `http://localhost:8501` in your browser.

---

### Test Case 1: Stage 1 Demo Mode Verification
1. **Sidebar:** Ensure `Assessment Mode` is set to `Demo mode`.
2. **Warning Banner:** Confirm amber warning: *"Demo mode uses simulated imagery and simulated damage output..."*
3. **Execution:** Click `Run AI Damage Assessment`.
4. **Verification:**
   - 18 simulated building footprints appear around center coordinates ($13.0827, 80.2707$).
   - Metrics cards display: Total Buildings Detected (18), Destroyed/Major Damage count, Mean Confidence.
   - Pre/post synthetic images and difference overlay render in 3 columns.
   - Interactive Folium map renders with colored building polygons.
   - Download buttons for GeoJSON, Summary JSON, CSV, and Overlay PNG operate correctly.

---

### Test Case 2: Stage 2 Real Model Mode Verification (Hurricane Michael)
1. **Sidebar Mode Selection:** Select **"Real Model Mode"** in the sidebar.
2. **Visual Distinction Banner:**
   - Confirm prominent blue banner appears:
     > `REAL MODEL INFERENCE OUTPUT`  
     > *Dataset: DamageTriage-Bench (Hurricane Michael, Mexico Beach FL) · Model: Bi-Temporal ONNX CNN / Classical CV*
3. **Model Backend Selection:**
   - Choose **"Real Damage Model (ONNX)"** (runs bi-temporal CNN via onnxruntime on CPU).
   - Alternatively, choose **"Classical Change Detection (OpenCV)"**.
4. **Parameters:**
   - Dataset: `Hurricane Michael (Mexico Beach, FL - Oct 2018)`.
   - Confidence Threshold: `0.35`.
   - Priority Zone Grid Size: `3 x 3 (9 Zones)`.
5. **Execute Inference:**
   - Click **"Run AI Damage Assessment"**.
   - Processing time toast appears indicating completion in ~100 ms.
6. **Tactical Assessment Summary:**
   - Total Buildings Detected: 42 to 51 structures.
   - Destroyed / Major Damage count and percentage.
   - Mean Assessment Confidence.
7. **Stage 2 Priority-Zone Intelligence Section:**
   - Metric card shows Top Priority Zone (e.g. `Z-002`).
   - Priority Score, Severe Structures, and Total Affected count.
   - Ranked table displays: `Priority Rank`, `Zone ID`, `Priority Score`, `Total Structures`, `Damaged Structures`, `Destroyed`, `Major Damage`, `Mean Confidence`, and `Triage Rationale`.
8. **Interactive Geospatial Folium Map:**
   - Centered on Mexico Beach, Florida ($30.1750^\circ \text{N}, -85.5500^\circ \text{W}$).
   - Building damage polygons colored by status (`#28a745` green, `#ffc107` yellow, `#fd7e14` orange, `#dc3545` red).
   - Priority zone boundary rectangles with rank callout markers (`Rank #1 (Z-002)`, `Rank #2 (Z-003)`).
   - Tooltips show damage level and model confidence; clicking shows detailed popups.
9. **Export Downloads:**
   - Click `📥 Download GeoJSON` → saves `meridian_damage_assessment.geojson`.
   - Click `📥 Download Priority Zones JSON` → saves `meridian_priority_zones.json`.
   - Click `📥 Download CSV Results` → saves `meridian_building_assessment.csv`.
   - Click `📥 Download Damage Overlay` → saves `meridian_damage_overlay.png`.

---

### Test Case 3: Custom Upload Mode Verification
1. **Sidebar:** Select `Upload mode`.
2. **File Uploads:** Upload custom pre-disaster and post-disaster PNG/JPEG image files.
3. **Execution:** Click `Run AI Damage Assessment`.
4. **Verification:** System processes the uploaded pair, resizes safely if resolutions differ, generates difference overlay, computes metrics and priority zones, and enables downloads.
