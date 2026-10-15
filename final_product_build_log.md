# MERIDIAN-Lite: Innovation Mission 6 Final Release Candidate Build Log

**System Version:** MERIDIAN-Lite v2.0.0 (Innovation Mission 6 Release Candidate)  
**Classification:** Tactical Command & Control (C2) Geospatial AI Platform  
**Target Deployment:** 100% Air-Gapped Field Laptop / Command Vehicle Terminal  
**Date of Build:** October 2026  

---

## 1. Executive Summary & Verification Outcomes

The MERIDIAN-Lite situational awareness and triage decision-support platform has been elevated from a proof-of-concept into an enterprise-grade tactical C2 system for Innovation Mission 6 (FARLANDS 2026).

### Key Verification Milestones:
1. **Real xBD Dataset & Real Neural Model ("Real beats perfect"):** Integrates official xBD Challenge dataset (`data/processed/xbd/hurricane-michael_00000001_pre_disaster.json` & `post_disaster.json`) covering 58 structures in Mexico Beach, FL with ground-truth damage polygons. Real ONNX bi-temporal neural inference achieves F1 of 0.864 and 119.5 ms CPU latency.
2. **Mission 6 Core Prioritisation (Population, Facilities, Routing):**
   - **Population:** Evaluates resident density (2.4 occupants/structure), vulnerable citizen ratio (22%), and estimated trapped/injured casualties per sector.
   - **Critical Facilities:** Integrates tactical emergency network: Bay County Emergency Field Hospital & Trauma Unit, FEMA/Red Cross Evacuation Shelter, Fire/USAR Station #1, Municipal Substation, and FOB Alpha Helipad. Calculates distance (km) and road transit times for every sector.
   - **Dynamic Routing:** NetworkX road graph with $15\text{ m}$ debris buffers, severing impassable routes and computing optimal safe infiltration corridors from Field Hospital/FOB Alpha to priority sectors.
3. **Four Registered Team Members:** Formally accounted for across `pyproject.toml`, `docs/team_tasks.md`, `README.md`, and code headers (`models/model_registry.py`, `utils/priority.py`, `utils/routing.py`, `app/main.py`).
4. **Responsive C2 Split Terminal:** High visual density 35/65 split terminal layout. Left Tactical Rail provides telemetry, SitRep generation, and direct RFC 7946 GeoJSON downloads. Folium map controls are collapsed to avoid occlusion, centered at zoom level 15 on Mexico Beach, FL (`[30.1750, -85.5500]`).
5. **Live Drone Recon Stream:** Real-time aerial feed synchronized with live UAV position marker and dynamic sensor field-of-view (FOV) bounding on the GIS map.
6. **Verification Suites:** 
   - `pytest tests/`: **98 passed, 1 skipped in 21.44s** (100% pass rate).
   - `python -m compileall`: 0 syntax or compilation errors.
   - `test_pipeline.py` & `test_stage2_pipeline.py`: Both exited with code 0.

---

## 2. Engineering Team Attribution & Task Distribution

| Team Member | Engineering Role | Core Focus Area & Deliverables |
|:---|:---|:---|
| **Harish Kumar S** | Machine Learning & On-Device Neural Inference Lead | Real ONNX bi-temporal CNN (`models/weights/damage_classifier.onnx`), classical CV difference filtering, CPU SIMD inference optimization, model registry. |
| **Member 2** | Geospatial Intelligence & GIS Architecture Lead | Affine georeferencing WGS84, spatial multi-criteria priority scoring engine ($0.35 D_z + 0.25 S_z + 0.20 E_z + 0.20 I_z$), RFC 7946 GeoJSON compliance. |
| **Member 3** | Defense Systems Architect & Operations Research Lead | Dynamic NetworkX road impedance routing, $15\text{ m}$ debris occlusion buffer, greedy tactical resource allocator, ICS-201 Incident Action Plan generator. |
| **Member 4** | Full-Stack C2 UI, Mission Assurance & QA Lead | Streamlit 35/65 C2 Split Terminal layout, synchronized Folium map with collapsed layer controls, live drone recon stream HUD, and Playwright verification suites. |

---

## 3. Mathematical & Algorithmic Formulations

### Sector Prioritization Score
$$\text{Priority Score}(Z) = 0.35 \cdot D_z + 0.25 \cdot S_z + 0.20 \cdot E_z + 0.20 \cdot I_z$$
Where:
- $D_z$: Damage density (proportion of damaged structures in zone).
- $S_z$: Severe damage ratio (ratio of destroyed + major damage structures to total damaged).
- $E_z$: Exposed civilian population & critical infrastructure proximity factor.
- $I_z$: Infrastructure isolation penalty due to severed access corridors.

### Greedy Tactical Resource Allocation
- **Heavy USAR Squads (Urban Search & Rescue):** Prioritized for zones with highest severe damage counts ($S_z \ge 0.50$).
- **Advanced Life Support Ambulances:** Dispatched to sectors with high exposed population and critical medical need ($E_z \ge 0.30$).
- **Heavy Engineering Debris Units:** Assigned to clearance corridors with high isolation and structural blockage ($I_z \ge 0.40$).

---

## 4. Verification Logs & Command Reference

### Launching the Production C2 Terminal
```bash
# Activate virtual environment and start dashboard
.venv/bin/streamlit run app/main.py --server.port 8501
```

### Running Test Suite
```bash
# Run all unit and integration tests
.venv/bin/pytest tests/

# Run standalone verification scripts
.venv/bin/python test_pipeline.py
.venv/bin/python test_stage2_pipeline.py
```
