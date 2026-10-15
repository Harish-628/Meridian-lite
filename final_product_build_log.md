# MERIDIAN-Lite: Innovation Mission 6 Final Release Candidate Build Log

**System Version:** MERIDIAN-Lite v2.0.0 (Innovation Mission 6 Release Candidate)  
**Classification:** Tactical Command & Control (C2) Geospatial AI Platform  
**Target Deployment:** 100% Air-Gapped Field Laptop / Command Vehicle Terminal  
**Date of Build:** October 2026  

---

## 1. Executive Summary & Verification Outcomes

The MERIDIAN-Lite situational awareness and triage decision-support platform has been elevated from a proof-of-concept into an enterprise-grade tactical C2 system for Innovation Mission 6 (FARLANDS 2026).

### Key Verification Milestones:
1. **Real Disaster Default ("Real beats perfect"):** Initial launch defaults strictly to real Hurricane Michael disaster imagery (`data/processed/hurricane_michael_pre.png` & `post.png`) and real ONNX neural damage assessment. Synthetic demo mode is removed from default selection.
2. **Four Registered Team Members:** Formally accounted for across `pyproject.toml`, `docs/team_tasks.md`, `README.md`, and code headers (`models/model_registry.py`, `utils/priority.py`, `utils/routing.py`, `app/main.py`).
3. **Advanced Prioritization & Asset Allocation:** Multi-criteria formulation combining Damage Density ($D_z$), Severe Damage Ratio ($S_z$), Exposed Civilian Population ($E_z$), and Infrastructure Isolation ($I_z$). Includes greedy dispatching for USAR Squads, Ambulances, and Heavy Debris Units.
4. **Dynamic Road Impedance Routing:** NetworkX routing topology incorporating $15\text{ m}$ debris occlusion buffers around damaged structures and coastal flood hazards.
5. **Responsive C2 Split Terminal:** High visual density 35/65 split terminal layout. Left Tactical Rail provides telemetry, SitRep generation, and direct RFC 7946 GeoJSON downloads. Folium map controls are collapsed to avoid occlusion, centered at zoom level 15 on Mexico Beach, FL (`[30.1750, -85.5500]`).
6. **Live Drone Recon Stream:** Real-time aerial feed synchronized with live UAV position marker and dynamic sensor field-of-view (FOV) bounding on the GIS map.
7. **Verification Suites:** 
   - `pytest tests/`: **97 passed, 1 skipped in 21.58s** (100% pass rate).
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
