# Cleanup & Archival Release Report

Conducted: 2026-10-14 23:32 IST

## 1. Summary
- **Files Safely Deleted:** 57 files (544,491 bytes / 0.52 MB)
- **Files Preserved in `archive/`:** 8 files (1,445,815 bytes / 1.38 MB)
- **Critical Core Files Protected:** 100% of model weights, raw/processed datasets, documentation, test suites, and production code preserved.

## 2. Archived Files (`archive/`)

| File Original Path | Size (Bytes) | Category / Rationale |
|:---|---:|:---|
| `outputs/temp_downloads/browser_test_buildings.csv` | 1,540 | Ephemeral browser test download artifact |
| `outputs/temp_downloads/stage2_browser_damage.geojson` | 79,802 | Ephemeral browser test download artifact |
| `outputs/temp_downloads/browser_test_overlay.png` | 1,045,147 | Ephemeral browser test download artifact |
| `outputs/temp_downloads/browser_test_summary.json` | 348 | Ephemeral browser test download artifact |
| `outputs/temp_downloads/stage2_browser_priority_zones.json` | 3,657 | Ephemeral browser test download artifact |
| `outputs/temp_downloads/browser_test_damage.geojson` | 13,853 | Ephemeral browser test download artifact |
| `outputs/temp_downloads/browser_onnx_damage.geojson` | 79,802 | Ephemeral browser test download artifact |
| `outputs/reports/meridian_stage1.log` | 221,666 | Legacy Stage 1 execution runtime log |

## 3. Deleted Files (Caches & Bytecode)

| File Path | Size (Bytes) | Reason |
|:---|---:|:---|
| `models/__pycache__/model_registry.cpython-314.pyc` | 29,289 | Compiled Python bytecode (.pyc) |
| `models/__pycache__/predict.cpython-314.pyc` | 6,685 | Compiled Python bytecode (.pyc) |
| `models/__pycache__/__init__.cpython-314.pyc` | 734 | Compiled Python bytecode (.pyc) |
| `models/__pycache__/real_predict.cpython-314.pyc` | 7,899 | Compiled Python bytecode (.pyc) |
| `__pycache__/test_stage2_pipeline.cpython-314-pytest-9.1.1.pyc` | 24,546 | Compiled Python bytecode (.pyc) |
| `__pycache__/test_stage2_pipeline.cpython-314.pyc` | 10,674 | Compiled Python bytecode (.pyc) |
| `__pycache__/test_pipeline.cpython-314-pytest-9.1.1.pyc` | 18,279 | Compiled Python bytecode (.pyc) |
| `__pycache__/test_pipeline.cpython-314.pyc` | 7,939 | Compiled Python bytecode (.pyc) |
| `tests/__pycache__/test_images.cpython-314-pytest-9.1.1.pyc` | 15,759 | Compiled Python bytecode (.pyc) |
| `tests/__pycache__/test_real_pipeline.cpython-314.pyc` | 5,466 | Compiled Python bytecode (.pyc) |
| `tests/__pycache__/__init__.cpython-314.pyc` | 166 | Compiled Python bytecode (.pyc) |
| `tests/__pycache__/test_georeferencing.cpython-314.pyc` | 5,784 | Compiled Python bytecode (.pyc) |
| `tests/__pycache__/test_priority.cpython-314.pyc` | 5,045 | Compiled Python bytecode (.pyc) |
| `tests/__pycache__/test_real_model.cpython-314-pytest-9.1.1.pyc` | 17,511 | Compiled Python bytecode (.pyc) |
| `tests/__pycache__/test_browser_live.cpython-314-pytest-9.1.1.pyc` | 16,424 | Compiled Python bytecode (.pyc) |
| `tests/__pycache__/test_predict.cpython-314-pytest-9.1.1.pyc` | 20,278 | Compiled Python bytecode (.pyc) |
| `tests/__pycache__/test_app_import.cpython-314.pyc` | 2,969 | Compiled Python bytecode (.pyc) |
| `tests/__pycache__/test_real_pipeline.cpython-314-pytest-9.1.1.pyc` | 20,289 | Compiled Python bytecode (.pyc) |
| `tests/__pycache__/test_geojson.cpython-314.pyc` | 6,257 | Compiled Python bytecode (.pyc) |
| `tests/__pycache__/test_images.cpython-314.pyc` | 6,192 | Compiled Python bytecode (.pyc) |
| `tests/__pycache__/test_real_model.cpython-314.pyc` | 5,332 | Compiled Python bytecode (.pyc) |
| `tests/__pycache__/test_browser_stage2_live.cpython-314.pyc` | 8,481 | Compiled Python bytecode (.pyc) |
| `tests/__pycache__/test_metrics.cpython-314.pyc` | 4,196 | Compiled Python bytecode (.pyc) |
| `tests/__pycache__/test_georeferencing.cpython-314-pytest-9.1.1.pyc` | 19,932 | Compiled Python bytecode (.pyc) |
| `tests/__pycache__/test_ui_app.cpython-314-pytest-9.1.1.pyc` | 17,202 | Compiled Python bytecode (.pyc) |
| `tests/__pycache__/test_browser_stage2_live.cpython-314-pytest-9.1.1.pyc` | 12,287 | Compiled Python bytecode (.pyc) |
| `tests/__pycache__/test_real_model_inference.cpython-314-pytest-9.1.1.pyc` | 30,960 | Compiled Python bytecode (.pyc) |
| `tests/__pycache__/test_browser_live.cpython-314.pyc` | 9,487 | Compiled Python bytecode (.pyc) |
| `tests/__pycache__/test_app_import.cpython-314-pytest-9.1.1.pyc` | 10,180 | Compiled Python bytecode (.pyc) |
| `tests/__pycache__/test_predict.cpython-314.pyc` | 5,562 | Compiled Python bytecode (.pyc) |
| `tests/__pycache__/test_browser_onnx_live.cpython-314-pytest-9.1.1.pyc` | 17,633 | Compiled Python bytecode (.pyc) |
| `tests/__pycache__/test_metrics.cpython-314-pytest-9.1.1.pyc` | 15,733 | Compiled Python bytecode (.pyc) |
| `tests/__pycache__/test_browser_onnx_live.cpython-314.pyc` | 9,315 | Compiled Python bytecode (.pyc) |
| `tests/__pycache__/test_geojson.cpython-314-pytest-9.1.1.pyc` | 21,436 | Compiled Python bytecode (.pyc) |
| `tests/__pycache__/test_priority.cpython-314-pytest-9.1.1.pyc` | 15,142 | Compiled Python bytecode (.pyc) |
| `utils/__pycache__/georeferencing.cpython-314.pyc` | 9,372 | Compiled Python bytecode (.pyc) |
| `utils/__pycache__/metrics.cpython-314.pyc` | 5,042 | Compiled Python bytecode (.pyc) |
| `utils/__pycache__/__init__.cpython-314.pyc` | 2,140 | Compiled Python bytecode (.pyc) |
| `utils/__pycache__/logging_utils.cpython-314.pyc` | 3,123 | Compiled Python bytecode (.pyc) |
| `utils/__pycache__/paths.cpython-314.pyc` | 2,349 | Compiled Python bytecode (.pyc) |
| *...and 17 additional cache files* | | |

## 4. Final Directory Layout (Post-Cleanup)

```
Meridian-lite/
├── app/                      # Streamlit application coordinator & UI modules
│   ├── main.py
│   └── ui/                   # Modular UI presentation views
├── archive/                  # Safely archived legacy test outputs and logs
├── data/                     # Offline disaster datasets (demo, raw, processed)
├── docs/                     # Architectural, audit, and demo documentation
├── models/                   # Neural and CV damage classification backends
│   └── weights/              # Local ONNX model weights
├── outputs/                  # Verified outputs (GeoJSON, overlays, reports)
│   ├── geojson/
│   ├── overlays/
│   ├── reports/
│   ├── temp_downloads/       # Empty .gitkeep
│   └── temp_uploads/         # Empty .gitkeep
├── tests/                    # Pytest unit, integration, and Playwright E2E suites
├── utils/                    # Geospatial, priority, metric, and logging utilities
├── pyproject.toml            # Modern PEP 517/518/621 packaging metadata
├── requirements.txt          # Python dependency specifications
└── README.md                 # Primary system guide
```
