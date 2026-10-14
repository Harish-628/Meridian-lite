# Dataset Card: Hurricane Michael Mexico Beach Damage Assessment Benchmark

## 1. Dataset Summary
- **Event Name:** Hurricane Michael (Category 5 landfall, October 10, 2018)
- **Geographic Area:** Bay County (Panama City / Callaway Area), Florida, USA
- **Bounding Box (WGS84 EPSG:4326):**
  - Minimum Longitude: -85.5501°
  - Minimum Latitude: 30.1530°
  - Maximum Longitude: -85.5473°
  - Maximum Latitude: 30.1558°
- **Primary Imagery Dimensions:** 1024 × 1024 pixels bi-temporal pair
- **Spatial Resolution (GSD):** ~0.3 meters / pixel (high-resolution aerial orthophotography)

---

## 2. Provenance & Attribution
- **Pre-Disaster Imagery Source:**
  - Esri World Imagery (High-Resolution Aerial Orthophoto, archived August 2018).
- **Post-Disaster Imagery Source:**
  - NOAA National Geodetic Survey (NGS) Emergency Response Imagery flight flightlines acquired on October 11-12, 2018.
- **Benchmark Curation & Annotation:**
  - DamageTriage-Bench / Open FEMA disaster analytics benchmark suite.
- **License:**
  - NOAA Emergency Response Aerial Imagery is in the **Public Domain** (U.S. Federal Government work).
  - DamageTriage-Bench curated annotations and alignment are released under **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**.
- **Citation:**
  > National Oceanic and Atmospheric Administration (NOAA) Remote Sensing Division, "Emergency Response Imagery - Hurricane Michael," 2018.
  > DamageTriage-Bench, "Bi-temporal Building Damage Assessment Benchmark Suite," 2024.

---

## 3. Data Structure & Directory Layout
```
data/
├── demo/
│   ├── buildings.geojson         # 18 vectorized building footprints (EPSG:4326)
│   ├── pre_disaster.png          # 800x800 RGB pre-event demo tile
│   ├── post_disaster.png         # 800x800 RGB post-event demo tile
│   └── README.md                 # Demo dataset documentation
├── raw/
│   ├── hurricane_michael_pre.png                 # 1024x1024 pre-hurricane aerial tile
│   ├── hurricane_michael_post.png                # 1024x1024 post-hurricane aerial tile
│   └── hurricane_michael_ground_truth_mask.png   # 1024x1024 damage annotation mask
└── processed/
    ├── dataset_metadata.json                     # Georeferencing bounds & CRS
    ├── hurricane_michael_pre.png
    ├── hurricane_michael_post.png
    └── hurricane_michael_ground_truth_mask.png
```

---

## 4. Damage Classification Schema (EMS-98 / Joint Damage Scale)
The dataset benchmarks 4 discrete structural damage tiers:
1. **No Damage (Class 0 / Green):** Intact structure, no visible roof or wall failure.
2. **Minor Damage (Class 1 / Yellow):** Partial shingle loss, broken windows, debris in yard.
3. **Major Damage (Class 2 / Orange):** Significant roof compromise (>25%), partial wall collapse.
4. **Destroyed (Class 3 / Red):** Total structural failure, roof and foundation collapsed or swept away by storm surge.

---

## 5. Limitations & Caveats
- **Coverage Extent:** This benchmark dataset focuses on a coastal strip of Mexico Beach, Florida that experienced catastrophic storm surge (>14 ft) and 160 mph sustained winds.
- **Occlusions & Debris:** Post-event imagery contains storm surge sand deposits and standing seawater which can alter spectral characteristics.
- **Offline Processing:** Imagery and vector layers are stored locally and execute 100% offline without requiring external network connectivity or cloud APIs.
