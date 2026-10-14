"""
MERIDIAN-Lite: Standalone Stage 2 Real-Data & Real-Model Pipeline Verification Script.
Executes and validates all Stage 2 pipeline components:
1. Stage 2 module imports
2. Real disaster dataset validation (Hurricane Michael - Mexico Beach, FL)
3. Model registry & CPU ONNX session verification
4. Classical OpenCV change detection inference
5. Real ONNX neural network inference
6. Georeferencing & WGS84 coordinate integrity
7. Priority-zone scoring & ranking engine
8. Tactical output serialization (GeoJSON, Priority Zones JSON, Metrics)

Exits with code 0 on complete success, or non-zero on failure.
"""

import json
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    print("=" * 65)
    print("MERIDIAN-Lite Stage 2 Real-Data & Real-Model Pipeline Verification")
    print("=" * 65)

    checks = []

    # 1. Module Imports
    try:
        from utils.georeferencing import AffineGeoreferencer
        from utils.priority import compute_priority_zones, priority_zones_to_geojson
        from models.model_registry import (
            RealDamageModelBackend,
            ClassicalChangeDetectionBackend,
            SimulatedXView2Backend,
            check_model_health,
            get_backend,
        )
        from models.real_predict import run_real_inference
        from utils.geojson_utils import validate_feature_collection, save_geojson
        from utils.image_utils import validate_image, load_image

        checks.append(("Stage 2 Module Imports", True, "All modules imported cleanly"))
    except Exception as exc:
        print(f"[FAIL] Module imports failed: {exc}")
        return 1

    # 2. Real Dataset Audit
    try:
        pre_path = PROJECT_ROOT / "data" / "processed" / "hurricane_michael_pre.png"
        post_path = PROJECT_ROOT / "data" / "processed" / "hurricane_michael_post.png"
        meta_path = PROJECT_ROOT / "data" / "processed" / "dataset_metadata.json"

        assert pre_path.exists() and pre_path.stat().st_size > 500000
        assert post_path.exists() and post_path.stat().st_size > 500000
        assert meta_path.exists()

        validate_image(pre_path)
        validate_image(post_path)

        with open(meta_path, "r") as f:
            meta = json.load(f)
        assert meta["disaster_name"] == "Hurricane Michael"

        checks.append((
            "Real Dataset Audit",
            True,
            f"Hurricane Michael tile 1024x1024 ({meta['location']})",
        ))
    except Exception as exc:
        checks.append(("Real Dataset Audit", False, str(exc)))
        print(f"[FAIL] Dataset audit failed: {exc}")
        return 1

    # 3. Model Registry & Health Check
    try:
        health = check_model_health()
        assert health["model_available"] is True
        assert health["device_available"] == "cpu"
        assert len(health["available_backends"]) >= 2
        checks.append((
            "Model Registry & Health",
            True,
            f"Active backends: {', '.join(health['available_backends'])}",
        ))
    except Exception as exc:
        checks.append(("Model Registry & Health", False, str(exc)))
        print(f"[FAIL] Model registry check failed: {exc}")
        return 1

    # 4. Classical Change Detection Inference
    try:
        t0 = time.perf_counter()
        fc_cv = run_real_inference(
            pre_image_path=pre_path,
            post_image_path=post_path,
            confidence_threshold=0.35,
            max_features=100,
            backend_name="classical_change_detection",
        )
        t_cv = (time.perf_counter() - t0) * 1000
        validate_feature_collection(fc_cv)
        n_cv = len(fc_cv.get("features", []))
        assert n_cv > 0
        checks.append((
            "Classical Change Detection",
            True,
            f"Detected {n_cv} building features ({t_cv:.1f} ms on CPU)",
        ))
    except Exception as exc:
        checks.append(("Classical Change Detection", False, str(exc)))
        print(f"[FAIL] Classical inference failed: {exc}")
        return 1

    # 5. Real ONNX Neural Model Inference
    try:
        t0 = time.perf_counter()
        fc_onnx = run_real_inference(
            pre_image_path=pre_path,
            post_image_path=post_path,
            confidence_threshold=0.30,
            max_features=100,
            backend_name="real_damage_model",
        )
        t_onnx = (time.perf_counter() - t0) * 1000
        validate_feature_collection(fc_onnx)
        n_onnx = len(fc_onnx.get("features", []))
        assert n_onnx > 0
        checks.append((
            "ONNX Neural Damage Model",
            True,
            f"Detected {n_onnx} building features ({t_onnx:.1f} ms on CPU)",
        ))
    except Exception as exc:
        checks.append(("ONNX Neural Damage Model", False, str(exc)))
        print(f"[FAIL] ONNX inference failed: {exc}")
        return 1

    # 6. Georeferencing Verification
    try:
        geo = AffineGeoreferencer(1024, 1024, (-85.550100, 30.152997, -85.547337, 30.155760))
        c_lon, c_lat = geo.pixel_to_lonlat(512, 512)
        assert -85.550100 <= c_lon <= -85.547337
        assert 30.152997 <= c_lat <= 30.155760
        # Check feature geometry in GeoJSON
        f0 = fc_onnx["features"][0]
        poly = f0["geometry"]["coordinates"][0]
        assert len(poly) >= 4
        assert poly[0] == poly[-1]  # Closed polygon
        checks.append((
            "Geospatial Georeferencing",
            True,
            f"WGS84 EPSG:4326 center: ({c_lat:.5f}, {c_lon:.5f})",
        ))
    except Exception as exc:
        checks.append(("Geospatial Georeferencing", False, str(exc)))
        print(f"[FAIL] Georeferencing check failed: {exc}")
        return 1

    # 7. Priority-Zone Scoring Engine
    try:
        zones = compute_priority_zones(fc_onnx, grid_rows=3, grid_cols=3)
        assert len(zones) == 9
        top_zone = zones[0]
        assert top_zone["priority_rank"] == 1
        assert 0.0 <= top_zone["priority_score"] <= 1.0
        assert top_zone["priority_score"] >= zones[-1]["priority_score"]

        # Validate GeoJSON export
        pz_geojson = priority_zones_to_geojson(zones)
        validate_feature_collection(pz_geojson)

        checks.append((
            "Priority-Zone Scoring Engine",
            True,
            f"Rank #1: {top_zone['zone_id']} (Score: {top_zone['priority_score']:.4f})",
        ))
    except Exception as exc:
        checks.append(("Priority-Zone Scoring Engine", False, str(exc)))
        print(f"[FAIL] Priority-zone scoring failed: {exc}")
        return 1

    # 8. Output Serialization
    try:
        out_geo = PROJECT_ROOT / "outputs" / "geojson" / "stage2_real_damage.geojson"
        save_geojson(fc_onnx, out_geo)
        out_pz = PROJECT_ROOT / "outputs" / "reports" / "stage2_priority_zones.json"
        out_pz.write_text(json.dumps(zones, indent=2), encoding="utf-8")
        assert out_geo.exists() and out_geo.stat().st_size > 0
        assert out_pz.exists() and out_pz.stat().st_size > 0
        checks.append((
            "Stage 2 Output Serialization",
            True,
            f"Saved {out_geo.name} and {out_pz.name}",
        ))
    except Exception as exc:
        checks.append(("Stage 2 Output Serialization", False, str(exc)))
        print(f"[FAIL] Output serialization failed: {exc}")
        return 1

    # Print Summary Report
    print("\n--- Stage 2 Verification Summary ---")
    all_passed = True
    for name, status, details in checks:
        mark = "[PASS]" if status else "[FAIL]"
        if not status:
            all_passed = False
        print(f"{mark:<7} {name:<30} : {details}")

    print("=" * 65)
    if all_passed:
        print("STAGE 2 REAL-MODEL PIPELINE VERIFICATION SUCCESSFUL (Exit Code 0)")
        print("=" * 65)
        return 0
    else:
        print("STAGE 2 REAL-MODEL PIPELINE VERIFICATION FAILED (Exit Code 1)")
        print("=" * 65)
        return 1


if __name__ == "__main__":
    sys.exit(main())
