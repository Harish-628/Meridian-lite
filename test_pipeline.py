"""
MERIDIAN-Lite: Standalone End-to-End Pipeline Verification Script.
Executes and validates all Stage 1 pipeline components:
1. Module imports
2. Demo asset generation
3. Simulated AI inference
4. GeoJSON compliance validation
5. Summary metrics computation
6. Overlay image generation
7. Tactical output serialization

Exits with code 0 on complete success, or non-zero on failure.
"""

import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    print("=" * 60)
    print("MERIDIAN-Lite Stage 1 Pipeline Verification")
    print("=" * 60)

    checks = []

    # 1. Module Imports
    try:
        from utils.paths import (
            PROJECT_ROOT,
            DATA_DIR,
            DEMO_DATA_DIR,
            OUTPUTS_DIR,
            DEFAULT_LOG_FILE,
            DEFAULT_GEOJSON_OUTPUT,
            DEFAULT_METRICS_JSON_OUTPUT,
            DEFAULT_METRICS_CSV_OUTPUT,
            DEFAULT_OVERLAY_OUTPUT,
        )
        from utils.logging_utils import get_logger, setup_logging
        from utils.image_utils import (
            create_demo_images,
            validate_image,
            load_image,
            create_damage_overlay,
        )
        from utils.geojson_utils import (
            validate_feature_collection,
            is_valid_feature_collection,
            save_geojson,
            load_geojson,
            get_bounds,
            damage_color,
        )
        from utils.metrics import calculate_summary_metrics, features_to_dataframe
        from models.predict import run_inference

        logger = setup_logging()
        logger.info("Pipeline verification initialized.")
        checks.append(("Module Imports", True, "All modules imported cleanly"))
    except Exception as exc:
        print(f"[FAIL] Module imports failed: {exc}")
        return 1

    # 2. Demo Data Generation
    try:
        pre_img, post_img = create_demo_images(DEMO_DATA_DIR)
        assert pre_img.exists() and pre_img.stat().st_size > 0
        assert post_img.exists() and post_img.stat().st_size > 0
        validate_image(pre_img)
        validate_image(post_img)
        checks.append(
            ("Demo Asset Generation", True, f"Generated {pre_img.name} and {post_img.name}")
        )
    except Exception as exc:
        checks.append(("Demo Asset Generation", False, str(exc)))
        print(f"[FAIL] Demo asset generation failed: {exc}")
        return 1

    # 3. Mock Inference Execution
    try:
        geojson_result = run_inference(
            pre_image_path=pre_img,
            post_image_path=post_img,
            center_lat=13.0827,
            center_lon=80.2707,
            seed=42,
            building_count=18,
        )
        feature_count = len(geojson_result.get("features", []))
        assert feature_count == 18
        checks.append(
            ("Mock Inference Engine", True, f"Generated {feature_count} building features")
        )
    except Exception as exc:
        checks.append(("Mock Inference Engine", False, str(exc)))
        print(f"[FAIL] Mock inference failed: {exc}")
        return 1

    # 4. GeoJSON Validation
    try:
        validate_feature_collection(geojson_result)
        assert is_valid_feature_collection(geojson_result) is True
        bounds = get_bounds(geojson_result)
        assert bounds != [[0.0, 0.0], [0.0, 0.0]]
        saved_geo = save_geojson(geojson_result, DEFAULT_GEOJSON_OUTPUT)
        reloaded = load_geojson(saved_geo)
        assert len(reloaded["features"]) == 18
        checks.append(
            ("GeoJSON RFC 7946 Validation", True, f"Valid polygons, saved to {saved_geo.name}")
        )
    except Exception as exc:
        checks.append(("GeoJSON RFC 7946 Validation", False, str(exc)))
        print(f"[FAIL] GeoJSON validation failed: {exc}")
        return 1

    # 5. Metrics Calculation & Dataframe Export
    try:
        metrics = calculate_summary_metrics(geojson_result)
        assert metrics["total_buildings"] == 18
        assert metrics["damaged_count"] >= 0
        assert 0.0 <= metrics["damaged_percentage"] <= 100.0
        assert 0.75 <= metrics["average_confidence"] <= 0.99

        # Test zero-division safety
        zero_metrics = calculate_summary_metrics({"type": "FeatureCollection", "features": []})
        assert zero_metrics["total_buildings"] == 0
        assert zero_metrics["damaged_percentage"] == 0.0

        # Save metrics JSON & CSV
        DEFAULT_METRICS_JSON_OUTPUT.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        df = features_to_dataframe(geojson_result)
        df.to_csv(DEFAULT_METRICS_CSV_OUTPUT, index=False)
        checks.append(
            ("Metrics & Serialization", True, f"Damaged: {metrics['damaged_percentage']}% (Total: 18)")
        )
    except Exception as exc:
        checks.append(("Metrics & Serialization", False, str(exc)))
        print(f"[FAIL] Metrics calculation failed: {exc}")
        return 1

    # 6. Visual Difference Overlay Generation
    try:
        overlay_arr, overlay_p = create_damage_overlay(
            pre_image_path=pre_img,
            post_image_path=post_img,
            output_path=DEFAULT_OVERLAY_OUTPUT,
        )
        assert overlay_p.exists() and overlay_p.stat().st_size > 0
        assert overlay_arr.shape == (800, 800, 3)
        checks.append(
            ("Damage Overlay Generation", True, f"Overlay saved to {overlay_p.name}")
        )
    except Exception as exc:
        checks.append(("Damage Overlay Generation", False, str(exc)))
        print(f"[FAIL] Overlay generation failed: {exc}")
        return 1

    # Summary Report
    print("\n--- Pipeline Verification Summary ---")
    all_passed = True
    for name, status, details in checks:
        mark = "[PASS]" if status else "[FAIL]"
        if not status:
            all_passed = False
        print(f"{mark:<7} {name:<30} : {details}")

    print("=" * 60)
    if all_passed:
        print("STAGE 1 PIPELINE VERIFICATION SUCCESSFUL (Exit Code 0)")
        print("=" * 60)
        return 0
    else:
        print("STAGE 1 PIPELINE VERIFICATION FAILED (Exit Code 1)")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
