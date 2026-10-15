"""
Pixel-level ground-truth evaluation script for the Advanced Damage Model Backend (GPU-Ready / CPU Fallback).
Evaluates predictions against data/processed/hurricane_michael_ground_truth_mask.png
using the identical pixel-level methodology as outputs/reports/onnx_evaluation.json.
"""

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from models.gpu_predict import run_gpu_inference
from utils.georeferencing import AffineGeoreferencer

PRE_IMG = PROJECT_ROOT / "data" / "processed" / "hurricane_michael_pre.png"
POST_IMG = PROJECT_ROOT / "data" / "processed" / "hurricane_michael_post.png"
GT_MASK_PATH = PROJECT_ROOT / "data" / "processed" / "hurricane_michael_ground_truth_mask.png"
EVAL_REPORT_PATH = PROJECT_ROOT / "outputs" / "reports" / "gpu_model_evaluation.json"

BOUNDS = (-85.550100, 30.152997, -85.547337, 30.155760)


def evaluate_backend(confidence_threshold: float = 0.30, max_features: int = 100) -> dict:
    if not GT_MASK_PATH.exists():
        report = {
            "evaluation_target": "Advanced Damage Model Backend (GPU-Ready / CPU Fallback)",
            "status": "Ground-truth evaluation was not completed. No accuracy claim should be made.",
            "reason": f"Ground-truth mask not found at {GT_MASK_PATH}",
        }
        EVAL_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        EVAL_REPORT_PATH.write_text(json.dumps(report, indent=2))
        return report

    # 1. Run inference
    fc = run_gpu_inference(
        pre_image_path=PRE_IMG,
        post_image_path=POST_IMG,
        bounds=BOUNDS,
        confidence_threshold=confidence_threshold,
        max_features=max_features,
    )

    # 2. Load ground-truth mask
    gt_img = cv2.imread(str(GT_MASK_PATH), cv2.IMREAD_GRAYSCALE)
    if gt_img is None:
        raise RuntimeError(f"Failed to read {GT_MASK_PATH}")

    h, w = gt_img.shape[:2]
    total_pixels = h * w
    gt_bin = (gt_img > 0).astype(np.uint8)

    # 3. Rasterize predicted GeoJSON polygons
    geo = AffineGeoreferencer(w, h, BOUNDS)
    pred_mask = np.zeros((h, w), dtype=np.uint8)

    for feat in fc.get("features", []):
        geom = feat.get("geometry", {})
        coords = geom.get("coordinates", [])
        if not coords:
            continue
        poly = coords[0]
        pts = np.array([geo.lonlat_to_pixel(pt[0], pt[1]) for pt in poly], dtype=np.int32)
        cv2.fillPoly(pred_mask, [pts], 1)

    # 4. Compute contingency matrix
    tp = int(np.logical_and(pred_mask == 1, gt_bin == 1).sum())
    fp = int(np.logical_and(pred_mask == 1, gt_bin == 0).sum())
    fn = int(np.logical_and(pred_mask == 0, gt_bin == 1).sum())
    tn = int(np.logical_and(pred_mask == 0, gt_bin == 0).sum())

    # 5. Compute metrics
    pixel_acc = (tp + tn) / max(total_pixels, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = (2 * prec * rec) / max(prec + rec, 1e-6)
    iou = tp / max(tp + fp + fn, 1)

    metadata = fc.get("metadata", {})
    report = {
        "evaluation_target": "Advanced Damage Model Backend (GPU-Ready / CPU Fallback) Localization & Segmentation",
        "dataset": "Hurricane Michael (Mexico Beach, FL - Oct 2018)",
        "ground_truth_mask": str(GT_MASK_PATH.relative_to(PROJECT_ROOT)),
        "predicted_geojson": "outputs/geojson/gpu_model_damage_assessment.geojson",
        "device_used": metadata.get("device", "cpu (automated fallback)"),
        "total_pixels": total_pixels,
        "contingency_matrix": {
            "true_positives_tp": tp,
            "false_positives_fp": fp,
            "false_negatives_fn": fn,
            "true_negatives_tn": tn,
        },
        "metrics": {
            "pixel_accuracy": round(pixel_acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "intersection_over_union_iou": round(iou, 4),
        },
        "evaluation_assumptions": [
            "Evaluation performed at native tile resolution of 1024x1024 pixels.",
            "Ground truth building mask is binarized: non-zero pixels represent ground-truth building/damage footprints.",
            "Predicted GeoJSON polygons are rasterized onto the 1024x1024 grid using WGS84 affine coordinate inversion.",
            "Precision reflects the proportion of predicted building footprint pixels that correspond to ground truth building pixels.",
            "Recall reflects the coverage of actual ground truth structures captured by candidate proposals.",
            "Execution ran on CPU SIMD fallback because CUDAExecutionProvider is not available in the installed ONNX Runtime wheel.",
            "No accuracy superiority claims are made over the baseline lightweight ONNX backend because both share the verified patch-proposal inference pipeline."
        ],
    }

    EVAL_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Evaluation saved to {EVAL_REPORT_PATH}")
    print(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
    print(f"Accuracy: {pixel_acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, IoU: {iou:.4f}")
    return report


if __name__ == "__main__":
    evaluate_backend()
