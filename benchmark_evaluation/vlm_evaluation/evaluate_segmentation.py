#!/usr/bin/env python3
"""
Evaluate VLM segmentation mask results against data_sample/images ground truth masks.

Data layout:
  - Ground truth masks:
      defect_bench/data_sample/images/{class_dir}/{image_stem}_mask.png
  - Model predictions:
      defect_bench/data_sample/images/{model_name}_masks/{class_dir}/{image_stem}_mask.png

model_name is taken from:
    model_name = os.environ.get("VLM_MODEL_NAME", DEFAULT_MODEL_NAME)
where DEFAULT_MODEL_NAME is imported from vlm_generate_qa.py

Masks are colored images with RGB colors:
  - Crack: (255, 0, 0) 红
  - Material_loss: (255, 140, 0) 橙
  - Stain: (30, 144, 255) 蓝
  - External Fixings: (0, 200, 0) 绿

Metrics per class:
  - mIoU: mean Intersection over Union
  - Precision: TP / (TP + FP)
  - Recall: TP / (TP + FN)
  - F1-score: 2 * Precision * Recall / (Precision + Recall)
  - PA: Pixel Accuracy (correct pixels / total pixels)
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

TEST100_DIR = Path("defect_bench/data_sample")
IMAGES_DIR = TEST100_DIR / "images"
CLASS_DIRS = ["images"]
DEFAULT_MODEL_NAME = "doubao-seedream-4-5-251128"
RESULTS_DIR = Path("defect_bench/results")


# Color mapping: primary_class -> RGB color (same as unify_mask.py)
CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
    "Crack": (255, 0, 0),             # Red
    "Material_loss": (255, 140, 0),   # Orange
    "Stain": (30, 144, 255),          # Blue
    "External Fixings": (0, 200, 0),  # Green
}

PRIMARY_CLASSES = list(CLASS_COLORS.keys())


def extract_class_mask_from_colored_mask(colored_mask: np.ndarray, class_color: Tuple[int, int, int]) -> np.ndarray:
    """
    Extract binary mask for a specific class from colored mask.
    
    Args:
        colored_mask: (H, W, 3) RGB image
        class_color: (R, G, B) tuple
    
    Returns:
        Binary mask (H, W) as uint8 array (0 or 255)
    """
    if len(colored_mask.shape) != 3 or colored_mask.shape[2] != 3:
        return np.zeros((colored_mask.shape[0], colored_mask.shape[1]), dtype=np.uint8)
    
    # Create mask where pixels match the class color (with tolerance for slight variations)
    r, g, b = class_color
    # Use tolerance of 5 for each channel to handle slight color variations
    mask = (
        (np.abs(colored_mask[:, :, 0].astype(np.int16) - r) <= 5) &
        (np.abs(colored_mask[:, :, 1].astype(np.int16) - g) <= 5) &
        (np.abs(colored_mask[:, :, 2].astype(np.int16) - b) <= 5)
    )
    
    return (mask * 255).astype(np.uint8)


def compute_segmentation_metrics(gt_mask: np.ndarray, pred_mask: np.ndarray) -> Tuple[float, float, float, float, float]:
    """
    Compute segmentation metrics: mIoU, Precision, Recall, F1-score, PA.
    
    This uses pixel-level (per-pixel) evaluation, which is standard for segmentation tasks.
    Each pixel is treated as an independent sample.
    
    Definitions:
    - TP (True Positive): pixels predicted as positive and actually positive
    - FP (False Positive): pixels predicted as positive but actually negative
    - FN (False Negative): pixels actually positive but predicted as negative
    - TN (True Negative): pixels predicted as negative and actually negative
    
    Metrics:
    - Precision = TP / (TP + FP): Among all predicted positive pixels, how many are correct?
    - Recall = TP / (TP + FN): Among all actual positive pixels, how many are correctly predicted?
    - IoU = TP / (TP + FP + FN): Intersection over Union
    - F1 = 2 * Precision * Recall / (Precision + Recall): Harmonic mean of Precision and Recall
    - PA = (TP + TN) / Total: Pixel Accuracy (all correct pixels / total pixels)
    
    Edge cases:
    - When no positive predictions (tp+fp=0): precision=1.0 (no false positives)
    - When no positive GT (tp+fn=0): recall=1.0 (no false negatives)
    
    Args:
        gt_mask: Ground truth binary mask (H, W) as uint8 (0 or 255)
        pred_mask: Predicted binary mask (H, W) as uint8 (0 or 255)
    
    Returns:
        (iou, precision, recall, f1, pa)
    """
    # Convert to boolean masks
    gt_bool = (gt_mask > 127).astype(np.uint8)
    pred_bool = (pred_mask > 127).astype(np.uint8)
    
    # Compute intersection and union
    intersection = np.logical_and(gt_bool, pred_bool).sum()
    union = np.logical_or(gt_bool, pred_bool).sum()
    
    # IoU
    if union == 0:
        iou = 1.0 if intersection == 0 else 0.0
    else:
        iou = float(intersection) / float(union)
    
    # TP, FP, FN, TN (pixel-level)
    tp = intersection
    fp = (pred_bool & (~gt_bool)).sum()
    fn = (gt_bool & (~pred_bool)).sum()
    tn = ((~gt_bool) & (~pred_bool)).sum()
    
    # Precision: TP / (TP + FP)
    # When no positive predictions (tp + fp == 0): precision is undefined.
    # Convention: set to 1.0 (no false positives, prediction is correct for this class)
    if tp + fp == 0:
        precision = 1.0
    else:
        precision = float(tp) / float(tp + fp)
    
    # Recall: TP / (TP + FN)
    # When no positive GT (tp + fn == 0): recall is undefined.
    # Convention: set to 1.0 (no false negatives, prediction is correct for this class)
    if tp + fn == 0:
        recall = 1.0
    else:
        recall = float(tp) / float(tp + fn)
    
    # F1-score
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    
    # Pixel Accuracy
    total_pixels = gt_bool.size
    correct_pixels = (gt_bool == pred_bool).sum()
    pa = float(correct_pixels) / float(total_pixels) if total_pixels > 0 else 0.0
    
    return iou, precision, recall, f1, pa


def main():
    model_name = os.environ.get("VLM_MODEL_NAME", DEFAULT_MODEL_NAME)
    pred_mask_root = RESULTS_DIR / f"{model_name}_masks"
    gt_mask_root = TEST100_DIR / "masks"
    
    print("=" * 80)
    print("Evaluating VLM segmentation mask results")
    print(f"Model name: {model_name}")
    print(f"Prediction mask root: {pred_mask_root}")
    print(f"GT mask root: {gt_mask_root}")
    print("=" * 80)
    
    # Aggregators per class: {class: {"iou": List[float], "precision": List[float], ...}}
    per_class_metrics: Dict[str, Dict[str, List[float]]] = {
        cls: {
            "iou": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "pa": [],
        }
        for cls in PRIMARY_CLASSES
    }
    
    n_images_processed = 0
    n_images_with_errors = 0
    
    for class_dir_name in CLASS_DIRS:
        gt_class_dir = gt_mask_root / class_dir_name
        pred_class_dir = pred_mask_root / class_dir_name
        
        if not gt_class_dir.exists():
            print(f"Warning: GT directory {gt_class_dir} does not exist, skipping...")
            continue
        
        if not pred_class_dir.exists():
            print(f"Warning: Prediction directory {pred_class_dir} does not exist, skipping...")
            continue
        
        print(f"\nProcessing {class_dir_name}...")
        
        # Find all GT mask files
        gt_mask_files = list(gt_class_dir.glob("*_mask.png"))
        
        for gt_mask_path in gt_mask_files:
            image_stem = gt_mask_path.name.replace("_mask.png", "")
            pred_mask_path = pred_class_dir / f"{image_stem}_mask.png"
            
            if not pred_mask_path.exists():
                print(f"  Warning: Prediction mask not found for {image_stem}, skipping...")
                n_images_with_errors += 1
                continue
            
            try:
                # Read masks
                gt_mask_colored = cv2.imread(str(gt_mask_path))
                pred_mask_colored = cv2.imread(str(pred_mask_path))
                
                if gt_mask_colored is None:
                    print(f"  Warning: Failed to read GT mask {gt_mask_path}, skipping...")
                    n_images_with_errors += 1
                    continue
                
                if pred_mask_colored is None:
                    print(f"  Warning: Failed to read prediction mask {pred_mask_path}, skipping...")
                    n_images_with_errors += 1
                    continue
                
                # Convert BGR to RGB (OpenCV reads as BGR)
                gt_mask_colored = cv2.cvtColor(gt_mask_colored, cv2.COLOR_BGR2RGB)
                pred_mask_colored = cv2.cvtColor(pred_mask_colored, cv2.COLOR_BGR2RGB)
                
                # Ensure same size
                if gt_mask_colored.shape[:2] != pred_mask_colored.shape[:2]:
                    print(f"  Warning: Size mismatch for {image_stem}, resizing prediction to match GT...")
                    pred_mask_colored = cv2.resize(
                        pred_mask_colored,
                        (gt_mask_colored.shape[1], gt_mask_colored.shape[0]),
                        interpolation=cv2.INTER_NEAREST
                    )
                
                # Extract masks for each class and compute metrics
                for cls in PRIMARY_CLASSES:
                    class_color = CLASS_COLORS[cls]
                    
                    # Extract binary masks
                    gt_binary = extract_class_mask_from_colored_mask(gt_mask_colored, class_color)
                    pred_binary = extract_class_mask_from_colored_mask(pred_mask_colored, class_color)
                    
                    # Compute metrics
                    iou, precision, recall, f1, pa = compute_segmentation_metrics(gt_binary, pred_binary)
                    
                    # Store metrics
                    per_class_metrics[cls]["iou"].append(iou)
                    per_class_metrics[cls]["precision"].append(precision)
                    per_class_metrics[cls]["recall"].append(recall)
                    per_class_metrics[cls]["f1"].append(f1)
                    per_class_metrics[cls]["pa"].append(pa)
                
                n_images_processed += 1
                
            except Exception as e:
                print(f"  Error processing {image_stem}: {e}")
                import traceback
                traceback.print_exc()
                n_images_with_errors += 1
                continue
    
    # Compute average metrics per class
    per_class_stats: Dict[str, Dict[str, float]] = {}
    for cls in PRIMARY_CLASSES:
        metrics = per_class_metrics[cls]
        if not metrics["iou"]:
            # No data for this class
            per_class_stats[cls] = {
                "mIoU": 0.0,
                "Precision": 0.0,
                "Recall": 0.0,
                "F1": 0.0,
                "PA": 0.0,
            }
        else:
            per_class_stats[cls] = {
                "mIoU": float(np.mean(metrics["iou"])),
                "Precision": float(np.mean(metrics["precision"])),
                "Recall": float(np.mean(metrics["recall"])),
                "F1": float(np.mean(metrics["f1"])),
                "PA": float(np.mean(metrics["pa"])),
            }
    
    # Compute global averages (macro average over classes)
    global_miou = float(np.mean([stats["mIoU"] for stats in per_class_stats.values()]))
    global_precision = float(np.mean([stats["Precision"] for stats in per_class_stats.values()]))
    global_recall = float(np.mean([stats["Recall"] for stats in per_class_stats.values()]))
    global_f1 = float(np.mean([stats["F1"] for stats in per_class_stats.values()]))
    global_pa = float(np.mean([stats["PA"] for stats in per_class_stats.values()]))
    
    # Print results
    print("\n" + "=" * 80)
    print("=== Evaluation Results ===")
    print(f"Images processed: {n_images_processed}")
    print(f"Images with errors: {n_images_with_errors}")
    print("\nGlobal metrics (macro average over classes):")
    print(f"  mIoU      : {global_miou:.4f}")
    print(f"  Precision : {global_precision:.4f}")
    print(f"  Recall    : {global_recall:.4f}")
    print(f"  F1-score  : {global_f1:.4f}")
    print(f"  PA        : {global_pa:.4f}")
    
    print("\nPer-class metrics:")
    print("  {:18s} {:>10s} {:>10s} {:>10s} {:>10s} {:>10s}".format(
        "Class", "mIoU", "Precision", "Recall", "F1-score", "PA"
    ))
    for cls in PRIMARY_CLASSES:
        stats = per_class_stats[cls]
        print(
            "  {:18s} {:10.4f} {:10.4f} {:10.4f} {:10.4f} {:10.4f}".format(
                cls,
                stats["mIoU"],
                stats["Precision"],
                stats["Recall"],
                stats["F1"],
                stats["PA"],
            )
        )
    
    # Export to CSV
    csv_path = pred_mask_root / f"{model_name}_segmentation_metrics.csv"
    try:
        with csv_path.open("w", encoding="utf-8") as f:
            # Global metrics
            f.write("# Segmentation metrics (global - macro average over classes)\n")
            f.write("metric,value\n")
            f.write(f"mIoU,{global_miou:.6f}\n")
            f.write(f"Precision,{global_precision:.6f}\n")
            f.write(f"Recall,{global_recall:.6f}\n")
            f.write(f"F1-score,{global_f1:.6f}\n")
            f.write(f"PA,{global_pa:.6f}\n\n")
            
            # Per-class metrics
            f.write("# Segmentation metrics (per-class)\n")
            f.write("class,mIoU,Precision,Recall,F1-score,PA\n")
            for cls in PRIMARY_CLASSES:
                stats = per_class_stats[cls]
                f.write(
                    f"{cls},"
                    f"{stats['mIoU']:.6f},"
                    f"{stats['Precision']:.6f},"
                    f"{stats['Recall']:.6f},"
                    f"{stats['F1']:.6f},"
                    f"{stats['PA']:.6f}\n"
                )
            # Add average row
            f.write(
                f"Average,{global_miou:.6f},{global_precision:.6f},{global_recall:.6f},{global_f1:.6f},{global_pa:.6f}\n"
            )
        
        print(f"\nSegmentation metrics CSV saved to: {csv_path}")
    except Exception as e:
        print(f"\nWarning: failed to write evaluation CSV: {e}")
        import traceback
        traceback.print_exc()
    
    print("=" * 80)


if __name__ == "__main__":
    main()
