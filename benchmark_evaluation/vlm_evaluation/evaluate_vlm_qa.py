#!/usr/bin/env python3
"""
Evaluate VLM QA results against results Visualization ground truth.

Data layout (reusing vlm_generate_qa.py conventions):
  - Ground truth QA (Q1–Q3):
      defect_bench/results/Visualization/{class}/{stem}_qa.json
  - Ground truth topology QA (Q4):
      defect_bench/results/Visualization/{class}/{stem}_topology_qa.json
  - Model predictions (Q1–Q3):
      defect_bench/results/{model_name}/{class}/{stem}_qa.json
  - Model topology predictions (Q4, if available):
      defect_bench/results/{model_name}/{class}/{stem}_topology_qa.json

model_name is taken from:
    model_name = os.environ.get("VLM_MODEL_NAME", DEFAULT_MODEL_NAME)
where DEFAULT_MODEL_NAME is imported from vlm_generate_qa.py

Metrics:
  Q1: "What defects are in the image?"
      - Precision (micro over images)
      - Recall   (micro over images)
      - Hit Rate (per-image: at least one correct defect; dataset = average)

  Q2: "How many instances of each defect type (...) ?"
      - Mean Absolute Error (MAE) over (image, class) pairs where GT count > 0
      - Relative Error = |pred - gt| / gt, averaged over same pairs

  Q3: "What are the bounding box coordinates of each defect instance (x, y, width, height)?"
      - mAP50: mean AP over classes at IoU 0.5
      - mAP50-95: mean AP over IoU thresholds [0.5, 0.55, ..., 0.95] (COCO-style)
      - F1-score: macro average over classes at IoU 0.5 using a single operating point

  Q4: topology QA (e.g. "[1#Crack, adjacency, 2#Material_loss]")
      - Precision (micro over relation triplets)
      - Recall   (micro)
      - F1-score (micro)
"""

import os
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any

import numpy as np

TEST100_DIR = Path("defect_bench/data_sample")
IMAGES_DIR = TEST100_DIR / "images"
CLASS_DIRS = ["images"]
DEFAULT_MODEL_NAME = "doubao-seed-1-8-251228"
RESULTS_DIR = Path("defect_bench/results")


GT_VIS_DIR = RESULTS_DIR / "ground_truth" / "Visualization"


PRIMARY_CLASSES = ["Crack", "Material_loss", "Stain", "External Fixings"]


def normalize_class_name(name: str) -> Optional[str]:
    """Normalize a class name to one of the 4 primary classes."""
    if not name:
        return None
    s = name.strip().lower().replace("-", "_").replace(" ", "_")
    mapping = {
        "crack": "Crack",
        "cracks": "Crack",
        "material_loss": "Material_loss",
        "stain": "Stain",
        "stains": "Stain",
        "external_fixings": "External Fixings",
        "external_fixing": "External Fixings",
    }
    return mapping.get(s)


# ----------------------------------------------------------------------
# Parsing helpers for Q1 / Q2
# ----------------------------------------------------------------------

def parse_defect_list(answer: str) -> List[str]:
    """
    Parse a defect list answer like "Crack, Material_loss" into an ordered list
    of canonical primary class names (duplicates removed, order preserved).
    """
    if not answer:
        return []
    tokens = [t.strip() for t in answer.replace(";", ",").split(",")]
    seen: Set[str] = set()
    result: List[str] = []
    for t in tokens:
        if not t:
            continue
        cls = normalize_class_name(t)
        if not cls:
            continue
        if cls not in seen:
            seen.add(cls)
            result.append(cls)
    return result


def parse_counts_by_class(answer1: str, answer2: str) -> Dict[str, int]:
    """
    Given:
      answer1: defect types string (e.g. "Crack, Material_loss")
      answer2: counts string (e.g. "1, 2")
    return dict {primary_class: count}.

    If parsing fails (length mismatch etc.), returns empty dict.
    """
    classes = parse_defect_list(answer1)
    if not classes:
        return {}
    if not answer2:
        # No counts returned; treat as zeros elsewhere
        return {}

    parts = [p.strip() for p in answer2.replace(";", ",").split(",") if p.strip()]
    # 即使长度不一致，也按 zip 对齐使用前 min(len(classes), len(parts)) 个，
    # 这样 GT 永远可用，预测的异常也不会整张丢弃。
    counts: Dict[str, int] = {}
    for cls, p in zip(classes, parts):
        try:
            c = int(round(float(p)))
        except Exception:
            c = 0
        counts[cls] = c
    return counts


# ----------------------------------------------------------------------
# Parsing helpers for Q3 (bbox strings)
# ----------------------------------------------------------------------

def parse_bbox_answer(answer: str) -> Dict[str, List[Tuple[float, float, float, float]]]:
    """
    Parse bbox answer like:
        "Crack: [264.0, 346.0, 760.0, 588.0]; Material_loss: [3, 51, 1021, 298]"
        or with multiple boxes:
        "Crack: [x1,y1,w1,h1]; [x2,y2,w2,h2]; Material_loss: [..]"

    Returns:
        {primary_class: [(x, y, w, h), ...]}
    """
    result: Dict[str, List[Tuple[float, float, float, float]]] = {}
    if not answer:
        return result

    tokens = [t.strip() for t in answer.split(";") if t.strip()]
    current_class: Optional[str] = None

    for tok in tokens:
        cls_part = None
        box_part = tok
        if ":" in tok:
            cls_raw, box_part = tok.split(":", 1)
            cls_part = normalize_class_name(cls_raw)
            current_class = cls_part

        if not current_class:
            # No valid class yet, skip
            continue

        # Extract first bracketed segment
        m = re.search(r"\[([^\]]+)\]", box_part)
        if not m:
            continue
        nums = [n.strip() for n in m.group(1).split(",")]
        if len(nums) != 4:
            continue
        try:
            x, y, w, h = [float(v) for v in nums]
        except Exception:
            continue

        result.setdefault(current_class, []).append((x, y, w, h))

    return result


# ----------------------------------------------------------------------
# IoU / detection metrics for Q3
# ----------------------------------------------------------------------

def bbox_iou_xywh(b1: Tuple[float, float, float, float],
                  b2: Tuple[float, float, float, float]) -> float:
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    x1_min, y1_min, x1_max, y1_max = x1, y1, x1 + w1, y1 + h1
    x2_min, y2_min, x2_max, y2_max = x2, y2, x2 + w2, y2 + h2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_w = max(0.0, inter_x_max - inter_x_min)
    inter_h = max(0.0, inter_y_max - inter_y_min)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area1 = max(0.0, w1) * max(0.0, h1)
    area2 = max(0.0, w2) * max(0.0, h2)
    union = area1 + area2 - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """
    VOC-style 11-point interpolated AP.
    recalls, precisions: cumulative arrays sorted by prediction rank.
    """
    if recalls.size == 0:
        return 0.0

    # Append boundary points
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    # Make precision non-increasing
    for i in range(mpre.size - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    ap = 0.0
    for r in np.linspace(0, 1, 11):
        indices = np.where(mrec >= r)[0]
        p = np.max(mpre[indices]) if indices.size > 0 else 0.0
        ap += p
    return ap / 11.0


def compute_detection_metrics(
    all_gt: Dict[str, List[Tuple[str, Tuple[float, float, float, float]]]],
    all_pred: Dict[str, List[Tuple[str, Tuple[float, float, float, float]]]],
) -> Tuple[float, float, float, float, float, Dict[str, Dict[str, float]]]:
    """
    Compute (mAP50, mAP50_95, F1_at_50, Precision_at_50, Recall_at_50) across all primary classes.

    all_gt / all_pred:
        {class_name: [(image_id, (x,y,w,h)), ...]}
    """
    iou_thresholds = np.arange(0.5, 0.96, 0.05)
    ap_per_class_per_thr: Dict[str, Dict[float, float]] = {}
    f1_per_class: Dict[str, float] = {}
    precision_per_class: Dict[str, float] = {}
    recall_per_class: Dict[str, float] = {}
    per_class_stats: Dict[str, Dict[str, float]] = {}
    
    # Global aggregators for Precision and Recall
    global_tp_50 = 0
    global_fp_50 = 0
    global_fn_50 = 0

    for cls in PRIMARY_CLASSES:
        gt_list = all_gt.get(cls, [])
        pred_list = all_pred.get(cls, [])
        if not gt_list and not pred_list:
            continue

        # Group GT by image for matching
        gt_by_img: Dict[str, List[Tuple[float, float, float, float]]] = {}
        for img_id, box in gt_list:
            gt_by_img.setdefault(img_id, []).append(box)

        n_gt = len(gt_list)
        if n_gt == 0:
            # No GT for this class: AP is undefined; we skip it in mAP.
            continue

        # For F1 we take IoU=0.5 only; reuse below
        tp_final_50 = 0
        fp_final_50 = 0

        ap_per_thr: Dict[float, float] = {}

        for thr in iou_thresholds:
            # Prepare per-GT matched flags per image
            gt_matched: Dict[str, List[bool]] = {
                img_id: [False] * len(boxes) for img_id, boxes in gt_by_img.items()
            }

            tps: List[int] = []
            fps: List[int] = []

            # Predictions are treated in given order (no scores)
            for img_id, pbox in pred_list:
                gts = gt_by_img.get(img_id, [])
                best_iou = 0.0
                best_idx = -1
                for i, gbox in enumerate(gts):
                    if gt_matched[img_id][i]:
                        continue
                    iou = bbox_iou_xywh(pbox, gbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = i

                if best_iou >= thr and best_idx >= 0:
                    gt_matched[img_id][best_idx] = True
                    tps.append(1)
                    fps.append(0)
                else:
                    tps.append(0)
                    fps.append(1)

            if not tps:
                # No predictions -> AP = 0 for this class at this IoU
                ap_per_thr[thr] = 0.0
                if thr == 0.5:
                    tp_final_50 = 0
                    fp_final_50 = 0
                continue

            tp_cum = np.cumsum(tps)
            fp_cum = np.cumsum(fps)
            recalls = tp_cum / float(n_gt)
            precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-8)

            ap = compute_ap(recalls, precisions)
            ap_per_thr[thr] = ap

            if thr == 0.5:
                tp_final_50 = int(tp_cum[-1])
                fp_final_50 = int(fp_cum[-1])

        # F1, Precision, Recall at IoU=0.5
        fn_final_50 = n_gt - tp_final_50
        precision_50 = tp_final_50 / float(tp_final_50 + fp_final_50) if (tp_final_50 + fp_final_50) > 0 else 0.0
        recall_50 = tp_final_50 / float(n_gt) if n_gt > 0 else 0.0
        if precision_50 + recall_50 > 0:
            f1 = 2.0 * precision_50 * recall_50 / (precision_50 + recall_50)
        else:
            f1 = 0.0
        f1_per_class[cls] = f1
        precision_per_class[cls] = precision_50
        recall_per_class[cls] = recall_50
        
        # Accumulate global TP, FP, FN
        global_tp_50 += tp_final_50
        global_fp_50 += fp_final_50
        global_fn_50 += fn_final_50

        # Per-class AP stats
        if ap_per_thr:
            ap50_cls = ap_per_thr.get(0.5, 0.0)
            mean_ap_cls = float(np.mean(list(ap_per_thr.values())))
        else:
            ap50_cls = 0.0
            mean_ap_cls = 0.0

        ap_per_class_per_thr[cls] = ap_per_thr
        per_class_stats[cls] = {
            "AP50": ap50_cls,
            "AP50_95": mean_ap_cls,
            "F1": f1,
            "Precision": precision_50,
            "Recall": recall_50,
        }

    # Aggregate mAP50 and mAP50-95
    mAP50_list: List[float] = []
    mAP5095_list: List[float] = []
    for cls, ap_dict in ap_per_class_per_thr.items():
        if not ap_dict:
            continue
        ap50 = ap_dict.get(0.5, 0.0)
        mAP50_list.append(ap50)
        mAP5095_list.append(float(np.mean(list(ap_dict.values()))))

    mAP50 = float(np.mean(mAP50_list)) if mAP50_list else 0.0
    mAP5095 = float(np.mean(mAP5095_list)) if mAP5095_list else 0.0

    f1_vals = list(f1_per_class.values())
    f1_macro = float(np.mean(f1_vals)) if f1_vals else 0.0
    
    # Global Precision and Recall at IoU=0.5
    global_precision = global_tp_50 / float(global_tp_50 + global_fp_50) if (global_tp_50 + global_fp_50) > 0 else 0.0
    global_recall = global_tp_50 / float(global_tp_50 + global_fn_50) if (global_tp_50 + global_fn_50) > 0 else 0.0

    return mAP50, mAP5095, f1_macro, global_precision, global_recall, per_class_stats


# ----------------------------------------------------------------------
# Parsing helpers for Q4 (topology relations)
# ----------------------------------------------------------------------

RELATION_PATTERN = re.compile(r"\[([^\]]+)\]")


def parse_topology_relations(answer: str) -> Set[Tuple[str, str, str]]:
    """
    Parse topology answer like:
        "[1#Crack, adjacency, 2#Material_loss]"
        or multiple:
        "[1#Crack, adjacency, 2#Material_loss]; [2#Stain, above, 3#Crack]"

    We turn each relation into a canonical (cls1, relation, cls2) triple,
    ignoring instance ids (the "1#" prefix).
    """
    relations: Set[Tuple[str, str, str]] = set()
    if not answer:
        return relations

    # Find all [...] groups
    matches = RELATION_PATTERN.findall(answer)
    for m in matches:
        parts = [p.strip() for p in m.split(",")]
        if len(parts) != 3:
            continue
        subj_raw, rel_raw, obj_raw = parts

        def extract_cls(token: str) -> Optional[str]:
            # Token may look like "1#Crack" or "Crack"
            tok = token.strip()
            if "#" in tok:
                tok = tok.split("#", 1)[1]
            return normalize_class_name(tok)

        cls1 = extract_cls(subj_raw)
        cls2 = extract_cls(obj_raw)
        relation = rel_raw.strip().lower()
        if not cls1 or not cls2 or not relation:
            continue
        relations.add((cls1, relation, cls2))
    return relations


# ----------------------------------------------------------------------
# Main evaluation
# ----------------------------------------------------------------------

def main():
    model_name = os.environ.get("VLM_MODEL_NAME", DEFAULT_MODEL_NAME)
    pred_root = RESULTS_DIR / model_name

    print("=" * 80)
    print("Evaluating VLM QA results")
    print(f"Model name: {model_name}")
    print(f"Prediction root: {pred_root}")
    print(f"GT root (Visualization): {GT_VIS_DIR}")
    print("=" * 80)

    # --- Aggregators for Q1 ---
    q1_tp = 0
    q1_pred_pos = 0
    q1_gt_pos = 0
    q1_hit_list: List[int] = []
    # Per-class Q1 metrics: {class: {"tp": int, "fp": int, "fn": int}}
    q1_per_class: Dict[str, Dict[str, int]] = {c: {"tp": 0, "fp": 0, "fn": 0} for c in PRIMARY_CLASSES}

    # --- Aggregators for Q2 ---
    q2_abs_errors: List[float] = []
    q2_rel_errors: List[float] = []
    # Per-class Q2 metrics: {class: {"abs_errors": List[float], "rel_errors": List[float]}}
    q2_per_class: Dict[str, Dict[str, List[float]]] = {c: {"abs_errors": [], "rel_errors": []} for c in PRIMARY_CLASSES}

    # --- Aggregators for Q3 ---
    # all_gt / all_pred: {class: [(image_id, (x,y,w,h)), ...]}
    det_gt: Dict[str, List[Tuple[str, Tuple[float, float, float, float]]]] = {c: [] for c in PRIMARY_CLASSES}
    det_pred: Dict[str, List[Tuple[str, Tuple[float, float, float, float]]]] = {c: [] for c in PRIMARY_CLASSES}

    # --- Aggregators for Q4 ---
    q4_tp = 0
    q4_pred_pos = 0
    q4_gt_pos = 0
    # Per-class Q4 metrics: {class: {"tp": int, "fp": int, "fn": int}}
    # For Q4, we count relations that involve each class
    q4_per_class: Dict[str, Dict[str, int]] = {c: {"tp": 0, "fp": 0, "fn": 0} for c in PRIMARY_CLASSES}

    n_images_q1q2q3 = 0
    n_images_q4 = 0

    for cls_dir in CLASS_DIRS:
        gt_cls_dir = GT_VIS_DIR / cls_dir
        pred_cls_dir = pred_root / cls_dir
        if not gt_cls_dir.exists():
            continue

        for gt_qa_path in gt_cls_dir.glob("*_qa.json"):
            stem = gt_qa_path.name.replace("_qa.json", "")
            image_id = f"{cls_dir}/{stem}"

            with open(gt_qa_path, "r", encoding="utf-8") as f:
                gt_data = json.load(f)

            gt_questions = gt_data.get("questions", [])
            if len(gt_questions) < 3:
                continue

            gt_q1 = str(gt_questions[0].get("answer", "")).strip()
            gt_q2 = str(gt_questions[1].get("answer", "")).strip()
            gt_q3 = str(gt_questions[2].get("answer", "")).strip()

            # Load prediction QA
            pred_qa_path = pred_cls_dir / f"{stem}_qa.json"
            if pred_qa_path.exists():
                with open(pred_qa_path, "r", encoding="utf-8") as f:
                    pred_data = json.load(f)
                pred_questions = pred_data.get("questions", [])
                if len(pred_questions) >= 3:
                    pred_q1 = str(pred_questions[0].get("answer", "")).strip()
                    pred_q2 = str(pred_questions[1].get("answer", "")).strip()
                    pred_q3 = str(pred_questions[2].get("answer", "")).strip()
                else:
                    pred_q1 = pred_q2 = pred_q3 = ""
            else:
                pred_q1 = pred_q2 = pred_q3 = ""

            # ---------------- Q1: defects list ----------------
            gt_set = set(parse_defect_list(gt_q1))
            pred_set = set(parse_defect_list(pred_q1))

            inter = gt_set & pred_set
            tp = len(inter)
            q1_tp += tp
            q1_pred_pos += len(pred_set)
            q1_gt_pos += len(gt_set)

            # Per-class Q1 metrics
            for cls in PRIMARY_CLASSES:
                gt_has = cls in gt_set
                pred_has = cls in pred_set
                if gt_has and pred_has:
                    q1_per_class[cls]["tp"] += 1
                elif pred_has and not gt_has:
                    q1_per_class[cls]["fp"] += 1
                elif gt_has and not pred_has:
                    q1_per_class[cls]["fn"] += 1

            # Hit Rate: 1 if at least one correct defect (or both empty and equal)
            if gt_set or pred_set:
                hit = 1 if tp > 0 else 0
            else:
                # Both empty: treat as a "hit"
                hit = 1
            q1_hit_list.append(hit)

            # ---------------- Q2: counts per defect type ----------------
            gt_counts = parse_counts_by_class(gt_q1, gt_q2)
            pred_counts = parse_counts_by_class(pred_q1, pred_q2)

            for c, g in gt_counts.items():
                if g <= 0:
                    continue
                p = pred_counts.get(c, 0)
                abs_err = abs(p - g)
                rel_err = abs_err / float(g)
                q2_abs_errors.append(abs_err)
                q2_rel_errors.append(rel_err)
                # Per-class Q2 metrics
                if c in PRIMARY_CLASSES:
                    q2_per_class[c]["abs_errors"].append(abs_err)
                    q2_per_class[c]["rel_errors"].append(rel_err)

            # ---------------- Q3: bounding boxes ----------------
            gt_boxes_by_class = parse_bbox_answer(gt_q3)
            pred_boxes_by_class = parse_bbox_answer(pred_q3)

            for c in PRIMARY_CLASSES:
                for box in gt_boxes_by_class.get(c, []):
                    det_gt.setdefault(c, []).append((image_id, box))
                for box in pred_boxes_by_class.get(c, []):
                    det_pred.setdefault(c, []).append((image_id, box))

            n_images_q1q2q3 += 1

            # ---------------- Q4: topology ----------------
            gt_top_path = gt_cls_dir / f"{stem}_topology_qa.json"
            if gt_top_path.exists():
                with open(gt_top_path, "r", encoding="utf-8") as f:
                    top_data = json.load(f)
                top_questions = top_data.get("questions", [])
                if not top_questions:
                    continue
                gt_top_ans = str(top_questions[0].get("answer", "")).strip()

                pred_top_path = pred_cls_dir / f"{stem}_topology_qa.json"
                if pred_top_path.exists():
                    with open(pred_top_path, "r", encoding="utf-8") as f:
                        pred_top_data = json.load(f)
                    pred_top_questions = pred_top_data.get("questions", [])
                    pred_top_ans = (
                        str(pred_top_questions[0].get("answer", "")).strip()
                        if pred_top_questions
                        else ""
                    )
                else:
                    pred_top_ans = ""

                gt_rels = parse_topology_relations(gt_top_ans)
                pred_rels = parse_topology_relations(pred_top_ans)

                inter_rel = gt_rels & pred_rels
                tp_rel = len(inter_rel)
                q4_tp += tp_rel
                q4_pred_pos += len(pred_rels)
                q4_gt_pos += len(gt_rels)

                # Per-class Q4 metrics: count relations that involve each class
                for cls in PRIMARY_CLASSES:
                    # GT relations involving this class
                    gt_rels_cls = {r for r in gt_rels if r[0] == cls or r[2] == cls}
                    # Pred relations involving this class
                    pred_rels_cls = {r for r in pred_rels if r[0] == cls or r[2] == cls}
                    # Intersection
                    inter_rels_cls = gt_rels_cls & pred_rels_cls
                    # TP: correctly predicted relations involving this class
                    q4_per_class[cls]["tp"] += len(inter_rels_cls)
                    # FP: predicted but not in GT
                    q4_per_class[cls]["fp"] += len(pred_rels_cls - gt_rels_cls)
                    # FN: in GT but not predicted
                    q4_per_class[cls]["fn"] += len(gt_rels_cls - pred_rels_cls)

                n_images_q4 += 1

    # ---- Aggregate Q1 ----
    q1_precision = q1_tp / float(q1_pred_pos) if q1_pred_pos > 0 else 0.0
    q1_recall = q1_tp / float(q1_gt_pos) if q1_gt_pos > 0 else 0.0
    q1_hit_rate = float(np.mean(q1_hit_list)) if q1_hit_list else 0.0

    # ---- Aggregate Q2 ----
    q2_mae = float(np.mean(q2_abs_errors)) if q2_abs_errors else 0.0
    q2_rel_err = float(np.mean(q2_rel_errors)) if q2_rel_errors else 0.0
    
    # Compute per-class Q2 metrics
    q2_per_class_stats: Dict[str, Dict[str, float]] = {}
    for cls in PRIMARY_CLASSES:
        abs_errs = q2_per_class[cls]["abs_errors"]
        rel_errs = q2_per_class[cls]["rel_errors"]
        mae_cls = float(np.mean(abs_errs)) if abs_errs else 0.0
        rel_err_cls = float(np.mean(rel_errs)) if rel_errs else 0.0
        q2_per_class_stats[cls] = {
            "MAE": mae_cls,
            "Relative_Error": rel_err_cls,
        }

    # ---- Q3 detection metrics ----
    mAP50, mAP5095, f1_det, precision_det, recall_det, per_cls_stats = compute_detection_metrics(det_gt, det_pred)

    # ---- Aggregate Q4 ----
    q4_precision = q4_tp / float(q4_pred_pos) if q4_pred_pos > 0 else 0.0
    q4_recall = q4_tp / float(q4_gt_pos) if q4_gt_pos > 0 else 0.0
    q4_f1 = 2.0 * q4_precision * q4_recall / (q4_precision + q4_recall) if (q4_precision + q4_recall) > 0 else 0.0
    
    # Compute per-class Q4 metrics
    q4_per_class_stats: Dict[str, Dict[str, float]] = {}
    for cls in PRIMARY_CLASSES:
        stats = q4_per_class[cls]
        tp = stats["tp"]
        fp = stats["fp"]
        fn = stats["fn"]
        precision_cls = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
        recall_cls = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
        f1_cls = 2.0 * precision_cls * recall_cls / (precision_cls + recall_cls) if (precision_cls + recall_cls) > 0 else 0.0
        q4_per_class_stats[cls] = {
            "precision": precision_cls,
            "recall": recall_cls,
            "f1": f1_cls,
        }

    print("\n=== Evaluation Results ===")
    print(f"Images evaluated for Q1–Q3: {n_images_q1q2q3}")
    print(f"Images evaluated for Q4 (topology): {n_images_q4}")
    print("\nQ1: What defects are in the image?")
    print(f"  Precision (micro): {q1_precision:.4f}")
    print(f"  Recall   (micro): {q1_recall:.4f}")
    print(f"  Hit Rate       : {q1_hit_rate:.4f}")
    
    # Per-class Q1 metrics
    print("\nPer-class Q1 metrics:")
    print("  {:18s} {:>10s} {:>10s} {:>10s}".format("Class", "Precision", "Recall", "F1"))
    q1_per_class_stats: Dict[str, Dict[str, float]] = {}
    for cls in PRIMARY_CLASSES:
        stats = q1_per_class[cls]
        tp = stats["tp"]
        fp = stats["fp"]
        fn = stats["fn"]
        precision_cls = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
        recall_cls = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
        f1_cls = 2.0 * precision_cls * recall_cls / (precision_cls + recall_cls) if (precision_cls + recall_cls) > 0 else 0.0
        q1_per_class_stats[cls] = {
            "precision": precision_cls,
            "recall": recall_cls,
            "f1": f1_cls,
        }
        print(
            "  {:18s} {:10.4f} {:10.4f} {:10.4f}".format(
                cls, precision_cls, recall_cls, f1_cls
            )
        )

    print("\nQ2: How many instances of each defect type?")
    print(f"  MAE (global)           : {q2_mae:.4f}")
    print(f"  Relative Error (global): {q2_rel_err:.4f}")
    
    # Per-class Q2 metrics
    print("\nPer-class Q2 metrics:")
    print("  {:18s} {:>12s} {:>15s}".format("Class", "MAE", "Relative Error"))
    for cls in PRIMARY_CLASSES:
        stats = q2_per_class_stats.get(cls, {})
        print(
            "  {:18s} {:12.4f} {:15.4f}".format(
                cls, stats.get("MAE", 0.0), stats.get("Relative_Error", 0.0)
            )
        )

    print("\nQ3: Bounding boxes")
    print(f"  mAP@0.5       : {mAP50:.4f}")
    print(f"  mAP@0.5:0.95  : {mAP5095:.4f}")
    print(f"  F1-score@0.5  : {f1_det:.4f}")
    print(f"  Precision@0.5 : {precision_det:.4f}")
    print(f"  Recall@0.5    : {recall_det:.4f}")

    # Per-class table for Q3
    if per_cls_stats:
        print("\nPer-class detection metrics (Q3):")
        print("  {:18s} {:>10s} {:>14s} {:>10s} {:>12s} {:>10s}".format("Class", "AP@0.5", "AP@0.5:0.95", "F1@0.5", "Precision@0.5", "Recall@0.5"))
        for cls in PRIMARY_CLASSES:
            stats = per_cls_stats.get(cls)
            if not stats:
                continue
            print(
                "  {:18s} {:10.4f} {:14.4f} {:10.4f} {:12.4f} {:10.4f}".format(
                    cls, stats["AP50"], stats["AP50_95"], stats["F1"], stats.get("Precision", 0.0), stats.get("Recall", 0.0)
                )
            )

        # Export full evaluation metrics (Q1–Q4 + per-class detection) to CSV
        # Path example:
        #   defect_bench/results/doubao-seed-1-8-251228/doubao-seed-1-8-251228_detection_metrics.csv
        csv_path = RESULTS_DIR / model_name / f"{model_name}_detection_metrics.csv"
        try:
            with csv_path.open("w", encoding="utf-8") as f:
                # Q1 global
                f.write("# Q1: defect types (global metrics)\n")
                f.write("metric,value\n")
                f.write(f"precision_micro,{q1_precision:.6f}\n")
                f.write(f"recall_micro,{q1_recall:.6f}\n")
                f.write(f"hit_rate,{q1_hit_rate:.6f}\n\n")
                
                # Q1 per-class
                f.write("# Q1: defect types (per-class metrics)\n")
                f.write("class,precision,recall,f1\n")
                for cls in PRIMARY_CLASSES:
                    stats = q1_per_class_stats.get(cls, {})
                    f.write(
                        f"{cls},{stats.get('precision', 0.0):.6f},{stats.get('recall', 0.0):.6f},{stats.get('f1', 0.0):.6f}\n"
                    )
                # Add average row
                if q1_per_class_stats:
                    avg_prec = float(np.mean([s["precision"] for s in q1_per_class_stats.values()]))
                    avg_rec = float(np.mean([s["recall"] for s in q1_per_class_stats.values()]))
                    avg_f1 = float(np.mean([s["f1"] for s in q1_per_class_stats.values()]))
                    f.write(f"Average,{avg_prec:.6f},{avg_rec:.6f},{avg_f1:.6f}\n")
                f.write("\n")

                # Q2 global
                f.write("# Q2: counts per defect type (global metrics)\n")
                f.write("metric,value\n")
                f.write(f"MAE,{q2_mae:.6f}\n")
                f.write(f"relative_error,{q2_rel_err:.6f}\n\n")
                
                # Q2 per-class
                f.write("# Q2: counts per defect type (per-class metrics)\n")
                f.write("class,MAE,relative_error\n")
                for cls in PRIMARY_CLASSES:
                    stats = q2_per_class_stats.get(cls, {})
                    f.write(
                        f"{cls},{stats.get('MAE', 0.0):.6f},{stats.get('Relative_Error', 0.0):.6f}\n"
                    )
                # Add average row (only for classes with data)
                valid_mae = [stats["MAE"] for cls, stats in q2_per_class_stats.items() if len(q2_per_class[cls]["abs_errors"]) > 0]
                valid_rel_err = [stats["Relative_Error"] for cls, stats in q2_per_class_stats.items() if len(q2_per_class[cls]["rel_errors"]) > 0]
                if valid_mae or valid_rel_err:
                    avg_mae = float(np.mean(valid_mae)) if valid_mae else 0.0
                    avg_rel_err = float(np.mean(valid_rel_err)) if valid_rel_err else 0.0
                    f.write(f"Average,{avg_mae:.6f},{avg_rel_err:.6f}\n")
                f.write("\n")

                # Q3 global
                f.write("# Q3: bounding boxes (global metrics)\n")
                f.write("metric,value\n")
                f.write(f"mAP@0.5,{mAP50:.6f}\n")
                f.write(f"mAP@0.5:0.95,{mAP5095:.6f}\n")
                f.write(f"F1@0.5,{f1_det:.6f}\n")
                f.write(f"Precision@0.5,{precision_det:.6f}\n")
                f.write(f"Recall@0.5,{recall_det:.6f}\n\n")

                # Q3 per-class
                f.write("# Q3: bounding boxes (per-class metrics)\n")
                f.write("class,AP50,AP50_95,F1@0.5,Precision@0.5,Recall@0.5\n")
                for cls, stats in per_cls_stats.items():
                    f.write(
                        f"{cls},{stats['AP50']:.6f},{stats['AP50_95']:.6f},{stats['F1']:.6f},{stats.get('Precision', 0.0):.6f},{stats.get('Recall', 0.0):.6f}\n"
                    )
                # Add macro average over classes as an extra row
                ap50_values = [s["AP50"] for s in per_cls_stats.values()]
                ap5095_values = [s["AP50_95"] for s in per_cls_stats.values()]
                f1_values = [s["F1"] for s in per_cls_stats.values()]
                precision_values = [s.get("Precision", 0.0) for s in per_cls_stats.values()]
                recall_values = [s.get("Recall", 0.0) for s in per_cls_stats.values()]
                if ap50_values:
                    avg_ap50_classes = float(np.mean(ap50_values))
                    avg_ap5095_classes = float(np.mean(ap5095_values))
                    avg_f1_classes = float(np.mean(f1_values))
                    avg_precision_classes = float(np.mean(precision_values))
                    avg_recall_classes = float(np.mean(recall_values))
                    f.write(
                        f"Average,{avg_ap50_classes:.6f},{avg_ap5095_classes:.6f},{avg_f1_classes:.6f},{avg_precision_classes:.6f},{avg_recall_classes:.6f}\n"
                    )
                f.write("\n")

                # Q4 global
                f.write("# Q4: topology relations (global metrics)\n")
                f.write("metric,value\n")
                f.write(f"precision_micro,{q4_precision:.6f}\n")
                f.write(f"recall_micro,{q4_recall:.6f}\n")
                f.write(f"f1_micro,{q4_f1:.6f}\n\n")
                
                # Q4 per-class
                f.write("# Q4: topology relations (per-class metrics)\n")
                f.write("class,precision,recall,f1\n")
                for cls in PRIMARY_CLASSES:
                    stats = q4_per_class_stats.get(cls, {})
                    f.write(
                        f"{cls},{stats.get('precision', 0.0):.6f},{stats.get('recall', 0.0):.6f},{stats.get('f1', 0.0):.6f}\n"
                    )
                # Add average row
                if q4_per_class_stats:
                    avg_prec = float(np.mean([s["precision"] for s in q4_per_class_stats.values()]))
                    avg_rec = float(np.mean([s["recall"] for s in q4_per_class_stats.values()]))
                    avg_f1 = float(np.mean([s["f1"] for s in q4_per_class_stats.values()]))
                    f.write(f"Average,{avg_prec:.6f},{avg_rec:.6f},{avg_f1:.6f}\n")

            print(f"\nFull evaluation metrics CSV saved to: {csv_path}")
        except Exception as e:
            print(f"\nWarning: failed to write evaluation CSV: {e}")

    print("\nQ4: Topology relations")
    print(f"  Precision (micro): {q4_precision:.4f}")
    print(f"  Recall   (micro): {q4_recall:.4f}")
    print(f"  F1-score (micro): {q4_f1:.4f}")
    
    # Per-class Q4 metrics
    print("\nPer-class Q4 metrics:")
    print("  {:18s} {:>10s} {:>10s} {:>10s}".format("Class", "Precision", "Recall", "F1"))
    for cls in PRIMARY_CLASSES:
        stats = q4_per_class_stats.get(cls, {})
        print(
            "  {:18s} {:10.4f} {:10.4f} {:10.4f}".format(
                cls, stats.get("precision", 0.0), stats.get("recall", 0.0), stats.get("f1", 0.0)
            )
        )
    print("=" * 80)


if __name__ == "__main__":
    main()


