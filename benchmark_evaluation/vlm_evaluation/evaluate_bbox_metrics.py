#!/usr/bin/env python3
"""
Evaluate predicted bbox answers against label JSON ground truth.

Inputs:
- Ground truth labels: defect_bench/data_sample/labels/{stem}.json
- Predicted QA: defect_bench/results/{MODEL_NAME}/images/{stem}_qa.json

Output:
- defect_bench/results/{MODEL_NAME}/{MODEL_NAME}_bbox_metrics.csv
"""

from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple


DATA_ROOT = Path("defect_bench/data_sample")
LABELS_DIR = DATA_ROOT / "labels"
RESULTS_DIR = Path("defect_bench/results")
DEFAULT_MODEL_NAME = "doubao-seed-1-8-251228"


def iou_xywh(a: List[float], b: List[float]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def load_gt_boxes(label_json: Path) -> List[List[float]]:
    data = json.loads(label_json.read_text(encoding="utf-8"))
    out: List[List[float]] = []
    for item in data.get("bboxes", []):
        box = item.get("bbox", [])
        if len(box) == 4:
            out.append([float(box[0]), float(box[1]), float(box[2]), float(box[3])])
    return out


def parse_pred_boxes(qa_json: Path) -> List[List[float]]:
    data = json.loads(qa_json.read_text(encoding="utf-8"))
    questions = data.get("questions", [])
    answer3 = ""
    for q in questions:
        qtext = str(q.get("question", "")).lower()
        if "bounding box coordinates" in qtext:
            answer3 = str(q.get("answer", ""))
            break
    # Parse all [x, y, w, h]
    boxes = []
    for m in re.finditer(r"\[\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\]", answer3):
        boxes.append([float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))])
    return boxes


def match_counts(gt: List[List[float]], pred: List[List[float]], thr: float = 0.5) -> Tuple[int, int, int]:
    used = [False] * len(pred)
    tp = 0
    for g in gt:
        best_j = -1
        best_iou = 0.0
        for j, p in enumerate(pred):
            if used[j]:
                continue
            v = iou_xywh(g, p)
            if v > best_iou:
                best_iou = v
                best_j = j
        if best_j >= 0 and best_iou >= thr:
            used[best_j] = True
            tp += 1
    fp = len(pred) - tp
    fn = len(gt) - tp
    return tp, fp, fn


def main() -> None:
    model_name = os.environ.get("VLM_MODEL_NAME", DEFAULT_MODEL_NAME)
    pred_dir = RESULTS_DIR / model_name / "images"
    out_dir = RESULTS_DIR / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{model_name}_bbox_metrics.csv"

    rows = []
    total_tp = total_fp = total_fn = 0
    for gt_json in sorted(LABELS_DIR.glob("*.json")):
        stem = gt_json.stem
        pred_json = pred_dir / f"{stem}_qa.json"
        gt_boxes = load_gt_boxes(gt_json)
        pred_boxes = parse_pred_boxes(pred_json) if pred_json.exists() else []
        tp, fp, fn = match_counts(gt_boxes, pred_boxes, thr=0.5)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        p = tp / (tp + fp) if tp + fp > 0 else 0.0
        r = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = (2 * p * r / (p + r)) if p + r > 0 else 0.0
        rows.append([stem, len(gt_boxes), len(pred_boxes), tp, fp, fn, p, r, f1])

    P = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0.0
    R = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0.0
    F1 = (2 * P * R / (P + R)) if P + R > 0 else 0.0

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_stem", "gt_count", "pred_count", "tp", "fp", "fn", "precision", "recall", "f1"])
        w.writerows(rows)
        w.writerow([])
        w.writerow(["OVERALL", "", "", total_tp, total_fp, total_fn, P, R, F1])

    print(f"Saved bbox metrics: {out_csv}")
    print(f"Overall @IoU0.5 -> P={P:.4f}, R={R:.4f}, F1={F1:.4f}")


if __name__ == "__main__":
    main()

