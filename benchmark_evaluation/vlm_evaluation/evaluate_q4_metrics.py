#!/usr/bin/env python3
"""
Evaluate Q4 (topology relations) metrics for models in opensrc_q4 directory.

This script processes CSV files containing topology predictions and ground truth,
computes Q4 metrics (Precision, Recall, F1) using the same methodology as 
evaluate_vlm_qa.py Q4 evaluation.

Input: CSV files in opensrc_q4 directory with columns:
  - Image Path: path to image
  - Ground Truth: GT topology answer string
  - Model Prediction: JSON string with "relationships" field
  - Defect List Used: (optional)

Output: Metrics CSV file with all models' results
"""

import os
import json
import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import numpy as np

# Import functions from evaluate_vlm_qa.py
import sys
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from evaluate_vlm_qa import (
    PRIMARY_CLASSES,
    normalize_class_name,
    parse_topology_relations,
)

# Directory containing CSV files
SCRIPT_DIR = Path(__file__).parent


def parse_prediction_relationships(pred_json_str: str) -> str:
    """
    Parse Model Prediction JSON and convert to string format for parse_topology_relations.
    
    Input format: '{"relationships": [["1#Crack", "adjacency", "2#Crack"], ...]}'
    Output format: "[1#Crack, adjacency, 2#Crack]; [2#Stain, above, 3#Crack]"
    """
    if not pred_json_str or not pred_json_str.strip():
        return ""
    
    try:
        pred_data = json.loads(pred_json_str)
        relationships = pred_data.get("relationships", [])
        if not relationships:
            return ""
        
        # Convert each relationship to string format
        relation_strs = []
        for rel in relationships:
            if isinstance(rel, list) and len(rel) == 3:
                # Format: [subject, relation, object]
                subj, rel_name, obj = rel
                relation_strs.append(f"[{subj}, {rel_name}, {obj}]")
            elif isinstance(rel, dict):
                # If it's a dict, try to extract fields
                subj = rel.get("subject", rel.get("from", ""))
                rel_name = rel.get("relation", rel.get("type", ""))
                obj = rel.get("object", rel.get("to", ""))
                if subj and rel_name and obj:
                    relation_strs.append(f"[{subj}, {rel_name}, {obj}]")
        
        return "; ".join(relation_strs)
    
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
        # If parsing fails, try to extract relationships from string directly
        return pred_json_str


def process_csv_file(csv_path: Path) -> Tuple[str, Dict[str, int], Dict[str, Dict[str, int]]]:
    """
    Process a CSV file and extract GT and prediction relations.
    
    Returns:
        (model_name, global_stats, per_class_stats)
        where global_stats: {"tp": int, "fp": int, "fn": int}
        and per_class_stats: {class: {"tp": int, "fp": int, "fn": int}}
    """
    model_name = csv_path.stem.replace("-topo", "").replace("_topo", "")
    
    # Global aggregators
    q4_tp = 0
    q4_pred_pos = 0
    q4_gt_pos = 0
    
    # Per-class aggregators
    q4_per_class: Dict[str, Dict[str, int]] = {
        c: {"tp": 0, "fp": 0, "fn": 0} for c in PRIMARY_CLASSES
    }
    
    with open(csv_path, "r", encoding="utf-8-sig") as f:  # utf-8-sig handles BOM
        reader = csv.DictReader(f)
        for row in reader:
            gt_str = row.get("Ground Truth", "").strip()
            pred_json_str = row.get("Model Prediction", "").strip()
            
            # Check if GT is "Only one defect." (no relations)
            gt_is_empty = (gt_str == "Only one defect." or not gt_str)
            
            # Parse prediction
            pred_rels_str = parse_prediction_relationships(pred_json_str)
            pred_rels = parse_topology_relations(pred_rels_str)
            pred_is_empty = (not pred_rels_str or pred_rels_str.strip() == "" or len(pred_rels) == 0)
            
            # If both GT and prediction are empty (no relations), treat as correct
            # Both empty means: GT="Only one defect." and pred="[]" -> correct, skip counting
            if gt_is_empty and pred_is_empty:
                continue  # Both empty: correct prediction, no TP/FP/FN to count
            
            # If GT is empty but prediction has relations -> FP
            if gt_is_empty and not pred_is_empty:
                q4_pred_pos += len(pred_rels)
                # Count FP per class
                for cls in PRIMARY_CLASSES:
                    pred_rels_cls = {r for r in pred_rels if r[0] == cls or r[2] == cls}
                    q4_per_class[cls]["fp"] += len(pred_rels_cls)
                continue
            
            # If GT has relations but prediction is empty -> FN
            if not gt_is_empty and pred_is_empty:
                gt_rels = parse_topology_relations(gt_str)
                q4_gt_pos += len(gt_rels)
                # Count FN per class
                for cls in PRIMARY_CLASSES:
                    gt_rels_cls = {r for r in gt_rels if r[0] == cls or r[2] == cls}
                    q4_per_class[cls]["fn"] += len(gt_rels_cls)
                continue
            
            # Both GT and prediction have relations -> normal matching
            gt_rels = parse_topology_relations(gt_str)
            
            # Global metrics
            inter_rel = gt_rels & pred_rels
            tp_rel = len(inter_rel)
            q4_tp += tp_rel
            q4_pred_pos += len(pred_rels)
            q4_gt_pos += len(gt_rels)
            
            # Per-class metrics: count relations that involve each class
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
    
    # Compute global metrics
    global_stats = {
        "tp": q4_tp,
        "fp": q4_pred_pos - q4_tp,
        "fn": q4_gt_pos - q4_tp,
    }
    
    # Compute per-class metrics
    per_class_stats: Dict[str, Dict[str, float]] = {}
    for cls in PRIMARY_CLASSES:
        stats = q4_per_class[cls]
        tp = stats["tp"]
        fp = stats["fp"]
        fn = stats["fn"]
        precision_cls = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
        recall_cls = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
        f1_cls = 2.0 * precision_cls * recall_cls / (precision_cls + recall_cls) if (precision_cls + recall_cls) > 0 else 0.0
        per_class_stats[cls] = {
            "precision": precision_cls,
            "recall": recall_cls,
            "f1": f1_cls,
        }
    
    return model_name, global_stats, per_class_stats


def main():
    """Process all CSV files in opensrc_q4 directory and compute metrics."""
    
    print("=" * 80)
    print("Evaluating Q4 (topology relations) metrics for opensrc_q4 models")
    print(f"Directory: {SCRIPT_DIR}")
    print("=" * 80)
    
    # Find all CSV files (exclude metrics files)
    csv_files = sorted([f for f in SCRIPT_DIR.glob("*.csv") if not f.name.endswith("_metrics.csv")])
    if not csv_files:
        print("No CSV files found in directory")
        return
    
    print(f"\nFound {len(csv_files)} CSV file(s)")
    
    # Store all results
    all_results = []
    
    # Process each CSV file
    for csv_path in csv_files:
        print(f"\nProcessing: {csv_path.name}")
        
        try:
            model_name, global_stats, per_class_stats = process_csv_file(csv_path)
            
            # Compute global metrics
            tp = global_stats["tp"]
            fp = global_stats["fp"]
            fn = global_stats["fn"]
            precision = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # Print results
            print(f"\nResults for {model_name}:")
            print(f"  Precision (micro): {precision:.4f}")
            print(f"  Recall   (micro): {recall:.4f}")
            print(f"  F1-score (micro): {f1:.4f}")
            print(f"  TP: {tp}, FP: {fp}, FN: {fn}")
            
            if per_class_stats:
                print("\nPer-class Q4 metrics:")
                print("  {:18s} {:>10s} {:>10s} {:>10s}".format(
                    "Class", "Precision", "Recall", "F1"
                ))
                for cls in PRIMARY_CLASSES:
                    stats = per_class_stats.get(cls, {})
                    print(
                        "  {:18s} {:10.4f} {:10.4f} {:10.4f}".format(
                            cls,
                            stats.get("precision", 0.0),
                            stats.get("recall", 0.0),
                            stats.get("f1", 0.0),
                        )
                    )
            
            # Store results
            all_results.append({
                "model": model_name,
                "global_stats": {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                },
                "per_class_stats": per_class_stats,
            })
        
        except Exception as e:
            print(f"Error processing {csv_path.name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save all results to a single CSV file
    if all_results:
        output_csv = SCRIPT_DIR / "all_models_q4_metrics.csv"
        with open(output_csv, "w", encoding="utf-8") as f:
            # Write results for each model: global metrics first, then per-class metrics
            for result in all_results:
                model_name = result['model']
                global_stats = result['global_stats']
                per_cls_stats = result['per_class_stats']
                
                # Global metrics for this model
                f.write(f"# {model_name} - Global metrics\n")
                f.write("metric,value\n")
                f.write(f"precision_micro,{global_stats['precision']:.6f}\n")
                f.write(f"recall_micro,{global_stats['recall']:.6f}\n")
                f.write(f"f1_micro,{global_stats['f1']:.6f}\n")
                f.write(f"TP,{global_stats['tp']}\n")
                f.write(f"FP,{global_stats['fp']}\n")
                f.write(f"FN,{global_stats['fn']}\n")
                f.write("\n")
                
                # Per-class metrics for this model
                f.write(f"# {model_name} - Per-class metrics\n")
                f.write("class,precision,recall,f1\n")
                for cls in PRIMARY_CLASSES:
                    stats = per_cls_stats.get(cls, {})
                    if not stats:
                        # Write zeros if no stats for this class
                        f.write(
                            f"{cls},0.000000,0.000000,0.000000\n"
                        )
                    else:
                        f.write(
                            f"{cls},"
                            f"{stats.get('precision', 0.0):.6f},"
                            f"{stats.get('recall', 0.0):.6f},"
                            f"{stats.get('f1', 0.0):.6f}\n"
                        )
                f.write("\n")
        
        print(f"\n{'=' * 80}")
        print(f"All metrics saved to: {output_csv}")
        print(f"{'=' * 80}")
    
    print("\n" + "=" * 80)
    print("Evaluation complete")
    print("=" * 80)


if __name__ == "__main__":
    main()

