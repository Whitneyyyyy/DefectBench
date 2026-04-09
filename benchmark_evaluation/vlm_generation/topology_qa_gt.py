#!/usr/bin/env python3
"""
Generate topology QA ground truth JSON files for images in data_sample/images directory.

For each image, generates a QA JSON file with bbox relationship information.
Question: What are the relationships between the defects in the image?
Answer: List of [bbox1#primary_class, relation, bbox2#primary_class] pairs.

Relations:
- inclusion: defect1 is completely within defect2's boundaries
- overlapping: two different types of defects partially overlap in space
- adjacency: two defects' boundaries (with buffer) are in contact but do not overlap
- disjoint: defects have no spatial intersection or contact

Output: JSON files saved to {TEST100_DIR}/Visualization/{class_dir_name}/{image_name}_topology_qa.json
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import math

# Base directory
TEST100_DIR = Path("defect_bench/data_sample")
CLASS_DIRS = ["images"]

# Note: We use primary_class names directly without mapping

# Buffer distance for adjacency detection (in pixels)
ADJACENCY_BUFFER = 10.0

# IoU threshold: if IoU < this value, treat as adjacency instead of overlapping
OVERLAPPING_IOU_THRESHOLD = 0.05


def bbox_to_xyxy(bbox: List[float]) -> Tuple[float, float, float, float]:
    """Convert bbox from [x, y, width, height] to [x1, y1, x2, y2]."""
    x, y, w, h = bbox
    return x, y, x + w, y + h


def bbox_area(bbox: List[float]) -> float:
    """Calculate bbox area."""
    _, _, w, h = bbox
    return max(0.0, float(w)) * max(0.0, float(h))


def bbox_iou(b1: List[float], b2: List[float]) -> float:
    """Calculate IoU (Intersection over Union) between two bboxes."""
    x1_min, y1_min, x1_max, y1_max = bbox_to_xyxy(b1)
    x2_min, y2_min, x2_max, y2_max = bbox_to_xyxy(b2)

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_w = max(0.0, inter_x_max - inter_x_min)
    inter_h = max(0.0, inter_y_max - inter_y_min)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area1 = bbox_area(b1)
    area2 = bbox_area(b2)
    union = area1 + area2 - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def bbox_containment_ratio(b1: List[float], b2: List[float]) -> float:
    """
    Calculate containment ratio: intersection area / smaller bbox area.
    Returns ratio >= 1.0 if one bbox is completely contained in the other.
    """
    x1_min, y1_min, x1_max, y1_max = bbox_to_xyxy(b1)
    x2_min, y2_min, x2_max, y2_max = bbox_to_xyxy(b2)

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_w = max(0.0, inter_x_max - inter_x_min)
    inter_h = max(0.0, inter_y_max - inter_y_min)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area1 = bbox_area(b1)
    area2 = bbox_area(b2)
    smaller = max(1e-6, min(area1, area2))
    return inter_area / smaller


def bbox_boundary_distance(b1: List[float], b2: List[float]) -> float:
    """
    Calculate the minimum distance between two bbox boundaries.
    Returns 0 if they overlap, positive value if separated.
    """
    x1_min, y1_min, x1_max, y1_max = bbox_to_xyxy(b1)
    x2_min, y2_min, x2_max, y2_max = bbox_to_xyxy(b2)

    # Check if they overlap
    if not (x1_max < x2_min or x2_max < x1_min or y1_max < y2_min or y2_max < y1_min):
        return 0.0  # They overlap

    # Calculate minimum distance between boundaries
    # Horizontal distance
    if x1_max < x2_min:
        h_dist = x2_min - x1_max
    elif x2_max < x1_min:
        h_dist = x1_min - x2_max
    else:
        h_dist = 0.0

    # Vertical distance
    if y1_max < y2_min:
        v_dist = y2_min - y1_max
    elif y2_max < y1_min:
        v_dist = y1_min - y2_max
    else:
        v_dist = 0.0

    # If both are non-zero, they are diagonally separated
    if h_dist > 0 and v_dist > 0:
        return math.sqrt(h_dist ** 2 + v_dist ** 2)
    else:
        return max(h_dist, v_dist)


def determine_relation(b1: List[float], b2: List[float], 
                      primary_class1: str, primary_class2: str) -> Optional[str]:
    """
    Determine the spatial relationship between two bboxes.
    
    Args:
        b1: First bbox [x, y, width, height]
        b2: Second bbox [x, y, width, height]
        primary_class1: Primary class of first bbox
        primary_class2: Primary class of second bbox
    
    Returns:
        Relation type: "inclusion", "overlapping", "adjacency", "disjoint", or None
    """
    # Check inclusion: one bbox is completely within the other
    # containment_ratio = intersection_area / smaller_bbox_area
    # If ratio >= 1.0, the smaller bbox is completely contained in the larger one
    containment_ratio_1_2 = bbox_containment_ratio(b1, b2)
    containment_ratio_2_1 = bbox_containment_ratio(b2, b1)
    
    # If containment ratio >= 1.0, one is completely contained in the other
    if containment_ratio_1_2 >= 1.0 or containment_ratio_2_1 >= 1.0:
        return "inclusion"
    
    # Calculate IoU
    iou = bbox_iou(b1, b2)
    
    # Check overlapping: IoU must be significant (>= threshold) to be considered overlapping
    # If IoU is very small (< threshold), treat as adjacency instead
    if iou >= OVERLAPPING_IOU_THRESHOLD:
        return "overlapping"
    
    # Check adjacency: boundaries are close (within buffer) or have very small overlap
    boundary_dist = bbox_boundary_distance(b1, b2)
    if boundary_dist == 0.0:
        # They overlap but IoU < threshold, treat as adjacency
        return "adjacency"
    elif 0 < boundary_dist <= ADJACENCY_BUFFER:
        return "adjacency"
    
    # Otherwise, they are disjoint
    if boundary_dist > ADJACENCY_BUFFER:
        return "disjoint"
    
    return None


def generate_topology_qa_for_image(json_path: Path, image_path: Path) -> Optional[Dict]:
    """
    Generate topology QA ground truth for a single image.
    
    Args:
        json_path: Path to bbox JSON file
        image_path: Path to image file
    
    Returns:
        QA dictionary with question and answer about bbox relationships
    """
    # Load bbox data
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            bbox_data = json.load(f)
    except Exception as e:
        print(f"Error loading {json_path}: {e}")
        return None
    
    bboxes = bbox_data.get("bboxes", [])
    if not bboxes:
        return None
    
    # Filter valid bboxes (must have bbox and taxonomy)
    valid_bboxes = []
    for bbox_item in bboxes:
        bbox = bbox_item.get("bbox", [])
        primary_class = bbox_item.get("taxonomy", {}).get("primary_class")
        if len(bbox) == 4 and primary_class:
            valid_bboxes.append({
                "bbox": bbox,
                "primary_class": primary_class
            })
    
    if len(valid_bboxes) == 0:
        # No valid bboxes found
        return None
    
    if len(valid_bboxes) == 1:
        # Only one bbox, no relationships
        answer = "Only one defect."
    else:
        # Generate relationships (only for pairs where i < j to avoid duplicates)
        relationships = []
        for i in range(len(valid_bboxes)):
            for j in range(i + 1, len(valid_bboxes)):
                b1 = valid_bboxes[i]["bbox"]
                b2 = valid_bboxes[j]["bbox"]
                pc1 = valid_bboxes[i]["primary_class"]
                pc2 = valid_bboxes[j]["primary_class"]
                
                # Determine relation
                relation = determine_relation(b1, b2, pc1, pc2)
                if relation is None:
                    continue
                
                # Format: [1#primary_class, relation, 2#primary_class]
                # Note: numbering starts from 1 (matching visualize_annotations.py)
                bbox1_label = f"{i + 1}#{pc1}"
                bbox2_label = f"{j + 1}#{pc2}"
                
                relationships.append([bbox1_label, relation, bbox2_label])
        
        if not relationships:
            # No relationships found (all disjoint or other edge cases)
            answer = "No relationships detected between defects."
        else:
            # Format answer as list of relationship strings
            answer_parts = []
            for rel in relationships:
                answer_parts.append(f"[{rel[0]}, {rel[1]}, {rel[2]}]")
            answer = "; ".join(answer_parts)
    
    # Construct QA dictionary
    qa_data = {
        "image_path": str(image_path.name),
        "questions": [
            {
                "question": "What are the relationships between the defects in the image?",
                "answer": answer
            }
        ]
    }
    
    return qa_data


def main():
    """Main function to generate topology QA files for all images."""
    print("Generating topology QA ground truth files...")
    print("=" * 60)
    
    total_processed = 0
    total_errors = 0
    
    visualization_dir = Path("defect_bench/results/ground_truth/Visualization")
    
    for class_dir_name in CLASS_DIRS:
        class_dir = TEST100_DIR / "labels"
        if not class_dir.exists():
            print(f"Warning: Directory {class_dir} does not exist, skipping...")
            continue
        
        # Save QA files to Visualization/{class_dir_name}/
        qa_dir = visualization_dir / class_dir_name
        qa_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nProcessing {class_dir_name}...")
        
        # Find all JSON files (excluding QA files and mask files)
        json_files = [
            f for f in (TEST100_DIR / "labels").glob("*.json")
            if not f.name.endswith("_qa.json") and not f.name.endswith("_topology_qa.json")
        ]
        
        processed = 0
        errors = 0
        
        for json_file in json_files:
            # Find corresponding image file
            image_name = json_file.stem
            image_path = None
            
            # Try different image extensions
            for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
                potential_image = TEST100_DIR / "images" / f"{image_name}{ext}"
                if potential_image.exists():
                    image_path = potential_image
                    break
            
            if image_path is None:
                print(f"  Warning: No image found for {json_file.name}, skipping...")
                errors += 1
                continue
            
            # Generate topology QA data
            qa_data = generate_topology_qa_for_image(json_file, image_path)
            if qa_data is None:
                print(f"  Warning: Could not generate topology QA for {json_file.name}, skipping...")
                errors += 1
                continue
            
            # Save topology QA JSON file (with _topology_qa.json suffix to distinguish from _qa.json)
            qa_output_path = qa_dir / f"{image_name}_topology_qa.json"
            try:
                with open(qa_output_path, 'w', encoding='utf-8') as f:
                    json.dump(qa_data, f, indent=2, ensure_ascii=False)
                processed += 1
            except Exception as e:
                print(f"  Error saving topology QA file {qa_output_path}: {e}")
                errors += 1
        
        print(f"  {class_dir_name}: Processed {processed}, Errors {errors}")
        total_processed += processed
        total_errors += errors
    
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Processed: {total_processed} images")
    print(f"  Errors: {total_errors}")
    print(f"  Topology QA files saved to: {visualization_dir}/*/")
    print("=" * 60)


if __name__ == "__main__":
    main()
