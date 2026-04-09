#!/usr/bin/env python3
"""
Generate QA ground truth JSON files for images in data_sample/images directory.

For each image, generates a QA JSON file with 3 questions and answers:
1. What defects are in the image? (based on primary_class set)
2. How many instances of each defect type? (count bboxes per type)
3. What are the bounding box coordinates of each defect instance? (outputs raw bbox coordinates)

Output: JSON files saved to {TEST100_DIR}/Visualization/{class_dir_name}/{image_name}_qa.json
"""

import json
from pathlib import Path
from collections import defaultdict
from PIL import Image
from typing import Dict, List, Tuple, Set

# Base directory
TEST100_DIR = Path("defect_bench/data_sample")
CLASS_DIRS = ["images"]

# Note: We use primary_class names directly without mapping

# 9-grid position names
POSITION_NAMES = [
    "top-left", "top-center", "top-right",
    "middle-left", "center", "middle-right",
    "bottom-left", "bottom-center", "bottom-right"
]


def get_image_size(image_path: Path) -> Tuple[int, int]:
    """Get image dimensions."""
    try:
        img = Image.open(image_path)
        return img.size  # (width, height)
    except Exception as e:
        print(f"Warning: Could not read image {image_path}: {e}")
        return None, None


def get_bbox_position(bbox: List[float], img_width: int, img_height: int) -> str:
    """
    Determine the 9-grid position of a bbox.
    
    Args:
        bbox: [x, y, width, height]
        img_width: Image width
        img_height: Image height
    
    Returns:
        Position name (e.g., "top-left", "center")
    """
    x, y, w, h = bbox
    # Calculate bbox center
    center_x = x + w / 2.0
    center_y = y + h / 2.0
    
    # Divide image into 3x3 grid
    third_w = img_width / 3.0
    third_h = img_height / 3.0
    
    # Determine horizontal position
    if center_x < third_w:
        h_pos = "left"
    elif center_x < 2 * third_w:
        h_pos = "center"
    else:
        h_pos = "right"
    
    # Determine vertical position
    if center_y < third_h:
        v_pos = "top"
    elif center_y < 2 * third_h:
        v_pos = "middle"
    else:
        v_pos = "bottom"
    
    # Combine to get position name
    if v_pos == "middle" and h_pos == "center":
        return "center"
    elif v_pos == "top":
        return f"top-{h_pos}"
    elif v_pos == "middle":
        return f"middle-{h_pos}"
    else:  # bottom
        return f"bottom-{h_pos}"


def generate_qa_for_image(json_path: Path, image_path: Path) -> Dict:
    """
    Generate QA ground truth for a single image.
    
    Args:
        json_path: Path to bbox JSON file
        image_path: Path to image file
    
    Returns:
        QA dictionary with questions and answers
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
    
    # Get image size
    img_width, img_height = get_image_size(image_path)
    if img_width is None or img_height is None:
        # Try to infer from bboxes
        max_x = max(bbox[0] + bbox[2] for bbox in [b.get("bbox", []) for b in bboxes] if len(b.get("bbox", [])) == 4)
        max_y = max(bbox[1] + bbox[3] for bbox in [b.get("bbox", []) for b in bboxes] if len(b.get("bbox", [])) == 4)
        img_width = int(max_x) if max_x else 1024
        img_height = int(max_y) if max_y else 1024
        print(f"Warning: Could not read image size, inferred {img_width}x{img_height} from bboxes")
    
    # Question 1: What defects are in the image?
    primary_classes = set()
    for bbox_item in bboxes:
        primary_class = bbox_item.get("taxonomy", {}).get("primary_class")
        if primary_class:
            primary_classes.add(primary_class)
    
    # Sort for consistent output
    primary_classes_sorted = sorted(primary_classes)
    answer1 = ", ".join(primary_classes_sorted)
    
    # Question 2: How many instances of each defect type?
    type_counts = defaultdict(int)
    for bbox_item in bboxes:
        primary_class = bbox_item.get("taxonomy", {}).get("primary_class")
        if primary_class:
            type_counts[primary_class] += 1
    
    # Format answer as "3, 4" (counts in same order as question 1)
    answer2_parts = []
    for pc in primary_classes_sorted:
        count = type_counts[pc]
        answer2_parts.append(str(count))
    answer2 = ", ".join(answer2_parts)
    
    # Question 3: What are the bounding box coordinates of each defect instance?
    # Answer format example:
    # "Crack: [x1, y1, w1, h1]; [x2, y2, w2, h2]; Spalling: [x3, y3, w3, h3]"
    type_bboxes = defaultdict(list)
    for bbox_item in bboxes:
        primary_class = bbox_item.get("taxonomy", {}).get("primary_class")
        bbox = bbox_item.get("bbox", [])
        if primary_class and len(bbox) == 4:
            type_bboxes[primary_class].append(bbox)

    answer3_parts = []
    for pc in primary_classes_sorted:
        bboxes_list = type_bboxes.get(pc)
        if not bboxes_list:
            continue
        coord_strs = []
        for bbox in bboxes_list:
            # Use raw bbox coordinates [x, y, width, height]
            coord_strs.append(f"[{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")
        answer3_parts.append(f"{pc}: " + "; ".join(coord_strs))

    answer3 = "; ".join(answer3_parts)
    
    # Construct QA dictionary
    qa_data = {
        "image_path": str(image_path.name),
        "questions": [
            {
                "question": "What defects are in the image?",
                "answer": answer1
            },
            {
                "question": f"How many instances of each defect type ({answer1})?",
                "answer": answer2
            },
            {
                "question": "What are the bounding box coordinates of each defect instance (x, y, width, height)?",
                "answer": answer3
            }
        ]
    }
    
    return qa_data


def main():
    """Main function to generate QA files for all images."""
    print("Generating QA ground truth files...")
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
        
        # Find all JSON files (excluding QA directory and mask files)
        json_files = [f for f in (TEST100_DIR / "labels").glob("*.json") if not f.name.endswith("_qa.json")]
        
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
                total_errors += 1
                continue
            
            # Generate QA data
            qa_data = generate_qa_for_image(json_file, image_path)
            if qa_data is None:
                print(f"  Warning: Could not generate QA for {json_file.name}, skipping...")
                total_errors += 1
                continue
            
            # Save QA JSON file
            qa_output_path = qa_dir / f"{image_name}_qa.json"
            try:
                with open(qa_output_path, 'w', encoding='utf-8') as f:
                    json.dump(qa_data, f, indent=2, ensure_ascii=False)
                total_processed += 1
            except Exception as e:
                print(f"  Error saving QA file {qa_output_path}: {e}")
                total_errors += 1
    
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Processed: {total_processed} images")
    print(f"  Errors: {total_errors}")
    print(f"  QA files saved to: {visualization_dir}/*/")
    print("=" * 60)


if __name__ == "__main__":
    main()

