#!/usr/bin/env python3
"""
统一各数据集的bbox标注格式，转换为统一的JSON格式。

输出格式：
{
  "instance_id": "id name",
  "taxonomy": {
    "primary_class": "Crack",
    "sub_type": "crack",
  },
  "visual_features": {
    "bbox": [x, y, width, height]
  },
}
"""

import os
import json
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set, Any
import cv2
import numpy as np
import math

# 标签映射规则：sub_type -> primary_class
LABEL_MAPPING = {
    # Crack
    'major_crack': 'Crack',
    'minor_crack': 'Crack',
    'crack': 'Crack',
    'stairstep_crack': 'Crack',
    'cracks': 'Crack',
    
    # Material Loss
    'peeling': 'Material_loss',
    'spalling': 'Material_loss',
    'Spalling': 'Material_loss',
    'flakes': 'Material_loss',
    'peeling_paint': 'Material_loss',
    'Abscission': 'Material_loss',
    'abscission': 'Material_loss',  # 小写版本
    'Bulge': 'Material_loss',
    'bulge': 'Material_loss',  # 小写版本
    'Material_loss': 'Material_loss',
    
    # Stain
    'algae': 'Stain',
    'stain': 'Stain',
    'biological_deteriorations': 'Stain',
    'mold': 'Stain',
    'water_seepage': 'Stain',
    'Dampness': 'Stain',
    'Efflorescence': 'Stain',
    'Leakage': 'Stain',
    'Corrosion': 'Stain',
    'chemical_deteriorations': 'Stain',
    
    # External Fixings
    'human_caused_damages': 'External Fixings',
    'Vegetation': 'External Fixings',
}

# sub_type 统一映射规则：将各种 sub_type 统一到标准分类
# Material Loss: Spalling, Peeling, Bulging
# Stain: Corrosion, Rust stain, Leakage stain
# External Fixings: Grafiti, Vegetation growth, Surface contaminants
# Crack: Linear crack, Map cracking
SUB_TYPE_MAPPING = {
    # Crack -> Map cracking
    'major_crack': 'Map cracking',
    
    # Crack -> Linear crack
    'crack': 'Linear crack',
    'minor_crack': 'Linear crack',
    'stairstep_crack': 'Linear crack',
    'cracks': 'Linear crack',
    'Concrete_Crack': 'Linear crack',
    
    # Material Loss -> Spalling
    'spalling': 'Spalling',
    'Spalling': 'Spalling',
    'Concrete_Spalling': 'Spalling',
    'Tile_spalling': 'Spalling',
    'Material_loss': 'Spalling',
    'Degraded Plaster': 'Spalling',
    'Degraded_Plaster': 'Spalling',
    
    # Material Loss -> Peeling
    'peeling': 'Peeling',
    'flakes': 'Peeling',
    'peeling_paint': 'Peeling',
    'Abscission': 'Peeling',
    'abscission': 'Peeling',
    'Concrete_Delamination': 'Peeling',
    
    # Material Loss -> Bulging
    'Bulge': 'Bulging',
    'bulge': 'Bulging',
    
    # Stain -> Corrosion
    'Corrosion': 'Corrosion',
    'biological_deteriorations': 'Corrosion',
    
    # Stain -> Rust stain
    'Rust_Stain': 'Rust stain',
    
    # Stain -> Leakage stain
    'algae': 'Leakage stain',
    'stain': 'Leakage stain',
    'mold': 'Leakage stain',
    'water_seepage': 'Leakage stain',
    'Dampness': 'Leakage stain',
    'Efflorescence': 'Leakage stain',
    'Leakage': 'Leakage stain',
    'chemical_deteriorations': 'Leakage stain',
    'moisture': 'Leakage stain',
    'Water_Stain': 'Leakage stain',
    
    # External Fixings -> Grafiti
    'human_caused_damages': 'Grafiti',
    
    # External Fixings -> Vegetation growth
    'Vegetation': 'Vegetation growth',
    'Plant_growth': 'Vegetation growth',
    'Vegeterian': 'Vegetation growth',  # 拼写错误，统一为 Vegetation growth
    
    # External Fixings -> Surface contaminants
    'external_fixings': 'Surface contaminants',
    'Contaminants': 'Surface contaminants',
}

# Input root (set by user when preprocessing raw datasets).
# Keep empty by default to avoid accidental coupling to machine-specific paths.
RAW_DATA_ROOT = Path("")
BASE_DIR = RAW_DATA_ROOT

# Fixed outputs inside defect_bench/data_sample
OUTPUT_IMAGE_DIR = Path("defect_bench/data_sample/images")
OUTPUT_LABEL_DIR = Path("defect_bench/data_sample/labels")
OUTPUT_MASK_DIR = Path("defect_bench/data_sample/masks")
OUTPUT_DIR = OUTPUT_LABEL_DIR


# ==== BBox merge helpers (shared logic with copy_candidates_to_test100.py) ====

# 合并策略参数（基于空间距离的聚类）
IOU_MERGE_THRESHOLD = 0.05         # IoU >= 0.05 才合并（避免轻微重叠就合并）
CONTAINMENT_RATIO_THRESHOLD = 0.7  # 较小框有 >=70% 面积被大框覆盖就合并
CENTER_DIST_RATIO_THRESHOLD = 0.4  # 中心距离 <= 0.4 * 平均对角线长度 -> 合并（更严格）
UNION_AREA_RATIO_THRESHOLD = 1.4   # 如果 union 面积 <= 1.4 * (area1 + area2)，说明很接近，合并


def _bbox_to_xyxy(bbox: List[float]) -> Tuple[float, float, float, float]:
    x, y, w, h = bbox
    return x, y, x + w, y + h


def _bbox_area(bbox: List[float]) -> float:
    _, _, w, h = bbox
    return max(0.0, float(w)) * max(0.0, float(h))


def _bbox_iou(b1: List[float], b2: List[float]) -> float:
    x1_min, y1_min, x1_max, y1_max = _bbox_to_xyxy(b1)
    x2_min, y2_min, x2_max, y2_max = _bbox_to_xyxy(b2)

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_w = max(0.0, inter_x_max - inter_x_min)
    inter_h = max(0.0, inter_y_max - inter_y_min)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area1 = _bbox_area(b1)
    area2 = _bbox_area(b2)
    union = area1 + area2 - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def _bbox_containment_ratio(b1: List[float], b2: List[float]) -> float:
    x1_min, y1_min, x1_max, y1_max = _bbox_to_xyxy(b1)
    x2_min, y2_min, x2_max, y2_max = _bbox_to_xyxy(b2)

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_w = max(0.0, inter_x_max - inter_x_min)
    inter_h = max(0.0, inter_y_max - inter_y_min)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area1 = _bbox_area(b1)
    area2 = _bbox_area(b2)
    smaller = max(1e-6, min(area1, area2))
    return inter_area / smaller


def _bbox_avg_diagonal(b1: List[float], b2: List[float]) -> float:
    """计算两个框的平均对角线长度"""
    _, _, w1, h1 = b1
    _, _, w2, h2 = b2
    diag1 = math.hypot(w1, h1)
    diag2 = math.hypot(w2, h2)
    return (diag1 + diag2) / 2.0


def _bbox_union_area_ratio(b1: List[float], b2: List[float]) -> float:
    """计算 union 面积与两框面积之和的比值。比值越小，说明两框越接近。"""
    x1_min, y1_min, x1_max, y1_max = _bbox_to_xyxy(b1)
    x2_min, y2_min, x2_max, y2_max = _bbox_to_xyxy(b2)
    
    union_x_min = min(x1_min, x2_min)
    union_y_min = min(y1_min, y2_min)
    union_x_max = max(x1_max, x2_max)
    union_y_max = max(y1_max, y2_max)
    
    union_area = (union_x_max - union_x_min) * (union_y_max - union_y_min)
    area1 = _bbox_area(b1)
    area2 = _bbox_area(b2)
    sum_area = area1 + area2
    
    if sum_area <= 0:
        return float('inf')
    return union_area / sum_area


def _bbox_center_distance_ratio(b1: List[float], b2: List[float]) -> float:
    """计算中心距离与平均对角线长度的比值"""
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    c1x = x1 + w1 / 2.0
    c1y = y1 + h1 / 2.0
    c2x = x2 + w2 / 2.0
    c2y = y2 + h2 / 2.0
    dist = math.hypot(c1x - c2x, c1y - c2y)
    avg_diagonal = _bbox_avg_diagonal(b1, b2)
    if avg_diagonal <= 0:
        return float('inf')
    return dist / avg_diagonal


def should_merge_bboxes(b1: List[float], b2: List[float]) -> bool:
    """
    Decide whether two boxes of the same type should be merged.
    This helper is exposed for potential future per-image aggregation.
    
    新的聚类策略（基于空间距离）：
      1. 大框包小框：containment >= 0.7 -> 合并
      2. 有显著重叠：IoU >= 0.05 -> 合并
      3. 空间上很接近：
         - 中心距离 <= 0.4 * 平均对角线长度 -> 合并
         - 或者 union 面积 <= 1.4 * (area1 + area2) -> 合并（说明两框很接近）
      4. 对于小框（面积 < 1000），使用更宽松的条件：
         - 中心距离 <= 100 像素 -> 合并（小框之间距离很近就合并）
         - 或者 union 面积 <= 5.0 * (area1 + area2) -> 合并（小框的最小外接矩形合理）
    """
    area1 = _bbox_area(b1)
    area2 = _bbox_area(b2)
    min_area = min(area1, area2)
    
    # 1. 大框包小框
    containment_ratio = _bbox_containment_ratio(b1, b2)
    if containment_ratio >= CONTAINMENT_RATIO_THRESHOLD:
        return True

    # 2. 有显著重叠
    iou = _bbox_iou(b1, b2)
    if iou >= IOU_MERGE_THRESHOLD:
        return True

    # 3. 空间上很接近：union 面积接近两框面积之和
    union_ratio = _bbox_union_area_ratio(b1, b2)
    if union_ratio <= UNION_AREA_RATIO_THRESHOLD:
        return True

    # 4. 中心距离很近（相对于框的大小）
    center_dist_ratio = _bbox_center_distance_ratio(b1, b2)
    if center_dist_ratio <= CENTER_DIST_RATIO_THRESHOLD:
        return True

    # 5. 对于小框的特殊处理
    if min_area < 1000:  # 至少有一个框是小框（面积 < 1000）
        # 计算绝对中心距离（像素）
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        c1x = x1 + w1 / 2.0
        c1y = y1 + h1 / 2.0
        c2x = x2 + w2 / 2.0
        c2y = y2 + h2 / 2.0
        center_dist = math.hypot(c1x - c2x, c1y - c2y)
        
        # 小框之间中心距离 <= 100 像素就合并
        if center_dist <= 100.0:
            return True
        
        # 小框的最小外接矩形：union_ratio <= 5.0 就合并（更宽松）
        if union_ratio <= 5.0:
            return True

    return False


def get_primary_class(sub_type: str) -> Optional[str]:
    """根据sub_type获取primary_class，如果不在映射表中则返回None"""
    return LABEL_MAPPING.get(sub_type)


def normalize_sub_type(sub_type: str, primary_class: str) -> str:
    """
    将 sub_type 统一到标准分类
    
    Material Loss: Spalling, Peeling, Bulging
    Stain: Corrosion, Rust stain, Leakage stain
    External Fixings: Grafiti, Vegetation growth, Surface contaminants
    Crack: Linear crack, Map cracking
    
    Args:
        sub_type: 原始 sub_type
        primary_class: primary_class，用于确定映射规则
    
    Returns:
        统一后的 sub_type
    """
    # 如果 sub_type 已经在映射表中，直接返回
    if sub_type in SUB_TYPE_MAPPING:
        return SUB_TYPE_MAPPING[sub_type]
    
    # 如果 primary_class 是 Crack，但 sub_type 不在映射表中，默认映射为 Linear crack
    if primary_class == 'Crack':
        return 'Linear crack'
    
    # 其他情况，如果不在映射表中，返回原值（可能需要警告）
    return sub_type


def mask_to_bbox_from_binary(binary_mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """从二值mask中提取bbox"""
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for contour in contours:
        if cv2.contourArea(contour) > 10:
            x, y, w, h = cv2.boundingRect(contour)
            bboxes.append((x, y, w, h))
    return bboxes


def mask_to_bbox(mask_path: Path) -> List[Tuple[int, int, int, int]]:
    """从mask图像中提取所有非零区域的bounding box"""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []
    
    return mask_to_bbox_from_binary(mask)


def process_classification_dataset(dataset_name: str, dataset_path: Path):
    """处理classification数据集：整个图片就是一个bbox"""
    print(f"\n处理 Classification - {dataset_name}...")
    
    # 获取图片尺寸和类别
    if dataset_name == 'HS-23K':
        # HS-23K: 文件夹名即类别（可能有数字前缀，如"3.cracks"）
        for class_dir in dataset_path.iterdir():
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            # 去掉数字前缀（如"3.cracks" -> "cracks"）
            sub_type = class_name.split('.', 1)[-1] if '.' in class_name else class_name
            primary_class = get_primary_class(sub_type)
            if primary_class is None:
                continue
            
            for img_file in class_dir.rglob('*.jpg'):
                if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                    continue
                
                img = cv2.imread(str(img_file))
                if img is None:
                    continue
                h, w = img.shape[:2]
                
                instance_id = f"{dataset_name}_{img_file.stem}"
                normalized_sub_type = normalize_sub_type(sub_type, primary_class)
                output_data = {
                    "instance_id": instance_id,
                    "taxonomy": {
                        "primary_class": primary_class,
                        "sub_type": normalized_sub_type,
                    },
                    "visual_features": {
                        "bbox": [0, 0, w, h]  # 整个图片
                    },
                    "image_path": str(img_file.relative_to(BASE_DIR))
                }
                
                output_file = OUTPUT_DIR / f"{instance_id}.json"
                output_file.parent.mkdir(parents=True, exist_ok=True)
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    elif dataset_name == 'BD3':
        # BD3: 文件夹名即类别
        for class_dir in dataset_path.iterdir():
            if not class_dir.is_dir():
                continue
            sub_type = class_dir.name
            primary_class = get_primary_class(sub_type)
            if primary_class is None:
                continue
            
            for img_file in class_dir.rglob('*.jpg'):
                img = cv2.imread(str(img_file))
                if img is None:
                    continue
                h, w = img.shape[:2]
                
                instance_id = f"{dataset_name}_{img_file.stem}"
                normalized_sub_type = normalize_sub_type(sub_type, primary_class)
                output_data = {
                    "instance_id": instance_id,
                    "taxonomy": {
                        "primary_class": primary_class,
                        "sub_type": normalized_sub_type,
                    },
                    "visual_features": {
                        "bbox": [0, 0, w, h]
                    },
                    "image_path": str(img_file.relative_to(BASE_DIR))
                }
                
                output_file = OUTPUT_DIR / f"{instance_id}.json"
                output_file.parent.mkdir(parents=True, exist_ok=True)
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    elif dataset_name == 'CCIC':
        # CCIC: 所有图片都是crack
        sub_type = 'crack'
        primary_class = get_primary_class(sub_type)
        
        for img_file in dataset_path.glob('*.jpg'):
            img = cv2.imread(str(img_file))
            if img is None:
                continue
            h, w = img.shape[:2]
            
            instance_id = f"{dataset_name}_{img_file.stem}"
            normalized_sub_type = normalize_sub_type(sub_type, primary_class)
            output_data = {
                "instance_id": instance_id,
                "taxonomy": {
                    "primary_class": primary_class,
                    "sub_type": normalized_sub_type,
                },
                "visual_features": {
                    "bbox": [0, 0, w, h]
                },
                "image_path": str(img_file.relative_to(BASE_DIR))
            }
            
            output_file = OUTPUT_DIR / f"{instance_id}.json"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    elif dataset_name == 'BDD':
        # BDD: 结构是 train_set/类别 和 test_set/类别
        for split_dir in ['train_set', 'test_set']:
            split_path = dataset_path / split_dir
            if not split_path.exists():
                continue
            
            for class_dir in split_path.iterdir():
                if not class_dir.is_dir():
                    continue
                sub_type = class_dir.name
                primary_class = get_primary_class(sub_type)
                if primary_class is None:
                    continue
                
                for img_file in class_dir.rglob('*.jpg'):
                    img = cv2.imread(str(img_file))
                    if img is None:
                        continue
                    h, w = img.shape[:2]
                    
                    instance_id = f"{dataset_name}_{split_dir}_{img_file.stem}"
                    normalized_sub_type = normalize_sub_type(sub_type, primary_class)
                    output_data = {
                        "instance_id": instance_id,
                        "taxonomy": {
                            "primary_class": primary_class,
                            "sub_type": normalized_sub_type,
                        },
                        "visual_features": {
                            "bbox": [0, 0, w, h]
                        },
                        "image_path": str(img_file.relative_to(BASE_DIR))
                    }
                    
                    output_file = OUTPUT_DIR / f"{instance_id}.json"
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(output_data, f, indent=2, ensure_ascii=False)


def process_bdw(dataset_path: Path):
    """处理BDW数据集：COCO格式标注"""
    print(f"\n处理 Detection - BDW (COCO格式)...")
    
    coco_file = dataset_path / 'train' / '_annotations.coco.json'
    if not coco_file.exists():
        print(f"  警告: 找不到COCO标注文件 {coco_file}")
        return
    
    with open(coco_file, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    # 建立image_id到image信息的映射
    image_map = {img['id']: img for img in coco_data['images']}
    category_map = {cat['id']: cat for cat in coco_data['categories']}
    
    # 按image分组annotations
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)
    
    for image_id, image_info in image_map.items():
        if image_id not in annotations_by_image:
            continue
        
        image_name = image_info['file_name']
        image_path = dataset_path / 'train' / image_name
        
        for ann in annotations_by_image[image_id]:
            category_id = ann['category_id']
            category = category_map.get(category_id)
            if category is None:
                continue
            
            sub_type = category['name']
            primary_class = get_primary_class(sub_type)
            if primary_class is None:
                continue
            
            # COCO格式bbox: [x, y, width, height]
            bbox = ann['bbox']
            
            instance_id = f"BDW_{image_name}_{ann['id']}"
            normalized_sub_type = normalize_sub_type(sub_type, primary_class)
            output_data = {
                "instance_id": instance_id,
                "taxonomy": {
                    "primary_class": primary_class,
                    "sub_type": normalized_sub_type,
                },
                "visual_features": {
                    "bbox": bbox
                },
                "image_path": str(image_path.relative_to(BASE_DIR))
            }
            
            output_file = OUTPUT_DIR / f"{instance_id}.json"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)


def process_cubit_det(dataset_path: Path):
    """处理cubit-det数据集：YOLO格式标注"""
    print(f"\n处理 Detection - cubit-det (YOLO格式)...")
    
    images_dir = dataset_path / 'images'
    labels_dir = dataset_path / 'labels'
    
    # 遍历所有子目录（train2017, val2017, test2017）
    for split_dir in images_dir.iterdir():
        if not split_dir.is_dir():
            continue
        
        split_name = split_dir.name
        label_split_dir = labels_dir / split_name
        
        if not label_split_dir.exists():
            continue
        
        for img_file in split_dir.glob('*.jpg'):
            label_file = label_split_dir / f"{img_file.stem}.txt"
            if not label_file.exists():
                continue
            
            img = cv2.imread(str(img_file))
            if img is None:
                continue
            img_h, img_w = img.shape[:2]
            
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            for idx, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                
                class_id = int(parts[0])
                
                # cubit-det只处理class_id 0和1，忽略2
                if class_id == 2:
                    continue
                
                # 类别映射：0 = crack, 1 = spalling
                class_name_map = {0: 'crack', 1: 'spalling'}
                sub_type = class_name_map.get(class_id)
                if sub_type is None:
                    continue  # 未知类别，跳过
                
                primary_class = get_primary_class(sub_type)
                if primary_class is None:
                    continue
                
                # YOLO格式: class_id x_center y_center width height (归一化)
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # 转换为绝对坐标 [x, y, width, height]
                x = int((x_center - width / 2) * img_w)
                y = int((y_center - height / 2) * img_h)
                w = int(width * img_w)
                h = int(height * img_h)
                
                instance_id = f"cubit-det_{split_name}_{img_file.stem}_{idx}"
                normalized_sub_type = normalize_sub_type(sub_type, primary_class)
                output_data = {
                    "instance_id": instance_id,
                    "taxonomy": {
                        "primary_class": primary_class,
                        "sub_type": normalized_sub_type,
                    },
                    "visual_features": {
                        "bbox": [x, y, w, h]
                    },
                    "image_path": str(img_file.relative_to(BASE_DIR))
                }
                
                output_file = OUTPUT_DIR / f"{instance_id}.json"
                output_file.parent.mkdir(parents=True, exist_ok=True)
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)


def process_mbdd2025(dataset_path: Path):
    """处理MBDD2025数据集：Pascal VOC XML格式"""
    print(f"\n处理 Detection - MBDD2025 (Pascal VOC格式)...")
    
    annotations_dir = dataset_path / 'Annotations'
    images_dir = dataset_path / 'JPEGImages'
    
    for xml_file in annotations_dir.glob('*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # 获取图片信息
        filename_elem = root.find('filename')
        if filename_elem is None:
            continue
        image_name = filename_elem.text
        image_path = images_dir / image_name
        
        size_elem = root.find('size')
        if size_elem is not None:
            width = int(size_elem.find('width').text)
            height = int(size_elem.find('height').text)
        else:
            img = cv2.imread(str(image_path))
            if img is None:
                continue
            height, width = img.shape[:2]
        
        # 处理所有object
        for idx, obj in enumerate(root.findall('object')):
            name_elem = obj.find('name')
            if name_elem is None:
                continue
            # MBDD2025的类别名称可能是小写，需要映射
            sub_type_raw = name_elem.text
            # 转换为小写并映射
            sub_type = sub_type_raw.lower()
            # 特殊映射：abscission -> Abscission (首字母大写)
            if sub_type == 'abscission':
                sub_type = 'Abscission'
            elif sub_type == 'bulge':
                sub_type = 'Bulge'
            primary_class = get_primary_class(sub_type)
            if primary_class is None:
                continue
            
            bbox_elem = obj.find('bndbox')
            if bbox_elem is None:
                continue

            # 有些标注是浮点数（如 "1179.92"），先按float解析再转为int
            xmin_f = float(bbox_elem.find('xmin').text)
            ymin_f = float(bbox_elem.find('ymin').text)
            xmax_f = float(bbox_elem.find('xmax').text)
            ymax_f = float(bbox_elem.find('ymax').text)

            # 转换为 [x, y, width, height]，用int截断到像素
            xmin = int(xmin_f)
            ymin = int(ymin_f)
            xmax = int(xmax_f)
            ymax = int(ymax_f)
            bbox = [xmin, ymin, xmax - xmin, ymax - ymin]
            
            instance_id = f"MBDD2025_{xml_file.stem}_{idx}"
            normalized_sub_type = normalize_sub_type(sub_type, primary_class)
            output_data = {
                "instance_id": instance_id,
                "taxonomy": {
                    "primary_class": primary_class,
                    "sub_type": normalized_sub_type,
                },
                "visual_features": {
                    "bbox": bbox
                },
                "image_path": str(image_path.relative_to(BASE_DIR))
            }
            
            output_file = OUTPUT_DIR / f"{instance_id}.json"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)


def mask_to_bbox_by_color(mask_path: Path) -> Dict[str, List[Tuple[int, int, int, int]]]:
    """从s2ds的彩色mask中按颜色提取不同类别的bbox"""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
    if mask is None:
        return {}
    
    # s2ds颜色编码映射：BGR格式（OpenCV使用BGR）
    # 根据README，颜色编码应该是：
    # - 白-Crack, 红-Spalling, 黄-Corrosion, 浅蓝-Efflorescence, 绿-Vegetation, 深蓝-Control Point, 黑-背景
    # 但实际测试发现mask可能是二值的，需要根据实际颜色值来判断
    color_to_class = {
        (255, 255, 255): 'crack',           # 白-Crack (RGB: 255,255,255 -> BGR: 255,255,255)
        (0, 0, 255): 'Spalling',            # 红-Spalling (RGB: 255,0,0 -> BGR: 0,0,255)
        (0, 255, 255): 'Corrosion',         # 黄-Corrosion (RGB: 255,255,0 -> BGR: 0,255,255)
        (255, 255, 0): 'Efflorescence',    # 浅蓝-Efflorescence (RGB: 0,255,255 -> BGR: 255,255,0)
        (0, 255, 0): 'Vegetation',          # 绿-Vegetation (RGB: 0,255,0 -> BGR: 0,255,0)
        (255, 0, 0): 'Control Point',       # 深蓝-Control Point (RGB: 0,0,255 -> BGR: 255,0,0)
    }
    
    result = {}
    for color_bgr, sub_type in color_to_class.items():
        # 创建该颜色的二值mask
        color_mask = cv2.inRange(mask, color_bgr, color_bgr)
        bboxes = mask_to_bbox_from_binary(color_mask)
        if bboxes:
            result[sub_type] = bboxes
    
    return result
def process_s2ds(dataset_path: Path):
    """处理s2ds数据集：mask文件名是"原图片名_lab.png"，按颜色编码提取多类别"""
    print(f"\n处理 Segmentation - s2ds...")
    
    for split_dir in ['train', 'val', 'test']:
        split_path = dataset_path / split_dir
        if not split_path.exists():
            continue
        
        for img_file in split_path.glob('*.png'):
            if img_file.name.endswith('_lab.png'):
                continue
            
            mask_file = split_path / f"{img_file.stem}_lab.png"
            if not mask_file.exists():
                continue
            
            # 按颜色提取不同类别的bbox
            bboxes_by_class = mask_to_bbox_by_color(mask_file)
            if not bboxes_by_class:
                continue
            
            for sub_type, bboxes in bboxes_by_class.items():
                primary_class = get_primary_class(sub_type)
                if primary_class is None:
                    continue
                
                for idx, bbox in enumerate(bboxes):
                    normalized_sub_type = normalize_sub_type(sub_type, primary_class)
                    instance_id = f"s2ds_{split_dir}_{img_file.stem}_{sub_type}_{idx}"
                    output_data = {
                        "instance_id": instance_id,
                        "taxonomy": {
                            "primary_class": primary_class,
                            "sub_type": normalized_sub_type,
                        },
                        "visual_features": {
                            "bbox": list(bbox)
                        },
                        "image_path": str(img_file.relative_to(BASE_DIR))
                    }
                    
                    output_file = OUTPUT_DIR / f"{instance_id}.json"
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(output_data, f, indent=2, ensure_ascii=False)


def process_bai2020(dataset_path: Path):
    """处理Bai-2020数据集：image在image目录，mask在label目录"""
    print(f"\n处理 Segmentation - Bai-2020...")
    
    image_dir = dataset_path / 'Data' / 'Object-and_structural-level_image&label' / 'image'
    label_dir = dataset_path / 'Data' / 'Object-and_structural-level_image&label' / 'label'
    
    if not image_dir.exists() or not label_dir.exists():
        print(f"  警告: 目录不存在")
        return
    
    sub_type = 'crack'
    primary_class = get_primary_class(sub_type)
    
    for img_file in image_dir.rglob('*'):
        if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
            continue
        
        # 查找对应的mask文件
        mask_file = label_dir / img_file.name
        if not mask_file.exists():
            # 尝试不同的扩展名
            for ext in ['.png', '.jpg', '.jpeg']:
                mask_file = label_dir / f"{img_file.stem}{ext}"
                if mask_file.exists():
                    break
            else:
                continue
        
        bboxes = mask_to_bbox(mask_file)
        if not bboxes:
            continue
        
        for idx, bbox in enumerate(bboxes):
            instance_id = f"Bai-2020_{img_file.stem}_{idx}"
            normalized_sub_type = normalize_sub_type(sub_type, primary_class)
            output_data = {
                "instance_id": instance_id,
                "taxonomy": {
                    "primary_class": primary_class,
                    "sub_type": normalized_sub_type,
                },
                "visual_features": {
                    "bbox": list(bbox)
                },
                "image_path": str(img_file.relative_to(BASE_DIR))
            }
            
            output_file = OUTPUT_DIR / f"{instance_id}.json"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)


def process_csd(dataset_path: Path):
    """处理CSD数据集：mask在masks目录"""
    print(f"\n处理 Segmentation - CSD...")
    
    sub_type = 'crack'  # CSD主要是crack相关
    primary_class = get_primary_class(sub_type)
    
    # CSD的目录结构是 train/images, train/masks, test/images, test/masks
    for split_dir in ['train', 'test']:
        split_path = dataset_path / split_dir
        if not split_path.exists():
            continue
        
        images_dir = split_path / 'images'
        masks_dir = split_path / 'masks'
        
        if not images_dir.exists() or not masks_dir.exists():
            continue
        
        for img_file in images_dir.rglob('*.jpg'):
            # 查找对应的mask文件（mask文件名可能与图片名相同或不同）
            mask_file = masks_dir / img_file.name
            if not mask_file.exists():
                # 尝试不同的扩展名
                for ext in ['.png', '.jpg', '.jpeg']:
                    mask_file = masks_dir / f"{img_file.stem}{ext}"
                    if mask_file.exists():
                        break
                else:
                    continue
            
            # 跳过noncrack的图片
            if mask_file.name.startswith('noncrack'):
                continue
            
            bboxes = mask_to_bbox(mask_file)
            if not bboxes:
                continue
            
            for idx, bbox in enumerate(bboxes):
                instance_id = f"CSD_{split_dir}_{img_file.stem}_{idx}"
                normalized_sub_type = normalize_sub_type(sub_type, primary_class)
                output_data = {
                    "instance_id": instance_id,
                    "taxonomy": {
                        "primary_class": primary_class,
                        "sub_type": normalized_sub_type,
                    },
                    "visual_features": {
                        "bbox": list(bbox)
                    },
                    "image_path": str(img_file.relative_to(BASE_DIR))
                }
                
                output_file = OUTPUT_DIR / f"{instance_id}.json"
                output_file.parent.mkdir(parents=True, exist_ok=True)
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)


def process_cubit_seg(dataset_path: Path):
    """处理cubit-seg数据集：mask文件名是原图片名.png（图片是.jpg）"""
    print(f"\n处理 Segmentation - cubit-seg...")
    
    crack_dir = dataset_path / 'crack_org'
    spalling_dir = dataset_path / 'spalling_org'
    
    for class_dir, sub_type in [(crack_dir, 'crack'), (spalling_dir, 'spalling')]:
        if not class_dir.exists():
            continue
        
        primary_class = get_primary_class(sub_type)
        if primary_class is None:
            continue
        
        for img_file in class_dir.glob('*.jpg'):
            mask_file = class_dir / f"{img_file.stem}.png"
            if not mask_file.exists():
                continue
            
            bboxes = mask_to_bbox(mask_file)
            if not bboxes:
                continue
            
            for idx, bbox in enumerate(bboxes):
                normalized_sub_type = normalize_sub_type(sub_type, primary_class)
                instance_id = f"cubit-seg_{sub_type}_{img_file.stem}_{idx}"
                output_data = {
                    "instance_id": instance_id,
                    "taxonomy": {
                        "primary_class": primary_class,
                        "sub_type": normalized_sub_type,
                    },
                    "visual_features": {
                        "bbox": list(bbox)
                    },
                    "image_path": str(img_file.relative_to(BASE_DIR))
                }
                
                output_file = OUTPUT_DIR / f"{instance_id}.json"
                output_file.parent.mkdir(parents=True, exist_ok=True)
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)


def process_deepcarck(dataset_path: Path):
    """处理deepcarck数据集：mask在train_lab和test_lab目录"""
    print(f"\n处理 Segmentation - deepcarck...")
    
    data_dir = dataset_path / 'Data'
    if not data_dir.exists():
        print(f"  警告: Data目录不存在")
        return
    
    sub_type = 'crack'
    primary_class = get_primary_class(sub_type)
    
    # deepcarck的目录结构是 train_img, train_lab, test_img, test_lab
    for split in ['train', 'test']:
        img_dir = data_dir / f"{split}_img"
        lab_dir = data_dir / f"{split}_lab"
        
        if not img_dir.exists() or not lab_dir.exists():
            continue
        
        for img_file in img_dir.glob('*.jpg'):
            mask_file = lab_dir / f"{img_file.stem}.png"
            if not mask_file.exists():
                continue
            
            bboxes = mask_to_bbox(mask_file)
            if not bboxes:
                continue
            
            for idx, bbox in enumerate(bboxes):
                instance_id = f"deepcarck_{split}_{img_file.stem}_{idx}"
                normalized_sub_type = normalize_sub_type(sub_type, primary_class)
                output_data = {
                    "instance_id": instance_id,
                    "taxonomy": {
                        "primary_class": primary_class,
                        "sub_type": normalized_sub_type,
                    },
                    "visual_features": {
                        "bbox": list(bbox)
                    },
                    "image_path": str(img_file.relative_to(BASE_DIR))
                }
                
                output_file = OUTPUT_DIR / f"{instance_id}.json"
                output_file.parent.mkdir(parents=True, exist_ok=True)
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)


def process_masonry(dataset_path: Path):
    """处理masonry数据集：mask在crack_detection_224_masks目录"""
    print(f"\n处理 Segmentation - masonry...")
    
    images_dir = dataset_path / 'Data' / 'crack_detection_224_images'
    masks_dir = dataset_path / 'Data' / 'crack_detection_224_masks'
    
    if not images_dir.exists() or not masks_dir.exists():
        print(f"  警告: 目录不存在")
        return
    
    sub_type = 'crack'
    primary_class = get_primary_class(sub_type)
    
    for img_file in images_dir.glob('*.png'):
        mask_file = masks_dir / img_file.name
        if not mask_file.exists():
            continue
        
        bboxes = mask_to_bbox(mask_file)
        if not bboxes:
            continue
        
        for idx, bbox in enumerate(bboxes):
            instance_id = f"masonry_{img_file.stem}_{idx}"
            normalized_sub_type = normalize_sub_type(sub_type, primary_class)
            output_data = {
                "instance_id": instance_id,
                "taxonomy": {
                    "primary_class": primary_class,
                    "sub_type": normalized_sub_type,
                },
                "visual_features": {
                    "bbox": list(bbox)
                },
                "image_path": str(img_file.relative_to(BASE_DIR))
            }
            
            output_file = OUTPUT_DIR / f"{instance_id}.json"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)


def process_uav75(dataset_path: Path):
    """处理uav75数据集：test_img/test_lab, train_img/train_lab配对"""
    print(f"\n处理 Segmentation - uav75...")
    
    data_dir = dataset_path / 'Data'
    if not data_dir.exists():
        print(f"  警告: Data目录不存在")
        return
    
    sub_type = 'crack'
    primary_class = get_primary_class(sub_type)
    
    for split in ['test', 'train']:
        img_dir = data_dir / f"{split}_img"
        lab_dir = data_dir / f"{split}_lab"
        
        if not img_dir.exists() or not lab_dir.exists():
            continue
        
        for img_file in img_dir.glob('*.jpg'):
            # 查找对应的mask文件（可能是.png）
            mask_file = lab_dir / f"{img_file.stem}.png"
            if not mask_file.exists():
                continue
            
            bboxes = mask_to_bbox(mask_file)
            if not bboxes:
                continue
            
            for idx, bbox in enumerate(bboxes):
                instance_id = f"uav75_{split}_{img_file.stem}_{idx}"
                normalized_sub_type = normalize_sub_type(sub_type, primary_class)
                output_data = {
                    "instance_id": instance_id,
                    "taxonomy": {
                        "primary_class": primary_class,
                        "sub_type": normalized_sub_type,
                    },
                    "visual_features": {
                        "bbox": list(bbox)
                    },
                    "image_path": str(img_file.relative_to(BASE_DIR))
                }
                
                output_file = OUTPUT_DIR / f"{instance_id}.json"
                output_file.parent.mkdir(parents=True, exist_ok=True)
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)


def main():
    # 命令行参数：可以只处理指定数据集，避免已完成的数据集重复跑
    parser = argparse.ArgumentParser(description="统一各数据集bbox标注格式")
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help="只处理指定数据集，逗号分隔，名称为数据集目录名，例如："
             "HS-23K,BD3,CCIC,BDD,BDW,cubit-det,MBDD2025,"
             "s2ds,Bai-2020,CSD,cubit-seg,deepcarck,masonry,uav75",
    )
    args = parser.parse_args()

    only_set: Optional[Set[str]] = None
    if args.only:
        only_set = {name.strip() for name in args.only.split(",") if name.strip()}
        if not only_set:
            only_set = None

    OUTPUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_LABEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_MASK_DIR.mkdir(parents=True, exist_ok=True)

    print("Start bbox label unification...")
    print(f"Output images dir: {OUTPUT_IMAGE_DIR}")
    print(f"Output labels dir: {OUTPUT_LABEL_DIR}")
    print(f"Output masks dir: {OUTPUT_MASK_DIR}")

    if not BASE_DIR or str(BASE_DIR) == "." or not BASE_DIR.exists():
        print("Raw input path is not configured. Set RAW_DATA_ROOT before running actual conversion.")
        return
    if only_set:
        print(f"  仅处理数据集: {sorted(only_set)}")
    
    # 处理Classification数据集
    classification_dir = BASE_DIR / 'classification'
    if classification_dir.exists():
        for dataset_dir in classification_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            dataset_name = dataset_dir.name
            if only_set and dataset_name not in only_set:
                continue
            process_classification_dataset(dataset_name, dataset_dir)
    
    # 处理Detection数据集
    detection_dir = BASE_DIR / 'detection'
    if detection_dir.exists():
        # BDW
        bdw_path = detection_dir / 'BDW'
        if bdw_path.exists():
            if (not only_set) or ('BDW' in only_set):
                process_bdw(bdw_path)
        
        # cubit-det
        cubit_det_path = detection_dir / 'cubit-det'
        if cubit_det_path.exists():
            if (not only_set) or ('cubit-det' in only_set):
                process_cubit_det(cubit_det_path)
        
        # MBDD2025
        mbdd_path = detection_dir / 'MBDD2025'
        if mbdd_path.exists():
            if (not only_set) or ('MBDD2025' in only_set):
                process_mbdd2025(mbdd_path)
    
    # 处理Segmentation数据集
    segmentation_dir = BASE_DIR / 'segmentation'
    if segmentation_dir.exists():
        # s2ds
        s2ds_path = segmentation_dir / 's2ds'
        if s2ds_path.exists():
            if (not only_set) or ('s2ds' in only_set):
                process_s2ds(s2ds_path)
        
        # Bai-2020
        bai_path = segmentation_dir / 'Bai-2020'
        if bai_path.exists():
            if (not only_set) or ('Bai-2020' in only_set):
                process_bai2020(bai_path)
        
        # CSD
        csd_path = segmentation_dir / 'CSD'
        if csd_path.exists():
            if (not only_set) or ('CSD' in only_set):
                process_csd(csd_path)
        
        # cubit-seg
        cubit_seg_path = segmentation_dir / 'cubit-seg'
        if cubit_seg_path.exists():
            if (not only_set) or ('cubit-seg' in only_set):
                process_cubit_seg(cubit_seg_path)
        
        # deepcarck
        deepcarck_path = segmentation_dir / 'deepcarck'
        if deepcarck_path.exists():
            if (not only_set) or ('deepcarck' in only_set):
                process_deepcarck(deepcarck_path)
        
        # masonry
        masonry_path = segmentation_dir / 'masonry'
        if masonry_path.exists():
            if (not only_set) or ('masonry' in only_set):
                process_masonry(masonry_path)
        
        # uav75
        uav75_path = segmentation_dir / 'uav75'
        if uav75_path.exists():
            if (not only_set) or ('uav75' in only_set):
                process_uav75(uav75_path)
    
    print(f"\n完成！统一后的标注已保存到: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()

