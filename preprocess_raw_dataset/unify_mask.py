#!/usr/bin/env python3
"""
统一生成所有数据集的彩色 mask。

功能：
1. 对于有 mask 的 segmentation 数据集：读取原始 mask，按类别转换为统一颜色
2. 对于没有 mask 的 classification/detection 数据集：从 bbox-label 读取 bbox，使用 SAM 生成 mask

颜色映射（与 view_topk_images.py 一致）：
- Crack: (255, 0, 0) 红
- Material_loss: (255, 140, 0) 橙
- Stain: (30, 144, 255) 蓝
- External Fixings: (0, 200, 0) 绿
"""

import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import argparse

# Optional service imports from defect_bench annotation toolkit backend.
# Keep best-effort behavior for preprocessing scripts.
try:
    from sam_logic import sam_service
    from crack_service import crack_service
except ImportError as e:
    print(f"Warning: Could not import services: {e}")
    print("SAM and crack services will not be available for mask generation.")
    sam_service = None
    crack_service = None

# 颜色映射：primary_class -> RGB 颜色
CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
    "Crack": (255, 0, 0),             # 红
    "Material_loss": (255, 140, 0),   # 橙
    "Stain": (30, 144, 255),          # 蓝
    "External Fixings": (0, 200, 0),  # 绿
}

# Input root (set by user when preprocessing raw datasets).
RAW_DATA_ROOT = Path("")
BASE_DIR = RAW_DATA_ROOT

# Fixed outputs inside defect_bench/data_sample
OUTPUT_IMAGE_DIR = Path("defect_bench/data_sample/images")
OUTPUT_LABEL_DIR = Path("defect_bench/data_sample/labels")
OUTPUT_MASK_DIR = Path("defect_bench/data_sample/masks")

# s2ds 颜色编码映射（BGR格式，因为 OpenCV 使用 BGR）
S2DS_COLOR_TO_CLASS = {
    (255, 255, 255): 'crack',           # 白 -> Crack
    (0, 0, 255): 'Spalling',            # 红 -> Material_loss
    (0, 255, 255): 'Corrosion',         # 黄 -> Stain
    (255, 255, 0): 'Efflorescence',    # 浅蓝 -> Stain
    (0, 255, 0): 'Vegetation',          # 绿 -> External Fixings
    (255, 0, 0): 'Control Point',       # 深蓝 -> (未映射，跳过)
}

# 标签映射：sub_type -> primary_class（与 unify_bbox_labels.py 一致）
LABEL_MAPPING = {
    'major_crack': 'Crack',
    'minor_crack': 'Crack',
    'crack': 'Crack',
    'stairstep_crack': 'Crack',
    'cracks': 'Crack',
    'peeling': 'Material_loss',
    'spalling': 'Material_loss',
    'Spalling': 'Material_loss',
    'flakes': 'Material_loss',
    'peeling_paint': 'Material_loss',
    'Abscission': 'Material_loss',
    'abscission': 'Material_loss',
    'Bulge': 'Material_loss',
    'bulge': 'Material_loss',
    'Material_loss': 'Material_loss',
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
    'human_caused_damages': 'External Fixings',
    'Vegetation': 'External Fixings',
}


def get_primary_class(sub_type: str) -> Optional[str]:
    """根据 sub_type 获取 primary_class"""
    return LABEL_MAPPING.get(sub_type)


def create_bbox_mask(h: int, w: int, bboxes: List[List[float]]) -> np.ndarray:
    """
    根据一组 [x1, y1, x2, y2] bbox 生成二值 mask，用于限制 SAM 结果不超过 bbox 范围。
    参考 backend 中 SAMService.predict_bboxes 的实现。
    """
    bbox_mask = np.zeros((h, w), dtype=np.uint8)
    if not bboxes:
        return bbox_mask
    
    for bbox in bboxes:
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = map(int, bbox)
        # 裁剪到图像范围内
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        if x2 > x1 and y2 > y1:
            bbox_mask[y1:y2, x1:x2] = 255
    return bbox_mask


def convert_binary_mask_to_colored(mask: np.ndarray, primary_class: str) -> np.ndarray:
    """将二值 mask 转换为彩色 mask"""
    color = CLASS_COLORS.get(primary_class, (128, 128, 128))  # 默认灰色
    colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    mask_bool = mask > 0
    colored_mask[mask_bool] = color
    return colored_mask


def process_s2ds_mask(mask_path: Path) -> Optional[np.ndarray]:
    """处理 s2ds 的彩色 mask，转换为统一颜色"""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
    if mask is None:
        return None
    
    h, w = mask.shape[:2]
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for color_bgr, sub_type in S2DS_COLOR_TO_CLASS.items():
        # 创建该颜色的二值 mask
        color_mask = cv2.inRange(mask, color_bgr, color_bgr)
        if np.any(color_mask > 0):
            primary_class = get_primary_class(sub_type)
            if primary_class and primary_class in CLASS_COLORS:
                color = CLASS_COLORS[primary_class]
                colored_mask[color_mask > 0] = color
    
    return colored_mask


def process_binary_mask_dataset(mask_path: Path, primary_class: str) -> Optional[np.ndarray]:
    """处理二值 mask 数据集，转换为彩色"""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    
    return convert_binary_mask_to_colored(mask, primary_class)


def load_bbox_annotations_for_image(image_path: str) -> List[Dict]:
    """从 bbox-label 目录加载指定图片的所有标注"""
    annotations = []
    image_path_normalized = image_path.replace("\\", "/")
    
    for json_path in BBOX_LABEL_DIR.glob("*.json"):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        
        json_image_path = str(data.get("image_path", "")).replace("\\", "/")
        if json_image_path != image_path_normalized:
            continue
        
        bbox = data.get("visual_features", {}).get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        
        taxonomy = data.get("taxonomy", {})
        annotations.append({
            "bbox": bbox,  # [x, y, w, h]
            "primary_class": taxonomy.get("primary_class"),
            "sub_type": taxonomy.get("sub_type"),
        })
    
    return annotations


def generate_mask_with_sam(image_np: np.ndarray, annotations: List[Dict]) -> np.ndarray:
    """使用 SAM 和 crack_service 为没有 mask 的图片生成彩色 mask"""
    h, w = image_np.shape[:2]
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    if not sam_service or not sam_service.initialized:
        print("  Warning: SAM service not initialized, skipping mask generation")
        return colored_mask
    
    # 按 primary_class 分组
    annotations_by_class = defaultdict(list)
    for ann in annotations:
        primary_class = ann.get("primary_class")
        if primary_class and primary_class in CLASS_COLORS:
            annotations_by_class[primary_class].append(ann)
    
    # 处理 Crack 类别：使用 crack_service + 三种 SAM
    if "Crack" in annotations_by_class:
        crack_annos = annotations_by_class["Crack"]
        crack_mask = generate_crack_mask(image_np, crack_annos)
        if np.any(crack_mask > 0):
            color = CLASS_COLORS["Crack"]
            colored_mask[crack_mask > 0] = color
    
    # 处理其他类别：使用 SAM with bbox + text
    for primary_class, annos in annotations_by_class.items():
        if primary_class == "Crack":
            continue  # 已经处理过了
        
        color = CLASS_COLORS[primary_class]
        # 针对该 primary_class 的临时二值 mask，用于在上色前做 bbox 裁剪
        class_mask_binary = np.zeros((h, w), dtype=np.uint8)
        
        # 准备 bboxes 和 text prompts
        bboxes: List[List[int]] = []
        text_list: List[str] = []
        
        for ann in annos:
            x, y, w_bbox, h_bbox = ann["bbox"]
            # 转换为 [x1, y1, x2, y2] 格式
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w_bbox), int(y + h_bbox)
            bboxes.append([x1, y1, x2, y2])
            # 直接使用 primary_class 作为 text prompt（不转换）
            text_list.append(primary_class)
        
        if not bboxes:
            continue
        
        # 使用 SAM 生成 mask
        try:
            sam_service.predictor.set_image(image_np)
            # 同时使用 bboxes 和 text
            results = sam_service.predictor(bboxes=bboxes, text=text_list)
            
            if results and len(results) > 0:
                for result in results:
                    if result.masks is not None:
                        # 提取 mask
                        if hasattr(result.masks, "data"):
                            masks = result.masks.data.cpu().numpy()
                        elif hasattr(result.masks, "cpu"):
                            masks = result.masks.cpu().numpy()
                        else:
                            masks = np.array(result.masks)
                        
                        for m in masks:
                            # 调整大小
                            if m.shape[:2] != (h, w):
                                m = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
                            
                            # 转换为二值 mask，并累积到该类别的 mask 中
                            m_binary = (m > 0.5).astype(np.uint8) * 255
                            class_mask_binary = cv2.bitwise_or(class_mask_binary, m_binary)
        except Exception as e:
            print(f"  Error generating SAM mask for {primary_class}: {e}")
            continue
        
        # 使用 bbox 对该类别的 SAM mask 做后处理，限制在 bbox 范围内
        if bboxes:
            bbox_mask = create_bbox_mask(h, w, bboxes)
            class_mask_binary = cv2.bitwise_and(class_mask_binary, bbox_mask)
        
        # 上色到总的彩色 mask 中
        colored_mask[class_mask_binary > 0] = color
    
    return colored_mask


def generate_crack_mask(image_np: np.ndarray, crack_annotations: List[Dict]) -> np.ndarray:
    """为 Crack 类别生成 mask：使用 crack_service + 三种 SAM"""
    h, w = image_np.shape[:2]
    crack_mask = np.zeros((h, w), dtype=np.uint8)
    
    if not crack_service:
        print("  Warning: crack_service not available")
        return crack_mask
    
    # 1. 使用 crack_service 处理 crack bboxes
    for ann in crack_annotations:
        x, y, w_bbox, h_bbox = ann["bbox"]
        x1, y1 = max(0, int(x)), max(0, int(y))
        x2, y2 = min(w, int(x + w_bbox)), min(h, int(y + h_bbox))
        
        if x2 <= x1 or y2 <= y1:
            continue
        
        crop = image_np[y1:y2, x1:x2]
        
        try:
            crop_mask = crack_service.predict(crop, mode='union')
            if crop_mask.shape[:2] != (y2 - y1, x2 - x1):
                crop_mask = cv2.resize(crop_mask, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
            
            mask_region = crack_mask[y1:y2, x1:x2]
            crack_mask[y1:y2, x1:x2] = cv2.bitwise_or(mask_region, crop_mask)
        except Exception as e:
            print(f"  Error in crack_service for bbox [{x1},{y1},{x2},{y2}]: {e}")
            continue
    
    # 2. 使用三种 SAM 方法增强（可选，这里简化处理，只使用 bbox SAM）
    if sam_service and sam_service.initialized:
        crack_bboxes: List[List[int]] = []
        for ann in crack_annotations:
            x, y, w_bbox, h_bbox = ann["bbox"]
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w_bbox), int(y + h_bbox)
            crack_bboxes.append([x1, y1, x2, y2])
        
        if crack_bboxes:
            try:
                sam_service.predictor.set_image(image_np)
                results = sam_service.predictor(bboxes=crack_bboxes)
                
                if results and len(results) > 0:
                    for result in results:
                        if result.masks is not None:
                            if hasattr(result.masks, "data"):
                                masks = result.masks.data.cpu().numpy()
                            elif hasattr(result.masks, "cpu"):
                                masks = result.masks.cpu().numpy()
                            else:
                                masks = np.array(result.masks)
                            
                            for m in masks:
                                if m.shape[:2] != (h, w):
                                    m = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
                                m_binary = (m > 0.5).astype(np.uint8) * 255
                                crack_mask = cv2.bitwise_or(crack_mask, m_binary)
            except Exception as e:
                print(f"  Error in SAM for crack: {e}")
    
    # 最后一步：对 crack 的最终 mask 做 bbox 限制，确保 crack 不会超出标注 bbox 范围
    if crack_annotations:
        crack_bboxes_xyxy: List[List[int]] = []
        for ann in crack_annotations:
            x, y, w_bbox, h_bbox = ann["bbox"]
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w_bbox), int(y + h_bbox)
            crack_bboxes_xyxy.append([x1, y1, x2, y2])
        
        bbox_mask = create_bbox_mask(h, w, crack_bboxes_xyxy)
        crack_mask = cv2.bitwise_and(crack_mask, bbox_mask)
    
    return crack_mask


def process_segmentation_datasets():
    """处理有 mask 的 segmentation 数据集"""
    segmentation_dir = BASE_DIR / "segmentation"
    if not segmentation_dir.exists():
        return
    
    print("\n" + "=" * 80)
    print("处理 Segmentation 数据集（有原始 mask）")
    print("=" * 80)
    
    # s2ds
    s2ds_path = segmentation_dir / "s2ds"
    if s2ds_path.exists():
        print("\n处理 s2ds...")
        count = 0
        for split_dir in ['train', 'val', 'test']:
            split_path = s2ds_path / split_dir
            if not split_path.exists():
                continue
            
            for img_file in split_path.glob('*.png'):
                if img_file.name.endswith('_lab.png'):
                    continue
                
                mask_file = split_path / f"{img_file.stem}_lab.png"
                if not mask_file.exists():
                    continue
                
                colored_mask = process_s2ds_mask(mask_file)
                if colored_mask is not None and np.any(colored_mask > 0):
                    # 保存到输出目录
                    rel_path = img_file.relative_to(BASE_DIR)
                    output_path = OUTPUT_MASK_DIR / rel_path.parent / f"{img_file.stem}_mask.png"
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(output_path), cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR))
                    count += 1
        
        print(f"  处理了 {count} 张图片")
    
    # Bai-2020
    bai_path = segmentation_dir / "Bai-2020"
    if bai_path.exists():
        print("\n处理 Bai-2020...")
        count = 0
        image_dir = bai_path / 'Data' / 'Object-and_structural-level_image&label' / 'image'
        label_dir = bai_path / 'Data' / 'Object-and_structural-level_image&label' / 'label'
        
        if image_dir.exists() and label_dir.exists():
            for img_file in image_dir.rglob('*'):
                if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                    continue
                
                mask_file = label_dir / img_file.name
                if not mask_file.exists():
                    for ext in ['.png', '.jpg', '.jpeg']:
                        mask_file = label_dir / f"{img_file.stem}{ext}"
                        if mask_file.exists():
                            break
                    else:
                        continue
                
                colored_mask = process_binary_mask_dataset(mask_file, "Crack")
                if colored_mask is not None and np.any(colored_mask > 0):
                    rel_path = img_file.relative_to(BASE_DIR)
                    output_path = OUTPUT_MASK_DIR / rel_path.parent / f"{img_file.stem}_mask.png"
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(output_path), cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR))
                    count += 1
        
        print(f"  处理了 {count} 张图片")
    
    # CSD
    csd_path = segmentation_dir / "CSD"
    if csd_path.exists():
        print("\n处理 CSD...")
        count = 0
        for split_dir in ['train', 'test']:
            split_path = csd_path / split_dir
            if not split_path.exists():
                continue
            
            images_dir = split_path / 'images'
            masks_dir = split_path / 'masks'
            
            if not images_dir.exists() or not masks_dir.exists():
                continue
            
            for img_file in images_dir.rglob('*.jpg'):
                mask_file = masks_dir / img_file.name
                if not mask_file.exists():
                    for ext in ['.png', '.jpg', '.jpeg']:
                        mask_file = masks_dir / f"{img_file.stem}{ext}"
                        if mask_file.exists():
                            break
                    else:
                        continue
                
                if mask_file.name.startswith('noncrack'):
                    continue
                
                colored_mask = process_binary_mask_dataset(mask_file, "Crack")
                if colored_mask is not None and np.any(colored_mask > 0):
                    rel_path = img_file.relative_to(BASE_DIR)
                    output_path = OUTPUT_MASK_DIR / rel_path.parent / f"{img_file.stem}_mask.png"
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(output_path), cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR))
                    count += 1
        
        print(f"  处理了 {count} 张图片")
    
    # cubit-seg
    cubit_seg_path = segmentation_dir / "cubit-seg"
    if cubit_seg_path.exists():
        print("\n处理 cubit-seg...")
        count = 0
        for class_dir, sub_type in [('crack_org', 'crack'), ('spalling_org', 'spalling')]:
            class_path = cubit_seg_path / class_dir
            if not class_path.exists():
                continue
            
            primary_class = get_primary_class(sub_type)
            if not primary_class:
                continue
            
            for img_file in class_path.glob('*.jpg'):
                mask_file = class_path / f"{img_file.stem}.png"
                if not mask_file.exists():
                    continue
                
                colored_mask = process_binary_mask_dataset(mask_file, primary_class)
                if colored_mask is not None and np.any(colored_mask > 0):
                    rel_path = img_file.relative_to(BASE_DIR)
                    output_path = OUTPUT_MASK_DIR / rel_path.parent / f"{img_file.stem}_mask.png"
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(output_path), cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR))
                    count += 1
        
        print(f"  处理了 {count} 张图片")
    
    # deepcarck
    deepcarck_path = segmentation_dir / "deepcarck"
    if deepcarck_path.exists():
        print("\n处理 deepcarck...")
        count = 0
        data_dir = deepcarck_path / 'Data'
        if data_dir.exists():
            for split in ['train', 'test']:
                img_dir = data_dir / f"{split}_img"
                lab_dir = data_dir / f"{split}_lab"
                
                if not img_dir.exists() or not lab_dir.exists():
                    continue
                
                for img_file in img_dir.glob('*.jpg'):
                    mask_file = lab_dir / f"{img_file.stem}.png"
                    if not mask_file.exists():
                        continue
                    
                    colored_mask = process_binary_mask_dataset(mask_file, "Crack")
                    if colored_mask is not None and np.any(colored_mask > 0):
                        rel_path = img_file.relative_to(BASE_DIR)
                        output_path = OUTPUT_MASK_DIR / rel_path.parent / f"{img_file.stem}_mask.png"
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(output_path), cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR))
                        count += 1
        
        print(f"  处理了 {count} 张图片")
    
    # masonry
    masonry_path = segmentation_dir / "masonry"
    if masonry_path.exists():
        print("\n处理 masonry...")
        count = 0
        images_dir = masonry_path / 'Data' / 'crack_detection_224_images'
        masks_dir = masonry_path / 'Data' / 'crack_detection_224_masks'
        
        if images_dir.exists() and masks_dir.exists():
            for img_file in images_dir.glob('*.png'):
                mask_file = masks_dir / img_file.name
                if not mask_file.exists():
                    continue
                
                colored_mask = process_binary_mask_dataset(mask_file, "Crack")
                if colored_mask is not None and np.any(colored_mask > 0):
                    rel_path = img_file.relative_to(BASE_DIR)
                    output_path = OUTPUT_MASK_DIR / rel_path.parent / f"{img_file.stem}_mask.png"
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(output_path), cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR))
                    count += 1
        
        print(f"  处理了 {count} 张图片")
    
    # uav75
    uav75_path = segmentation_dir / "uav75"
    if uav75_path.exists():
        print("\n处理 uav75...")
        count = 0
        data_dir = uav75_path / 'Data'
        if data_dir.exists():
            for split in ['test', 'train']:
                img_dir = data_dir / f"{split}_img"
                lab_dir = data_dir / f"{split}_lab"
                
                if not img_dir.exists() or not lab_dir.exists():
                    continue
                
                for img_file in img_dir.glob('*.jpg'):
                    mask_file = lab_dir / f"{img_file.stem}.png"
                    if not mask_file.exists():
                        continue
                    
                    colored_mask = process_binary_mask_dataset(mask_file, "Crack")
                    if colored_mask is not None and np.any(colored_mask > 0):
                        rel_path = img_file.relative_to(BASE_DIR)
                        output_path = OUTPUT_MASK_DIR / rel_path.parent / f"{img_file.stem}_mask.png"
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(output_path), cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR))
                        count += 1
        
        print(f"  处理了 {count} 张图片")


def process_classification_detection_datasets():
    """处理没有 mask 的 classification 和 detection 数据集，使用 SAM 生成"""
    print("\n" + "=" * 80)
    print("处理 Classification/Detection 数据集（使用 SAM 生成 mask）")
    print("=" * 80)
    
    if not sam_service or not sam_service.initialized:
        print("SAM service not available, skipping classification/detection datasets")
        return
    
    # 收集所有需要处理的图片（从 bbox-label 中）
    image_to_annotations = defaultdict(list)
    
    print("\n加载 bbox-label 标注...")
    for json_path in BBOX_LABEL_DIR.glob("*.json"):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        
        image_path = str(data.get("image_path", "")).replace("\\", "/")
        if not image_path:
            continue
        
        # 检查是否来自 classification 或 detection
        if "classification" not in image_path and "detection" not in image_path:
            continue
        
        bbox = data.get("visual_features", {}).get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        
        taxonomy = data.get("taxonomy", {})
        image_to_annotations[image_path].append({
            "bbox": bbox,
            "primary_class": taxonomy.get("primary_class"),
            "sub_type": taxonomy.get("sub_type"),
        })
    
    print(f"找到 {len(image_to_annotations)} 张需要处理的图片")
    
    # 处理每张图片
    count = 0
    for image_path_str, annotations in image_to_annotations.items():
        image_path = BASE_DIR / image_path_str.lstrip("/\\")
        
        if not image_path.exists():
            continue
        
        try:
            # 读取图片
            img_bgr = cv2.imread(str(image_path))
            if img_bgr is None:
                continue
            
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            # 生成彩色 mask
            colored_mask = generate_mask_with_sam(img_rgb, annotations)
            
            if np.any(colored_mask > 0):
                # 保存 mask
                rel_path = image_path.relative_to(BASE_DIR)
                output_path = OUTPUT_MASK_DIR / rel_path.parent / f"{image_path.stem}_mask.png"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(output_path), cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR))
                count += 1
                
                if count % 10 == 0:
                    print(f"  已处理 {count} 张图片...")
        
        except Exception as e:
            print(f"  Error processing {image_path}: {e}")
            continue
    
    print(f"\n处理了 {count} 张图片")


def main():
    parser = argparse.ArgumentParser(description="统一生成所有数据集的彩色 mask")
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help="只处理指定类型：segmentation（有mask）或 classification_detection（无mask），或 all（默认）",
    )
    args = parser.parse_args()
    
    OUTPUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_LABEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_MASK_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Output images dir: {OUTPUT_IMAGE_DIR}")
    print(f"Output labels dir: {OUTPUT_LABEL_DIR}")
    print(f"Output masks dir: {OUTPUT_MASK_DIR}")

    if not BASE_DIR or str(BASE_DIR) == "." or not BASE_DIR.exists():
        print("Raw input path is not configured. Set RAW_DATA_ROOT before running actual conversion.")
        return
    
    # 初始化服务
    if sam_service:
        print("初始化 SAM 服务...")
        sam_service.initialize()
        if not sam_service.initialized:
            print("警告: SAM 服务初始化失败，classification/detection 数据集将无法处理")
    
    if crack_service:
        print("初始化 Crack 服务...")
        crack_service.initialize()
        print("Crack 服务初始化成功")
    
    # 处理数据集
    only_type = args.only.lower() if args.only else "all"
    
    if only_type in ["segmentation", "all"]:
        process_segmentation_datasets()
    
    if only_type in ["classification_detection", "all"]:
        process_classification_detection_datasets()
    
    print("\n" + "=" * 80)
    print(f"Done. Unified masks are saved to: {OUTPUT_MASK_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()

