#!/usr/bin/env python3
"""
对所有数据集进行图片筛选（太小、太模糊）和分类（patch-level vs bbox-level）
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import json
import pandas as pd
from typing import Tuple, List

import torch
from PIL import Image
import clip
import timm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

# 支持的图片格式
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

# 每个 level 要保留的 Top-K 数量
TOPK_PATCH = 5000
TOPK_BBOX = 5000

# 是否启用 CLIP 语义筛选（删除明显不是户外墙面的图片）
USE_CLIP_FILTER = True
CLIP_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# CLIP 语义筛选的 prompt 设计：
# - 正向：包含“墙 + 户外/天空”的描述
# - 反向：室内、人、车等与目标无关的场景
CLIP_POSITIVE_PROMPTS = [
    "outdoor building facade with wall and sky",
    "outdoor masonry or concrete wall with cracks or defects",
    "exterior wall of a building with visible texture",
]

CLIP_NEGATIVE_PROMPTS = [
    # 非房屋建筑或与建筑立面无关的结构/场景
    "concrete bridge pier or bridge column structure",
    "bridge deck or road pavement without building facades",
    "highway or road surface with guardrails and vehicles but no building walls",
    "tunnel interior or underground passage without building facades",
    "large industrial plant structure or steel frame without clear wall surfaces",
]

_clip_model = None
_clip_preprocess = None
_clip_text_tokens = None
_clip_pos_len = None
_dino_model = None
_dino_transform = None


def _init_clip():
    """惰性加载 CLIP 模型和文本特征，仅在需要语义筛选时调用。"""
    global _clip_model, _clip_preprocess, _clip_text_tokens, _clip_pos_len
    if _clip_model is not None:
        return _clip_model, _clip_preprocess, _clip_text_tokens, _clip_pos_len

    print(f"Loading CLIP ViT-B/32 on {CLIP_DEVICE} for semantic filtering...")
    model, preprocess = clip.load("ViT-B/32", device=CLIP_DEVICE)
    all_prompts = CLIP_POSITIVE_PROMPTS + CLIP_NEGATIVE_PROMPTS
    text_tokens = clip.tokenize(all_prompts).to(CLIP_DEVICE)

    _clip_model = model
    _clip_preprocess = preprocess
    _clip_text_tokens = text_tokens
    _clip_pos_len = len(CLIP_POSITIVE_PROMPTS)
    return _clip_model, _clip_preprocess, _clip_text_tokens, _clip_pos_len


def clip_semantic_scores(image_path: str) -> Tuple[float, float]:
    """
    使用 CLIP 计算图像与正向/反向 prompt 的概率之和。
    返回 (positive_score, negative_score)。
    """
    try:
        model, preprocess, text_tokens, pos_len = _init_clip()
        img = Image.open(image_path).convert("RGB")
        img_input = preprocess(img).unsqueeze(0).to(CLIP_DEVICE)
        with torch.no_grad():
            logits_per_image, _ = model(img_input, text_tokens)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

        pos_score = float(probs[:pos_len].sum())
        neg_score = float(probs[pos_len:].sum())
        return pos_score, neg_score
    except Exception as e:
        print(f"[CLIP] Error processing {image_path}: {e}")
        # 出错时当作语义不可靠，返回较差的分数
        return 0.0, 1.0


# ----------------------- DINO 特征与多样性评分 ----------------------- #

DINO_MODEL_NAME = 'vit_base_patch16_dinov3_qkvb'
DINO_DEVICE = CLIP_DEVICE  # 复用同一设备
LAMBDA_DIVERSITY = 0.3  # 多样性在质量分中的权重


def _init_dino():
    """惰性加载 DINO 模型和预处理，用于全局特征提取。"""
    global _dino_model, _dino_transform
    if _dino_model is not None:
        return _dino_model, _dino_transform

    print(f"Loading DINO model '{DINO_MODEL_NAME}' on {DINO_DEVICE} for diversity scoring...")
    try:
        model = timm.create_model(DINO_MODEL_NAME, pretrained=True, num_classes=0)
    except Exception as e:
        print(f"[DINO] Failed to load {DINO_MODEL_NAME}: {e}")
        # 尝试使用常见的 DINOv2 作为备选
        fallback = 'vit_base_patch14_dinov2.lvd142m'
        print(f"[DINO] Fallback to {fallback}")
        try:
            model = timm.create_model(fallback, pretrained=True, num_classes=0)
        except Exception as e2:
            print(f"[DINO] Fallback model load failed: {e2}")
            return None, None

    model.to(DINO_DEVICE)
    model.eval()
    data_config = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_config, is_training=False)

    _dino_model = model
    _dino_transform = transform
    return _dino_model, _dino_transform


def extract_dino_features(paths: List[str]) -> np.ndarray:
    """
    为一组图像路径提取 DINO 特征（L2 归一化），返回 shape = (N, D) 的 numpy 数组。
    如果加载失败，返回空数组。
    """
    model, transform = _init_dino()
    if model is None or transform is None or not paths:
        return np.zeros((0, 0), dtype=np.float32)

    feats_list = []
    batch_size = 16
    with torch.no_grad():
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i:i + batch_size]
            batch_imgs = []
            valid_idx = []
            for j, p in enumerate(batch_paths):
                try:
                    img = Image.open(p).convert('RGB')
                    img_t = transform(img)
                    batch_imgs.append(img_t)
                    valid_idx.append(j)
                except Exception as e:
                    print(f"[DINO] Skip {p}: {e}")
                    continue
            if not batch_imgs:
                continue
            batch = torch.stack(batch_imgs).to(DINO_DEVICE)
            out = model(batch)  # (B, D)
            feats_list.append(out.cpu().numpy())

    if not feats_list:
        return np.zeros((0, 0), dtype=np.float32)

    feats = np.vstack(feats_list).astype(np.float32)
    # L2 归一化
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    feats = feats / norms
    return feats

def is_image_file(filepath):
    """检查文件是否是图片文件"""
    return Path(filepath).suffix.lower() in IMAGE_EXTENSIONS

def calculate_sharpness(image):
    """计算图像清晰度（使用拉普拉斯方差）"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return laplacian.var()

def should_exclude_generic(filepath, dataset_path):
    """通用的mask/label文件排除逻辑"""
    path_parts = Path(filepath).parts
    
    # 检查是否在明确的label/mask目录中
    exclude_dirs = ['label', 'labels', 'mask', 'masks', 'annotation', 'annotations', 
                    'gt', 'groundtruth', 'ground_truth', 'semantic',
                    'instance', 'class_mask', 'class_label']
    
    for part in path_parts:
        if part.lower() in exclude_dirs:
            return True
    
    # 检查文件名是否明确包含mask/label关键词
    filename_lower = Path(filepath).name.lower()
    exclude_patterns = ['_mask.', '_label.', '_gt.', '_annotation.', 
                       'mask_', 'label_', 'gt_', 'annotation_',
                       '.mask.', '.label.', '.gt.', '.annotation.']
    
    for pattern in exclude_patterns:
        if pattern in filename_lower:
            return True
    
    return False

def scan_bbox_labels(bbox_label_dir):
    """
    扫描 bbox-label 目录，建立「按图片聚合」的标注映射，并统计类别分布。

    返回:
        - label_map: dict
            key   : 图片的绝对路径字符串
            value : {
                "instances": [  # 每个 bbox 为一个 instance
                    {
                        "primary_class": str,
                        "sub_type": str,
                        "bbox": [x, y, w, h],
                        "instance_id": str,
                    },
                    ...
                ],
                "per_class_counts": {primary_class: count, ...},  # 这张图中，各类 bbox 数
                "total_instances": int,  # 这张图中 bbox 总数
            }
        - class_stats: dict, 统计全局每个 primary_class 的 bbox 总数
    """
    label_map = {}
    class_stats = defaultdict(int)
    
    if not bbox_label_dir.exists():
        print(f"[WARNING] bbox-label 目录不存在: {bbox_label_dir}")
        return label_map, dict(class_stats)
    
    print(f"\n扫描 bbox-label 目录: {bbox_label_dir}")
    json_files = list(bbox_label_dir.glob("*.json"))
    print(f"找到 {len(json_files)} 个标注文件")
    
    base_dir = bbox_label_dir.parent
    valid_count = 0
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                label_data = json.load(f)
            
            # 获取图片路径
            image_path_rel = label_data.get('image_path', '')
            if not image_path_rel:
                continue
            
            # 转换为绝对路径并规范化
            image_path_abs = (base_dir / image_path_rel).resolve()
            if not image_path_abs.exists():
                # 尝试直接使用相对路径（如果 image_path_rel 已经是绝对路径的一部分）
                # 或者尝试从 building-open-source 目录开始
                if image_path_rel.startswith('/'):
                    # 如果已经是绝对路径，直接使用
                    image_path_abs = Path(image_path_rel).resolve()
                else:
                    # 尝试不同的基础路径
                    alt_paths = [
                        base_dir / image_path_rel,
                        base_dir.parent / image_path_rel,
                    ]
                    found = False
                    for alt_path in alt_paths:
                        alt_resolved = alt_path.resolve()
                        if alt_resolved.exists():
                            image_path_abs = alt_resolved
                            found = True
                            break
                    if not found:
                        continue  # 跳过找不到的图片
            
            # 获取类别信息
            taxonomy = label_data.get('taxonomy', {})
            primary_class = taxonomy.get('primary_class', 'Unknown')
            sub_type = taxonomy.get('sub_type', '')
            
            # 检查是否有有效的 bbox
            visual_features = label_data.get('visual_features', {})
            bbox = visual_features.get('bbox', [])
            if not bbox or len(bbox) < 4:
                continue  # 跳过没有有效 bbox 的标注
            
            # 按图片聚合：同一张图可能有多个 bbox / 多个类别
            key = str(image_path_abs)
            if key not in label_map:
                label_map[key] = {
                    'instances': [],
                    'per_class_counts': defaultdict(int),
                    'total_instances': 0,
                }
            info = label_map[key]
            info['instances'].append({
                'primary_class': primary_class,
                'sub_type': sub_type,
                'bbox': bbox,
                'instance_id': label_data.get('instance_id', ''),
            })
            info['per_class_counts'][primary_class] += 1
            info['total_instances'] += 1
            
            # 全局统计：按 bbox（instance）数计
            class_stats[primary_class] += 1
            valid_count += 1
            
        except Exception as e:
            print(f"  错误读取 {json_file}: {e}")
            continue
    
    # 将 per_class_counts 从 defaultdict 转成普通 dict，便于后续序列化 / DataFrame 使用
    for key, info in label_map.items():
        if isinstance(info.get('per_class_counts'), defaultdict):
            info['per_class_counts'] = dict(info['per_class_counts'])
    
    print(f"  有效标注: {valid_count} 个")
    print(f"  类别分布(按 bbox 数量统计): {dict(class_stats)}")
    
    return label_map, dict(class_stats)


def collect_all_images(base_dir):
    """收集所有数据集的图片信息"""
    all_images = []
    
    # 收集所有数据集
    datasets = []
    
    classification_dir = base_dir / 'classification'
    if classification_dir.exists():
        for dataset in classification_dir.iterdir():
            if dataset.is_dir():
                datasets.append(('classification', dataset.name, dataset))
    
    detection_dir = base_dir / 'detection'
    if detection_dir.exists():
        for dataset in detection_dir.iterdir():
            if dataset.is_dir():
                datasets.append(('detection', dataset.name, dataset))
    
    segmentation_dir = base_dir / 'segmentation'
    if segmentation_dir.exists():
        for dataset in segmentation_dir.iterdir():
            if dataset.is_dir():
                datasets.append(('segmentation', dataset.name, dataset))
    
    print(f"开始收集图片信息，共 {len(datasets)} 个数据集...\n")
    
    for category, dataset_name, dataset_path in datasets:
        full_name = f"{category}_{dataset_name}"
        print(f"处理 [{full_name}]...")
        
        image_count = 0
        
        # 根据数据集类型使用不同的遍历策略
        if category == 'classification':
            # Classification数据集：通常图片在子目录中
            # 特殊处理：CCIC数据集图片直接在根目录
            if dataset_name == 'CCIC':
                # CCIC: 图片直接在根目录
                for file in os.listdir(dataset_path):
                    filepath = dataset_path / file
                    if not filepath.is_file() or not is_image_file(filepath):
                        continue
                    
                    if should_exclude_generic(filepath, dataset_path):
                        continue
                    
                    try:
                        img = cv2.imread(str(filepath))
                        if img is None:
                            continue
                        h, w = img.shape[:2]
                        sharpness = calculate_sharpness(img)
                        
                        all_images.append({
                            'filepath': str(filepath),
                            'category': category,
                            'dataset': dataset_name,
                            'full_name': full_name,
                            'width': w,
                            'height': h,
                            'area': w * h,
                            'sharpness': float(sharpness),
                            'aspect_ratio': float(w / h) if h > 0 else 0,
                        })
                        image_count += 1
                    except Exception as e:
                        print(f"  错误处理 {filepath}: {e}")
                        continue
            else:
                # 其他classification数据集：图片在子目录中
                for root, dirs, files in os.walk(dataset_path):
                    # 跳过根目录（如果根目录没有图片）
                    if Path(root) == dataset_path:
                        # 检查根目录是否有图片
                        has_images = any(is_image_file(Path(root) / f) for f in files)
                        if not has_images:
                            continue
                    
                    for file in files:
                        filepath = Path(root) / file
                        if not is_image_file(filepath):
                            continue
                        
                        if should_exclude_generic(filepath, dataset_path):
                            continue
                        
                        try:
                            img = cv2.imread(str(filepath))
                            if img is None:
                                continue
                            h, w = img.shape[:2]
                            sharpness = calculate_sharpness(img)
                            
                            all_images.append({
                                'filepath': str(filepath),
                                'category': category,
                                'dataset': dataset_name,
                                'full_name': full_name,
                                'width': w,
                                'height': h,
                                'area': w * h,
                                'sharpness': float(sharpness),
                                'aspect_ratio': float(w / h) if h > 0 else 0,
                            })
                            image_count += 1
                        except Exception as e:
                            print(f"  错误处理 {filepath}: {e}")
                            continue
        
        elif category == 'detection':
            # Detection数据集：可能有特定目录结构
            if dataset_name == 'cubit-det':
                # cubit-det: 图片在images目录中
                images_dir = dataset_path / 'images'
                if images_dir.exists():
                    for root, dirs, files in os.walk(images_dir):
                        for file in files:
                            filepath = Path(root) / file
                            if not is_image_file(filepath):
                                continue
                            
                            try:
                                img = cv2.imread(str(filepath))
                                if img is None:
                                    continue
                                h, w = img.shape[:2]
                                sharpness = calculate_sharpness(img)
                                
                                all_images.append({
                                    'filepath': str(filepath),
                                    'category': category,
                                    'dataset': dataset_name,
                                    'full_name': full_name,
                                    'width': w,
                                    'height': h,
                                    'area': w * h,
                                    'sharpness': float(sharpness),
                                    'aspect_ratio': float(w / h) if h > 0 else 0,
                                })
                                image_count += 1
                            except Exception as e:
                                print(f"  错误处理 {filepath}: {e}")
                                continue
            elif dataset_name == 'MBDD2025':
                # MBDD2025: 图片在JPEGImages目录中
                images_dir = dataset_path / 'JPEGImages'
                if images_dir.exists():
                    for root, dirs, files in os.walk(images_dir):
                        for file in files:
                            filepath = Path(root) / file
                            if not is_image_file(filepath):
                                continue
                            
                            try:
                                img = cv2.imread(str(filepath))
                                if img is None:
                                    continue
                                h, w = img.shape[:2]
                                sharpness = calculate_sharpness(img)
                                
                                all_images.append({
                                    'filepath': str(filepath),
                                    'category': category,
                                    'dataset': dataset_name,
                                    'full_name': full_name,
                                    'width': w,
                                    'height': h,
                                    'area': w * h,
                                    'sharpness': float(sharpness),
                                    'aspect_ratio': float(w / h) if h > 0 else 0,
                                })
                                image_count += 1
                            except Exception as e:
                                print(f"  错误处理 {filepath}: {e}")
                                continue
            else:
                # 其他detection数据集：通用处理
                for root, dirs, files in os.walk(dataset_path):
                    # 跳过labels目录
                    if 'labels' in Path(root).parts or 'label' in Path(root).parts:
                        continue
                    
                    for file in files:
                        filepath = Path(root) / file
                        if not is_image_file(filepath):
                            continue
                        
                        # 排除JSON等标注文件
                        if filepath.suffix.lower() == '.json':
                            continue
                        
                        if should_exclude_generic(filepath, dataset_path):
                            continue
                        
                        try:
                            img = cv2.imread(str(filepath))
                            if img is None:
                                continue
                            h, w = img.shape[:2]
                            sharpness = calculate_sharpness(img)
                            
                            all_images.append({
                                'filepath': str(filepath),
                                'category': category,
                                'dataset': dataset_name,
                                'full_name': full_name,
                                'width': w,
                                'height': h,
                                'area': w * h,
                                'sharpness': float(sharpness),
                                'aspect_ratio': float(w / h) if h > 0 else 0,
                            })
                            image_count += 1
                        except Exception as e:
                            print(f"  错误处理 {filepath}: {e}")
                            continue
        
        else:  # segmentation
            if dataset_name == 'Bai-2020':
                # Bai-2020: 只处理Object-and_structural-level_image&label/image目录（场景级图片）
                # 不处理pixel-level_image&label/image（像素级patch）
                image_dir = dataset_path / 'Data' / 'Object-and_structural-level_image&label' / 'image'
                
                if image_dir.exists():
                    for root, dirs, files in os.walk(image_dir):
                        for file in files:
                            filepath = Path(root) / file
                            if not is_image_file(filepath):
                                continue
                            
                            try:
                                img = cv2.imread(str(filepath))
                                if img is None:
                                    continue
                                h, w = img.shape[:2]
                                sharpness = calculate_sharpness(img)
                                
                                all_images.append({
                                    'filepath': str(filepath),
                                    'category': category,
                                    'dataset': dataset_name,
                                    'full_name': full_name,
                                    'width': w,
                                    'height': h,
                                    'area': w * h,
                                    'sharpness': float(sharpness),
                                    'aspect_ratio': float(w / h) if h > 0 else 0,
                                })
                                image_count += 1
                            except Exception as e:
                                print(f"  错误处理 {filepath}: {e}")
                                continue
            elif dataset_name == 'CSD':
                # CSD: 图片在images目录中
                images_dir = dataset_path / 'images'
                if images_dir.exists():
                    for root, dirs, files in os.walk(images_dir):
                        for file in files:
                            filepath = Path(root) / file
                            if not is_image_file(filepath):
                                continue
                            
                            try:
                                img = cv2.imread(str(filepath))
                                if img is None:
                                    continue
                                h, w = img.shape[:2]
                                sharpness = calculate_sharpness(img)
                                
                                all_images.append({
                                    'filepath': str(filepath),
                                    'category': category,
                                    'dataset': dataset_name,
                                    'full_name': full_name,
                                    'width': w,
                                    'height': h,
                                    'area': w * h,
                                    'sharpness': float(sharpness),
                                    'aspect_ratio': float(w / h) if h > 0 else 0,
                                })
                                image_count += 1
                            except Exception as e:
                                print(f"  错误处理 {filepath}: {e}")
                                continue
            else:
                # 其他segmentation数据集：排除masks/labels目录
                for root, dirs, files in os.walk(dataset_path):
                    path_parts = Path(root).parts
                    skip_dirs = ['masks', 'labels', 'label', 'mask', 'annotations', 'annotation']
                    if any(part.lower() in skip_dirs for part in path_parts):
                        continue
                    
                    for file in files:
                        filepath = Path(root) / file
                        if not is_image_file(filepath):
                            continue
                        
                        if should_exclude_generic(filepath, dataset_path):
                            continue
                        
                        try:
                            img = cv2.imread(str(filepath))
                            if img is None:
                                continue
                            h, w = img.shape[:2]
                            sharpness = calculate_sharpness(img)
                            
                            all_images.append({
                                'filepath': str(filepath),
                                'category': category,
                                'dataset': dataset_name,
                                'full_name': full_name,
                                'width': w,
                                'height': h,
                                'area': w * h,
                                'sharpness': float(sharpness),
                                'aspect_ratio': float(w / h) if h > 0 else 0,
                            })
                            image_count += 1
                        except Exception as e:
                            print(f"  错误处理 {filepath}: {e}")
                            continue
        
        print(f"  找到 {image_count} 张图片")
    
    print(f"\n总共收集到 {len(all_images)} 张图片")
    return all_images


def filter_images_with_labels(all_images, label_map, base_dir):
    """
    只保留在 label_map 中有标注的图片，并添加标注信息。
    同时标记图片是否来自 segmentation 目录（用于后续优先选择）。
    """
    filtered = []
    segmentation_count = 0
    other_count = 0
    
    for img_info in all_images:
        filepath = img_info['filepath']
        
        # 规范化路径以便匹配
        filepath_normalized = str(Path(filepath).resolve())
        
        # 检查是否有标注（使用规范化路径）
        if filepath_normalized not in label_map:
            continue
        
        # ------------ 添加按图片聚合的 bbox / 类别信息 ------------
        label_info = label_map[filepath_normalized]
        instances = label_info.get('instances', [])
        per_class_counts = label_info.get('per_class_counts', {}) or {}
        total_instances = label_info.get('total_instances', len(instances))
        
        # 主导类别：这张图中 bbox 数量最多的 primary_class，用于兼容原有按图片统计逻辑
        if per_class_counts:
            dominant_primary_class = max(per_class_counts.items(), key=lambda kv: kv[1])[0]
        else:
            dominant_primary_class = 'Unknown'
        
        img_info['primary_class'] = dominant_primary_class
        img_info['bbox_total'] = int(total_instances)
        img_info['bbox_per_class'] = dict(per_class_counts)
        img_info['has_annotation'] = True
        
        # 检查是否来自 segmentation 目录
        path_obj = Path(filepath)
        path_parts = path_obj.parts
        is_segmentation = 'segmentation' in path_parts
        img_info['is_segmentation'] = is_segmentation
        
        if is_segmentation:
            segmentation_count += 1
        else:
            other_count += 1
        
        filtered.append(img_info)
    
    print(f"\n过滤后保留 {len(filtered)} 张有标注的图片")
    print(f"  来自 segmentation 目录: {segmentation_count}")
    print(f"  来自其他目录: {other_count}")
    
    return filtered

def analyze_and_recommend_filters(all_images):
    """分析所有图片并给出筛选建议"""
    if not all_images:
        return None, None, None
    
    df = pd.DataFrame(all_images)
    
    # 计算整体统计信息（用于参考）
    stats = {
        'total': len(df),
        'width': {
            'min': int(df['width'].min()),
            'max': int(df['width'].max()),
            'mean': float(df['width'].mean()),
            'median': float(df['width'].median()),
            'std': float(df['width'].std()),
            'percentile_5': float(df['width'].quantile(0.05)),
            'percentile_25': float(df['width'].quantile(0.25)),
            'percentile_75': float(df['width'].quantile(0.75)),
            'percentile_95': float(df['width'].quantile(0.95)),
        },
        'height': {
            'min': int(df['height'].min()),
            'max': int(df['height'].max()),
            'mean': float(df['height'].mean()),
            'median': float(df['height'].median()),
            'std': float(df['height'].std()),
            'percentile_5': float(df['height'].quantile(0.05)),
            'percentile_25': float(df['height'].quantile(0.25)),
            'percentile_75': float(df['height'].quantile(0.75)),
            'percentile_95': float(df['height'].quantile(0.95)),
        },
        'area': {
            'min': int(df['area'].min()),
            'max': int(df['area'].max()),
            'mean': float(df['area'].mean()),
            'median': float(df['area'].median()),
            'std': float(df['area'].std()),
            'percentile_5': float(df['area'].quantile(0.05)),
            'percentile_25': float(df['area'].quantile(0.25)),
            'percentile_75': float(df['area'].quantile(0.75)),
            'percentile_95': float(df['area'].quantile(0.95)),
        },
        'sharpness': {
            'min': float(df['sharpness'].min()),
            'max': float(df['sharpness'].max()),
            'mean': float(df['sharpness'].mean()),
            'median': float(df['sharpness'].median()),
            'std': float(df['sharpness'].std()),
            'percentile_5': float(df['sharpness'].quantile(0.05)),
            'percentile_25': float(df['sharpness'].quantile(0.25)),
            'percentile_75': float(df['sharpness'].quantile(0.75)),
            'percentile_95': float(df['sharpness'].quantile(0.95)),
        },
    }
    
    # 按类别分析
    category_stats = {}
    for category in df['category'].unique():
        cat_df = df[df['category'] == category]
        category_stats[category] = {
            'count': len(cat_df),
            'mean_width': float(cat_df['width'].mean()),
            'mean_height': float(cat_df['height'].mean()),
            'mean_area': float(cat_df['area'].mean()),
            'mean_sharpness': float(cat_df['sharpness'].mean()),
            'median_area': float(cat_df['area'].median()),
            'median_sharpness': float(cat_df['sharpness'].median()),
        }
    
    # 先按现有规则计算 level（patch-level / bbox-level）
    df_level = df.copy()
    df_level['level'] = df_level.apply(
        lambda row: classify_patch_vs_bbox(row.to_dict(), category_stats, stats), axis=1
    )
    
    # 为每个 level 分别计算 10% 分位阈值
    per_level_recs = {}
    for level_name in ['patch-level', 'bbox-level']:
        sub = df_level[df_level['level'] == level_name]
        if sub.empty:
            continue
        
        w_q10 = float(sub['width'].quantile(0.10))
        h_q10 = float(sub['height'].quantile(0.10))
        a_q10 = float(sub['area'].quantile(0.10))
        s_q10 = float(sub['sharpness'].quantile(0.10))
        
        per_level_recs[level_name] = {
            'min_width': max(32, int(w_q10)),
            'min_height': max(32, int(h_q10)),
            'min_area': max(1024, int(a_q10)),
            'min_sharpness': max(10.0, s_q10),
        }
    
    # 整体（占位统计，真实过滤率在后面根据should_filter计算）
    recommendations = {
        'per_level': per_level_recs,
    }
    
    return stats, category_stats, recommendations

def classify_patch_vs_bbox(image_info, category_stats, all_images_stats=None):
    """将图片分类为patch-level或bbox-level
    
    分类规则：
    1. Patch-level: 通常是小的、裁剪的图片，用于分类任务
       - 特征：面积较小（通常<200000 pixels²），宽高比接近1:1
       - Classification数据集中的大多数图片
       - 文件名可能包含patch、crop等关键词
    
    2. Bbox-level: 通常是包含完整目标或场景的图片，用于检测/分割任务
       - 特征：面积较大（通常>=200000 pixels²），可能有不规则的宽高比
       - Detection和Segmentation数据集中的大多数图片
    """
    category = image_info['category']
    width = image_info['width']
    height = image_info['height']
    area = image_info['area']
    aspect_ratio = image_info['aspect_ratio']
    filepath = image_info['filepath'].lower()
    
    # 规则1: 基于文件名关键词（最可靠）
    patch_keywords = ['patch', 'crop', 'cls', 'tile', 'window']
    bbox_keywords = ['full', 'scene', 'image', 'original']
    
    filename = Path(image_info['filepath']).name.lower()
    if any(kw in filename for kw in patch_keywords):
        return 'patch-level'
    if any(kw in filename for kw in bbox_keywords):
        return 'bbox-level'
    
    # 规则2: 基于数据集类别和尺寸的综合判断
    if category == 'classification':
        # Classification数据集：全部都是patch-level（用于分类任务的裁剪图片）
        # 根据summary.json，所有classification数据集都是小尺寸patch（227x227到512x512）
        return 'patch-level'
    
    elif category == 'detection':
        # Detection数据集：大多数是bbox-level（完整场景）
        # 但如果面积很小（<50000），可能是patch-level
        if area < 50000:  # 小于约224x224
            return 'patch-level'
        # 如果面积很大（>200000），肯定是bbox-level
        elif area > 200000:
            return 'bbox-level'
        # 中等尺寸：根据宽高比判断
        else:
            # 接近正方形且面积较小 -> patch-level
            if 0.7 <= aspect_ratio <= 1.4 and area < 100000:
                return 'patch-level'
            else:
                return 'bbox-level'  # detection默认bbox-level
    
    else:  # segmentation
        # Segmentation数据集：大多数是bbox-level（完整场景）
        # 但如果面积很小（<50000），可能是patch-level
        if area < 50000:  # 小于约224x224
            return 'patch-level'
        # 如果面积很大（>200000），肯定是bbox-level
        elif area > 200000:
            return 'bbox-level'
        # 中等尺寸：根据宽高比判断
        else:
            # 接近正方形且面积较小 -> patch-level
            if 0.7 <= aspect_ratio <= 1.4 and area < 100000:
                return 'patch-level'
            else:
                return 'bbox-level'  # segmentation默认bbox-level

def apply_filters_and_classify(all_images, stats, category_stats, recommendations):
    """应用筛选规则并分类"""
    df = pd.DataFrame(all_images)
    
    # 分类为patch-level或bbox-level
    df['level'] = df.apply(lambda row: classify_patch_vs_bbox(row.to_dict(), category_stats, stats), axis=1)
    
    # 应用按 level 的筛选阈值
    per_level = recommendations.get('per_level', {})
    
    def compute_filter_flags(row):
        lvl = row['level']
        rec = per_level.get(lvl)
        if not rec:
            # 如果没有对应阈值，就认为不过滤
            return pd.Series({'too_small': False, 'too_blur': False, 'should_filter': False})
        
        too_small = (row['width'] < rec['min_width']) or (row['height'] < rec['min_height']) or (row['area'] < rec['min_area'])
        too_blur = row['sharpness'] < rec['min_sharpness']
        should = too_small or too_blur
        return pd.Series({'too_small': too_small, 'too_blur': too_blur, 'should_filter': should})
    
    flags = df.apply(compute_filter_flags, axis=1)
    df[['too_small', 'too_blur', 'should_filter']] = flags

    # 可选：使用 CLIP 做语义筛选，删除明显不是“户外墙面/建筑立面”的图片
    if USE_CLIP_FILTER:
        print("\n[CLIP] 开始语义筛选（保留户外建筑墙面相关图片）...")
        semantic_keep = []
        for idx, row in df.iterrows():
            pos_score, neg_score = clip_semantic_scores(row['filepath'])
            # 简单规则：正向分数要足够高，且显著大于负向分数
            keep = (pos_score >= 0.3) and (pos_score > neg_score)
            semantic_keep.append(keep)
        df['semantic_keep'] = semantic_keep
        # 语义不过关的图片也标记为 should_filter
        df.loc[~df['semantic_keep'], 'should_filter'] = True
        print(f"[CLIP] 语义筛选保留 {df['semantic_keep'].sum()} / {len(df)} 张图片")
    else:
        df['semantic_keep'] = True

    # 计算基础质量评分：综合面积和清晰度
    # 使用 log1p(area) * sharpness，既考虑分辨率又考虑清晰度
    df['base_quality'] = np.log1p(df['area']) * df['sharpness']

    # 使用 DINO 在每个 (level, dataset) 内进行“相似度 > 0.95 的贪心去重”
    # 逻辑：在每个子集中，按 base_quality 从高到低遍历，高分样本优先保留；
    #       对于每一个被保留的样本，删除与其余弦相似度 > 0.95 的样本（视为近重复）。
    print("\n[DINO] 开始按 level + dataset 进行去重（相似度 > 0.95 视为近重复）...")
    df['dedup_keep'] = True
    valid_mask = df['semantic_keep']
    for level_name in ['patch-level', 'bbox-level']:
        level_mask = valid_mask & (df['level'] == level_name)
        if not level_mask.any():
            continue
        for dataset_name, sub in df[level_mask].groupby('dataset'):
            if sub.empty:
                continue
            # 为了控制计算量，只在本数据集内 base_quality 较高的前若干样本上做去重
            if level_name == 'patch-level':
                cap = 2000  # 只是用于控制候选数量的上限尺度
            else:
                cap = 2000
            max_candidates = min(len(sub), cap * 3)
            sub_sorted = sub.sort_values('base_quality', ascending=False).iloc[:max_candidates]
            sub_idx = sub_sorted.index
            sub_paths = sub_sorted['filepath'].tolist()
            feats = extract_dino_features(sub_paths)
            if feats.size == 0:
                continue
            N = feats.shape[0]
            keep_flags = np.ones(N, dtype=bool)
            cos_thresh = 0.95
            # 贪心：从高分到低分遍历，删除与已选样本过于相似的样本
            for i in range(N):
                if not keep_flags[i]:
                    continue
                vi = feats[i]
                sims = feats @ vi
                # 将与当前样本相似度 > 0.95 的样本标记为删除（包括自身，但自身会重新标记为保留）
                dup_mask = sims > cos_thresh
                keep_flags[dup_mask] = False
                keep_flags[i] = True  # 当前样本保留
            # 将被去重掉的样本标记为 dedup_keep=False
            removed_idx = [sub_idx[i] for i in range(N) if not keep_flags[i]]
            if removed_idx:
                df.loc[removed_idx, 'dedup_keep'] = False

    # 最终质量分目前直接使用 base_quality（去重逻辑已经优先保留高质量样本）
    df['quality'] = df['base_quality']

    # 按 level 选 Top-K 质量最好的图片
    df['topk_keep'] = False

    def select_topk_for_level(level_name: str, k: int):
        """
        按 level 选出 Top-K，考虑类别平衡和 segmentation 优先：
        1）优先选择 segmentation 目录的图片
        2）在配额约束下，尽量保证类别平衡
        3）从高到低遍历，如果某个样本所属的数据集尚未用完自己的配额(cap)，则选入；
        4）直到选满 K 张或没有可选样本。

        这样可以保证：
        - 每个数据集最终被选中的样本数不超过其 cap（避免单一大数据集垄断）；
        - 在配额约束下，尽量接近"全局最高质量"的 Top-K；
        - 优先选择 segmentation 目录的图片；
        - 尽量保证类别平衡。
        """
        nonlocal df
        # Top-K 只在语义通过且未被DINO去重剔除的样本中选
        sub = df[(df['level'] == level_name) & (df['semantic_keep']) & (df['dedup_keep'])]
        if sub.empty:
            return

        # ---------------- 基于 bbox 数量的细粒度类别平衡 ----------------
        # 思路：
        #   1. 在当前候选子集 sub 中，统计每个类别的 bbox 总数；
        #   2. 按「如果随机选 k/N 比例的图片」的期望，计算每类应保留的目标 bbox 数；
        #   3. 选图时维护 bbox_selected[cls]，在后 20% 名额阶段尽量避免让某类 bbox
        #      远超自己的目标数量，从而实现基于 bbox 的更细粒度平衡。
        from collections import defaultdict as _dd
        target_bbox_per_class = {}
        if 'bbox_per_class' in sub.columns:
            bbox_totals = _dd(int)
            for _, row in sub.iterrows():
                per_class = row.get('bbox_per_class', {}) or {}
                for cls, cnt in per_class.items():
                    try:
                        bbox_totals[cls] += int(cnt)
                    except Exception:
                        continue
            
            N = len(sub)
            if N > 0 and k > 0 and bbox_totals:
                alpha = k / float(N)
                for cls, total_cnt in bbox_totals.items():
                    # 期望保留 alpha 比例的 bbox，至少 1 个
                    tgt = int(round(alpha * total_cnt))
                    if tgt <= 0:
                        tgt = 1
                    target_bbox_per_class[cls] = tgt
        
        # 分离 segmentation 和其他目录的样本
        # 如果 is_segmentation 列不存在，默认为 False
        if 'is_segmentation' not in sub.columns:
            sub['is_segmentation'] = False
        sub_seg = sub[sub['is_segmentation'] == True]
        sub_other = sub[sub['is_segmentation'] == False]
        
        # 先按 is_segmentation 排序（True 在前），再按 quality 排序
        sub_sorted = pd.concat([
            sub_seg.sort_values('quality', ascending=False),
            sub_other.sort_values('quality', ascending=False)
        ])

        if level_name == 'patch-level':
            # Patch-level：按每个数据集配额选 Top-K
            # 多类缺陷分类数据集配额较高；只有 crack 的分类数据集（如 CCIC）配额按「每类平均」压缩到接近其他数据集。
            dataset_caps = {
                # 大型分类数据集：主力墙面多类缺陷 patch
                # 配额总和约 10000，确保能选够 5000 张
                'HS-23K': 5364,   # 6 类（含 cracks），约 300+ / 类
                'BD3':     1877,   # 7 类（algae / major_crack / minor_crack / peeling / plain / spalling / stain），约 100 / 类
                # CCIC 只有 crack 一类：配额按每类 ~300 的量级给一个较小值，避免单一 crack 集合压制多类数据
                'CCIC':    1072,   # 1 类（crack），总配额 ~1072
                'BDD':     669,   # 多类（crack / flakes 等），几乎全保留
                # 分割中提供 crack/background 场景级/局部 patch，代表性为主即可
                'masonry': 401,   # 480 个 crack/background patch，取约 30%
                'Bai-2020':401,   # 659 个 Object-level crack/background 小场景，取约 1/4
                # 其他少量被判为 patch 的检测图（如 BDW）
                'BDW':     213,
            }
            # 其他 patch-level 数据集（如 uav75 的小 patch、零散来源）默认每个最多 213 张
            default_cap = 213

            selected_idx = []
            used = defaultdict(int)
            # 基于 bbox 数量的已选统计
            bbox_selected = _dd(int)

            for idx, row in sub_sorted.iterrows():
                ds = row['dataset']
                cap = dataset_caps.get(ds, default_cap)
                if used[ds] >= cap:
                    continue  # 该数据集已达配额，跳过

                per_class = row.get('bbox_per_class', {}) or {}

                # 在已选数量达到 80% 之后，启动更严格的 bbox 平衡约束：
                # 如果一张图只会显著增加已经超额的类别 bbox，则尽量跳过。
                if target_bbox_per_class and len(selected_idx) >= k * 0.8:
                    too_much = True
                    for cls, cnt in per_class.items():
                        tgt = target_bbox_per_class.get(cls)
                        if tgt is None:
                            # 对于没有目标的类别，不强行限制
                            too_much = False
                            break
                        if bbox_selected[cls] + cnt <= tgt:
                            # 至少有一个类别尚未达到目标，这张图仍然有平衡价值
                            too_much = False
                            break
                    if too_much:
                        continue

                selected_idx.append(idx)
                used[ds] += 1
                for cls, cnt in per_class.items():
                    try:
                        bbox_selected[cls] += int(cnt)
                    except Exception:
                        continue
                if len(selected_idx) >= k:
                    break

            df.loc[selected_idx, 'topk_keep'] = True
        else:
            # Bbox-level：按每个数据集配额 + 全局排序选 Top-K
            # s2ds：多类像素级缺陷（crack / spalling / corrosion / efflorescence / vegetation / control point / background），
            #       希望全部计入 bbox-level 的 Top-K 候选，以保证非 crack 类别有足够覆盖。
            # segmentation 目录下其他数据集统一视为以 crack 为主的来源，通过较小配额控制其总量，
            # 同时将 CSD 也视为 crack+其他墙体缺陷的来源之一，配额与 crack-only 数据集同一量级，
            # 并将 MBDD2025 的配额砍半，以避免桥梁/构件数据过度主导。
            bbox_caps = {
                # 多类检测/分割场景
                # 配额总和约 10000，确保能选够 5000 张
                'MBDD2025': 1886,    # ~14K，多类桥/构件缺陷，砍半以降低占比
                'cubit-det': 2612,   # ~5.5K，多类桥/建筑检测

                # s2ds：多类像素级分割，全部纳入候选集合（保障非 crack 类别覆盖）
                's2ds':     4314,   # s2ds bbox-level 总量约 1486，全保留

                # CSD + 其他 crack-only 或 crack 为主的分割/检测数据集：
                # 统一作为 crack 及少数附加缺陷的来源，总 cap 控制在与 s2ds 其他类别平均数同一量级。
                'CSD':      232,     # ~11K，多类墙体缺陷，这里配额压到与 crack-only 数据集同级
                'deepcarck':232,     # ~1K crack/background
                'uav75':    116,     # ~150 crack/background
                'cubit-seg':173,     # ~400+ crack/spalling/background（这里只算作 crack 来源的一部分）
                'Bai-2020': 57,     # Object-level 大场景中的 crack 代表
                'BDW':      232,     # 道路裂缝检测
                # 其他少量 bbox 级样本
                'BDD':      144,
                # 若 Heritage defects 以某个 dataset 名称出现，可在此补充配额，例如：
                # 'Heritage Building Defect Detection Dataset': 200,
            }
            # 其他极小众 bbox-level 数据集默认每个最多 232 张
            default_cap_bbox = 232

            selected_idx = []
            used = defaultdict(int)
            bbox_selected = _dd(int)

            for idx, row in sub_sorted.iterrows():
                ds = row['dataset']
                cap = bbox_caps.get(ds, default_cap_bbox)
                if used[ds] >= cap:
                    continue

                per_class = row.get('bbox_per_class', {}) or {}

                if target_bbox_per_class and len(selected_idx) >= k * 0.8:
                    too_much = True
                    for cls, cnt in per_class.items():
                        tgt = target_bbox_per_class.get(cls)
                        if tgt is None:
                            too_much = False
                            break
                        if bbox_selected[cls] + cnt <= tgt:
                            too_much = False
                            break
                    if too_much:
                        continue

                selected_idx.append(idx)
                used[ds] += 1
                for cls, cnt in per_class.items():
                    try:
                        bbox_selected[cls] += int(cnt)
                    except Exception:
                        continue
                if len(selected_idx) >= k:
                    break

            df.loc[selected_idx, 'topk_keep'] = True

    # Patch-level Top-K
    select_topk_for_level('patch-level', TOPK_PATCH)
    # Bbox-level Top-K
    select_topk_for_level('bbox-level', TOPK_BBOX)
    
    return df

def generate_report(df, stats, category_stats, recommendations, output_dir):
    """生成分析报告"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 1. 保存详细数据
    df.to_csv(output_dir / 'all_images_detailed.csv', index=False, encoding='utf-8')
    
    # 2. 生成统计报告
    report = {
        'summary': {
            'total_images': len(df),
            'after_filter': len(df[~df['should_filter']]),
            'filtered_count': len(df[df['should_filter']]),
            'filtered_percentage': len(df[df['should_filter']]) / len(df) * 100,
        },
        'filter_recommendations': recommendations,
        'overall_statistics': stats,
        'category_statistics': category_stats,
        'level_distribution': {
            'patch-level': int((df['level'] == 'patch-level').sum()),
            'bbox-level': int((df['level'] == 'bbox-level').sum()),
        },
        'level_by_category': {},
        'filter_by_level': {},
        'topk_by_level': {},
    }
    
    # 按类别统计level分布
    for category in df['category'].unique():
        cat_df = df[df['category'] == category]
        report['level_by_category'][category] = {
            'patch-level': int((cat_df['level'] == 'patch-level').sum()),
            'bbox-level': int((cat_df['level'] == 'bbox-level').sum()),
            'total': len(cat_df),
        }
    
    # 按level统计过滤情况
    for level_name in ['patch-level', 'bbox-level']:
        sub = df[df['level'] == level_name]
        if sub.empty:
            continue
        total = len(sub)
        filtered = int(sub['should_filter'].sum())
        report['filter_by_level'][level_name] = {
            'total': total,
            'filtered': filtered,
            'filtered_percentage': filtered / total * 100,
        }

        # Top-K 情况
        topk_sub = sub[sub['topk_keep']]
        report['topk_by_level'][level_name] = {
            'total': total,
            'topk': len(topk_sub),
        }

    # 保存每个 level 的 Top-K 列表
    patch_top_path = output_dir / f'patch_level_top{TOPK_PATCH}.csv'
    bbox_top_path = output_dir / f'bbox_level_top{TOPK_BBOX}.csv'
    df[(df['level'] == 'patch-level') & (df['topk_keep'])].to_csv(patch_top_path, index=False, encoding='utf-8')
    df[(df['level'] == 'bbox-level') & (df['topk_keep'])].to_csv(bbox_top_path, index=False, encoding='utf-8')
    
    # 保存报告
    with open(output_dir / 'filter_and_classify_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # 3. 生成可视化
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 如果有按 level 的阈值，构造一个全局参考阈值用于可视化（取各 level 最小值）
    per_level_recs = recommendations.get('per_level', {})
    if per_level_recs:
        min_width_vis = min(r['min_width'] for r in per_level_recs.values())
        min_height_vis = min(r['min_height'] for r in per_level_recs.values())
        min_area_vis = min(r['min_area'] for r in per_level_recs.values())
        min_sharpness_vis = min(r['min_sharpness'] for r in per_level_recs.values())
    else:
        min_width_vis = min_height_vis = min_area_vis = min_sharpness_vis = None
    
    # 尺寸分布
    axes[0, 0].scatter(df['width'], df['height'], alpha=0.3, s=1)
    if min_width_vis is not None:
        axes[0, 0].axvline(min_width_vis, color='red', linestyle='--', label=f'Min Width (per-level min): {min_width_vis}')
    if min_height_vis is not None:
        axes[0, 0].axhline(min_height_vis, color='orange', linestyle='--', label=f'Min Height (per-level min): {min_height_vis}')
    axes[0, 0].set_xlabel('Width (pixels)')
    axes[0, 0].set_ylabel('Height (pixels)')
    axes[0, 0].set_title('Image Size Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 面积分布
    axes[0, 1].hist(df['area'], bins=100, edgecolor='black', alpha=0.7)
    if min_area_vis is not None:
        axes[0, 1].axvline(min_area_vis, color='red', linestyle='--', label=f'Min Area (per-level min): {min_area_vis}')
    axes[0, 1].set_xlabel('Area (pixels²)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Image Area Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xscale('log')
    
    # 清晰度分布
    axes[0, 2].hist(df['sharpness'], bins=100, edgecolor='black', alpha=0.7)
    if min_sharpness_vis is not None:
        axes[0, 2].axvline(min_sharpness_vis, color='red', linestyle='--', label=f'Min Sharpness (per-level min): {min_sharpness_vis:.2f}')
    axes[0, 2].set_xlabel('Sharpness (Laplacian Variance)')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Image Sharpness Distribution')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Level分类分布
    level_counts = df['level'].value_counts()
    axes[1, 0].bar(level_counts.index, level_counts.values, color=['skyblue', 'lightgreen'])
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Patch-level vs Bbox-level Distribution')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 按类别的Level分布
    category_level = pd.crosstab(df['category'], df['level'])
    category_level.plot(kind='bar', ax=axes[1, 1], color=['skyblue', 'lightgreen'])
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Level Distribution by Category')
    axes[1, 1].legend(title='Level')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # 筛选状态
    filter_counts = df['should_filter'].value_counts()
    axes[1, 2].bar(['Keep', 'Filter'], [filter_counts.get(False, 0), filter_counts.get(True, 0)], 
                   color=['green', 'red'])
    axes[1, 2].set_ylabel('Count')
    axes[1, 2].set_title('Filter Status')
    axes[1, 2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'filter_and_classify_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. 打印摘要
    print("\n" + "="*80)
    print("图片筛选和分类分析报告")
    print("="*80)
    total_imgs = len(df)
    filtered_count = int(df['should_filter'].sum())
    kept_count = total_imgs - filtered_count
    filtered_pct = filtered_count / total_imgs * 100 if total_imgs > 0 else 0.0
    print(f"\n总图片数: {total_imgs}")
    print(f"建议保留: {kept_count} ({100 - filtered_pct:.2f}%)")
    print(f"建议筛选: {filtered_count} ({filtered_pct:.2f}%)")
    
    print(f"\n筛选规则（按 level 的详细阈值已写入 filter_and_classify_report.json['filter_recommendations']['per_level']）")
    
    print(f"\nLevel分类:")
    print(f"  Patch-level: {report['level_distribution']['patch-level']} ({report['level_distribution']['patch-level']/len(df)*100:.2f}%)")
    print(f"  Bbox-level: {report['level_distribution']['bbox-level']} ({report['level_distribution']['bbox-level']/len(df)*100:.2f}%)")
    
    # 按level的过滤率
    print(f"\n按Level的过滤率:")
    for level_name, lvl_stats in report['filter_by_level'].items():
        print(f"  {level_name}: 筛选 {lvl_stats['filtered']} / {lvl_stats['total']} ({lvl_stats['filtered_percentage']:.2f}%)")

    # Top-K 摘要
    print(f"\n按Level的Top-K保留情况:")
    for level_name, lvl_stats in report['topk_by_level'].items():
        print(f"  {level_name}: 保留Top-K {lvl_stats['topk']} / {lvl_stats['total']}")
    print(f"\nTop-K列表已保存为:")
    print(f"  Patch-level: {patch_top_path}")
    print(f"  Bbox-level:  {bbox_top_path}")
    
    print(f"\n按类别统计:")
    for category, cat_stats in report['level_by_category'].items():
        print(f"  {category}:")
        print(f"    总数: {cat_stats['total']}")
        print(f"    Patch-level: {cat_stats['patch-level']} ({cat_stats['patch-level']/cat_stats['total']*100:.2f}%)")
        print(f"    Bbox-level: {cat_stats['bbox-level']} ({cat_stats['bbox-level']/cat_stats['total']*100:.2f}%)")
    
    print(f"\n报告已保存到: {output_dir}")
    print("="*80)

def main():
    base_dir = Path('defect_bench/raw_data')
    output_dir = Path('defect_bench/annotation_toolkit/sample_pipeline/results')
    bbox_label_dir = base_dir / 'bbox-label'
    
    # 1. 扫描 bbox-label 目录，建立标注映射和类别统计
    label_map, class_stats = scan_bbox_labels(bbox_label_dir)
    
    if not label_map:
        print("[ERROR] 没有找到任何有效的标注文件！")
        return
    
    # 2. 收集所有图片
    all_images = collect_all_images(base_dir)
    
    if not all_images:
        print("没有找到任何图片！")
        return
    
    # 3. 只保留有标注的图片，并添加标注信息
    all_images = filter_images_with_labels(all_images, label_map, base_dir)
    
    if not all_images:
        print("过滤后没有剩余图片！")
        return

    # ---------------- 预截断步骤：在进入 CLIP / DINO 之前按配额粗筛一遍 ----------------
    # 目的：大幅减少后续需要做 CLIP 语义筛选和 DINO 特征计算的图片数量，减轻计算压力。
    print("\n[Pre-filter] 根据各数据集配额进行预截断，减少后续 CLIP / DINO 计算量...")
    df0 = pd.DataFrame(all_images)
    # 计算基础质量分：只用 area + sharpness，先做一个便宜的排序指标
    df0['area'] = df0['width'] * df0['height']
    df0['aspect_ratio'] = df0['width'] / df0['height']
    df0['base_quality'] = np.log1p(df0['area']) * df0['sharpness']
    # 使用轻量级的 level 判定（不依赖 stats/category_stats），只看类别和尺寸
    df0['level'] = df0.apply(lambda r: classify_patch_vs_bbox(r.to_dict(), {}, None), axis=1)

    # 与 Top-K 阶段一致的配额表（但这里只做“配额的若干倍”作为预筛上限）
    PATCH_CAPS = {
        # 与 patch-level Top-K 阶段保持一致的配额，用于预截断控制各数据集候选数量
        'HS-23K': 5364,
        'BD3':     1877,
        'CCIC':    1072,
        'BDD':     669,
        'masonry': 401,
        'Bai-2020':401,
        'BDW':     213,
    }
    PATCH_DEFAULT_CAP = 213

    BBOX_CAPS = {
        # 与 bbox-level Top-K 阶段保持一致的配额，用于预截断时限制各数据集的候选数量尺度
        'MBDD2025': 1886,
        'cubit-det': 2612,
        's2ds':     4314,
        'CSD':      232,
        'deepcarck':232,
        'uav75':    116,
        'cubit-seg':173,
        'Bai-2020': 57,
        'BDW':      232,
        'BDD':      144,
        # 如果 Heritage defects 出现在 dataset 字段中，可在此添加：
        # 'Heritage Building Defect Detection Dataset': 200,
    }
    BBOX_DEFAULT_CAP = 232

    PRE_MULT = 5.0  # 预截断时，每个数据集最多保留 cap 的若干倍，给后续 CLIP / DINO 足够空间
    # 增加到 5.0 以确保经过 CLIP/DINO 筛选后仍能选出足够的图片

    keep_mask = np.zeros(len(df0), dtype=bool)

    for level_name, caps, default_cap in [
        ('patch-level', PATCH_CAPS, PATCH_DEFAULT_CAP),
        ('bbox-level',  BBOX_CAPS,  BBOX_DEFAULT_CAP),
    ]:
        sub = df0[df0['level'] == level_name]
        if sub.empty:
            continue
        for ds, g in sub.groupby('dataset'):
            cap = caps.get(ds, default_cap)
            pre_cap = int(cap * PRE_MULT)
            if pre_cap <= 0:
                continue
            if len(g) <= pre_cap:
                keep_mask[g.index] = True
            else:
                g_sorted = g.sort_values('base_quality', ascending=False)
                keep_idx = g_sorted.index[:pre_cap]
                keep_mask[keep_idx] = True

    # 应用预截断结果：后续分析/CLIP/DINO/Top-K 只在这一子集上进行
    before_count = len(df0)
    df0 = df0[keep_mask].copy()
    after_count = len(df0)
    print(f"[Pre-filter] 从 {before_count} 张预筛到 {after_count} 张，将在其上进行 CLIP / DINO / Top-K 计算。\n")

    all_images = df0.to_dict(orient='records')
    
    # 分析和推荐筛选规则
    stats, category_stats, recommendations = analyze_and_recommend_filters(all_images)
    
    # 应用筛选和分类
    df = apply_filters_and_classify(all_images, stats, category_stats, recommendations)
    
    # 生成报告
    generate_report(df, stats, category_stats, recommendations, output_dir)

if __name__ == '__main__':
    main()

