#!/usr/bin/env python3
"""
分析building-open-source数据集中各数据集的图片大小和清晰度分布
为每个数据集定制不同的图片查找逻辑
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import json
import argparse
import time

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

# 支持的图片格式
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

def is_image_file(filepath):
    """检查文件是否是图片文件"""
    return Path(filepath).suffix.lower() in IMAGE_EXTENSIONS

def calculate_sharpness(image):
    """计算图像清晰度（使用拉普拉斯方差）"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    # 加速：对大图先缩放再算拉普拉斯（清晰度分布趋势仍然稳定）
    h, w = gray.shape[:2]
    max_side = max(h, w)
    if max_side > 512:
        scale = 512 / max_side
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return laplacian.var()

def should_exclude_generic(filepath, dataset_path):
    """通用的mask/label文件排除逻辑"""
    path_parts = Path(filepath).parts
    
    # 检查是否在明确的label/mask目录中
    exclude_dirs = ['label', 'labels', 'mask', 'masks', 'annotation', 'annotations', 
                    'gt', 'groundtruth', 'ground_truth', 'semantic',
                    'instance', 'class_mask', 'class_label']
    
    # 检查路径中是否包含这些目录名
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

def get_dataset_specific_logic(dataset_name):
    """为每个数据集返回特定的查找逻辑"""

    def _maybe_progress(total_images: int, start_ts: float, every: int = 500):
        if every > 0 and total_images > 0 and total_images % every == 0:
            elapsed = time.time() - start_ts
            rate = total_images / elapsed if elapsed > 0 else 0
            print(f"  已处理 {total_images} 张，用时 {elapsed:.1f}s，{rate:.1f} img/s")
    
    def analyze_classification_BD3(dataset_path):
        """BD3: 图片在子目录中（algae, major_crack等），文件名是cls00_xxx.jpg"""
        widths, heights, sharpness_scores = [], [], []
        total_images, skipped = 0, 0
        
        for root, dirs, files in os.walk(dataset_path):
            # 跳过根目录下的非图片文件
            if Path(root) == dataset_path:
                continue
                
            for file in files:
                filepath = Path(root) / file
                if not is_image_file(filepath):
                    continue
                
                # BD3的图片都在子目录中，直接包含
                try:
                    img = cv2.imread(str(filepath))
                    if img is None:
                        continue
                    h, w = img.shape[:2]
                    widths.append(w)
                    heights.append(h)
                    sharpness = calculate_sharpness(img)
                    sharpness_scores.append(sharpness)
                    total_images += 1
                except Exception as e:
                    print(f"  处理文件时出错 {filepath}: {e}")
                    continue
        
        return {'widths': widths, 'heights': heights, 'sharpness': sharpness_scores, 'total': total_images}
    
    def analyze_classification_BDD(dataset_path):
        """BDD: 图片在train_set/test_set的子目录中"""
        widths, heights, sharpness_scores = [], [], []
        total_images, skipped = 0, 0
        
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                filepath = Path(root) / file
                if not is_image_file(filepath):
                    continue
                
                if should_exclude_generic(filepath, dataset_path):
                    skipped += 1
                    continue
                
                try:
                    img = cv2.imread(str(filepath))
                    if img is None:
                        continue
                    h, w = img.shape[:2]
                    widths.append(w)
                    heights.append(h)
                    sharpness = calculate_sharpness(img)
                    sharpness_scores.append(sharpness)
                    total_images += 1
                except Exception as e:
                    print(f"  处理文件时出错 {filepath}: {e}")
                    continue
        
        return {'widths': widths, 'heights': heights, 'sharpness': sharpness_scores, 'total': total_images}
    
    def analyze_classification_CCIC(dataset_path):
        """CCIC: 图片直接在根目录，文件名是00001.jpg等"""
        widths, heights, sharpness_scores = [], [], []
        total_images, skipped = 0, 0
        start_ts = time.time()
        
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                filepath = Path(root) / file
                if not is_image_file(filepath):
                    continue
                
                if should_exclude_generic(filepath, dataset_path):
                    skipped += 1
                    continue
                
                try:
                    img = cv2.imread(str(filepath))
                    if img is None:
                        continue
                    h, w = img.shape[:2]
                    widths.append(w)
                    heights.append(h)
                    sharpness = calculate_sharpness(img)
                    sharpness_scores.append(sharpness)
                    total_images += 1
                    _maybe_progress(total_images, start_ts, every=500)
                except Exception as e:
                    print(f"  处理文件时出错 {filepath}: {e}")
                    continue
        
        return {'widths': widths, 'heights': heights, 'sharpness': sharpness_scores, 'total': total_images}
    
    def analyze_classification_HS23K(dataset_path):
        """HS-23K: 图片在子目录中（3.cracks等），文件名是03_cracks_xxx.jpg"""
        widths, heights, sharpness_scores = [], [], []
        total_images, skipped = 0, 0
        
        for root, dirs, files in os.walk(dataset_path):
            # 跳过README文件
            if 'README' in root.upper():
                continue
                
            for file in files:
                filepath = Path(root) / file
                if not is_image_file(filepath):
                    continue
                
                if should_exclude_generic(filepath, dataset_path):
                    skipped += 1
                    continue
                
                try:
                    img = cv2.imread(str(filepath))
                    if img is None:
                        continue
                    h, w = img.shape[:2]
                    widths.append(w)
                    heights.append(h)
                    sharpness = calculate_sharpness(img)
                    sharpness_scores.append(sharpness)
                    total_images += 1
                except Exception as e:
                    print(f"  处理文件时出错 {filepath}: {e}")
                    continue
        
        return {'widths': widths, 'heights': heights, 'sharpness': sharpness_scores, 'total': total_images}
    
    def analyze_detection_BDW(dataset_path):
        """BDW: 图片在train目录中"""
        widths, heights, sharpness_scores = [], [], []
        total_images, skipped = 0, 0
        start_ts = time.time()
        
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                filepath = Path(root) / file
                if not is_image_file(filepath):
                    continue
                
                # 排除JSON文件（标注文件）
                if filepath.suffix.lower() == '.json':
                    continue
                
                if should_exclude_generic(filepath, dataset_path):
                    skipped += 1
                    continue
                
                try:
                    img = cv2.imread(str(filepath))
                    if img is None:
                        continue
                    h, w = img.shape[:2]
                    widths.append(w)
                    heights.append(h)
                    sharpness = calculate_sharpness(img)
                    sharpness_scores.append(sharpness)
                    total_images += 1
                    _maybe_progress(total_images, start_ts, every=500)
                except Exception as e:
                    print(f"  处理文件时出错 {filepath}: {e}")
                    continue
        
        return {'widths': widths, 'heights': heights, 'sharpness': sharpness_scores, 'total': total_images}
    
    def analyze_detection_cubit_det(dataset_path):
        """cubit-det: 图片在images/train2017等目录中"""
        widths, heights, sharpness_scores = [], [], []
        total_images, skipped = 0, 0
        start_ts = time.time()
        
        images_dir = dataset_path / 'images'
        if not images_dir.exists():
            return {'widths': [], 'heights': [], 'sharpness': [], 'total': 0}
        
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
                    widths.append(w)
                    heights.append(h)
                    sharpness = calculate_sharpness(img)
                    sharpness_scores.append(sharpness)
                    total_images += 1
                    _maybe_progress(total_images, start_ts, every=500)
                except Exception as e:
                    print(f"  处理文件时出错 {filepath}: {e}")
                    continue
        
        return {'widths': widths, 'heights': heights, 'sharpness': sharpness_scores, 'total': total_images}
    
    def analyze_detection_MBDD2025(dataset_path):
        """MBDD2025: 图片在JPEGImages目录中"""
        widths, heights, sharpness_scores = [], [], []
        total_images, skipped = 0, 0
        start_ts = time.time()
        
        images_dir = dataset_path / 'JPEGImages'
        if not images_dir.exists():
            return {'widths': [], 'heights': [], 'sharpness': [], 'total': 0}
        
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
                    widths.append(w)
                    heights.append(h)
                    sharpness = calculate_sharpness(img)
                    sharpness_scores.append(sharpness)
                    total_images += 1
                    _maybe_progress(total_images, start_ts, every=500)
                except Exception as e:
                    print(f"  处理文件时出错 {filepath}: {e}")
                    continue
        
        return {'widths': widths, 'heights': heights, 'sharpness': sharpness_scores, 'total': total_images}
    
    def analyze_segmentation_Bai2020(dataset_path):
        """Bai-2020: 只统计 Object-and_structural-level_image&label/image 里的场景级图片（忽略 pixel-level）"""
        widths, heights, sharpness_scores = [], [], []
        total_images, skipped = 0, 0
        start_ts = time.time()
        
        image_dir = dataset_path / 'Data' / 'Object-and_structural-level_image&label' / 'image'
        if not image_dir.exists():
            return {'widths': [], 'heights': [], 'sharpness': [], 'total': 0}

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
                    widths.append(w)
                    heights.append(h)
                    sharpness = calculate_sharpness(img)
                    sharpness_scores.append(sharpness)
                    total_images += 1
                    _maybe_progress(total_images, start_ts, every=500)
                except Exception as e:
                    print(f"  处理文件时出错 {filepath}: {e}")
                    continue
        
        return {'widths': widths, 'heights': heights, 'sharpness': sharpness_scores, 'total': total_images}

    def analyze_segmentation_cubit_seg(dataset_path: Path):
        """cubit-seg: 正确图片在 crack_org / spalling_org（同时可能存在成对 jpg/png，png 多为 mask）"""
        widths, heights, sharpness_scores = [], [], []
        total_images = 0

        def iter_images_in_dir(img_dir: Path):
            if not img_dir.exists():
                return
            # 如果同名同时存在 jpg 和 png，默认 png 是 mask，跳过 png
            jpg_stems = {p.stem for p in img_dir.glob("*.jpg")}
            jpg_stems |= {p.stem for p in img_dir.glob("*.jpeg")}

            for p in sorted(img_dir.iterdir()):
                if not p.is_file():
                    continue
                if not is_image_file(p):
                    continue
                if p.suffix.lower() in {".png"} and p.stem in jpg_stems:
                    continue
                yield p

        for img_dir in [dataset_path / "crack_org", dataset_path / "spalling_org"]:
            for filepath in iter_images_in_dir(img_dir):
                try:
                    img = cv2.imread(str(filepath))
                    if img is None:
                        continue
                    h, w = img.shape[:2]
                    widths.append(w)
                    heights.append(h)
                    sharpness_scores.append(calculate_sharpness(img))
                    total_images += 1
                except Exception as e:
                    print(f"  处理文件时出错 {filepath}: {e}")
                    continue

        return {'widths': widths, 'heights': heights, 'sharpness': sharpness_scores, 'total': total_images}

    def analyze_segmentation_uav75(dataset_path: Path):
        """uav75: 正确图片在 Data/{train_img,val_img,test_img}"""
        widths, heights, sharpness_scores = [], [], []
        total_images = 0

        for img_dir in [dataset_path / "Data" / "train_img", dataset_path / "Data" / "val_img", dataset_path / "Data" / "test_img"]:
            if not img_dir.exists():
                continue
            for root, _, files in os.walk(img_dir):
                for file in files:
                    filepath = Path(root) / file
                    if not is_image_file(filepath):
                        continue
                    try:
                        img = cv2.imread(str(filepath))
                        if img is None:
                            continue
                        h, w = img.shape[:2]
                        widths.append(w)
                        heights.append(h)
                        sharpness_scores.append(calculate_sharpness(img))
                        total_images += 1
                    except Exception as e:
                        print(f"  处理文件时出错 {filepath}: {e}")
                        continue

        return {'widths': widths, 'heights': heights, 'sharpness': sharpness_scores, 'total': total_images}

    def analyze_segmentation_deepcarck(dataset_path: Path):
        """deepcarck: 正确图片在 Data/{train_img,test_img}"""
        widths, heights, sharpness_scores = [], [], []
        total_images = 0

        for img_dir in [dataset_path / "Data" / "train_img", dataset_path / "Data" / "test_img"]:
            if not img_dir.exists():
                continue
            for root, _, files in os.walk(img_dir):
                for file in files:
                    filepath = Path(root) / file
                    if not is_image_file(filepath):
                        continue
                    try:
                        img = cv2.imread(str(filepath))
                        if img is None:
                            continue
                        h, w = img.shape[:2]
                        widths.append(w)
                        heights.append(h)
                        sharpness_scores.append(calculate_sharpness(img))
                        total_images += 1
                    except Exception as e:
                        print(f"  处理文件时出错 {filepath}: {e}")
                        continue

        return {'widths': widths, 'heights': heights, 'sharpness': sharpness_scores, 'total': total_images}

    def analyze_segmentation_masonry(dataset_path: Path):
        """masonry: 正确图片在 Data/crack_detection_224_images（patch 级）"""
        widths, heights, sharpness_scores = [], [], []
        total_images = 0

        img_dir = dataset_path / "Data" / "crack_detection_224_images"
        if not img_dir.exists():
            return {'widths': [], 'heights': [], 'sharpness': [], 'total': 0}

        for root, _, files in os.walk(img_dir):
            for file in files:
                filepath = Path(root) / file
                if not is_image_file(filepath):
                    continue
                try:
                    img = cv2.imread(str(filepath))
                    if img is None:
                        continue
                    h, w = img.shape[:2]
                    widths.append(w)
                    heights.append(h)
                    sharpness_scores.append(calculate_sharpness(img))
                    total_images += 1
                except Exception as e:
                    print(f"  处理文件时出错 {filepath}: {e}")
                    continue

        return {'widths': widths, 'heights': heights, 'sharpness': sharpness_scores, 'total': total_images}

    def analyze_segmentation_s2ds(dataset_path: Path):
        """s2ds: 正确图片在 train/val/test"""
        widths, heights, sharpness_scores = [], [], []
        total_images = 0

        for img_dir in [dataset_path / "train", dataset_path / "val", dataset_path / "test"]:
            if not img_dir.exists():
                continue
            for root, _, files in os.walk(img_dir):
                for file in files:
                    filepath = Path(root) / file
                    if not is_image_file(filepath):
                        continue
                    try:
                        img = cv2.imread(str(filepath))
                        if img is None:
                            continue
                        h, w = img.shape[:2]
                        widths.append(w)
                        heights.append(h)
                        sharpness_scores.append(calculate_sharpness(img))
                        total_images += 1
                    except Exception as e:
                        print(f"  处理文件时出错 {filepath}: {e}")
                        continue

        return {'widths': widths, 'heights': heights, 'sharpness': sharpness_scores, 'total': total_images}

    def analyze_segmentation_CSD(dataset_path: Path):
        """CSD: 正确图片在 images/（mask 在 masks/）"""
        widths, heights, sharpness_scores = [], [], []
        total_images = 0

        img_dir = dataset_path / "images"
        if not img_dir.exists():
            return {'widths': [], 'heights': [], 'sharpness': [], 'total': 0}

        for root, _, files in os.walk(img_dir):
            for file in files:
                filepath = Path(root) / file
                if not is_image_file(filepath):
                    continue
                try:
                    img = cv2.imread(str(filepath))
                    if img is None:
                        continue
                    h, w = img.shape[:2]
                    widths.append(w)
                    heights.append(h)
                    sharpness_scores.append(calculate_sharpness(img))
                    total_images += 1
                except Exception as e:
                    print(f"  处理文件时出错 {filepath}: {e}")
                    continue

        return {'widths': widths, 'heights': heights, 'sharpness': sharpness_scores, 'total': total_images}
    
    def analyze_segmentation_generic(dataset_path):
        """通用的segmentation数据集分析：排除masks/labels目录"""
        widths, heights, sharpness_scores = [], [], []
        total_images, skipped = 0, 0
        
        for root, dirs, files in os.walk(dataset_path):
            # 跳过明确的mask/label目录
            path_parts = Path(root).parts
            skip_dirs = ['masks', 'labels', 'label', 'mask', 'annotations', 'annotation']
            if any(part.lower() in skip_dirs for part in path_parts):
                continue
            
            for file in files:
                filepath = Path(root) / file
                if not is_image_file(filepath):
                    continue
                
                if should_exclude_generic(filepath, dataset_path):
                    skipped += 1
                    continue
                
                try:
                    img = cv2.imread(str(filepath))
                    if img is None:
                        continue
                    h, w = img.shape[:2]
                    widths.append(w)
                    heights.append(h)
                    sharpness = calculate_sharpness(img)
                    sharpness_scores.append(sharpness)
                    total_images += 1
                except Exception as e:
                    print(f"  处理文件时出错 {filepath}: {e}")
                    continue
        
        return {'widths': widths, 'heights': heights, 'sharpness': sharpness_scores, 'total': total_images}
    
    # 返回特定数据集的逻辑
    logic_map = {
        'classification_BD3': analyze_classification_BD3,
        'classification_BDD': analyze_classification_BDD,
        'classification_CCIC': analyze_classification_CCIC,
        'classification_HS-23K': analyze_classification_HS23K,
        'detection_BDW': analyze_detection_BDW,
        'detection_cubit-det': analyze_detection_cubit_det,
        'detection_MBDD2025': analyze_detection_MBDD2025,
        'segmentation_Bai-2020': analyze_segmentation_Bai2020,
        'segmentation_cubit-seg': analyze_segmentation_cubit_seg,
        'segmentation_uav75': analyze_segmentation_uav75,
        'segmentation_deepcarck': analyze_segmentation_deepcarck,
        'segmentation_masonry': analyze_segmentation_masonry,
        'segmentation_s2ds': analyze_segmentation_s2ds,
        'segmentation_CSD': analyze_segmentation_CSD,
    }
    
    return logic_map.get(dataset_name, analyze_segmentation_generic)

def analyze_dataset(dataset_path, dataset_name, max_images=None, progress_every=500):
    """分析单个数据集"""
    print(f"正在分析: {dataset_path}")
    
    # 获取特定数据集的逻辑
    analyze_func = get_dataset_specific_logic(dataset_name)
    data = analyze_func(dataset_path)
    
    print(f"  找到 {data['total']} 张图片")
    
    return data

def plot_distribution(dataset_name, data, output_dir):
    """为单个数据集生成分布图"""
    if data['total'] == 0:
        print(f"  警告: {dataset_name} 没有找到图片，跳过绘图")
        return
    
    widths = data['widths']
    heights = data['heights']
    sharpness = data['sharpness']
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Image Distribution Analysis: {dataset_name}\n(Total: {data["total"]} images)', 
                 fontsize=14, fontweight='bold')
    
    # 1. 宽度分布
    axes[0, 0].hist(widths, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    axes[0, 0].set_xlabel('Width (pixels)', fontsize=10)
    axes[0, 0].set_ylabel('Frequency', fontsize=10)
    axes[0, 0].set_title('Width Distribution', fontsize=11, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axvline(np.mean(widths), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(widths):.0f}')
    axes[0, 0].legend()
    
    # 2. 高度分布
    axes[0, 1].hist(heights, bins=50, edgecolor='black', alpha=0.7, color='lightgreen')
    axes[0, 1].set_xlabel('Height (pixels)', fontsize=10)
    axes[0, 1].set_ylabel('Frequency', fontsize=10)
    axes[0, 1].set_title('Height Distribution', fontsize=11, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axvline(np.mean(heights), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(heights):.0f}')
    axes[0, 1].legend()
    
    # 3. 清晰度分布
    axes[1, 0].hist(sharpness, bins=50, edgecolor='black', alpha=0.7, color='coral')
    axes[1, 0].set_xlabel('Sharpness (Laplacian Variance)', fontsize=10)
    axes[1, 0].set_ylabel('Frequency', fontsize=10)
    axes[1, 0].set_title('Sharpness Distribution', fontsize=11, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axvline(np.mean(sharpness), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(sharpness):.2f}')
    axes[1, 0].legend()
    
    # 4. 宽高散点图
    axes[1, 1].scatter(widths, heights, alpha=0.5, s=10, color='purple')
    axes[1, 1].set_xlabel('Width (pixels)', fontsize=10)
    axes[1, 1].set_ylabel('Height (pixels)', fontsize=10)
    axes[1, 1].set_title('Width vs Height Scatter', fontsize=11, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    output_path = output_dir / f'{dataset_name}_distribution.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  已保存分布图: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Analyze image size and sharpness distributions per dataset.")
    parser.add_argument("--max-images", type=int, default=0, help="Limit images processed per dataset (0 = no limit).")
    args = parser.parse_args()

    base_dir = Path('defect_bench/annotation_toolkit/sample_pipeline/results')
    output_dir = Path('defect_bench/annotation_toolkit/sample_pipeline/results/image_distribution_plots')
    output_dir.mkdir(exist_ok=True)
    
    # 收集所有数据集
    datasets = []
    
    # Classification数据集
    classification_dir = base_dir / 'classification'
    if classification_dir.exists():
        for dataset in classification_dir.iterdir():
            if dataset.is_dir():
                datasets.append(('classification', dataset.name, dataset))
    
    # Detection数据集
    detection_dir = base_dir / 'detection'
    if detection_dir.exists():
        for dataset in detection_dir.iterdir():
            if dataset.is_dir():
                datasets.append(('detection', dataset.name, dataset))
    
    # Segmentation数据集
    segmentation_dir = base_dir / 'segmentation'
    if segmentation_dir.exists():
        for dataset in segmentation_dir.iterdir():
            if dataset.is_dir():
                datasets.append(('segmentation', dataset.name, dataset))
    
    print(f"找到 {len(datasets)} 个数据集\n")
    
    # 分析每个数据集
    results = {}
    for category, dataset_name, dataset_path in datasets:
        full_name = f"{category}_{dataset_name}"
        print(f"\n[{full_name}]")
        data = analyze_dataset(dataset_path, full_name, max_images=args.max_images if args.max_images > 0 else None)
        results[full_name] = {
            'category': category,
            'dataset': dataset_name,
            'total_images': data['total'],
            'mean_width': float(np.mean(data['widths'])) if data['widths'] else 0,
            'mean_height': float(np.mean(data['heights'])) if data['heights'] else 0,
            'mean_sharpness': float(np.mean(data['sharpness'])) if data['sharpness'] else 0,
            'std_width': float(np.std(data['widths'])) if data['widths'] else 0,
            'std_height': float(np.std(data['heights'])) if data['heights'] else 0,
            'std_sharpness': float(np.std(data['sharpness'])) if data['sharpness'] else 0,
        }
        
        # 生成分布图
        if data['total'] > 0:
            plot_distribution(full_name, data, output_dir)
    
    # 保存统计结果
    summary_path = output_dir / 'summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n\n分析完成！")
    print(f"分布图保存在: {output_dir}")
    print(f"统计摘要保存在: {summary_path}")
    
    # 打印摘要
    print("\n=== 数据集统计摘要 ===")
    for name, stats in results.items():
        if stats['total_images'] > 0:
            print(f"\n{name}:")
            print(f"  图片数量: {stats['total_images']}")
            print(f"  平均尺寸: {stats['mean_width']:.0f} x {stats['mean_height']:.0f}")
            print(f"  平均清晰度: {stats['mean_sharpness']:.2f}")

if __name__ == '__main__':
    main()
