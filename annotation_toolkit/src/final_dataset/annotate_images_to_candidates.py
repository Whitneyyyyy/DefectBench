#!/usr/bin/env python3
"""
Image annotation and refinement tool for data_sample.

Features:
  - Read images from defect_bench/data_sample/images
  - Read and visualize bbox annotations from defect_bench/data_sample/labels
  - Read and visualize colored masks from defect_bench/data_sample/masks
  - Support manual bbox refinement (drag, resize, delete, add)
  - Support manual mask refinement (point operations, brush operations, maintain class colors)

Usage:
  cd defect_bench
  python annotation_toolkit/backend/annotate_images_to_candidates.py

Then access in browser: http://<server_ip>:5000
"""

import base64
import io
import json
import os
import cv2
import numpy as np
import sys
from pathlib import Path

# defect_bench path resolution
_THIS_FILE = Path(__file__).resolve()
_DEFAULT_OPEN_DATASET_ROOT = next((p / "data_sample" for p in [_THIS_FILE.parent, *_THIS_FILE.parents] if p.name == "defect_bench"), _THIS_FILE.parent)
OPEN_DATASET_ROOT = Path(os.environ.get("DEFECT_BENCH_OPEN_DATASET_ROOT", str(_DEFAULT_OPEN_DATASET_ROOT)))

from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

import pandas as pd
from flask import Flask, jsonify, render_template_string, request
from PIL import Image, ImageDraw

# Import SAM service and Detection Agent
# Import from local defect_bench modules
try:
    from sam_logic import sam_service
    SAM_AVAILABLE = True
except ImportError:
    print("Warning: SAM service not available. Mask refinement will be disabled.")
    SAM_AVAILABLE = False
    sam_service = None

try:
    from detection_agent import detection_service
    import asyncio
    DETECTION_AVAILABLE = True
except ImportError:
    print("Warning: Detection service not available. Detection will be disabled.")
    DETECTION_AVAILABLE = False
    detection_service = None

app = Flask(__name__)

# Directory configuration
BASE_DIR = OPEN_DATASET_ROOT
IMAGES_DIR = BASE_DIR / "images"
LABELS_DIR = BASE_DIR / "labels"
MASKS_DIR = BASE_DIR / "masks"
CANDIDATES_CSV = BASE_DIR / "candidates.csv"

# Class color mapping (RGB)
CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
    "Crack": (255, 0, 0),             # Red
    "Material_loss": (255, 140, 0),   # Orange
    "Stain": (30, 144, 255),          # Blue
    "External Fixings": (0, 200, 0),  # Green
}

# sub_type (detection model output) -> primary_class mapping
SUBTYPE_TO_PRIMARY_CLASS: Dict[str, str] = {
    "Concrete_Crack": "Crack",
    "Concrete_Delamination": "Material_loss",
    "Concrete_Spalling": "Material_loss",
    "Rust_Stain": "Stain",
    "Vegeterian": "External Fixings",
    "Degraded_Plaster": "Material_loss",
    "Craquelure": "Crack",
    "Tile_Crack": "Crack",
    "Tile_spalling": "Material_loss",
    "Water_Stain": "Stain",
    "Bulging": "Material_loss",
    "Contaminants": "External Fixings",
}
DEFAULT_COLOR = (255, 255, 0)  # Yellow

# Image list cache
_image_list_cache: List[Dict[str, Any]] = []
_image_list_loaded = False


def _load_image_list():
    """Load all images from images directory and categorize by naming pattern"""
    global _image_list_cache, _image_list_loaded
    if _image_list_loaded:
        return
    
    print("[Cache] Loading image list...")
    _image_list_cache = []
    
    if not IMAGES_DIR.exists():
        print(f"[Cache] Images directory not found: {IMAGES_DIR}")
        _image_list_loaded = True
        return
    
    # Get all image files
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        image_files.extend(IMAGES_DIR.glob(f"*{ext}"))
    
    for img_file in sorted(image_files):
        stem = img_file.stem
        img_name = img_file.name
        
        # Determine dataset based on naming pattern
        if stem.startswith('sp_'):
            dataset = 'sp_renamed'  # Renamed from DJI files
        elif stem.startswith('hk'):
            dataset = 'hk_original'  # Original HK files
        else:
            dataset = 'other'  # Other naming patterns
        
        # Check if label exists
        label_path = LABELS_DIR / f"{stem}.json"
        has_label = label_path.exists()
        
        # Check if mask exists
        mask_png = MASKS_DIR / f"{stem}_mask.png"
        mask_jpg = MASKS_DIR / f"{stem}_mask.jpg"
        has_mask = mask_png.exists() or mask_jpg.exists()
        
        _image_list_cache.append({
            'stem': stem,
            'name': img_name,
            'path': str(img_file),
            'dataset': dataset,
            'has_label': has_label,
            'has_mask': has_mask
        })
    
    print(f"[Cache] Loaded {len(_image_list_cache)} images")
    _image_list_loaded = True


def _get_filtered_images(dataset: Optional[str] = None, primary_class: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get filtered image list based on dataset and primary_class"""
    _load_image_list()
    
    filtered = _image_list_cache.copy()
    
    # Filter by dataset
    if dataset:
        filtered = [img for img in filtered if img['dataset'] == dataset]
    
    # Filter by primary_class (need to check labels)
    if primary_class:
        matching_images = []
        for img in filtered:
            label_path = LABELS_DIR / f"{img['stem']}.json"
            if not label_path.exists():
                continue
            
            try:
                with open(label_path, 'r', encoding='utf-8') as f:
                    label_data = json.load(f)
                bboxes = label_data.get('bboxes', [])
                
                # Check if any bbox has the specified primary_class
                for bbox in bboxes:
                    taxonomy = bbox.get('taxonomy', {})
                    if taxonomy.get('primary_class') == primary_class:
                        matching_images.append(img)
                        break
            except Exception:
                continue
        
        filtered = matching_images
    
    return filtered


def _load_bboxes_for_image(stem: str) -> List[Dict[str, Any]]:
    """Load bboxes for a given image stem"""
    label_path = LABELS_DIR / f"{stem}.json"
    if not label_path.exists():
        return []
    
    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        bboxes = data.get('bboxes', [])
        bbox_list = []
        for bbox_data in bboxes:
            bbox_xywh = bbox_data.get('bbox', [])
            if len(bbox_xywh) != 4:
                continue
            
            taxonomy = bbox_data.get('taxonomy', {})
            bbox_list.append({
                'bbox': bbox_xywh,  # [x, y, w, h] format
                'primary_class': taxonomy.get('primary_class'),
                'sub_type': taxonomy.get('sub_type', ''),
            })
        
        return bbox_list
    except Exception as e:
        print(f"[Error] Failed to load bboxes for {stem}: {e}")
        return []


def _load_mask_for_image(stem: str) -> Optional[np.ndarray]:
    """Load mask for a given image stem"""
    mask_png = MASKS_DIR / f"{stem}_mask.png"
    mask_jpg = MASKS_DIR / f"{stem}_mask.jpg"
    
    mask_path = None
    if mask_png.exists():
        mask_path = mask_png
    elif mask_jpg.exists():
        mask_path = mask_jpg
    
    if not mask_path:
        return None
    
    try:
        mask_bgr = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
        if mask_bgr is None:
            return None
        mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)
        return mask_rgb
    except Exception as e:
        print(f"[Error] Failed to load mask for {stem}: {e}")
        return None


def _get_color_for_class(primary_class: Optional[str]) -> Tuple[int, int, int]:
    """Get color for the given class"""
    if primary_class:
        return CLASS_COLORS.get(primary_class, DEFAULT_COLOR)
    return DEFAULT_COLOR


def _get_class_for_color(color: Tuple[int, int, int], tolerance: int = 10) -> Optional[str]:
    """Get class name from color (for mask editing)"""
    r, g, b = color
    for class_name, class_color in CLASS_COLORS.items():
        cr, cg, cb = class_color
        if abs(r - cr) <= tolerance and abs(g - cg) <= tolerance and abs(b - cb) <= tolerance:
            return class_name
    return None


# HTML Template (contains frontend interaction logic)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <title>Image Annotation to Candidates Tool</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1800px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 2px solid #ddd;
        }
        .controls {
            display: flex;
            align-items: center;
            gap: 10px;
            flex-wrap: wrap;
        }
        select, button {
            padding: 8px 12px;
            font-size: 14px;
            border: 1px solid #ccc;
            border-radius: 4px;
        }
        button {
            background-color: #4CAF50;
            color: white;
            cursor: pointer;
        }
        button:hover { background-color: #45a049; }
        button:disabled { background-color: #ccc; cursor: not-allowed; }
        .info {
            margin-bottom: 15px;
            padding: 10px;
            background-color: #e3f2fd;
            border-radius: 4px;
            font-size: 14px;
        }
        .main-content {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        .panel {
            border: 2px solid #ddd;
            border-radius: 4px;
            padding: 15px;
            background-color: #fafafa;
        }
        .panel-title {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid #ddd;
        }
        .toolbar {
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
            flex-wrap: wrap;
        }
        .toolbar button {
            padding: 6px 12px;
            font-size: 13px;
        }
        .toolbar button.active {
            background-color: #2196F3;
        }
        .canvas-container {
            position: relative;
            background-color: #000;
            border: 2px solid #333;
            border-radius: 4px;
            overflow: hidden;
            text-align: center;
        }
        canvas {
            max-width: 100%;
            max-height: 600px;
            display: block;
            cursor: crosshair;
        }
        .color-legend {
            margin-top: 15px;
            padding: 10px;
            background-color: #fff;
            border-radius: 4px;
            font-size: 12px;
        }
        .color-item {
            display: inline-block;
            margin-right: 15px;
            margin-bottom: 5px;
        }
        .color-box {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 1px solid #333;
            vertical-align: middle;
            margin-right: 5px;
        }
        .status {
            margin-top: 10px;
            padding: 8px;
            background-color: #fff3cd;
            border-radius: 4px;
            font-size: 13px;
        }
        .status.success {
            background-color: #d4edda;
        }
        .status.error {
            background-color: #f8d7da;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Image Annotation to Candidates Tool</h1>
            <div class="controls">
                <select id="datasetSelect">
                    <option value="">All Datasets</option>
                    <option value="sp_renamed">SP Renamed (sp_*)</option>
                    <option value="hk_original">HK Original (hk*)</option>
                    <option value="other">Other</option>
                </select>
                <select id="primaryClassSelect">
                    <option value="">All Classes</option>
                    <option value="Crack">Crack</option>
                    <option value="Material_loss">Material_loss</option>
                    <option value="Stain">Stain</option>
                    <option value="External Fixings">External Fixings</option>
                </select>
                <button onclick="loadImage(0)">Load First</button>
                <button onclick="saveChanges()">Save Changes</button>
                <button onclick="addToCandidates()" style="background-color: #FF9800; color: white;">Add to Candidates (C)</button>
            </div>
        </div>
        
        <div class="info" id="infoDiv">
            Please select dataset and class, then load image
        </div>
        
        <div class="main-content">
            <!-- BBox Editing Panel -->
            <div class="panel">
                <div class="panel-title">BBox Editing</div>
                <div class="toolbar">
                    <button id="bboxViewBtn" class="active" onclick="setBBoxMode('view')">View</button>
                    <button id="bboxDrawBtn" onclick="setBBoxMode('draw')">Draw</button>
                    <button id="bboxDeleteBtn" onclick="setBBoxMode('delete')">Delete</button>
                    <button onclick="clearBBoxes()">Clear All</button>
                </div>
                <div style="margin-bottom: 10px;">
                    <label>Detection Model:</label>
                    <select id="detectionModelSelect">
                        <option value="ensemble">Ensemble</option>
                        <option value="intersection">Intersection</option>
                        <option value="yolo12m">YOLO12m</option>
                        <option value="yolo11m">YOLO11m</option>
                        <option value="faster_rcnn">Faster R-CNN</option>
                        <option value="rtdetr">RT-DETR</option>
                    </select>
                    <button onclick="runDetection()" style="margin-left: 10px; background-color: #2196F3; color: white;">Run Detection</button>
                </div>
                <div class="canvas-container">
                    <canvas id="bboxCanvas"></canvas>
                </div>
                <div class="color-legend">
                    <strong>BBox Color Legend:</strong>
                    <span class="color-item"><span class="color-box" style="background-color: rgb(255,0,0);"></span>Crack</span>
                    <span class="color-item"><span class="color-box" style="background-color: rgb(255,140,0);"></span>Material_loss</span>
                    <span class="color-item"><span class="color-box" style="background-color: rgb(30,144,255);"></span>Stain</span>
                    <span class="color-item"><span class="color-box" style="background-color: rgb(0,200,0);"></span>External Fixings</span>
                </div>
                <div class="status" id="bboxStatus"></div>
            </div>
            
            <!-- Label Selection Dialog -->
            <div id="labelDialog" style="display: none; position: fixed; inset: 0; background-color: rgba(0,0,0,0.5); z-index: 1000; align-items: center; justify-content: center;">
                <div style="background-color: white; border-radius: 8px; padding: 24px; max-width: 400px; width: 90%;">
                    <h3 style="font-size: 18px; font-weight: bold; margin-bottom: 16px;">Select Defect Class</h3>
                    <div id="labelOptions" style="display: flex; flex-direction: column; gap: 8px; margin-bottom: 16px;">
                        <!-- Options will be dynamically generated by JavaScript -->
                    </div>
                    <button onclick="cancelLabelSelection()" style="width: 100%; padding: 8px; background-color: #e0e0e0; border: none; border-radius: 4px; cursor: pointer; font-size: 14px;">Cancel</button>
                </div>
            </div>
            
            <!-- Mask Editing Panel -->
            <div class="panel">
                <div class="panel-title">Mask Editing</div>
                <div class="toolbar">
                    <button id="maskViewBtn" class="active" onclick="setMaskMode('view')">View</button>
                    <button id="maskPointPosBtn" onclick="setMaskMode('point-pos')">Positive Point</button>
                    <button id="maskPointNegBtn" onclick="setMaskMode('point-neg')">Negative Point</button>
                    <button id="maskRefineBtn" onclick="executeMaskRefine()" style="background-color: #4CAF50; color: white; display: none;">Execute Refine (SAM)</button>
                    <button id="maskBrushBtn" onclick="setMaskMode('brush-add')">Brush</button>
                    <button id="maskEraseBtn" onclick="setMaskMode('brush-remove')">Erase</button>
                    <button onclick="clearMask()">Clear Mask</button>
                </div>
                <div style="margin-bottom: 10px;">
                    <label>Class Selection (for adding mask):</label>
                    <select id="maskClassSelect">
                        <option value="Crack">Crack (Red)</option>
                        <option value="Material_loss">Material_loss (Orange)</option>
                        <option value="Stain">Stain (Blue)</option>
                        <option value="External Fixings">External Fixings (Green)</option>
                    </select>
                    <label style="margin-left: 15px;">Brush Size:</label>
                    <input type="range" id="brushSize" min="1" max="100" value="20" style="width: 100px;">
                    <span id="brushSizeValue">20px</span>
                </div>
                <div class="canvas-container" id="maskContainer" style="position: relative;">
                    <canvas id="maskCanvas"></canvas>
                    <canvas id="maskBrushLayer" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; cursor: crosshair; z-index: 10;"></canvas>
                </div>
                <div class="status" id="maskStatus"></div>
            </div>
        </div>
    </div>

    <script>
        // Global state
        let currentIndex = 0;
        let currentDataset = null;
        let currentPrimaryClass = null;
        let currentImageData = null;
        let currentBBoxes = [];
        let currentMask = null;
        let originalMask = null;
        
        // Canvas and context
        const bboxCanvas = document.getElementById('bboxCanvas');
        const bboxCtx = bboxCanvas.getContext('2d');
        const maskCanvas = document.getElementById('maskCanvas');
        const maskCtx = maskCanvas.getContext('2d');
        const maskBrushLayer = document.getElementById('maskBrushLayer');
        const maskBrushCtx = maskBrushLayer.getContext('2d');
        
        // Modes
        let bboxMode = 'view'; // 'view', 'draw', 'delete'
        let maskMode = 'view'; // 'view', 'point', 'brush'
        
        // Interaction state
        let bboxDragging = null;
        let bboxDrawing = null;
        let bboxResizing = null;
        let pendingBBox = null;
        
        // Available defect classes
        const defectClasses = [
            'Crack',
            'Material_loss',
            'Stain',
            'External Fixings'
        ];
        let maskDrawing = false;
        let maskIsDrawing = false;
        let maskCursorPos = null;
        let maskPoints = [];
        let maskLastBrushPos = null;
        
        // Color mapping
        const classColors = {
            'Crack': [255, 0, 0],
            'Material_loss': [255, 140, 0],
            'Stain': [30, 144, 255],
            'External Fixings': [0, 200, 0]
        };
        
        // Initialize
        document.getElementById('datasetSelect').addEventListener('change', async (e) => {
            currentDataset = e.target.value || null;
            currentIndex = 0;
            await updateImageList();
        });
        
        document.getElementById('brushSize').addEventListener('input', (e) => {
            document.getElementById('brushSizeValue').textContent = e.target.value + 'px';
        });
        
        document.getElementById('primaryClassSelect').addEventListener('change', async (e) => {
            currentPrimaryClass = e.target.value || null;
            currentIndex = 0;
            await updateImageList();
            if (currentImageData && currentImageData.total > 0) {
                loadImage(0);
            } else {
                currentImageData = null;
                currentBBoxes = [];
                currentMask = null;
                drawBBoxCanvas();
                drawMaskCanvas();
                document.getElementById('infoDiv').textContent = `No images found with ${currentPrimaryClass || 'specified class'}`;
            }
        });
        
        // Update image list
        async function updateImageList() {
            const params = new URLSearchParams();
            if (currentDataset) params.append('dataset', currentDataset);
            if (currentPrimaryClass) params.append('primary_class', currentPrimaryClass);
            
            try {
                const response = await fetch(`/api/images?${params}`);
                const data = await response.json();
                if (data.total > 0) {
                    loadImage(0);
                } else {
                    currentImageData = null;
                    currentBBoxes = [];
                    currentMask = null;
                    drawBBoxCanvas();
                    drawMaskCanvas();
                    document.getElementById('infoDiv').textContent = `No images found`;
                }
            } catch (error) {
                console.error('Failed to load image list:', error);
            }
        }
        
        // Load image
        async function loadImage(index) {
            try {
                const params = new URLSearchParams({
                    index: index.toString()
                });
                if (currentDataset) params.append('dataset', currentDataset);
                if (currentPrimaryClass) params.append('primary_class', currentPrimaryClass);
                
                const response = await fetch(`/api/image/${index}?${params}`);
                const data = await response.json();
                
                if (data.error) {
                    showStatus('bboxStatus', data.error, 'error');
                    return;
                }
                
                currentIndex = index;
                currentImageData = data;
                currentBBoxes = data.bboxes || [];
                
                // Ensure each bbox has id
                currentBBoxes.forEach((bbox, idx) => {
                    if (bbox.id === undefined) {
                        bbox.id = idx;
                    }
                });
                
                // Clear pending box
                pendingBBox = null;
                cancelLabelSelection();
                
                // Convert mask Image to canvas if exists
                if (data.mask) {
                    const maskImg = await loadImageFromBase64(data.mask);
                    const img = new Image();
                    await new Promise((resolve) => {
                        img.onload = () => {
                            currentMask = document.createElement('canvas');
                            currentMask.width = img.width;
                            currentMask.height = img.height;
                            const maskCtx2 = currentMask.getContext('2d');
                            maskCtx2.drawImage(maskImg, 0, 0, img.width, img.height);
                            resolve();
                        };
                        img.src = currentImageData.image;
                    });
                    originalMask = await cloneImage(maskImg);
                } else {
                    currentMask = null;
                    originalMask = null;
                }
                
                // Update info
                document.getElementById('infoDiv').innerHTML = `
                    <strong>Image ${index + 1} / ${data.total}</strong> | 
                    ${data.filename} | 
                    BBox: ${currentBBoxes.length} | 
                    Mask: ${currentMask ? 'Yes' : 'No'}
                `;
                
                // Draw
                drawBBoxCanvas();
                drawMaskCanvas();
                
                showStatus('bboxStatus', 'Image loaded successfully', 'success');
                showStatus('maskStatus', 'Image loaded successfully', 'success');
                
            } catch (error) {
                console.error('Failed to load image:', error);
                showStatus('bboxStatus', 'Load failed: ' + error.message, 'error');
            }
        }
        
        // Load image from base64
        function loadImageFromBase64(base64Str) {
            return new Promise((resolve) => {
                const img = new Image();
                img.onload = () => resolve(img);
                img.src = base64Str;
            });
        }
        
        // Clone Image object
        function cloneImage(img) {
            return new Promise((resolve) => {
                const canvas = document.createElement('canvas');
                canvas.width = img.width;
                canvas.height = img.height;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0);
                const newImg = new Image();
                newImg.onload = () => resolve(newImg);
                newImg.src = canvas.toDataURL();
            });
        }
        
        // Draw BBox Canvas (keeping the same drawing logic from original)
        let imageWidth = 0;
        let currentDisplayScale = 1;
        let currentAdjustedLineWidth = 2;
        
        function drawBBoxCanvas() {
            if (!currentImageData || !currentImageData.image) return;
            
            const img = new Image();
            img.onload = () => {
                imageWidth = img.width;
                bboxCanvas.width = img.width;
                bboxCanvas.height = img.height;
                
                const maxWidth = bboxCanvas.parentElement.clientWidth - 20;
                const maxHeight = 600;
                const scale = Math.min(maxWidth / img.width, maxHeight / img.height, 1);
                bboxCanvas.style.width = (img.width * scale) + 'px';
                bboxCanvas.style.height = (img.height * scale) + 'px';
                
                currentDisplayScale = scale;
                
                const baseLineWidth = 2;
                const baseFontSize = 12;
                const baseHandleSize = 6;
                const baseLabelHeight = 18;
                
                const adjustedLineWidth = Math.max(baseLineWidth, Math.min(baseLineWidth / scale, 8));
                const adjustedFontSize = Math.max(baseFontSize, Math.min(baseFontSize / scale, 24));
                const adjustedHandleSize = Math.max(baseHandleSize, Math.min(baseHandleSize / scale, 12));
                const adjustedLabelHeight = Math.max(baseLabelHeight, Math.min(baseLabelHeight / scale, 30));
                const adjustedLabelPadding = Math.max(2, Math.min(2 / scale, 6));
                
                currentAdjustedLineWidth = adjustedLineWidth;
                
                bboxCtx.clearRect(0, 0, bboxCanvas.width, bboxCanvas.height);
                bboxCtx.drawImage(img, 0, 0, bboxCanvas.width, bboxCanvas.height);
                
                currentBBoxes.forEach((bbox, idx) => {
                    const [x, y, w, h] = bbox.bbox;
                    const color = bbox.primary_class ? classColors[bbox.primary_class] || [255, 255, 0] : [255, 255, 0];
                    
                    bboxCtx.strokeStyle = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
                    bboxCtx.lineWidth = adjustedLineWidth;
                    bboxCtx.strokeRect(x, y, w, h);
                    
                    if (bboxMode === 'view') {
                        const points = [
                            [x, y], [x + w, y], [x, y + h], [x + w, y + h]
                        ];
                        bboxCtx.fillStyle = 'white';
                        bboxCtx.strokeStyle = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
                        bboxCtx.lineWidth = adjustedLineWidth;
                        points.forEach(([px, py]) => {
                            bboxCtx.fillRect(px - adjustedHandleSize/2, py - adjustedHandleSize/2, adjustedHandleSize, adjustedHandleSize);
                            bboxCtx.strokeRect(px - adjustedHandleSize/2, py - adjustedHandleSize/2, adjustedHandleSize, adjustedHandleSize);
                        });
                    }
                    
                    const label = bbox.primary_class || 'Unknown';
                    const labelWidth = label.length * (adjustedFontSize * 0.6);
                    bboxCtx.fillStyle = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
                    bboxCtx.fillRect(x, y - adjustedLabelHeight, labelWidth, adjustedLabelHeight);
                    bboxCtx.fillStyle = 'black';
                    bboxCtx.font = `${adjustedFontSize}px Arial`;
                    bboxCtx.fillText(label, x + adjustedLabelPadding, y - adjustedLabelPadding);
                });
                
                if (pendingBBox) {
                    const [x, y, w, h] = pendingBBox.bbox;
                    bboxCtx.strokeStyle = 'yellow';
                    bboxCtx.lineWidth = adjustedLineWidth;
                    bboxCtx.setLineDash([5 * (1/scale), 5 * (1/scale)]);
                    bboxCtx.strokeRect(x, y, w, h);
                    bboxCtx.setLineDash([]);
                    
                    const labelText = 'Waiting for class selection...';
                    const labelWidth = labelText.length * (adjustedFontSize * 0.6);
                    bboxCtx.fillStyle = 'yellow';
                    bboxCtx.fillRect(x, y - adjustedLabelHeight, labelWidth, adjustedLabelHeight);
                    bboxCtx.fillStyle = 'black';
                    bboxCtx.font = `${adjustedFontSize}px Arial`;
                    bboxCtx.fillText(labelText, x + adjustedLabelPadding, y - adjustedLabelPadding);
                }
            };
            img.src = currentImageData.image;
        }
        
        // Get mouse coordinates in original image
        function getImageCoords(e) {
            const rect = bboxCanvas.getBoundingClientRect();
            const displayScale = bboxCanvas.width / rect.width;
            const x = (e.clientX - rect.left) * displayScale;
            const y = (e.clientY - rect.top) * displayScale;
            return { x, y, displayX: x, displayY: y };
        }
        
        // Get mask image coordinates
        function getMaskImageCoords(e) {
            const rect = maskCanvas.getBoundingClientRect();
            const scaleX = maskCanvas.width / rect.width;
            const scaleY = maskCanvas.height / rect.height;
            return {
                x: (e.clientX - rect.left) * scaleX,
                y: (e.clientY - rect.top) * scaleY
            };
        }
        
        // Draw Mask Canvas
        function drawMaskCanvas() {
            if (!currentImageData || !currentImageData.image) return;
            
            const img = new Image();
            img.onload = () => {
                maskCanvas.width = img.width;
                maskCanvas.height = img.height;
                maskBrushLayer.width = img.width;
                maskBrushLayer.height = img.height;
                
                const maxWidth = maskCanvas.parentElement.clientWidth - 20;
                const maxHeight = 600;
                const scale = Math.min(maxWidth / img.width, maxHeight / img.height, 1);
                maskCanvas.style.width = (img.width * scale) + 'px';
                maskCanvas.style.height = (img.height * scale) + 'px';
                maskBrushLayer.style.width = (img.width * scale) + 'px';
                maskBrushLayer.style.height = (img.height * scale) + 'px';
                
                maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
                maskCtx.drawImage(img, 0, 0);
                
                if (currentMask && currentMask.width === img.width && currentMask.height === img.height) {
                    maskCtx.globalAlpha = 0.4;
                    maskCtx.drawImage(currentMask, 0, 0);
                    maskCtx.globalAlpha = 1.0;
                } else if (currentMask) {
                    maskCtx.globalAlpha = 0.4;
                    maskCtx.drawImage(currentMask, 0, 0, img.width, img.height);
                    maskCtx.globalAlpha = 1.0;
                }
                
                maskPoints.forEach(p => {
                    maskCtx.beginPath();
                    maskCtx.arc(p.x, p.y, 8, 0, 2 * Math.PI);
                    maskCtx.fillStyle = p.label === 1 ? '#00ff00' : '#ff0000';
                    maskCtx.fill();
                    maskCtx.strokeStyle = 'white';
                    maskCtx.lineWidth = 2;
                    maskCtx.stroke();
                });
            };
            img.src = currentImageData.image;
        }
        
        // BBox mode setting
        function setBBoxMode(mode) {
            bboxMode = mode;
            document.getElementById('bboxViewBtn').classList.toggle('active', mode === 'view');
            document.getElementById('bboxDrawBtn').classList.toggle('active', mode === 'draw');
            document.getElementById('bboxDeleteBtn').classList.toggle('active', mode === 'delete');
            bboxCanvas.style.cursor = mode === 'draw' ? 'crosshair' : mode === 'delete' ? 'not-allowed' : 'default';
        }
        
        // Mask mode setting
        function setMaskMode(mode) {
            maskMode = mode;
            document.getElementById('maskViewBtn').classList.toggle('active', mode === 'view');
            document.getElementById('maskPointPosBtn').classList.toggle('active', mode === 'point-pos');
            document.getElementById('maskPointNegBtn').classList.toggle('active', mode === 'point-neg');
            document.getElementById('maskBrushBtn').classList.toggle('active', mode === 'brush-add');
            document.getElementById('maskEraseBtn').classList.toggle('active', mode === 'brush-remove');
            
            const isPointMode = (mode === 'point-pos' || mode === 'point-neg');
            document.getElementById('maskRefineBtn').style.display = isPointMode ? 'inline-block' : 'none';
            
            const isInteractive = (mode === 'point-pos' || mode === 'point-neg' || mode === 'brush-add' || mode === 'brush-remove');
            maskBrushLayer.style.pointerEvents = isInteractive ? 'auto' : 'none';
            maskBrushLayer.style.cursor = isInteractive ? 'crosshair' : 'default';
        }
        
        // Check if click is on resize handle
        function getResizeHandle(bbox, x, y) {
            const [bx, by, bw, bh] = bbox.bbox;
            const size = 8;
            const handles = [
                { name: 'tl', x: bx, y: by },
                { name: 'tr', x: bx + bw, y: by },
                { name: 'bl', x: bx, y: by + bh },
                { name: 'br', x: bx + bw, y: by + bh }
            ];
            for (const handle of handles) {
                if (Math.abs(x - handle.x) < size && Math.abs(y - handle.y) < size) {
                    return handle.name;
                }
            }
            return null;
        }
        
        // BBox Canvas events (keeping same logic from original)
        bboxCanvas.addEventListener('mousedown', (e) => {
            if (bboxMode === 'draw') {
                const coords = getImageCoords(e);
                bboxDrawing = { startX: coords.displayX, startY: coords.displayY };
            } else if (bboxMode === 'delete') {
                const coords = getImageCoords(e);
                const clickedIndex = currentBBoxes.findIndex(bbox => {
                    const [bx, by, bw, bh] = bbox.bbox;
                    return coords.x >= bx && coords.x <= bx + bw && coords.y >= by && coords.y <= by + bh;
                });
                if (clickedIndex >= 0) {
                    currentBBoxes.splice(clickedIndex, 1);
                    drawBBoxCanvas();
                    showStatus('bboxStatus', 'Deleted one BBox', 'success');
                }
            } else if (bboxMode === 'view') {
                const coords = getImageCoords(e);
                for (let i = 0; i < currentBBoxes.length; i++) {
                    const handle = getResizeHandle(currentBBoxes[i], coords.x, coords.y);
                    if (handle) {
                        bboxResizing = { index: i, handle: handle, startX: coords.x, startY: coords.y, startBox: [...currentBBoxes[i].bbox] };
                        return;
                    }
                }
                const clickedIndex = currentBBoxes.findIndex(bbox => {
                    const [bx, by, bw, bh] = bbox.bbox;
                    return coords.x >= bx && coords.x <= bx + bw && coords.y >= by && coords.y <= by + bh;
                });
                if (clickedIndex >= 0) {
                    bboxDragging = { index: clickedIndex, startX: coords.x, startY: coords.y, startBox: [...currentBBoxes[clickedIndex].bbox] };
                }
            }
        });
        
        bboxCanvas.addEventListener('mousemove', (e) => {
            if (bboxDrawing) {
                const coords = getImageCoords(e);
                drawBBoxCanvas();
                bboxCtx.strokeStyle = 'yellow';
                bboxCtx.lineWidth = currentAdjustedLineWidth;
                const dashSize = Math.max(5, Math.min(5 / currentDisplayScale, 15));
                bboxCtx.setLineDash([dashSize, dashSize]);
                bboxCtx.strokeRect(bboxDrawing.startX, bboxDrawing.startY, coords.displayX - bboxDrawing.startX, coords.displayY - bboxDrawing.startY);
                bboxCtx.setLineDash([]);
            } else if (bboxDragging) {
                const coords = getImageCoords(e);
                const bbox = currentBBoxes[bboxDragging.index];
                const [bx, by, bw, bh] = bboxDragging.startBox;
                const dx = coords.x - bboxDragging.startX;
                const dy = coords.y - bboxDragging.startY;
                bbox.bbox = [bx + dx, by + dy, bw, bh];
                drawBBoxCanvas();
            } else if (bboxResizing) {
                const coords = getImageCoords(e);
                const bbox = currentBBoxes[bboxResizing.index];
                const [bx, by, bw, bh] = bboxResizing.startBox;
                const dx = coords.x - bboxResizing.startX;
                const dy = coords.y - bboxResizing.startY;
                if (bboxResizing.handle === 'tl') {
                    bbox.bbox = [bx + dx, by + dy, bw - dx, bh - dy];
                } else if (bboxResizing.handle === 'tr') {
                    bbox.bbox = [bx, by + dy, bw + dx, bh - dy];
                } else if (bboxResizing.handle === 'bl') {
                    bbox.bbox = [bx + dx, by, bw - dx, bh + dy];
                } else if (bboxResizing.handle === 'br') {
                    bbox.bbox = [bx, by, bw + dx, bh + dy];
                }
                if (bbox.bbox[2] < 10) bbox.bbox[2] = 10;
                if (bbox.bbox[3] < 10) bbox.bbox[3] = 10;
                drawBBoxCanvas();
            }
        });
        
        bboxCanvas.addEventListener('mouseup', (e) => {
            if (bboxDrawing) {
                const coords = getImageCoords(e);
                const x = Math.min(bboxDrawing.startX, coords.displayX);
                const y = Math.min(bboxDrawing.startY, coords.displayY);
                const w = Math.abs(coords.displayX - bboxDrawing.startX);
                const h = Math.abs(coords.displayY - bboxDrawing.startY);
                if (w > 10 && h > 10) {
                    const bboxCoords = [x, y, w, h];
                    const newId = currentBBoxes.length > 0 ? Math.max(...currentBBoxes.map(b => b.id || 0)) + 1 : 0;
                    pendingBBox = { id: newId, bbox: bboxCoords };
                    showLabelDialog();
                }
                bboxDrawing = null;
            }
            bboxDragging = null;
            bboxResizing = null;
        });
        
        // Mask Canvas interaction events (keeping same logic from original)
        maskBrushLayer.addEventListener('mousedown', (e) => {
            const pos = getMaskImageCoords(e);
            if (maskMode === 'point-pos' || maskMode === 'point-neg') {
                const label = maskMode === 'point-pos' ? 1 : 0;
                maskPoints.push({ x: pos.x, y: pos.y, label });
                drawMaskCanvas();
                showStatus('maskStatus', `Added ${maskPoints.length} points, click "Execute Refine" to apply SAM`, 'success');
            } else if (maskMode === 'brush-add' || maskMode === 'brush-remove') {
                maskIsDrawing = true;
                maskLastBrushPos = null;
                maskBrushCtx.clearRect(0, 0, maskBrushLayer.width, maskBrushLayer.height);
                drawMaskBrush(pos.x, pos.y);
            }
        });
        
        // Execute SAM refine
        async function executeMaskRefine() {
            if (maskPoints.length === 0) {
                showStatus('maskStatus', 'Please add points first', 'error');
                return;
            }
            showStatus('maskStatus', 'Executing SAM refine...', '');
            try {
                const points = maskPoints.map(p => [Math.round(p.x), Math.round(p.y)]);
                const labels = maskPoints.map(p => p.label);
                const bboxes = currentBBoxes.map(b => {
                    const [x, y, w, h] = b.bbox;
                    return [x, y, x + w, y + h];
                });
                const response = await fetch('/api/refine_mask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        image_data: currentImageData.image,
                        mask_data: currentMask ? currentMask.toDataURL() : null,
                        points: points,
                        labels: labels,
                        bboxes: bboxes
                    })
                });
                const responseData = await response.json();
                if (responseData.success) {
                    const img = new Image();
                    img.onload = () => {
                        const tempCanvas = document.createElement('canvas');
                        tempCanvas.width = img.width;
                        tempCanvas.height = img.height;
                        const tempCtx = tempCanvas.getContext('2d');
                        tempCtx.drawImage(img, 0, 0);
                        const imageData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
                        const pixelData = imageData.data;
                        const selectedClass = document.getElementById('maskClassSelect').value;
                        const color = classColors[selectedClass];
                        for (let i = 0; i < pixelData.length; i += 4) {
                            if (pixelData[i] > 0 || pixelData[i + 1] > 0 || pixelData[i + 2] > 0) {
                                pixelData[i] = color[0];
                                pixelData[i + 1] = color[1];
                                pixelData[i + 2] = color[2];
                                pixelData[i + 3] = 255;
                            }
                        }
                        tempCtx.putImageData(imageData, 0, 0);
                        if (!currentMask || currentMask.width !== img.width || currentMask.height !== img.height) {
                            currentMask = document.createElement('canvas');
                            currentMask.width = img.width;
                            currentMask.height = img.height;
                        }
                        const maskCtx = currentMask.getContext('2d');
                        maskCtx.clearRect(0, 0, currentMask.width, currentMask.height);
                        maskCtx.drawImage(tempCanvas, 0, 0);
                        originalMask = cloneImage(currentMask);
                        maskPoints = [];
                        drawMaskCanvas();
                        showStatus('maskStatus', 'SAM refine completed', 'success');
                    };
                    img.src = responseData.mask;
                } else {
                    showStatus('maskStatus', 'Refine failed: ' + responseData.error, 'error');
                }
            } catch (error) {
                console.error('SAM refine error:', error);
                showStatus('maskStatus', 'Refine failed: ' + error.message, 'error');
            }
        }
        
        maskBrushLayer.addEventListener('mousemove', (e) => {
            const pos = getMaskImageCoords(e);
            maskCursorPos = pos;
            if (maskIsDrawing) {
                drawMaskBrush(pos.x, pos.y);
            } else {
                updateBrushPreview();
            }
        });
        
        maskBrushLayer.addEventListener('mouseup', () => {
            if (maskIsDrawing) {
                maskIsDrawing = false;
                const brushDataURL = maskBrushLayer.toDataURL('image/png');
                applyBrushToMask(brushDataURL);
                maskBrushCtx.clearRect(0, 0, maskBrushLayer.width, maskBrushLayer.height);
            }
            maskLastBrushPos = null;
        });
        
        maskBrushLayer.addEventListener('mouseleave', () => {
            if (maskIsDrawing) {
                maskIsDrawing = false;
                const brushDataURL = maskBrushLayer.toDataURL('image/png');
                applyBrushToMask(brushDataURL);
                maskBrushCtx.clearRect(0, 0, maskBrushLayer.width, maskBrushLayer.height);
            }
            maskCursorPos = null;
            maskLastBrushPos = null;
            maskBrushCtx.clearRect(0, 0, maskBrushLayer.width, maskBrushLayer.height);
        });
        
        // Draw brush
        function drawMaskBrush(x, y) {
            const brushSize = parseInt(document.getElementById('brushSize').value);
            maskBrushCtx.beginPath();
            if (maskLastBrushPos) {
                maskBrushCtx.moveTo(maskLastBrushPos.x, maskLastBrushPos.y);
                maskBrushCtx.lineTo(x, y);
                maskBrushCtx.lineWidth = brushSize * 2;
                maskBrushCtx.lineCap = 'round';
                maskBrushCtx.strokeStyle = maskMode === 'brush-add' ? 'rgba(0, 255, 0, 1.0)' : 'rgba(255, 0, 0, 1.0)';
                maskBrushCtx.stroke();
            } else {
                maskBrushCtx.arc(x, y, brushSize, 0, 2 * Math.PI);
                maskBrushCtx.fillStyle = maskMode === 'brush-add' ? 'rgba(0, 255, 0, 1.0)' : 'rgba(255, 0, 0, 1.0)';
                maskBrushCtx.fill();
            }
            maskLastBrushPos = { x, y };
        }
        
        // Update brush preview
        function updateBrushPreview() {
            if (!maskIsDrawing && maskCursorPos) {
                const brushSize = parseInt(document.getElementById('brushSize').value);
                maskBrushCtx.clearRect(0, 0, maskBrushLayer.width, maskBrushLayer.height);
                if (maskMode === 'brush-add' || maskMode === 'brush-remove') {
                    maskBrushCtx.beginPath();
                    maskBrushCtx.arc(maskCursorPos.x, maskCursorPos.y, brushSize, 0, 2 * Math.PI);
                    maskBrushCtx.strokeStyle = maskMode === 'brush-add' ? 'lime' : 'red';
                    maskBrushCtx.lineWidth = 2;
                    maskBrushCtx.stroke();
                }
            }
        }
        
        // Apply brush to mask
        function applyBrushToMask(brushDataURL) {
            const brushImg = new Image();
            brushImg.onload = () => {
                const img = new Image();
                img.onload = () => {
                    if (!currentMask) {
                        currentMask = document.createElement('canvas');
                        currentMask.width = img.width;
                        currentMask.height = img.height;
                        const maskCtx2 = currentMask.getContext('2d');
                        maskCtx2.clearRect(0, 0, currentMask.width, currentMask.height);
                    } else {
                        if (currentMask.width !== img.width || currentMask.height !== img.height) {
                            const oldMask = currentMask;
                            currentMask = document.createElement('canvas');
                            currentMask.width = img.width;
                            currentMask.height = img.height;
                            const maskCtx2 = currentMask.getContext('2d');
                            maskCtx2.drawImage(oldMask, 0, 0, img.width, img.height);
                        }
                    }
                    const maskCtx2 = currentMask.getContext('2d');
                    applyBrushOperation(maskCtx2, brushImg, img.width, img.height);
                };
                img.src = currentImageData.image;
            };
            brushImg.src = brushDataURL;
        }
        
        // Apply brush operation
        function applyBrushOperation(ctx, brushImg, maskWidth, maskHeight) {
            if (maskMode === 'brush-add') {
                const selectedClass = document.getElementById('maskClassSelect').value;
                const color = classColors[selectedClass];
                const tempCanvas = document.createElement('canvas');
                tempCanvas.width = maskWidth;
                tempCanvas.height = maskHeight;
                const tempCtx = tempCanvas.getContext('2d');
                tempCtx.drawImage(brushImg, 0, 0, maskWidth, maskHeight);
                const imageData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
                for (let i = 0; i < imageData.data.length; i += 4) {
                    if (imageData.data[i + 3] > 0) {
                        imageData.data[i] = color[0];
                        imageData.data[i + 1] = color[1];
                        imageData.data[i + 2] = color[2];
                        imageData.data[i + 3] = 255;
                    }
                }
                tempCtx.putImageData(imageData, 0, 0);
                ctx.drawImage(tempCanvas, 0, 0);
            } else if (maskMode === 'brush-remove') {
                const imageData = ctx.getImageData(0, 0, maskWidth, maskHeight);
                const tempCanvas = document.createElement('canvas');
                tempCanvas.width = maskWidth;
                tempCanvas.height = maskHeight;
                const tempCtx = tempCanvas.getContext('2d');
                tempCtx.drawImage(brushImg, 0, 0, maskWidth, maskHeight);
                const brushImageData = tempCtx.getImageData(0, 0, maskWidth, maskHeight);
                for (let i = 0; i < imageData.data.length; i += 4) {
                    if (brushImageData.data[i + 3] > 0 && imageData.data[i + 3] > 0) {
                        imageData.data[i] = 0;
                        imageData.data[i + 1] = 0;
                        imageData.data[i + 2] = 0;
                    }
                }
                ctx.putImageData(imageData, 0, 0);
            }
            drawMaskCanvas();
        }
        
        // Show label selection dialog
        function showLabelDialog() {
            if (!pendingBBox) return;
            const dialog = document.getElementById('labelDialog');
            const optionsDiv = document.getElementById('labelOptions');
            optionsDiv.innerHTML = '';
            defectClasses.forEach(className => {
                const button = document.createElement('button');
                button.textContent = className;
                const color = classColors[className] || [255, 255, 0];
                const bgColor = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
                const hoverColor = `rgb(${Math.min(255, color[0] + 30)}, ${Math.min(255, color[1] + 30)}, ${Math.min(255, color[2] + 30)})`;
                button.style.cssText = `width: 100%; padding: 10px; text-align: left; background-color: ${bgColor}; color: black; border: 1px solid #333; border-radius: 4px; cursor: pointer; font-size: 14px; font-weight: bold; transition: background-color 0.2s;`;
                button.onmouseover = () => button.style.backgroundColor = hoverColor;
                button.onmouseout = () => button.style.backgroundColor = bgColor;
                button.onclick = () => handleLabelSelected(className);
                optionsDiv.appendChild(button);
            });
            dialog.style.display = 'flex';
        }
        
        // Handle label selection
        function handleLabelSelected(className) {
            if (!pendingBBox) return;
            const subTypeMap = {
                'Crack': 'Linear crack',
                'Material_loss': 'Spalling',
                'Stain': 'Leakage stain',
                'External Fixings': 'Surface contaminants'
            };
            const newBBox = {
                id: pendingBBox.id,
                bbox: pendingBBox.bbox,
                primary_class: className,
                sub_type: subTypeMap[className] || className.toLowerCase().replace(' ', '_')
            };
            currentBBoxes.push(newBBox);
            drawBBoxCanvas();
            showStatus('bboxStatus', `Added one BBox: ${className}`, 'success');
            cancelLabelSelection();
        }
        
        // Cancel label selection
        function cancelLabelSelection() {
            pendingBBox = null;
            const dialog = document.getElementById('labelDialog');
            dialog.style.display = 'none';
        }
        
        // Clear BBoxes
        function clearBBoxes() {
            if (confirm('Are you sure you want to clear all BBoxes?')) {
                currentBBoxes = [];
                pendingBBox = null;
                cancelLabelSelection();
                drawBBoxCanvas();
                showStatus('bboxStatus', 'Cleared all BBoxes', 'success');
            }
        }
        
        // Clear Mask
        function clearMask() {
            if (confirm('Are you sure you want to clear the Mask?')) {
                currentMask = null;
                originalMask = null;
                maskPoints = [];
                maskBrushCtx.clearRect(0, 0, maskBrushLayer.width, maskBrushLayer.height);
                drawMaskCanvas();
                showStatus('maskStatus', 'Cleared Mask', 'success');
            }
        }
        
        // Save changes
        async function saveChanges() {
            if (pendingBBox) {
                cancelLabelSelection();
                showStatus('bboxStatus', 'Cancelled pending box', 'error');
                return;
            }
            try {
                let maskDataURL = null;
                if (currentMask) {
                    try {
                        if (currentMask instanceof HTMLCanvasElement) {
                            maskDataURL = currentMask.toDataURL('image/png');
                        } else {
                            showStatus('maskStatus', 'Mask format error, cannot save', 'error');
                            return;
                        }
                    } catch (e) {
                        showStatus('maskStatus', 'Failed to convert mask: ' + e.message, 'error');
                        return;
                    }
                }
                const response = await fetch('/api/save', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        index: currentIndex,
                        dataset: currentDataset,
                        primary_class: currentPrimaryClass,
                        bboxes: currentBBoxes,
                        mask: maskDataURL
                    })
                });
                const data = await response.json();
                if (data.success) {
                    showStatus('bboxStatus', 'Saved successfully', 'success');
                    showStatus('maskStatus', maskDataURL ? 'Mask saved successfully' : 'Mask not modified', 'success');
                    setTimeout(() => {
                        loadImage(currentIndex);
                    }, 500);
                } else {
                    showStatus('bboxStatus', 'Save failed: ' + data.error, 'error');
                    showStatus('maskStatus', 'Save failed: ' + data.error, 'error');
                }
            } catch (error) {
                console.error('Save failed:', error);
                showStatus('bboxStatus', 'Save failed: ' + error.message, 'error');
                showStatus('maskStatus', 'Save failed: ' + error.message, 'error');
            }
        }
        
        // Show status
        function showStatus(elementId, message, type = '') {
            const element = document.getElementById(elementId);
            element.textContent = message;
            element.className = 'status ' + type;
            setTimeout(() => {
                element.textContent = '';
                element.className = 'status';
            }, 3000);
        }
        
        // Run detection
        async function runDetection() {
            if (!currentImageData || !currentImageData.image) {
                showStatus('bboxStatus', 'Please load image first', 'error');
                return;
            }
            const model = document.getElementById('detectionModelSelect').value;
            showStatus('bboxStatus', 'Running detection...', '');
            try {
                const response = await fetch('/api/detect', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        image_data: currentImageData.image,
                        model: model,
                        filename: currentImageData.filename || 'unknown.jpg'
                    })
                });
                const data = await response.json();
                if (data.success && data.bboxes) {
                    const maxId = currentBBoxes.length > 0 ? Math.max(...currentBBoxes.map(b => b.id || 0)) : -1;
                    let nextId = maxId + 1;
                    const existingCount = currentBBoxes.length;
                    data.bboxes.forEach((bbox) => {
                        currentBBoxes.push({
                            id: nextId++,
                            bbox: bbox.bbox,
                            primary_class: bbox.primary_class,
                            sub_type: bbox.sub_type
                        });
                    });
                    const newCount = currentBBoxes.length - existingCount;
                    drawBBoxCanvas();
                    showStatus('bboxStatus', `Detection completed, added ${newCount} boxes (total ${currentBBoxes.length})`, 'success');
                } else {
                    showStatus('bboxStatus', 'Detection failed: ' + (data.error || 'Unknown error'), 'error');
                }
            } catch (error) {
                console.error('Detection failed:', error);
                showStatus('bboxStatus', 'Detection failed: ' + error.message, 'error');
            }
        }
        
        // Add to candidates
        async function addToCandidates() {
            if (!currentImageData) {
                showStatus('bboxStatus', 'No image loaded', 'error');
                return;
            }
            let primaryClass = null;
            if (currentBBoxes.length > 0) {
                const classCount = {};
                currentBBoxes.forEach(bbox => {
                    const pc = bbox.primary_class;
                    if (pc) {
                        classCount[pc] = (classCount[pc] || 0) + 1;
                    }
                });
                const classNames = Object.keys(classCount);
                if (classNames.length > 0) {
                    primaryClass = classNames.reduce((a, b) => classCount[a] > classCount[b] ? a : b);
                    if (classNames.length > 1) {
                        const counts = classNames.map(c => `${c}: ${classCount[c]}`).join(', ');
                        console.log(`[Candidate] Image contains multiple classes: ${counts}, selected most common: ${primaryClass}`);
                    }
                }
            }
            if (!primaryClass) {
                showStatus('bboxStatus', 'Current image has no bbox annotations, cannot add to candidates', 'error');
                return;
            }
            try {
                const response = await fetch('/api/add_candidate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        index: currentIndex,
                        dataset: currentDataset,
                        primary_class: primaryClass,
                        filename: currentImageData.filename
                    })
                });
                const data = await response.json();
                if (data.success) {
                    showStatus('bboxStatus', `Added to candidates (${primaryClass}): ${data.message}`, 'success');
                } else {
                    showStatus('bboxStatus', 'Failed to add to candidates: ' + data.error, 'error');
                }
            } catch (error) {
                console.error('Failed to add to candidates:', error);
                showStatus('bboxStatus', 'Failed to add to candidates: ' + error.message, 'error');
            }
        }
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            const isInputElement = e.target.tagName === 'INPUT' || 
                                   e.target.tagName === 'TEXTAREA' || 
                                   e.target.tagName === 'SELECT';
            if (e.key === 'ArrowLeft') {
                e.preventDefault();
                if (currentIndex > 0) loadImage(currentIndex - 1);
            } else if (e.key === 'ArrowRight') {
                e.preventDefault();
                loadImage(currentIndex + 1);
            } else if ((e.key === 'c' || e.key === 'C') && !e.ctrlKey && !e.metaKey) {
                if (e.target.tagName === 'SELECT') {
                    e.preventDefault();
                    e.stopPropagation();
                    e.target.blur();
                } else if (!isInputElement) {
                    e.preventDefault();
                    addToCandidates();
                }
            } else if ((e.key === 's' || e.key === 'S') && !e.ctrlKey && !e.metaKey) {
                if (e.target.tagName === 'SELECT') {
                    e.preventDefault();
                    e.stopPropagation();
                    e.target.blur();
                } else if (!isInputElement) {
                    e.preventDefault();
                    saveChanges();
                }
            }
        });
        
        // Initialize
        updateImageList();
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/images")
def api_images():
    """Get total number of filtered images"""
    dataset = request.args.get("dataset") or None
    primary_class = request.args.get("primary_class") or None
    
    filtered = _get_filtered_images(dataset, primary_class)
    return jsonify({"total": len(filtered)})


@app.route("/api/image/<int:index>")
def api_image(index: int):
    """Get image data by index"""
    dataset = request.args.get("dataset") or None
    primary_class = request.args.get("primary_class") or None
    
    filtered = _get_filtered_images(dataset, primary_class)
    
    if index < 0 or index >= len(filtered):
        return jsonify({"error": "index out of range"})
    
    img_info = filtered[index]
    stem = img_info['stem']
    img_path = Path(img_info['path'])
    
    # Read image and convert to base64
    img = Image.open(img_path)
    img_rgb = img.convert("RGB")
    buffer = io.BytesIO()
    img_rgb.save(buffer, format="JPEG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    image_data_url = f"data:image/jpeg;base64,{img_base64}"
    
    # Load bboxes
    bboxes = _load_bboxes_for_image(stem)
    bbox_list = []
    for bbox_data in bboxes:
        bbox_list.append({
            "bbox": bbox_data["bbox"],
            "primary_class": bbox_data.get("primary_class"),
            "sub_type": bbox_data.get("sub_type", ""),
        })
    
    # Load mask
    mask_data_url = None
    mask_rgb = _load_mask_for_image(stem)
    if mask_rgb is not None:
        mask_pil = Image.fromarray(mask_rgb)
        mask_buffer = io.BytesIO()
        mask_pil.save(mask_buffer, format="PNG")
        mask_base64 = base64.b64encode(mask_buffer.getvalue()).decode("utf-8")
        mask_data_url = f"data:image/png;base64,{mask_base64}"
    
    return jsonify({
        "index": index,
        "total": len(filtered),
        "filename": img_info['name'],
        "image": image_data_url,
        "bboxes": bbox_list,
        "mask": mask_data_url,
    })


@app.route("/api/save", methods=["POST"])
def api_save():
    """Save bbox and mask modifications"""
    try:
        data = request.json
        index = data.get("index")
        dataset = data.get("dataset")
        primary_class = data.get("primary_class")
        bboxes = data.get("bboxes", [])
        mask_base64 = data.get("mask")
        
        # Get current image info
        filtered = _get_filtered_images(dataset, primary_class)
        if index < 0 or index >= len(filtered):
            return jsonify({"success": False, "error": "index out of range"})
        
        img_info = filtered[index]
        stem = img_info['stem']
        
        # Save bbox modifications
        label_path = LABELS_DIR / f"{stem}.json"
        label_data = {
            "image_path": img_info['name'],
            "bboxes": []
        }
        
        for i, bbox_data in enumerate(bboxes):
            bbox_xywh = bbox_data.get("bbox", [])
            if len(bbox_xywh) != 4:
                continue
            
            taxonomy = {
                "primary_class": bbox_data.get("primary_class"),
                "sub_type": bbox_data.get("sub_type", "unknown")
            }
            
            if not taxonomy.get("sub_type") or taxonomy["sub_type"] == "unknown":
                sub_type_map = {
                    'Crack': 'Linear crack',
                    'Material_loss': 'Spalling',
                    'Stain': 'Leakage stain',
                    'External Fixings': 'Surface contaminants'
                }
                taxonomy["sub_type"] = sub_type_map.get(taxonomy.get("primary_class"), "unknown")
            
            instance_id = f"{stem}_{i}"
            
            label_data["bboxes"].append({
                "instance_id": instance_id,
                "taxonomy": taxonomy,
                "bbox": bbox_xywh  # [x, y, w, h] format
            })
        
        # Save label file
        label_path.parent.mkdir(parents=True, exist_ok=True)
        with open(label_path, 'w', encoding='utf-8') as f:
            json.dump(label_data, f, indent=2, ensure_ascii=False)
        
        # Save mask modifications
        if mask_base64:
            try:
                mask_path = MASKS_DIR / f"{stem}_mask.png"
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                
                if "," in mask_base64:
                    mask_data = base64.b64decode(mask_base64.split(",", 1)[1])
                else:
                    mask_data = base64.b64decode(mask_base64)
                
                mask_img = Image.open(io.BytesIO(mask_data))
                mask_img.save(mask_path, "PNG")
                print(f"[Save] Mask saved: {mask_path} (size: {mask_img.size})")
            except Exception as e:
                print(f"[Save] Failed to save mask: {e}")
                import traceback
                traceback.print_exc()
        
        # Reload image list cache
        global _image_list_loaded
        _image_list_loaded = False
        _load_image_list()
        
        return jsonify({"success": True})
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/refine_mask", methods=["POST"])
def api_refine_mask():
    """Refine mask using SAM (point operations)"""
    if not SAM_AVAILABLE or sam_service is None:
        return jsonify({"success": False, "error": "SAM service not available"})
    
    try:
        data = request.json
        image_base64 = data.get("image_data")
        mask_base64 = data.get("mask_data")
        points = data.get("points", [])
        labels = data.get("labels", [])
        bboxes = data.get("bboxes", [])
        
        if not image_base64:
            return jsonify({"success": False, "error": "image_data is required"})
        
        if len(points) == 0:
            return jsonify({"success": False, "error": "points are required"})
        
        # Decode image
        image_data = base64.b64decode(image_base64.split(",")[1] if "," in image_base64 else image_base64)
        image_np = np.frombuffer(image_data, np.uint8)
        image_np = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        if image_np is None:
            return jsonify({"success": False, "error": "Failed to decode image"})
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        
        # Decode current mask (if exists)
        current_mask = None
        if mask_base64:
            mask_data = base64.b64decode(mask_base64.split(",")[1] if "," in mask_base64 else mask_base64)
            mask_np = np.frombuffer(mask_data, np.uint8)
            mask_img = cv2.imdecode(mask_np, cv2.IMREAD_UNCHANGED)
            if mask_img is not None:
                if len(mask_img.shape) == 3:
                    if mask_img.shape[2] == 4:
                        current_mask = mask_img[:, :, 3]
                    else:
                        current_mask = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
                else:
                    current_mask = mask_img
                
                if current_mask.shape[:2] != image_np.shape[:2]:
                    current_mask = cv2.resize(current_mask, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            current_mask = np.zeros(image_np.shape[:2], dtype=np.uint8)
        
        # Ensure SAM is initialized
        if not sam_service.initialized:
            sam_service.initialize()
        
        # Call SAM refine
        refined_mask = sam_service.refine_mixed(
            image_np=image_np,
            current_mask=current_mask,
            points=points,
            labels=labels,
            bboxes=bboxes if bboxes else None,
            brush_mask_b64=None,
            operation='point'
        )
        
        # Convert mask to base64
        mask_pil = Image.fromarray(refined_mask)
        mask_buffer = io.BytesIO()
        mask_pil.save(mask_buffer, format="PNG")
        mask_base64_result = base64.b64encode(mask_buffer.getvalue()).decode("utf-8")
        mask_data_url = f"data:image/png;base64,{mask_base64_result}"
        
        return jsonify({
            "success": True,
            "mask": mask_data_url
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/detect", methods=["POST"])
def api_detect():
    """Run detection model and return detection results"""
    if not DETECTION_AVAILABLE or detection_service is None:
        return jsonify({"success": False, "error": "Detection service not available"})
    
    try:
        data = request.json
        image_base64 = data.get("image_data")
        model = data.get("model", "ensemble")
        filename = data.get("filename", "unknown.jpg")
        
        if not image_base64:
            return jsonify({"success": False, "error": "image_data is required"})
        
        # Decode image
        image_data = base64.b64decode(image_base64.split(",")[1] if "," in image_base64 else image_base64)
        image_np = np.frombuffer(image_data, np.uint8)
        image_np = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        if image_np is None:
            return jsonify({"success": False, "error": "Failed to decode image"})
        
        # Set detection model
        detection_service.model = model
        
        # Run detection (async)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            detection_service.detect(image_np, text_prompt="Detect defects in this image.", filename=filename)
        )
        loop.close()
        
        if "error" in result:
            return jsonify({"success": False, "error": result["error"]})
        
        # Convert detection results format
        annotations = result.get("annotations_in_crop", [])
        bboxes = []
        h, w = image_np.shape[:2]
        
        for ann in annotations:
            # Detection results use normalized coordinates, convert to pixel coordinates
            x_center_norm = ann.get("x_center_norm", 0)
            y_center_norm = ann.get("y_center_norm", 0)
            width_norm = ann.get("width_norm", 0)
            height_norm = ann.get("height_norm", 0)
            
            # Convert to pixel coordinates [x, y, w, h]
            x_center = x_center_norm * w
            y_center = y_center_norm * h
            bbox_w = width_norm * w
            bbox_h = height_norm * h
            x = x_center - bbox_w / 2
            y = y_center - bbox_h / 2
            
            # Get class name (sub_type)
            class_name = ann.get("class_name", "unknown")
            
            # Map to primary_class
            primary_class = SUBTYPE_TO_PRIMARY_CLASS.get(class_name, "Crack")
            
            bboxes.append({
                "bbox": [x, y, bbox_w, bbox_h],  # [x, y, w, h] format
                "primary_class": primary_class,
                "sub_type": class_name  # Keep original class name
            })
        
        return jsonify({
            "success": True,
            "bboxes": bboxes
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/add_candidate", methods=["POST"])
def api_add_candidate():
    """Add current image to candidates list"""
    try:
        data = request.json
        index = data.get("index")
        dataset = data.get("dataset")
        primary_class = data.get("primary_class")
        filename = data.get("filename")
        
        if not primary_class:
            return jsonify({"success": False, "error": "primary_class is required"})
        
        # Read or create candidates CSV
        if CANDIDATES_CSV.exists():
            df_candidates = pd.read_csv(CANDIDATES_CSV)
        else:
            df_candidates = pd.DataFrame(columns=["primary_class", "dataset", "index", "filename", "added_at"])
        
        # Check if already exists (avoid duplicates)
        if len(df_candidates[(df_candidates["primary_class"] == primary_class) & 
                            (df_candidates["filename"] == filename)]) > 0:
            return jsonify({
                "success": False,
                "error": "This image already exists in the candidates list"
            })
        
        # Add new record
        from datetime import datetime
        new_row = {
            "primary_class": primary_class,
            "dataset": dataset or "",
            "index": index,
            "filename": filename,
            "added_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        df_candidates = pd.concat([df_candidates, pd.DataFrame([new_row])], ignore_index=True)
        
        # Save CSV
        CANDIDATES_CSV.parent.mkdir(parents=True, exist_ok=True)
        df_candidates.to_csv(CANDIDATES_CSV, index=False, encoding='utf-8')
        
        new_count = len(df_candidates[df_candidates["primary_class"] == primary_class])
        return jsonify({
            "success": True,
            "message": f"Added to candidates ({new_count} total)"
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})


if __name__ == "__main__":
    # Load image list cache
    _load_image_list()
    
    print("=" * 80)
    print("Image Annotation to Candidates Tool")
    print("=" * 80)
    print(f"Images: {len(_image_list_cache)} files")
    print(f"Access URL: http://localhost:5000")
    print("=" * 80)
    app.run(host="0.0.0.0", port=5000, debug=True)