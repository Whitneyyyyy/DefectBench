#!/usr/bin/env python3
"""
Generate bbox-only visualization images for VLM pipelines.

Input:
- Images: defect_bench/data_sample/images
- Labels: defect_bench/data_sample/labels

Output:
- defect_bench/results/visualization
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
    "Crack": (255, 0, 0),
    "Material_loss": (255, 140, 0),
    "Stain": (30, 144, 255),
    "External Fixings": (0, 200, 0),
}
DEFAULT_COLOR = (255, 255, 0)

DATA_ROOT = Path("defect_bench/data_sample")
IMAGES_DIR = DATA_ROOT / "images"
LABELS_DIR = DATA_ROOT / "labels"
OUTPUT_DIR = Path("defect_bench/results/visualization/images")


def get_color_for_class(primary_class: Optional[str]) -> Tuple[int, int, int]:
    return CLASS_COLORS.get(primary_class, DEFAULT_COLOR) if primary_class else DEFAULT_COLOR


def draw_bboxes(image: np.ndarray, bboxes: List[Dict]) -> np.ndarray:
    out = image.copy()
    img_h, img_w = out.shape[:2]
    diag = float(np.hypot(img_w, img_h))
    font_scale = max(0.5, min(2.0, diag / 1000.0))
    line_thickness = max(2, int(min(img_w, img_h) / 400))

    for idx, bbox_data in enumerate(bboxes):
        bbox = bbox_data.get("bbox", [])
        if len(bbox) != 4:
            continue
        x, y, w, h = bbox
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)
        primary = bbox_data.get("taxonomy", {}).get("primary_class")
        rgb = get_color_for_class(primary)
        bgr = (rgb[2], rgb[1], rgb[0])

        cv2.rectangle(out, (x1, y1), (x2, y2), bgr, line_thickness)
        label = f"{idx + 1}#{primary or 'Unknown'}"
        thickness = max(1, int(font_scale * 1.5))
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        lx = max(0, min(x1, img_w - tw - 2))
        ly = max(th + 4, y1)
        cv2.rectangle(out, (lx - 4, ly - th - bl - 4), (lx + tw + 4, ly + bl + 4), bgr, -1)
        cv2.putText(out, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    return out


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    images = sorted([p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    ok = 0
    err = 0
    for img_path in images:
        stem = img_path.stem
        json_path = LABELS_DIR / f"{stem}.json"
        if not json_path.exists():
            err += 1
            continue
        image = cv2.imread(str(img_path))
        if image is None:
            err += 1
            continue
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            bboxes = data.get("bboxes", [])
        except Exception:
            err += 1
            continue
        vis = draw_bboxes(image, bboxes)
        out_path = OUTPUT_DIR / f"{stem}_visualized.jpg"
        cv2.imwrite(str(out_path), vis)
        ok += 1
    print(f"Done. visualized={ok}, skipped={err}, output={OUTPUT_DIR}")


if __name__ == "__main__":
    main()
