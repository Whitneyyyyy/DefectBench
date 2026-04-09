import os
import base64
from pathlib import Path
from typing import List

import cv2
import numpy as np
from ultralytics.models.sam import SAM3SemanticPredictor
from ultralytics import SAM


class SAMService:
    """
    Lightweight SAM service used by data_sample.

    - Loads SAM3 model from a local .pt file (by default `model_weights/sam3.pt` under this directory).
    - Supports:
        * `predict_bboxes`: generate masks from bbox prompts.
        * `_predict_with_points_only`: generate masks from point prompts.
        * `refine_mixed`: mixed refinement with points and brush (used by /api/refine_mask).
    - All Doubao-related quality analysis and external dependencies are intentionally removed.
    """

    def __init__(self) -> None:
        self.predictor = None
        self.sam_refiner = None
        self.initialized = False

        # Resolve model path from defect_bench/model_weights.
        current_dir = os.path.dirname(os.path.abspath(__file__))
        bench_root = next(
            (
                str(parent)
                for parent in Path(current_dir).resolve().parents
                if parent.name == "defect_bench"
            ),
            None,
        )
        if not bench_root:
            raise RuntimeError("Cannot resolve defect_bench root for SAM model path.")
        self.default_model_path = os.path.join(bench_root, "model_weights", "sam3.pt")

    def initialize(self, model_path: str | None = None) -> None:
        """Initialize SAM models (SAM3SemanticPredictor + SAM refiner)."""
        if self.initialized:
            return

        if model_path is None:
            model_path = self.default_model_path

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"SAM3 weight not found at required path: {model_path}")

        print(f"Initializing SAM3 models from: {model_path}")
        overrides = dict(
            conf=0.25,
            task="segment",
            mode="predict",
            model=model_path,
            half=False,
            save=False,
            verbose=False,
        )
        try:
            self.predictor = SAM3SemanticPredictor(overrides=overrides)
            self.sam_refiner = SAM(model_path)
            self.initialized = True
            print("SAM3 initialized successfully.")
        except Exception as e:  # pragma: no cover - runtime/weight errors
            print(f"Failed to initialize SAM3: {e}")
            self.predictor = None
            self.sam_refiner = None
            self.initialized = False

    # -------------------------------------------------------------------------
    # Core prediction utilities
    # -------------------------------------------------------------------------
    def predict_bboxes(self, image_np: np.ndarray, bboxes: List[List[float]]) -> np.ndarray:
        """
        Run SAM3 on image with bbox prompts.

        Args:
            image_np: RGB numpy array (H, W, 3)
            bboxes: list of [x1, y1, x2, y2] in pixel coordinates

        Returns:
            Binary mask (uint8, 0/255) with the union of all masks, clamped to bboxes.
        """
        if not self.initialized or self.predictor is None:
            raise RuntimeError("SAM model not initialized")

        h, w = image_np.shape[:2]
        combined_mask = np.zeros((h, w), dtype=np.uint8)

        if not bboxes:
            return combined_mask

        try:
            self.predictor.set_image(image_np)
            results = self.predictor(bboxes=bboxes)

            if results:
                for result in results:
                    if result.masks is None:
                        continue
                    if hasattr(result.masks, "data"):
                        masks = result.masks.data.cpu().numpy()
                    elif hasattr(result.masks, "cpu"):
                        masks = result.masks.cpu().numpy()
                    else:
                        masks = np.array(result.masks)

                    for m in masks:
                        if m.shape[:2] != (h, w):
                            m = cv2.resize(
                                m.astype(np.float32),
                                (w, h),
                                interpolation=cv2.INTER_NEAREST,
                            )
                        combined_mask = np.maximum(
                            combined_mask, (m > 0.5).astype(np.uint8) * 255
                        )

            # Clamp mask to bbox regions
            bbox_mask = np.zeros((h, w), dtype=np.uint8)
            for bbox in bboxes:
                x1, y1, x2, y2 = map(int, bbox)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                if x2 > x1 and y2 > y1:
                    bbox_mask[y1:y2, x1:x2] = 255
            combined_mask = cv2.bitwise_and(combined_mask, bbox_mask)

        except Exception as e:  # pragma: no cover - runtime errors
            print(f"SAM bbox prediction error: {e}")
            raise

        return combined_mask

    def _predict_with_points_only(
        self, image_np: np.ndarray, bboxes: List[List[float]]
    ) -> np.ndarray:
        """
        Predict mask using only point prompts (center points from bboxes).

        This is mainly a fallback or alternative interaction mode.
        """
        if not self.initialized or self.sam_refiner is None:
            raise RuntimeError("SAM model not initialized")

        h, w = image_np.shape[:2]
        combined_mask = np.zeros((h, w), dtype=np.uint8)

        points: List[List[float]] = []
        labels: List[int] = []

        for bbox in bboxes:
            x1, y1, x2, y2 = map(float, bbox)
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w - 1, x2))
            y2 = max(0, min(h - 1, y2))

            if x2 <= x1 or y2 <= y1:
                continue

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            points.append([cx, cy])
            labels.append(1)

        if not points:
            return combined_mask

        try:
            results = self.sam_refiner.predict(
                source=image_np,
                points=points,
                labels=labels,
                imgsz=max(h, w),
                verbose=False,
            )

            if results:
                for result in results:
                    if result.masks is None:
                        continue
                    if hasattr(result.masks, "data"):
                        masks = result.masks.data.cpu().numpy()
                    elif hasattr(result.masks, "cpu"):
                        masks = result.masks.cpu().numpy()
                    else:
                        masks = np.array(result.masks)

                    for mask in masks:
                        if mask.shape[:2] != (h, w):
                            mask = cv2.resize(
                                mask.astype(np.float32),
                                (w, h),
                                interpolation=cv2.INTER_NEAREST,
                            )
                        combined_mask = np.maximum(
                            combined_mask, (mask > 0.5).astype(np.uint8) * 255
                        )

            if bboxes:
                bbox_mask = np.zeros((h, w), dtype=np.uint8)
                for bbox in bboxes:
                    x1, y1, x2, y2 = map(int, bbox)
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w, x2)
                    y2 = min(h, y2)
                    if x2 > x1 and y2 > y1:
                        bbox_mask[y1:y2, x1:x2] = 255
                combined_mask = cv2.bitwise_and(combined_mask, bbox_mask)

        except Exception as e:  # pragma: no cover
            print(f"SAM point-only prediction error: {e}")
            raise

        return combined_mask

    # -------------------------------------------------------------------------
    # Mixed refinement (points + brush) used by /api/refine_mask
    # -------------------------------------------------------------------------
    def refine_mixed(
        self,
        image_np: np.ndarray,
        current_mask: np.ndarray,
        points: List[List[int]],
        labels: List[int],
        bboxes: List[List[float]] | None = None,
        brush_mask_b64: str | None = None,
        operation: str = "point",
    ) -> np.ndarray:
        """
        Unified refinement handling Points and Brush.

        Args:
            image_np: RGB or BGR numpy array
            current_mask: uint8 mask (0/255)
            points: list of [x, y] points
            labels: list of 1 (positive) or 0 (negative)
            bboxes: optional list of [x1, y1, x2, y2] to constrain SAM output
            brush_mask_b64: optional brush mask (for add/remove)
            operation: 'point', 'brush-add', or 'brush-remove'
        """
        if not self.initialized or self.sam_refiner is None:
            raise RuntimeError("SAM model not initialized")

        h, w = image_np.shape[:2]
        final_mask = current_mask.copy()

        # 1. Brush operation (pixel-wise override)
        if operation.startswith("brush") and brush_mask_b64:
            try:
                b_bytes = base64.b64decode(
                    brush_mask_b64.split(",")[1]
                    if "," in brush_mask_b64
                    else brush_mask_b64
                )
                b_arr = np.frombuffer(b_bytes, np.uint8)
                brush_mask_img = cv2.imdecode(b_arr, cv2.IMREAD_GRAYSCALE)

                if brush_mask_img.shape[:2] != (h, w):
                    brush_mask_img = cv2.resize(
                        brush_mask_img, (w, h), interpolation=cv2.INTER_NEAREST
                    )

                brush_mask_bool = brush_mask_img > 0

                if operation == "brush-add":
                    # Optional: respect bbox constraints
                    if bboxes:
                        bbox_constraint = np.zeros((h, w), dtype=np.uint8)
                        for bbox in bboxes:
                            x1, y1, x2, y2 = map(int, bbox)
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(w, x2), min(h, y2)
                            if x2 > x1 and y2 > y1:
                                bbox_constraint[y1:y2, x1:x2] = 255
                        brush_mask_bool = brush_mask_bool & (bbox_constraint > 0)

                    final_mask[brush_mask_bool] = 255

                elif operation == "brush-remove":
                    final_mask[brush_mask_bool] = 0

                return final_mask

            except Exception as e:  # pragma: no cover
                print(f"Brush refinement error: {e}")
                # fall through to point refinement if needed

        # 2. SAM point refinement
        if points:
            results = self.sam_refiner.predict(
                source=image_np,
                points=points,
                labels=labels,
                imgsz=max(h, w),
                verbose=False,
            )

            new_mask = np.zeros((h, w), dtype=np.uint8)

            if results:
                for res in results:
                    if res.masks is None:
                        continue
                    if hasattr(res.masks, "data"):
                        m_arr = res.masks.data.cpu().numpy()
                    elif hasattr(res.masks, "cpu"):
                        m_arr = res.masks.cpu().numpy()
                    else:
                        m_arr = np.array(res.masks)

                    if m_arr is not None:
                        for mask in m_arr:
                            if mask.shape[:2] != (h, w):
                                mask = cv2.resize(
                                    mask.astype(np.float32),
                                    (w, h),
                                    interpolation=cv2.INTER_NEAREST,
                                )
                            new_mask = np.maximum(
                                new_mask, (mask > 0.5).astype(np.uint8) * 255
                            )

            # Optionally clamp SAM output to bboxes
            if bboxes:
                bbox_mask = np.zeros((h, w), dtype=np.uint8)
                for bbox in bboxes:
                    x1, y1, x2, y2 = map(int, bbox)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    if x2 > x1 and y2 > y1:
                        bbox_mask[y1:y2, x1:x2] = 255
                new_mask = cv2.bitwise_and(new_mask, bbox_mask)

            has_negative = any(l == 0 for l in labels)
            has_positive = any(l == 1 for l in labels)

            if has_negative and not has_positive:
                # Only negative points: subtract region
                final_mask = cv2.bitwise_and(final_mask, cv2.bitwise_not(new_mask))
            else:
                # Default: union with existing mask
                final_mask = cv2.bitwise_or(final_mask, new_mask)

        return final_mask


sam_service = SAMService()


