import os
import json
import base64
from pathlib import Path
from typing import Optional, Dict, Any, List

import cv2
import numpy as np
from ultralytics import YOLO


class DetectionAgent:
    """
    Lightweight detection agent for data_sample.

    - Uses only local detection models (YOLO, optional Faster R-CNN and RT-DETR).
    - No Doubao quality analysis, no external HTTP APIs.
    - API surface:
        * detect(image_input, text_prompt, filename) -> dict with `annotations_in_crop`.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "ensemble",
        api_url: str = "",
        local_model_path: Optional[str] = None,
    ) -> None:
        self.api_key = api_key  # kept for compatibility, not used
        self.model = model
        self.api_url = api_url

        # Model containers
        self.yolo_models: Dict[str, Any] = {}
        self.yolo_model_paths: Dict[str, str] = {}
        self.local_yolo = None
        self.frcnn_predictor = None
        self.rtdetr_model = None

        # Resolve model weight directory from defect_bench/model_weights.
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
            raise RuntimeError("Cannot resolve defect_bench root for model weights.")
        weights_dir = os.path.join(bench_root, "model_weights")

        yolo12m_path = local_model_path or os.path.join(weights_dir, "yolo12m_building.pt")
        yolo11m_path = os.path.join(weights_dir, "yolo11m_building.pt")
        self.yolo_model_paths["yolo12m"] = yolo12m_path
        self.yolo_model_paths["yolo11m"] = yolo11m_path

        self.frcnn_model_path = os.path.join(weights_dir, "Faster_R-CNN.pth")
        # RT-DETR model path (can be overridden to a local path if needed)
        self.rtdetr_model_path = os.path.join(weights_dir, "rtdetr-l-bst.pt")

        self.frcnn_class_names: List[str] = [
            "Concrete_Crack",
            "Concrete_Delamination",
            "Concrete_Spalling",
            "Rust_Stain",
            "Vegeterian",
            "Degraded_Plaster",
            "Craquelure",
            "Tile_Crack",
            "Tile_spalling",
            "Water_Stain",
            "Bulging",
            "Contaminants",
        ]

        # Load available YOLO models
        for name, path in self.yolo_model_paths.items():
            self._load_yolo_model(name, path)
        self.local_yolo = self.yolo_models.get("yolo12m")

        # Load RT-DETR model (optional)
        self._load_rtdetr_model()

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    def _encode_image(self, image_path_or_array) -> str:
        if isinstance(image_path_or_array, str):
            if image_path_or_array.startswith("data:image"):
                return image_path_or_array.split(",", 1)[1]
            if os.path.exists(image_path_or_array):
                with open(image_path_or_array, "rb") as image_file:
                    return base64.b64encode(image_file.read()).decode("utf-8")
            return image_path_or_array
        elif isinstance(image_path_or_array, np.ndarray):
            success, encoded_image = cv2.imencode(".jpg", image_path_or_array)
            if success:
                return base64.b64encode(encoded_image).decode("utf-8")
        return ""

    def _load_yolo_model(self, name: str, weight_path: str) -> None:
        """Load a YOLO model if the weight file exists."""
        if not os.path.exists(weight_path):
            print(f"Local YOLO weight for {name} not found at {weight_path}.")
            return
        try:
            self.yolo_models[name] = YOLO(weight_path)
            print(f"Local YOLO model '{name}' loaded from {weight_path}.")
        except Exception as e:  # pragma: no cover
            print(f"Failed to load YOLO model '{name}' from {weight_path}: {e}")

    def _load_rtdetr_model(self) -> None:
        """Load RT-DETR model if the weight file exists."""
        if not os.path.exists(self.rtdetr_model_path):
            print(f"RT-DETR weight not found at {self.rtdetr_model_path}.")
            self.rtdetr_model = None
            return
        try:
            from ultralytics import RTDETR

            self.rtdetr_model = RTDETR(self.rtdetr_model_path)
            print(f"RT-DETR model loaded from {self.rtdetr_model_path}.")
        except Exception as e:  # pragma: no cover
            print(f"Failed to load RT-DETR model from {self.rtdetr_model_path}: {e}")
            self.rtdetr_model = None

    def _bbox_from_norm(self, ann: Dict[str, Any], width: int, height: int) -> List[float]:
        """Convert normalized bbox to xyxy pixels."""
        cx = ann.get("x_center_norm", 0) * width
        cy = ann.get("y_center_norm", 0) * height
        bw = ann.get("width_norm", 0) * width
        bh = ann.get("height_norm", 0) * height
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2
        return [x1, y1, x2, y2]

    def _bbox_to_norm(self, box: List[float], width: int, height: int) -> Dict[str, float]:
        """Convert xyxy pixels to normalized center format."""
        x1, y1, x2, y2 = box
        bw = x2 - x1
        bh = y2 - y1
        cx = x1 + bw / 2
        cy = y1 + bh / 2
        return {
            "x_center_norm": cx / width if width > 0 else 0,
            "y_center_norm": cy / height if height > 0 else 0,
            "width_norm": bw / width if width > 0 else 0,
            "height_norm": bh / height if height > 0 else 0,
        }

    def _compute_overlap_score(self, box1: List[float], box2: List[float]) -> float:
        """
        Compute max(IoU, IoA_small) for two xyxy boxes.
        IoA_small = intersection area / min(box areas), capturing containment.
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter_w = max(0.0, x2 - x1)
        inter_h = max(0.0, y2 - y1)
        inter_area = inter_w * inter_h
        if inter_area <= 0:
            return 0.0
        area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
        area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
        if area1 <= 0 or area2 <= 0:
            return 0.0
        iou = inter_area / (area1 + area2 - inter_area) if (area1 + area2 - inter_area) > 0 else 0.0
        ioa_small = inter_area / min(area1, area2)
        return max(iou, ioa_small)

    def _merge_annotations(
        self, annotations: List[Dict[str, Any]], width: int, height: int, iou_thresh: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Merge overlapping boxes of the same class by averaging coordinates/confidence.
        """
        merged: Dict[str, List[Dict[str, Any]]] = {}
        for ann in annotations:
            cls = ann.get("class_name", "unknown")
            merged.setdefault(cls, []).append(ann)

        final_annotations: List[Dict[str, Any]] = []
        for cls, cls_anns in merged.items():
            cls_anns_sorted = sorted(cls_anns, key=lambda a: a.get("confidence", 0), reverse=True)
            kept: List[Dict[str, Any]] = []
            for ann in cls_anns_sorted:
                box = self._bbox_from_norm(ann, width, height)
                merged_flag = False
                for kept_ann in kept:
                    kept_box = kept_ann["_box_xyxy"]
                    score = self._compute_overlap_score(box, kept_box)
                    if score >= iou_thresh:
                        new_box = [
                            (kept_box[0] + box[0]) / 2,
                            (kept_box[1] + box[1]) / 2,
                            (kept_box[2] + box[2]) / 2,
                            (kept_box[3] + box[3]) / 2,
                        ]
                        kept_ann["_box_xyxy"] = new_box
                        kept_ann["confidence"] = float(
                            (kept_ann.get("confidence", 0) + ann.get("confidence", 0)) / 2
                        )
                        merged_flag = True
                        break
                if not merged_flag:
                    new_ann = ann.copy()
                    new_ann["_box_xyxy"] = box
                    kept.append(new_ann)

            for kept_ann in kept:
                box_xyxy = kept_ann.pop("_box_xyxy")
                norm_fields = self._bbox_to_norm(box_xyxy, width, height)
                kept_ann.update(norm_fields)
                final_annotations.append(kept_ann)

        return final_annotations

    # ------------------------------------------------------------------
    # Model-specific inference
    # ------------------------------------------------------------------
    def _run_local_inference(
        self,
        yolo_model: Any,
        img_input: Any,
        original_h: int,
        original_w: int,
        filename: str = "unknown.jpg",
    ) -> Dict[str, Any]:
        """Run inference using a local YOLO model and return formatted JSON."""
        if not yolo_model:
            return {"error": "Local YOLO model not initialized"}

        inference_input = img_input
        if isinstance(img_input, str) and img_input.startswith("data:image"):
            try:
                encoded_data = img_input.split(",", 1)[1]
                nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
                inference_input = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except Exception as e:  # pragma: no cover
                print(f"Error decoding data URI for local inference: {e}")
                return {"error": "Failed to decode image"}

        results = yolo_model(inference_input)
        annotations: List[Dict[str, Any]] = []

        for result in results:
            boxes = result.boxes
            names = result.names
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                class_name = names[cls_id]

                w_box = x2 - x1
                h_box = y2 - y1
                x_center = x1 + w_box / 2
                y_center = y1 + h_box / 2

                x_center_norm = x_center / original_w if original_w > 0 else 0
                y_center_norm = y_center / original_h if original_h > 0 else 0
                width_norm = w_box / original_w if original_w > 0 else 0
                height_norm = h_box / original_h if original_h > 0 else 0

                annotations.append(
                    {
                        "class_id": cls_id,
                        "class_name": class_name,
                        "confidence": confidence,
                        "x_center_norm": x_center_norm,
                        "y_center_norm": y_center_norm,
                        "width_norm": width_norm,
                        "height_norm": height_norm,
                        "description": f"Detected {class_name} with {confidence:.2f} confidence",
                    }
                )

        output_json = {
            "crop_filename": filename,
            "crop_coordinates_pixels": {"x1": 0, "y1": 0, "x2": original_w, "y2": original_h},
            "crop_width_pixels": original_w,
            "crop_height_pixels": original_h,
            "overlap_ratio": 1.0,
            "area_ratio": 1.0,
            "annotations_in_crop": annotations,
        }
        return output_json

    def _ensure_faster_rcnn_model(self) -> bool:
        """Lazy-load Faster R-CNN predictor if requested."""
        if self.frcnn_predictor is not None:
            return True

        if not os.path.exists(self.frcnn_model_path):
            print(f"Faster R-CNN weight not found at {self.frcnn_model_path}.")
            return False

        try:
            from detectron2.config import get_cfg
            from detectron2 import model_zoo
            from detectron2.engine import DefaultPredictor
            import torch  # noqa: F401
        except Exception as e:  # pragma: no cover
            print(f"Failed to import detectron2 components: {e}")
            return False

        try:
            cfg = get_cfg()
            cfg.merge_from_file(
                model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
            )
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(self.frcnn_class_names)
            cfg.MODEL.WEIGHTS = self.frcnn_model_path
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
            cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # type: ignore[name-defined]
            self.frcnn_predictor = DefaultPredictor(cfg)
            print(f"Faster R-CNN model loaded from {self.frcnn_model_path} on {cfg.MODEL.DEVICE}.")
            return True
        except Exception as e:  # pragma: no cover
            print(f"Failed to initialize Faster R-CNN model: {e}")
            self.frcnn_predictor = None
            return False

    def _run_faster_rcnn_inference(
        self,
        img_input: Any,
        original_h: int,
        original_w: int,
        filename: str = "unknown.jpg",
    ) -> Dict[str, Any]:
        """Run inference using Faster R-CNN and return formatted JSON."""
        if not self.frcnn_predictor:
            return {"error": "Faster R-CNN model not initialized"}

        inference_input = img_input
        if isinstance(img_input, str) and img_input.startswith("data:image"):
            try:
                encoded_data = img_input.split(",", 1)[1]
                nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
                inference_input = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except Exception as e:  # pragma: no cover
                print(f"Error decoding data URI for Faster R-CNN inference: {e}")
                return {"error": "Failed to decode image"}

        try:
            outputs = self.frcnn_predictor(inference_input)
        except Exception as e:  # pragma: no cover
            print(f"Faster R-CNN inference failed: {e}")
            return {"error": f"Faster R-CNN inference failed: {e}"}

        instances = outputs["instances"].to("cpu")
        boxes = instances.pred_boxes.tensor.numpy() if instances.has("pred_boxes") else []
        scores = instances.scores.tolist() if instances.has("scores") else []
        classes = instances.pred_classes.tolist() if instances.has("pred_classes") else []

        annotations: List[Dict[str, Any]] = []
        for box, score, cls_id in zip(boxes, scores, classes):
            x1, y1, x2, y2 = box.tolist()
            w_box = x2 - x1
            h_box = y2 - y1
            x_center = x1 + w_box / 2
            y_center = y1 + h_box / 2

            x_center_norm = x_center / original_w if original_w > 0 else 0
            y_center_norm = y_center / original_h if original_h > 0 else 0
            width_norm = w_box / original_w if original_w > 0 else 0
            height_norm = h_box / original_h if original_h > 0 else 0

            class_name = (
                self.frcnn_class_names[cls_id]
                if 0 <= cls_id < len(self.frcnn_class_names)
                else str(cls_id)
            )

            annotations.append(
                {
                    "class_id": cls_id,
                    "class_name": class_name,
                    "confidence": float(score),
                    "x_center_norm": x_center_norm,
                    "y_center_norm": y_center_norm,
                    "width_norm": width_norm,
                    "height_norm": height_norm,
                    "description": f"Detected {class_name} with {score:.2f} confidence",
                }
            )

        output_json = {
            "crop_filename": filename,
            "crop_coordinates_pixels": {"x1": 0, "y1": 0, "x2": original_w, "y2": original_h},
            "crop_width_pixels": original_w,
            "crop_height_pixels": original_h,
            "overlap_ratio": 1.0,
            "area_ratio": 1.0,
            "annotations_in_crop": annotations,
        }
        return output_json

    def _run_rtdetr_inference(
        self,
        img_input: Any,
        original_h: int,
        original_w: int,
        filename: str = "unknown.jpg",
    ) -> Dict[str, Any]:
        """Run inference using RT-DETR model and return formatted JSON."""
        if not self.rtdetr_model:
            return {"error": "RT-DETR model not initialized"}

        inference_input = img_input
        if isinstance(img_input, str) and img_input.startswith("data:image"):
            try:
                encoded_data = img_input.split(",", 1)[1]
                nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
                inference_input = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except Exception as e:  # pragma: no cover
                print(f"Error decoding data URI for RT-DETR inference: {e}")
                return {"error": "Failed to decode image"}

        try:
            results = self.rtdetr_model(inference_input)
        except Exception as e:  # pragma: no cover
            print(f"RT-DETR inference failed: {e}")
            return {"error": f"RT-DETR inference failed: {e}"}

        annotations: List[Dict[str, Any]] = []
        for result in results:
            boxes = result.boxes
            names = result.names
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                class_name = names[cls_id]

                w_box = x2 - x1
                h_box = y2 - y1
                x_center = x1 + w_box / 2
                y_center = y1 + h_box / 2

                x_center_norm = x_center / original_w if original_w > 0 else 0
                y_center_norm = y_center / original_h if original_h > 0 else 0
                width_norm = w_box / original_w if original_w > 0 else 0
                height_norm = h_box / original_h if original_h > 0 else 0

                annotations.append(
                    {
                        "class_id": cls_id,
                        "class_name": class_name,
                        "confidence": confidence,
                        "x_center_norm": x_center_norm,
                        "y_center_norm": y_center_norm,
                        "width_norm": width_norm,
                        "height_norm": height_norm,
                        "description": f"Detected {class_name} with {confidence:.2f} confidence",
                    }
                )

        output_json = {
            "crop_filename": filename,
            "crop_coordinates_pixels": {"x1": 0, "y1": 0, "x2": original_w, "y2": original_h},
            "crop_width_pixels": original_w,
            "crop_height_pixels": original_h,
            "overlap_ratio": 1.0,
            "area_ratio": 1.0,
            "annotations_in_crop": annotations,
        }
        return output_json

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def detect(
        self,
        image_input: Any,
        text_prompt: str = "Detect defects in this image.",
        filename: str = "unknown.jpg",
    ) -> Dict[str, Any]:
        """
        Run detection using local models and return a JSON structure with `annotations_in_crop`.

        Supported models (self.model, case-insensitive):
            - "yolo12m", "yolo11m"
            - "faster_rcnn"
            - "rtdetr"
            - "ensemble" (default): merge results from all available models
            - "intersection": intersection of detections across models
        """
        import asyncio as _asyncio  # local import to avoid global dependency if unused

        h, w = 0, 0
        img_for_local = image_input

        try:
            if isinstance(image_input, np.ndarray):
                h, w = image_input.shape[:2]
            elif isinstance(image_input, str):
                if os.path.exists(image_input):
                    img = cv2.imread(image_input)
                    if img is not None:
                        h, w = img.shape[:2]
                        img_for_local = img
                elif image_input.startswith("data:image"):
                    encoded_data = image_input.split(",", 1)[1]
                    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if img is not None:
                        h, w = img.shape[:2]
                        img_for_local = img
        except Exception as e:  # pragma: no cover
            print(f"Error getting image dimensions: {e}")

        model_lower = (self.model or "").lower()

        # YOLO branch
        if model_lower.startswith("yolo"):
            chosen_model = "yolo11m" if "11" in model_lower else "yolo12m"
            yolo_model = self.yolo_models.get(chosen_model) or self.local_yolo
            if not yolo_model:
                return {"error": f"YOLO model '{chosen_model}' requested but not loaded/available."}
            try:
                return await _asyncio.to_thread(
                    self._run_local_inference, yolo_model, img_for_local, h, w, filename
                )
            except Exception as e:  # pragma: no cover
                print(f"Local YOLO inference failed: {e}.")
                return {"error": f"Local YOLO execution failed: {e}"}

        # RT-DETR branch
        if "rtdetr" in model_lower or "rt-detr" in model_lower:
            if not self.rtdetr_model:
                return {"error": "RT-DETR model requested but not loaded/available."}
            try:
                return await _asyncio.to_thread(
                    self._run_rtdetr_inference, img_for_local, h, w, filename
                )
            except Exception as e:  # pragma: no cover
                print(f"RT-DETR inference failed: {e}.")
                return {"error": f"RT-DETR execution failed: {e}"}

        # Faster R-CNN branch
        if "faster" in model_lower and "rcnn" in model_lower:
            if not self._ensure_faster_rcnn_model():
                return {"error": "Faster R-CNN model requested but not loaded/available."}
            try:
                return await _asyncio.to_thread(
                    self._run_faster_rcnn_inference, img_for_local, h, w, filename
                )
            except Exception as e:  # pragma: no cover
                print(f"Faster R-CNN inference failed: {e}.")
                return {"error": f"Faster R-CNN execution failed: {e}"}

        # Ensemble / intersection: run all models and combine
        if model_lower in ["ensemble", "default", ""] or model_lower == "intersection":
            all_results = await self._run_all_models_detection(img_for_local, filename)
            if not all_results:
                return {"error": "All local models failed or not available."}

            if model_lower in ["ensemble", "default", ""]:
                if "ensemble" in all_results:
                    return all_results["ensemble"]
                return {"error": "Ensemble detection failed or produced no annotations."}

            if model_lower == "intersection":
                if "intersection" in all_results:
                    return all_results["intersection"]
                if "ensemble" in all_results:
                    print("Warning: Intersection not available, falling back to ensemble")
                    return all_results["ensemble"]
                return {"error": "Intersection detection failed or produced no annotations."}

        # Unsupported model
        return {
            "error": (
                f"Model '{self.model}' is not a supported local model. "
                "Supported models: yolo12m, yolo11m, faster_rcnn, rtdetr, ensemble, intersection"
            )
        }

    async def _run_all_models_detection(
        self,
        image_input: Any,
        filename: str = "unknown.jpg",
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run detection with all available models and return results.

        Returns dict with keys: yolo12m, yolo11m, faster_rcnn, rtdetr, ensemble, intersection.
        """
        import asyncio as _asyncio

        h, w = 0, 0
        img_for_local = image_input

        try:
            if isinstance(image_input, np.ndarray):
                h, w = image_input.shape[:2]
            elif isinstance(image_input, str):
                if os.path.exists(image_input):
                    img = cv2.imread(image_input)
                    if img is not None:
                        h, w = img.shape[:2]
                        img_for_local = img
                elif image_input.startswith("data:image"):
                    encoded_data = image_input.split(",", 1)[1]
                    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if img is not None:
                        h, w = img.shape[:2]
                        img_for_local = img
        except Exception as e:  # pragma: no cover
            print(f"Error getting image dimensions: {e}")
            return {}

        results: Dict[str, Dict[str, Any]] = {}

        # YOLO12m
        yolo12 = self.yolo_models.get("yolo12m") or self.local_yolo
        if yolo12:
            try:
                res12 = await _asyncio.to_thread(
                    self._run_local_inference, yolo12, img_for_local, h, w, filename
                )
                if "annotations_in_crop" in res12:
                    results["yolo12m"] = res12
            except Exception as e:  # pragma: no cover
                print(f"YOLO12m detection failed: {e}")

        # YOLO11m
        yolo11 = self.yolo_models.get("yolo11m")
        if yolo11:
            try:
                res11 = await _asyncio.to_thread(
                    self._run_local_inference, yolo11, img_for_local, h, w, filename
                )
                if "annotations_in_crop" in res11:
                    results["yolo11m"] = res11
            except Exception as e:  # pragma: no cover
                print(f"YOLO11m detection failed: {e}")

        # Faster R-CNN
        if self._ensure_faster_rcnn_model():
            try:
                res_frcnn = await _asyncio.to_thread(
                    self._run_faster_rcnn_inference, img_for_local, h, w, filename
                )
                if "annotations_in_crop" in res_frcnn:
                    results["faster_rcnn"] = res_frcnn
            except Exception as e:  # pragma: no cover
                print(f"Faster R-CNN detection failed: {e}")

        # RT-DETR
        if self.rtdetr_model:
            try:
                res_rtdetr = await _asyncio.to_thread(
                    self._run_rtdetr_inference, img_for_local, h, w, filename
                )
                if "annotations_in_crop" in res_rtdetr:
                    results["rtdetr"] = res_rtdetr
            except Exception as e:  # pragma: no cover
                print(f"RT-DETR detection failed: {e}")

        # Ensemble
        if results:
            annotations_all: List[Dict[str, Any]] = []
            for res in results.values():
                if "annotations_in_crop" in res:
                    annotations_all.extend(res["annotations_in_crop"])

            if annotations_all:
                merged_annotations = self._merge_annotations(annotations_all, w, h)
                results["ensemble"] = {
                    "crop_filename": filename,
                    "crop_coordinates_pixels": {"x1": 0, "y1": 0, "x2": w, "y2": h},
                    "crop_width_pixels": w,
                    "crop_height_pixels": h,
                    "overlap_ratio": 1.0,
                    "area_ratio": 1.0,
                    "annotations_in_crop": merged_annotations,
                }

        # Intersection: boxes detected by at least 2 models
        if len(results) >= 2:
            box_votes: Dict[tuple, int] = {}
            iou_thresh = 0.5

            for model_name, res in results.items():
                if model_name in ("ensemble", "intersection"):
                    continue
                if "annotations_in_crop" not in res:
                    continue
                for ann in res["annotations_in_crop"]:
                    box = self._bbox_from_norm(ann, w, h)
                    matched = False
                    for (cls, existing_box), count in list(box_votes.items()):
                        if cls == ann.get("class_name", "unknown"):
                            score = self._compute_overlap_score(box, existing_box)
                            if score >= iou_thresh:
                                box_votes[(cls, existing_box)] = count + 1
                                matched = True
                                break
                    if not matched:
                        box_votes[(ann.get("class_name", "unknown"), tuple(box))] = 1

            intersection_annotations: List[Dict[str, Any]] = []
            for (cls, box_tuple), count in box_votes.items():
                if count < 2:
                    continue
                box = list(box_tuple)
                for model_name, res in results.items():
                    if model_name in ("ensemble", "intersection"):
                        continue
                    if "annotations_in_crop" not in res:
                        continue
                    for ann in res["annotations_in_crop"]:
                        ann_box = self._bbox_from_norm(ann, w, h)
                        if ann.get("class_name", "unknown") == cls:
                            score = self._compute_overlap_score(box, ann_box)
                            if score >= iou_thresh:
                                intersection_annotations.append(ann)
                                break
                    if intersection_annotations and intersection_annotations[-1].get("class_name") == cls:
                        break

            if intersection_annotations:
                intersection_merged = self._merge_annotations(intersection_annotations, w, h)
                results["intersection"] = {
                    "crop_filename": filename,
                    "crop_coordinates_pixels": {"x1": 0, "y1": 0, "x2": w, "y2": h},
                    "crop_width_pixels": w,
                    "crop_height_pixels": h,
                    "overlap_ratio": 1.0,
                    "area_ratio": 1.0,
                    "annotations_in_crop": intersection_merged,
                }

        return results


detection_service = DetectionAgent()


