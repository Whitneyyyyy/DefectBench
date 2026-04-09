import sys
import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import SegformerForSemanticSegmentation, AutoImageProcessor
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import torchvision.transforms as transforms
import traceback
from pathlib import Path

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))

# Always resolve crack dependencies from defect_bench/model_weights
bench_root = next(
    (str(parent) for parent in Path(current_dir).resolve().parents if parent.name == "defect_bench"),
    None,
)
if not bench_root:
    raise RuntimeError("Cannot resolve defect_bench/model_weights path.")
base_weights_dir = os.path.join(bench_root, "model_weights")

crack_seg_dir = os.path.join(base_weights_dir, "crack_segmentation")
cracksam_dir = os.path.join(base_weights_dir, "CrackSAM", "CrackSAM")

class CrackSegmentationService:
    def __init__(self):
        self.models = {}
        self.initialized = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.unet_input_size = [320, 320]
        self.cracksam_img_size = 448
        self.cracksam_input_size = 224

    def initialize(self):
        if self.initialized:
            return
        
        print(f"Initializing Crack Segmentation Service on {self.device}...")

        # 1. Segformer B0
        try:
            print("Loading Segformer B0...")
            model_id_b0 = "eoruadl/segformer-b0-finetuned-segments-crop_crack-2"
            self.proc_b0 = AutoImageProcessor.from_pretrained(model_id_b0)
            self.model_b0 = SegformerForSemanticSegmentation.from_pretrained(model_id_b0).to(self.device)
            self.models['b0'] = True
        except Exception as e:
            print(f"Failed to load Segformer B0: {e}")
            traceback.print_exc()

        # 2. Segformer B4
        try:
            print("Loading Segformer B4...")
            model_id_b4 = "varcoder/segformer-b4-crack-segmentation-dataset"
            try:
                self.proc_b4 = AutoImageProcessor.from_pretrained(model_id_b4)
            except OSError:
                print(f"Warning: Could not load processor from {model_id_b4}. Falling back to nvidia/segformer-b4-finetuned-ade-512-512 (cleaned reduce_labels)")
                from huggingface_hub import hf_hub_download
                import json
                preproc_path = hf_hub_download(repo_id="nvidia/segformer-b4-finetuned-ade-512-512", filename="preprocessor_config.json")
                with open(preproc_path, "r", encoding="utf-8") as f:
                    proc_cfg = json.load(f)
                # Remove deprecated or invalid parameters
                proc_cfg.pop("reduce_labels", None)
                proc_cfg.pop("feature_extractor_type", None)  # Remove deprecated parameter
                from transformers import SegformerImageProcessor
                self.proc_b4 = SegformerImageProcessor(**proc_cfg)
            
            self.model_b4 = SegformerForSemanticSegmentation.from_pretrained(model_id_b4).to(self.device)
            self.models['b4'] = True
        except Exception as e:
            print(f"Failed to load Segformer B4: {e}")
            traceback.print_exc()

        # 3. YOLOv8-crack-seg
        try:
            print("Loading YOLOv8-crack-seg...")
            repo_id = "OpenSistemas/YOLOv8-crack-seg"
            filename = "yolov8n/weights/best.pt"
            weight_path = hf_hub_download(repo_id=repo_id, filename=filename)
            self.model_YOLOv8_crack_seg = YOLO(weight_path)
            self.models['YOLOv8-crack-seg'] = True
        except Exception as e:
            print(f"Failed to load YOLOv8-crack-seg: {e}")
            traceback.print_exc()

        # 4. unet_crack (local model)
        try:
            print("Loading unet_crack (local)...")
            if crack_seg_dir not in sys.path:
                sys.path.insert(0, crack_seg_dir)
            else:
                sys.path.remove(crack_seg_dir)
                sys.path.insert(0, crack_seg_dir)

            model_path = os.path.join(crack_seg_dir, "models/model_unet_vgg_16_best.pt")
            if not os.path.exists(model_path):
                print(f"unet_crack weights not found at {model_path}")
            else:
                from utils import load_unet_vgg16
                from unet.unet_transfer import input_size
                self.unet_input_size = input_size
                self.model_unet_crack = load_unet_vgg16(model_path)
                if not torch.cuda.is_available():
                    self.model_unet_crack = self.model_unet_crack.to(self.device)
                self.models['unet_crack'] = True
        except Exception as e:
            print(f"Failed to load unet_crack: {e}")
            traceback.print_exc()

        # 5. CrackSAM
        try:
            print("Loading CrackSAM...")
            if cracksam_dir not in sys.path:
                sys.path.insert(0, cracksam_dir)
            else:
                sys.path.remove(cracksam_dir)
                sys.path.insert(0, cracksam_dir)

            from importlib import import_module
            from segment_anything import sam_model_registry

            ckpt_backbone = os.path.join(cracksam_dir, "checkpoints/sam_vit_h_4b8939.pth")
            ckpt_delta = os.path.join(cracksam_dir, "checkpoints/CrackSAM_adapter_d32.pth")
            if not os.path.exists(ckpt_backbone) or not os.path.exists(ckpt_delta):
                print(f"CrackSAM checkpoints missing: {ckpt_backbone} or {ckpt_delta}")
            else:
                sam, _ = sam_model_registry["vit_h"](
                    image_size=self.cracksam_img_size,
                    num_classes=1,
                    checkpoint=ckpt_backbone,
                    pixel_mean=[0, 0, 0],
                    pixel_std=[1, 1, 1],
                )
                pkg = import_module("delta.sam_adapter_image_encoder")
                net = pkg.Adapter_Sam(sam, middle_dim=32, scaling_factor=0.2).to(self.device)
                net.load_delta_parameters(ckpt_delta)
                net.eval()
                self.model_cracksam = net
                self.models["CrackSAM"] = True
        except Exception as e:
            print(f"Failed to load CrackSAM: {e}")
            traceback.print_exc()

        if not self.models:
            print("CRITICAL: No models loaded successfully.")
        else:
            self.initialized = True
            print(f"Crack Segmentation Service initialized with models: {list(self.models.keys())}")

    def predict(self, image_np, model_name=None, mode='union'):
        """
        image_np: RGB numpy array
        model_name: Optional[str]. 
                    If provided, runs only that model.
                    If None, runs all available models.
        mode: 'union' (default) -> OR logic (any model says crack, it's crack)
              'voting' -> At least 2 models must agree (used for full-image scan)
        """
        if not self.initialized:
            self.initialize()
            if not self.models:
                raise RuntimeError("No crack models available.")

        return self._evaluate_ensemble(image_np, target_model=model_name, mode=mode)

    def _evaluate_ensemble(self, img, target_model=None, mode='union'):
        h, w = img.shape[:2]
        
        # If voting mode, we need to sum up votes (float). If union, we OR them (uint8).
        # To unify, let's just accumulate counts.
        # If mode='union', threshold=1. If mode='voting', threshold=2.
        
        vote_map = np.zeros((h, w), dtype=np.uint8)
        valid_votes = 0

        # Helper to process model output
        def add_vote(mask_float):
            nonlocal vote_map, valid_votes
            # mask_float is 0.0 or 1.0 (float32)
            vote_map += mask_float.astype(np.uint8)
            valid_votes += 1

        # --- Model 1: Segformer B0 ---
        if (target_model is None or target_model == 'b0') and self.models.get('b0'):
            try:
                inputs = self.proc_b0(images=img, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.model_b0(**inputs)
                    logits = F.interpolate(outputs.logits, size=(h, w), mode="bilinear", align_corners=False)
                    pred = logits.argmax(dim=1)[0].cpu().numpy()
                    mask = (pred == 1).astype(np.float32)
                    add_vote(mask)
            except Exception as e:
                print(f"Error executing Segformer B0: {e}")

        # --- Model 2: Segformer B4 ---
        if (target_model is None or target_model == 'b4') and self.models.get('b4'):
            try:
                inputs = self.proc_b4(images=img, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.model_b4(**inputs)
                    logits = F.interpolate(outputs.logits, size=(h, w), mode="bilinear", align_corners=False)
                    pred = logits.argmax(dim=1)[0].cpu().numpy()
                    mask = (pred == 1).astype(np.float32)
                    add_vote(mask)
            except Exception as e:
                print(f"Error executing Segformer B4: {e}")

        # --- Model 3: YOLOv8-crack-seg ---
        if (target_model is None or target_model == 'YOLOv8-crack-seg') and self.models.get('YOLOv8-crack-seg'):
            try:
                results = self.model_YOLOv8_crack_seg(img, verbose=False)
                mask = np.zeros((h, w), dtype=np.float32)
                if results and results[0].masks is not None:
                    masks_data = results[0].masks.data
                    if masks_data.numel() > 0:
                        masks_resized = F.interpolate(masks_data.unsqueeze(1).float(), size=(h, w), mode="bilinear", align_corners=False)
                        masks_resized = masks_resized.squeeze(1).cpu().numpy()
                        combined = np.max(masks_resized, axis=0)
                        mask = (combined > 0.5).astype(np.float32)
                add_vote(mask)
            except Exception as e:
                print(f"Error executing YOLOv8-crack-seg: {e}")

        # --- Model 4: unet_crack ---
        if (target_model is None or target_model == 'unet_crack') and self.models.get('unet_crack'):
            try:
                mask = self._run_unet_crack(img)
                add_vote(mask)
            except Exception as e:
                print(f"Error executing unet_crack: {e}")

        # --- Model 5: CrackSAM ---
        if (target_model is None or target_model == 'CrackSAM') and self.models.get('CrackSAM'):
            try:
                mask = self._run_cracksam(img)
                add_vote(mask)
            except Exception as e:
                print(f"Error executing CrackSAM: {e}")

        # Decision Logic
        if valid_votes == 0:
            return np.zeros((h, w), dtype=np.uint8)

        # If a specific model was requested, any vote > 0 is valid (since only 1 voter)
        if target_model is not None:
            final_mask = (vote_map >= 1).astype(np.uint8) * 255
            return final_mask

        # Ensemble logic
        if mode == 'voting':
            # At least 2 models must agree
            threshold = 2
            # If fewer than 2 models ran successfully, fallback to 1
            if valid_votes < 2:
                threshold = 1
        else:
            # Union mode (default for bbox crop): Any model says yes
            threshold = 1
        
        final_mask = (vote_map >= threshold).astype(np.uint8) * 255
        return final_mask

    def _run_unet_crack(self, img):
        input_width, input_height = self.unet_input_size[0], self.unet_input_size[1]
        img_height, img_width = img.shape[:2]
        img_resized = cv2.resize(img, (input_width, input_height), interpolation=cv2.INTER_AREA)
        
        channel_means = [0.485, 0.456, 0.406]
        channel_stds = [0.229, 0.224, 0.225]
        train_tfms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(channel_means, channel_stds)])
        
        X = train_tfms(Image.fromarray(img_resized)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mask = self.model_unet_crack(X)
        mask = torch.sigmoid(mask[0, 0]).cpu().numpy()
        mask = cv2.resize(mask, (img_width, img_height), interpolation=cv2.INTER_AREA)
        return (mask > 0.2).astype(np.float32)

    def _run_cracksam(self, img):
        img_height, img_width = img.shape[:2]
        img_resized = cv2.resize(img, (self.cracksam_img_size, self.cracksam_img_size), interpolation=cv2.INTER_AREA)
        img_norm = img_resized.astype(np.float32) / 255.0
        tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model_cracksam(tensor, False, self.cracksam_img_size)
            output_masks = outputs["masks"]
            out = torch.argmax(torch.softmax(output_masks, dim=1), dim=1).squeeze(0).cpu().numpy()
        if out.shape != (self.cracksam_img_size, self.cracksam_img_size):
            out = cv2.resize(out, (self.cracksam_img_size, self.cracksam_img_size), interpolation=cv2.INTER_NEAREST)
        out = cv2.resize(out, (img_width, img_height), interpolation=cv2.INTER_NEAREST)
        return (out > 0).astype(np.float32)

crack_service = CrackSegmentationService()
