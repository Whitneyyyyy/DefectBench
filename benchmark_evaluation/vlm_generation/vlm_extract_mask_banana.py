#!/usr/bin/env python3
"""
Use image generation API (AIML API) to extract defect masks from bbox-only visualized images.

For each visualized image in bbox_only_vis (with numbered bounding boxes and labels):
  1. Read the corresponding JSON file to get all defect types
  2. For each defect type, call image generation API to generate a binary mask
  3. Resize generated image to match original image size
  4. Extract white regions within bboxes for that type
  5. Combine all type masks into a colored mask using unify_mask.py color scheme

The masks are saved under:
    defect_bench/results/{MODEL_NAME}_masks/{class_dir}/{image_stem}_mask.png
"""

import os
import json
import base64
import cv2
import numpy as np
import requests
from pathlib import Path
from typing import Dict, List, Tuple, Optional



# Base directory
TEST100_DIR = Path("defect_bench/data_sample")
IMAGES_DIR = TEST100_DIR / "images"
LABELS_DIR = TEST100_DIR / "labels"
RESULTS_DIR = Path("defect_bench/results")

# Class subdirectories
CLASS_DIRS = ["images"]

# Color mapping: primary_class -> RGB color (same as unify_mask.py)
CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
    "Crack": (255, 0, 0),             # Red
    "Material_loss": (255, 140, 0),   # Orange
    "Stain": (30, 144, 255),          # Blue
    "External Fixings": (0, 200, 0),  # Green
}

# Model configuration
DEFAULT_MODEL_NAME = "google/nano-banana-pro-edit"

# AIML API configuration
API_URL = "https://api.aimlapi.com/v1/images/generations"


def resize_to_square(img: np.ndarray, target_size: Optional[int] = None) -> Tuple[np.ndarray, int, int, int, int]:
    """
    Resize image to square by padding with black borders, maintaining aspect ratio.
    
    Args:
        img: Input image (BGR format from OpenCV)
        target_size: Target square size. If None, use max(img_w, img_h)
    
    Returns:
        Tuple of (square_image, original_w, original_h, pad_left, pad_top)
        pad_left and pad_top are the padding offsets for reverse resize
    """
    h, w = img.shape[:2]
    max_dim = max(w, h)
    
    if target_size is None:
        target_size = max_dim
    
    # Calculate padding to make it square
    pad_w = (target_size - w) // 2
    pad_h = (target_size - h) // 2
    pad_left = pad_w
    pad_top = pad_h
    pad_right = target_size - w - pad_left
    pad_bottom = target_size - h - pad_top
    
    # Pad with black (0, 0, 0)
    square_img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, 
                                     cv2.BORDER_CONSTANT, value=[0, 0, 0])
    
    return square_img, w, h, pad_left, pad_top


def reverse_square_resize(square_img: np.ndarray, orig_w: int, orig_h: int, 
                          pad_left: int, pad_top: int) -> np.ndarray:
    """
    Reverse the square resize operation, cropping back to original dimensions.
    
    Args:
        square_img: Square image (from API output)
        orig_w: Original image width
        orig_h: Original image height
        pad_left: Left padding that was added
        pad_top: Top padding that was added
    
    Returns:
        Image cropped back to original dimensions
    """
    h, w = square_img.shape[:2]
    
    # Crop back to original size
    # Remove the padding we added
    crop_x1 = pad_left
    crop_y1 = pad_top
    crop_x2 = crop_x1 + orig_w
    crop_y2 = crop_y1 + orig_h
    
    # Ensure crop coordinates are within bounds
    crop_x1 = max(0, min(crop_x1, w - 1))
    crop_y1 = max(0, min(crop_y1, h - 1))
    crop_x2 = max(crop_x1 + 1, min(crop_x2, w))
    crop_y2 = max(crop_y1 + 1, min(crop_y2, h))
    
    cropped = square_img[crop_y1:crop_y2, crop_x1:crop_x2]
    
    # If the cropped image is smaller than original (shouldn't happen, but just in case)
    if cropped.shape[1] != orig_w or cropped.shape[0] != orig_h:
        cropped = cv2.resize(cropped, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    
    return cropped


def encode_image_square(image_path: Path, target_size: Optional[int] = None) -> Tuple[str, int, int, int, int]:
    """
    将本地图片转换为正方形并编码为 Base64 data URL 字符串
    
    Args:
        image_path: 图片文件路径
        target_size: 目标正方形尺寸，如果为None则使用max(w, h)
    
    Returns:
        Tuple of (base64_data_url, original_w, original_h, pad_left, pad_top)
    """
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")
    
    # Resize to square
    square_img, orig_w, orig_h, pad_left, pad_top = resize_to_square(img, target_size)
    
    # Encode to base64
    # Get file extension for mime type
    ext = image_path.suffix.lower().replace(".", "")
    if ext == "jpg":
        ext = "jpeg"
    elif ext == "png":
        ext = "png"
    else:
        ext = "jpeg"
    
    # Encode square image
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95] if ext == "jpeg" else []
    success, img_bytes = cv2.imencode(f'.{ext}', square_img, encode_param)
    if not success:
        raise ValueError(f"Failed to encode image: {image_path}")
    
    encoded_string = base64.b64encode(img_bytes.tobytes()).decode('utf-8')
    data_url = f"data:image/{ext};base64,{encoded_string}"
    
    return data_url, orig_w, orig_h, pad_left, pad_top


def build_prompt_for_mask_generation(primary_class: str) -> str:
    """
    Build prompt for image generation API to generate a binary mask for a specific defect type.
    
    Args:
        primary_class: One of "Crack", "Material_loss", "Stain", "External Fixings"
    
    Returns:
        Prompt string for image generation
    """
    # Get class definition from vlm_generate_qa.py style
    class_definitions = {
        "Crack": "Any type of crack or fissure in the building surface, including fine cracks, wide cracks, through-cracks, and stair-step cracks.",
        "Material_loss": "Loss of material from the building surface, including peeling, spalling, flakes, peeling_paint, Abscission, Bulge, and similar forms of material detachment or bulging.",
        "Stain": "Discoloration or staining on the building surface, including algae, stain, biological_deteriorations, mold, water_seepage, Dampness, Efflorescence, Leakage, Corrosion, and chemical_deteriorations—any kind of color change, seepage marks, efflorescence, or corrosion stains on the surface.",
        "External Fixings": "External objects or human-made additions attached to the building surface, including human_caused_damages (such as graffiti, vandalism, scratches, or other man-made marks) and Vegetation (plants, moss, or other vegetation growing on the surface)."
    }
    
    class_def = class_definitions.get(primary_class, "")
    
    prompt = (
        f"You will be given an image of defects on a building surface with numbered bounding boxes and labels.\n"
        f"Your task is to **edit the input image directly** by painting black and white to create a **binary segmentation mask** for **{primary_class} only**.\n\n"
        f"**Definition of {primary_class}:**\n"
        f"{class_def}\n\n"
        f"CRITICAL Requirements:\n"
        f"1. **Edit the input image directly** - Do NOT generate a new image. Start from the input image and modify it by painting:\n"
        f"   - Paint defect pixels (belonging to **{primary_class}**): use white (255)\n"
        f"   - Paint all other pixels (background, non-{primary_class} areas): use black (0)\n"
        f"2. **Remove ALL annotations** - You MUST paint over and remove:\n"
        f"   - All bounding boxes (rectangles)\n"
        f"   - All labels and text (like \"1#Crack\", \"2#Material_loss\", etc.)\n"
        f"   - All numbers and annotations\n"
        f"   - Paint all of these as black (0) - they should NOT appear in the final mask\n"
        f"3. **Keep the exact same dimensions** - The output must have exactly the same width and height as the input image. Do not change the aspect ratio, make it square, or add any borders.\n"
        f"4. **Maintain the same composition** - Do not change the camera viewpoint, scaling, or crop.\n"
        f"5. Focus on the areas within the bounding boxes labeled with \"{primary_class}\" in the image, but extract ONLY the actual defect pixels. Do NOT include the bounding boxes, labels, numbers, or any text in the mask.\n\n"
        f"Output the edited input image as a pure binary mask: white pixels ONLY for **{primary_class}** defect regions, black pixels for everything else (including all annotations, boxes, labels, and text), with the exact same dimensions as the input image."
    )
    return prompt


def extract_text_from_response(response) -> Optional[str]:
    """
    Extract main text content from Ark response.
    """
    try:
        for item in response.output:
            if hasattr(item, "content") and item.content:
                first = item.content[0]
                if hasattr(first, "text"):
                    return first.text.strip()
    except Exception:
        return None
    return None


def extract_thinking_from_response(response) -> Optional[str]:
    """
    Extract thinking/chain-of-thought content from Ark response if available.
    """
    try:
        for item in response.output:
            if hasattr(item, "type") and item.type == "reasoning" and hasattr(item, "summary"):
                summary = item.summary
                if summary and len(summary) > 0:
                    first = summary[0]
                    if hasattr(first, "text"):
                        return str(first.text).strip()
                    return str(first).strip()
    except Exception:
        return None
    return None


def extract_json_from_text(text: str) -> Optional[str]:
    """
    Extract JSON object from text that may contain extra commentary.
    Finds the first complete JSON object (from first '{' to matching '}').
    """
    if not text:
        return None
    
    # Try direct parse first (most common case)
    try:
        json.loads(text)
        return text
    except:
        pass
    
    # Find first '{' and try to extract complete JSON object
    start_idx = text.find('{')
    if start_idx == -1:
        return None
    
    # Find matching closing '}'
    brace_count = 0
    for i in range(start_idx, len(text)):
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                # Found complete JSON object
                json_candidate = text[start_idx:i+1]
                try:
                    json.loads(json_candidate)
                    return json_candidate
                except:
                    pass
    
    return None


def polygon_to_mask(polygon: List[List[float]], img_h: int, img_w: int) -> np.ndarray:
    """
    Convert polygon coordinates to binary mask.
    
    Args:
        polygon: List of [x, y] coordinates (absolute pixel coordinates)
        img_h: Image height
        img_w: Image width
    
    Returns:
        Binary mask (uint8, 0 or 255)
    """
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    
    if not polygon or len(polygon) < 3:
        return mask
    
    # Convert to numpy array of points
    points = np.array(polygon, dtype=np.int32)
    
    # Clip points to image bounds
    points[:, 0] = np.clip(points[:, 0], 0, img_w - 1)
    points[:, 1] = np.clip(points[:, 1], 0, img_h - 1)
    
    # Fill polygon
    cv2.fillPoly(mask, [points], 255)
    
    return mask




def call_image_generation_for_mask(api_key: str, api_url: str, model_name: str, image_path: Path,
                                    primary_class: str, img_h: int, img_w: int,
                                    save_raw_image_path: Optional[Path] = None) -> Optional[np.ndarray]:
    """
    Call AIML API image generation to generate a binary mask for a specific defect type.
    
    Args:
        api_key: AIML API key
        api_url: API endpoint URL
        model_name: Model name for image generation
        image_path: Path to input image file
        primary_class: Defect type ("Crack", "Material_loss", "Stain", "External Fixings")
        img_h: Original image height
        img_w: Original image width
        save_raw_image_path: Optional path to save the raw generated image before processing (for debugging)
    
    Returns:
        Binary mask (H, W) as uint8 array (0 or 255), or None on failure
    """
    prompt = build_prompt_for_mask_generation(primary_class)
    
    try:
        # Resize input image to square and convert to base64
        # This ensures API-generated square mask will match the square input
        base64_image, orig_w, orig_h, pad_left, pad_top = encode_image_square(image_path)
        
        print(f"    Resized input to square: original {orig_w}x{orig_h}, padding: left={pad_left}, top={pad_top}")
        
        # Build payload according to AIML API format
        payload = {
            "model": model_name,
            "image_urls": [base64_image],  # Use base64 data directly (square image)
            "prompt": prompt
        }
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        print(f"    Calling AIML API: model={model_name}, image_urls=[base64 data], prompt_length={len(prompt)}")
        
        # Call AIML API
        response = requests.post(api_url, headers=headers, json=payload, timeout=120)
        
        if response.status_code != 200:
            print(f"  Warning: API request failed with status {response.status_code}: {response.text}")
            return None
        
        # Parse response JSON
        try:
            result = response.json()
        except Exception as e:
            print(f"  Warning: Failed to parse JSON response: {e}")
            print(f"  Response text: {response.text[:500]}")
            return None
        
        # AIML API response format: {"data": [{"url": "...", "b64_json": null}]}
        img_url = None
        img_b64 = None
        
        if isinstance(result, dict):
            if "data" in result and isinstance(result["data"], list) and len(result["data"]) > 0:
                first_item = result["data"][0]
                if isinstance(first_item, dict):
                    # Try "url" field first
                    if "url" in first_item and first_item["url"]:
                        img_url = first_item["url"]
                        print(f"    Found image URL in response")
                    # Try "b64_json" as fallback
                    elif "b64_json" in first_item and first_item["b64_json"]:
                        img_b64 = first_item["b64_json"]
                        print(f"    Found base64 image in response")
        
        if not img_url and not img_b64:
            print(f"  Warning: No image data found in response for {primary_class}")
            print(f"  Response structure: {type(result)}")
            if isinstance(result, dict):
                print(f"  Response keys: {list(result.keys())}")
                if "data" in result:
                    print(f"  Data: {result['data']}")
            print(f"  Full response (first 1000 chars): {json.dumps(result, indent=2)[:1000]}")
            return None
        
        # Decode image from URL or base64
        if img_b64:
            # Decode base64
            img_bytes = base64.b64decode(img_b64)
            img_array = np.frombuffer(img_bytes, np.uint8)
            generated_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        elif img_url:
            # Download from URL
            img_response = requests.get(img_url, timeout=60)
            img_array = np.frombuffer(img_response.content, np.uint8)
            generated_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        else:
            return None
        
        if generated_img is None:
            print(f"  Warning: Failed to decode generated image for {primary_class}")
            return None
        
        # Save raw generated image before any processing (for debugging)
        if save_raw_image_path is not None:
            try:
                cv2.imwrite(str(save_raw_image_path), generated_img)
                print(f"    Saved raw generated image to: {save_raw_image_path.name}")
            except Exception as e:
                print(f"    Warning: Failed to save raw image: {e}")
        
        # Reverse the square resize: crop back to original dimensions
        # API returns square image, we need to remove the padding we added
        gen_h, gen_w = generated_img.shape[:2]
        print(f"    Generated image size: {gen_w}x{gen_h}, reversing to original: {orig_w}x{orig_h}")
        
        # Calculate expected square size (should match the square input we sent)
        expected_square_size = max(orig_w, orig_h)
        
        # If generated image is not square or wrong size, resize it first
        if gen_w != gen_h or gen_w != expected_square_size:
            print(f"    Warning: Generated image is not square or wrong size. Resizing to {expected_square_size}x{expected_square_size}")
            generated_img = cv2.resize(generated_img, (expected_square_size, expected_square_size), interpolation=cv2.INTER_NEAREST)
        
        # Reverse square resize: crop out the padding we added
        generated_img = reverse_square_resize(generated_img, orig_w, orig_h, pad_left, pad_top)
        
        # Verify final size matches original
        final_h, final_w = generated_img.shape[:2]
        if final_w != img_w or final_h != img_h:
            print(f"    Warning: Size mismatch after reverse resize. Expected {img_w}x{img_h}, got {final_w}x{final_h}")
            # Force resize to match
            generated_img = cv2.resize(generated_img, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        
        # Convert to grayscale
        if len(generated_img.shape) == 3:
            gray = cv2.cvtColor(generated_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = generated_img
        
        # Threshold: anything close to white (>= 200) becomes 255, else 0
        _, binary_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        return binary_mask
        
    except Exception as e:
        print(f"  Error calling image generation for {primary_class}: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_defect_types_from_json(json_path: Path) -> set:
    """
    Read JSON file and extract all unique primary_class values.
    
    Returns:
        Set of primary_class strings (e.g., {"Crack", "Material_loss"})
    """
    if not json_path.exists():
        return set()
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        bboxes = data.get("bboxes", [])
        types = set()
        for bbox_data in bboxes:
            taxonomy = bbox_data.get("taxonomy", {})
            primary_class = taxonomy.get("primary_class")
            if primary_class:
                types.add(primary_class)
        return types
    except Exception as e:
        print(f"  Warning: Failed to read JSON {json_path}: {e}")
        return set()


def get_bboxes_for_class(json_path: Path, primary_class: str) -> List[Dict]:
    """
    Read JSON file and return all bboxes for a specific primary_class.
    
    Returns:
        List of bbox dicts with "bbox" and "taxonomy" keys
    """
    if not json_path.exists():
        return []
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        bboxes = data.get("bboxes", [])
        result = []
        for bbox_data in bboxes:
            taxonomy = bbox_data.get("taxonomy", {})
            if taxonomy.get("primary_class") == primary_class:
                result.append(bbox_data)
        return result
    except Exception as e:
        print(f"  Warning: Failed to read JSON {json_path}: {e}")
        return []


def extract_mask_from_generated_image(generated_mask: np.ndarray, bboxes_for_class: List[Dict],
                                      img_h: int, img_w: int, primary_class: str) -> np.ndarray:
    """
    Extract mask regions from generated image within specified bboxes.
    Only keep white regions that are within the bboxes for this class.
    
    For Crack type: if white pixels > black pixels in a bbox, invert that bbox region.
    
    Args:
        generated_mask: Binary mask from image generation (H, W)
        bboxes_for_class: List of bbox dicts for this class
        img_h: Image height
        img_w: Image width
        primary_class: Defect type (to apply special logic for Crack)
    
    Returns:
        Binary mask (H, W) with only bbox regions kept
    """
    result_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    
    for bbox_data in bboxes_for_class:
        bbox = bbox_data.get("bbox", [])
        if len(bbox) != 4:
            continue
        
        x, y, w, h = bbox
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)
        
        # Clip to image bounds
        x1 = max(0, min(x1, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        x2 = max(x1 + 1, min(x2, img_w))
        y2 = max(y1 + 1, min(y2, img_h))
        
        # Extract region from generated mask
        region = generated_mask[y1:y2, x1:x2].copy()
        
        # Special logic for Crack: if white pixels > black pixels, invert
        if primary_class == "Crack":
            white_count = np.sum(region == 255)
            black_count = np.sum(region == 0)
            if white_count > black_count:
                # Invert the region (255 -> 0, 0 -> 255)
                region = cv2.bitwise_not(region)
                print(f"      Inverted Crack bbox [{x1},{y1},{x2},{y2}] (white={white_count} > black={black_count})")
        
        result_mask[y1:y2, x1:x2] = cv2.bitwise_or(result_mask[y1:y2, x1:x2], region)
    
    return result_mask


def generate_colored_mask_from_defects(defects: List[Dict], img_h: int, img_w: int) -> np.ndarray:
    """
    Generate colored mask from VLM-extracted defect masks.
    
    Args:
        defects: List of defect dicts with mask_polygon
        img_h: Image height
        img_w: Image width
    
    Returns:
        Colored mask (H, W, 3) with RGB colors
    """
    colored_mask = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    
    for defect in defects:
        defect_type = defect.get("type", "")
        # Normalize class name
        if defect_type.lower() in ["crack", "cracks"]:
            primary_class = "Crack"
        elif defect_type.lower() in ["material_loss", "material loss"]:
            primary_class = "Material_loss"
        elif defect_type.lower() in ["stain", "stains"]:
            primary_class = "Stain"
        elif defect_type.lower() in ["external fixings", "external_fixings"]:
            primary_class = "External Fixings"
        else:
            print(f"  Warning: Unknown defect type '{defect_type}', skipping")
            continue
        
        if primary_class not in CLASS_COLORS:
            print(f"  Warning: No color mapping for '{primary_class}', skipping")
            continue
        
        color = CLASS_COLORS[primary_class]
        mask_polygon = defect.get("mask_polygon", [])
        
        if not mask_polygon or len(mask_polygon) < 3:
            print(f"  Warning: Invalid mask_polygon for defect {defect.get('number')}, skipping")
            continue
        
        # Convert polygon to binary mask
        binary_mask = polygon_to_mask(mask_polygon, img_h, img_w)
        
        # Apply color
        colored_mask[binary_mask > 0] = color
    
    return colored_mask


def main():
    """
    Main entry: iterate over bbox-only visualized images, use image generation to create masks.
    """
    api_key = os.environ.get("AIMLAPI_KEY")
    if not api_key:
        print("ERROR: AIMLAPI_KEY is not set in environment.")
        return

    model_name = os.environ.get("VLM_MODEL_NAME", DEFAULT_MODEL_NAME)
    api_url = os.environ.get("API_URL", API_URL)
    
    # Sanitize model_name for use in directory path (replace / with -)
    model_name_safe = model_name.replace("/", "-").replace("\\", "-")
    output_root = RESULTS_DIR / f"{model_name_safe}_masks"
    output_root.mkdir(parents=True, exist_ok=True)
    bbox_vis_dir = Path("defect_bench/results/visualization/images")

    print("Extracting defect masks using image generation API (AIML API)...")
    print("=" * 60)
    print(f"API URL: {api_url}")
    print(f"Model name: {model_name}")
    print(f"Output root: {output_root}")

    total_processed = 0
    total_errors = 0

    for class_dir_name in CLASS_DIRS:
        vis_class_dir = bbox_vis_dir / class_dir_name
        if not vis_class_dir.exists():
            print(f"Warning: Directory {vis_class_dir} does not exist, skipping...")
            continue

        # Original class dir for JSON files
        orig_class_dir = LABELS_DIR

        # Save masks to {MODEL_NAME}_masks/{class_dir_name}/
        mask_output_dir = output_root / class_dir_name
        mask_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing {class_dir_name}...")

        # Iterate over visualized images
        vis_images = list(vis_class_dir.glob("*_visualized.jpg")) + list(vis_class_dir.glob("*_visualized.png"))
        
        for vis_image_path in vis_images:
            image_stem = vis_image_path.stem.replace("_visualized", "")
            
            # Output file path
            mask_output_path = mask_output_dir / f"{image_stem}_mask.png"
            if mask_output_path.exists():
                # Skip if already generated
                total_processed += 1
                continue

            try:
                # Read image to get dimensions
                img_bgr = cv2.imread(str(vis_image_path))
                if img_bgr is None:
                    print(f"  Warning: Failed to read image {vis_image_path}, skipping...")
                    total_errors += 1
                    continue
                
                img_h, img_w = img_bgr.shape[:2]
                
                # Read JSON to get defect types
                json_path = orig_class_dir / f"{image_stem}.json"
                defect_types = get_defect_types_from_json(json_path)
                
                if not defect_types:
                    print(f"  Warning: No defect types found in {json_path}, skipping...")
                    total_errors += 1
                    continue
                
                print(f"  Processing {image_stem}: found types {defect_types}")
                
                # Generate mask for each type
                colored_mask = np.zeros((img_h, img_w, 3), dtype=np.uint8)
                
                for primary_class in defect_types:
                    if primary_class not in CLASS_COLORS:
                        print(f"  Warning: Unknown class {primary_class}, skipping...")
                        continue
                    
                    print(f"    Generating mask for {primary_class}...")
                    
                    # Path to save raw generated image
                    raw_image_path = mask_output_dir / f"{image_stem}_{primary_class}_raw_generated.png"
                    
                    # Call image generation API
                    binary_mask = call_image_generation_for_mask(
                        api_key, api_url, model_name, vis_image_path, primary_class, img_h, img_w,
                        save_raw_image_path=raw_image_path
                    )
                    
                    if binary_mask is None:
                        print(f"    Warning: Failed to generate mask for {primary_class}")
                        continue
                    
                    # Get bboxes for this class
                    bboxes_for_class = get_bboxes_for_class(json_path, primary_class)
                    
                    # Extract mask regions within bboxes
                    if bboxes_for_class:
                        extracted_mask = extract_mask_from_generated_image(
                            binary_mask, bboxes_for_class, img_h, img_w, primary_class
                        )
                    else:
                        extracted_mask = binary_mask
                    
                    # Save individual type mask (binary, 0/255)
                    type_mask_path = mask_output_dir / f"{image_stem}_{primary_class}_mask.png"
                    cv2.imwrite(str(type_mask_path), extracted_mask)
                    print(f"    Saved {primary_class} mask to {type_mask_path.name}")
                    
                    # Apply color
                    color = CLASS_COLORS[primary_class]
                    colored_mask[extracted_mask > 0] = color
                
                if not np.any(colored_mask > 0):
                    print(f"  Warning: No valid masks generated for {vis_image_path}, skipping...")
                    total_errors += 1
                    continue
                
                # Save mask (convert RGB to BGR for OpenCV)
                cv2.imwrite(str(mask_output_path), cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR))
                
                # Generate overlay visualization (mask on bbox-only visualized image)
                overlay_path = mask_output_dir / f"{image_stem}_mask_overlay.jpg"
                try:
                    # Read the bbox-only visualized image
                    vis_img = cv2.imread(str(vis_image_path))
                    if vis_img is not None:
                        # Ensure mask and image have same size
                        if colored_mask.shape[:2] != vis_img.shape[:2]:
                            mask_resized = cv2.resize(
                                cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR),
                                (vis_img.shape[1], vis_img.shape[0]),
                                interpolation=cv2.INTER_NEAREST
                            )
                        else:
                            mask_resized = cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR)
                        
                        # Create alpha channel from mask (non-black regions)
                        mask_gray = cv2.cvtColor(mask_resized, cv2.COLOR_BGR2GRAY)
                        alpha = (mask_gray > 0).astype(np.float32)[:, :, np.newaxis]
                        
                        # Overlay mask on visualized image (40% transparency)
                        vis_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
                        mask_rgb = cv2.cvtColor(mask_resized, cv2.COLOR_BGR2RGB)
                        overlay = vis_rgb.astype(np.float32) * (1 - 0.4 * alpha) + mask_rgb.astype(np.float32) * (0.4 * alpha)
                        overlay = overlay.astype(np.uint8)
                        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                        
                        cv2.imwrite(str(overlay_path), overlay_bgr)
                        print(f"  Saved overlay visualization to {overlay_path.name}")
                except Exception as e:
                    print(f"  Warning: Failed to create overlay visualization: {e}")
                
                total_processed += 1
                print(f"  Saved mask for {image_stem} -> {mask_output_path}")

            except Exception as e:
                print(f"  Error processing {vis_image_path}: {e}")
                import traceback
                traceback.print_exc()
                total_errors += 1

    print("\n" + "=" * 60)
    print("Mask extraction finished.")
    print(f"  Processed (including skipped-existing): {total_processed}")
    print(f"  Errors: {total_errors}")
    print(f"  Output directory: {output_root}")
    print("=" * 60)


if __name__ == "__main__":
    main()

