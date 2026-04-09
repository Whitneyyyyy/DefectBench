#!/usr/bin/env python3
"""
Use image generation API (DashScope/Bailian) to extract defect masks from bbox-only visualized images.

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

import dashscope
from dashscope.aigc.image_generation import ImageGeneration
from dashscope.api_entities.dashscope_response import Message


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
DEFAULT_MODEL_NAME = "wan2.6-image"

# DashScope API configuration
# 以下为北京地域base_url，各地域的base_url不同
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/api/v1"


def encode_image_to_data_url(image_path: Path) -> str:
    """Encode image file to data URL (base64) for Ark input_image."""
    suffix = image_path.suffix.lower()
    if suffix in [".png"]:
        mime = "image/png"
    else:
        mime = "image/jpeg"

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


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


def get_size_string(img_w: int, img_h: int) -> Optional[str]:
    """
    Convert image dimensions to DashScope size format.
    Returns None to let API auto-determine size, or a size string if needed.
    
    DashScope API requires total pixels between 589824 (768*768) and 1638400 (1280*1280).
    If we need to specify size, we'll use a square size that fits within limits.
    """
    # Try to let API auto-determine size by returning None
    # If API requires size parameter, we'll use square format
    # But ideally we should not specify size to preserve aspect ratio
    return None  # Let API decide based on input image


def call_image_generation_for_mask(api_key: str, model_name: str, image_path: Path,
                                    primary_class: str, img_h: int, img_w: int) -> Optional[np.ndarray]:
    """
    Call DashScope image generation API to generate a binary mask for a specific defect type.
    
    Args:
        api_key: DashScope API key
        model_name: Model name for image generation
        image_path: Path to input image file
        primary_class: Defect type ("Crack", "Material_loss", "Stain", "External Fixings")
        img_h: Original image height
        img_w: Original image width
    
    Returns:
        Binary mask (H, W) as uint8 array (0 or 255), or None on failure
    """
    prompt = build_prompt_for_mask_generation(primary_class)
    
    try:
        size_str = get_size_string(img_w, img_h)
        
        # Check if image dimensions are within API limits [384, 5000]
        # API requires dimensions in [384, 5000]
        max_dim = max(img_w, img_h)
        min_dim = min(img_w, img_h)
        temp_image_path = None
        scale_factor = 1.0
        needs_resize = False
        
        # Check if resize is needed
        if max_dim > 5000 or min_dim < 384:
            needs_resize = True
            
            # Read image first
            img = cv2.imread(str(image_path))
            if img is None:
                print(f"  Warning: Failed to read image {image_path}")
                return None
            
            # Calculate scale factor
            if max_dim > 5000:
                # Need to shrink: scale down to fit max dimension
                scale_factor = 5000.0 / max_dim
                new_w = int(img_w * scale_factor)
                new_h = int(img_h * scale_factor)
                
                # After shrinking, check if min dimension is still >= 384
                if min(new_w, new_h) < 384:
                    # Need to scale up to meet minimum, but ensure max doesn't exceed 5000
                    min_scale = 384.0 / min_dim
                    max_scale = 5000.0 / max_dim
                    # Use the smaller scale to ensure both constraints are met
                    scale_factor = min(min_scale, max_scale)
                    new_w = int(img_w * scale_factor)
                    new_h = int(img_h * scale_factor)
            elif min_dim < 384:
                # Need to enlarge: scale up to meet minimum dimension
                scale_factor = 384.0 / min_dim
                new_w = int(img_w * scale_factor)
                new_h = int(img_h * scale_factor)
                
                # After enlarging, check if max dimension exceeds 5000
                if max(new_w, new_h) > 5000:
                    # Need to scale down to fit max dimension
                    max_scale = 5000.0 / max(new_w, new_h)
                    scale_factor = scale_factor * max_scale
                    new_w = int(img_w * scale_factor)
                    new_h = int(img_h * scale_factor)
            
            # Ensure final dimensions are within bounds
            new_w = max(384, min(5000, new_w))
            new_h = max(384, min(5000, new_h))
            
            print(f"    Resizing input image from {img_w}x{img_h} to {new_w}x{new_h} (scale={scale_factor:.3f}) to fit API limits [384, 5000]")
            
            # Resize image
            resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # Save to temporary file
            import tempfile
            temp_dir = tempfile.gettempdir()
            temp_image_path = Path(temp_dir) / f"temp_bailian_{image_path.stem}_{primary_class}.jpg"
            cv2.imwrite(str(temp_image_path), resized_img)
            image_file_path = f"file://{temp_image_path.absolute()}"
        else:
            # Use original image (dimensions are within valid range)
            image_file_path = f"file://{image_path.absolute()}"
        
        message = Message(
            role="user",
            content=[
                {
                    "text": prompt
                },
                {
                    "image": image_file_path
                }
            ]
        )
        
        # Call DashScope ImageGeneration API
        # Try without size parameter first to preserve aspect ratio
        call_params = {
            "model": model_name,
            "api_key": api_key,
            "messages": [message],
            "negative_prompt": "",
            "prompt_extend": True,
            "watermark": False,
            "n": 1,
            "enable_interleave": False
        }
        
        # Only add size if specified (None means let API decide)
        if size_str is not None:
            call_params["size"] = size_str
        
        rsp = ImageGeneration.call(**call_params)
        
        # Check if response is successful
        if rsp.status_code != 200:
            error_msg = getattr(rsp, 'message', 'Unknown error')
            print(f"  Warning: API request failed with status {rsp.status_code}: {error_msg}")
            return None
        
        # Extract image URL from response
        # DashScope response structure: rsp.output.choices[0].message.content[0].image
        img_url = None
        try:
            if hasattr(rsp, 'output') and rsp.output:
                # Check for choices structure (standard DashScope format)
                if hasattr(rsp.output, 'choices') and rsp.output.choices:
                    choice = rsp.output.choices[0]
                    if hasattr(choice, 'message') and choice.message:
                        if hasattr(choice.message, 'content') and choice.message.content:
                            # content is a list, find the image item
                            for item in choice.message.content:
                                if isinstance(item, dict):
                                    if 'image' in item:
                                        img_url = item['image']
                                        break
                                elif hasattr(item, 'image'):
                                    img_url = item.image
                                    break
                                elif hasattr(item, 'get'):
                                    img_url = item.get('image')
                                    if img_url:
                                        break
                
                # Fallback: try results structure (if API uses different format)
                if not img_url and hasattr(rsp.output, 'results') and rsp.output.results:
                    result = rsp.output.results[0]
                    if isinstance(result, dict):
                        img_url = result.get('url') or result.get('image')
                    elif hasattr(result, 'url'):
                        img_url = result.url
                    elif hasattr(result, 'image'):
                        img_url = result.image
                
                # Fallback: try data structure
                if not img_url and hasattr(rsp.output, 'data') and rsp.output.data:
                    data = rsp.output.data[0]
                    if isinstance(data, dict):
                        img_url = data.get('url') or data.get('image')
                    elif hasattr(data, 'url'):
                        img_url = data.url
                    elif hasattr(data, 'image'):
                        img_url = data.image
        except Exception as e:
            print(f"  Warning: Error parsing response structure: {e}")
            import traceback
            traceback.print_exc()
        
        if not img_url:
            print(f"  Warning: No URL found in response for {primary_class}")
            print(f"  Response structure: {type(rsp)}")
            if hasattr(rsp, 'output'):
                print(f"  Output type: {type(rsp.output)}")
                if hasattr(rsp.output, 'choices'):
                    print(f"  Choices: {rsp.output.choices}")
            return None
        
        # Download image from URL
        img_response = requests.get(img_url)
        img_array = np.frombuffer(img_response.content, np.uint8)
        generated_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if generated_img is None:
            print(f"  Warning: Failed to decode generated image for {primary_class}")
            # Clean up temp file if exists
            if temp_image_path and temp_image_path.exists():
                try:
                    temp_image_path.unlink()
                except:
                    pass
            return None
        
        # If we resized the input, we need to resize the output back to original size
        if scale_factor != 1.0:
            print(f"    Resizing generated image back to original size: {img_w}x{img_h}")
            generated_img = cv2.resize(generated_img, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        
        # Clean up temp file if exists
        if temp_image_path and temp_image_path.exists():
            try:
                temp_image_path.unlink()
            except:
                pass
        
        # Resize to match original image size
        # Handle aspect ratio mismatch (API may return square images)
        gen_h, gen_w = generated_img.shape[:2]
        
        if gen_h != img_h or gen_w != img_w:
            gen_aspect = gen_w / gen_h
            orig_aspect = img_w / img_h
            
            # If aspect ratios differ significantly, we need to handle it carefully
            if abs(gen_aspect - orig_aspect) > 0.01:
                print(f"      Resizing from {gen_w}x{gen_h} (aspect {gen_aspect:.3f}) to {img_w}x{img_h} (aspect {orig_aspect:.3f})")
                
                # Strategy: Scale to cover the larger dimension, then crop center
                # This ensures we don't lose important mask regions
                scale_w = img_w / gen_w
                scale_h = img_h / gen_h
                scale = max(scale_w, scale_h)  # Use larger scale to ensure coverage
                
                new_w = int(gen_w * scale)
                new_h = int(gen_h * scale)
                
                # Resize maintaining aspect ratio
                generated_img = cv2.resize(generated_img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                
                # Crop center to match original dimensions
                if new_w > img_w:
                    crop_left = (new_w - img_w) // 2
                    generated_img = generated_img[:, crop_left:crop_left+img_w]
                if new_h > img_h:
                    crop_top = (new_h - img_h) // 2
                    generated_img = generated_img[crop_top:crop_top+img_h, :]
                
                # If we need to pad (shouldn't happen with max scale, but just in case)
                if generated_img.shape[1] < img_w:
                    pad_left = (img_w - generated_img.shape[1]) // 2
                    pad_right = img_w - generated_img.shape[1] - pad_left
                    generated_img = cv2.copyMakeBorder(generated_img, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
                if generated_img.shape[0] < img_h:
                    pad_top = (img_h - generated_img.shape[0]) // 2
                    pad_bottom = img_h - generated_img.shape[0] - pad_top
                    generated_img = cv2.copyMakeBorder(generated_img, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=0)
            else:
                # Aspect ratios match, direct resize
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
        # Clean up temp file if exists
        if 'temp_image_path' in locals() and temp_image_path and temp_image_path.exists():
            try:
                temp_image_path.unlink()
            except:
                pass
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
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        print("ERROR: DASHSCOPE_API_KEY is not set in environment.")
        return

    model_name = os.environ.get("VLM_MODEL_NAME", DEFAULT_MODEL_NAME)
    base_url = os.environ.get("DASHSCOPE_BASE_URL", DASHSCOPE_BASE_URL)
    
    # Set DashScope base URL
    dashscope.base_http_api_url = base_url
    
    output_root = RESULTS_DIR / f"{model_name}_masks"
    output_root.mkdir(parents=True, exist_ok=True)
    bbox_vis_dir = Path("defect_bench/results/visualization/images")

    print("Extracting defect masks using image generation API (DashScope/Bailian)...")
    print("=" * 60)
    print(f"Base URL: {base_url}")
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
                    
                    # Call image generation API
                    binary_mask = call_image_generation_for_mask(
                        api_key, model_name, vis_image_path, primary_class, img_h, img_w
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

