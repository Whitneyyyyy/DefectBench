#!/usr/bin/env python3
"""
Use image generation API (OpenAI) to extract defect masks from bbox-only visualized images.

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
DEFAULT_MODEL_NAME = "chatgpt-image-latest"

# API configuration
API_URL = "https://vip.yi-zhan.top/v1/images/edits"


def encode_image_to_data_url(image_path: Path) -> str:
    """Encode image file to data URL (base64) for OpenAI input_image."""
    suffix = image_path.suffix.lower()
    if suffix in [".png"]:
        mime = "image/png"
    else:
        mime = "image/jpeg"

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def create_white_mask(image_path: Path) -> Path:
    """
    Create a white mask image (same size as input) for OpenAI images.edit().
    The mask marks the entire image as editable region.
    
    Returns:
        Path to temporary mask file
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")
    
    h, w = img.shape[:2]
    white_mask = np.ones((h, w), dtype=np.uint8) * 255
    
    # Save to temporary file
    mask_path = image_path.parent / f"{image_path.stem}_temp_mask.png"
    cv2.imwrite(str(mask_path), white_mask)
    return mask_path


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
        f"Your task is to generate a **binary segmentation mask** for **{primary_class} only**.\n\n"
        f"**Definition of {primary_class}:**\n"
        f"{class_def}\n\n"
        f"Requirements:\n"
        f"1. The mask must have **exactly the same resolution and crop** as the input image. Do not change the camera viewpoint, scaling, or add any borders.\n"
        f"2. In the mask image:\n"
        f"   - Defect pixels (belonging to **{primary_class}**): use 255 (white)\n"
        f"   - Background and all other pixels (non-{primary_class} areas): use 0 (black)\n"
        f"3. Focus on the areas within the bounding boxes labeled with \"{primary_class}\" in the image.\n\n"
        f"Output a single PNG image that is a binary segmentation mask for **{primary_class}** with the same dimensions as the input image."
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


def get_size_string(img_w: int, img_h: int) -> str:
    """
    Convert image dimensions to OpenAI size format.
    OpenAI supports: "256x256", "512x512", "1024x1024"
    We'll use the closest supported size, then resize to match original.
    """
    # Find closest supported size
    supported_sizes = [256, 512, 1024]
    max_dim = max(img_w, img_h)
    
    # Find the smallest supported size that is >= max_dim, or use largest if max_dim > 1024
    target_size = 1024
    for size in supported_sizes:
        if size >= max_dim:
            target_size = size
            break
    
    return f"{target_size}x{target_size}"


def call_image_generation_for_mask(api_key: str, api_url: str, model_name: str, image_path: Path,
                                    primary_class: str, img_h: int, img_w: int) -> Optional[np.ndarray]:
    """
    Call image edit API via requests to generate a binary mask for a specific defect type.
    
    Args:
        api_key: API key for authentication
        api_url: API endpoint URL
        model_name: Model name for image generation
        image_path: Path to input image file
        primary_class: Defect type ("Crack", "Material_loss", "Stain", "External Fixings")
        img_h: Original image height
        img_w: Original image width
    
    Returns:
        Binary mask (H, W) as uint8 array (0 or 255), or None on failure
    """
    prompt = build_prompt_for_mask_generation(primary_class)
    
    # Create a white mask (marks entire image as editable)
    mask_path = None
    try:
        mask_path = create_white_mask(image_path)
        size_str = get_size_string(img_w, img_h)
        
        # Prepare headers
        headers = {
            "Authorization": f"Bearer {api_key}"
        }
        
        # Prepare form-data payload with files (using context manager for proper file handling)
        image_mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
        with open(image_path, "rb") as img_file, open(mask_path, "rb") as mask_file:
            files = {
                "image": (image_path.name, img_file, image_mime),
                "mask": (mask_path.name, mask_file, "image/png")
            }
            
            data = {
                "prompt": prompt,
                "size": size_str,
                "model": model_name
            }
            
            # Make API request
            response = requests.post(api_url, headers=headers, files=files, data=data)
        
        # Check response status
        if response.status_code != 200:
            print(f"  Warning: API request failed with status {response.status_code}: {response.text}")
            return None
        
        # Parse response JSON
        try:
            response_data = response.json()
        except Exception as e:
            print(f"  Warning: Failed to parse JSON response: {e}")
            print(f"  Response text: {response.text[:500]}")
            return None
        
        # Get generated image URL
        # Expected format: {"data": [{"url": "...", "b64_json": null}]}
        img_url = None
        
        try:
            # Check for "data" field
            if "data" in response_data:
                data_list = response_data["data"]
                if isinstance(data_list, list) and len(data_list) > 0:
                    first_item = data_list[0]
                    if isinstance(first_item, dict):
                        # Try "url" field
                        if "url" in first_item and first_item["url"]:
                            img_url = first_item["url"]
                        # Try "b64_json" as fallback (base64 encoded image)
                        elif "b64_json" in first_item and first_item["b64_json"]:
                            # Decode base64 image
                            img_data = base64.b64decode(first_item["b64_json"])
                            img_array = np.frombuffer(img_data, np.uint8)
                            generated_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                            if generated_img is not None:
                                # Skip URL download, go directly to resize
                                gen_h, gen_w = generated_img.shape[:2]
                                if gen_h != img_h or gen_w != img_w:
                                    generated_img = cv2.resize(generated_img, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                                if len(generated_img.shape) == 3:
                                    gray = cv2.cvtColor(generated_img, cv2.COLOR_BGR2GRAY)
                                else:
                                    gray = generated_img
                                _, binary_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
                                return binary_mask
                else:
                    print(f"  Warning: 'data' field is empty or not a list for {primary_class}")
            else:
                print(f"  Warning: No 'data' field in response for {primary_class}")
                print(f"  Response keys: {list(response_data.keys())}")
        except Exception as e:
            print(f"  Warning: Error parsing response structure: {e}")
            print(f"  Response data: {str(response_data)[:500]}")
            import traceback
            traceback.print_exc()
        
        if not img_url:
            print(f"  Warning: No URL found in response for {primary_class}")
            print(f"  Response structure: {type(response_data)}")
            if isinstance(response_data, dict):
                print(f"  Response keys: {list(response_data.keys())}")
                if "data" in response_data:
                    print(f"  Data type: {type(response_data['data'])}")
                    if isinstance(response_data["data"], list) and len(response_data["data"]) > 0:
                        print(f"  First data item: {response_data['data'][0]}")
            return None
        
        # Download image from URL
        img_response = requests.get(img_url)
        img_array = np.frombuffer(img_response.content, np.uint8)
        generated_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if generated_img is None:
            print(f"  Warning: Failed to decode generated image for {primary_class}")
            return None
        
        # Resize to match original image size
        gen_h, gen_w = generated_img.shape[:2]
        if gen_h != img_h or gen_w != img_w:
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
    finally:
        # Clean up temp mask file
        if mask_path and mask_path.exists():
            mask_path.unlink()


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
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY is not set in environment.")
        return

    model_name = os.environ.get("VLM_MODEL_NAME", DEFAULT_MODEL_NAME)
    api_url = os.environ.get("API_URL", API_URL)
    output_root = RESULTS_DIR / f"{model_name}_masks"
    output_root.mkdir(parents=True, exist_ok=True)
    bbox_vis_dir = Path("defect_bench/results/visualization/images")

    print("Extracting defect masks using image generation API...")
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
                    
                    # Call image generation API
                    binary_mask = call_image_generation_for_mask(
                        api_key, api_url, model_name, vis_image_path, primary_class, img_h, img_w
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

