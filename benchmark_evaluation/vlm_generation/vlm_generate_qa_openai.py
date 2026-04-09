#!/usr/bin/env python3
"""
Use a VLM (OpenAI-compatible API) to generate QA JSON files for images in data_sample/images.

This script is similar to vlm_generate_qa.py but uses OpenAI-compatible API format.

For each image, we:
  - First call a QA-style VLM prompt to answer 2 questions (defect types, counts)
  - Then call a separate grounding-style VLM prompt to get bounding boxes
    using <bbox>x_min y_min x_max y_max</bbox> tags with normalized [0, 999] coords,
    and convert them locally to (x, y, width, height) in image space.

The final answers (3 questions) are saved under:

    defect_bench/results/{MODEL_NAME}/{class_dir}/{image_stem}_qa.json

High-level QA questions (first call):
1. What defects are in the image?
2. How many instances of each defect type?

Bounding-box question (second call):
3. What are the bounding box coordinates of each defect instance (x, y, width, height)?
"""

import os
import json
import base64
import io
from pathlib import Path
from typing import Dict, Optional

from openai import OpenAI


# Base directory
TEST100_DIR = Path("defect_bench/data_sample")
IMAGES_DIR = TEST100_DIR / "images"
LABELS_DIR = TEST100_DIR / "labels"
RESULTS_DIR = Path("defect_bench/results")

# Class subdirectories (same as generate_qa_ground_truth.py)
CLASS_DIRS = ["images"]

# Note: We use primary_class names directly without mapping

# Model configuration
DEFAULT_MODEL_NAME = "gpt-4o"  # Update with actual model name

# BBox tag configuration for grounding-style call (normalized 0-999 coords)
BBOX_TAG_START = "<bbox>"
BBOX_TAG_END = "</bbox>"

# Maximum base64-encoded image size in bytes (5MB limit, use 4.5MB to leave margin)
# Note: base64 encoding increases size by ~33%, so we need to check the encoded size
# For a 5MB limit, the raw image should be ~3.75MB (5MB / 1.33)
MAX_BASE64_SIZE_BYTES = 4_500_000
MAX_RAW_IMAGE_SIZE_BYTES = int(4_500_000 / 1.33)  # ~3.38MB to account for base64 overhead


def encode_image_to_data_url(image_path: Path) -> str:
    """
    Encode image file to data URL (base64) for OpenAI API.
    Automatically compresses images that exceed MAX_IMAGE_SIZE_BYTES.
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError(
            "PIL/Pillow is required for image compression. "
            "Please install it with: pip install Pillow"
        )
    
    # Read original image
    img = Image.open(image_path)
    original_format = img.format or "JPEG"
    
    # Determine output format and MIME type
    suffix = image_path.suffix.lower()
    if suffix in [".png"]:
        output_format = "PNG"
        mime = "image/png"
    else:
        output_format = "JPEG"
        mime = "image/jpeg"
    
    # Convert RGBA to RGB for JPEG
    if output_format == "JPEG" and img.mode in ("RGBA", "LA", "P"):
        # Create white background for transparency
        background = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "P":
            img = img.convert("RGBA")
        background.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
        img = background
    elif img.mode != "RGB":
        img = img.convert("RGB")
    
    # Try to encode without compression first
    buffer = io.BytesIO()
    if output_format == "PNG":
        img.save(buffer, format="PNG", optimize=True)
    else:
        img.save(buffer, format="JPEG", quality=95, optimize=True)
    
    image_bytes = buffer.getvalue()
    raw_size = len(image_bytes)
    
    # Check base64-encoded size (base64 increases size by ~33%)
    b64_encoded = base64.b64encode(image_bytes)
    base64_size = len(b64_encoded)
    
    # If base64-encoded size is too large, compress progressively
    if base64_size > MAX_BASE64_SIZE_BYTES:
        print(f"  Compressing image {image_path.name} (raw size: {raw_size / 1024 / 1024:.2f} MB, base64 size: {base64_size / 1024 / 1024:.2f} MB)")
        
        # Strategy: progressive compression to minimize quality loss
        # 1. For PNG: convert to JPEG first (usually much smaller)
        # 2. Gradually reduce JPEG quality (from 90 to 70, step by 5)
        # 3. Only if still too large: slightly reduce resolution (max 20% reduction)
        # 4. Last resort: further reduce quality (down to 60 minimum)
        
        # Step 1: If PNG, try converting to JPEG first (much better compression)
        already_tested_quality_90 = False
        if output_format == "PNG":
            buffer = io.BytesIO()
            # Ensure RGB mode for JPEG
            if img.mode != "RGB":
                if img.mode in ("RGBA", "LA", "P"):
                    background = Image.new("RGB", img.size, (255, 255, 255))
                    if img.mode == "P":
                        img = img.convert("RGBA")
                    background.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
                    img = background
                else:
                    img = img.convert("RGB")
            img.save(buffer, format="JPEG", quality=90, optimize=True)
            image_bytes = buffer.getvalue()
            raw_size = len(image_bytes)
            b64_encoded = base64.b64encode(image_bytes)
            base64_size = len(b64_encoded)
            output_format = "JPEG"
            mime = "image/jpeg"
            already_tested_quality_90 = True
            if base64_size <= MAX_BASE64_SIZE_BYTES:
                print(f"    Converted PNG to JPEG (quality 90): raw {raw_size / 1024 / 1024:.2f} MB, base64 {base64_size / 1024 / 1024:.2f} MB")
            else:
                print(f"    Converted PNG to JPEG (quality 90): raw {raw_size / 1024 / 1024:.2f} MB, base64 {base64_size / 1024 / 1024:.2f} MB (still too large)")
        
        # Step 2: Gradually reduce JPEG quality (90 -> 85 -> 80 -> 75 -> 70)
        # If we already tested quality 90 in Step 1, start from 85
        quality = 95  # Track current quality (original was 95)
        if output_format == "JPEG" and base64_size > MAX_BASE64_SIZE_BYTES:
            quality_list = [85, 80, 75, 70] if already_tested_quality_90 else [90, 85, 80, 75, 70]
            for test_quality in quality_list:
                if base64_size <= MAX_BASE64_SIZE_BYTES:
                    break
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=test_quality, optimize=True)
                image_bytes = buffer.getvalue()
                raw_size = len(image_bytes)
                b64_encoded = base64.b64encode(image_bytes)
                base64_size = len(b64_encoded)
                quality = test_quality
                # Always print the test result, whether it meets the requirement or not
                if base64_size <= MAX_BASE64_SIZE_BYTES:
                    print(f"    Reduced quality to {quality}: raw {raw_size / 1024 / 1024:.2f} MB, base64 {base64_size / 1024 / 1024:.2f} MB")
                    break
                else:
                    print(f"    Tested quality {quality}: raw {raw_size / 1024 / 1024:.2f} MB, base64 {base64_size / 1024 / 1024:.2f} MB (still too large)")
        
        # Step 3: If still too large, slightly reduce resolution (max 20% reduction)
        scale_factor = 1.0
        resized_img = None
        if base64_size > MAX_BASE64_SIZE_BYTES:
            # Try reducing resolution in small steps (5% each time, max 20% total)
            for scale in [0.95, 0.90, 0.85, 0.80]:
                if base64_size <= MAX_BASE64_SIZE_BYTES:
                    break
                scale_factor = scale
                new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
                resized_img = img.resize(new_size, Image.Resampling.LANCZOS)
                buffer = io.BytesIO()
                if output_format == "PNG":
                    resized_img.save(buffer, format="PNG", optimize=True)
                else:
                    resized_img.save(buffer, format="JPEG", quality=quality, optimize=True)
                image_bytes = buffer.getvalue()
                raw_size = len(image_bytes)
                b64_encoded = base64.b64encode(image_bytes)
                base64_size = len(b64_encoded)
                if base64_size <= MAX_BASE64_SIZE_BYTES:
                    print(f"    Reduced resolution to {int(scale_factor*100)}%: raw {raw_size / 1024 / 1024:.2f} MB, base64 {base64_size / 1024 / 1024:.2f} MB")
                    break
        
        # Step 4: Last resort - further reduce quality (down to 60 minimum)
        if base64_size > MAX_BASE64_SIZE_BYTES and output_format == "JPEG":
            # Use resized image if we resized, otherwise original
            work_img = resized_img if resized_img is not None else img
            for test_quality in [65, 60]:
                if base64_size <= MAX_BASE64_SIZE_BYTES:
                    break
                buffer = io.BytesIO()
                work_img.save(buffer, format="JPEG", quality=test_quality, optimize=True)
                image_bytes = buffer.getvalue()
                raw_size = len(image_bytes)
                b64_encoded = base64.b64encode(image_bytes)
                base64_size = len(b64_encoded)
                quality = test_quality
                if base64_size <= MAX_BASE64_SIZE_BYTES:
                    print(f"    Further reduced quality to {quality}: raw {raw_size / 1024 / 1024:.2f} MB, base64 {base64_size / 1024 / 1024:.2f} MB")
                    break
        
        if base64_size > MAX_BASE64_SIZE_BYTES:
            print(f"  Warning: Image {image_path.name} base64-encoded size still too large after compression ({base64_size / 1024 / 1024:.2f} MB)")
        else:
            print(f"  Compressed to raw: {raw_size / 1024 / 1024:.2f} MB, base64: {base64_size / 1024 / 1024:.2f} MB")
    else:
        # Image is already small enough, use the already encoded base64
        pass
    
    b64 = b64_encoded.decode("utf-8")
    return f"data:{mime};base64,{b64}"


def build_prompt() -> str:
    """
    Build the text prompt for the VLM.

    We ask the model to analyze the image and return a strict JSON with 2 answers:
    1) defects list, 2) counts per defect type.
    """
    prompt = (
        "You are an expert in building defect analysis. You will see one image.\n"
        "Your task is to answer TWO questions about visible defects in this image.\n\n"
        "Only consider these four primary defect types, and use EXACTLY these English names in your answers:\n\n"
        "1. Crack: Any type of crack or fissure in the building surface, including:\n"
        "2. Material_loss: Loss of material from the building surface, including:\n"
        "   - peeling, spalling, flakes, peeling_paint, Abscission, Bulge\n"
        "3. Stain: Discoloration or staining on the building surface, including:\n"
        "   - algae, stain, biological_deteriorations, mold, water_seepage, Dampness, Efflorescence, Leakage, Corrosion, chemical_deteriorations\n"
        "4. External Fixings: External objects or human-made additions on the building surface, including:\n"
        "   - human_caused_damages (such as graffiti, vandalism, or other human-made marks)\n"
        "   - Vegetation (plants, moss, or other vegetation growing on the surface)\n\n"
        "The two questions are:\n"
        "1) What defects are in the image? (list defect types, separated by comma)\n"
        "2) How many instances of each defect type? (follow the same order as in Q1, give counts separated by comma)\n\n"
        "Return your answers STRICTLY as a single JSON object with exactly these keys:\n"
        "  {\n"
        "    \"answer1\": \"...\",  // Answer to question 1 (defect types)\n"
        "    \"answer2\": \"...\"   // Answer to question 2 (counts)\n"
        "  }\n"
        "Do NOT add any extra commentary, explanations, or markdown outside of this JSON.\n"
        "The JSON must be valid and parseable by Python json.loads.\n"
    )
    return prompt


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


def extract_json_array_from_text(text: str) -> Optional[str]:
    """
    Extract JSON array from text that may contain extra commentary.
    Finds the first complete JSON array (from first '[' to matching ']').
    """
    if not text:
        return None
    
    # Try direct parse first (most common case)
    try:
        json.loads(text)
        return text
    except:
        pass
    
    # Find first '[' and try to extract complete JSON array
    start_idx = text.find('[')
    if start_idx == -1:
        return None
    
    # Find matching closing ']'
    bracket_count = 0
    for i in range(start_idx, len(text)):
        if text[i] == '[':
            bracket_count += 1
        elif text[i] == ']':
            bracket_count -= 1
            if bracket_count == 0:
                # Found complete JSON array
                json_candidate = text[start_idx:i+1]
                try:
                    json.loads(json_candidate)
                    return json_candidate
                except:
                    pass
    
    return None


def call_vlm_for_image(client: OpenAI, model_name: str, image_data_url: str):
    """
    Call VLM for a single image (first round of conversation).
    Returns (response, answers_dict) where response contains message for multi-turn.
    """
    prompt = build_prompt()

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    )

    text = response.choices[0].message.content
    if not text:
        print("Warning: empty response text from model")
        return None, None

    # Try to extract JSON object from text (model may add extra commentary)
    json_text = extract_json_from_text(text)
    if not json_text:
        print(f"Warning: failed to extract JSON from model output")
        print("Raw model output:")
        print(text)
        return None, None

    try:
        data = json.loads(json_text)
    except Exception as e:
        print(f"Warning: failed to parse extracted JSON: {e}")
        print("Raw model output:")
        print(text)
        print("Extracted JSON text:")
        print(json_text)
        return None, None

    answers = {
        "answer1": str(data.get("answer1", "")).strip(),
        "answer2": str(data.get("answer2", "")).strip(),
    }
    # Store messages for multi-turn conversation
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_data_url
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        },
        {
            "role": "assistant",
            "content": text
        }
    ]
    return messages, answers


def call_vlm_for_bboxes(client: OpenAI, model_name: str, image_path: Path, 
                        messages: list, answer1: str, answer2: str) -> Optional[str]:
    """
    Call VLM in the second round of conversation to get bounding box coordinates.
    Uses messages list to maintain conversation context and references
    the first round's answers (defect types and counts).

    The model is asked to return ONLY a JSON array like:
      [
        {"category": "Crack", "bbox": "<bbox>x1 y1 x2 y2</bbox>"},
        {"category": "Material_loss", "bbox": "<bbox>x1 y1 x2 y2</bbox>"}
      ]

    We then:
      - parse this JSON
      - convert normalized [0, 999] coords to (x, y, width, height) in image space
      - group by category
      - build a string such as:
          "Crack: [x1, y1, w1, h1]; [x2, y2, w2, h2]; Material_loss: [x3, y3, w3, h3]"

    Returns:
        The formatted string above, or None on failure.
    """
    # Build prompt that references first round's answers
    prompt = (
        f"Based on your previous analysis, you identified these defects: {answer1}\n"
        f"And the counts are: {answer2}\n\n"
        "Now, please detect the bounding boxes for each defect instance you identified.\n"
        "For each detected object, return its category and bounding box.\n\n"
        "IMPORTANT: Bounding box coordinate format and requirements:\n"
        f"- The bounding box must be written as \"bbox\": \"{BBOX_TAG_START}x_min y_min x_max y_max{BBOX_TAG_END}\"\n"
        "- Coordinates are NORMALIZED to the range [0, 999], where:\n"
        "  * 0 represents the leftmost/topmost edge of the image\n"
        "  * 999 represents the rightmost/bottommost edge of the image\n"
        "- x_min, y_min are the coordinates of the TOP-LEFT corner of the bounding box\n"
        "- x_max, y_max are the coordinates of the BOTTOM-RIGHT corner of the bounding box\n"
        "- All coordinates must be integers in the range [0, 999]\n"
        "- The bounding box should cover the ENTIRE defect region, not just a small part of it\n"
        "  For example, if you see a long crack, the box should encompass the full length of the crack\n"
        "- Make sure x_max > x_min and y_max > y_min\n\n"
        "Use EXACTLY these category names: Crack, Material_loss, Stain, External Fixings\n"
        "Return ONLY a JSON array like:\n"
        f"[{{\"category\": \"Crack\", \"bbox\": \"{BBOX_TAG_START}x1 y1 x2 y2{BBOX_TAG_END}\"}}, "
        f"{{\"category\": \"Material_loss\", \"bbox\": \"{BBOX_TAG_START}x3 y3 x4 y4{BBOX_TAG_END}\"}}]"
    )

    # Add second round message to conversation
    messages.append({
        "role": "user",
        "content": prompt
    })

    # Use chat.completions.create with messages for multi-turn conversation
    response = client.chat.completions.create(
        model=model_name,
        messages=messages
    )

    bbox_content = response.choices[0].message.content
    if not bbox_content:
        print(f"Warning: empty bbox response text for {image_path}")
        return None

    # Try to extract JSON array from text (model may add extra commentary)
    json_text = extract_json_array_from_text(bbox_content)
    if not json_text:
        print(f"Warning: failed to extract JSON array from bbox response for {image_path}")
        print("Raw bbox output:")
        print(bbox_content)
        return None

    # According to official guidance, we don't strongly enforce JSON in the prompt,
    # but we still TRY to parse JSON array if the model follows the example format.
    try:
        items = json.loads(json_text)
    except Exception as e:
        print(f"Warning: failed to parse extracted bbox JSON for {image_path}: {e}")
        print("Raw bbox output:")
        print(bbox_content)
        print("Extracted JSON text:")
        print(json_text)
        return None

    if not isinstance(items, list):
        print(f"Warning: bbox JSON root is not a list for {image_path}")
        return None

    # Load image to get real width/height
    try:
        import cv2  # Local import to avoid hard dependency if not used
    except ImportError:
        print("Warning: cv2 not available, cannot scale bbox coordinates.")
        return None

    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Warning: failed to read image for bbox scaling: {image_path}")
        return None
    h, w = img.shape[:2]

    allowed_categories = {"Crack", "Material_loss", "Stain", "External Fixings"}
    grouped: Dict[str, list] = {}

    for obj in items:
        if not isinstance(obj, dict):
            continue
        category = str(obj.get("category", "")).strip()
        if category not in allowed_categories:
            continue

        bbox_field = str(obj.get("bbox", "")).strip()
        if not (bbox_field.startswith(BBOX_TAG_START) and bbox_field.endswith(BBOX_TAG_END)):
            print(f"Warning: bbox field missing tags for {image_path}: {bbox_field}")
            continue

        coords_str = bbox_field[len(BBOX_TAG_START) : -len(BBOX_TAG_END)].strip()
        try:
            parts = coords_str.split()
            if len(parts) != 4:
                raise ValueError(f"Expected 4 coords, got {len(parts)}")
            x_min_n, y_min_n, x_max_n, y_max_n = map(int, parts)
        except Exception as e:
            print(f"Warning: failed to parse bbox coords '{coords_str}' for {image_path}: {e}")
            continue

        # Map normalized [0,999] coords to real image coords
        x_min_real = int(x_min_n * w / 1000.0)
        y_min_real = int(y_min_n * h / 1000.0)
        x_max_real = int(x_max_n * w / 1000.0)
        y_max_real = int(y_max_n * h / 1000.0)

        x_real = max(0, min(x_min_real, w - 1))
        y_real = max(0, min(y_min_real, h - 1))
        box_w = max(1, min(x_max_real - x_min_real, w - x_real))
        box_h = max(1, min(y_max_real - y_min_real, h - y_real))

        grouped.setdefault(category, []).append(f"[{x_real}, {y_real}, {box_w}, {box_h}]")

    if not grouped:
        return None

    # Build answer string grouped by category, in a stable order
    category_order = ["Crack", "Material_loss", "Stain", "External Fixings"]
    parts_out = []
    for cat in category_order:
        boxes = grouped.get(cat)
        if not boxes:
            continue
        parts_out.append(f"{cat}: " + "; ".join(boxes))

    if not parts_out:
        return None

    return "; ".join(parts_out)


def build_qa_json(image_name: str, answers: Dict[str, str]) -> Dict:
    """
    Build QA JSON structure matching generate_qa_ground_truth.py.
    """
    answer1 = answers.get("answer1", "")
    answer2 = answers.get("answer2", "")
    # answer3 will be filled from separate bbox grounding call (bbox coordinates)
    answer3 = answers.get("answer3", "")

    qa_data = {
        "image_path": image_name,
        "questions": [
            {
                "question": "What defects are in the image?",
                "answer": answer1,
            },
            {
                "question": f"How many instances of each defect type ({answer1})?",
                "answer": answer2,
            },
            {
                "question": "What are the bounding box coordinates of each defect instance (x, y, width, height)?",
                "answer": answer3,
            },
        ],
    }
    return qa_data


def main():
    """
    Main entry: iterate over data_sample/images, call VLM, and save QA JSONs under results/{MODEL_NAME}/.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY is not set in environment.")
        return

    model_name = os.environ.get("VLM_MODEL_NAME", DEFAULT_MODEL_NAME)
    base_url = os.environ.get("OPENAI_BASE_URL", "https://vip.yi-zhan.top/v1")
    
    # Remove duplicate https:// if present
    if base_url.startswith("https://https://"):
        base_url = base_url.replace("https://https://", "https://", 1)
    elif not base_url.startswith("http"):
        base_url = f"https://{base_url}"
    
    output_root = RESULTS_DIR / model_name
    output_root.mkdir(parents=True, exist_ok=True)

    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )

    print("Generating VLM QA predictions (OpenAI format)...")
    print("=" * 60)
    print(f"Model name: {model_name}")
    print(f"Base URL: {base_url}")
    print(f"Output root: {output_root}")

    total_processed = 0
    total_errors = 0

    for class_dir_name in CLASS_DIRS:
        class_dir = IMAGES_DIR
        if not class_dir.exists():
            print(f"Warning: Directory {class_dir} does not exist, skipping...")
            continue

        output_class_dir = output_root / class_dir_name
        output_class_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing {class_dir_name}...")

        # Iterate over bbox JSONs in class directory
        json_files = [f for f in LABELS_DIR.glob("*.json")]

        for json_file in json_files:
            image_stem = json_file.stem
            image_name = None
            image_path = None

            # Look for original raw image in class_dir only
            for ext in [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]:
                candidate = IMAGES_DIR / f"{image_stem}{ext}"
                if candidate.exists():
                    image_path = candidate
                    image_name = candidate.name
                    break

            if image_path is None:
                print(f"  Warning: No image found for {json_file.name}, skipping...")
                total_errors += 1
                continue

            # Output file path
            qa_output_path = output_class_dir / f"{image_stem}_qa.json"
            if qa_output_path.exists():
                # Skip if already generated
                total_processed += 1
                continue

            try:
                # First round: high-level QA (answers 1-2)
                image_data_url = encode_image_to_data_url(image_path)
                messages, answers = call_vlm_for_image(client, model_name, image_data_url)
                if not answers or not messages:
                    print(f"  Warning: VLM QA (first round) failed for {image_path}, skipping...")
                    total_errors += 1
                    continue

                # Second round: multi-turn conversation to get bounding boxes based on first round's answers
                bbox_answer = call_vlm_for_bboxes(
                    client, model_name, image_path, 
                    messages=messages,
                    answer1=answers["answer1"],
                    answer2=answers["answer2"]
                )
                if bbox_answer:
                    answers["answer3"] = bbox_answer
                else:
                    print(f"  Warning: VLM bbox extraction (second round) failed for {image_path}, keeping empty bbox answer.")

                qa_data = build_qa_json(image_name, answers)

                with open(qa_output_path, "w", encoding="utf-8") as f:
                    json.dump(qa_data, f, indent=2, ensure_ascii=False)

                total_processed += 1
                print(f"  Saved VLM QA for {image_name} -> {qa_output_path}")

            except Exception as e:
                print(f"  Error processing {image_path}: {e}")
                import traceback
                traceback.print_exc()
                total_errors += 1

    print("\n" + "=" * 60)
    print("VLM QA generation finished.")
    print(f"  Processed (including skipped-existing): {total_processed}")
    print(f"  Errors: {total_errors}")
    print(f"  Output directory: {output_root}")
    print("=" * 60)


if __name__ == "__main__":
    main()

