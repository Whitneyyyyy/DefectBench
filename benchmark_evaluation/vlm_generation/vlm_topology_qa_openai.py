#!/usr/bin/env python3
"""
Use a VLM (OpenAI-compatible API) to generate topology QA JSON files for visualized images in data_sample/images.

This script is similar to vlm_topology_qa.py but uses OpenAI-compatible API format.

For each visualized image (with numbered bounding boxes), we:
  - First call a VLM prompt to identify all defects with their numbered bounding boxes
  - Then call a second prompt to analyze spatial relationships between defects

The final topology QA is saved under:

    defect_bench/results/{MODEL_NAME}/{class_dir}/{image_stem}_topology_qa.json

Question: What are the relationships between the defects in the image?
Answer format: "[1#primary_class, relation, 2#primary_class]; [3#primary_class, relation, 4#primary_class]"

Relations:
- inclusion: defect1 is completely within defect2's boundaries
- overlapping: two different types of defects partially overlap in space
- adjacency: two defects' boundaries are in contact but do not overlap
- disjoint: defects have no spatial intersection or contact
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


def build_prompt_for_defects() -> str:
    """
    Build the first prompt to identify all defects with their numbered bounding boxes.
    The image should have numbered bounding boxes visible.
    """
    prompt = (
        "You will see an image with numbered bounding boxes and their labels.\n"
        "Each bounding box has a label in the format: \"{number}#{defect_type}\"\n"
        "For example: \"1#Crack\", \"2#Stain\", \"3#Material_loss\", \"4#External Fixings\"\n\n"
        "Your task is to identify all defects and their corresponding box numbers by reading the labels shown in the image.\n\n"
        "Use EXACTLY these defect type names (read from the image labels):\n"
        "- Crack\n"
        "- Material_loss\n"
        "- Stain\n"
        "- External Fixings\n\n"
        "For each numbered bounding box in the image, read its label (format: number#type) and extract the number and defect type.\n"
        "Return your answer as a JSON object with a list of defects:\n"
        "  {\n"
        "    \"defects\": [\n"
        "      {\"number\": 1, \"type\": \"Crack\"},\n"
        "      {\"number\": 2, \"type\": \"Material_loss\"}\n"
        "    ]\n"
        "  }\n"
        "Do NOT add any extra commentary, explanations, or markdown outside of this JSON.\n"
        "The JSON must be valid and parseable by Python json.loads.\n"
    )
    return prompt


def build_prompt_for_relationships(defects_list: str) -> str:
    """
    Build the second prompt to analyze spatial relationships between defects.
    """
    prompt = (
        f"Based on your previous analysis, you identified these defects: {defects_list}\n\n"
        "Now, analyze the spatial relationships between all pairs of defects in the image.\n\n"
        "Relationship types:\n"
        "- inclusion: one defect is completely within another defect's boundaries\n"
        "- overlapping: two different types of defects partially overlap in space\n"
        "- adjacency: two defects' boundaries are in contact or very close (within ~10 pixels) but do not overlap\n"
        "- disjoint: defects have no spatial intersection or contact\n\n"
        "For each pair of defects (i, j) where i < j, determine their relationship.\n"
        "Return your answer as a JSON object with a list of relationships:\n"
        "  {\n"
        "    \"relationships\": [\n"
        "      [\"1#Crack\", \"adjacency\", \"2#Material_loss\"],\n"
        "      [\"1#Crack\", \"overlapping\", \"3#Stain\"]\n"
        "    ]\n"
        "  }\n"
        "Format: [\"{number}#{type}\", \"{relation}\", \"{number}#{type}\"]\n"
        "Use EXACTLY these relation names: inclusion, overlapping, adjacency, disjoint\n"
        "Do NOT add any extra commentary, explanations, or markdown outside of this JSON.\n"
        "The JSON must be valid and parseable by Python json.loads.\n"
    )
    return prompt


def build_prompt_for_topology_complete() -> str:
    """
    Build a single prompt that combines defect identification and relationship analysis.
    This is more efficient than two separate calls.
    """
    prompt = (
        "You will see an image with numbered bounding boxes and their labels.\n"
        "Each bounding box has a label in the format: \"{number}#{defect_type}\"\n"
        "For example: \"1#Crack\", \"2#Stain\", \"3#Material_loss\", \"4#External Fixings\"\n\n"
        "Your task has two parts:\n\n"
        "Part 1: Identify all defects and their corresponding box numbers by reading the labels shown in the image.\n"
        "Use EXACTLY these defect type names (read from the image labels):\n"
        "- Crack\n"
        "- Material_loss\n"
        "- Stain\n"
        "- External Fixings\n\n"
        "Part 2: Analyze the spatial relationships between all pairs of defects in the image.\n"
        "Relationship types:\n"
        "- inclusion: one defect is completely within another defect's boundaries\n"
        "- overlapping: two different types of defects partially overlap in space\n"
        "- adjacency: two defects' boundaries are in contact or very close (within ~10 pixels) but do not overlap\n"
        "- disjoint: defects have no spatial intersection or contact\n\n"
        "For each pair of defects (i, j) where i < j, determine their relationship.\n\n"
        "Return your answer as a JSON object with both defects and relationships:\n"
        "  {\n"
        "    \"defects\": [\n"
        "      {\"number\": 1, \"type\": \"Crack\"},\n"
        "      {\"number\": 2, \"type\": \"Material_loss\"}\n"
        "    ],\n"
        "    \"relationships\": [\n"
        "      [\"1#Crack\", \"adjacency\", \"2#Material_loss\"],\n"
        "      [\"1#Crack\", \"overlapping\", \"3#Stain\"]\n"
        "    ]\n"
        "  }\n"
        "Format for relationships: [\"{number}#{type}\", \"{relation}\", \"{number}#{type}\"]\n"
        "Use EXACTLY these relation names: inclusion, overlapping, adjacency, disjoint\n"
        "If there is only one defect or no defects, return an empty relationships list.\n"
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


def call_vlm_for_defects(client: OpenAI, model_name: str, image_data_url: str):
    """
    Call VLM to identify defects with numbered bounding boxes (first round).
    Returns (messages, defects_list, reasoning_text) where defects_list is a list of {number, type} dicts.
    """
    prompt = build_prompt_for_defects()

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

    message = response.choices[0].message
    # Optional deep reasoning content (for reasoning models)
    reasoning = getattr(message, "reasoning_content", None)
    if reasoning:
        # Print reasoning chain for debugging/inspection
        print("Defect reasoning_content:")
        print(reasoning)

    text = message.content
    if not text:
        print("Warning: empty response text from model")
        return None, None, None

    # Try to extract JSON object from text (model may add extra commentary)
    json_text = extract_json_from_text(text)
    if not json_text:
        print(f"Warning: failed to extract JSON from model output")
        print("Raw model output:")
        print(text)
        return None, None, None

    try:
        data = json.loads(json_text)
    except Exception as e:
        print(f"Warning: failed to parse extracted JSON: {e}")
        print("Raw model output:")
        print(text)
        print("Extracted JSON text:")
        print(json_text)
        return None, None, None

    defects = data.get("defects", [])
    if not isinstance(defects, list):
        print(f"Warning: defects field is not a list")
        return None, None, None

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

    return messages, defects, reasoning


def call_vlm_for_topology_complete(client: OpenAI, model_name: str, image_data_url: str):
    """
    Call VLM in a single round to both identify defects and analyze relationships.
    This is more efficient than two separate calls.
    
    Returns:
        (relationships_string, reasoning_text) where relationships_string is like
        "[1#Crack, adjacency, 2#Material_loss]; [1#Crack, overlapping, 3#Stain]",
        or (None, None) on failure.
    """
    prompt = build_prompt_for_topology_complete()

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

    message = response.choices[0].message
    # Optional deep reasoning content (for reasoning models)
    reasoning = getattr(message, "reasoning_content", None)
    if reasoning:
        print("Topology reasoning_content:")
        print(reasoning)

    text = message.content
    if not text:
        print("Warning: empty response text from model")
        return None, None

    # Try to extract JSON object from text
    json_text = extract_json_from_text(text)
    if not json_text:
        print(f"Warning: failed to extract JSON from model output")
        print("Raw model output:")
        print(text)
        return None, reasoning

    try:
        data = json.loads(json_text)
    except Exception as e:
        print(f"Warning: failed to parse extracted JSON: {e}")
        print("Raw model output:")
        print(text)
        print("Extracted JSON text:")
        print(json_text)
        return None, reasoning

    defects = data.get("defects", [])
    relationships = data.get("relationships", [])

    if not isinstance(defects, list):
        print(f"Warning: defects field is not a list")
        return None, reasoning
    if not isinstance(relationships, list):
        print(f"Warning: relationships field is not a list")
        return None, reasoning

    if len(defects) <= 1:
        # Only one or no defects, no relationships
        return "Only one defect." if len(defects) == 1 else "No defects found.", reasoning

    if not relationships:
        return "No relationships detected between defects.", reasoning

    # Format relationships as answer string
    # Format: "[1#Crack, adjacency, 2#Material_loss]; [1#Crack, overlapping, 3#Stain]"
    answer_parts = []
    for rel in relationships:
        if isinstance(rel, list) and len(rel) == 3:
            answer_parts.append(f"[{rel[0]}, {rel[1]}, {rel[2]}]")
    
    if not answer_parts:
        return "No relationships detected between defects.", reasoning
    
    return "; ".join(answer_parts), reasoning


def call_vlm_for_relationships(client: OpenAI, model_name: str,
                                messages: list, defects: list) -> Optional[str]:
    """
    Call VLM in the second round to analyze spatial relationships between defects.
    Uses messages list to maintain conversation context.

    Returns:
        (relationships_string, reasoning_text) where relationships_string is like
        "[1#Crack, adjacency, 2#Material_loss]; [1#Crack, overlapping, 3#Stain]",
        or (None, None) on failure.
    """
    # Format defects list for prompt
    defects_str = ", ".join([f"{d.get('number')}#{d.get('type')}" for d in defects])
    
    prompt = build_prompt_for_relationships(defects_str)

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

    # Extract raw text content & optional reasoning
    message = response.choices[0].message
    reasoning = getattr(message, "reasoning_content", None)
    if reasoning:
        print("Relationship reasoning_content:")
        print(reasoning)

    rel_content = message.content
    if not rel_content:
        print(f"Warning: empty relationship response text")
        return None, None

    # Try to extract JSON object from text
    json_text = extract_json_from_text(rel_content)
    if not json_text:
        print(f"Warning: failed to extract JSON from relationship response")
        print("Raw relationship output:")
        print(rel_content)
        return None, reasoning

    try:
        data = json.loads(json_text)
    except Exception as e:
        print(f"Warning: failed to parse relationship JSON: {e}")
        print("Raw relationship output:")
        print(rel_content)
        print("Extracted JSON text:")
        print(json_text)
        return None, reasoning

    relationships = data.get("relationships", [])
    if not isinstance(relationships, list):
        print(f"Warning: relationships field is not a list")
        return None, reasoning

    if not relationships:
        return "No relationships detected between defects.", reasoning

    # Format relationships as answer string
    # Format: "[1#Crack, adjacency, 2#Material_loss]; [1#Crack, overlapping, 3#Stain]"
    answer_parts = []
    for rel in relationships:
        if isinstance(rel, list) and len(rel) == 3:
            answer_parts.append(f"[{rel[0]}, {rel[1]}, {rel[2]}]")
    
    if not answer_parts:
        return "No relationships detected between defects.", reasoning
    
    return "; ".join(answer_parts), reasoning


def build_topology_qa_json(image_name: str, relationships: str, reasoning: Optional[str] = None) -> Dict:
    """
    Build topology QA JSON structure matching topology_qa_gt.py, with optional reasoning.
    """
    qa_data = {
        "image_path": image_name,
        "questions": [
            {
                "question": "What are the relationships between the defects in the image?",
                "answer": relationships,
            },
        ],
    }
    # Append model's chain-of-thought reasoning (if available) for offline analysis
    if reasoning:
        qa_data["reasoning"] = reasoning
    return qa_data


def main():
    """
    Main entry: iterate over data_sample/images, call VLM, and save topology QA JSONs under results/{MODEL_NAME}/.
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
    visualization_dir = RESULTS_DIR / "ground_truth" / "Visualization"

    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )

    print("Generating VLM topology QA predictions (OpenAI format)...")
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

        # Save topology QA files to {MODEL_NAME}/{class_dir_name}/
        qa_output_dir = output_root / class_dir_name
        qa_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing {class_dir_name}...")

        # Iterate over bbox JSONs in class directory
        json_files = [f for f in LABELS_DIR.glob("*.json")
                      if not f.name.endswith("_qa.json") and not f.name.endswith("_topology_qa.json")]

        for json_file in json_files:
            image_stem = json_file.stem
            image_name = None
            image_path = None

            # Look for visualized image (with numbered bounding boxes)
            vis_jpg = visualization_dir / class_dir_name / f"{image_stem}_visualized.jpg"
            vis_png = visualization_dir / class_dir_name / f"{image_stem}_visualized.png"
            if vis_jpg.exists():
                image_path = vis_jpg
                image_name = vis_jpg.name.replace("_visualized.jpg", ".png")  # Use original name in output
            elif vis_png.exists():
                image_path = vis_png
                image_name = vis_png.name.replace("_visualized.png", ".png")
            else:
                print(f"  Warning: No visualized image found for {json_file.name}, skipping...")
                total_errors += 1
                continue

            # Output file path (topology_qa.json)
            qa_output_path = qa_output_dir / f"{image_stem}_topology_qa.json"
            if qa_output_path.exists():
                # Skip if already generated
                total_processed += 1
                continue

            try:
                # Single round: identify defects and analyze relationships in one call
                image_data_url = encode_image_to_data_url(image_path)
                relationships, reasoning = call_vlm_for_topology_complete(client, model_name, image_data_url)
                if not relationships:
                    print(f"  Warning: VLM topology analysis failed for {image_path}, using default.")
                    relationships = "No relationships detected between defects."
                    reasoning = None

                qa_data = build_topology_qa_json(image_name, relationships, reasoning)

                with open(qa_output_path, "w", encoding="utf-8") as f:
                    json.dump(qa_data, f, indent=2, ensure_ascii=False)

                total_processed += 1
                print(f"  Saved topology QA for {image_name} -> {qa_output_path}")

            except Exception as e:
                print(f"  Error processing {image_path}: {e}")
                import traceback
                traceback.print_exc()
                total_errors += 1

    print("\n" + "=" * 60)
    print("VLM topology QA generation finished.")
    print(f"  Processed (including skipped-existing): {total_processed}")
    print(f"  Errors: {total_errors}")
    print(f"  Output directory: {output_root}")
    print("=" * 60)


if __name__ == "__main__":
    main()

