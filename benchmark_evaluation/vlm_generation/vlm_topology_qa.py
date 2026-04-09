#!/usr/bin/env python3
"""
Use a VLM (Doubao via Ark) to generate topology QA JSON files for visualized images in data_sample/images.

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
from pathlib import Path
from typing import Dict, Optional

from volcenginesdkarkruntime import Ark


# Base directory
TEST100_DIR = Path("defect_bench/data_sample")
IMAGES_DIR = TEST100_DIR / "images"
LABELS_DIR = TEST100_DIR / "labels"
RESULTS_DIR = Path("defect_bench/results")

# Class subdirectories (same as generate_qa_ground_truth.py)
CLASS_DIRS = ["images"]

# Note: We use primary_class names directly without mapping

# Model configuration
DEFAULT_MODEL_NAME = "doubao-seed-1-8-251228"


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


def extract_text_from_response(response) -> Optional[str]:
    """
    Extract main text content from Ark response (same pattern as building backend).
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
        # Doubao returns reasoning as a separate output item with type == "reasoning"
        # and a summary list, same pattern as used in building/ui/backend/main.py.
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


def call_vlm_for_defects(client: Ark, model_name: str, image_data_url: str):
    """
    Call Doubao VLM to identify defects with numbered bounding boxes (first round).
    Returns (response, defects_list, thinking_text) where defects_list is a list of
    {number, type} dicts and thinking_text is the model's chain-of-thought if available.
    """
    prompt = build_prompt_for_defects()

    response = client.responses.create(
        model=model_name,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": image_data_url,
                    },
                    {
                        "type": "input_text",
                        "text": prompt,
                    },
                ],
            }
        ],
        extra_body={
            # Enable Doubao's thinking/chain-of-thought
            "thinking": {"type": "enabled"}
        }
    )

    text = extract_text_from_response(response)
    thinking = extract_thinking_from_response(response)
    if thinking:
        print("Defect thinking:")
        print(thinking)
    if not text:
        print("Warning: empty response text from model")
        return None, None, None

    # Try to extract JSON object from text (model may add extra commentary)
    json_text = extract_json_from_text(text)
    if not json_text:
        print(f"Warning: failed to extract JSON from model output")
        print("Raw model output:")
        print(text)
        return None, None, thinking

    try:
        data = json.loads(json_text)
    except Exception as e:
        print(f"Warning: failed to parse extracted JSON: {e}")
        print("Raw model output:")
        print(text)
        print("Extracted JSON text:")
        print(json_text)
        return None, None, thinking

    defects = data.get("defects", [])
    if not isinstance(defects, list):
        print(f"Warning: defects field is not a list")
        return None, None, thinking

    return response, defects, thinking


def call_vlm_for_topology_complete(client: Ark, model_name: str, image_data_url: str):
    """
    Call VLM in a single round to both identify defects and analyze relationships.
    This is more efficient than two separate calls.
    
    Returns:
        (relationships_string, thinking_text) where relationships_string is like
        "[1#Crack, adjacency, 2#Material_loss]; [1#Crack, overlapping, 3#Stain]",
        or (None, None) on failure.
    """
    prompt = build_prompt_for_topology_complete()

    response = client.responses.create(
        model=model_name,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": image_data_url,
                    },
                    {
                        "type": "input_text",
                        "text": prompt,
                    },
                ],
            }
        ],
        extra_body={
            # Enable Doubao's thinking/chain-of-thought
            "thinking": {"type": "enabled"}
        }
    )

    text = extract_text_from_response(response)
    thinking = extract_thinking_from_response(response)
    if thinking:
        print("Topology thinking:")
        print(thinking)
    if not text:
        print("Warning: empty response text from model")
        return None, None

    # Try to extract JSON object from text
    json_text = extract_json_from_text(text)
    if not json_text:
        print(f"Warning: failed to extract JSON from model output")
        print("Raw model output:")
        print(text)
        return None, thinking

    try:
        data = json.loads(json_text)
    except Exception as e:
        print(f"Warning: failed to parse extracted JSON: {e}")
        print("Raw model output:")
        print(text)
        print("Extracted JSON text:")
        print(json_text)
        return None, thinking

    defects = data.get("defects", [])
    relationships = data.get("relationships", [])

    if not isinstance(defects, list):
        print(f"Warning: defects field is not a list")
        return None, thinking
    if not isinstance(relationships, list):
        print(f"Warning: relationships field is not a list")
        return None, thinking

    if len(defects) <= 1:
        # Only one or no defects, no relationships
        return "Only one defect." if len(defects) == 1 else "No defects found.", thinking

    if not relationships:
        return "No relationships detected between defects.", thinking

    # Format relationships as answer string
    # Format: "[1#Crack, adjacency, 2#Material_loss]; [1#Crack, overlapping, 3#Stain]"
    answer_parts = []
    for rel in relationships:
        if isinstance(rel, list) and len(rel) == 3:
            answer_parts.append(f"[{rel[0]}, {rel[1]}, {rel[2]}]")
    
    if not answer_parts:
        return "No relationships detected between defects.", thinking
    
    return "; ".join(answer_parts), thinking


def call_vlm_for_relationships(client: Ark, model_name: str,
                                previous_response_id: str, defects: list) -> Optional[str]:
    """
    Call VLM in the second round to analyze spatial relationships between defects.
    Uses previous_response_id to maintain conversation context.

    Returns:
        (relationships_string, thinking_text) where relationships_string is like
        "[1#Crack, adjacency, 2#Material_loss]; [1#Crack, overlapping, 3#Stain]",
        or (None, None) on failure.
    """
    # Format defects list for prompt
    defects_str = ", ".join([f"{d.get('number')}#{d.get('type')}" for d in defects])
    
    prompt = build_prompt_for_relationships(defects_str)

    # Use responses.create with previous_response_id for multi-turn conversation
    response = client.responses.create(
        model=model_name,
        previous_response_id=previous_response_id,
        input=prompt,
        extra_body={
            # Enable Doubao's thinking/chain-of-thought
            "thinking": {"type": "enabled"}
        }
    )

    # Extract raw text content
    rel_content = extract_text_from_response(response)
    thinking = extract_thinking_from_response(response)
    if thinking:
        print("Relationship thinking:")
        print(thinking)
    if not rel_content:
        print(f"Warning: empty relationship response text")
        return None, thinking

    # Try to extract JSON object from text
    json_text = extract_json_from_text(rel_content)
    if not json_text:
        print(f"Warning: failed to extract JSON from relationship response")
        print("Raw relationship output:")
        print(rel_content)
        return None, thinking

    try:
        data = json.loads(json_text)
    except Exception as e:
        print(f"Warning: failed to parse relationship JSON: {e}")
        print("Raw relationship output:")
        print(rel_content)
        print("Extracted JSON text:")
        print(json_text)
        return None, thinking

    relationships = data.get("relationships", [])
    if not isinstance(relationships, list):
        print(f"Warning: relationships field is not a list")
        return None, thinking

    if not relationships:
        return "No relationships detected between defects.", thinking

    # Format relationships as answer string
    # Format: "[1#Crack, adjacency, 2#Material_loss]; [1#Crack, overlapping, 3#Stain]"
    answer_parts = []
    for rel in relationships:
        if isinstance(rel, list) and len(rel) == 3:
            answer_parts.append(f"[{rel[0]}, {rel[1]}, {rel[2]}]")
    
    if not answer_parts:
        return "No relationships detected between defects.", thinking
    
    return "; ".join(answer_parts), thinking


def build_topology_qa_json(image_name: str, relationships: str, thinking: Optional[str] = None) -> Dict:
    """
    Build topology QA JSON structure matching topology_qa_gt.py, with optional thinking field.
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
    if thinking:
        qa_data["thinking"] = thinking
    return qa_data


def main():
    """
    Main entry: iterate over data_sample/images, call VLM, and save QA JSONs under results/{MODEL_NAME}/.
    """
    api_key = os.environ.get("ARK_API_KEY")
    if not api_key:
        print("ERROR: ARK_API_KEY is not set in environment.")
        return

    model_name = os.environ.get("VLM_MODEL_NAME", DEFAULT_MODEL_NAME)
    output_root = RESULTS_DIR / model_name
    output_root.mkdir(parents=True, exist_ok=True)
    visualization_dir = RESULTS_DIR / "ground_truth" / "Visualization"

    client = Ark(
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        api_key=api_key,
    )

    print("Generating VLM topology QA predictions...")
    print("=" * 60)
    print(f"Model name: {model_name}")
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
                relationships, thinking = call_vlm_for_topology_complete(client, model_name, image_data_url)
                if not relationships:
                    print(f"  Warning: VLM topology analysis failed for {image_path}, using default.")
                    relationships = "No relationships detected between defects."
                    thinking = None

                qa_data = build_topology_qa_json(image_name, relationships, thinking)

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


