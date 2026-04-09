# DefectBench — Dataset Documentation

Detailed documentation for the DefectBench dataset, including annotation format, data curation pipeline, source dataset mapping, and evaluation prompts.

## Table of Contents

- [Data Curation Pipeline](#data-curation-pipeline)
- [Defect Taxonomy](#defect-taxonomy)
- [Directory Structure](#directory-structure)
- [Annotation Format](#annotation-format)
- [Source Datasets](#source-datasets)
- [Benchmark Tasks & Evaluation Prompts](#benchmark-tasks--evaluation-prompts)
- [Evaluation Metrics](#evaluation-metrics)
- [Statistics](#statistics)

---

## Data Curation Pipeline

DefectBench is constructed through a **human-in-the-loop semi-automated annotation framework** consisting of two core modules:

### 1. Interactive Detection Module
- For datasets with pre-existing annotations: human-verified refinement to correct box drifts.
- For unlabeled raw images: an ensemble of SOTA detectors generates candidate proposals:
  - **YOLO12-M**, **YOLO11-M**, **Faster R-CNN**, **RT-DETR**
- All proposals are manually calibrated by domain experts.

### 2. Interactive Segmentation Module
- Leverages refined bounding boxes as visual prompts.
- Employs **SAM-3** for zero-shot mask generation.
- Domain-specific model zoo: **SegFormer (b0/b4)**, **UNet-VGG16**, **YOLOv8-crack-seg**, **SCSegamba**.
- Interactive tools (point-based prompting, brush-based refinement) for expert fine-grained correction.

---

![alt text](assets/fig.s1.jpg)

## Defect Taxonomy

DefectBench defines a standardized, two-level taxonomy:

| Primary Class | Sub-type | Instances | Description |
|---|---|---:|---|
| **Crack** | Linear crack | 2,042 | Single-path structural cracks |
| | Map cracking | 80 | Network/pattern cracking |
| **Material Loss** | Spalling | 906 | Concrete/plaster detachment |
| | Peeling | 51 | Surface layer separation |
| **Surface Stain** | Corrosion | 254 | Metal oxidation staining |
| | Rust stain | 21 | Rust-originated discoloration |
| | Leakage stain | 803 | Water infiltration marks |
| **External Fixings** | Vegetation growth | 221 | Biological colonization |
| | Graffiti | 30 | Human-made surface markings |
| | Surface contaminants | 119 | Dirt, deposits, foreign matter |
| | **Total** | | **4,527** |

---

### Directory Overview

```text
DefectBench/
├── annotation_toolkit/
│   ├── annotation.py
│   ├── backend/
│   │   ├── annotate_images_to_candidates.py
│   │   ├── detection_agent.py
│   │   ├── sam_logic.py
│   │   └── crack_service.py
│   ├── frontend/
│   ├── sample_pipeline/
│   │   ├── filter_and_classify_images.py
│   │   └── analyze_image_distributions.py
│   └── src/final_dataset/
├── benchmark_evaluation/
│   ├── vlm_generation/
│   │   ├── generate_visualization.py
│   │   ├── generate_qa_ground_truth.py
│   │   ├── topology_qa_gt.py
│   │   ├── vlm_generate_qa.py
│   │   ├── vlm_generate_qa_openai.py
│   │   ├── vlm_topology_qa.py
│   │   ├── vlm_topology_qa_openai.py
│   │   └── vlm_extract_mask*.py
│   └── vlm_evaluation/
│       ├── evaluate_vlm_qa.py
│       ├── evaluate_segmentation.py
│       ├── evaluate_q4_metrics.py
│       └── evaluate_bbox_metrics.py
├── preprocess_raw_dataset/
│   ├── unify_bbox_labels.py
│   └── unify_mask.py
├── data_sample/
│   ├── images/
│   ├── labels/
│   └── masks/
├── model_weights/
└── core/
```

## Annotation Format

Each image has a corresponding `.json` label file with the following schema:

```jsonc
{
  "image_path": "003.png",           // Filename of the corresponding image
  "bboxes": [
    {
      "instance_id": "003_0",        // Unique ID: <image_id>_<instance_index>
      "taxonomy": {
        "primary_class": "Material_loss",   // One of: Crack, Material_loss, Stain, External_Fixings
        "sub_type": "Peeling"               // See taxonomy table for valid sub-types
      },
      "bbox": [293, 369, 420, 296]   // [x_center, y_center, width, height]
    },
    {
      "instance_id": "003_1",
      "taxonomy": {
        "primary_class": "Stain",
        "sub_type": "Leakage stain"
      },
      "bbox": [533.53, 547.20, 121.17, 116.05]
    }
  ]
}
```

### Field Descriptions

| Field | Type | Description |
|---|---|---|
| `image_path` | string | Image filename (matches file in `images/` directory) |
| `bboxes` | array | List of all defect instances in the image |
| `bboxes[].instance_id` | string | Unique identifier: `<image_id>_<index>` |
| `bboxes[].taxonomy.primary_class` | string | Primary defect category (4 classes) |
| `bboxes[].taxonomy.sub_type` | string | Fine-grained sub-type (10 sub-types) |
| `bboxes[].bbox` | array[4] | Bounding box as `[x_center, y_center, width, height]` |

### Mask Files

- Binary PNG images: `<image_id>_mask.png`
- Pixel value `255` = defect region, `0` = background
- Same spatial resolution as the corresponding source image

### Primary Class Labels (Standardized)

| Label in JSON | Display Name |
|---|---|
| `Crack` | Crack |
| `Material_loss` | Material Loss |
| `Stain` | Surface Stain |
| `External_Fixings` | External Fixings |

---

## Source Datasets

DefectBench integrates and harmonizes 12 open-source datasets spanning classification, detection, and segmentation:

### Classification Datasets

| Dataset | Size | Categories | Annotation |
|---|---:|---|---|
| CCIC | 20,000 | Surface cracks | Patch-level |
| BDD | 436 | Roof defect, cracks, flakes | Patch-level |
| BD3 | 3,965 | Algae, major/minor crack, peeling, spalling, stain | Patch-level |
| HS-23K | 23,688 | Biological/chemical deterioration, cracks, human-induced damage, material loss, undamaged | Patch-level |

### Detection Datasets

| Dataset | Size | Categories | Annotation |
|---|---:|---|---|
| MBDD2025 | 14,471 | Crack, leakage, abscission, corrosion, bulge | Bounding-box |
| BDW | 1,417 | Crack, mold, peeling, stairstep crack, water seepage | Bounding-box |
| CUBIT-Det | 5,527 | Crack, spalling, moisture | Bounding-box |

### Segmentation Datasets

| Dataset | Size | Categories | Annotation |
|---|---:|---|---|
| Bai-2020 | 1,221 | Crack | Pixel-level |
| CSD | 11,298 | Crack | Pixel-level |
| DeepCrack | 537 | Crack | Pixel-level |
| CUBIT-Seg | 6,622 | Crack, spalling | Pixel-level |
| S2DS | 743 | Crack, spalling, corrosion, efflorescence, vegetation | Pixel-level |

### Label Harmonization

Cross-dataset label ambiguity was resolved through expert re-annotation. Example mappings:

| Original Label (various sources) | DefectBench Label |
|---|---|
| flake, peeling, degraded plaster | `Material_loss` → `Peeling` |
| spalling, abscission | `Material_loss` → `Spalling` |
| mold, water seepage, leakage | `Stain` → `Leakage stain` |
| corrosion, rust | `Stain` → `Corrosion` |
| vegetation, biological deterioration | `External_Fixings` → `Vegetation growth` |

---

## Benchmark Tasks & Evaluation Prompts

DefectBench defines 5 hierarchical tasks (Q1–Q5) across 3 cognitive levels:

### Level 1: Semantic Perception ("What")

**Q1 — Defect Identification** & **Q2 — Defect Counting**

```
       "You are an expert in building defect analysis. You will see one image.\n"
        "Your task is to answer TWO questions about visible defects in this image.\n\n"
        "Only consider these four primary defect types, and use EXACTLY these English names in your answers:\n\n"
        "1. Crack: Any type of crack or fissure in the building surface, including:\n"
        "2. material_loss: Loss of material from the building surface, including:\n"
        "   - peeling, spalling, flakes, peeling_paint, Abscission, Bulge\n"
        "3. Stain: Discoloration or staining on the building surface, including:\n"
        "   - algae, stain, biological_deteriorations, mold, water_seepage, Dampness, Efflorescence, Leakage, Corrosion, chemical_deteriorations\n"
        "4. External Fixings: External objects or human-made additions on the building surface, including:\n"
        "   - human_caused_damages (such as graffiti, vandalism, or other human-made marks)\n"
        "   - Vegetation (plants, moss, or other vegetation growing on the surface)\n\n"
        "The two questions are:\n"
        "1. What defects are in the image? (list defect types, separated by comma)\n"
        "2. How many instances of each defect type? (follow the same order as in Q1, give counts separated by comma)\n\n"
        "Return your answers STRICTLY as a single JSON object with exactly these keys:\n"
        "  {\n"
        "    \"answer1\": \"...\",  // Answer to question 1 (defect types)\n"
        "    \"answer2\": \"...\"   // Answer to question 2 (counts)\n"
        "  }\n"
        "Do NOT add any extra commentary, explanations, or markdown outside of this JSON.\n"
        "The JSON must be valid and parseable by Python json.loads.\n"
    
```


### Level 2: Spatial Localization ("Where")

**Q3 — Object Detection**

```
 "Based on your previous analysis, you identified these defects: {answer1}\n"
        f"And the counts are: {answer2}\n\n"
        "Now, please detect the bounding boxes for each defect instance you identified.\n"
        "For each detected object, return its category and bounding box.\n"
        f"The bounding box must be written as \"bbox\": \"{BBOX_TAG_START}x_min y_min x_max y_max{BBOX_TAG_END}\".\n"
        "Use EXACTLY these category names: Crack, material_loss, Stain, External Fixings\n"
        "Return ONLY a JSON array like:\n"
        f"[{{\"category\": \"Crack\", \"bbox\": \"{BBOX_TAG_START}x1 y1 x2 y2{BBOX_TAG_END}\"}}, "
        f"{{\"category\": \"material_loss\", \"bbox\": \"{BBOX_TAG_START}x3 y3 x4 y4{BBOX_TAG_END}\"}}]"
```

**Q4 — Visual Spatial Reasoning**

```
"Based on your previous analysis, you identified these defects: {defects_list}\n\n"
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
        "      [\"1#Crack\", \"adjacency\", \"2#material_loss\"],\n"
        "      [\"1#Crack\", \"overlapping\", \"3#Stain\"]\n"
        "    ]\n"
        "  }\n"
        "Format: [\"{number}#{type}\", \"{relation}\", \"{number}#{type}\"]\n"
        "Use EXACTLY these relation names: inclusion, overlapping, adjacency, disjoint\n"
        "Do NOT add any extra commentary, explanations, or markdown outside of this JSON.\n"
        "The JSON must be valid and parseable by Python json.loads.\n"

```

### Level 3: Generative Geometry Segmentation ("How")

**Q5 — Geometry Segmentation**

```
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
    
```

---

## Evaluation Metrics

| Level | Task | Metrics | Description |
|---|---|---|---|
| L1 | Q1: Classification | P, R, F1 | Per-class and average precision, recall, F1-score |
| L1 | Q2: Counting | MAE, RE | Mean Absolute Error and Relative Error per category |
| L2 | Q3: Detection | P, R, F1 | IoU-thresholded bounding box evaluation |
| L2 | Q4: Spatial Reasoning | P, R, F1 | Relation triplet matching accuracy |
| L3 | Q5: Segmentation | mIoU, P, R, F1, AP | Pixel-level mask quality metrics |

---

## Statistics

- **Total images**: 1,488 (1,485 in `final_dataset/`)
- **Total defect instances**: 4,527
- **Defect classes**: 4 primary / 10 sub-types
- **Image format**: PNG
- **Annotation format**: JSON (per-image)
- **Mask format**: Binary PNG

### Instance Distribution

```
Crack             ████████████████████████████████████████  2,122  (46.9%)
  ├─ Linear crack ██████████████████████████████████████    2,042
  └─ Map cracking █                                           80

Surface Stain     ████████████████████                      1,078  (23.8%)
  ├─ Leakage stain███████████████                             803
  ├─ Corrosion    █████                                       254
  └─ Rust stain                                                21

Material Loss     ██████████████████                          957  (21.1%)
  ├─ Spalling     █████████████████                            906
  └─ Peeling      █                                             51

External Fixings  ███████                                     370   (8.2%)
  ├─ Vegetation   ████                                         221
  ├─ Contaminants ██                                           119
  └─ Graffiti     █                                             30
```
