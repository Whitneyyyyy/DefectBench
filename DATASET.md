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

## Directory Structure

```
DefectBench/
├── final_dataset/                  # Full dataset (1,485 samples)
│   ├── images/                     # Facade images
│   │   ├── 003.png
│   │   ├── 007.png
│   │   └── ...
│   ├── labels/                     # JSON annotations (bbox + taxonomy)
│   │   ├── 003.json
│   │   ├── 007.json
│   │   └── ...
│   └── masks/                      # Binary segmentation masks
│       ├── 003_mask.png
│       ├── 007_mask.png
│       └── ...
│
├── test_100/                       # 100-image evaluation subset
│   ├── images/
│   ├── labels/
│   ├── masks/
│   ├── Crack/                      # Category-wise organized
│   ├── Material_loss/
│   ├── Stain/
│   └── External_Fixings/
│
├── Visualization_selected_100/     # Visualization outputs
│   ├── images/
│   ├── labels/
│   └── masks/
│
├── README.md
└── DATASET.md                      # This file
```

---

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

**Q1 — Defect Identification**

```
You are an expert in building defect analysis. You will see one image.
Your task is to answer two questions about visible defects in this image.

Only consider the following four primary defect types, and use exactly
these English names: Crack; material_loss; Stain; External Fixings.

Return your answers strictly as a single JSON object:
{"answer1": "...", "answer2": "..."}
```

**Q2 — Defect Counting**

```
Based on the identified defect types, report the number of instances
for each defect class. Counts must follow the same order as in Q1
and be separated by commas.
```

### Level 2: Spatial Localization ("Where")

**Q3 — Object Detection**

```
Detect bounding boxes for each defect instance identified previously.
Each object must be returned with its category and bounding box in
the format: bbox: "<x_min y_min x_max y_max>".

Return only a JSON array:
[{"category": "...", "bbox": "..."}]
Use exactly the category names: Crack, material_loss, Stain, External Fixings.
```

**Q4 — Visual Spatial Reasoning**

```
Analyze spatial relationships between all pairs of detected defects.
Possible relations include: inclusion, overlapping, adjacency, and disjoint.

Return a JSON object with the structure:
{"relationships": [[i#type, relation, j#type]]}
The output must be valid JSON without any additional text.
```

### Level 3: Generative Geometry Segmentation ("How")

**Q5 — Geometry Segmentation**

```
Edit the input image directly to generate a binary segmentation mask
for all identified defects.
```

---

## Evaluation Metrics

| Level | Task | Metrics | Description |
|---|---|---|---|
| L1 | Q1: Classification | P, R, F1 | Per-class and average precision, recall, F1-score |
| L1 | Q2: Counting | MAE, RE | Mean Absolute Error and Relative Error per category |
| L2 | Q3: Detection | P, R, F1 | IoU-thresholded bounding box evaluation |
| L2 | Q4: Spatial Reasoning | P, R, F1 | Relation triplet matching accuracy |
| L3 | Q5: Segmentation | IoU, Dice, mIoU | Pixel-level mask quality metrics |

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
