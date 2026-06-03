# Labelme fork

## Custom Modifications

This fork extends the original Labelme with enhanced COCO dataset support, a Guided Review Mode for systematic annotation validation, keyboard-based shape navigation, and group ID validation for bbox-polygon pairing.

### Key Features

**COCO Dataset Integration**
- Import and visualize COCO-format datasets (polygon and RLE formats)
- Export annotations to COCO polygon format via **File → Export COCO Annotations**
- Lazy loading support for large datasets
- Dataset integrity verification on load
- Ground-truth location overlay: displays ground truth attributes from COCO annotations as circles with metadata labels (location, object ID, extracted frame, original frame). These attributes come from the original .npy annotations file used to create the COCO annotations.

**Saving & Exporting**

This fork replaces the original Labelme save workflow with a COCO-centric one. The relevant File menu entries behave as follows:

| Menu item | Behavior |
|-----------|----------|
| **Save** (`Ctrl+S`) | Stages the current frame's annotations in memory. Does **not** write to disk. |
| **Save As** | Disabled — it wrote to the original Labelme JSON format, which is not used in the COCO workflow. |
| **Save Automatically** | Disabled |
| **Save With Image Data** | Disabled — embedding raw image bytes in the annotations file is not relevant for the COCO workflow. |
| **Export COCO Annotations** | Writes all staged annotations to a COCO JSON file on disk. Run this when you are done editing a session. |

**Guided Review Mode** ([details](#guided-review-mode))
- Review bbox-polygon pairs grouped by Object ID
- Keyboard-driven workflow: Confirm, Edit, or Delete annotations
- Progress tracking with auto-save to `.labelme_review.json`
- Auto-saves deleted/edited annotations to `incorrect_predictions.json` for hard negative mining

**Enhanced Navigation**
- Keyboard shortcuts: `W`/`S` for shape navigation, `A`/`D` for frame navigation
- Auto-centering on selected shapes
- Shape type indicators in label list
- Polygon Labels list sorted by label name, then group ID

**Persistent Polygon Visibility**
- Unchecking a shape in the Polygon Labels panel hides it across frame switches
- Visibility is tracked per (label, group ID, shape type) combination
- Re-checking a shape restores visibility on all frames

**Linked Polygon-BBox Behavior**
- Auto-creates bounding boxes around polygons with synchronized group IDs
- Moving a polygon also moves its paired bounding box (not vice versa)
- Synchronized label and ObjID editing between paired shapes
- Auto-updates bounding box dimensions when polygon is resized

**Utility Scripts** ([details](#utility-scripts))
- Extract image crops and masks from COCO annotations for hard negative training

### Guided Review Mode

Guided Review Mode provides a structured workflow for validating and reviewing annotations. 

#### Starting Guided Review Mode

1. Open a directory containing COCO annotated images
2. Press `Ctrl+G` or click the button "Guided Review" found in the Tools bar
3. The review dock widget will appear showing progress and controls

#### Review Workflow

When review mode is active:
- Annotations are grouped by their Object ID (`group_id`)
- The current annotation pair (bbox + polygon) is highlighted, while other annotations are dimmed
- For each annotation pair, you can:
  - **Confirm** (`C` or `Enter`): Mark as correct and move to next
  - **Edit** (`E`): Mark the shape as "to edit" and exit review mode. While outside review mode, modify the shape as needed, then re-enter review mode and press `C` to confirm. The shape will then be marked as "edited".
  - **Delete** (`Backspace`): Mark as deleted and move to next
  - **Reset Frame** (`R`): Reset all review progress for the current frame

#### Keyboard Shortcuts

| Action | Shortcut |
|--------|----------|
| Start Guided Review | `Ctrl+G` |
| Confirm | `C` or `Enter` |
| Edit | `E` |
| Delete | `Backspace` |
| Reset Frame | `R` |
| Exit Review | `Escape` |

#### Progress Tracking

- **Overall Progress**: Shows how many frames have been completed across the dataset
- **Frame Progress**: Shows how many annotation pairs have been reviewed in the current frame
- **View Summary**: Click to see a detailed breakdown of review statistics for the current frame, all frames, and individual annotations

#### Review State Persistence

Review progress is automatically saved to `.labelme_review.json` in the dataset directory. The file is saved immediately after every review action (confirming, editing, deleting, or resetting), so there's no risk of losing progress. This allows you to:
- Resume reviewing where you left off
- Track which frames have been completed
- See the review status of each annotation

#### Annotation Statuses

| Status | Description |
|--------|-------------|
| Pending | Not yet reviewed |
| To Edit | Marked for editing, awaiting modifications and confirmation |
| Edited | Was edited and then confirmed |
| Confirmed | Reviewed and marked as correct |
| Deleted | Marked for deletion |

#### Frame Completion

After reviewing all annotations in a frame:
1. A dialog appears asking if you missed any annotations
2. Choose "Yes, Add Annotations" to exit review mode and add missing annotations
3. Choose "No, Continue to Next Frame" to save and auto-advance to the next image

#### Group ID Validation

Each `group_id` represents a single physical object and may contain:
- **At most one rectangle** (the bounding box).
- **One or more polygons** — multiple polygons per group are supported for objects whose mask is split into disjoint regions (e.g., an animal partially occluded so one body is visible as two separate areas). The group's bounding box is auto-recomputed to enclose the union of all its polygons.

When you assign a group ID to a new shape, the system rejects the assignment if it would produce a second rectangle in the same group. This keeps the Guided Review pairing (one bbox + one or more polygons) well-defined.

### Hard Negative Mining

Hard negatives are training examples where the model made incorrect predictions with high confidence. By collecting these failure cases and including them in training, the model can learn to avoid these specific types of errors, leading to more robust performance.

#### How It Works

During Guided Review, when an annotation is **deleted** or **edited**:
1. The original annotation (as it was when loaded from disk) is captured
2. It is saved to `incorrect_predictions.json` in the dataset directory
3. The annotation includes a `rejection_type` field: `"deleted"` or `"edited"`

#### Output Format

The `incorrect_predictions.json` file follows standard COCO format:

```json
{
  "images": [...],
  "categories": [...],
  "annotations": [
    {
      "id": 123,
      "image_id": 456,
      "category_id": 1,
      "bbox": [x, y, width, height],
      "segmentation": [[...]],
      "rejection_type": "deleted"
    }
  ]
}
```

#### Use Cases

**Deleted annotations (false positive reduction)**

Bounding boxes from deleted annotations can be used to train a binary classifier that filters false positives from an object detector. The classifier learns to distinguish between true detections and the types of false positives the detector commonly produces.

Masks from deleted annotations can also be used to train an anomaly classification model that rejects incorrectly segmented objects.

**Edited annotations (mask quality assessment)**

Pre-edit masks from edited annotations can be useful for training an anomaly classification model that rejects incorrectly segmented objects. The model learns what "wrong" masks look like.

**Important:** These pre-edit masks should be manually reviewed before training. If a user edited an almost-perfect mask, the model would incorrectly learn that near-perfect masks are unacceptable. Before training, one should filter the pre-edit masks to only preserve those that were genuinely problematic (e.g., masks missing half of the object).

**General use cases**

- **Error analysis**: Reviewing patterns in model failures can guide data collection
- **Crop extraction**: The `extract_crops_from_coco.py` script can extract image crops from rejected predictions for training

### Utility Scripts

#### Extract Crops from COCO Annotations

The `scripts/extract_crops_from_coco.py` script extracts cropped image regions from COCO annotation files. It's primarily intended to be used with the `incorrect_predictions.json` file that stores deleted and edited annotations from Guided Review Mode. For each annotation in the COCO JSON file, it:
1. Reads the bounding box coordinates from the annotation
2. Crops the corresponding region from the source image
3. Optionally converts the polygon segmentation to a binary mask image
4. Saves the crops and masks with descriptive filenames that include the rejection type (deleted/edited)

This is particularly useful for extracting hard negatives to train classifiers that filter false positives or reject poor-quality segmentations.

**Usage:**

```bash
python scripts/extract_crops_from_coco.py \
    --coco-file path/to/incorrect_predictions.json \
    --images-dir path/to/images \
    --output-dir ./output \
    --extract-masks  # Optional: also extract binary masks
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `--coco-file` | Yes | Path to COCO annotation JSON file |
| `--images-dir` | Yes | Directory containing source images |
| `--output-dir` | No | Output directory (default: `./sam2_outputs`) |
| `--extract-masks` | No | Also extract binary masks from segmentation polygons |

**Output Structure:**

```
output_dir/
├── crops/
│   └── {image_name}_ann{id}_{category}_{rejection_type}_crop.png
└── masks/  (if --extract-masks is used)
    ├── {image_name}_ann{id}_{category}_{rejection_type}_mask.png
    └── {image_name}_ann{id}_{category}_{rejection_type}_mask_crop.png
```

### Technical Details

**COCO Format Handling**
- Polygon annotations (`iscrowd=0`) are loaded without approximation, preserving original precision
- RLE annotations (`iscrowd=1`) are converted to masks, then approximated to polygons for editing
- RLE annotations are expected to be in compressed format (list of ints).
- All exports use COCO polygon format (`iscrowd=0`) to avoid precision loss
- Bidirectional RLE-to-polygon conversion with configurable approximation tolerance (default: 0.008)
- Custom COCO attributes like object IDs are preserved during import/export

**Dataset Management**
- Uses `LazyCOCODataset` class for efficient handling of large datasets
- Validates datasets on load: duplicate IDs, orphan annotations, missing required fields
- Review state persists to `.labelme_review.json` with immediate saves after each action

**Review State File Format**

The `.labelme_review.json` file stores the review progress for each frame and annotation in the dataset. The file is automatically created in the dataset directory when you start a Guided Review session and is updated immediately after each review action.

Structure:
```json
{
  "version": "1.0",
  "frames": {
    "00001.jpg": {
      "status": "in_progress",
      "annotations": {
        "1": {
          "status": "confirmed",
          "reviewed_at": "2026-01-22T00:20:44.040144Z"
        },
        "9999": {
          "status": "deleted",
          "reviewed_at": "2026-01-22T00:20:45.357352Z"
        },
        "3": {
          "status": "to_edit",
          "reviewed_at": "2026-01-23T07:30:47.788155Z"
        }
      }
    },
    "00002.jpg": {
      "status": "pending",
      "annotations": {
        "999": {
          "status": "confirmed",
          "reviewed_at": "2026-01-22T00:20:34.053147Z"
        }
      }
    },
    "00003.jpg": {
      "status": "completed",
      "annotations": {
        "1": {
          "status": "deleted",
          "reviewed_at": "2026-01-21T08:32:45.711032Z"
        }
      }
    }
  }
}
```

Fields:
- `version`: Format version for compatibility tracking
- `frames`: Dictionary mapping image filenames to their review state
  - `status`: Frame-level status — one of:
    - `"pending"` — frame has been seen but no annotation in it has been reviewed or saved
    - `"in_progress"` — at least one annotation has been saved or reviewed, but the frame has not been marked complete
    - `"completed"` — the user has finished reviewing the frame
  - `annotations`: Dictionary mapping group IDs (Object IDs) to their review state
    - `status`: Annotation status (`"pending"`, `"confirmed"`, `"to_edit"`, `"edited"`, or `"deleted"`)
    - `reviewed_at`: ISO 8601 timestamp of when the annotation was last reviewed

The file enables:
- Resuming review sessions from where you left off
- Tracking which frames are complete vs. in-progress
- Monitoring individual annotation review history
- Generating review statistics and progress reports

**Ground-Truth Location Visualization**
- When COCO annotations contain a `gt_location` attribute (an `[x, y]` coordinate), a circle shape is rendered at that position on the canvas
- The circle displays metadata from the annotation's attributes as a text label: `gt_obj_id` (object ID), `gt_frame_extracted` (extracted frame number), and `gt_frame_original` (original frame number)
- Text is rendered with a semi-transparent black background and white font, positioned above the circle center

**Linked Shape Implementation**
- Bounding box padding configurable via `canvas.bbox_padding` in config (default: 3 pixels)
- Shapes linked via `group_id` field for synchronized operations
- Movement synchronization: moving a polygon automatically updates its paired bbox, but moving the bbox alone does not affect the polygon (allows manual bbox margin adjustments)

**Architecture**
- Type definitions in `labelme_types.py`, COCO module in `coco_dataset.py`
- Uses `supervision` library for COCO operations
- Fixed panning behavior at different zoom levels

---

<h1 align="center">
  <img src="labelme/icons/icon-256.png" width="200" height="200"><br/>labelme
</h1>

<h4 align="center">
  Image Polygonal Annotation with Python
</h4>

<div align="center">
  <a href="https://pypi.python.org/pypi/labelme"><img src="https://img.shields.io/pypi/v/labelme.svg"></a>
  <!-- <a href="https://pypi.org/project/labelme"><img src="https://img.shields.io/pypi/pyversions/labelme.svg"></a> -->
  <a href="https://github.com/wkentaro/labelme/actions"><img src="https://github.com/wkentaro/labelme/actions/workflows/ci.yml/badge.svg?branch=main&event=push"></a>
  <a href="https://discord.com/invite/uAjxGcJm83"><img src="https://dcbadge.limes.pink/api/server/uAjxGcJm83?style=flat"></a>
</div>

<div align="center">
  <a href="#installation"><b>Installation</b></a>
  | <a href="#usage"><b>Usage</b></a>
  | <a href="#examples"><b>Examples</b></a>
  <!-- | <a href="https://github.com/wkentaro/labelme/discussions"><b>Community</b></a> -->
  <!-- | <a href="https://www.youtube.com/playlist?list=PLI6LvFw0iflh3o33YYnVIfOpaO0hc5Dzw"><b>Youtube FAQ</b></a> -->
</div>

<br/>

<div align="center">
  <img src="examples/instance_segmentation/.readme/annotation.jpg" width="70%">
</div>

## Description

Labelme is a graphical image annotation tool inspired by <http://labelme.csail.mit.edu>.  
It is written in Python and uses Qt for its graphical interface.

<img src="examples/instance_segmentation/data_dataset_voc/JPEGImages/2011_000006.jpg" width="19%" /> <img src="examples/instance_segmentation/data_dataset_voc/SegmentationClass/2011_000006.png" width="19%" /> <img src="examples/instance_segmentation/data_dataset_voc/SegmentationClassVisualization/2011_000006.jpg" width="19%" /> <img src="examples/instance_segmentation/data_dataset_voc/SegmentationObject/2011_000006.png" width="19%" /> <img src="examples/instance_segmentation/data_dataset_voc/SegmentationObjectVisualization/2011_000006.jpg" width="19%" />  
<i>VOC dataset example of instance segmentation.</i>

<img src="examples/semantic_segmentation/.readme/annotation.jpg" width="30%" /> <img src="examples/bbox_detection/.readme/annotation.jpg" width="30%" /> <img src="examples/classification/.readme/annotation_cat.jpg" width="35%" />  
<i>Other examples (semantic segmentation, bbox detection, and classification).</i>

<img src="https://user-images.githubusercontent.com/4310419/47907116-85667800-de82-11e8-83d0-b9f4eb33268f.gif" width="30%" /> <img src="https://user-images.githubusercontent.com/4310419/47922172-57972880-deae-11e8-84f8-e4324a7c856a.gif" width="30%" /> <img src="https://user-images.githubusercontent.com/14256482/46932075-92145f00-d080-11e8-8d09-2162070ae57c.png" width="32%" />  
<i>Various primitives (polygon, rectangle, circle, line, and point).</i>


## Features

- [x] Image annotation for polygon, rectangle, circle, line and point. ([tutorial](examples/tutorial))
- [x] Image flag annotation for classification and cleaning. ([#166](https://github.com/wkentaro/labelme/pull/166))
- [x] Video annotation. ([video annotation](examples/video_annotation))
- [x] GUI customization (predefined labels / flags, auto-saving, label validation, etc). ([#144](https://github.com/wkentaro/labelme/pull/144))
- [x] Exporting VOC-format dataset for semantic/instance segmentation. ([semantic segmentation](examples/semantic_segmentation), [instance segmentation](examples/instance_segmentation))
- [x] Exporting COCO-format dataset for instance segmentation. ([instance segmentation](examples/instance_segmentation))


## Installation

There are 3 options to install labelme:

### Option 1: Using pip

For more detail, check ["Install Labelme using Terminal"](https://www.labelme.io/docs/install-labelme-terminal)

```bash
pip install labelme

# To install the latest version from GitHub:
# pip install git+https://github.com/wkentaro/labelme.git
```

### Option 2: Using standalone executable (Easiest)

If you're willing to invest in the convenience of simple installation without any dependencies (Python, Qt),
you can download the standalone executable from ["Install Labelme as App"](https://www.labelme.io/docs/install-labelme-app).

It's a one-time payment for lifetime access, and it helps us to maintain this project.

### Option 3: Using a package manager in each Linux distribution

In some Linux distributions, you can install labelme via their package managers (e.g., apt, pacman). The following systems are currently available:

[![Packaging status](https://repology.org/badge/vertical-allrepos/labelme.svg)](https://repology.org/project/labelme/versions)

## Usage

Run `labelme --help` for detail.  
The annotations are saved as a [JSON](http://www.json.org/) file.

```bash
labelme  # just open gui

# tutorial (single image example)
cd examples/tutorial
labelme apc2016_obj3.jpg  # specify image file
labelme apc2016_obj3.jpg -O apc2016_obj3.json  # close window after the save
labelme apc2016_obj3.jpg --nodata  # not include image data but relative image path in JSON file
labelme apc2016_obj3.jpg \
  --labels highland_6539_self_stick_notes,mead_index_cards,kong_air_dog_squeakair_tennis_ball  # specify label list

# semantic segmentation example
cd examples/semantic_segmentation
labelme data_annotated/  # Open directory to annotate all images in it
labelme data_annotated/ --labels labels.txt  # specify label list with a file
```

### Command Line Arguments
- `--output` specifies the location that annotations will be written to. If the location ends with .json, a single annotation will be written to this file. Only one image can be annotated if a location is specified with .json. If the location does not end with .json, the program will assume it is a directory. Annotations will be stored in this directory with a name that corresponds to the image that the annotation was made on.
- The first time you run labelme, it will create a config file in `~/.labelmerc`. You can edit this file and the changes will be applied the next time that you launch labelme. If you would prefer to use a config file from another location, you can specify this file with the `--config` flag.
- Without the `--nosortlabels` flag, the program will list labels in alphabetical order. When the program is run with this flag, it will display labels in the order that they are provided.
- Flags are assigned to an entire image. [Example](examples/classification)
- Labels are assigned to a single polygon. [Example](examples/bbox_detection)

### FAQ

- **How to convert JSON file to numpy array?** See [examples/tutorial](examples/tutorial#convert-to-dataset).
- **How to load label PNG file?** See [examples/tutorial](examples/tutorial#how-to-load-label-png-file).
- **How to get annotations for semantic segmentation?** See [examples/semantic_segmentation](examples/semantic_segmentation).
- **How to get annotations for instance segmentation?** See [examples/instance_segmentation](examples/instance_segmentation).


## Examples

* [Image Classification](examples/classification)
* [Bounding Box Detection](examples/bbox_detection)
* [Semantic Segmentation](examples/semantic_segmentation)
* [Instance Segmentation](examples/instance_segmentation)
* [Video Annotation](examples/video_annotation)


## How to build standalone executable

```bash
LABELME_PATH=./labelme
OSAM_PATH=$(python -c 'import os, osam; print(os.path.dirname(osam.__file__))')
pyinstaller labelme/labelme/__main__.py \
  --name=Labelme \
  --windowed \
  --noconfirm \
  --specpath=build \
  --add-data=$(OSAM_PATH)/_models/yoloworld/clip/bpe_simple_vocab_16e6.txt.gz:osam/_models/yoloworld/clip \
  --add-data=$(LABELME_PATH)/config/default_config.yaml:labelme/config \
  --add-data=$(LABELME_PATH)/icons/*:labelme/icons \
  --add-data=$(LABELME_PATH)/translate/*:translate \
  --icon=$(LABELME_PATH)/icons/icon-256.png \
  --onedir
```


## Acknowledgement

This repo is the fork of [mpitid/pylabelme](https://github.com/mpitid/pylabelme).
