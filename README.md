# Labelme fork

## Custom Modifications

This fork extends the original Labelme with enhanced COCO dataset support, a Guided Review Mode for systematic annotation validation, keyboard-based shape navigation, and group ID validation for bbox-polygon pairing.

### Key Features

**COCO Dataset Integration**
- **Load existing COCO annotations**: Import and visualize COCO-format datasets directly in the GUI
- **Lazy loading support**: Efficiently handle large COCO datasets with `LazyCOCODataset` class
- **Export to COCO format**: Save annotations in COCO polygon format
  - **Polygon format preservation**: Annotations are stored as COCO polygons (`iscrowd=0`) to preserve precision
  - When loading existing COCO annotations:
    - Polygon annotations (`iscrowd=0`) are loaded directly without approximation, preserving their original precision
    - RLE annotations (`iscrowd=1`) are converted to masks, then approximated to polygons for editing
  - When saving, all polygon shapes are exported as COCO polygon format (`iscrowd=0`) to avoid precision loss
- **Bidirectional conversion**: Convert between COCO RLE masks and polygon representations with configurable approximation tolerance
- **Custom attributes support**: Store non-standard COCO fields like object IDs and custom attributes

**Guided Review Mode**
- **Systematic annotation validation**: Review annotation pairs (bounding box + polygon) grouped by Object ID
- **Review actions**: Confirm, Edit, or Delete each annotation pair with keyboard shortcuts
- **Progress tracking**: Track review progress per frame and across the entire dataset
- **Persistent state**: Review progress is automatically saved to `.labelme_review.json`
- **Visual highlighting**: Current annotation pair is highlighted while others are dimmed
- **Auto-advance**: Automatically moves to the next frame after completing review

**Enhanced Navigation & Workflow**
- **Keyboard shortcuts for shape navigation**: Switch between bounding boxes/masks using `W` (previous) and `S` (next) keys
- **Auto-centering**: Selected shapes automatically center on screen when navigating with keyboard
- **Shape type indicators**: Polygon label list now displays shape types for better visibility
- **Improved file browsing**: Enhanced file list preview in dialog

**Technical Improvements**
- **Type safety**: Added comprehensive type definitions (`labelme_types.py`) for COCO structures and shape dictionaries
- **Modular architecture**: Separated COCO dataset handling into dedicated module (`coco_dataset.py`)
- **Mask-to-polygon optimization**: Improved polygon approximation from masks with adjustable tolerance (default: 0.008)
- **Supervision library integration**: Leverages `supervision` library for robust COCO operations

### Guided Review Mode

Guided Review Mode provides a structured workflow for validating and reviewing annotations. 

#### Starting Review Mode

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
  - **Delete** (`Delete`): Mark as deleted and move to next
  - **Reset Frame** (`R`): Reset all review progress for the current frame

#### Keyboard Shortcuts

| Action | Shortcut |
|--------|----------|
| Start Guided Review | `Ctrl+G` |
| Confirm | `C` or `Enter` |
| Edit | `E` |
| Delete | `Delete` |
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

Enforce that each `group_id` contains exactly one rectangle and one polygon. When assigning a group ID to a shape, the system validates that:
- The group doesn't already contain a shape of the same type
- Each group has at most one bounding box and one polygon

This ensures proper pairing for the Guided Review workflow.

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
