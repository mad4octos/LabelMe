"""
Extract cropped image regions from COCO annotations, with optional mask generation.

This script processes COCO annotation files (including incorrect_predictions.json)
and generates:
1. Cropped image regions using bounding boxes (saved to output_dir/crops/)
2. Optionally, binary masks from polygon segmentations (saved to output_dir/masks/)
   - Full-size masks and cropped masks (when bbox crops are generated)

Usage:
    python extract_crops_from_coco.py --coco-file annotations.json \
        --images-dir ./images --output-dir ./output [--extract-masks]
"""

from __future__ import annotations

# Standard Library imports
import argparse
import json
from pathlib import Path

# External imports
import cv2
import numpy as np
import supervision as sv
from supervision.utils.image import crop_image

# Local imports
from labelme.coco_dataset import coco_annotations_to_masks


def process_coco_file(
    coco_file: Path, images_dir: Path, output_dir: Path, extract_masks: bool = False
) -> None:
    """
    Process a COCO annotation file and generate cropped images and/or binary masks.

    For each annotation, generates:
    - A cropped image region using the bbox (saved to output_dir/crops/)

    Optionally:
    - A binary mask from the segmentation polygon (saved to output_dir/masks/)
    - A cropped mask combining both (saved to output_dir/masks/)

    Args:
        coco_file: Path to the COCO JSON file.
        images_dir: Directory containing the source images.
        output_dir: Directory to save outputs.
        extract_masks: Whether to extract binary masks from segmentations.
    """
    with open(coco_file, encoding="utf-8") as f:
        coco_data = json.load(f)

    # Build image lookup
    images = {img["id"]: img for img in coco_data.get("images", [])}
    categories = {cat["id"]: cat for cat in coco_data.get("categories", [])}
    annotations = coco_data.get("annotations", [])

    # Create output directories
    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    if extract_masks:
        masks_dir = output_dir / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing {len(annotations)} annotations from {coco_file.name}")

    for ann in annotations:
        ann_id = ann["id"]
        image_id = ann["image_id"]
        category_id = ann["category_id"]

        if image_id not in images:
            print(
                f"Warning: Image ID {image_id} not found, skipping annotation {ann_id}"
            )
            continue

        image_info = images[image_id]
        image_path = images_dir / image_info["file_name"]

        if not image_path.exists():
            print(
                f"Warning: Image file not found: {image_path}, skipping annotation {ann_id}"
            )
            continue

        img_h = image_info["height"]
        img_w = image_info["width"]
        category_name = categories.get(category_id, {}).get("name")
        assert category_name is not None

        # Get rejection type if present (for incorrect_predictions.json)
        rejection_type = ann.get("rejection_type", "")
        suffix = f"_{rejection_type}" if rejection_type else ""

        # Base filename for outputs
        base_name = (
            f"{Path(image_info['file_name']).stem}_ann{ann_id}_{category_name}{suffix}"
        )

        # Generate mask from segmentation (optional)
        mask = None
        if extract_masks and "segmentation" in ann:
            masks = coco_annotations_to_masks([ann], (img_w, img_h))
            mask = (masks[0] * 255).astype(np.uint8)
            mask_path = masks_dir / f"{base_name}_mask.png"
            cv2.imwrite(str(mask_path), mask)
            print(f"Saved mask: {mask_path.name}")

        # Crop image using bbox
        if "bbox" in ann:
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Warning: Could not read image: {image_path}")
                continue

            xywh = np.array([ann["bbox"]])
            xyxy = sv.xywh_to_xyxy(xywh)[0]
            crop = crop_image(image=image, xyxy=xyxy)
            crop_path = crops_dir / f"{base_name}_crop.png"
            cv2.imwrite(str(crop_path), crop)
            print(f"Saved crop: {crop_path.name}")

            # Also save cropped mask if mask extraction is enabled
            if mask is not None:
                mask_crop = crop_image(image=mask, xyxy=xyxy)
                mask_crop_path = masks_dir / f"{base_name}_mask_crop.png"
                cv2.imwrite(str(mask_crop_path), mask_crop)

    print(f"\nDone! Outputs saved to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert COCO segmentations to SAM2 masks and crop frames using bbox"
    )
    parser.add_argument(
        "--coco-file",
        type=Path,
        required=True,
        help="Path to COCO annotation JSON file (e.g., incorrect_predictions.json)",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        required=True,
        help="Directory containing source images",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./sam2_outputs"),
        help="Output directory for masks and crops (default: ./sam2_outputs)",
    )
    parser.add_argument(
        "--extract-masks",
        action="store_true",
        help="Extract binary masks from segmentation polygons (disabled by default)",
    )

    args = parser.parse_args()

    if not args.coco_file.exists():
        print(f"Error: COCO file not found: {args.coco_file}")
        return

    if not args.images_dir.exists():
        print(f"Error: Images directory not found: {args.images_dir}")
        return

    process_coco_file(
        coco_file=args.coco_file,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        extract_masks=args.extract_masks,
    )


if __name__ == "__main__":
    main()
