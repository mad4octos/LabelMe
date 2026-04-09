"""Persistence layer for incorrect/edited predictions in COCO format."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
from loguru import logger
from supervision.dataset.formats.yolo import _polygons_to_masks
from supervision.dataset.utils import rle_to_mask

from labelme.coco_dataset import extract_labelme_polygons_from_coco_annotation
from labelme.guided_review_mode import AnnotationPair
from labelme.labelme_types import CocoAnnotation
from labelme.labelme_types import RejectedCocoAnnotation
from labelme.labelme_types import is_coco_annotation
from labelme.labelme_types import is_compressed_rle
from labelme.labelme_types import is_polygon_segmentation
from labelme.shape import Shape
from labelme.utils.shape import shape_to_mask

# Minimum IoU change required to consider an edit "significant".
# If pre-edit and post-edit masks overlap by more than this fraction,
# the edit is considered a no-op and will not be saved.
_EDIT_SIGNIFICANCE_IOU_THRESHOLD = 0.95


class IncorrectPredictionsPersistence:
    """
    Manages saving rejected/edited annotations in COCO format for training.

    Creates an 'incorrect_predictions.json' file in the dataset directory,
    storing COCO annotations when they are deleted or edited.
    These can be used as hard negative examples during model training.

    Output format follows COCO structure:
    {
        "images": [...],
        "categories": [...],
        "annotations": [
            {
                ...standard COCO annotation fields...,
                "rejection_type": "deleted" | "edited",
            }
        ]
    }
    """

    FILENAME = "incorrect_predictions.json"

    def __init__(
        self,
        dataset_dir: str | Path,
        category_names: dict[int, str],
        image_id_by_filename: dict[str, int] | None = None,
    ) -> None:
        self._dataset_dir = Path(dataset_dir)
        self._filepath = self._dataset_dir / self.FILENAME
        self._category_names = category_names
        # Reverse mapping: category name -> category id
        self._category_id_by_name = {name: id_ for id_, name in category_names.items()}
        # Mapping: filename -> image_id (for GUI-created shapes)
        self._image_id_by_filename = image_id_by_filename or {}
        # group_id -> list of COCO annotations pending edit
        self._pending_edits: dict[int, list[CocoAnnotation]] = {}
        # Counter for generating unique annotation IDs for GUI-created shapes
        self._next_generated_id = -1  # Use negative IDs to avoid conflicts

        # In-memory cache of the COCO data
        self._images: dict[int, dict] = {}  # image_id -> image info
        self._categories: dict[int, dict] = {}  # category_id -> category info
        self._annotations: list[RejectedCocoAnnotation] = []

        # Load existing data if file exists
        self._load()

    def _load(self) -> None:
        """Load existing incorrect predictions from disk."""
        if not self._filepath.exists():
            return

        try:
            with open(self._filepath, encoding="utf-8") as f:
                data = json.load(f)

            for img in data.get("images", []):
                self._images[img["id"]] = img

            for cat in data.get("categories", []):
                self._categories[cat["id"]] = cat

            self._annotations = data.get("annotations", [])

            logger.debug(
                f"Loaded {len(self._annotations)} incorrect predictions "
                f"from {self._filepath}"
            )
        except Exception as e:
            logger.error(f"Failed to load incorrect predictions: {e}")

    def _save(self) -> bool:
        """Save current state to disk."""
        data = {
            "images": list(self._images.values()),
            "categories": list(self._categories.values()),
            "annotations": self._annotations,
        }

        try:
            with open(self._filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.debug(f"Saved incorrect predictions to {self._filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save incorrect predictions: {e}")
            return False

    def _extract_coco_annotation(self, shape: Shape) -> dict | None:
        """Extract COCO annotation from a shape's other_data."""
        if not shape.other_data:
            return None
        return shape.other_data.get("original_annotation")

    def _generate_annotation_id(self) -> int:
        """Generate a unique negative ID for GUI-created annotations."""
        id_ = self._next_generated_id
        self._next_generated_id -= 1
        return id_

    def _add_image_if_needed(
        self,
        image_id: int,
        frame_name: str,
        image_height: int,
        image_width: int,
    ) -> None:
        """Add image info to the images list if not already present."""
        if image_id not in self._images:
            self._images[image_id] = {
                "id": image_id,
                "width": image_width,
                "height": image_height,
                "file_name": frame_name,
            }

    def _add_category_if_needed(
        self, category_id: int, category_name: str
    ) -> None:
        """Add category info if not already present."""
        if category_id not in self._categories:
            self._categories[category_id] = {
                "id": category_id,
                "name": category_name,
                "supercategory": "",
            }

    def _annotation_exists(self, annotation_id: int) -> bool:
        """Check if an annotation with this ID already exists."""
        return any(ann.get("id") == annotation_id for ann in self._annotations)

    def _save_coco_annotations(
        self,
        coco_annotations: list[CocoAnnotation],
        frame_name: str,
        image_height: int,
        image_width: int,
        rejection_type: Literal["deleted", "edited"],
        group_id: int,
    ) -> bool:
        """
        Save COCO annotations to the incorrect predictions file.

        Args:
            coco_annotations: List of COCO annotation dicts to save.
            frame_name: Name of the image file.
            image_height: Height of the image.
            image_width: Width of the image.
            rejection_type: Type of rejection ("deleted" or "edited").
            group_id: The annotation group ID (for logging).

        Returns:
            True if save was successful, False otherwise.
        """
        saved_count = 0

        for coco_ann in coco_annotations:
            # Skip if this annotation was already saved
            if self._annotation_exists(coco_ann["id"]):
                logger.debug(
                    f"Annotation {coco_ann['id']} already exists, skipping"
                )
                continue

            # Add image and category info
            self._add_image_if_needed(
                image_id=coco_ann["image_id"],
                image_width=image_width,
                image_height=image_height,
                frame_name=frame_name,
            )

            category_id = coco_ann["category_id"]
            category_name = self._category_names.get(category_id)
            assert category_name is not None

            self._add_category_if_needed(
                category_id=category_id, category_name=category_name
            )

            # Create enriched annotation with rejection metadata
            enriched_ann: RejectedCocoAnnotation = {
                **coco_ann,
                "rejection_type": rejection_type,
            }

            self._annotations.append(enriched_ann)
            saved_count += 1

        if saved_count > 0:
            success = self._save()
            if success:
                logger.info(
                    f"Saved {saved_count} {rejection_type} annotation(s) "
                    f"for group_id={group_id}"
                )
            return success

        return True  # No annotations to save is not an error

    def _collect_coco_annotations(self, pair: AnnotationPair) -> list[CocoAnnotation]:
        """Extract valid original COCO annotations from polygon shapes in a pair."""
        coco_annotations: list[CocoAnnotation] = []
        for shape in pair.shapes:
            if shape.shape_type != "polygon":
                continue

            coco_ann = self._extract_coco_annotation(shape)

            # This is a shape that was manually created during the session
            if coco_ann is None:
                continue

            if not is_coco_annotation(coco_ann):
                raise Exception(
                    f"Shape '{shape.label}' (group_id={pair.group_id}) "
                    "has an invalid COCO annotation."
                )

            coco_annotations.append(coco_ann)

        return coco_annotations

    def save_deleted_shapes(
        self, frame_name: str, pair: AnnotationPair, image_height: int, image_width: int
    ) -> bool:
        """Save deleted shapes to the incorrect predictions file in COCO format."""
        return self._save_coco_annotations(
            coco_annotations=self._collect_coco_annotations(pair),
            frame_name=frame_name,
            image_height=image_height,
            image_width=image_width,
            rejection_type="deleted",
            group_id=pair.group_id,
        )

    def capture_for_edit(self, pair: AnnotationPair) -> None:
        """
        Capture original COCO annotations before user edits them.

        The annotations are held in memory until the edit is confirmed or cancelled.

        Args:
            group_id: The annotation group ID being edited
            shapes: List of Shape objects to capture
        """

        coco_annotations = self._collect_coco_annotations(pair)
        if coco_annotations:
            self._pending_edits[pair.group_id] = coco_annotations
            logger.debug(
                f"Captured {len(coco_annotations)} COCO annotation(s) "
                f"for potential edit (group_id={pair.group_id})"
            )

    def finalize_edit(
        self,
        frame_name: str,
        group_id: int,
        image_height: int,
        image_width: int,
    ) -> bool:
        """
        Finalize and save the previously captured annotations for an edit.

        Called when user confirms the edit (TO_EDIT -> EDITED transition).

        Returns:
            True if save was successful, False otherwise.
        """
        if group_id not in self._pending_edits:
            logger.warning(f"No pending edit found for group_id={group_id}")
            return False

        coco_annotations = self._pending_edits.pop(group_id)

        return self._save_coco_annotations(
            coco_annotations=coco_annotations,
            frame_name=frame_name,
            image_height=image_height,
            image_width=image_width,
            rejection_type="edited",
            group_id=group_id,
        )

    @staticmethod
    def _coco_annotation_to_mask(
        coco_ann: CocoAnnotation, image_height: int, image_width: int
    ) -> npt.NDArray[np.bool_]:
        """Decode a COCO annotation's segmentation to a binary mask."""
        mask = np.zeros((image_height, image_width), dtype=bool)

        segmentation = coco_ann.get("segmentation")
        if is_compressed_rle(segmentation):
            rle = np.array(segmentation["counts"])
            h, w = segmentation["size"]
            mask = rle_to_mask(rle=rle, resolution_wh=(w, h))
            mask = mask.astype(bool)
            assert mask.shape == (image_height, image_width)
        elif is_polygon_segmentation(segmentation):
            list_of_polygons = extract_labelme_polygons_from_coco_annotation(coco_ann)
            if list_of_polygons:
                masks = _polygons_to_masks(
                    list_of_polygons, (image_width, image_height)
                )
                mask = np.max(masks, axis=0).astype(bool)

        return mask

    @staticmethod
    def _shapes_to_mask(
        shapes: list[Shape], group_id: int, image_height: int, image_width: int
    ) -> npt.NDArray[np.bool_]:
        """Convert canvas shapes for a given group_id to a combined binary mask."""
        mask = np.zeros((image_height, image_width), dtype=bool)
        for shape in shapes:
            if shape.shape_type != "polygon":
                continue

            if shape.group_id != group_id:
                continue

            if shape.mask is not None:
                if shape.mask.shape == (image_height, image_width):
                    mask |= shape.mask.astype(bool)
            elif shape.points:
                points = [[p.x(), p.y()] for p in shape.points]
                mask |= shape_to_mask(
                    (image_height, image_width), points, shape_type="polygon"
                )
        return mask

    def has_significant_changes(
        self,
        group_id: int,
        current_shapes: list[Shape],
        image_height: int,
        image_width: int,
    ) -> bool:
        """Return True if the edit made significant changes (mask IoU < threshold).

        Compares the original captured annotation mask with the current canvas
        shapes. If the masks are nearly identical the edit is considered a no-op
        and the annotation should not be saved.
        """
        if group_id not in self._pending_edits:
            logger.debug(f"No pending edit for group_id={group_id}, assuming changed")
            return True

        # Build original mask from captured COCO annotations
        original_mask = np.zeros((image_height, image_width), dtype=bool)
        for ann in self._pending_edits[group_id]:
            original_mask |= self._coco_annotation_to_mask(
                ann, image_height, image_width
            )

        # Build current mask from canvas shapes with matching group_id
        current_mask = self._shapes_to_mask(
            current_shapes, group_id, image_height, image_width
        )

        union = np.logical_or(original_mask, current_mask).sum()
        if union == 0:
            return False  # Both masks empty — nothing to compare

        intersection = np.logical_and(original_mask, current_mask).sum()
        iou = float(intersection / union)

        logger.debug(
            f"Edit IoU for group_id={group_id}: {iou:.3f} "
            f"(threshold={_EDIT_SIGNIFICANCE_IOU_THRESHOLD})"
        )
        return iou < _EDIT_SIGNIFICANCE_IOU_THRESHOLD

    def cancel_pending_edit(self, group_id: int) -> None:
        """Discard a pending edit capture without saving (no significant changes)."""
        self._pending_edits.pop(group_id, None)
        logger.debug(
            f"Cancelled pending edit for group_id={group_id} (no significant change)"
        )

    def clear_pending_edits(self) -> None:
        """Clear all pending edit captures (e.g., when switching frames)."""
        if self._pending_edits:
            logger.debug(
                f"Clearing {len(self._pending_edits)} pending edits on frame change"
            )
        self._pending_edits.clear()
