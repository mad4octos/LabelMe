"""Persistence layer for incorrect/edited predictions in COCO format."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Literal

from loguru import logger

from labelme.guided_review_mode import AnnotationPair
from labelme.labelme_types import CocoAnnotation
from labelme.labelme_types import RejectedCocoAnnotation
from labelme.shape import Shape


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

    def _extract_coco_annotation(self, shape: Shape) -> CocoAnnotation | None:
        """Extract COCO annotation from a shape's other_data."""
        if not shape.other_data:
            return None
        return shape.other_data.get("original_annotation")

    def _is_valid_coco_annotation(self, coco_ann: CocoAnnotation) -> bool:
        """Check if a COCO annotation has all required fields."""
        required_keys = ("id", "image_id", "category_id")
        return all(key in coco_ann for key in required_keys)

    def _generate_annotation_id(self) -> int:
        """Generate a unique negative ID for GUI-created annotations."""
        id_ = self._next_generated_id
        self._next_generated_id -= 1
        return id_

    def _create_coco_annotation_from_shape(
        self, shape: Shape, frame_name: str
    ) -> CocoAnnotation | None:
        """
        Create a COCO annotation from a GUI-created Shape.

        Returns None if the shape cannot be converted (missing label, unknown category,
        or unknown image).
        """
        if not shape.label:
            logger.warning("Shape has no label, cannot create COCO annotation")
            return None

        category_id = self._category_id_by_name.get(shape.label)
        if category_id is None:
            logger.warning(
                f"Unknown category '{shape.label}', cannot create COCO annotation"
            )
            return None

        image_id = self._image_id_by_filename.get(frame_name)
        if image_id is None:
            logger.warning(
                f"Unknown image '{frame_name}', cannot create COCO annotation"
            )
            return None

        # Compute bounding box from shape points
        bbox = shape.get_bounding_box()
        if bbox is None:
            logger.warning("Shape has no points, cannot create COCO annotation")
            return None

        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min
        area = width * height

        coco_ann: CocoAnnotation = {
            "id": self._generate_annotation_id(),
            "image_id": image_id,
            "category_id": category_id,
            "bbox": [x_min, y_min, width, height],
            "area": area,
            "iscrowd": 0,
        }

        return coco_ann

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

    def save_deleted_shapes(
        self,
        frame_name: str,
        pair: AnnotationPair,
        image_height: int,
        image_width: int,
    ) -> bool:
        """Save deleted shapes to the incorrect predictions file in COCO format."""
        coco_annotations: list[CocoAnnotation] = []

        for shape in pair.shapes:
            coco_ann = self._extract_coco_annotation(shape)

            # If no existing COCO annotation, create one from the shape
            if coco_ann is None:
                coco_ann = self._create_coco_annotation_from_shape(shape, frame_name)
                if coco_ann is None:
                    logger.warning(
                        f"Shape '{shape.label}' (group_id={pair.group_id}) "
                        "could not be converted to COCO annotation, skipping"
                    )
                    continue

            if not self._is_valid_coco_annotation(coco_ann):
                logger.warning(
                    f"Shape '{shape.label}' (group_id={pair.group_id}) "
                    "has invalid COCO annotation (missing required fields), "
                    "skipping"
                )
                continue

            coco_annotations.append(coco_ann)

        return self._save_coco_annotations(
            coco_annotations=coco_annotations,
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
        coco_annotations: list[CocoAnnotation] = []

        for shape in pair.shapes:
            coco_ann = self._extract_coco_annotation(shape)
            if coco_ann and self._is_valid_coco_annotation(coco_ann):
                coco_annotations.append(deepcopy(coco_ann))

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

    def clear_pending_edits(self) -> None:
        """Clear all pending edit captures (e.g., when switching frames)."""
        if self._pending_edits:
            logger.debug(
                f"Clearing {len(self._pending_edits)} pending edits on frame change"
            )
        self._pending_edits.clear()
