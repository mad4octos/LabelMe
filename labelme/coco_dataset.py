# Standard Library imports
from pathlib import Path

# External imports
import numpy as np
import numpy.typing as npt
from loguru import logger
from supervision.dataset.formats.coco import coco_categories_to_classes
from supervision.dataset.formats.coco import group_coco_annotations_by_image_id
from supervision.dataset.utils import rle_to_mask
from supervision.utils.file import read_json_file
from supervision.utils.file import save_json_file

# Local imports
from labelme.labelme_types import CocoAnnotation
from labelme.labelme_types import CocoFile


def extract_labelme_polygons_from_coco_annotation(
    annotation: CocoAnnotation,
) -> list[npt.NDArray[np.float32]]:
    """Extract polygon arrays from a COCO annotation with polygon segmentation.

    Only works for non-crowd annotations (iscrowd=0) where the segmentation
    is in polygon format (list of flat coordinate lists). Returns an empty list
    for crowd annotations or Compressed RLE segmentation.

    Parameters
    ----------
    annotation : CocoAnnotation
        A single COCO annotation dictionary.

    Returns
    -------
    list[npt.NDArray[np.float32]]
        List of polygon arrays, each of shape (N, 2) with xy coordinates.
    """
    iscrowd = annotation.get("iscrowd")
    if iscrowd != 0:
        return []

    segmentation = annotation.get("segmentation")
    if not isinstance(segmentation, list):
        return []

    polygons = []
    for flat_coords in segmentation:
        if isinstance(flat_coords, list) and len(flat_coords) >= 6:
            polygon_points = np.array(
                [
                    [flat_coords[j], flat_coords[j + 1]]
                    for j in range(0, len(flat_coords), 2)
                ],
                dtype=np.float32,
            )
            polygons.append(polygon_points)

    return polygons


def extract_labelme_polygons_from_coco_rle_annotation(
    annotation: CocoAnnotation,
) -> list[npt.NDArray[np.float32]]:
    """Extract polygons from a COCO annotation with Compressed RLE segmentation.

    Decodes the Compressed RLE mask and recovers an approximate polygon
    using compute_polygon_from_mask.

    Parameters
    ----------
    annotation : CocoAnnotation
        A single COCO annotation dictionary with Compressed RLE segmentation.

    Returns
    -------
    list[npt.NDArray[np.float32]]
        List of polygon arrays, each of shape (N, 2) with xy coordinates.
    """
    # Local import to avoid circular dependency with _label_file
    from labelme._label_file import compute_polygon_from_mask

    segmentation = annotation.get("segmentation")
    if (
        not isinstance(segmentation, dict)
        or "counts" not in segmentation
        or "size" not in segmentation
    ):
        return []

    rle = np.array(segmentation["counts"])
    h, w = segmentation["size"]
    mask = rle_to_mask(rle=rle, resolution_wh=(w, h))
    polygon = compute_polygon_from_mask(mask)
    if polygon.size > 0:
        return [polygon]

    return []


class LazyCOCODataset:
    """ """

    def __init__(
        self,
        images_directory_path: Path,
        annotations_file_path: Path,
        expected_category_names: list[str] | None = None,
    ):
        """ """
        self.images_directory_path = images_directory_path
        self.annotations_file_path = annotations_file_path

        self.coco_data: CocoFile = read_json_file(file_path=annotations_file_path)
        self._images = self.coco_data["images"]
        self.categories = self.coco_data["categories"]

        self.annotations_by_image_id: dict[int, list[CocoAnnotation]] = (
            group_coco_annotations_by_image_id(self.coco_data["annotations"])
        )

        self.image_id_by_filename: dict[str, int] = {
            img["file_name"]: img["id"] for img in self._images
        }

        self.image_filepaths: list[Path] = [
            images_directory_path / self._images[i]["file_name"]
            for i in range(len(self))
        ]

        # Verify dataset integrity at load time
        self.validation_results = self.verify(expected_category_names)
        self._log_validation_warnings()

    @property
    def category_id_to_name(self) -> dict[int, str]:
        return {cat["id"]: cat["name"] for cat in self.categories}

    @property
    def category_name_to_id(self) -> dict[str, int]:
        return {cat["name"]: cat["id"] for cat in self.categories}

    def _build_category_remap(self, expected_labels: list[str]) -> dict[int, int]:
        """Return a mapping of current category IDs to expected IDs for mismatched
        categories.

        Returns an empty dict if all category IDs already match the expected labels.
        """
        expected_cat_to_id = {cat: i + 1 for i, cat in enumerate(expected_labels)}

        curr_to_expected: dict[int, int] = {}
        for curr_cat_id, curr_cat_name in self.category_id_to_name.items():
            if (expected_cat_id := expected_cat_to_id.get(curr_cat_name)) is not None:
                if curr_cat_id != expected_cat_id:
                    curr_to_expected[curr_cat_id] = expected_cat_id

        return curr_to_expected

    def _enforce_categories(self, expected_labels: list[str]) -> dict[int, int]:
        """Normalize categories to match the expected label list.

        Ensures every expected label is present with its canonical ID
        (1-indexed by position in the config list). Any annotation
        category_ids that differ from the expected IDs are remapped
        simultaneously so no intermediate collision can occur.
        Categories that exist in the file but are not in the config
        are preserved as-is.
        """
        expected_cat_to_id = {cat: i + 1 for i, cat in enumerate(expected_labels)}
        curr_to_expected = self._build_category_remap(expected_labels)

        if curr_to_expected:
            for annotations in self.annotations_by_image_id.values():
                for ann in annotations:
                    ann["category_id"] = curr_to_expected.get(
                        ann["category_id"], ann["category_id"]
                    )

        # Rebuild categories: expected labels first (with canonical IDs),
        # then any extra categories not in the config
        expected_names = set(expected_labels)
        extra_cats = [c for c in self.categories if c["name"] not in expected_names]
        self.categories = [
            {"id": new_id, "name": name, "supercategory": ""}
            for name, new_id in expected_cat_to_id.items()
        ] + extra_cats

        # Assert unique category names and ids
        names = [c["name"] for c in self.categories]
        ids = [c["id"] for c in self.categories]
        assert len(set(names)) == len(names), f"Duplicate category names: {names}"
        assert len(set(ids)) == len(ids), f"Duplicate category ids: {ids}"

        return curr_to_expected
 

    def _log_validation_warnings(self) -> None:
        """Log warnings if validation issues are found."""
        if not self.validation_results["valid"]:
            warning_msg = self.get_validation_warning_message()
            if warning_msg:
                logger.warning(warning_msg)
        else:
            logger.info("Dataset consistency validated")

    def get_validation_warning_message(self) -> str | None:
        """Get a formatted warning message for validation issues."""
        if self.validation_results["valid"]:
            return None
        messages = []
        if self.validation_results["duplicate_annotation_ids"]:
            ids = self.validation_results["duplicate_annotation_ids"]
            messages.append(f"Duplicate annotation IDs: {ids}")
        if self.validation_results["duplicate_image_ids"]:
            ids = self.validation_results["duplicate_image_ids"]
            messages.append(f"Duplicate image IDs: {ids}")
        if self.validation_results["orphan_annotations"]:
            ids = self.validation_results["orphan_annotations"]
            messages.append(f"Orphan annotations: {ids}")
        if self.validation_results["missing_fields"]:
            fields = self.validation_results["missing_fields"]
            messages.append(f"Annotations with missing fields: {fields}")
        if self.validation_results["category_id_remaps"]:
            remaps = self.validation_results["category_id_remaps"]
            messages.append(f"Category IDs need remapping: {remaps}")
        if self.validation_results["missing_categories"]:
            missing = self.validation_results["missing_categories"]
            messages.append(f"Missing categories: {missing}")
        return (
            f"COCO dataset validation failed for "
            f"{self.annotations_file_path}:\n" + "\n".join(messages)
        )

    def __len__(self) -> int:
        return len(self.coco_data["images"])

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, idx) -> list[CocoAnnotation]:
        image_id = self._images[idx]["id"]
        return self.annotations_by_image_id.get(image_id, [])

    def _clamp_bboxes(self, annotations: list[CocoAnnotation]) -> None:
        """Clamp all bbox coordinates in-place to their image bounds."""
        image_wh: dict[int, tuple[int, int]] = {
            img["id"]: (img["width"], img["height"]) for img in self._images
        }
        for ann in annotations:
            if "bbox" not in ann:
                continue
            wh = image_wh.get(ann["image_id"])
            if wh is None:
                continue
            image_width, image_height = float(wh[0]), float(wh[1])
            x, y, bbox_width, bbox_height = ann["bbox"]
            x = max(0.0, x)
            y = max(0.0, y)
            bbox_width = max(0.0, min(image_width - x, bbox_width))
            bbox_height = max(0.0, min(image_height - y, bbox_height))
            ann["bbox"] = [x, y, bbox_width, bbox_height]

    def export_annotations(self, output_path: Path | None = None) -> None:
        """
        Export the current dataset to COCO format.

        Parameters
        ----------
        output_path : Path | None
            Where to save the COCO JSON. Defaults to original annotations_file_path.
        """

        if output_path is None:
            output_path = self.annotations_file_path

        # Rebuild annotations list from annotations_by_image_id
        all_annotations = []
        for image_id in sorted(self.annotations_by_image_id.keys()):
            all_annotations.extend(self.annotations_by_image_id[image_id])

        self._clamp_bboxes(all_annotations)

        coco_export = {
            "images": self._images,
            "annotations": all_annotations,
            "categories": self.categories,
        }

        save_json_file(file_path=output_path, data=coco_export)

    def verify(self, expected_category_names: list[str] | None = None) -> dict:
        """
        Verify the COCO dataset for common issues.

        Returns
        -------
        dict
            Dictionary containing verification results with keys:
            - 'valid': bool, True if no issues found
            - 'duplicate_annotation_ids': list of duplicate annotation IDs
            - 'duplicate_image_ids': list of duplicate image IDs
            - 'orphan_annotations': list of annotations with invalid image_id
            - 'missing_fields': list of annotations missing required fields
            - 'category_id_remaps': dict mapping old category IDs to new ones
              (empty if no remapping was needed)
            - 'missing_categories': list of expected category names that are
              absent from the COCO file (empty if all expected labels exist)
        """
        results = {
            "valid": True,
            "duplicate_annotation_ids": [],
            "duplicate_image_ids": [],
            "orphan_annotations": [],
            "missing_fields": [],
            "category_id_remaps": {},
            "missing_categories": [],
        }

        if expected_category_names:
            results["category_id_remaps"] = self._build_category_remap(
                expected_category_names
            )
            existing_names = set(self.category_id_to_name.values())
            results["missing_categories"] = [
                name for name in expected_category_names if name not in existing_names
            ]

        annotations = self.coco_data["annotations"]
        images = self.coco_data["images"]

        # Check for duplicate IDs
        def find_duplicates(items, key):
            seen, duplicates = set(), []
            for item in items:
                if item[key] in seen:
                    duplicates.append(item[key])
                seen.add(item[key])
            return duplicates

        results["duplicate_annotation_ids"] = find_duplicates(annotations, "id")
        results["duplicate_image_ids"] = find_duplicates(images, "id")

        # Check for orphan annotations (referencing non-existent images)
        valid_image_ids = {img["id"] for img in images}
        for ann in annotations:
            if ann["image_id"] not in valid_image_ids:
                results["orphan_annotations"].append(ann["id"])

        # Check for missing required fields in annotations
        required_fields = [
            "id",
            "image_id",
            "category_id",
            "bbox",
            "segmentation",
            "iscrowd",
            "attributes",
        ]
        for ann in annotations:
            missing = [f for f in required_fields if f not in ann]
            if missing:
                results["missing_fields"].append(
                    {
                        "annotation_id": ann.get("id", "unknown"),
                        "missing": missing,
                        "image_id": ann.get("image_id", "unknown"),
                    }
                )

        # Set valid flag
        if (
            results["duplicate_annotation_ids"]
            or results["duplicate_image_ids"]
            or results["orphan_annotations"]
            or results["missing_fields"]
            or results["category_id_remaps"]
            or results["missing_categories"]
        ):
            results["valid"] = False

        return results
