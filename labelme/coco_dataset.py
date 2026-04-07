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

    def __init__(self, images_directory_path: Path, annotations_file_path: Path):
        """ """
        self.images_directory_path = images_directory_path
        self.annotations_file_path = annotations_file_path

        self.coco_data: CocoFile = read_json_file(file_path=annotations_file_path)
        self._images = self.coco_data["images"]
        self.categories = self.coco_data["categories"]
        self.classes = coco_categories_to_classes(coco_categories=self.categories)

        self.category_id_to_name: dict[int, str] = {
            cat["id"]: cat["name"] for cat in self.categories
        }

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
        self.validation_results = self.verify()
        self._log_validation_warnings()

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

        coco_export = {
            "images": self._images,
            "annotations": all_annotations,
            "categories": self.categories,
        }

        save_json_file(file_path=output_path, data=coco_export)

    def verify(self) -> dict:
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
        """
        results = {
            "valid": True,
            "duplicate_annotation_ids": [],
            "duplicate_image_ids": [],
            "orphan_annotations": [],
            "missing_fields": [],
        }

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
        ):
            results["valid"] = False

        return results
