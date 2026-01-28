# Standard Library imports
from pathlib import Path

# External imports
import cv2
from loguru import logger
import numpy as np
import numpy.typing as npt
from supervision.dataset.formats.coco import build_coco_class_index_mapping
from supervision.dataset.formats.coco import coco_categories_to_classes
from supervision.dataset.formats.coco import group_coco_annotations_by_image_id
from supervision.dataset.utils import map_detections_class_id
from supervision.dataset.utils import rle_to_mask
from supervision.detection.utils.converters import polygon_to_mask
from supervision.detection.core import Detections
from supervision.utils.file import read_json_file
from supervision.utils.file import save_json_file

# Local imports
from labelme.labelme_types import CocoAnnotation
from labelme.labelme_types import CocoFile


def coco_annotations_to_detections(
    image_annotations: list[dict],
    resolution_wh: tuple[int, int],
    with_masks: bool,
) -> Detections:
    """
    Convert COCO annotations to Detections object.

    Parameters
    ----------
    image_annotations : list[dict]
        List of COCO annotation dictionaries for a single image.
    resolution_wh : tuple[int, int]
        Image resolution as (width, height).
    with_masks : bool
        If True, convert COCO segmentation data (polygons or RLE) to binary masks.
        If False, only bounding boxes are included (mask field will be None).

    Returns
    -------
    Detections
        Detection object with bounding boxes, class IDs, masks (if requested),
        and additional data (iscrowd, area, obj_id).

    Notes
    -----
    Modified from supervision.dataset.formats.coco.coco_annotations_to_detections
    """
    if not image_annotations:
        return Detections.empty()

    class_ids = [
        image_annotation["category_id"] for image_annotation in image_annotations
    ]
    xyxy = [image_annotation["bbox"] for image_annotation in image_annotations]
    xyxy = np.asarray(xyxy)
    xyxy[:, 2:4] += xyxy[:, 0:2]

    iscrowd = [image_annotation["iscrowd"] for image_annotation in image_annotations]
    area = [image_annotation["area"] for image_annotation in image_annotations]
    obj_id = [
        image_annotation["attributes"]["ObjID"]
        for image_annotation in image_annotations
    ]
    data = dict(
        iscrowd=np.asarray(iscrowd, dtype=int),
        area=np.asarray(area, dtype=float),
        obj_id=np.asarray(obj_id, dtype=int),
    )

    if with_masks:
        mask = coco_annotations_to_masks(
            image_annotations=image_annotations, resolution_wh=resolution_wh
        )
    else:
        mask = None

    return Detections(
        class_id=np.asarray(class_ids, dtype=int), xyxy=xyxy, mask=mask, data=data
    )


def coco_annotations_to_masks(
    image_annotations: list[dict], resolution_wh: tuple[int, int]
) -> npt.NDArray[np.bool_]:
    masks = []
    for annotation in image_annotations:
        if annotation["iscrowd"]:
            assert isinstance(annotation["segmentation"], dict)
            rle = np.array(annotation["segmentation"]["counts"])
            mask = rle_to_mask(rle=rle, resolution_wh=resolution_wh)
            masks.append(mask)
        else:
            if ("segmentation" not in annotation) or not isinstance(
                annotation["segmentation"], list
            ):
                # Create empty mask for annotations without valid segmentation
                mask = np.zeros((resolution_wh[1], resolution_wh[0]), dtype=np.bool_)
            else:
                polygon = np.reshape(
                    np.asarray(annotation["segmentation"], dtype=np.int32),
                    (-1, 2),
                )
                mask = polygon_to_mask(polygon=polygon, resolution_wh=resolution_wh)
            masks.append(mask)
    return np.array(masks, dtype=bool)

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

        self.class_index_mapping = build_coco_class_index_mapping(
            coco_categories=self.categories, target_classes=self.classes
        )

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

    def __getitem__(self, idx):
        """
        Modified from:
        https://github.com/roboflow/supervision/blob/a61440ee0b7d8dec9aff2896c78f03fb4f424c49/supervision/dataset/formats/coco.py#L212
        """

        coco_image = self._images[idx]
        image_name = coco_image["file_name"]
        image_width = coco_image["width"]
        image_height = coco_image["height"]
        image_id = coco_image["id"]

        image_annotations = self.annotations_by_image_id.get(image_id, [])

        detections = coco_annotations_to_detections(
            image_annotations=image_annotations,
            resolution_wh=(image_width, image_height),
            with_masks=True,
        )

        annotation = map_detections_class_id(
            source_to_target_mapping=self.class_index_mapping, detections=detections
        )

        image = cv2.imread(str(self.images_directory_path / image_name))

        return image, annotation

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
                    {"annotation_id": ann.get("id", "unknown"), "missing": missing}
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
