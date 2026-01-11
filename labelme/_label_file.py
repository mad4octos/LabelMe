import base64
import builtins
import contextlib
from copy import deepcopy
import io
import json
import os.path as osp
from typing import Optional, Literal

import numpy as np
import numpy.typing as npt
import skimage
import PIL.Image
from loguru import logger
from numpy.typing import NDArray
from supervision.detection.core import Detections
from supervision.dataset.formats.yolo import _polygons_to_masks
from supervision.dataset.utils import  mask_to_rle
from labelme import __version__
from labelme import utils
from labelme._automation import polygon_from_mask
from labelme.labelme_types import CocoAnnotation, OtherData, ShapeDict, ShapeByAnnIdx, CocoRLE
from labelme.coco_dataset import LazyCOCODataset
PIL.Image.MAX_IMAGE_PIXELS = None


@contextlib.contextmanager
def open(name, mode):
    assert mode in ["r", "w"]
    encoding = "utf-8"
    yield builtins.open(name, mode, encoding=encoding)
    return

def compute_polygon_from_mask(
    mask: npt.NDArray[np.bool_], polygon_approx_tolerance=0.008
) -> npt.NDArray[np.float32]:
    """
    Modified from:
    labelme._automation.polygon_from_mask.compute_polygon_from_mask

    Changed polygon approx tolerance default from 0.004 to 0.008 to further simplify the
    recovered polygon.
    """
    contours: npt.NDArray[np.float32] = skimage.measure.find_contours(
        np.pad(mask, pad_width=1)
    )
    if len(contours) == 0:
        logger.warning("No contour found, so returning empty polygon.")
        return np.empty((0, 2), dtype=np.float32)

    contour: npt.NDArray[np.float32] = max(
        contours, key=polygon_from_mask._get_contour_length
    )
    polygon: npt.NDArray[np.float32] = skimage.measure.approximate_polygon(
        coords=contour,
        tolerance=np.ptp(contour, axis=0).max() * polygon_approx_tolerance,
    )
    polygon = np.clip(polygon, (0, 0), (mask.shape[0] - 1, mask.shape[1] - 1))
    polygon = polygon[:-1]  # drop last point that is duplicate of first point

    return polygon[:, ::-1]  # yx -> xy


def convert_coco_detections_to_shapes(
    detections: Detections,
    classes_names: list[str],
    image_annotations: list[CocoAnnotation],
    mask=False,
) -> list[ShapeDict]:
    """Convert COCO detections to labelme shape dictionaries.

    Parameters
    ----------
    detections : Detections
        Detection results containing bounding boxes, class IDs, masks, and
        additional data.
    classes_names : list[str]
        List of class names
    image_annotations : list[CocoAnnotation]
        Original COCO annotations for the image, used to preserve metadata.
        Each annotation is paired with the corresponding detection by index.
    mask : bool, optional
        If True, create polygon shapes from segmentation masks.
        If False, create rectangle shapes from bounding boxes
        (default: False).

    Returns
    -------
    list[ShapeDict]
        List of shape dictionaries compatible with labelme format.
    """

    SHAPE_KEYS: set[str] = {
        "label",
        "points",
        "group_id",
        "shape_type",
        "flags",
        "description",
        "mask",
    }

    results = []

    for i, (x1, y1, x2, y2) in enumerate(detections.xyxy.astype(int).tolist()):
        # Check if original annotation has polygon segmentation (iscrowd=0)
        polygon_points = None
        if mask and image_annotations and i < len(image_annotations):
            orig_ann = image_annotations[i]
            if (
                orig_ann.get("iscrowd") == 0
                and "segmentation" in orig_ann
                and isinstance(orig_ann["segmentation"], list)
            ):
                # Use original polygon points directly (avoid mask approximation)
                # COCO polygon format: [[x1, y1, x2, y2, ...], ...]
                segmentation = orig_ann["segmentation"]
                if (
                    segmentation
                    and isinstance(segmentation[0], list)
                    and len(segmentation[0]) >= 6
                ):
                    # Convert from flat list to numpy array of [x, y] pairs
                    flat_coords = segmentation[0]
                    polygon_points = np.array(
                        [
                            [flat_coords[j], flat_coords[j + 1]]
                            for j in range(0, len(flat_coords), 2)
                        ],
                        dtype=np.float32,
                    )

        # Fallback: compute polygon from mask if not available from original annotation
        if polygon_points is None and mask:
            polygon_points = compute_polygon_from_mask(detections.mask[i][y1:y2, x1:x2])
            polygon_points += [x1, y1]

        # Store original annotation data for reconstruction
        other_data = {}
        if image_annotations and i < len(image_annotations):
            orig_ann = image_annotations[i]
            other_data: OtherData = {
                "annotation_index": i,
                "original_annotation": orig_ann,
            }

        loaded: ShapeDict = ShapeDict(
            label=classes_names[detections.class_id[i]],
            points=polygon_points if mask else [[x1, y1], [x2, y2]],
            shape_type="polygon" if mask else "rectangle",
            flags={},
            description="",
            group_id=int(detections.data["obj_id"][i]),
            mask=None,
            other_data=other_data,
        )

        assert set(loaded.keys()) == SHAPE_KEYS | {"other_data"}
        results.append(loaded)

    return results


def _load_shape_json_obj(shape_json_obj: dict) -> ShapeDict:
    SHAPE_KEYS: set[str] = {
        "label",
        "points",
        "group_id",
        "shape_type",
        "flags",
        "description",
        "mask",
    }

    assert "label" in shape_json_obj, f"label is required: {shape_json_obj}"
    assert isinstance(shape_json_obj["label"], str), (
        f"label must be str: {shape_json_obj['label']}"
    )
    label: str = shape_json_obj["label"]

    assert "points" in shape_json_obj, f"points is required: {shape_json_obj}"
    assert isinstance(shape_json_obj["points"], list), (
        f"points must be list: {shape_json_obj['points']}"
    )
    assert shape_json_obj["points"], f"points must be non-empty: {shape_json_obj}"
    assert all(
        isinstance(point, list)
        and len(point) == 2
        and all(isinstance(xy, (int, float)) for xy in point)
        for point in shape_json_obj["points"]
    ), f"points must be list of [x, y]: {shape_json_obj['points']}"
    points: list[list[float]] = shape_json_obj["points"]

    assert "shape_type" in shape_json_obj, f"shape_type is required: {shape_json_obj}"
    assert isinstance(shape_json_obj["shape_type"], str), (
        f"shape_type must be str: {shape_json_obj['shape_type']}"
    )
    shape_type: str = shape_json_obj["shape_type"]

    flags: dict = {}
    if shape_json_obj.get("flags") is not None:
        assert isinstance(shape_json_obj["flags"], dict), (
            f"flags must be dict: {shape_json_obj['flags']}"
        )
        assert all(
            isinstance(k, str) and isinstance(v, bool)
            for k, v in shape_json_obj["flags"].items()
        ), f"flags must be dict of str to bool: {shape_json_obj['flags']}"
        flags = shape_json_obj["flags"]

    description: str = ""
    if shape_json_obj.get("description") is not None:
        assert isinstance(shape_json_obj["description"], str), (
            f"description must be str: {shape_json_obj['description']}"
        )
        description = shape_json_obj["description"]

    group_id: Optional[int] = None
    if shape_json_obj.get("group_id") is not None:
        assert isinstance(shape_json_obj["group_id"], int), (
            f"group_id must be int: {shape_json_obj['group_id']}"
        )
        group_id = shape_json_obj["group_id"]

    mask: Optional[NDArray[np.bool]] = None
    if shape_json_obj.get("mask") is not None:
        assert isinstance(shape_json_obj["mask"], str), (
            f"mask must be base64-encoded PNG: {shape_json_obj['mask']}"
        )
        mask = utils.img_b64_to_arr(shape_json_obj["mask"]).astype(bool)

    other_data = {k: v for k, v in shape_json_obj.items() if k not in SHAPE_KEYS}

    loaded: ShapeDict = ShapeDict(
        label=label,
        points=points,
        shape_type=shape_type,
        flags=flags,
        description=description,
        group_id=group_id,
        mask=mask,
        other_data=other_data,
    )
    assert set(loaded.keys()) == SHAPE_KEYS | {"other_data"}
    return loaded


class LabelFileError(Exception):
    pass


class LabelFile:
    shapes: list[ShapeDict]
    suffix = ".json"

    def __init__(self, filename=None):
        self.shapes = []
        self.imagePath = None
        self.imageData = None
        if filename is not None:
            self.load(filename)
        self.filename = filename

    @staticmethod
    def load_image_file(filename):
        try:
            image_pil = PIL.Image.open(filename)
        except OSError:
            logger.error(f"Failed opening image file: {filename}")
            return

        # apply orientation to image according to exif
        image_pil = utils.apply_exif_orientation(image_pil)

        with io.BytesIO() as f:
            ext = osp.splitext(filename)[1].lower()
            if ext in [".jpg", ".jpeg"]:
                format = "JPEG"
            else:
                format = "PNG"
            image_pil.save(f, format=format)
            f.seek(0)
            return f.read()

    def load(self, filename):
        keys = [
            "version",
            "imageData",
            "imagePath",
            "shapes",  # polygonal annotations
            "flags",  # image level flags
            "imageHeight",
            "imageWidth",
        ]
        try:
            with open(filename, "r") as f:
                data = json.load(f)

            if data["imageData"] is not None:
                imageData = base64.b64decode(data["imageData"])
            else:
                # relative path from label file to relative path from cwd
                imagePath = osp.join(osp.dirname(filename), data["imagePath"])
                imageData = self.load_image_file(imagePath)
            flags = data.get("flags") or {}
            imagePath = data["imagePath"]
            self._check_image_height_and_width(
                base64.b64encode(imageData).decode("utf-8"),
                data.get("imageHeight"),
                data.get("imageWidth"),
            )
            shapes: list[ShapeDict] = [
                _load_shape_json_obj(shape_json_obj=s) for s in data["shapes"]
            ]
        except Exception as e:
            raise LabelFileError(e)

        otherData = {}
        for key, value in data.items():
            if key not in keys:
                otherData[key] = value

        # Only replace data after everything is loaded.
        self.flags = flags
        self.shapes = shapes
        self.imagePath = imagePath
        self.imageData = imageData
        self.filename = filename
        self.otherData = otherData

    @staticmethod
    def _check_image_height_and_width(imageData, imageHeight, imageWidth):
        img_arr = utils.img_b64_to_arr(imageData)
        if imageHeight is not None and img_arr.shape[0] != imageHeight:
            logger.error(
                "imageHeight does not match with imageData or imagePath, "
                "so getting imageHeight from actual image."
            )
            imageHeight = img_arr.shape[0]
        if imageWidth is not None and img_arr.shape[1] != imageWidth:
            logger.error(
                "imageWidth does not match with imageData or imagePath, "
                "so getting imageWidth from actual image."
            )
            imageWidth = img_arr.shape[1]
        return imageHeight, imageWidth

    @staticmethod
    def _calculate_bbox_from_points(points: list[list[float]]):
        """Calculate COCO format bbox [x, y, width, height] from a list of points."""
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        bbox_x = round(min(xs), 0)
        bbox_y = round(min(ys), 0)
        bbox_width = round(max(xs) - bbox_x, 0)
        bbox_height = round(max(ys) - bbox_y, 0)
        return [bbox_x, bbox_y, bbox_width, bbox_height]

    @staticmethod
    def _get_next_annotation_id(dataset: LazyCOCODataset) -> int:
        """Find the first available annotation ID."""
        existing_ids = {
            ann["id"]
            for anns in dataset.annotations_by_image_id.values()
            for ann in anns
        }
        ann_id = 1
        while ann_id in existing_ids:
            ann_id += 1
        return ann_id

    @staticmethod
    def _labelme_polygon_to_coco_format(
        points: list[list[float]],
        resolution_wh: tuple[int, int],
        iscrowd: Literal[0, 1],
    ) -> tuple[float, CocoRLE | list[list[float]]]:
        """
        Convert polygon points to mask and COCO segmentation format.

        Returns:
            tuple: (area, segmentation) where segmentation is either
                   RLE format dict (iscrowd=1) or polygon format list (iscrowd=0)
        """
        # Convert polygon points to proper format
        segmentation_polygon = [[float(x), float(y)] for x, y in points]

        # Convert polygon to mask
        masks = _polygons_to_masks([np.array(segmentation_polygon)], resolution_wh)
        masks = masks.astype(np.uint8) * 255

        segmentation: CocoRLE | list[list[float]]
        if iscrowd == 1:
            # For crowd annotations, use RLE format
            masks_rle = mask_to_rle(masks[0])
            masks_rle = list(map(int, masks_rle))
            segmentation = {
                "counts": masks_rle,
                "size": list(masks[0].shape[:2]),
            }
        else:
            # For non-crowd annotations, use polygon format
            segmentation = [[coord for point in points for coord in point]]

        area = float(np.sum(masks[0] > 0))

        return area, segmentation

    @staticmethod
    def _process_existing_annotation(
        shape: dict,
        shapes_by_annotation_index: dict[int, ShapeByAnnIdx],
    ) -> None:
        """Add shape from existing COCO annotation to the grouped index."""
        ann_index = shape["annotation_index"]
        original_annotation = shape["original_annotation"]

        if ann_index not in shapes_by_annotation_index:
            shapes_by_annotation_index[ann_index] = {
                "annotation": deepcopy(original_annotation),
                "shapes": [],
            }

        shapes_by_annotation_index[ann_index]["shapes"].append(shape)

    def _create_new_annotation(
        self,
        shape: dict,
        dataset: LazyCOCODataset,
        image_id: int,
        category_name_to_id: dict[str, int],
        resolution_wh: tuple[int, int],
        shapes_by_annotation_index: dict[int, ShapeByAnnIdx],
    ) -> None:
        """Create a new COCO annotation from a user-drawn shape."""
        label = shape["label"]
        points = shape["points"]
        shape_type = shape["shape_type"]
        group_id = shape.get("group_id")

        # Validate category
        if label not in category_name_to_id:
            logger.warning(f"Label '{label}' not found in COCO categories, skipping")
            return

        # Validate shape type
        if shape_type not in ["rectangle", "polygon"]:
            logger.warning(
                f"Shape type '{shape_type}' not supported for COCO export, skipping"
            )
            return

        # Create base annotation
        bbox = self._calculate_bbox_from_points(points)
        bbox_x, bbox_y, bbox_width, bbox_height = bbox
        ann_id = self._get_next_annotation_id(dataset)

        coco_annotation = CocoAnnotation(
            id=ann_id,
            image_id=image_id,
            category_id=category_name_to_id[label],
            area=round(bbox_width * bbox_height, 0),
            bbox=[bbox_x, bbox_y, bbox_width, bbox_height],
            iscrowd=0,
            attributes={"ObjID": group_id if group_id is not None else -1},
        )

        # Add segmentation for polygons
        if shape_type == "polygon":
            area, segmentation = self._labelme_polygon_to_coco_format(
                points, resolution_wh, iscrowd=0
            )
            coco_annotation["segmentation"] = segmentation
            coco_annotation["area"] = area

        shapes_by_annotation_index[ann_id] = {
            "annotation": coco_annotation,
            "shapes": [shape],
        }

    def _update_annotation_from_shapes(
        self,
        ann_data: ShapeByAnnIdx,
        resolution_wh: tuple[int, int],
    ) -> CocoAnnotation:
        """Update COCO annotation from edited Labelme shapes (bbox and mask)."""
        annotation = ann_data["annotation"]

        # Find both rectangle and polygon shapes
        shapes_by_type = {shape["shape_type"]: shape for shape in ann_data["shapes"]}
        rectangle_shape = shapes_by_type.get("rectangle")
        polygon_shape = shapes_by_type.get("polygon")

        # Update bbox from rectangle shape (if exists)
        if rectangle_shape:
            annotation["bbox"] = self._calculate_bbox_from_points(
                rectangle_shape["points"]
            )

        # Update segmentation from polygon shape (if exists)
        if polygon_shape and "segmentation" in annotation:
            points = polygon_shape["points"]
            iscrowd = 0
            area, segmentation = self._labelme_polygon_to_coco_format(
                points, resolution_wh, iscrowd=iscrowd
            )
            annotation["area"] = area
            annotation["segmentation"] = segmentation
            annotation["iscrowd"] = iscrowd

        return annotation

    def save(
        self,
        filename,
        shapes,
        imagePath,
        imageHeight,
        imageWidth,
        imageData=None,
        otherData=None,
        flags=None,
    ):
        if imageData is not None:
            imageData = base64.b64encode(imageData).decode("utf-8")
            imageHeight, imageWidth = self._check_image_height_and_width(
                imageData, imageHeight, imageWidth
            )
        if otherData is None:
            otherData = {}
        if flags is None:
            flags = {}
        data = dict(
            version=__version__,
            flags=flags,
            shapes=shapes,
            imagePath=imagePath,
            imageData=imageData,
            imageHeight=imageHeight,
            imageWidth=imageWidth,
        )
        for key, value in otherData.items():
            assert key not in data
            data[key] = value
        try:
            with open(filename, "w") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self.filename = filename
        except Exception as e:
            raise LabelFileError(e)

    def _sync_labelme_shapes_to_coco_dataset(
        self,
        dataset: LazyCOCODataset,
        image_id: int,
        shapes: list[dict],
        im_height: int,
        im_width: int,
        imageData,
    ):
        """Convert Labelme shapes to COCO annotations and update the dataset."""
        imageData = base64.b64encode(imageData).decode("utf-8")
        im_height, im_width = self._check_image_height_and_width(
            imageData, im_height, im_width
        )

        resolution_wh = (im_width, im_height)

        category_name_to_id = {cat["name"]: cat["id"] for cat in dataset.categories}

        # Group Labelme shapes by annotation index.
        # Each COCO annotation produces two Labelme shapes: rectangle (bbox) and
        # polygon (mask). Both Labelme shapes share the same annotation_index and
        # represent different aspects of the same COCO annotation. We group them to
        # reconstruct the COCO annotation while allowing independent editing of bbox
        # and mask in Labelme.
        shapes_by_annotation_index: dict[int, ShapeByAnnIdx] = {}
        for shape in shapes:
            original_annotation = shape.get("original_annotation")

            # Preserve existing COCO annotations (loaded from dataset)
            if original_annotation is not None:
                self._process_existing_annotation(shape, shapes_by_annotation_index)

            # Create new COCO annotation from user-drawn Labelme shape
            else:
                self._create_new_annotation(
                    shape,
                    dataset,
                    image_id,
                    category_name_to_id,
                    resolution_wh,
                    shapes_by_annotation_index,
                )

        # Reconstruct COCO annotations from Labelme shapes.
        # Labelme uses separate rectangle (bbox) and polygon (mask) shapes,
        # while COCO uses a single annotation with bbox + segmentation fields.
        # User edits to rectangles/polygons in Labelme are synced back to COCO.
        new_annotations: list[CocoAnnotation] = []
        for ann_index in sorted(shapes_by_annotation_index.keys()):
            ann_data = shapes_by_annotation_index[ann_index]
            annotation = self._update_annotation_from_shapes(ann_data, resolution_wh)
            new_annotations.append(annotation)

        # Update the dataset's annotations for this image
        dataset.annotations_by_image_id[image_id] = new_annotations

    @staticmethod
    def is_label_file(filename):
        return osp.splitext(filename)[1].lower() == LabelFile.suffix
