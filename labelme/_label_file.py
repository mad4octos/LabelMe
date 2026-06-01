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
from supervision.dataset.formats.yolo import _polygons_to_masks
from supervision.dataset.utils import  mask_to_rle
from labelme import __version__
from labelme import utils
from labelme._automation import polygon_from_mask
from labelme.labelme_types import CocoAnnotation, ShapeDict, CompressedRLE, CocoPolygon, ShapePolygon
from labelme.coco_dataset import extract_labelme_polygons_from_coco_annotation
from labelme.coco_dataset import extract_labelme_polygons_from_coco_rle_annotation
from labelme.coco_dataset import LazyCOCODataset
PIL.Image.MAX_IMAGE_PIXELS = None

MISSING_OBJ_ID = -1


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


def convert_coco_annotations_to_shapes(
    image_annotations: list[CocoAnnotation],
    category_id_to_name: dict[int, str],
    rectangle: bool = False,
    gt_location: bool = False,
) -> list[ShapeDict]:
    """Convert COCO annotations to labelme shape dictionaries.
    Used to load COCO annotations into Labelme.

    Parameters
    ----------
    image_annotations : list[CocoAnnotation]
        COCO annotations for the image.
    category_id_to_name : dict[int, str]
        Mapping from COCO category ID to class name.
    rectangle : bool, optional
        If True, create rectangle shapes from bounding boxes.
        If False, create polygon shapes from segmentation
        (default: False).
    gt_location : bool, optional
        If True, create point shapes from the ``gt_location``
        attribute (closest known ground-truth location).
        (default: False).

    Returns
    -------
    list[ShapeDict]
        List of shape dictionaries compatible with labelme format.
    """

    def _make_shape(annotation: CocoAnnotation, points, shape_type) -> ShapeDict:
        return ShapeDict(
            label=category_id_to_name[annotation["category_id"]],
            points=points,
            shape_type=shape_type,
            flags={},
            description="",
            group_id=int(annotation["attributes"]["ObjID"]),
            mask=None,
            other_data={"original_annotation": annotation},
        )

    results: list[ShapeDict] = []
    for annotation in image_annotations:
        if gt_location:
            attrs = annotation.get("attributes", {})
            loc = attrs.get("gt_location")
            if loc is not None:
                GT_CIRCLE_RADIUS = 10
                cx, cy = loc[0], loc[1]
                shape = _make_shape(
                    annotation,
                    [[cx, cy], [cx + GT_CIRCLE_RADIUS, cy]],
                    "circle",
                )
                parts = []
                gt_obj_id = attrs.get("gt_obj_id")
                if gt_obj_id is not None:
                    parts.append(f"ObjID={gt_obj_id} | ")
                gt_extracted = attrs.get("gt_frame_extracted")
                if gt_extracted is not None:
                    parts.append(f"Fr extr={gt_extracted} | ")
                gt_original = attrs.get("gt_frame_original")
                if gt_original is not None:
                    parts.append(f"Fr orig={gt_original}")
                if parts:
                    shape["description"] = " ".join(parts)
                results.append(shape)
            continue

        # Convert COCO bbox [x, y, w, h] to [x1, y1, x2, y2]
        bx, by, bw, bh = annotation["bbox"]
        x1, y1, x2, y2 = int(bx), int(by), int(bx + bw), int(by + bh)

        if rectangle:
            results.append(_make_shape(annotation, [[x1, y1], [x2, y2]], "rectangle"))
            continue

        polygons = extract_labelme_polygons_from_coco_annotation(annotation)
        if not polygons:
            polygons = extract_labelme_polygons_from_coco_rle_annotation(annotation)
        if not polygons:
            logger.warning(
                f"Annotation {annotation.get('id')}: no polygons found in segmentation"
            )

        for polygon_points in polygons:
            results.append(_make_shape(annotation, polygon_points, "polygon"))

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
    def _calculate_bbox_from_points(
        points: ShapePolygon,
        image_wh: tuple[int, int],
    ) -> list[float]:
        """Calculate COCO format bbox [x, y, width, height] from a list of points."""
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        w = float(image_wh[0])
        h = float(image_wh[1])
        bbox_x = max(0.0, round(min(xs), 0))
        bbox_y = max(0.0, round(min(ys), 0))
        bbox_width = max(0.0, min(w - bbox_x, round(max(xs) - bbox_x, 0)))
        bbox_height = max(0.0, min(h - bbox_y, round(max(ys) - bbox_y, 0)))
        return [bbox_x, bbox_y, bbox_width, bbox_height]

    @staticmethod
    def _max_annotation_id(dataset: LazyCOCODataset) -> int:
        """Return the current maximum annotation ID in the dataset (0 if empty)."""
        return max(
            (
                ann["id"]
                for anns in dataset.annotations_by_image_id.values()
                for ann in anns
            ),
            default=0,
        )

    @staticmethod
    def _convert_polygons_to_coco_segmentation(
        list_of_polygons: list[ShapePolygon],
        resolution_wh: tuple[int, int],
        iscrowd: Literal[0, 1],
    ) -> tuple[float, CompressedRLE | list[CocoPolygon]]:
        """
        Convert polygon points to mask and COCO segmentation format.

        Returns:
            tuple: (area, segmentation) where segmentation is either
                   RLE (compressed) format dict (iscrowd=1) or polygon format list (iscrowd=0)
        """
        masks: npt.NDArray[np.bool_] = _polygons_to_masks(
            [np.array(points) for points in list_of_polygons], resolution_wh
        )
        collapsed_mask = np.max(masks, axis=0)
        area = float(np.sum(collapsed_mask > 0))

        segmentation: CompressedRLE | list[CocoPolygon]
        # For non-crowd annotations, use polygon format
        if iscrowd == 0:
            segmentation = [
                [coord for point in polygon for coord in point]
                for polygon in list_of_polygons
            ]
        # For crowd annotations, use compressed RLE format
        elif iscrowd == 1:
            compressed_counts = mask_to_rle(collapsed_mask)
            compressed_counts = list(map(int, compressed_counts))
            # pycocotools can also generate RLE masks, but in uncompressed format, with encoded counts
            # (a bytes string in LEB128 variable-length encoding)
            # segmentation = pycocotools.mask.encode(
            #     np.asfortranarray(collapsed_mask.astype(np.uint8))
            # )
            segmentation = {
                "counts": compressed_counts,
                "size": list(collapsed_mask.shape[:2]),
            }
        else:
            raise Exception("`iscrowd` expected values: 0 or 1")

        return area, segmentation

    @staticmethod
    def _split_shapes_by_type(
        group_shapes: list[dict],
    ) -> tuple[dict | None, list[dict]]:
        """Split a group of shapes into a rectangle and polygon shapes.

        Returns:
            Tuple of (rectangle_shape or None, list of polygon shapes)
        """
        rectangles = [s for s in group_shapes if s["shape_type"] == "rectangle"]
        polygon_shapes = [s for s in group_shapes if s["shape_type"] == "polygon"]

        if len(rectangles) > 1:
            msg = "Duplicate rectangle shape found in group. Data was not saved."
            logger.error(msg)
            raise LabelFileError(msg)

        rectangle_shape = rectangles[0] if rectangles else None
        return rectangle_shape, polygon_shapes

    @staticmethod
    def _extract_float_polygons(polygon_shapes: list[dict]) -> list[ShapePolygon]:
        """Convert polygon shape dicts to float coordinate lists."""
        return [
            [[float(x), float(y)] for x, y in ps["points"]] for ps in polygon_shapes
        ]

    def _create_annotation_from_shape_group(
        self,
        group_shapes: list[dict],
        image_id: int,
        category_name_to_id: dict[str, int],
        resolution_wh: tuple[int, int],
        ann_id: int,
    ) -> CocoAnnotation:
        """Create a COCO annotation from a group of shapes (polygon + rectangle).

        Args:
            group_shapes: List of shapes with the same group_id (polygon and rectangle)

        Returns:
            The created COCO annotation.

        Raises:
            LabelFileError: If no polygon is found or label is not in categories.
        """
        rectangle_shape, polygon_shapes = self._split_shapes_by_type(group_shapes)

        # We need at least a polygon to create a valid COCO annotation
        if not polygon_shapes:
            msg = "No polygon found in shape group. Data was not saved."
            logger.warning(msg)
            raise LabelFileError(msg)

        label = polygon_shapes[0]["label"]
        group_id = polygon_shapes[0].get("group_id")

        # Validate category
        if label not in category_name_to_id:
            msg = f"Label '{label}' not found in COCO categories. Data was not saved."
            logger.warning(msg)
            raise LabelFileError(msg)

        # Calculate bbox from rectangle shape if available, else from polygons
        if rectangle_shape is not None:
            bbox = self._calculate_bbox_from_points(
                rectangle_shape["points"], resolution_wh
            )
        else:
            # Combine all polygon points to compute bbox
            all_points = [p for ps in polygon_shapes for p in ps["points"]]
            bbox = self._calculate_bbox_from_points(all_points, resolution_wh)

        bbox_x, bbox_y, bbox_width, bbox_height = bbox

        # Calculate segmentation from polygon(s)
        iscrowd = 0
        list_of_polygons = self._extract_float_polygons(polygon_shapes)

        area, segmentation = self._convert_polygons_to_coco_segmentation(
            list_of_polygons, resolution_wh, iscrowd=iscrowd
        )

        coco_annotation = CocoAnnotation(
            id=ann_id,
            image_id=image_id,
            category_id=category_name_to_id[label],
            area=area,
            bbox=[bbox_x, bbox_y, bbox_width, bbox_height],
            iscrowd=iscrowd,
            segmentation=segmentation,
            attributes={"ObjID": group_id if group_id is not None else MISSING_OBJ_ID},
        )

        return coco_annotation

    def _update_annotation_from_shapes(
        self,
        annotation: CocoAnnotation,
        group_shapes: list[dict],
        resolution_wh: tuple[int, int],
        category_name_to_id: dict[str, int],
    ) -> CocoAnnotation:
        """Update COCO annotation from edited Labelme shapes (bbox and polygon(s)).

        If both polygon and rectangle shapes exist, the rectangle bbox is computed
        from the polygon points to ensure it stays centered around the mask.
        """
        rectangle_shape, polygon_shapes = self._split_shapes_by_type(group_shapes)

        # Update bbox from rectangle shape (if exists)
        if rectangle_shape:
            annotation["bbox"] = self._calculate_bbox_from_points(
                rectangle_shape["points"], resolution_wh
            )

        # Update or add segmentation from polygon shapes
        if polygon_shapes:
            iscrowd = 0
            list_of_polygons = self._extract_float_polygons(polygon_shapes)

            area, segmentation = self._convert_polygons_to_coco_segmentation(
                list_of_polygons, resolution_wh, iscrowd=iscrowd
            )

            annotation["area"] = area
            annotation["segmentation"] = segmentation
            annotation["iscrowd"] = iscrowd
        
        # Polygon was deleted: remove segmentation from annotation
        elif "segmentation" in annotation:
            del annotation["segmentation"]
            del annotation["iscrowd"]

        # Update ObjID and category_id from edited shape (use polygon preferentially)
        shape_for_updates = polygon_shapes[0] if polygon_shapes else rectangle_shape
        if shape_for_updates is not None:
            # Update category_id from edited label
            label = shape_for_updates.get("label")
            # Update ObjID from edited group_id
            group_id = shape_for_updates.get("group_id")
            if group_id is not None:
                annotation.setdefault("attributes", {})["ObjID"] = group_id

            if label is not None and label in category_name_to_id:
                annotation["category_id"] = category_name_to_id[label]

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

    @staticmethod
    def _group_shapes_by_group_id(
        shapes: list[dict],
    ) -> dict[int, tuple[CocoAnnotation | None, list[dict]]]:
        """Group Labelme shapes by group_id.

        Each COCO annotation produces two Labelme shapes: rectangle (bbox) and
        polygon (mask). Both Labelme shapes share the same group_id (ObjID) and
        represent different aspects of the same COCO annotation. We group them to
        reconstruct the COCO annotation while allowing independent editing of bbox
        and mask in Labelme.
        """
        shapes_by_group_id: dict[int, tuple[CocoAnnotation | None, list[dict]]] = {}

        for shape in shapes:
            if (group_id := shape.get("group_id")) is None:
                msg = "Shape without ObjId!"
                logger.error(msg)
                raise LabelFileError(msg)

            if group_id in shapes_by_group_id:
                shapes_by_group_id[group_id][1].append(shape)
            else:
                # Preserve existing COCO annotation if available,
                # otherwise it will be created later from the shapes.
                annotation = shape.get("original_annotation")
                annotation = deepcopy(annotation) if annotation else None
                shapes_by_group_id[group_id] = (annotation, [shape])

        return shapes_by_group_id

    def _rebuild_coco_annotations(
        self,
        data_by_group_id: dict[int, tuple[CocoAnnotation | None, list[dict]]],
        dataset: LazyCOCODataset,
        image_id: int,
        category_name_to_id: dict[str, int],
        resolution_wh: tuple[int, int],
    ) -> list[CocoAnnotation]:
        """Reconstruct COCO annotations from grouped Labelme shapes.

        Labelme uses separate rectangle (bbox) and polygon (mask) shapes,
        while COCO uses a single annotation with bbox + segmentation fields.
        User edits to rectangles/polygons in Labelme are synced back to COCO.
        """
        next_id = self._max_annotation_id(dataset) + 1
        new_annotations: list[CocoAnnotation] = []
        for group_id in sorted(data_by_group_id):
            annotation, group_shapes = data_by_group_id[group_id]

            # The Labelme shapes must be new (created by the user during this session) because
            # they had no COCO annotation attached.
            if annotation is None:
                annotation = self._create_annotation_from_shape_group(
                    group_shapes,
                    image_id,
                    category_name_to_id,
                    resolution_wh,
                    ann_id=next_id,
                )
                next_id += 1

            else:
                annotation = self._update_annotation_from_shapes(
                    annotation, group_shapes, resolution_wh, category_name_to_id
                )

            new_annotations.append(annotation)

        return new_annotations

    @staticmethod
    def find_label_mismatches(
        current_frame_shapes: list[dict],
        current_image_id: int,
        dataset: LazyCOCODataset,
    ) -> list[dict]:
        """Return ObjID/label conflicts between the current frame and other frames.

        Each list entry in the returned list is:
          {"object_id": int, "current_label": str, "existing_label": str}

        Only the first conflicting label found per ObjID is reported.
        """

        # Build a mapping between current frame shapes' object_ids and labels
        current_frame_objid_to_label: dict[int, str] = {}
        for shape in current_frame_shapes:
            object_id = shape.get("group_id")
            shape_type = shape.get("shape_type")
            if object_id is None or shape_type not in ("polygon", "rectangle"):
                continue

            # Prefer polygon over rectangle; only set if not already set by a polygon
            if object_id not in current_frame_objid_to_label or shape_type == "polygon":
                current_frame_objid_to_label[object_id] = shape["label"]

        # Flag objects whose label in other frames conflicts with the current frame.
        conflicts: list[dict] = []
        seen_obj_ids: set[int] = set()
        for image_id, annotations in dataset.annotations_by_image_id.items():
            if image_id == current_image_id:
                continue

            for ann in annotations:
                object_id = ann.get("attributes", {}).get("ObjID")

                if (
                    object_id not in current_frame_objid_to_label
                    or object_id in seen_obj_ids
                ):
                    continue

                label = dataset.category_id_to_name.get(ann.get("category_id"), "")
                current_label = current_frame_objid_to_label[object_id]
                if label and label != current_label:
                    conflicts.append(
                        {
                            "obj_id": object_id,
                            "existing_label": label,
                            "current_label": current_label,
                        }
                    )
                    seen_obj_ids.add(object_id)

        return conflicts

    def _sync_labelme_shapes_to_coco_dataset(
        self,
        dataset: LazyCOCODataset,
        image_id: int,
        shapes: list[dict],
        im_height: int,
        im_width: int,
    ):
        """Convert Labelme shapes to COCO annotations and update the dataset."""
        resolution_wh = (im_width, im_height)

        data_by_group_id = self._group_shapes_by_group_id(shapes)

        new_annotations = self._rebuild_coco_annotations(
            data_by_group_id,
            dataset,
            image_id,
            dataset.category_name_to_id,
            resolution_wh,
        )

        # Update the dataset's annotations for this image
        dataset.annotations_by_image_id[image_id] = new_annotations

    @staticmethod
    def is_label_file(filename):
        return osp.splitext(filename)[1].lower() == LabelFile.suffix
