# Standard Library imports
from collections.abc import Mapping
from typing import Any
from typing import Literal
from typing import NotRequired
from typing import Optional
from typing import TypedDict
from typing import TypeGuard

# External imports
import numpy as np
from numpy.typing import NDArray

# A flat list of coordinates in COCO polygon format: [x1, y1, x2, y2, ..., xn, yn]
CocoPolygon = list[float]

ShapePolygon = list[list[float]]


class CompressedRLE(TypedDict):
    counts: list[int]
    size: list[int]


def is_compressed_rle(d: Any) -> TypeGuard[CompressedRLE]:
    return (
        isinstance(d, dict)
        and "counts" in d
        and "size" in d
        and isinstance(d["counts"], list)
        and isinstance(d["size"], list)
        and len(d["size"]) == 2
        and all(isinstance(c, int) for c in d["size"])
        and (not d["counts"] or isinstance(d["counts"][0], int))
    )


def is_polygon_segmentation(seg: Any) -> TypeGuard[list[CocoPolygon]]:
    return isinstance(seg, list) and all(
        isinstance(p, list) and all(isinstance(v, (int, float)) for v in p) for p in seg
    )


class CocoAnnotation(TypedDict):
    id: int
    image_id: int
    category_id: int

    # Polygon segmentation (iscrowd == 0) OR RLE (iscrowd == 1)
    segmentation: list[CocoPolygon] | CompressedRLE

    area: float
    bbox: list[float]  # [x, y, width, height]

    # From the COCO specification:
    # https://cocodataset.org/#format-data
    #
    # The segmentation format depends on whether the instance represents
    # a single object (iscrowd=0 in which case polygons are used) or a
    # collection of objects (iscrowd=1 in which case RLE is used).
    # Note that a single object (iscrowd=0) may require multiple polygons,
    # for example if occluded. Crowd annotations (iscrowd=1) are used to label large
    # groups of objects (e.g. a crowd of people).
    iscrowd: Literal[0, 1]

    # Optional, non-standard COCO field
    attributes: NotRequired[dict[str, Any]]


def is_coco_annotation(d: Mapping[str, Any]) -> TypeGuard[CocoAnnotation]:
    """Return True if `d` is a valid CocoAnnotation with all required fields and types."""
    if not all(isinstance(d.get(k), int) for k in ("id", "image_id", "category_id")):
        return False
    if not isinstance(d.get("area"), (int, float)):
        return False
    bbox = d.get("bbox")
    if not (
        isinstance(bbox, list)
        and len(bbox) == 4
        and all(isinstance(v, (int, float)) for v in bbox)
    ):
        return False
    iscrowd = d.get("iscrowd")
    if iscrowd not in (0, 1):
        return False
    seg = d.get("segmentation")
    if iscrowd == 1:
        if not is_compressed_rle(seg):
            return False
    else:
        if not is_polygon_segmentation(seg):
            return False
    return True


class RejectedCocoAnnotation(CocoAnnotation):
    """COCO annotation that was rejected (deleted or edited) by the user."""

    rejection_type: Literal["deleted", "edited"]


class CocoCategories(TypedDict):
    id: int
    name: str
    supercategory: str


class CocoFile(TypedDict):
    images: list[dict]
    categories: list[CocoCategories]
    annotations: list[CocoAnnotation]


class OtherData(TypedDict, total=False):
    original_annotation: CocoAnnotation


class ShapeDict(TypedDict):
    label: str
    points: ShapePolygon
    shape_type: str
    flags: dict[str, bool]
    description: str
    group_id: Optional[int]
    mask: Optional[NDArray[np.bool]]
    other_data: OtherData


class AnnotationWithShapes(TypedDict):
    annotation: CocoAnnotation
    shapes: list[ShapeDict]
