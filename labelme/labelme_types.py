# Standard Library imports
from typing import Any
from typing import Literal
from typing import NotRequired
from typing import Optional
from typing import TypedDict

# External imports
import numpy as np
from numpy.typing import NDArray

# A flat list of coordinates in COCO polygon format: [x1, y1, x2, y2, ..., xn, yn]
CocoPolygon = list[float]

ShapePolygon = list[list[float]]


class CompressedRLE(TypedDict):
    counts: list[int]
    size: list[int]


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
