# Standard Library imports
from typing import Any
from typing import Literal
from typing import NotRequired
from typing import Optional
from typing import TypedDict

# External imports
import numpy as np
from numpy.typing import NDArray


class CocoRLE(TypedDict):
    counts: list[int]
    size: list[int]


class CocoAnnotation(TypedDict):
    id: int
    image_id: int
    category_id: int

    # Polygon segmentation (iscrowd == 0) OR RLE (iscrowd == 1)
    segmentation: NotRequired[list[list[float]] | CocoRLE]

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
    #
    # Having said that, RLE is being used regardless if a single fish is being annotated!
    # This allows to easily preserve storing the original segmentation mask produced by SAM.
    # However, when loading the segmentation with Labelme, it gets transformed to a polygon.
    # The quality of the transformation from RLE to polygon depends on the precision of
    # the approximation. A highly detailed transformation may be undesirable because of
    # the high number of vertices of the polygon. Controlling the approximation is doable,
    # but may not be something to be tuned now.
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
    points: list[list[float]]
    shape_type: str
    flags: dict[str, bool]
    description: str
    group_id: Optional[int]
    mask: Optional[NDArray[np.bool]]
    other_data: OtherData


class AnnotationWithShapes(TypedDict):
    annotation: CocoAnnotation
    shapes: list[ShapeDict]
