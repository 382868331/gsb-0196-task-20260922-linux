"""深度遮挡多视图可见性裁定库。

公开接口见 :mod:`visibility.core` 与 :mod:`visibility.types`。
本包仅依赖 Python 3.14 标准库。
"""

from .errors import (
    InvalidCameraError,
    InvalidDepthMapError,
    InvalidParameterError,
    InvalidPointError,
    InvalidUpdateError,
    StaleVersionError,
    VisibilityError,
)
from .types import Camera, DepthMap, Reason, View, ViewStatus, Visibility
from .core import (
    DEFAULT_ABS_TOL,
    DEFAULT_REL_TOL,
    classify_points,
    classify_view,
    count_visible_views,
    indices_visible_in_at_least_k,
)
from .cache import (
    DepthUpdate,
    MaskUpdate,
    UpdateReport,
    VisibilityCache,
    VisibilitySnapshot,
)
from .types import MAX_DEPTH_HEIGHT, MAX_DEPTH_WIDTH

__all__ = [
    "Camera",
    "DepthMap",
    "View",
    "ViewStatus",
    "Visibility",
    "Reason",
    "VisibilityError",
    "InvalidCameraError",
    "InvalidDepthMapError",
    "InvalidPointError",
    "InvalidParameterError",
    "InvalidUpdateError",
    "StaleVersionError",
    "classify_view",
    "classify_points",
    "count_visible_views",
    "indices_visible_in_at_least_k",
    "VisibilityCache",
    "VisibilitySnapshot",
    "DepthUpdate",
    "MaskUpdate",
    "UpdateReport",
    "DEFAULT_ABS_TOL",
    "DEFAULT_REL_TOL",
    "MAX_DEPTH_WIDTH",
    "MAX_DEPTH_HEIGHT",
]
