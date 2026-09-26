"""深度遮挡多视图可见性裁定库。

公开接口见 :mod:`visibility.core` 与 :mod:`visibility.types`。
本包仅依赖 Python 3.14 标准库。
"""

from .errors import (
    InfeasibleSelectionError,
    InvalidCameraError,
    InvalidDepthMapError,
    InvalidParameterError,
    InvalidPointError,
    InvalidSelectionError,
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
from .types import MAX_DEPTH_HEIGHT, MAX_DEPTH_WIDTH
from .selection import (
    MAX_CANDIDATES_PER_FACE,
    MAX_FACES,
    CandidateView,
    DataLedgerEntry,
    FaceCandidates,
    SeamEdge,
    SeamLedgerEntry,
    SelectionResult,
    select_cameras,
)

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
    "classify_view",
    "classify_points",
    "count_visible_views",
    "indices_visible_in_at_least_k",
    "DEFAULT_ABS_TOL",
    "DEFAULT_REL_TOL",
    "MAX_DEPTH_WIDTH",
    "MAX_DEPTH_HEIGHT",
    "CandidateView",
    "FaceCandidates",
    "SeamEdge",
    "SelectionResult",
    "DataLedgerEntry",
    "SeamLedgerEntry",
    "select_cameras",
    "MAX_FACES",
    "MAX_CANDIDATES_PER_FACE",
    "InvalidSelectionError",
    "InfeasibleSelectionError",
]
