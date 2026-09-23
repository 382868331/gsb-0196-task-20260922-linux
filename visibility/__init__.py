"""Depth-occlusion multi-view visibility adjudication library."""

from visibility.core import (
    MAX_DEPTH_MAP_SIZE,
    DepthMap,
    PinholeCamera,
    Status,
    Verdict,
    VisibilityError,
    classify_point,
    classify_points,
    points_visible_in_at_least,
)

__all__ = [
    "MAX_DEPTH_MAP_SIZE",
    "DepthMap",
    "PinholeCamera",
    "Status",
    "Verdict",
    "VisibilityError",
    "classify_point",
    "classify_points",
    "points_visible_in_at_least",
]
