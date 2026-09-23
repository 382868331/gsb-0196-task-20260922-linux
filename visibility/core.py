"""Depth-occlusion multi-view visibility adjudication.

Reusable module for offline visual-reconstruction pipelines.  Given known
pinhole cameras, world-space 3D points and one depth map (camera-space z)
per camera of at most 32x32 pixels, decide for every (point, camera) pair
whether the point is VISIBLE, OCCLUDED, IN_FRONT of the recorded surface,
or UNKNOWN, and aggregate the points visible in at least k views.

Contract highlights (see README.md for the full contract):

* Projection first requires camera-space z > 0; only then is the
  4-neighbourhood sampled.  Bilinear interpolation happens only when all
  four neighbours exist, are enabled and hold valid depths; any missing or
  disabled neighbour yields UNKNOWN.  Missing data is never zero-filled.
* Depth comparison tolerance is ``abs_tol + rel_tol * sampled_depth``.
* Per-camera verdicts for the same point are kept separate; aggregation
  only selects point indices and never rewrites a per-view verdict.
* Invalid input raises :class:`VisibilityError` with a message that names
  the offending argument.  NaN/Infinity are rejected for numeric input and
  bool is rejected where an int is required.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

__all__ = [
    "MAX_DEPTH_MAP_SIZE",
    "VisibilityError",
    "Status",
    "Verdict",
    "PinholeCamera",
    "DepthMap",
    "classify_point",
    "classify_points",
    "points_visible_in_at_least",
]

#: Maximum width/height of a depth map, per the task contract.
MAX_DEPTH_MAP_SIZE = 32


class VisibilityError(ValueError):
    """Invalid input.  The message names the offending argument."""


class Status(Enum):
    """Per-point, per-camera adjudication outcome."""

    VISIBLE = "visible"
    OCCLUDED = "occluded"
    IN_FRONT = "in_front"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Verdict:
    """Result of adjudicating one world point against one camera.

    ``z`` is the camera-space z of the point (always set once the point
    validated).  ``depth`` is the bilinearly sampled depth and ``u``/``v``
    the projected pixel coordinates; both are None when no complete
    4-neighbourhood was available.  ``reason`` is a human-readable
    explanation, mainly useful for UNKNOWN outcomes.
    """

    status: Status
    reason: str
    z: float | None = None
    depth: float | None = None
    u: float | None = None
    v: float | None = None


# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------

def _as_finite(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VisibilityError(
            f"{name}: expected a real number, got {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result):
        raise VisibilityError(f"{name}: must be finite, got {value!r}")
    return result


def _as_int(name, value, lo=None, hi=None):
    if isinstance(value, bool) or not isinstance(value, int):
        raise VisibilityError(
            f"{name}: expected an int, got {type(value).__name__}")
    if lo is not None and value < lo:
        raise VisibilityError(f"{name}: must be >= {lo}, got {value}")
    if hi is not None and value > hi:
        raise VisibilityError(f"{name}: must be <= {hi}, got {value}")
    return value


def _as_point3(name, value):
    try:
        items = tuple(value)
    except TypeError:
        raise VisibilityError(
            f"{name}: expected a sequence of 3 numbers") from None
    if len(items) != 3:
        raise VisibilityError(
            f"{name}: expected 3 components, got {len(items)}")
    return tuple(_as_finite(f"{name}[{i}]", item) for i, item in enumerate(items))


def _as_tolerance(name, value):
    result = _as_finite(name, value)
    if result < 0.0:
        raise VisibilityError(f"{name}: must be >= 0, got {result!r}")
    return result


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------

class PinholeCamera:
    """Known pinhole camera: intrinsics plus a world-to-camera rigid pose.

    ``rotation`` is a 3x3 nested sequence mapping world vectors into the
    camera frame and ``translation`` the corresponding 3-vector, i.e.
    ``p_cam = rotation @ p_world + translation``.  The rotation is checked
    for shape and finiteness only; orthonormality is the caller's
    responsibility (documented limitation).
    """

    __slots__ = ("fx", "fy", "cx", "cy", "rotation", "translation")

    def __init__(self, fx, fy, cx, cy, rotation, translation):
        self.fx = _as_finite("fx", fx)
        self.fy = _as_finite("fy", fy)
        self.cx = _as_finite("cx", cx)
        self.cy = _as_finite("cy", cy)
        if self.fx <= 0.0:
            raise VisibilityError(f"fx: must be > 0, got {self.fx!r}")
        if self.fy <= 0.0:
            raise VisibilityError(f"fy: must be > 0, got {self.fy!r}")
        try:
            rows = tuple(tuple(row) for row in rotation)
        except TypeError:
            raise VisibilityError(
                "rotation: expected a 3x3 nested sequence") from None
        if len(rows) != 3 or any(len(row) != 3 for row in rows):
            raise VisibilityError(
                f"rotation: expected a 3x3 nested sequence, got "
                f"{len(rows)} rows")
        self.rotation = tuple(
            tuple(_as_finite(f"rotation[{i}][{j}]", value)
                  for j, value in enumerate(row))
            for i, row in enumerate(rows))
        self.translation = _as_point3("translation", translation)

    def world_to_camera(self, point):
        """Transform a world point into camera coordinates."""
        px, py, pz = _as_point3("point", point)
        tx, ty, tz = self.translation
        r = self.rotation
        return (
            r[0][0] * px + r[0][1] * py + r[0][2] * pz + tx,
            r[1][0] * px + r[1][1] * py + r[1][2] * pz + ty,
            r[2][0] * px + r[2][1] * py + r[2][2] * pz + tz,
        )

    def __repr__(self):
        return (f"PinholeCamera(fx={self.fx!r}, fy={self.fy!r}, "
                f"cx={self.cx!r}, cy={self.cy!r})")


class DepthMap:
    """Per-camera depth map (camera-space z per pixel) plus disable mask.

    ``depths`` is a row-major sequence of ``width * height`` strictly
    positive finite values.  Invalid pixels must be marked through
    ``disabled`` (None or a row-major sequence of ``width * height``
    bools, True = do not use); zero is not a valid placeholder depth.
    """

    __slots__ = ("width", "height", "depths", "disabled")

    def __init__(self, width, height, depths, disabled=None):
        self.width = _as_int("width", width, lo=1, hi=MAX_DEPTH_MAP_SIZE)
        self.height = _as_int("height", height, lo=1, hi=MAX_DEPTH_MAP_SIZE)
        count = self.width * self.height
        try:
            raw_depths = tuple(depths)
        except TypeError:
            raise VisibilityError("depths: expected a sequence") from None
        if len(raw_depths) != count:
            raise VisibilityError(
                f"depths: expected {count} values for a "
                f"{self.width}x{self.height} map, got {len(raw_depths)}")
        checked = []
        for i, value in enumerate(raw_depths):
            depth = _as_finite(f"depths[{i}]", value)
            if depth <= 0.0:
                raise VisibilityError(
                    f"depths[{i}]: depth must be > 0, got {depth!r}; "
                    f"mark invalid pixels via the disabled mask instead")
            checked.append(depth)
        self.depths = tuple(checked)
        if disabled is None:
            self.disabled = None
        else:
            try:
                raw_mask = tuple(disabled)
            except TypeError:
                raise VisibilityError("disabled: expected a sequence") from None
            if len(raw_mask) != count:
                raise VisibilityError(
                    f"disabled: expected {count} values, got {len(raw_mask)}")
            for i, flag in enumerate(raw_mask):
                if not isinstance(flag, bool):
                    raise VisibilityError(
                        f"disabled[{i}]: expected bool, got "
                        f"{type(flag).__name__}")
            self.disabled = raw_mask

    def __repr__(self):
        return f"DepthMap(width={self.width}, height={self.height})"


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------

def _sample_bilinear(depth_map, u, v):
    """Bilinear sample at fractional pixel (u, v).

    Returns ``(depth, "")`` on success or ``(None, reason)`` when any of
    the four neighbours is out of bounds or disabled.  Never zero-fills.
    """
    x0 = math.floor(u)
    y0 = math.floor(v)
    if (x0 < 0 or y0 < 0 or x0 + 1 >= depth_map.width
            or y0 + 1 >= depth_map.height):
        return None, (
            f"projection ({u:.6g}, {v:.6g}) lacks a complete 4-pixel "
            f"neighbourhood inside the "
            f"{depth_map.width}x{depth_map.height} depth map")
    w = depth_map.width
    corners = ((x0, y0), (x0 + 1, y0), (x0, y0 + 1), (x0 + 1, y0 + 1))
    if depth_map.disabled is not None:
        for cx, cy in corners:
            if depth_map.disabled[cy * w + cx]:
                return None, (
                    f"interpolation neighbour pixel ({cx}, {cy}) is "
                    f"disabled by the mask")
    d00 = depth_map.depths[y0 * w + x0]
    d10 = depth_map.depths[y0 * w + x0 + 1]
    d01 = depth_map.depths[(y0 + 1) * w + x0]
    d11 = depth_map.depths[(y0 + 1) * w + x0 + 1]
    tx = u - x0
    ty = v - y0
    depth = (d00 * (1.0 - tx) * (1.0 - ty)
             + d10 * tx * (1.0 - ty)
             + d01 * (1.0 - tx) * ty
             + d11 * tx * ty)
    return depth, ""


def classify_point(point, camera, depth_map, abs_tol=0.0, rel_tol=0.0):
    """Adjudicate one world point against one camera's depth map.

    Returns a :class:`Verdict`.  The projection's camera-space z is
    checked (z > 0) before any neighbourhood sampling; the depth
    comparison uses tolerance ``abs_tol + rel_tol * sampled_depth`` with
    the VISIBLE band inclusive on both sides.
    """
    if not isinstance(camera, PinholeCamera):
        raise VisibilityError(
            f"camera: expected PinholeCamera, got {type(camera).__name__}")
    if not isinstance(depth_map, DepthMap):
        raise VisibilityError(
            f"depth_map: expected DepthMap, got {type(depth_map).__name__}")
    abs_tol = _as_tolerance("abs_tol", abs_tol)
    rel_tol = _as_tolerance("rel_tol", rel_tol)

    cx, cy, cz = camera.world_to_camera(point)
    if cz <= 0.0:
        return Verdict(
            Status.UNKNOWN,
            "point is behind or on the camera plane (z <= 0)",
            z=cz)
    u = camera.fx * cx / cz + camera.cx
    v = camera.fy * cy / cz + camera.cy
    depth, reason = _sample_bilinear(depth_map, u, v)
    if depth is None:
        return Verdict(Status.UNKNOWN, reason, z=cz, u=u, v=v)
    tol = abs_tol + rel_tol * depth
    if cz < depth - tol:
        status = Status.IN_FRONT
        why = "point is closer to the camera than the sampled surface"
    elif cz <= depth + tol:
        status = Status.VISIBLE
        why = "point lies on the sampled surface within tolerance"
    else:
        status = Status.OCCLUDED
        why = "point is hidden behind the sampled surface"
    return Verdict(status, why, z=cz, depth=depth, u=u, v=v)


def classify_points(points, views, abs_tol=0.0, rel_tol=0.0):
    """Adjudicate many points against many views.

    ``views`` is a sequence of ``(PinholeCamera, DepthMap)`` pairs.
    Returns ``result[i][j]``: the :class:`Verdict` for point ``i`` in view
    ``j``.  Per-view verdicts are kept separate; nothing is merged or
    majority-voted.

    Every point, view and tolerance is validated before any
    classification runs, so a rejected batch raises without producing or
    storing partial results (the function is pure and mutates nothing).
    """
    try:
        point_list = list(points)
    except TypeError:
        raise VisibilityError("points: expected a sequence") from None
    try:
        view_list = list(views)
    except TypeError:
        raise VisibilityError("views: expected a sequence") from None
    checked_views = []
    for j, view in enumerate(view_list):
        try:
            cam, depth_map = view
        except (TypeError, ValueError):
            raise VisibilityError(
                f"views[{j}]: expected a (camera, depth_map) pair") from None
        if not isinstance(cam, PinholeCamera):
            raise VisibilityError(
                f"views[{j}][0]: expected PinholeCamera, got "
                f"{type(cam).__name__}")
        if not isinstance(depth_map, DepthMap):
            raise VisibilityError(
                f"views[{j}][1]: expected DepthMap, got "
                f"{type(depth_map).__name__}")
        checked_views.append((cam, depth_map))
    checked_points = [
        _as_point3(f"points[{i}]", point) for i, point in enumerate(point_list)
    ]
    abs_tol = _as_tolerance("abs_tol", abs_tol)
    rel_tol = _as_tolerance("rel_tol", rel_tol)
    return [[classify_point(point, cam, depth_map, abs_tol, rel_tol)
             for cam, depth_map in checked_views]
            for point in checked_points]


def points_visible_in_at_least(per_view_statuses, k):
    """Indices of the points VISIBLE in at least ``k`` views.

    ``per_view_statuses[i]`` is the row of per-view :class:`Status` values
    for point ``i`` (e.g. ``verdict.status`` across a classify_points
    row).  Aggregation only selects point indices; the per-view statuses
    themselves are never altered, so an OCCLUDED vote is never overridden
    by a majority.  If ``k`` exceeds the number of views of a point, that
    point is provably not selectable and is simply absent from the result
    (this is a determined outcome, not a search-budget exhaustion — the
    algorithm performs no search).
    """
    k = _as_int("k", k, lo=1)
    try:
        rows = [tuple(row) for row in per_view_statuses]
    except TypeError:
        raise VisibilityError(
            "per_view_statuses: expected a sequence of sequences") from None
    for i, row in enumerate(rows):
        for j, status in enumerate(row):
            if not isinstance(status, Status):
                raise VisibilityError(
                    f"per_view_statuses[{i}][{j}]: expected Status, got "
                    f"{type(status).__name__}")
    return {i for i, row in enumerate(rows)
            if sum(1 for status in row if status is Status.VISIBLE) >= k}
