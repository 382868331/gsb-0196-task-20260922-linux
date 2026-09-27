"""核心算法：四邻域双线性采样的深度遮挡裁定与多视图聚合。

裁定顺序（严格按契约）
----------------------

对每个 (视图, 世界点)（透视/正交两类视图共用本流程，仅第 2 步投影不同）：

1. 刚体变换得相机坐标 ``(cx, cy, cz)``。**先判 ``cz > 0``**；
   ``cz <= 0`` 直接 ``unknown/behind_camera``，不继续投影。
2. 投影到像素 ``(u, v)``：透视相机做透视除法
   (``u = fx*cx/cz + ...``)；正交相机**不除 ``cz``**
   (``u = fx*cx + ...``)，投影像素与深度无关。
3. 取四邻域 stencil：``(floor u, floor v)`` 及其 +1 两点构成的 4 个
   像素。投影点落在图像覆盖范围之外 -> ``unknown/out_of_bounds``。
4. 四点必须**全部在界内且未被 mask 禁用**：stencil 越界 ->
   ``unknown/missing_neighbor``；任一像素被禁用 ->
   ``unknown/masked_pixel``。**绝不用 0 或默认值补点**。
5. 四点全有效才做双线性插值得到表面深度 ``d``。
6. 比较 ``cz`` 与 ``d``，容差 ``T = abs_tol + rel_tol * d``：

   - ``|cz - d| <= T`` -> ``visible``
   - ``cz - d > T``     -> ``occluded``（点在表面之后）
   - ``d - cz > T``     -> ``in_front``（点在表面之前）

每个点在每个相机的状态独立保留；聚合只数 ``visible`` 的视图数，
不会用多数票把任何 ``occluded`` 改成 ``visible``。
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from .errors import InvalidParameterError
from .types import (
    Camera,
    DepthMap,
    Reason,
    View,
    ViewStatus,
    Visibility,
)
from .validation import validate_k, validate_point, validate_tolerance

DEFAULT_ABS_TOL = 1e-6
"""默认绝对容差，与深度同单位。"""
DEFAULT_REL_TOL = 1e-6
"""默认相对容差（无量纲，乘以采样深度）。"""


def _sample_bilinear(
    dm: DepthMap, u: float, v: float
) -> tuple[float | None, Reason]:
    """四邻域采样。返回 (深度或None, unknown原因)。"""
    h, w = dm.height, dm.width
    if u < 0.0 or v < 0.0 or u >= w or v >= h:
        return None, Reason.OUT_OF_BOUNDS

    cu = math.floor(u)
    cv = math.floor(v)
    c1 = cu + 1
    r1 = cv + 1
    # 缺任一个 stencil 邻点即 unknown，不用 0 补。
    if c1 >= w or r1 >= h:
        return None, Reason.MISSING_NEIGHBOR

    mask = dm.disabled
    if (
        mask[cv][cu]
        or mask[cv][c1]
        or mask[r1][cu]
        or mask[r1][c1]
    ):
        return None, Reason.MASKED_PIXEL

    du = u - cu
    dv = v - cv
    vals = dm.values
    d00 = vals[cv][cu]
    d10 = vals[cv][c1]
    d01 = vals[r1][cu]
    d11 = vals[r1][c1]
    w00 = (1.0 - du) * (1.0 - dv)
    w10 = du * (1.0 - dv)
    w01 = (1.0 - du) * dv
    w11 = du * dv
    return w00 * d00 + w10 * d10 + w01 * d01 + w11 * d11, Reason.NONE


def classify_view(
    view: View,
    point: Sequence[float],
    abs_tol: float = DEFAULT_ABS_TOL,
    rel_tol: float = DEFAULT_REL_TOL,
    *,
    view_index: int = 0,
) -> ViewStatus:
    """裁定单个世界点在单个视图下的状态。

    ``view_index`` 仅写入返回的 :class:`ViewStatus`，便于批次结果定位。
    """
    if not isinstance(view, View):
        raise InvalidParameterError(
            f"view must be a visibility.View instance, got {type(view).__name__}"
        )
    p = validate_point(point)
    a_tol, r_tol = validate_tolerance(abs_tol, rel_tol)
    if isinstance(view_index, bool) or not isinstance(view_index, int):
        raise InvalidParameterError(
            f"view_index must be int, got {type(view_index).__name__}"
        )

    cx, cy, cz = view.camera.project(p)

    # 第一步：z > 0 判定优先于一切投影/采样。
    if cz <= 0.0:
        return ViewStatus(
            view_index=view_index,
            visibility=Visibility.UNKNOWN,
            reason=Reason.BEHIND_CAMERA,
            projected_z=cz,
            sampled_depth=None,
            u=None,
            v=None,
        )

    # 按相机模型投影：透视做 /cz，正交不做。采样与深度比较规则两者共用。
    u, v = view.camera.project_pixels(cx, cy, cz)
    sampled, reason = _sample_bilinear(view.depth_map, u, v)
    if sampled is None:
        return ViewStatus(
            view_index=view_index,
            visibility=Visibility.UNKNOWN,
            reason=reason,
            projected_z=cz,
            sampled_depth=None,
            u=u,
            v=v,
        )

    delta = cz - sampled
    tol = a_tol + r_tol * sampled
    if delta > tol:
        vis = Visibility.OCCLUDED
    elif delta < -tol:
        vis = Visibility.IN_FRONT
    else:
        vis = Visibility.VISIBLE
    return ViewStatus(
        view_index=view_index,
        visibility=vis,
        reason=Reason.NONE,
        projected_z=cz,
        sampled_depth=sampled,
        u=u,
        v=v,
    )


def _validate_views(views: object) -> Sequence[View]:
    if not isinstance(views, Sequence) or isinstance(views, (str, bytes)):
        raise InvalidParameterError("views must be a sequence of View")
    if len(views) == 0:
        raise InvalidParameterError("views must contain at least one view")
    for i, view in enumerate(views):
        if not isinstance(view, View):
            raise InvalidParameterError(
                f"views[{i}] must be a visibility.View instance, "
                f"got {type(view).__name__}"
            )
    return views


def classify_points(
    views: Sequence[View],
    points: Sequence[Sequence[float]],
    abs_tol: float = DEFAULT_ABS_TOL,
    rel_tol: float = DEFAULT_REL_TOL,
) -> list[list[ViewStatus]]:
    """批量裁定。

    返回 ``results[point_index][view_index]`` 的 :class:`ViewStatus`
    矩阵。每个 (点, 视图) 状态独立保留。

    原子性：所有输入先整体校验通过才开始计算；本函数不修改任何输入、
    不持有可变状态，因此校验失败不会产生部分结果。
    """
    views = _validate_views(views)
    if not isinstance(points, Sequence) or isinstance(points, (str, bytes)):
        raise InvalidParameterError("points must be a sequence of 3D points")
    # 先整体校验点与参数，再执行任何裁定。
    validated_points = [
        validate_point(pt, f"points[{i}]") for i, pt in enumerate(points)
    ]
    a_tol, r_tol = validate_tolerance(abs_tol, rel_tol)

    results: list[list[ViewStatus]] = []
    for pt in validated_points:
        row = [
            classify_view(view, pt, a_tol, r_tol, view_index=j)
            for j, view in enumerate(views)
        ]
        results.append(row)
    return results


def count_visible_views(statuses: Sequence[ViewStatus]) -> int:
    """统计一个点的各视图状态中 ``visible`` 的数量。"""
    if not isinstance(statuses, Sequence) or isinstance(statuses, (str, bytes)):
        raise InvalidParameterError("statuses must be a sequence of ViewStatus")
    return sum(1 for s in statuses if s.visibility is Visibility.VISIBLE)


def indices_visible_in_at_least_k(
    results: Sequence[Sequence[ViewStatus]], k: int
) -> list[int]:
    """聚合：返回至少在 ``k`` 个视图中 ``visible`` 的点下标。

    只计 ``visible``；``occluded`` / ``in_front`` / ``unknown`` 均不计，
    不做跨视图投票翻转。
    """
    if not isinstance(results, Sequence) or isinstance(results, (str, bytes)):
        raise InvalidParameterError("results must be the matrix from classify_points")
    if len(results) == 0:
        # 空点集：无法得知视图数，只要 k 是非 bool 的正整数即返回空集。
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise InvalidParameterError(
                f"k must be a positive int, got {k!r}"
            )
        return []
    n_views = len(results[0])
    kk = validate_k(k, n_views)
    chosen: list[int] = []
    for i, row in enumerate(results):
        if not isinstance(row, Sequence) or len(row) != n_views:
            raise InvalidParameterError(
                f"results[{i}] must be a sequence of {n_views} ViewStatus"
            )
        if count_visible_views(row) >= kk:
            chosen.append(i)
    return chosen
