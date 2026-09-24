"""核心算法：四邻域双线性采样的深度遮挡裁定、深度区间裁定与多视图聚合。

标量裁定顺序（严格按契约）
--------------------------

对每个 (视图, 世界点)：

1. 刚体变换得相机坐标 ``(cx, cy, cz)``。**先判 ``cz > 0``**；
   ``cz <= 0`` 直接 ``unknown/behind_camera``，不继续投影。
2. 针孔投影到像素 ``(u, v)``。
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

深度区间裁定（:class:`~visibility.types.DepthInterval`）
-------------------------------------------------------

当点沿视线的相机 z 只能确定为闭区间 ``[z_near, z_far]``（像素
``(u, v)`` 固定，即同一条针孔视线）时：

1. stencil 规则与标量路径完全一致（缺邻点/禁用像素均为 unknown，
   不用 0 补）；四点齐全后做同样的双线性插值得到表面深度 ``d``。
2. 表面容差带与标量规则同源：``T = abs_tol + rel_tol*d``，
   表面可能深度为闭区间 ``[d-T, d+T]``。零宽度点区间
   ``[z, z]`` 的裁定因此与标量裁定逐位一致。
3. **原点退化优先**：点区间与表面跨度合在一起若跨过 z=0
   （``z_near <= 0 < z_far``），投影在 z=0 处对针孔模型无定义，
   判 ``uncertain/depth_span_crosses_origin``，不猜可见性。
   ``z_far <= 0`` 整条视线段在相机后方 -> ``unknown/behind_camera``。
4. 闭区间比较（含边界）：

   - ``z_near >= d-T`` 且 ``z_far <= d+T`` -> ``visible``（确定可见）
   - ``z_near > d+T`` -> ``occluded``（确定遮挡）
   - ``z_far < d-T`` -> ``in_front``（确定在表面之前）
   - 其余（仅端点接触或部分重叠）-> ``uncertain/interval_overlap``。

   **重叠/接触绝不直接判可见**：例如 ``z_far > d+T`` 而
   ``z_near <= d+T`` 时存在遮挡可能，只能 ``uncertain``。

每个点在每个相机的状态独立保留；聚合只数 ``visible`` 的视图数，
不会用多数票把任何 ``occluded`` / ``uncertain`` 改成 ``visible``。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .errors import InvalidParameterError
from .types import (
    Camera,
    DepthMap,
    IntervalQuery,
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


def _pixel_coords(cam: Camera, cx: float, cy: float, cz: float) -> tuple[float, float]:
    """针孔投影：相机坐标 -> 像素 (u=列, v=行)。调用前已确保 cz > 0。"""
    K = cam.K
    inv_z = 1.0 / cz
    u = (K[0][0] * cx + K[0][1] * cy) * inv_z + K[0][2]
    v = K[1][1] * cy * inv_z + K[1][2]
    return u, v


@dataclass(frozen=True)
class _Stencil:
    """四邻域 stencil：四角 (row,col) 与对应的双线性权重。"""

    corners: tuple[tuple[int, int], ...]
    weights: tuple[float, ...]


def _stencil(dm: DepthMap, u: float, v: float) -> tuple[_Stencil | None, Reason]:
    """定位并校验四邻域；返回 (stencil或None, unknown原因)。

    越界 / 缺邻点 / 邻点被 mask 禁用均不采样，绝不用 0 补点。
    """
    h, w = dm.height, dm.width
    if u < 0.0 or v < 0.0 or u >= w or v >= h:
        return None, Reason.OUT_OF_BOUNDS

    cu = math.floor(u)
    cv = math.floor(v)
    c1 = cu + 1
    r1 = cv + 1
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

    fu = u - cu
    fv = v - cv
    corners = ((cv, cu), (cv, c1), (r1, cu), (r1, c1))
    weights = (
        (1.0 - fu) * (1.0 - fv),
        fu * (1.0 - fv),
        (1.0 - fu) * fv,
        fu * fv,
    )
    return _Stencil(corners=corners, weights=weights), Reason.NONE


def _sample_bilinear_with(
    dm: DepthMap, st: _Stencil
) -> float:
    """在已校验的 stencil 上做双线性插值。"""
    vals = dm.values
    return sum(
        w * vals[r][c] for w, (r, c) in zip(st.weights, st.corners)
    )


def _sample_bilinear(
    dm: DepthMap, u: float, v: float
) -> tuple[float | None, Reason]:
    """四邻域采样。返回 (深度或None, unknown原因)。"""
    st, reason = _stencil(dm, u, v)
    if st is None:
        return None, reason
    return _sample_bilinear_with(dm, st), Reason.NONE


def _surface_tolerance_band(
    sampled: float, a_tol: float, r_tol: float
) -> tuple[float, float]:
    """双线性表面深度 ``d`` 的容差闭区间 ``[d-T, d+T]``。

    ``T = abs_tol + rel_tol*d``，与标量裁定完全同源；相对项以
    采样（插值）深度为基准，不以点深度为基准。
    """
    t = a_tol + r_tol * sampled
    return sampled - t, sampled + t


def classify_view(
    view: View,
    point: Sequence[float] | IntervalQuery,
    abs_tol: float = DEFAULT_ABS_TOL,
    rel_tol: float = DEFAULT_REL_TOL,
    *,
    view_index: int = 0,
) -> ViewStatus:
    """裁定单个世界点（或深度区间查询）在单个视图下的状态。

    ``point`` 为三元组时走标量裁定（四态 visible/occluded/in_front/
    unknown）；为 :class:`IntervalQuery` 时走深度区间裁定，可能额外
    返回 ``uncertain``。``view_index`` 仅写入返回的
    :class:`ViewStatus`，便于批次结果定位。
    """
    if not isinstance(view, View):
        raise InvalidParameterError(
            f"view must be a visibility.View instance, got {type(view).__name__}"
        )
    a_tol, r_tol = validate_tolerance(abs_tol, rel_tol)
    if isinstance(view_index, bool) or not isinstance(view_index, int):
        raise InvalidParameterError(
            f"view_index must be int, got {type(view_index).__name__}"
        )
    if isinstance(point, IntervalQuery):
        return _classify_interval(view, point, a_tol, r_tol, view_index)

    p = validate_point(point)
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

    u, v = _pixel_coords(view.camera, cx, cy, cz)
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


def _classify_interval(
    view: View,
    query: IntervalQuery,
    a_tol: float,
    r_tol: float,
    view_index: int,
) -> ViewStatus:
    """深度区间裁定。规则见模块 docstring。"""
    u, v = query.u, query.v
    z_near, z_far = query.interval.as_tuple()
    zrange = (z_near, z_far)

    # 与标量路径一致：整条视线段在相机后方优先判 behind_camera。
    if z_far <= 0.0:
        return ViewStatus(
            view_index=view_index,
            visibility=Visibility.UNKNOWN,
            reason=Reason.BEHIND_CAMERA,
            projected_z=None,
            sampled_depth=None,
            u=u,
            v=v,
            depth_interval=zrange,
            surface_interval=None,
        )

    st, reason = _stencil(view.depth_map, u, v)
    if st is None:
        return ViewStatus(
            view_index=view_index,
            visibility=Visibility.UNKNOWN,
            reason=reason,
            projected_z=None,
            sampled_depth=None,
            u=u,
            v=v,
            depth_interval=zrange,
            surface_interval=None,
        )

    sampled = _sample_bilinear_with(view.depth_map, st)
    d_lo, d_hi = _surface_tolerance_band(sampled, a_tol, r_tol)
    surface = (d_lo, d_hi)

    # 区间跨过相机原点（z_far 已确保 > 0）：z=0 处针孔投影无定义，
    # 区间前段在相机后、后段在相机前，无法自洽裁定 -> uncertain。
    if z_near <= 0.0 < z_far:
        return ViewStatus(
            view_index=view_index,
            visibility=Visibility.UNCERTAIN,
            reason=Reason.DEPTH_SPAN_CROSSES_ORIGIN,
            projected_z=None,
            sampled_depth=sampled,
            u=u,
            v=v,
            depth_interval=zrange,
            surface_interval=surface,
        )

    # 闭区间比较：只有点区间被表面容差范围完全包含才确定可见。
    if z_near >= d_lo and z_far <= d_hi:
        vis = Visibility.VISIBLE
        reason = Reason.NONE
    elif z_near > d_hi:
        vis = Visibility.OCCLUDED
        reason = Reason.NONE
    elif z_far < d_lo:
        vis = Visibility.IN_FRONT
        reason = Reason.NONE
    else:
        # 仅端点接触或部分重叠：可见与遮挡均无法证明。
        vis = Visibility.UNCERTAIN
        reason = Reason.INTERVAL_OVERLAP
    return ViewStatus(
        view_index=view_index,
        visibility=vis,
        reason=reason,
        projected_z=None,
        sampled_depth=sampled,
        u=u,
        v=v,
        depth_interval=zrange,
        surface_interval=surface,
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
    points: Sequence[Sequence[float] | IntervalQuery],
    abs_tol: float = DEFAULT_ABS_TOL,
    rel_tol: float = DEFAULT_REL_TOL,
) -> list[list[ViewStatus]]:
    """批量裁定。

    ``points`` 每个元素可以是世界点三元组（标量裁定）或
    :class:`IntervalQuery`（深度区间裁定），可在同一批次中混用。
    返回 ``results[point_index][view_index]`` 的 :class:`ViewStatus`
    矩阵。每个 (点, 视图) 状态独立保留，区间视图的 ``uncertain``
    不会被融合或翻转。

    原子性：所有输入先整体校验通过才开始计算；本函数不修改任何输入、
    不持有可变状态，因此校验失败不会产生部分结果。
    """
    views = _validate_views(views)
    if not isinstance(points, Sequence) or isinstance(points, (str, bytes)):
        raise InvalidParameterError("points must be a sequence of 3D points")
    # 先整体校验点/区间查询与参数，再执行任何裁定。
    validated_queries: list[tuple[float, float, float] | IntervalQuery] = []
    for i, item in enumerate(points):
        if isinstance(item, IntervalQuery):
            validated_queries.append(item)
        else:
            validated_queries.append(validate_point(item, f"points[{i}]"))
    a_tol, r_tol = validate_tolerance(abs_tol, rel_tol)

    results: list[list[ViewStatus]] = []
    for query in validated_queries:
        row = [
            classify_view(view, query, a_tol, r_tol, view_index=j)
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
