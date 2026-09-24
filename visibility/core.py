"""核心算法：四邻域双线性采样的深度区间可见性裁定与多视图聚合。

点与表面都可以携带深度**区间**：

- 点侧：投影得到 ``cz``，投影深度区间为 ``[cz-depth_eps, cz+depth_eps]``；
  ``depth_eps=0``（默认）时退化为单点，旧版行为逐位保持。
- 表面侧：每个像素占据闭区间 ``[near, far]``（``DepthMap`` 的
  ``values``/``far``）；``far==near`` 时退化为单点深度。
- 四邻域双线性对 near、far 两层**分别**插值。双线性权重非负且和为 1，
  故对四个角点区间做分别端点插值得到的仍是真实表面深度范围的包络
  （对仿射深度斜面为精确值）。

裁定顺序（严格按契约）
----------------------

对每个 (视图, 世界点)：

1. 刚体变换得相机坐标 ``(cx, cy, cz)``。先判深度区间与相机前方的关系：
   区间整体 ``cz + depth_eps <= 0`` -> ``unknown/behind_camera``；
   ``cz - depth_eps <= 0 < cz + depth_eps``（区间跨过原点，无法合法
   透视投影）-> ``unknown/depth_interval_at_origin``。
2. 针孔投影到像素 ``(u, v)``（以区间中点 cz 投影像素位置）。
3. 取四邻域 stencil：``(floor u, floor v)`` 及其 +1 两点构成的 4 个
   像素。投影点落在图像覆盖范围之外 -> ``unknown/out_of_bounds``。
4. 四点必须**全部在界内且未被 mask 禁用**：stencil 越界 ->
   ``unknown/missing_neighbor``；任一像素被禁用 ->
   ``unknown/masked_pixel``。**绝不用 0 或默认值补点**。
5. 四点全有效才分别对 near、far 双线性插值，得表面区间 ``[d_lo,d_hi]``。
6. 容差在两端分别取 ``T_lo = abs_tol + rel_tol*d_lo``、
   ``T_hi = abs_tol + rel_tol*d_hi``，把表面板膨胀为
   ``[d_lo-T_lo, d_hi+T_hi]``（单点深度时即旧的 ``d +/- T``）：

   - 点区间整体落在板内（``z_lo >= d_lo-T_lo`` 且
     ``z_hi <= d_hi+T_hi``，闭区间）-> ``visible``；
   - 点区间整体在板后方（``z_lo > d_hi+T_hi``）-> ``occluded``；
   - 点区间整体在板前方（``z_hi < d_lo-T_lo``）-> ``in_front``；
   - 其余（区间部分重叠，或仅端点接触于板外边界）-> ``uncertain``。
     **重叠/接触绝不直接判 visible 或 occluded。**

每个点在每个相机的状态与原因独立保留；聚合只数 ``visible`` 的视图数，
不会用多数票把任何 ``occluded`` / ``uncertain`` 改成 ``visible``。
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
from .validation import (
    _require_finite_number,
    validate_depth_eps,
    validate_k,
    validate_point,
    validate_tolerance,
)

DEFAULT_ABS_TOL = 1e-6
"""默认绝对容差，与深度同单位。"""
DEFAULT_REL_TOL = 1e-6
"""默认相对容差（无量纲，乘以采样深度）。"""
DEFAULT_DEPTH_EPS = 0.0
"""默认投影深度不确定半径：0 表示点深度退化为单点（旧版语义）。"""


def _pixel_coords(cam: Camera, cx: float, cy: float, cz: float) -> tuple[float, float]:
    """针孔投影：相机坐标 -> 像素 (u=列, v=行)。调用前已确保 cz > 0。"""
    K = cam.K
    inv_z = 1.0 / cz
    u = (K[0][0] * cx + K[0][1] * cy) * inv_z + K[0][2]
    v = K[1][1] * cy * inv_z + K[1][2]
    return u, v


def _sample_bilinear(
    dm: DepthMap, u: float, v: float
) -> tuple[tuple[float, float] | None, Reason]:
    """四邻域采样。返回 ((near, far) 或 None, unknown 原因)。

    near/far 分别做双线性；near 层（``dm.values``）即旧的单点深度层。
    """
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
    w00 = (1.0 - du) * (1.0 - dv)
    w10 = du * (1.0 - dv)
    w01 = (1.0 - du) * dv
    w11 = du * dv

    def interp(layer: tuple[tuple[float, ...], ...]) -> float:
        return (
            w00 * layer[cv][cu]
            + w10 * layer[cv][c1]
            + w01 * layer[r1][cu]
            + w11 * layer[r1][c1]
        )

    return (interp(dm.values), interp(dm.far)), Reason.NONE


def classify_intervals(
    z_lo: float,
    z_hi: float,
    d_lo: float,
    d_hi: float,
    abs_tol: float = DEFAULT_ABS_TOL,
    rel_tol: float = DEFAULT_REL_TOL,
) -> Visibility:
    """两个闭深度区间的纯函数裁定（公开，便于独立核对）。

    点区间 ``[z_lo, z_hi]``、表面区间 ``[d_lo, d_hi]``（d_lo<=d_hi）。
    容差板在 near/far 端分别取 ``abs_tol + rel_tol*d`` 后膨胀。
    """
    for name, x in (
        ("z_lo", z_lo), ("z_hi", z_hi),
        ("d_lo", d_lo), ("d_hi", d_hi),
    ):
        _require_finite_number(x, name, InvalidParameterError)
    a_tol, r_tol = validate_tolerance(abs_tol, rel_tol)
    if z_lo > z_hi or d_lo > d_hi:
        raise InvalidParameterError(
            "interval lower bound must be <= upper bound: "
            f"point=[{z_lo}, {z_hi}], surface=[{d_lo}, {d_hi}]"
        )
    board_lo = d_lo - (a_tol + r_tol * d_lo)
    board_hi = d_hi + (a_tol + r_tol * d_hi)
    # 点区间整体落入容差板（含边界）-> visible。
    if z_lo >= board_lo and z_hi <= board_hi:
        return Visibility.VISIBLE
    # 严格分离（开区间意义）才是确定的遮挡/在前。
    if z_lo > board_hi:
        return Visibility.OCCLUDED
    if z_hi < board_lo:
        return Visibility.IN_FRONT
    # 部分重叠或恰好端点接触于板外边界 -> 不能下确定结论。
    return Visibility.UNCERTAIN


def classify_view(
    view: View,
    point: Sequence[float],
    abs_tol: float = DEFAULT_ABS_TOL,
    rel_tol: float = DEFAULT_REL_TOL,
    *,
    view_index: int = 0,
    depth_eps: float = DEFAULT_DEPTH_EPS,
) -> ViewStatus:
    """裁定单个世界点在单个视图下的状态。

    ``view_index`` 仅写入返回的 :class:`ViewStatus`，便于批次结果定位。
    ``depth_eps`` 为投影深度的不确定半径（>=0）：点深度按
    ``[cz-depth_eps, cz+depth_eps]`` 参与区间裁定；缺省 0 时与旧版
    单点深度行为完全一致。
    """
    if not isinstance(view, View):
        raise InvalidParameterError(
            f"view must be a visibility.View instance, got {type(view).__name__}"
        )
    p = validate_point(point)
    a_tol, r_tol = validate_tolerance(abs_tol, rel_tol)
    eps = validate_depth_eps(depth_eps)
    if isinstance(view_index, bool) or not isinstance(view_index, int):
        raise InvalidParameterError(
            f"view_index must be int, got {type(view_index).__name__}"
        )

    cx, cy, cz = view.camera.project(p)
    z_lo, z_hi = cz - eps, cz + eps

    # 第一步：深度区间相对相机原点的位置优先于一切投影/采样。
    if z_hi <= 0.0:
        return ViewStatus(
            view_index=view_index,
            visibility=Visibility.UNKNOWN,
            reason=Reason.BEHIND_CAMERA,
            projected_z=cz,
            sampled_depth=None,
            u=None,
            v=None,
            projected_z_low=z_lo,
            projected_z_high=z_hi,
        )
    if z_lo <= 0.0 < z_hi:
        # 区间跨过原点：透视投影 1/z 在原点无定义，不能合法投影。
        return ViewStatus(
            view_index=view_index,
            visibility=Visibility.UNKNOWN,
            reason=Reason.DEPTH_INTERVAL_AT_ORIGIN,
            projected_z=cz,
            sampled_depth=None,
            u=None,
            v=None,
            projected_z_low=z_lo,
            projected_z_high=z_hi,
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
            projected_z_low=z_lo,
            projected_z_high=z_hi,
        )

    d_lo, d_hi = sampled
    vis = classify_intervals(z_lo, z_hi, d_lo, d_hi, a_tol, r_tol)
    return ViewStatus(
        view_index=view_index,
        visibility=vis,
        reason=(
            Reason.DEPTH_INTERVAL_OVERLAP
            if vis is Visibility.UNCERTAIN
            else Reason.NONE
        ),
        projected_z=cz,
        sampled_depth=d_lo,
        u=u,
        v=v,
        projected_z_low=z_lo,
        projected_z_high=z_hi,
        sampled_depth_far=d_hi,
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
    *,
    depth_eps: float = DEFAULT_DEPTH_EPS,
) -> list[list[ViewStatus]]:
    """批量裁定。

    返回 ``results[point_index][view_index]`` 的 :class:`ViewStatus`
    矩阵。每个 (点, 视图) 状态与原因独立保留，不强制融合成二值。

    ``depth_eps`` 对全部 (点, 视图) 生效（投影深度不确定半径）。

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
    eps = validate_depth_eps(depth_eps)

    results: list[list[ViewStatus]] = []
    for pt in validated_points:
        row = [
            classify_view(
                view, pt, a_tol, r_tol, view_index=j, depth_eps=eps
            )
            for j, view in enumerate(views)
        ]
        results.append(row)
    return results


def count_visible_views(statuses: Sequence[ViewStatus]) -> int:
    """统计一个点的各视图状态中 ``visible`` 的数量。

    ``uncertain`` 不计为可见——区间证据不足时不允许凑数。
    """
    if not isinstance(statuses, Sequence) or isinstance(statuses, (str, bytes)):
        raise InvalidParameterError("statuses must be a sequence of ViewStatus")
    return sum(1 for s in statuses if s.visibility is Visibility.VISIBLE)


def indices_visible_in_at_least_k(
    results: Sequence[Sequence[ViewStatus]], k: int
) -> list[int]:
    """聚合：返回至少在 ``k`` 个视图中 ``visible`` 的点下标。

    只计 ``visible``；``occluded`` / ``in_front`` / ``uncertain`` /
    ``unknown`` 均不计，不做跨视图投票翻转。
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
