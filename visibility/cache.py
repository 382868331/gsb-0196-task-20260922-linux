"""版本化可见性缓存：固定相机、世界点与 k，按版本批量替换深度/mask。

设计要点
--------

- 构造时固定全部相机（内参/外参）、世界点、容差与 ``k``，并从头按
  原公式（:func:`visibility.core.classify_view`）全量计算一次，
  建立 **像素 -> 实际查询该像素四邻域的 (点, 相机) 对** 的反向依赖。
- 相机与点固定，投影位置不变，因此依赖关系在构造后保持不变；
  深度/mask 更新只改变裁定结果。
- 因像素被禁用（masked）或缺邻域（missing_neighbor）而为 unknown
  的查询 **同样登记四邻域依赖**，否则像素恢复有效后永远不会被重算。
- 相机后方（behind_camera）与投影出界（out_of_bounds）不登记任何
  像素依赖；stencil 中超出图像边界的“不存在的像素”也不登记。
- :meth:`VisibilityCache.update` 先整体校验全部相机下标、像素坐标、
  版本号与数值，再原子发布：校验失败时旧快照保持可查询，不留下
  半份计数；也绝不通过清空全部缓存来掩盖漏依赖。
- 每次更新只重算受影响的 (点, 相机) 对，并增量维护每点可见计数与
  “至少 k 视图可见”的入选集合；``last_recomputed`` 暴露本次实际
  重算的对列表，供测试核对局部性。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .core import (
    DEFAULT_ABS_TOL,
    DEFAULT_REL_TOL,
    _pixel_coords,
    _validate_views,
    classify_view,
)
from .errors import (
    InvalidDepthMapError,
    InvalidParameterError,
    VersionMismatchError,
)
from .types import DepthMap, View, ViewStatus, Visibility
from .validation import validate_k, validate_point, validate_tolerance


@dataclass(frozen=True)
class CameraDepthUpdate:
    """单相机深度替换描述（一批更新中的一项）。

    - ``camera_index``: 目标相机下标（构造时 ``views`` 的序号）。
    - ``values``: 完整替换的新深度值（行主序，形状可与旧图不同，
      仍需满足 1..32 x 1..32 且为有限正数）；``None`` 表示不替换。
    - ``disabled``: 完整替换的新禁用 mask（形状须与替换后的深度图
      一致）；``None`` 表示不替换。
    """

    camera_index: int
    values: object = None
    disabled: object = None


class VisibilityCache:
    """固定相机/点/k 的可见性缓存，支持按版本批量局部更新。"""

    def __init__(
        self,
        views: Sequence[View],
        points: Sequence[Sequence[float]],
        k: int,
        abs_tol: float = DEFAULT_ABS_TOL,
        rel_tol: float = DEFAULT_REL_TOL,
    ) -> None:
        views = _validate_views(views)
        if not isinstance(points, Sequence) or isinstance(points, (str, bytes)):
            raise InvalidParameterError("points must be a sequence of 3D points")
        if len(points) == 0:
            raise InvalidParameterError("points must contain at least one point")
        validated_points = [
            validate_point(pt, f"points[{i}]") for i, pt in enumerate(points)
        ]
        a_tol, r_tol = validate_tolerance(abs_tol, rel_tol)
        kk = validate_k(k, len(views))

        self._views: tuple[View, ...] = tuple(views)
        self._cameras = tuple(v.camera for v in self._views)
        self._points = tuple(validated_points)
        self._k = kk
        self._abs_tol = a_tol
        self._rel_tol = r_tol
        self._depth_maps: tuple[DepthMap, ...] = tuple(
            v.depth_map for v in self._views
        )

        # 反向依赖：deps[view_index][(row, col)] = 查询该像素的点下标集合。
        # 只登记真实存在的像素；投影位置固定，构造后不再变化。
        deps: list[dict[tuple[int, int], set[int]]] = [
            {} for _ in self._views
        ]
        statuses: list[list[ViewStatus]] = []
        for pi, pt in enumerate(self._points):
            row: list[ViewStatus] = []
            for vi, view in enumerate(self._views):
                st = classify_view(view, pt, a_tol, r_tol, view_index=vi)
                row.append(st)
                for pixel in self._stencil_pixels(view.depth_map, pt, vi):
                    deps[vi].setdefault(pixel, set()).add(pi)
            statuses.append(row)

        self._deps = deps
        self._statuses: list[list[ViewStatus]] = statuses
        self._counts: list[int] = [
            sum(1 for s in row if s.visibility is Visibility.VISIBLE)
            for row in statuses
        ]
        self._selected: set[int] = {
            i for i, c in enumerate(self._counts) if c >= self._k
        }
        self._version = 0
        self._last_recomputed: tuple[tuple[int, int], ...] = ()

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _stencil_pixels(
        self, dm: DepthMap, point: tuple[float, float, float], view_index: int
    ) -> tuple[tuple[int, int], ...]:
        """该点在该视图下实际依赖的、真实存在的四邻域像素。

        相机后方或投影出界返回空（不登记不存在的像素）；stencil 越界
        方向上的像素不登记，界内部分照常登记（包括被 mask 禁用的）。
        """
        cam = self._cameras[view_index]
        cx, cy, cz = cam.project(point)
        if cz <= 0.0:
            return ()
        u, v = _pixel_coords(cam, cx, cy, cz)
        h, w = dm.height, dm.width
        if u < 0.0 or v < 0.0 or u >= w or v >= h:
            return ()
        cu = math.floor(u)
        cv = math.floor(v)
        pixels = []
        for r in (cv, cv + 1):
            for c in (cu, cu + 1):
                if 0 <= r < h and 0 <= c < w:
                    pixels.append((r, c))
        return tuple(pixels)

    def _check_point_index(self, point_index: object) -> int:
        if isinstance(point_index, bool) or not isinstance(point_index, int):
            raise InvalidParameterError(
                f"point_index must be int, got {type(point_index).__name__}"
            )
        if not 0 <= point_index < len(self._points):
            raise InvalidParameterError(
                f"point_index must be in [0, {len(self._points) - 1}], "
                f"got {point_index}"
            )
        return point_index

    def _check_view_index(self, view_index: object) -> int:
        if isinstance(view_index, bool) or not isinstance(view_index, int):
            raise InvalidParameterError(
                f"view_index must be int, got {type(view_index).__name__}"
            )
        if not 0 <= view_index < len(self._views):
            raise InvalidParameterError(
                f"view_index must be in [0, {len(self._views) - 1}], "
                f"got {view_index}"
            )
        return view_index

    # ------------------------------------------------------------------
    # 查询（对任意已发布版本均一致）
    # ------------------------------------------------------------------

    @property
    def version(self) -> int:
        """当前已发布快照的版本号（构造后为 0，每次成功更新 +1）。"""
        return self._version

    @property
    def k(self) -> int:
        return self._k

    @property
    def last_recomputed(self) -> tuple[tuple[int, int], ...]:
        """最近一次更新实际重算的 ``(point_index, view_index)`` 对。"""
        return self._last_recomputed

    def status(self, point_index: int, view_index: int) -> ViewStatus:
        """查询当前版本中某点在某相机下的裁定。"""
        pi = self._check_point_index(point_index)
        vi = self._check_view_index(view_index)
        return self._statuses[pi][vi]

    def visible_count(self, point_index: int) -> int:
        """当前版本中该点被判 visible 的视图数。"""
        pi = self._check_point_index(point_index)
        return self._counts[pi]

    def selected(self) -> list[int]:
        """当前版本中可见视图数 >= k 的点下标（升序）。"""
        return sorted(self._selected)

    # ------------------------------------------------------------------
    # 批量更新：先全量校验，再原子发布
    # ------------------------------------------------------------------

    def update(
        self,
        updates: Sequence[CameraDepthUpdate],
        *,
        expected_version: int,
    ) -> int:
        """按版本批量替换若干相机的深度值/禁用 mask，返回新版本号。

        - ``expected_version`` 必须等于当前版本号，否则抛
          :class:`VersionMismatchError`（过期版本拒绝）。
        - 同一批中同一相机只允许出现一次。
        - 先校验全部下标、形状（像素坐标隐含于形状）、数值与版本，
          再原子发布；校验失败时旧快照完全不变。
        """
        # --- 版本校验 -------------------------------------------------
        if isinstance(expected_version, bool) or not isinstance(
            expected_version, int
        ):
            raise InvalidParameterError(
                "expected_version must be int, got "
                f"{type(expected_version).__name__}"
            )
        if expected_version != self._version:
            raise VersionMismatchError(
                f"stale update: expected_version={expected_version} but "
                f"current version is {self._version}"
            )

        # --- 结构校验 -------------------------------------------------
        if not isinstance(updates, Sequence) or isinstance(updates, (str, bytes)):
            raise InvalidParameterError(
                "updates must be a sequence of CameraDepthUpdate"
            )
        seen: set[int] = set()
        prepared: dict[int, DepthMap] = {}
        for i, upd in enumerate(updates):
            if not isinstance(upd, CameraDepthUpdate):
                raise InvalidParameterError(
                    f"updates[{i}] must be a CameraDepthUpdate, got "
                    f"{type(upd).__name__}"
                )
            vi = upd.camera_index
            if isinstance(vi, bool) or not isinstance(vi, int):
                raise InvalidParameterError(
                    f"updates[{i}].camera_index must be int, got "
                    f"{type(vi).__name__}"
                )
            if not 0 <= vi < len(self._views):
                raise InvalidParameterError(
                    f"updates[{i}].camera_index must be in "
                    f"[0, {len(self._views) - 1}], got {vi}"
                )
            if vi in seen:
                raise InvalidParameterError(
                    f"updates[{i}].camera_index={vi} appears more than once "
                    "in the same batch"
                )
            seen.add(vi)
            if upd.values is None and upd.disabled is None:
                raise InvalidParameterError(
                    f"updates[{i}] must replace values and/or disabled"
                )
            old = self._depth_maps[vi]
            new_values = old.values if upd.values is None else upd.values
            new_disabled = (
                old.disabled if upd.disabled is None else upd.disabled
            )
            # DepthMap 构造即完整校验形状/数值/布尔性，错误可定位。
            try:
                prepared[vi] = DepthMap(new_values, new_disabled)
            except InvalidDepthMapError as exc:
                raise InvalidDepthMapError(
                    f"updates[{i}] (camera {vi}): {exc}"
                ) from exc

        # --- 计算受影响像素与 (点, 相机) 对 ---------------------------
        affected: set[tuple[int, int]] = set()
        new_depth_maps = list(self._depth_maps)
        for vi, new_dm in prepared.items():
            old_dm = self._depth_maps[vi]
            changed_pixels: set[tuple[int, int]] = set()
            if (new_dm.height, new_dm.width) != (old_dm.height, old_dm.width):
                # 形状变化会改变 stencil 的界内部分：该相机所有已登记
                # 依赖像素都可能受影响，全部波及（仍只重算登记过的对）。
                for pixel, pts in self._deps[vi].items():
                    for pi in pts:
                        affected.add((pi, vi))
            else:
                for r in range(new_dm.height):
                    for c in range(new_dm.width):
                        if (
                            new_dm.values[r][c] != old_dm.values[r][c]
                            or new_dm.disabled[r][c] != old_dm.disabled[r][c]
                        ):
                            changed_pixels.add((r, c))
                for pixel in changed_pixels:
                    for pi in self._deps[vi].get(pixel, ()):
                        affected.add((pi, vi))
            new_depth_maps[vi] = new_dm

        # --- 只重算受影响的对（用新深度图，相机/点不变） ---------------
        recomputed: list[tuple[int, int]] = sorted(affected)
        new_statuses = [row[:] for row in self._statuses]
        new_counts = list(self._counts)
        new_selected = set(self._selected)
        for pi, vi in recomputed:
            view = View(self._cameras[vi], new_depth_maps[vi])
            st = classify_view(
                view, self._points[pi], self._abs_tol, self._rel_tol,
                view_index=vi,
            )
            old_vis = new_statuses[pi][vi].visibility is Visibility.VISIBLE
            new_vis = st.visibility is Visibility.VISIBLE
            new_statuses[pi][vi] = st
            if old_vis != new_vis:
                new_counts[pi] += 1 if new_vis else -1
                if new_counts[pi] >= self._k:
                    new_selected.add(pi)
                else:
                    new_selected.discard(pi)

        # --- 原子发布：全部计算完成后一次性交换 ------------------------
        self._depth_maps = tuple(new_depth_maps)
        self._statuses = new_statuses
        self._counts = new_counts
        self._selected = new_selected
        self._version += 1
        self._last_recomputed = tuple(recomputed)
        return self._version
