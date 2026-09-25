"""版本化可见性缓存：深度/mask 局部更新时只重算受影响的 (点, 相机)。

固定相机集合、世界点集合与 k，缓存每个 (点, 相机) 的
:class:`~visibility.types.ViewStatus`、每点可见计数与入选集合。

反向依赖
--------
首次裁定每个 (点, 相机) 时，若投影 stencil 落在图内，则把该对登记为
其**实际存在的**四邻域像素的反向依赖：

- 相机后方（``cz <= 0``）：不投影，不登记任何像素；
- 投影整体出界：不登记不存在的像素；
- stencil 落在最后一行/列而缺 +1 邻点：登记界内实际存在的角点；
- 因像素被 mask 禁用而 ``unknown/masked_pixel`` 的对同样登记四个角点，
  像素恢复有效后该对才会被重新裁定。

原子性与版本
------------
一次 :meth:`VisibilityCache.apply_updates` 先在**旧状态**上验证全部
坐标、版本与数值，全部通过后才构造新的深度图/mask，重算受影响对并
一次性发布；失败时缓存状态（含计数与入选集合）保持不变。每个相机
维护独立整数版本，替换深度或实际改动 mask 后该相机版本 +1。旧快照
是不可变对象，发布新版本后仍可查询。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Union

from .core import _pixel_coords, classify_view
from .errors import (
    InvalidParameterError,
    InvalidUpdateError,
    StaleVersionError,
)
from .types import Camera, DepthMap, View, ViewStatus, Visibility
from .validation import validate_k, validate_point, validate_tolerance


def _validate_index(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidParameterError(
            f"{path} must be an int, got {type(value).__name__}"
        )
    if value < 0:
        raise InvalidParameterError(f"{path} must be >= 0, got {value}")
    return value


@dataclass(frozen=True)
class DepthUpdate:
    """整体替换某相机的深度图（含其 mask），尺寸必须与原图一致。"""

    view_index: int
    depth_map: DepthMap
    expected_version: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.depth_map, DepthMap):
            raise InvalidParameterError(
                "DepthUpdate.depth_map must be a visibility.DepthMap, "
                f"got {type(self.depth_map).__name__}"
            )
        _validate_index(self.view_index, "DepthUpdate.view_index")
        _validate_index(self.expected_version, "DepthUpdate.expected_version")


@dataclass(frozen=True)
class MaskUpdate:
    """批量修改某相机若干像素的启用/禁用状态。

    ``pixels`` 每项为 ``(row, col, enabled)``；``enabled=False`` 表示
    禁用该像素，``True`` 表示恢复启用。
    """

    view_index: int
    pixels: tuple[tuple[int, int, bool], ...]
    expected_version: int

    def __init__(
        self,
        view_index: int,
        pixels: Sequence[tuple[int, int, bool]],
        expected_version: int = 0,
    ) -> None:
        if not isinstance(pixels, Sequence) or isinstance(pixels, (str, bytes)):
            raise InvalidParameterError("MaskUpdate.pixels must be a sequence")
        norm: list[tuple[int, int, bool]] = []
        seen: set[tuple[int, int]] = set()
        for i, px in enumerate(pixels):
            if (
                not isinstance(px, Sequence)
                or isinstance(px, (str, bytes))
                or len(px) != 3
            ):
                raise InvalidUpdateError(
                    f"MaskUpdate.pixels[{i}] must be a (row, col, enabled) triple"
                )
            row = _validate_index(px[0], f"MaskUpdate.pixels[{i}][0] (row)")
            col = _validate_index(px[1], f"MaskUpdate.pixels[{i}][1] (col)")
            enabled = px[2]
            if not isinstance(enabled, bool):
                raise InvalidUpdateError(
                    f"MaskUpdate.pixels[{i}][2] (enabled) must be bool, "
                    f"got {type(enabled).__name__}"
                )
            if (row, col) in seen:
                raise InvalidUpdateError(
                    f"MaskUpdate.pixels duplicates pixel (row={row}, col={col})"
                )
            seen.add((row, col))
            norm.append((row, col, enabled))
        _validate_index(view_index, "MaskUpdate.view_index")
        _validate_index(expected_version, "MaskUpdate.expected_version")
        object.__setattr__(self, "view_index", view_index)
        object.__setattr__(self, "pixels", tuple(norm))
        object.__setattr__(self, "expected_version", expected_version)


Update = Union[DepthUpdate, MaskUpdate]


@dataclass(frozen=True)
class UpdateReport:
    """一次原子更新的结果。

    - ``recomputed``：实际重算的 ``(point_index, view_index)`` 集合；
    - ``versions``：发布后每个相机的版本；
    - ``changed_pixels``：每个相机实际取值/状态发生变化的像素。
    """

    recomputed: frozenset[tuple[int, int]]
    versions: tuple[int, ...]
    changed_pixels: tuple[frozenset[tuple[int, int]], ...]


@dataclass(frozen=True)
class VisibilitySnapshot:
    """某一时刻缓存状态的不可变快照，旧快照在后续更新后仍可查询。"""

    statuses: tuple[tuple[ViewStatus, ...], ...]
    visible_counts: tuple[int, ...]
    selected: frozenset[int]
    k: int
    versions: tuple[int, ...]

    def status(self, point_index: int, view_index: int) -> ViewStatus:
        return self.statuses[point_index][view_index]

    def visible_count(self, point_index: int) -> int:
        return self.visible_counts[point_index]

    def selected_indices(self) -> tuple[int, ...]:
        return tuple(sorted(self.selected))


class VisibilityCache:
    """固定相机/世界点/k 的增量可见性缓存。"""

    def __init__(
        self,
        cameras: Sequence[Camera],
        depth_maps: Sequence[DepthMap],
        points: Sequence[Sequence[float]],
        k: int,
        abs_tol: float = 1e-6,
        rel_tol: float = 1e-6,
    ) -> None:
        if not isinstance(cameras, Sequence) or isinstance(cameras, (str, bytes)):
            raise InvalidParameterError("cameras must be a sequence of Camera")
        if len(cameras) == 0:
            raise InvalidParameterError("cameras must contain at least one camera")
        for j, cam in enumerate(cameras):
            if not isinstance(cam, Camera):
                raise InvalidParameterError(
                    f"cameras[{j}] must be a visibility.Camera, "
                    f"got {type(cam).__name__}"
                )
        if not isinstance(depth_maps, Sequence) or isinstance(
            depth_maps, (str, bytes)
        ):
            raise InvalidParameterError("depth_maps must be a sequence of DepthMap")
        if len(depth_maps) != len(cameras):
            raise InvalidParameterError(
                f"depth_maps has {len(depth_maps)} entries, expected {len(cameras)}"
            )
        for j, dm in enumerate(depth_maps):
            if not isinstance(dm, DepthMap):
                raise InvalidParameterError(
                    f"depth_maps[{j}] must be a visibility.DepthMap, "
                    f"got {type(dm).__name__}"
                )
        if not isinstance(points, Sequence) or isinstance(points, (str, bytes)):
            raise InvalidParameterError("points must be a sequence of 3D points")
        validated_points = [
            validate_point(pt, f"points[{i}]") for i, pt in enumerate(points)
        ]
        self._k = validate_k(k, len(cameras))
        self._abs_tol, self._rel_tol = validate_tolerance(abs_tol, rel_tol)

        self._cameras: tuple[Camera, ...] = tuple(cameras)
        self._depth_maps: list[DepthMap] = list(depth_maps)
        self._points: tuple[tuple[float, float, float], ...] = tuple(
            validated_points
        )
        self._versions: list[int] = [0 for _ in self._cameras]

        n_views = len(self._cameras)
        n_points = len(self._points)
        self._statuses: list[list[ViewStatus | None]] = [
            [None] * n_views for _ in range(n_points)
        ]
        self._deps: list[dict[tuple[int, int], set[int]]] = [
            {} for _ in range(n_views)
        ]
        for j, cam in enumerate(self._cameras):
            view = View(cam, self._depth_maps[j])
            for p in range(n_points):
                self._register(p, j)
                self._statuses[p][j] = classify_view(
                    view,
                    self._points[p],
                    self._abs_tol,
                    self._rel_tol,
                    view_index=j,
                )
        self._visible_counts = [
            sum(
                1
                for j in range(n_views)
                if self._statuses[p][j].visibility is Visibility.VISIBLE
            )
            for p in range(n_points)
        ]

    # ---- 基本只读信息 -------------------------------------------------

    @property
    def k(self) -> int:
        return self._k

    @property
    def versions(self) -> tuple[int, ...]:
        return tuple(self._versions)

    @property
    def n_views(self) -> int:
        return len(self._cameras)

    @property
    def n_points(self) -> int:
        return len(self._points)

    def depth_map(self, view_index: int) -> DepthMap:
        self._check_view_index(view_index)
        return self._depth_maps[view_index]

    def dependencies(
        self, view_index: int
    ) -> dict[tuple[int, int], frozenset[int]]:
        """调试/测试用：返回某相机 像素 -> 依赖该像素的点下标集合。"""
        self._check_view_index(view_index)
        return {
            pixel: frozenset(pts) for pixel, pts in self._deps[view_index].items()
        }

    # ---- 查询 ---------------------------------------------------------

    def _check_view_index(self, view_index: int) -> None:
        _validate_index(view_index, "view_index")
        if not 0 <= view_index < self.n_views:
            raise InvalidParameterError(
                f"view_index must be in [0, {self.n_views}), got {view_index}"
            )

    def _check_point_index(self, point_index: int) -> None:
        _validate_index(point_index, "point_index")
        if not 0 <= point_index < self.n_points:
            raise InvalidParameterError(
                f"point_index must be in [0, {self.n_points}), got {point_index}"
            )

    def status(self, point_index: int, view_index: int) -> ViewStatus:
        self._check_point_index(point_index)
        self._check_view_index(view_index)
        return self._statuses[point_index][view_index]

    def visible_count(self, point_index: int) -> int:
        self._check_point_index(point_index)
        return self._visible_counts[point_index]

    def selected_indices(self) -> tuple[int, ...]:
        return tuple(
            p for p, cnt in enumerate(self._visible_counts) if cnt >= self._k
        )

    def snapshot(self) -> VisibilitySnapshot:
        """捕获当前状态的不可变快照（旧快照后续仍可查询）。"""
        selected = frozenset(
            p for p, cnt in enumerate(self._visible_counts) if cnt >= self._k
        )
        return VisibilitySnapshot(
            statuses=tuple(
                tuple(self._statuses[p][j] for j in range(self.n_views))
                for p in range(self.n_points)
            ),
            visible_counts=tuple(self._visible_counts),
            selected=selected,
            k=self._k,
            versions=tuple(self._versions),
        )

    # ---- 反向依赖 -----------------------------------------------------

    def _stencil_pixels(
        self, view_index: int, point_index: int
    ) -> set[tuple[int, int]]:
        """返回该 (点,相机) stencil 在界内实际存在的像素集合。

        相机后方或投影整体出界时返回空集（不登记不存在的像素）。
        """
        cam = self._cameras[view_index]
        dm = self._depth_maps[view_index]
        cx, cy, cz = cam.project(self._points[point_index])
        if cz <= 0.0:
            return set()
        u, v = _pixel_coords(cam, cx, cy, cz)
        h, w = dm.height, dm.width
        if u < 0.0 or v < 0.0 or u >= w or v >= h:
            return set()
        cu, cv = math.floor(u), math.floor(v)
        pixels: set[tuple[int, int]] = set()
        for r in (cv, cv + 1):
            for c in (cu, cu + 1):
                if 0 <= r < h and 0 <= c < w:
                    pixels.add((r, c))
        return pixels

    def _register(self, point_index: int, view_index: int) -> None:
        table = self._deps[view_index]
        for px in self._stencil_pixels(view_index, point_index):
            table.setdefault(px, set()).add(point_index)

    # ---- 原子增量更新 -------------------------------------------------

    def apply_updates(self, updates: Sequence[Update]) -> UpdateReport:
        """按版本批量替换深度或修改 mask，原子发布。

        先在旧状态上验证全部坐标、版本与数值，再构造新数据、重算受影响
        对，最后一次性发布；任何校验失败都不改变缓存。
        """
        if not isinstance(updates, Sequence) or isinstance(
            updates, (str, bytes)
        ):
            raise InvalidParameterError("updates must be a sequence")

        # 1) 类型检查：每相机至多一个更新。
        ops: dict[int, Update] = {}
        for i, upd in enumerate(updates):
            if not isinstance(upd, (DepthUpdate, MaskUpdate)):
                raise InvalidUpdateError(
                    f"updates[{i}] must be DepthUpdate or MaskUpdate, "
                    f"got {type(upd).__name__}"
                )
            j = upd.view_index
            if not 0 <= j < self.n_views:
                raise InvalidUpdateError(
                    f"updates[{i}].view_index must be in [0, {self.n_views}), "
                    f"got {j}"
                )
            if j in ops:
                raise InvalidUpdateError(
                    f"updates contains multiple operations for view_index={j}; "
                    "merge them into one update"
                )
            # 2) 版本检查（过期/超前均拒绝）。
            if upd.expected_version != self._versions[j]:
                raise StaleVersionError(
                    f"updates[{i}] targets view_index={j} with "
                    f"expected_version={upd.expected_version}, but current "
                    f"version is {self._versions[j]}"
                )
            ops[j] = upd

        # 3) 数值与坐标检查，全部基于旧快照；同时推导变化像素。
        new_maps: dict[int, DepthMap] = {}
        changed: dict[int, frozenset[tuple[int, int]]] = {}
        for j, upd in ops.items():
            old = self._depth_maps[j]
            if isinstance(upd, DepthUpdate):
                new_dm = upd.depth_map
                if (new_dm.height, new_dm.width) != (old.height, old.width):
                    raise InvalidUpdateError(
                        f"DepthUpdate for view_index={j} has shape "
                        f"({new_dm.height},{new_dm.width}), expected "
                        f"({old.height},{old.width}); camera/points are fixed"
                    )
                changed_set = {
                    (r, c)
                    for r in range(old.height)
                    for c in range(old.width)
                    if new_dm.values[r][c] != old.values[r][c]
                    or new_dm.disabled[r][c] != old.disabled[r][c]
                }
            else:
                for r, c, _enabled in upd.pixels:
                    if r >= old.height or c >= old.width:
                        raise InvalidUpdateError(
                            f"MaskUpdate for view_index={j} addresses pixel "
                            f"(row={r}, col={c}) outside depth map of shape "
                            f"({old.height},{old.width})"
                        )
                mask_rows = [list(row) for row in old.disabled]
                changed_set = set()
                for r, c, enabled in upd.pixels:
                    if bool(mask_rows[r][c]) != (not enabled):
                        changed_set.add((r, c))
                    mask_rows[r][c] = not enabled
                new_dm = DepthMap(old.values, mask_rows)
            new_maps[j] = new_dm
            changed[j] = frozenset(changed_set)

        # 4) 由像素 -> 依赖点集合，求受影响 (点, 相机) 对。
        affected: set[tuple[int, int]] = set()
        for j, changed_set in changed.items():
            table = self._deps[j]
            for px in changed_set:
                affected.update((p, j) for p in table.get(px, ()))

        # 5) 重算到新表（所有新对象已构造完毕，之后才触碰缓存自身）。
        new_statuses: dict[tuple[int, int], ViewStatus] = {}
        for p, j in affected:
            new_statuses[(p, j)] = classify_view(
                View(self._cameras[j], new_maps[j]),
                self._points[p],
                self._abs_tol,
                self._rel_tol,
                view_index=j,
            )

        # 6) 原子发布。
        for (p, j), st in new_statuses.items():
            old_vis = self._statuses[p][j].visibility is Visibility.VISIBLE
            new_vis = st.visibility is Visibility.VISIBLE
            if old_vis != new_vis:
                self._visible_counts[p] += 1 if new_vis else -1
            self._statuses[p][j] = st
        for j, new_dm in new_maps.items():
            # 每个显式更新操作都推进该相机版本，即使像素值恰好不变。
            self._depth_maps[j] = new_dm
            self._versions[j] += 1

        changed_pixels = tuple(
            changed.get(j, frozenset()) for j in range(self.n_views)
        )
        return UpdateReport(
            recomputed=frozenset(affected),
            versions=tuple(self._versions),
            changed_pixels=changed_pixels,
        )
