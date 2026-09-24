"""公开数据结构。

全部为不可值对象（``frozen=True``），深度图在构造时做防御性拷贝，
外部后续修改传入的列表不会影响已构造的 :class:`DepthMap`。

相机约定
--------
世界点 ``P = (X, Y, Z)`` 先经世界->相机刚体变换::

    p_cam = R @ P + t

其中 ``p_cam = (cx, cy, cz)``。**深度就是相机坐标 z 分量** ``cz``。
针孔内参 ``K``（行主序 3x3）把 *已除以深度* 的相机坐标映射到像素::

    u = K00 * cx / cz + K01 * cy / cz + K02
    v = K11 * cy / cz + K12
    (w = 1)

像素中心位于整数坐标 (u, v)；深度图像素 (row=v, col=u)。因此
``K02`` 为列方向主点、``K12`` 为行方向主点。``K[2]`` 必须恰为
``(0, 0, 1)``；``K01``（斜切）允许非零。
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import TypeAlias

from .errors import InvalidCameraError, InvalidDepthMapError

Vec3: TypeAlias = tuple[float, float, float]
Mat3: TypeAlias = tuple[Vec3, Vec3, Vec3]

MAX_DEPTH_WIDTH = 32
"""每张深度图允许的最大列数（接口边界）。"""
MAX_DEPTH_HEIGHT = 32
"""每张深度图允许的最大行数（接口边界）。"""


class Visibility(enum.Enum):
    """单视图对单点的裁定结果。"""

    VISIBLE = "visible"
    """投影深度区间整体落入表面深度区间（含容差膨胀）内。"""

    OCCLUDED = "occluded"
    """点在表面后方（投影深度区间整体明显远于表面 far 端）。"""

    IN_FRONT = "in_front"
    """点在表面前方（投影深度区间整体明显近于表面 near 端）。"""

    UNCERTAIN = "uncertain"
    """投影深度区间与表面区间部分重叠/接触：既不能确定可见，也不能
    确定遮挡。重叠区间绝不直接判 visible。"""

    UNKNOWN = "unknown"
    """点在相机后方、投影深度区间退化到原点、投影越界、插值邻域
    缺失/被禁用。"""


class Reason(enum.Enum):
    """裁定的细分原因。

    - ``NONE``：visible / occluded / in_front 等确定裁定；
    - ``DEPTH_INTERVAL_OVERLAP``：``UNCERTAIN``（区间重叠或仅端点接触）；
    - 其余成员对应 ``UNKNOWN`` 的成因。
    """

    NONE = "none"
    DEPTH_INTERVAL_OVERLAP = "depth_interval_overlap"
    BEHIND_CAMERA = "behind_camera"
    DEPTH_INTERVAL_AT_ORIGIN = "depth_interval_at_origin"
    OUT_OF_BOUNDS = "out_of_bounds"
    MISSING_NEIGHBOR = "missing_neighbor"
    MASKED_PIXEL = "masked_pixel"


@dataclass(frozen=True)
class Camera:
    """针孔相机：3x3 内参 ``K``、3x3 旋转 ``R``、3 维平移 ``t``。"""

    K: Mat3
    R: Mat3
    t: Vec3

    def __post_init__(self) -> None:
        from .validation import _require_mat3, _require_vec3, validate_camera

        K = _require_mat3(self.K, "K", InvalidCameraError)
        R = _require_mat3(self.R, "R", InvalidCameraError)
        t = _require_vec3(self.t, "t", InvalidCameraError)
        object.__setattr__(self, "K", K)
        object.__setattr__(self, "R", R)
        object.__setattr__(self, "t", t)
        validate_camera(self)

    def project(self, point: Vec3) -> tuple[float, float, float]:
        """投影世界点，返回相机坐标 ``(cx, cy, cz)``（cz 即深度）。

        纯刚体变换与针孔线性部分，不做 ``cz > 0`` 判断；是否在相机
        后方由裁定逻辑处理。
        """
        r0, r1, r2 = self.R
        x = r0[0] * point[0] + r0[1] * point[1] + r0[2] * point[2] + self.t[0]
        y = r1[0] * point[0] + r1[1] * point[1] + r1[2] * point[2] + self.t[1]
        z = r2[0] * point[0] + r2[1] * point[1] + r2[2] * point[2] + self.t[2]
        return x, y, z


@dataclass(frozen=True)
class DepthMap:
    """相机深度图。

    - ``values``: 行主序二维序列，``values[row][col]`` 为该像素处表面
      的相机 z（深度）。尺寸不超过 32x32。
    - ``disabled``: 与 ``values`` 同形状的布尔二维序列；``True`` 表示
      该像素被显式禁用。默认为全 False。
    - ``far``: 可选的逐像素表面**远端**深度（相机 z）。给出后该像素
      表面占据闭区间 ``[values[row][col], far[row][col]]``；缺省时
      区间退化为单点 ``values``，即旧版点深度语义。

    另见 :meth:`from_intervals`——以 (near, far) 两张图构造区间深度图。
    """

    values: tuple[tuple[float, ...], ...] = field(init=False)
    disabled: tuple[tuple[bool, ...], ...] = field(init=False)
    far: tuple[tuple[float, ...], ...] = field(init=False)
    """逐像素远端深度；与 ``values`` 同形，逐元素 ``far >= values``。"""

    def __init__(
        self,
        values: SequenceLike,
        disabled: SequenceLike | None = None,
        far: SequenceLike | None = None,
    ) -> None:
        from .validation import _freeze_depth

        vals, mask, fars = _freeze_depth(values, disabled, far)
        object.__setattr__(self, "values", vals)
        object.__setattr__(self, "disabled", mask)
        object.__setattr__(self, "far", fars)

    @classmethod
    def from_intervals(
        cls,
        near: SequenceLike,
        far: SequenceLike,
        disabled: SequenceLike | None = None,
    ) -> DepthMap:
        """以逐像素的近端/远端深度构造区间深度图。

        ``near[row][col] <= far[row][col]``（必须严格有限且为正）；
        两者相等即该像素退化为单点深度。
        """
        return cls(near, disabled, far)

    @property
    def is_interval(self) -> bool:
        """是否至少有一个像素的 near < far（非退化区间）。"""
        for r in range(len(self.values)):
            vr = self.values[r]
            fr = self.far[r]
            for c in range(len(vr)):
                if fr[c] != vr[c]:
                    return True
        return False

    @property
    def height(self) -> int:
        return len(self.values)

    @property
    def width(self) -> int:
        return len(self.values[0]) if self.values else 0


@dataclass(frozen=True)
class View:
    """一个相机与其深度图、禁用 mask 的绑定。"""

    camera: Camera
    depth_map: DepthMap

    def __post_init__(self) -> None:
        if not isinstance(self.camera, Camera):
            raise InvalidCameraError(
                "view.camera must be a visibility.Camera instance"
            )
        if not isinstance(self.depth_map, DepthMap):
            raise InvalidDepthMapError(
                "view.depth_map must be a visibility.DepthMap instance"
            )


@dataclass(frozen=True)
class ViewStatus:
    """一个点在一个视图下的完整裁定。"""

    view_index: int
    visibility: Visibility
    reason: Reason
    projected_z: float | None
    """相机坐标 z（点区间的中点）；``behind_camera`` 时仍给出（<=0），
    其余无法投影时为 None。"""
    sampled_depth: float | None
    """双线性采样的表面**近端**深度；退化（单点）深度图即表面深度
    本身。无法采样时为 None。"""
    u: float | None
    """像素列坐标；无法计算时为 None。"""
    v: float | None
    """像素行坐标；无法计算时为 None。"""
    projected_z_low: float | None = None
    """投影深度区间近端 ``cz - depth_eps``（区间可能退化）；
    ``behind_camera`` 时仍给出，其余无法投影时为 None。"""
    projected_z_high: float | None = None
    """投影深度区间远端 ``cz + depth_eps``。"""
    sampled_depth_far: float | None = None
    """双线性采样的表面**远端**深度；退化深度图时等于
    ``sampled_depth``；无法采样时为 None。"""

    @property
    def depth_interval(self) -> tuple[float, float] | None:
        """表面采样深度区间 ``[near, far]``；无法采样时为 None。"""
        if self.sampled_depth is None or self.sampled_depth_far is None:
            return None
        return self.sampled_depth, self.sampled_depth_far

    @property
    def projected_interval(self) -> tuple[float, float] | None:
        """投影深度区间 ``[cz-eps, cz+eps]``；点整体在相机后方时仍给出
        （两端 <=0），其他无法投影的情形为 None。"""
        if self.projected_z_low is None or self.projected_z_high is None:
            return None
        return self.projected_z_low, self.projected_z_high


# 构造时接受的“类序列”输入（list/tuple 等可下标、可迭代的有限容器）。
SequenceLike: TypeAlias = object
