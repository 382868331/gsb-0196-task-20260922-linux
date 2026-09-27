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

正交模型（``Camera.model is CameraModel.ORTHOGRAPHIC``）不除深度::

    u = K00 * cx + K01 * cy + K02
    v = K11 * cy + K12

此时 ``K00/K11`` 为像素/世界长度比例尺。两种模型深度均为 ``cz``。
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
    """投影 z 在深度图表面容差内。"""

    OCCLUDED = "occluded"
    """点在表面后方（投影 z 明显大于采样深度）。"""

    IN_FRONT = "in_front"
    """点在表面前方（投影 z 明显小于采样深度）。"""

    UNKNOWN = "unknown"
    """点在相机后方、投影越界、插值邻域缺失/被禁用。"""


class CameraModel(enum.Enum):
    """相机投影模型。

    - ``PINHOLE``：针孔（透视）投影，像素坐标除以相机 z；
    - ``ORTHOGRAPHIC``：正交投影，像素坐标**不除 z**。

    两种模型共用同一套世界->相机刚体变换 ``p_cam = R @ P + t``，
    深度含义一致：均为相机坐标 z 分量，裁定前都必须先 ``cz > 0``。
    """

    PINHOLE = "pinhole"
    ORTHOGRAPHIC = "orthographic"


class Reason(enum.Enum):
    """``UNKNOWN`` 的细分原因；非 unknown 状态为 ``NONE``。"""

    NONE = "none"
    BEHIND_CAMERA = "behind_camera"
    OUT_OF_BOUNDS = "out_of_bounds"
    MISSING_NEIGHBOR = "missing_neighbor"
    MASKED_PIXEL = "masked_pixel"


@dataclass(frozen=True)
class Camera:
    """相机：3x3 内参 ``K``、3x3 旋转 ``R``、3 维平移 ``t``、投影模型。

    ``model`` 选择投影模型（见 :class:`CameraModel`），缺省为
    针孔透视模型。``K`` 的结构约束（``K00>0``、``K11>0``、
    ``K10==0``、``K[2]==(0,0,1)``）两种模型相同，但尺度含义不同：

    - 针孔模型下 ``K00/K11`` 为以像素计的**焦距**（无量纲比值），
      投影时除以深度 ``cz``；
    - 正交模型下 ``K00/K11`` 为**像素/世界长度**的比例尺，
      投影时直接乘相机坐标，**不做透视除法**。
    """

    K: Mat3
    R: Mat3
    t: Vec3
    model: CameraModel = CameraModel.PINHOLE

    def __post_init__(self) -> None:
        from .validation import _require_mat3, _require_vec3, validate_camera

        if not isinstance(self.model, CameraModel):
            raise InvalidCameraError(
                "model must be a CameraModel member (PINHOLE/ORTHOGRAPHIC), "
                f"got {type(self.model).__name__}: {self.model!r}"
            )
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

    def pixel_coords(
        self, cx: float, cy: float, cz: float
    ) -> tuple[float, float]:
        """相机坐标 -> 像素 ``(u=列, v=行)``；调用前需确保 ``cz > 0``。

        两种模型的深度都是相机 z，差别仅在投影：

        - 针孔：``u = (K00*cx + K01*cy)/cz + K02``，透视除法；
        - 正交：``u = K00*cx + K01*cy + K02``，**无除法**，
          沿光轴移动（只改 ``cz``）不改变像素位置。
        """
        K = self.K
        if self.model is CameraModel.ORTHOGRAPHIC:
            u = K[0][0] * cx + K[0][1] * cy + K[0][2]
            v = K[1][1] * cy + K[1][2]
            return u, v
        inv_z = 1.0 / cz
        u = (K[0][0] * cx + K[0][1] * cy) * inv_z + K[0][2]
        v = K[1][1] * cy * inv_z + K[1][2]
        return u, v


@dataclass(frozen=True)
class DepthMap:
    """相机深度图。

    - ``values``: 行主序二维序列，``values[row][col]`` 为该像素处表面
      的相机 z（深度）。尺寸不超过 32x32。
    - ``disabled``: 与 ``values`` 同形状的布尔二维序列；``True`` 表示
      该像素被显式禁用。默认为全 False。
    """

    values: tuple[tuple[float, ...], ...] = field(init=False)
    disabled: tuple[tuple[bool, ...], ...] = field(init=False)

    def __init__(
        self,
        values: SequenceLike,
        disabled: SequenceLike | None = None,
    ) -> None:
        from .validation import _freeze_depth

        vals, mask = _freeze_depth(values, disabled)
        object.__setattr__(self, "values", vals)
        object.__setattr__(self, "disabled", mask)

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
    """相机坐标 z；``behind_camera`` 时仍给出（<=0），其余无法投影时为 None。"""
    sampled_depth: float | None
    """双线性采样深度；无法采样时为 None。"""
    u: float | None
    """像素列坐标；无法计算时为 None。"""
    v: float | None
    """像素行坐标；无法计算时为 None。"""


# 构造时接受的“类序列”输入（list/tuple 等可下标、可迭代的有限容器）。
SequenceLike: TypeAlias = object
