"""公开数据结构。

全部为不可值对象（``frozen=True``），深度图在构造时做防御性拷贝，
外部后续修改传入的列表不会影响已构造的 :class:`DepthMap`。

相机约定（两类视图共用同一套深度/采样语义）
------------------------------------------------

支持两类投影模型，由 :class:`CameraModel` 区分，两类相机可以混合出现在
同一次多视图裁定中，彼此独立、互不影响：

**透视（针孔）相机** :attr:`CameraModel.PERSPECTIVE`（默认）。
世界点 ``P = (X, Y, Z)`` 先经世界->相机刚体变换::

    p_cam = R @ P + t

其中 ``p_cam = (cx, cy, cz)``。**深度就是相机坐标 z 分量** ``cz``。
针孔内参 ``K``（行主序 3x3）把 *已除以深度* 的相机坐标映射到像素::

    u = K00 * cx / cz + K01 * cy / cz + K02
    v = K11 * cy / cz + K12
    (w = 1)

**正交相机** :attr:`CameraModel.ORTHOGRAPHIC`：相机坐标与深度定义完全
相同（``p_cam = R @ P + t``，深度仍为 ``cz``），但内参 ``K`` 直接把相机
x/y 仿射映射到像素，**不做透视除法**::

    u = K00 * cx + K01 * cy + K02
    v = K11 * cy + K12

即 ``K00/K11`` 是“像素/世界单位”的正交比例尺（必须 > 0），``K02/K12``
仍是像素主点。无论点远近，投影像素只由相机 x/y 决定；绝不能把透视公式
里的 ``/cz`` 套用到正交相机。两类模型都先判 ``cz > 0`` 再投影。

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
    """投影 z 在深度图表面容差内。"""

    OCCLUDED = "occluded"
    """点在表面后方（投影 z 明显大于采样深度）。"""

    IN_FRONT = "in_front"
    """点在表面前方（投影 z 明显小于采样深度）。"""

    UNKNOWN = "unknown"
    """点在相机后方、投影越界、插值邻域缺失/被禁用。"""


class CameraModel(enum.Enum):
    """相机投影模型。两类视图可混合参与同一次裁定。"""

    PERSPECTIVE = "perspective"
    """透视（针孔）：像素 = K @ (cx/cz, cy/cz, 1)，含透视除法。"""

    ORTHOGRAPHIC = "orthographic"
    """正交：像素 = K @ (cx, cy, 1)，不做透视除法；深度仍为相机 z。"""


class Reason(enum.Enum):
    """``UNKNOWN`` 的细分原因；非 unknown 状态为 ``NONE``。"""

    NONE = "none"
    BEHIND_CAMERA = "behind_camera"
    OUT_OF_BOUNDS = "out_of_bounds"
    MISSING_NEIGHBOR = "missing_neighbor"
    MASKED_PIXEL = "masked_pixel"


@dataclass(frozen=True)
class Camera:
    """相机：3x3 内参 ``K``、3x3 旋转 ``R``、3 维平移 ``t``。

    ``model`` 选择投影模型（见 :class:`CameraModel`），默认为
    :attr:`CameraModel.PERSPECTIVE`（透视/针孔）。
    """

    K: Mat3
    R: Mat3
    t: Vec3
    model: CameraModel = CameraModel.PERSPECTIVE

    def __post_init__(self) -> None:
        from .validation import _require_mat3, _require_vec3, validate_camera

        K = _require_mat3(self.K, "K", InvalidCameraError)
        R = _require_mat3(self.R, "R", InvalidCameraError)
        t = _require_vec3(self.t, "t", InvalidCameraError)
        object.__setattr__(self, "K", K)
        object.__setattr__(self, "R", R)
        object.__setattr__(self, "t", t)
        if not isinstance(self.model, CameraModel):
            raise InvalidCameraError(
                "model must be a visibility.CameraModel member, got "
                f"{type(self.model).__name__}"
            )
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

    def project_pixels(
        self, cx: float, cy: float, cz: float
    ) -> tuple[float, float]:
        """相机坐标 -> 像素 ``(u, v)``。调用前必须已确保 ``cz > 0``。

        透视模型除以 ``cz``；正交模型不除（投影与深度无关）。
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


class OrthographicCamera(Camera):
    """正交相机：``K`` 为像素仿射，投影不做透视除法。

    其余字段（``R``、``t``）与深度含义（相机 z）与透视相机完全相同。
    是 :class:`Camera` 的子类，可直接用于 :class:`View`。
    """

    def __init__(self, K: Mat3, R: Mat3, t: Vec3) -> None:
        super().__init__(
            K=K, R=R, t=t, model=CameraModel.ORTHOGRAPHIC
        )


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
