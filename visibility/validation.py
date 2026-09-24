"""输入校验与防御性冻结。

规则（本题契约）：

- 所有浮点/数值输入必须有限：拒绝 NaN 与 +/-Infinity；
- 整数参数拒绝 ``bool``（``True`` 不等于 1）；
- 深度图尺寸 1..32 x 1..32、矩形、深度值为有限正数；
- 禁用 mask 必须与深度图同形且元素为真正的 ``bool``；
- 内参 K 为有限针孔结构：``K00>0``、``K11>0``、``K[2]=(0,0,1)``。

错误信息携带字段/下标路径，例如 ``K[1][2]``、``values[3][7]``、
``points[2][1]``，保证可定位，且不做任何静默纠正。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from .errors import (
    InvalidCameraError,
    InvalidDepthMapError,
    InvalidParameterError,
    InvalidPointError,
)
from .types import MAX_DEPTH_HEIGHT, MAX_DEPTH_WIDTH, Camera


def is_finite_number(x: Any) -> bool:
    """真数值（不含 bool）且有限。"""
    if isinstance(x, bool):
        return False
    if isinstance(x, int):
        return True  # int 不存在 NaN/Infinity
    if isinstance(x, float):
        return math.isfinite(x)
    return False


def _require_finite_number(x: Any, path: str, err: type[ValueError]) -> float:
    if isinstance(x, bool):
        raise err(f"{path} must be a finite real number, got bool")
    if isinstance(x, int):
        return float(x)
    if isinstance(x, float):
        if not math.isfinite(x):
            raise err(f"{path} must be finite, got {x}")
        return x
    raise err(f"{path} must be a finite real number, got {type(x).__name__}")


def _require_vec3(v: Any, path: str, err: type[ValueError]) -> tuple[float, float, float]:
    if not isinstance(v, Sequence) or isinstance(v, (str, bytes)):
        raise err(f"{path} must be a sequence of 3 finite numbers")
    if len(v) != 3:
        raise err(f"{path} must have length 3, got {len(v)}")
    return tuple(_require_finite_number(v[i], f"{path}[{i}]", err) for i in range(3))  # type: ignore[return-value]


def _require_mat3(m: Any, path: str, err: type[ValueError]) -> tuple[tuple[float, float, float], ...]:
    if not isinstance(m, Sequence) or isinstance(m, (str, bytes)):
        raise err(f"{path} must be a 3x3 sequence of finite numbers")
    if len(m) != 3:
        raise err(f"{path} must have 3 rows, got {len(m)}")
    rows = []
    for i in range(3):
        row = m[i]
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)) or len(row) != 3:
            raise err(f"{path}[{i}] must be a sequence of 3 finite numbers")
        rows.append(
            tuple(
                _require_finite_number(row[j], f"{path}[{i}][{j}]", err)
                for j in range(3)
            )
        )
    return tuple(rows)  # type: ignore[return-value]


def validate_camera(cam: Camera) -> None:
    if not isinstance(cam, Camera):
        raise InvalidCameraError(
            f"expected a Camera instance, got {type(cam).__name__}"
        )
    K = _require_mat3(cam.K, "K", InvalidCameraError)
    _require_mat3(cam.R, "R", InvalidCameraError)
    _require_vec3(cam.t, "t", InvalidCameraError)
    if K[2][0] != 0.0 or K[2][1] != 0.0 or K[2][2] != 1.0:
        raise InvalidCameraError(
            f"K[2] must be exactly (0, 0, 1), got ({K[2][0]}, {K[2][1]}, {K[2][2]})"
        )
    if K[1][0] != 0.0:
        raise InvalidCameraError(f"K[1][0] must be exactly 0, got {K[1][0]}")
    if K[0][0] <= 0.0:
        raise InvalidCameraError(f"K[0][0] (fx) must be > 0, got {K[0][0]}")
    if K[1][1] <= 0.0:
        raise InvalidCameraError(f"K[1][1] (fy) must be > 0, got {K[1][1]}")


def validate_point(point: Any, path: str = "point") -> tuple[float, float, float]:
    """校验单个世界点，返回浮点三元组。"""
    return _require_vec3(point, path, InvalidPointError)


def _freeze_depth(
    values: Any, disabled: Any
) -> tuple[tuple[tuple[float, ...], ...], tuple[tuple[bool, ...], ...]]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise InvalidDepthMapError(
            "depth_map.values must be a 2D rectangular sequence of finite numbers"
        )
    h = len(values)
    if not 1 <= h <= MAX_DEPTH_HEIGHT:
        raise InvalidDepthMapError(
            f"depth map height must be in [1, {MAX_DEPTH_HEIGHT}], got {h}"
        )
    width: int | None = None
    frozen_rows: list[tuple[float, ...]] = []
    for r in range(h):
        row = values[r]
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            raise InvalidDepthMapError(f"values[{r}] must be a sequence of numbers")
        if width is None:
            width = len(row)
            if not 1 <= width <= MAX_DEPTH_WIDTH:
                raise InvalidDepthMapError(
                    f"depth map width must be in [1, {MAX_DEPTH_WIDTH}], got {width}"
                )
        elif len(row) != width:
            raise InvalidDepthMapError(
                f"values[{r}] has length {len(row)}, expected {width} (rectangular depth map)"
            )
        fr: list[float] = []
        for c in range(width):
            x = row[c]
            v = _require_finite_number(x, f"values[{r}][{c}]", InvalidDepthMapError)
            if v <= 0.0:
                raise InvalidDepthMapError(
                    f"values[{r}][{c}] depth (camera z) must be > 0, got {v}"
                )
            fr.append(v)
        frozen_rows.append(tuple(fr))
    assert width is not None

    if disabled is None:
        mask = tuple(tuple(False for _ in range(width)) for _ in range(h))
        return tuple(frozen_rows), mask

    if not isinstance(disabled, Sequence) or isinstance(disabled, (str, bytes)):
        raise InvalidDepthMapError("disabled must be a 2D rectangular bool sequence")
    if len(disabled) != h:
        raise InvalidDepthMapError(
            f"disabled has {len(disabled)} rows, expected {h} (same shape as values)"
        )
    frozen_mask: list[tuple[bool, ...]] = []
    for r in range(h):
        row = disabled[r]
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            raise InvalidDepthMapError(f"disabled[{r}] must be a sequence of bool")
        if len(row) != width:
            raise InvalidDepthMapError(
                f"disabled[{r}] has length {len(row)}, expected {width}"
            )
        fb: list[bool] = []
        for c in range(width):
            x = row[c]
            if not isinstance(x, bool):
                raise InvalidDepthMapError(
                    f"disabled[{r}][{c}] must be bool, got {type(x).__name__}"
                )
            fb.append(x)
        frozen_mask.append(tuple(fb))
    return tuple(frozen_rows), tuple(frozen_mask)


def validate_tolerance(abs_tol: Any, rel_tol: Any) -> tuple[float, float]:
    a = _require_finite_number(abs_tol, "abs_tol", InvalidParameterError)
    r = _require_finite_number(rel_tol, "rel_tol", InvalidParameterError)
    if a < 0.0:
        raise InvalidParameterError(f"abs_tol must be >= 0, got {a}")
    if r < 0.0:
        raise InvalidParameterError(f"rel_tol must be >= 0, got {r}")
    return a, r


def validate_k(k: Any, n_views: int) -> int:
    if isinstance(k, bool) or not isinstance(k, int):
        raise InvalidParameterError(f"k must be an int, got {type(k).__name__}")
    if not 1 <= k <= n_views:
        raise InvalidParameterError(
            f"k must be in [1, {n_views}] (number of views), got {k}"
        )
    return k


def validate_depth_interval(z_near: Any, z_far: Any) -> tuple[float, float]:
    """校验相机 z 深度闭区间 ``[z_near, z_far]``。

    两端必须是有限实数（拒绝 bool / NaN / Infinity）；允许 <= 0，
    因为“区间跨过相机原点”是裁定阶段的显式退化而非构造错误。
    要求 ``z_near <= z_far``，否则抛出 :class:`InvalidParameterError`。
    """
    lo = _require_finite_number(
        z_near, "depth_interval.z_near", InvalidParameterError
    )
    hi = _require_finite_number(
        z_far, "depth_interval.z_far", InvalidParameterError
    )
    if lo > hi:
        raise InvalidParameterError(
            f"depth_interval requires z_near <= z_far, got [{lo}, {hi}]"
        )
    return lo, hi
