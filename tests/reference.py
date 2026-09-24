"""独立参考实现（仅测试使用）。

本模块**不导入** ``visibility`` 包的任何内容，用独立写法重新实现
针孔投影、四邻域双线性采样与容差裁定，供测试交叉核对。
期望值的主要来源仍是测试中手工推算的常量；本参考与被测实现
互为对照，任何共用的概念性错误通过“像素中心手算用例”兜底。
"""

from __future__ import annotations

import math


def transform(R, t, P):
    """相机坐标 p_cam = R @ P + t（独立实现）。"""
    return (
        R[0][0] * P[0] + R[0][1] * P[1] + R[0][2] * P[2] + t[0],
        R[1][0] * P[0] + R[1][1] * P[1] + R[1][2] * P[2] + t[1],
        R[2][0] * P[0] + R[2][1] * P[1] + R[2][2] * P[2] + t[2],
    )


def project(K, R, t, P):
    """返回 (u, v, z)；z<=0 时只返回 z（第三分量），u/v 为 None。"""
    x, y, z = transform(R, t, P)
    if z <= 0.0:
        return None, None, z
    qx, qy = x / z, y / z
    u = K[0][0] * qx + K[0][1] * qy + K[0][2]
    v = K[1][1] * qy + K[1][2]
    return u, v, z


def world_point_at_pixel(K, R, t, u, v, z):
    """逆投影：给定像素 (u,v) 与相机 z，反解世界点 P = R^T(q - t)。"""
    qx = (u - K[0][2] - K[0][1] * (v - K[1][2]) / K[1][1]) / K[0][0]
    qy = (v - K[1][2]) / K[1][1]
    q = (qx * z, qy * z, z)
    # P = R^T (q - t)
    d = (q[0] - t[0], q[1] - t[1], q[2] - t[2])
    return (
        R[0][0] * d[0] + R[1][0] * d[1] + R[2][0] * d[2],
        R[0][1] * d[0] + R[1][1] * d[1] + R[2][1] * d[2],
        R[0][2] * d[0] + R[1][2] * d[1] + R[2][2] * d[2],
    )


def sample(values, disabled, u, v):
    """参考双线性采样：返回 (depth, code)；code 为失败原因字符串。"""
    h = len(values)
    w = len(values[0])
    if u < 0.0 or v < 0.0 or u >= w or v >= h:
        return None, "out_of_bounds"
    cu, cv = math.floor(u), math.floor(v)
    if cu + 1 > w - 1 or cv + 1 > h - 1:
        return None, "missing_neighbor"
    corners = ((cv, cu), (cv, cu + 1), (cv + 1, cu), (cv + 1, cu + 1))
    if any(disabled[r][c] for r, c in corners):
        return None, "masked_pixel"
    fu, fv = u - cu, v - cv
    return (
        (1 - fu) * (1 - fv) * values[cv][cu]
        + fu * (1 - fv) * values[cv][cu + 1]
        + (1 - fu) * fv * values[cv + 1][cu]
        + fu * fv * values[cv + 1][cu + 1]
    ), "ok"


def classify(K, R, t, values, disabled, P, abs_tol, rel_tol):
    """完整参考裁定，返回字典。"""
    u, v, z = project(K, R, t, P)
    if u is None:
        return {"visibility": "unknown", "reason": "behind_camera",
                "u": None, "v": None, "z": z, "depth": None}
    depth, code = sample(values, disabled, u, v)
    if depth is None:
        return {"visibility": "unknown", "reason": code,
                "u": u, "v": v, "z": z, "depth": None}
    diff = z - depth
    tol = abs_tol + rel_tol * depth
    if abs(diff) <= tol:
        vis = "visible"
    elif diff > tol:
        vis = "occluded"
    else:
        vis = "in_front"
    return {"visibility": vis, "reason": "none",
            "u": u, "v": v, "z": z, "depth": depth, "tol": tol}


def const_depth(w, h, d, disabled=None):
    vals = [[d for _ in range(w)] for _ in range(h)]
    if disabled is None:
        mask = [[False for _ in range(w)] for _ in range(h)]
    else:
        mask = [row[:] for row in disabled]
    return vals, mask


# --- 区间深度参考（near/far 两层 + 投影深度不确定半径 depth_eps） ----------


def _bilerp(grid, cu, cv, fu, fv):
    return (
        (1 - fu) * (1 - fv) * grid[cv][cu]
        + fu * (1 - fv) * grid[cv][cu + 1]
        + (1 - fu) * fv * grid[cv + 1][cu]
        + fu * fv * grid[cv + 1][cu + 1]
    )


def sample_interval(near, far, disabled, u, v):
    """参考区间双线性采样：返回 (near_depth, far_depth, code)。"""
    h = len(near)
    w = len(near[0])
    if u < 0.0 or v < 0.0 or u >= w or v >= h:
        return None, None, "out_of_bounds"
    cu, cv = math.floor(u), math.floor(v)
    if cu + 1 > w - 1 or cv + 1 > h - 1:
        return None, None, "missing_neighbor"
    corners = ((cv, cu), (cv, cu + 1), (cv + 1, cu), (cv + 1, cu + 1))
    if any(disabled[r][c] for r, c in corners):
        return None, None, "masked_pixel"
    fu, fv = u - cu, v - cv
    return _bilerp(near, cu, cv, fu, fv), _bilerp(far, cu, cv, fu, fv), "ok"


def classify_interval(K, R, t, near, far, disabled, P,
                      depth_eps, abs_tol, rel_tol):
    """完整参考区间裁定，返回字典。独立于被测核心实现。"""
    x, y, z = transform(R, t, P)
    z_lo, z_hi = z - depth_eps, z + depth_eps
    if z_hi <= 0.0:
        return {"visibility": "unknown", "reason": "behind_camera",
                "u": None, "v": None, "z": z,
                "z_lo": z_lo, "z_hi": z_hi, "d_lo": None, "d_hi": None}
    if z_lo <= 0.0:
        return {"visibility": "unknown", "reason": "depth_interval_at_origin",
                "u": None, "v": None, "z": z,
                "z_lo": z_lo, "z_hi": z_hi, "d_lo": None, "d_hi": None}
    qx, qy = x / z, y / z
    u = K[0][0] * qx + K[0][1] * qy + K[0][2]
    v = K[1][1] * qy + K[1][2]
    d_lo, d_hi, code = sample_interval(near, far, disabled, u, v)
    if d_lo is None:
        return {"visibility": "unknown", "reason": code,
                "u": u, "v": v, "z": z,
                "z_lo": z_lo, "z_hi": z_hi, "d_lo": None, "d_hi": None}
    board_lo = d_lo - (abs_tol + rel_tol * d_lo)
    board_hi = d_hi + (abs_tol + rel_tol * d_hi)
    if z_lo >= board_lo and z_hi <= board_hi:
        vis, reason = "visible", "none"
    elif z_lo > board_hi:
        vis, reason = "occluded", "none"
    elif z_hi < board_lo:
        vis, reason = "in_front", "none"
    else:
        vis, reason = "uncertain", "depth_interval_overlap"
    return {"visibility": vis, "reason": reason, "u": u, "v": v, "z": z,
            "z_lo": z_lo, "z_hi": z_hi,
            "d_lo": d_lo, "d_hi": d_hi,
            "board_lo": board_lo, "board_hi": board_hi}
