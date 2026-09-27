"""独立参考实现（仅测试使用）。

本模块**不导入** ``visibility`` 包的任何内容，用独立写法重新实现
针孔/正交投影、四邻域双线性采样与容差裁定，供测试交叉核对。
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


def project(K, R, t, P, model="pinhole"):
    """返回 (u, v, z)；z<=0 时只返回 z（第三分量），u/v 为 None。

    model="orthographic" 时不做透视除法；两者深度都是相机 z。
    """
    x, y, z = transform(R, t, P)
    if z <= 0.0:
        return None, None, z
    if model == "orthographic":
        u = K[0][0] * x + K[0][1] * y + K[0][2]
        v = K[1][1] * y + K[1][2]
    else:
        qx, qy = x / z, y / z
        u = K[0][0] * qx + K[0][1] * qy + K[0][2]
        v = K[1][1] * qy + K[1][2]
    return u, v, z


def world_point_at_pixel(K, R, t, u, v, z, model="pinhole"):
    """逆投影：给定像素 (u,v) 与相机 z，反解世界点 P = R^T(q - t)。"""
    qx = (u - K[0][2] - K[0][1] * (v - K[1][2]) / K[1][1]) / K[0][0]
    qy = (v - K[1][2]) / K[1][1]
    if model == "orthographic":
        q = (qx, qy, z)
    else:
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


def classify(K, R, t, values, disabled, P, abs_tol, rel_tol, model="pinhole"):
    """完整参考裁定，返回字典。"""
    u, v, z = project(K, R, t, P, model)
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
