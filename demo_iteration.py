#!/usr/bin/env python3
"""纹理相机选择演示：可见性裁定 -> 树 DP 选相机 -> 打印账本。

流程（全部真实计算、确定性、离线）：

1. 用 visibility 核心搭建 4 个针孔相机与解析平面深度图，
   对 6 个面代表点做可见性裁定（classify_points）。
2. 可见的 (面, 视图) 组合成为候选，数据代价由确定性公式给出
   （离轴角越大代价越高，模拟纹理投影失真）。
3. 逐面贪心（每面单独最优）作为“选择前”账本；
   select_texture_cameras 的树 DP 解作为“选择后”账本。
4. 边界演示：一个所有视图均不可见的面，触发
   NoVisibleCandidateError 并报告具体面 ID。

无 sleep、无预录输出；实测运行远低于 90 秒。
"""

from __future__ import annotations

from visibility import (
    Camera,
    DepthMap,
    FaceCandidate,
    NoVisibleCandidateError,
    View,
    Visibility,
    classify_points,
    select_texture_cameras,
)


def build_views() -> list[View]:
    """4 个相机环绕 z=0 平面上的面片；深度图是解析平面 z=10。"""
    views: list[View] = []
    for i in range(4):
        # 相机沿 x 轴等距排布，均朝向 +z 方向的墙；32x32 视野使
        # 每片墙至少被正对相机看到，多数面片还被相邻相机看到。
        cam = Camera(
            K=((50.0, 0.0, 15.5), (0.0, 50.0, 15.5), (0.0, 0.0, 1.0)),
            R=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            t=(float(1.8 * i - 2.7), 0.0, 0.0),
        )
        depth = DepthMap([[10.0] * 32 for _ in range(32)])
        views.append(View(camera=cam, depth_map=depth))
    return views


def face_points() -> list[tuple[float, float, float]]:
    """6 个面代表点，z=10 的墙面上沿 x 排开。"""
    return [(-4.5 + 1.8 * i, 0.0, 10.0) for i in range(6)]


def data_cost(u: float | None) -> float:
    """确定性数据代价：投影离主点越远代价越高（模拟纹理失真）。"""
    assert u is not None
    return round(0.1 + 0.05 * abs(u - 15.5), 6)


def greedy_pick(faces):
    """选择前基线：每面单独取数据代价最小的可见候选。"""
    vec = []
    total = 0.0
    for fid, cands in faces:
        visible = [c for c in cands if c.status.visibility is Visibility.VISIBLE]
        best = min(visible, key=lambda c: (c.data_cost, c.status.view_index))
        vec.append(best.status.view_index)
        total += best.data_cost
    return vec, total


def seam_total(vec, face_ids, edges) -> float:
    pos = {f: i for i, f in enumerate(face_ids)}
    return sum(
        seam for a, b, seam in edges if vec[pos[a]] != vec[pos[b]]
    )


def print_ledger(title, face_ids, vec, faces, edges) -> None:
    print(f"--- {title} ---")
    data_sum = 0.0
    for fid, vi, (_, cands) in zip(face_ids, vec, faces):
        cost = next(
            c.data_cost for c in cands
            if c.status.view_index == vi
            and c.status.visibility is Visibility.VISIBLE
        )
        data_sum += cost
        print(f"  face {fid}: camera {vi}, data_cost {cost:.3f}")
    seam = seam_total(vec, face_ids, edges)
    pos = {f: i for i, f in enumerate(face_ids)}
    for a, b, s in edges:
        if vec[pos[a]] != vec[pos[b]]:
            print(f"  seam ({a},{b}): camera {vec[pos[a]]} != {vec[pos[b]]}, cost {s:.3f}")
    print(f"  data total {data_sum:.3f} + seam total {seam:.3f} = {data_sum + seam:.3f}")


def main() -> None:
    views = build_views()
    points = face_points()
    results = classify_points(views, points)

    face_ids = list(range(6))
    faces = []
    for fid, row in zip(face_ids, results):
        cands = [
            FaceCandidate(status=st, data_cost=data_cost(st.u))
            for st in row
        ]
        n_vis = sum(1 for c in cands if c.status.visibility is Visibility.VISIBLE)
        print(f"face {fid}: {n_vis}/{len(cands)} views visible")
        faces.append((fid, cands))

    # 树状邻接：一条链 0-1-2-3-4-5，接缝代价随“距离”增大。
    edges = [(i, i + 1, 0.5) for i in range(5)]

    greedy_vec, _ = greedy_pick(faces)
    print_ledger("before: per-face greedy", face_ids, greedy_vec, faces, edges)

    sel = select_texture_cameras(faces, edges)
    print("--- after: tree DP optimum ---")
    for f in sel.faces:
        print(f"  face {f.face_id}: camera {f.view_index}, data_cost {f.data_cost:.3f}")
    for s in sel.seams:
        if s.cost > 0.0:
            print(
                f"  seam ({s.face_a},{s.face_b}): camera {s.view_a} != "
                f"{s.view_b}, cost {s.cost:.3f}"
            )
    print(f"  camera vector (by face id): {list(sel.camera_vector)}")
    print(f"  total cost: {sel.total_cost:.3f}")

    # 边界：面 9 在所有视图都被遮挡（点在相机后方），无可见候选。
    behind = classify_points(views, [(0.0, 0.0, -1.0)])
    bad_faces = faces + [(9, [FaceCandidate(status=st, data_cost=0.1)
                              for st in behind[0]])]
    bad_edges = edges + [(5, 9, 0.5)]
    try:
        select_texture_cameras(bad_faces, bad_edges)
    except NoVisibleCandidateError as exc:
        print(f"boundary: rejected as expected -> {exc}")


if __name__ == "__main__":
    main()
