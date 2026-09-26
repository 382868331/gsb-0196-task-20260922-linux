"""纹理相机选择演示：树 DP 选择前后的账本对照。

真实计算、离线、确定性，无 sleep、无预录输出。演示三部分：

1. 用 visibility 核心裁定一棵 5 面邻接树的候选视图可见性；
2. 对照"每面单独最优（不看接缝）"与"树 DP 全局最优"两份账本；
3. 边界：某面所有候选不可见时，报告具体面 ID 而不伪造相机。
"""

from __future__ import annotations

from visibility import (
    Camera,
    CandidateView,
    DepthMap,
    FaceCandidates,
    InfeasibleSelectionError,
    SeamEdge,
    View,
    Visibility,
    classify_points,
    select_cameras,
)

K = ((100.0, 0.0, 12.0), (0.0, 100.0, 4.0), (0.0, 0.0, 1.0))
R = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def make_view(cam_index: int) -> View:
    """第 cam_index 台相机：沿 x 平移，深度图常数 10（8x8）。"""
    cam = Camera(K=K, R=R, t=(float(cam_index), 0.0, 0.0))
    dm = DepthMap([[10.0] * 24 for _ in range(8)])
    return View(cam, dm)


def main() -> None:
    # 5 个面各取一个代表点；相机沿 x 依次平移，横向覆盖错开，
    # 使每个面在 1~3 台相机中可见（深度均为 10，与深度图一致）。
    points = [
        (0.0, 0.0, 10.0),
        (-0.5, 0.0, 10.0),
        (-1.0, 0.0, 10.0),
        (-1.5, 0.0, 10.0),
        (-2.5, 0.0, 10.0),
    ]
    views = [make_view(i) for i in range(3)]
    results = classify_points(views, points)

    # 候选数据代价（确定性构造）：代价随相机编号与面变化。
    data_cost = [
        [1.0, 2.5, 4.0],
        [3.0, 1.0, 2.0],
        [2.0, 3.5, 1.0],
        [1.5, 1.5, 5.0],
        [0.5, 0.6, 0.4],
    ]
    faces = []
    for face_id, row in enumerate(results):
        cands = [
            CandidateView(
                view_id=v,
                data_cost=data_cost[face_id][v],
                visible=st.visibility is Visibility.VISIBLE,
            )
            for v, st in enumerate(row)
        ]
        faces.append(FaceCandidates(face_id, cands))
    edges = [
        SeamEdge(0, 1, 2.0),
        SeamEdge(1, 2, 2.0),
        SeamEdge(1, 3, 3.0),
        SeamEdge(3, 4, 1.0),
    ]

    print("== 可见性裁定（visible 视图 / 面）==")
    for f in faces:
        vis = [c.view_id for c in f.candidates if c.visible]
        print(f"  face {f.face_id}: visible views = {vis}")

    def ledger_of(picks):
        """给定每面所选 view_id，计算 (数据账本, 接缝账本, 总成本)。"""
        by_id = {f.face_id: f for f in faces}
        data = []
        for fid, vid in picks.items():
            c = next(c for c in by_id[fid].candidates if c.view_id == vid)
            data.append((fid, vid, c.data_cost))
        seam = []
        for e in edges:
            cost = e.cost if picks[e.face_a] != picks[e.face_b] else 0.0
            seam.append((e.face_a, e.face_b, picks[e.face_a], picks[e.face_b], cost))
        total = sum(d[2] for d in data) + sum(s[4] for s in seam)
        return data, seam, total

    def show(title, data, seam, total, vec):
        print(title)
        print(f"  相机向量（按面 ID 顺序）: {vec}")
        for fid, vid, cost in data:
            print(f"  数据项  face {fid}: view {vid}, data_cost = {cost:.2f}")
        for a, b, va, vb, cost in seam:
            print(f"  接缝项  edge {a}-{b}: views {va}/{vb}, seam_cost = {cost:.2f}")
        print(f"  总成本 = {total:.2f}")

    # 选择前：每面单独最优（不可见候选不可选），不看接缝。
    greedy = {
        f.face_id: min(
            (c for c in f.candidates if c.visible), key=lambda c: c.data_cost
        ).view_id
        for f in faces
    }
    d, s, t = ledger_of(greedy)
    show("== 选择前：每面单独最优（忽略接缝）==", d, s, t,
         tuple(greedy[f.face_id] for f in faces))

    # 选择后：树 DP 全局最优。
    res = select_cameras(faces, edges)
    show("== 选择后：树 DP 全局最优 ==",
         [(e.face_id, e.view_id, e.data_cost) for e in res.data_ledger],
         [(e.face_a, e.face_b, e.view_a, e.view_b, e.cost) for e in res.seam_ledger],
         res.total_cost, res.camera_vector)
    print(f"  （数据 {res.total_data_cost:.2f} + 接缝 {res.total_seam_cost:.2f}，"
          f"相比每面单独最优节省 {t - res.total_cost:.2f}）")

    # 边界：face 4 的唯一便宜候选在相机 2 后方不可见还不够——构造一个
    # 全部候选都不可见的面，验证报告具体面而不伪造默认相机。
    print("== 边界：某面所有候选均不可见 ==")
    bad_faces = list(faces)
    bad_faces[4] = FaceCandidates(
        4, [CandidateView(v, 0.1, visible=False) for v in range(3)]
    )
    try:
        select_cameras(bad_faces, edges)
    except InfeasibleSelectionError as exc:
        print(f"  按预期拒绝: {exc}")


if __name__ == "__main__":
    main()
