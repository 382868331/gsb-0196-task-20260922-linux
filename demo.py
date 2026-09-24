"""固定输入演示：深度遮挡多视图可见性裁定（含深度区间）。

直接运行：``python demo.py``（90 秒内完成，实际计算，无网络、无
sleep、无预录结果）。演示数据在本文件内确定性构造：

- 3 个针孔视图（8x8 深度图），第三视图含一个显式禁用像素；
- 6 个世界点，分别触发 visible / occluded / in_front 以及
  behind_camera / missing_neighbor / masked_pixel 三类 unknown；
- 多视图聚合（至少 k 个视图 visible 才入选）；
- **深度区间**：表面 [near, far] 与投影深度不确定半径 depth_eps，
  展示确定 visible / occluded / in_front、接触或重叠的 uncertain、
  区间跨过相机原点的 unknown；
- 末尾实际触发两次输入拒绝（NaN 世界点、负 depth_eps）并打印定位。
"""

from __future__ import annotations

from visibility import (
    Camera,
    DepthMap,
    InvalidParameterError,
    InvalidPointError,
    Reason,
    View,
    Visibility,
    classify_points,
    classify_view,
    indices_visible_in_at_least_k,
)

# --- 固定相机与深度图 -------------------------------------------------------

FX = FY = 20.0
CX = CY = 3.5
W = H = 8
SURFACE_Z = 6.0

K = (
    (FX, 0.0, CX),
    (0.0, FY, CY),
    (0.0, 0.0, 1.0),
)
IDENT = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
ORIGIN = (0.0, 0.0, 0.0)


def point_at_pixel(u: float, v: float, z: float) -> tuple[float, float, float]:
    """由像素与相机 z 反解世界点（相机位于原点、无旋转）。"""
    return (u - CX) * z / FX, (v - CY) * z / FY, z


def tag_of(st) -> str:
    tag = st.visibility.value
    if st.visibility is Visibility.UNKNOWN or st.visibility is Visibility.UNCERTAIN:
        tag += "/" + st.reason.value
    return tag


def main() -> None:
    plane = [[SURFACE_Z for _ in range(W)] for _ in range(H)]
    # 视图 2 显式禁用像素 (row=3, col=3)。
    mask = [[False for _ in range(W)] for _ in range(H)]
    mask[3][3] = True

    cam = Camera(K=K, R=IDENT, t=ORIGIN)
    views = [
        View(cam, DepthMap(plane)),
        View(cam, DepthMap([row[:] for row in plane])),
        View(cam, DepthMap(plane, mask)),
    ]

    # --- 确定性构造的 6 个探针点 -------------------------------------------
    probes = [
        ("表面上(visible)      ", point_at_pixel(2.5, 2.5, SURFACE_Z)),
        ("表面后(occluded)     ", point_at_pixel(2.5, 2.5, 7.5)),
        ("表面前(in_front)     ", point_at_pixel(2.5, 2.5, 4.5)),
        ("相机后(behind)       ", point_at_pixel(2.5, 2.5, -1.0)),
        ("右边缘缺邻域(missing)", point_at_pixel(7.0, 3.0, SURFACE_Z)),
        ("落在禁用像素(masked) ", point_at_pixel(2.5, 2.5, SURFACE_Z)),
    ]

    print("== 单点 x 单视图裁定（容差 1e-6 绝对 + 1e-6 相对） ==")
    header = f"{'探针点':<24}" + "".join(f"{'视图'+str(i):>30}" for i in range(3))
    print(header)
    for name, P in probes:
        cells = [f"{tag_of(classify_view(vw, P)):>30}" for vw in views]
        print(f"{name:<24}" + "".join(cells))

    # 抽查具体数值，证明是真实计算而非标签查表。
    st = classify_view(views[0], probes[1][1])
    print(
        f"\n抽查 occluded: projected_z={st.projected_z:.3f} "
        f"sampled_depth={st.sampled_depth:.3f} (u,v)=({st.u:.2f},{st.v:.2f})"
    )

    # --- 多视图聚合 --------------------------------------------------------
    print("\n== 多视图聚合（每点至少 k 个视图 visible 才入选） ==")
    points = [P for _, P in probes]
    results = classify_points(views, points)
    for i, row in enumerate(results):
        states = ", ".join(
            f"v{i_view}:{s.visibility.value}" for i_view, s in enumerate(row)
        )
        print(f"点{i} [{states}]")
    for k in (1, 2, 3):
        chosen = indices_visible_in_at_least_k(results, k)
        print(f"k={k}: 入选点下标 = {chosen}")
    print("说明：occluded/in_front/unknown 均不计为可见，且不做跨视图投票翻转。")

    # --- 深度区间裁定（本轮迭代） ------------------------------------------
    print("\n== 深度区间裁定（表面占 [near, far]，点占 [cz-eps, cz+eps]） ==")
    # 视图 A：常量厚度表面 [5, 7]；探针像素 (2.5, 2.5) 四邻点齐全。
    slab = DepthMap.from_intervals(
        [[5.0] * W for _ in range(H)],
        [[7.0] * W for _ in range(H)],
    )
    view_a = View(cam, slab)

    interval_cases = [
        # (说明, 点中心 z, depth_eps, 期望的手算结论)
        ("[5.5,6.5] 整体落在板 [5,7] 内", 6.0, 0.5),
        ("[7.5,8.5] 整体远于 far=7     ", 8.0, 0.5),
        ("[3.5,4.5] 整体近于 near=5    ", 4.0, 0.5),
        ("[4.5,5.5] 越过 near 面(重叠) ", 5.0, 0.5),
        ("[4,5] 与板仅在 near 端接触   ", 4.5, 0.5),
        ("[7,8] 与板仅在 far 端接触    ", 7.5, 0.5),
        ("[6,10] 整块包住表面(也不判可见)", 8.0, 2.0),
    ]
    for desc, z, eps in interval_cases:
        P = point_at_pixel(2.5, 2.5, z)
        s = classify_view(view_a, P, abs_tol=0.0, rel_tol=0.0, depth_eps=eps)
        pi = s.projected_interval
        di = s.depth_interval
        print(
            f"{desc}: {tag_of(s):<34} "
            f"点区间=[{pi[0]:.2f},{pi[1]:.2f}] 表面区间=[{di[0]:.1f},{di[1]:.1f}]"
        )
    print("规则：重叠或仅端点接触一律 uncertain，不直接判 visible/occluded。")

    # 容差板边界（手算：near=5, far=7, abs_tol=0.2 -> 板 [4.8, 7.2]）。
    s_in = classify_view(view_a, point_at_pixel(2.5, 2.5, 7.2),
                         abs_tol=0.2, rel_tol=0.0)
    s_out = classify_view(view_a, point_at_pixel(2.5, 2.5, 7.2 + 1e-9),
                          abs_tol=0.2, rel_tol=0.0)
    print(
        f"容差板边界：z=7.2(板边) -> {s_in.visibility.value}；"
        f"z=7.2+1e-9 -> {s_out.visibility.value}"
    )

    # near/far 分别双线性的真实数值抽查：
    # 列 2 像素为厚表面 [6,10]，其余列薄 [5,5]；u=2.5 权重各半
    # -> 插值区间恰为 [5.5, 7.5]。
    near_grid = [[5.0] * W for _ in range(H)]
    far_grid = [[5.0] * W for _ in range(H)]
    for r in range(H):
        near_grid[r][2] = 6.0
        far_grid[r][2] = 10.0
    view_step = View(cam, DepthMap.from_intervals(near_grid, far_grid))
    s_edge = classify_view(view_step, point_at_pixel(2.5, 2.5, 7.5),
                           abs_tol=0.0, rel_tol=0.0)
    s_over = classify_view(view_step, point_at_pixel(2.5, 2.5, 7.6),
                           abs_tol=0.0, rel_tol=0.0)
    print(
        f"分层双线性抽查：u=2.5 插值表面区间=["
        f"{s_edge.sampled_depth:.2f}, {s_edge.sampled_depth_far:.2f}]；"
        f"z=7.5 -> {s_edge.visibility.value}，z=7.6 -> {s_over.visibility.value}"
    )

    # 原点深度退化：点中心 cz=0.5、eps=1 -> 区间 [-0.5,1.5] 跨过原点。
    s_origin = classify_view(
        view_a, (0.0, 0.0, 0.5), depth_eps=1.0, abs_tol=0.0, rel_tol=0.0
    )
    print(
        f"原点退化：点区间 {s_origin.projected_interval} -> "
        f"{s_origin.visibility.value}/{s_origin.reason.value}（1/z 在 0 无定义）"
    )

    # 多视图保留逐视图原因（不融合成二值）：
    # 同一带厚度的点在三视图分别得到 uncertain / visible / occluded。
    mv_views = [
        View(cam, slab),                                            # [5,7]
        View(cam, DepthMap.from_intervals(                          # [4,8]
            [[4.0] * W for _ in range(H)],
            [[8.0] * W for _ in range(H)])),
        View(cam, DepthMap.from_intervals(                          # [2,3]
            [[2.0] * W for _ in range(H)],
            [[3.0] * W for _ in range(H)])),
    ]
    P = point_at_pixel(2.5, 2.5, 6.0)
    mv = classify_points(mv_views, [P], abs_tol=0.0, rel_tol=0.0,
                         depth_eps=1.5)[0]
    print(
        "多视图逐视图原因（同一点 z 区间 [4.5,7.5]）：" + "；".join(
            f"v{i}:{tag_of(s)}" for i, s in enumerate(mv)
        )
    )
    print(f"visible 视图数 = {sum(1 for s in mv if s.visibility is Visibility.VISIBLE)}"
          "（uncertain 不凑数、不被投票翻转）")

    # --- 实际触发的拒绝边界 ------------------------------------------------
    print("\n== 输入拒绝边界（非法输入必须被拒绝，不留半更新状态） ==")
    try:
        classify_view(views[0], (0.0, float("nan"), SURFACE_Z))
    except InvalidPointError as exc:
        print(f"实际捕获 {type(exc).__name__}: {exc}")
    else:
        raise AssertionError("NaN 输入应当被拒绝")

    try:
        classify_view(view_a, point_at_pixel(2.5, 2.5, 6.0), depth_eps=-0.01)
    except InvalidParameterError as exc:
        print(f"实际捕获 {type(exc).__name__}: {exc}")
    else:
        raise AssertionError("负 depth_eps 应当被拒绝")

    print("\n演示正常结束。")


if __name__ == "__main__":
    main()
